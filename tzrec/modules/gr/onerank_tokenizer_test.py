# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``tzrec.modules.gr.onerank_tokenizer``.

The core assertion is :meth:`OneRankFuncTensorTest.test_matches_reference_mask`:
the NFUNC=3 func tensor, once decoded to a dense mask, must equal the mask
written out directly from the paper's group layout.  Everything downstream --
task-token isolation, cross-candidate isolation, the causal prefix -- is a
property of that mask, so it is the one place the whole tokenization scheme
can be checked for correctness without a GPU.
"""

import unittest
from typing import List, Optional, Tuple

import torch
from parameterized import parameterized
from torch import nn

from tzrec.modules.gr.onerank_tokenizer import (
    OneRankSTULayer,
    OneRankTokenizer,
    build_onerank_func_tensor,
)
from tzrec.ops._pytorch.pt_hstu_attention import _decode_attn_func_to_mask
from tzrec.utils.test_util import (
    TestGraphType,
    create_test_module,
    mark_ci_scope,
)

_GRAPH_TYPES = [TestGraphType.NORMAL, TestGraphType.FX_TRACE]

# (name, group_size, [(prefix_len, num_candidates), ...]) per batch sample.
_LAYOUT_CASES = [
    ("single_task", 2, [(4, 1)]),
    ("no_prefix", 4, [(0, 2)]),
    ("seven_tasks", 14, [(5, 3)]),
    ("no_candidates", 6, [(6, 0)]),
    ("mixed_batch", 6, [(3, 2), (7, 1), (1, 4)]),
    ("ragged_with_empty", 4, [(0, 0), (5, 2), (2, 1)]),
]


def _xprod(cases: List[Tuple]) -> List[Tuple]:
    return [
        (f"{case[0]}_{gt.name.lower()}", gt, *case[1:])
        for gt in _GRAPH_TYPES
        for case in cases
    ]


class _BuildOneRankFuncTensorWrapper(nn.Module):
    def __init__(self, group_size: int, nheads: int = 1) -> None:
        super().__init__()
        self._group_size = group_size
        self._nheads = nheads

    def forward(
        self,
        x: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        return build_onerank_func_tensor(
            nheads=self._nheads,
            seq_offsets=seq_offsets,
            total_q=x.size(0),
            num_targets=num_targets,
            group_size=self._group_size,
        )


def _reference_mask(
    layout: List[Tuple[int, int]], group_size: int, padded_len: int
) -> torch.Tensor:
    """Write out the paper's group layout as a dense mask, row by row.

    ``X = [prefix (H) | G_1 | ... | G_N]`` with
    ``G_i = [c_i^(1), t_1, ..., c_i^(K), t_K]``:

    * prefix rows are plain causal;
    * a candidate replica sees the whole prefix plus itself;
    * a task token sees the whole prefix plus its own replica and itself.

    Returns:
        torch.Tensor: ``(B, padded_len, padded_len)`` bool.
    """
    mask = torch.zeros(len(layout), padded_len, padded_len, dtype=torch.bool)
    for b, (prefix_len, num_candidates) in enumerate(layout):
        for q in range(prefix_len):
            mask[b, q, : q + 1] = True
        for i in range(num_candidates):
            base = prefix_len + i * group_size
            for slot in range(group_size):
                q = base + slot
                mask[b, q, :prefix_len] = True
                mask[b, q, q] = True
                if slot % 2 == 1:
                    # The replica this task token is paired with.
                    mask[b, q, q - 1] = True
    return mask


def _build_func_tensor(
    layout: List[Tuple[int, int]],
    group_size: int,
    nheads: int = 1,
    graph_type: TestGraphType = TestGraphType.NORMAL,
    num_targets_override: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run ``build_onerank_func_tensor`` on an expanded jagged batch."""
    seq_lengths = [p + n * group_size for p, n in layout]
    num_targets = (
        num_targets_override
        if num_targets_override is not None
        else [n * group_size for _, n in layout]
    )
    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(seq_lengths, dtype=torch.int32)
    )
    # Dummy x so that total_q = x.size(0) is a Proxy under fx-trace, which
    # is how STULayer.forward calls it.
    x = torch.zeros(int(sum(seq_lengths)), 1)
    module = create_test_module(
        _BuildOneRankFuncTensorWrapper(group_size=group_size, nheads=nheads),
        graph_type,
    )
    func = module(x, seq_offsets, torch.tensor(num_targets, dtype=torch.int32))
    return func, seq_offsets, max(seq_lengths + [1])


@mark_ci_scope("h20", "gpu")
class OneRankFuncTensorTest(unittest.TestCase):
    """Verify the NFUNC=3 encoding of the OneRank group layout."""

    @parameterized.expand(_xprod(_LAYOUT_CASES))
    def test_matches_reference_mask(
        self,
        _name: str,
        graph_type: TestGraphType,
        group_size: int,
        layout: List[Tuple[int, int]],
    ) -> None:
        """Decoded func tensor == mask built directly from the layout.

        ``_decode_attn_func_to_mask`` is the PyTorch kernel's own reader of
        the encoding, so comparing against it checks exactly what attention
        will see -- not our re-reading of the convention.
        """
        func, seq_offsets, padded_len = _build_func_tensor(
            layout, group_size, graph_type=graph_type
        )
        self.assertEqual(tuple(func.shape), (1, 3, int(seq_offsets[-1])))
        got = _decode_attn_func_to_mask(func, seq_offsets, padded_len)
        want = _reference_mask(layout, group_size, padded_len)
        torch.testing.assert_close(got[:, 0], want)

    @parameterized.expand([(gt.name.lower(), gt) for gt in _GRAPH_TYPES])
    def test_all_heads_share_one_row(
        self, _name: str, graph_type: TestGraphType
    ) -> None:
        """The mask is head-independent, so it is broadcast, not copied."""
        func, _, _ = _build_func_tensor(
            [(4, 2)], group_size=6, nheads=4, graph_type=graph_type
        )
        self.assertEqual(func.size(0), 4)
        for head in range(1, 4):
            torch.testing.assert_close(func[head], func[0])

    def test_task_tokens_within_a_group_are_isolated(self) -> None:
        """``t_k`` never sees ``t_j`` for ``j != k``, nor another candidate.

        This is the property the whole replica layout exists to buy: a task
        token must not read another task's state, or the cross-task head in
        the tower would be modelling an attention path that already leaked.
        """
        prefix_len, num_candidates, group_size = 3, 3, 8
        func, seq_offsets, padded_len = _build_func_tensor(
            [(prefix_len, num_candidates)], group_size
        )
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]

        for i in range(num_candidates):
            base = prefix_len + i * group_size
            for k in range(group_size // 2):
                q = base + 2 * k + 1  # a task token
                visible = mask[q].nonzero().flatten().tolist()
                self.assertEqual(visible, list(range(prefix_len)) + [q - 1, q])

        # Cross-group: no row of group 0 sees any column of group 1.
        g0 = slice(prefix_len, prefix_len + group_size)
        g1 = slice(prefix_len + group_size, prefix_len + 2 * group_size)
        self.assertFalse(mask[g0, g1].any())
        self.assertFalse(mask[g1, g0].any())
        # And no prefix row sees any group column.
        self.assertFalse(mask[:prefix_len, prefix_len:].any())

    def test_prefix_is_strictly_causal(self) -> None:
        """Prefix rows keep the baseline's causal mask, contextual included.

        ``OneRankSTULayer`` forces ``contextual_seq_len = 0``, so there is
        no bidirectional block to carve out here.
        """
        func, seq_offsets, padded_len = _build_func_tensor([(6, 1)], group_size=2)
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        for q in range(6):
            self.assertEqual(mask[q].nonzero().flatten().tolist(), list(range(q + 1)))

    def test_num_targets_over_length_is_clamped(self) -> None:
        """A prefix boundary can never go negative.

        Silent-NaN guard: a negative boundary would make ``col_max0``
        negative, every interval empty, and the attention denominator zero.
        """
        func, seq_offsets, padded_len = _build_func_tensor(
            [(2, 1)], group_size=4, num_targets_override=[12]
        )
        self.assertGreaterEqual(int(func.min()), 0)
        mask = _decode_attn_func_to_mask(func, seq_offsets, padded_len)[0, 0]
        # Boundary clamped to 0 -> every row is a group row, so each row
        # still attends to at least itself.
        self.assertTrue(mask.any(dim=-1).all())

    def test_odd_or_non_positive_group_size_raises(self) -> None:
        """``group_size`` is ``2 * num_tasks``; anything else is a bug."""
        seq_offsets = torch.tensor([0, 4], dtype=torch.int32)
        for bad in (0, -2, 3):
            with self.assertRaisesRegex(ValueError, "group_size"):
                build_onerank_func_tensor(
                    nheads=1,
                    seq_offsets=seq_offsets,
                    total_q=4,
                    num_targets=torch.tensor([2], dtype=torch.int32),
                    group_size=bad,
                )


class OneRankTokenizerTest(unittest.TestCase):
    """Verify the interleaved ``[replica, task token] * K`` expansion."""

    def test_expansion_layout(self) -> None:
        """Slot ``2k`` is the candidate, slot ``2k+1`` is task token ``k``.

        The func tensor's parity rule is hard-coded, so a layout that
        alternates the other way round would mask correctly and mean the
        wrong thing.
        """
        num_tasks, dim, total_targets = 3, 8, 4
        tokenizer = OneRankTokenizer(num_tasks=num_tasks, embedding_dim=dim)
        candidates = torch.randn(total_targets, dim)

        out = tokenizer(candidates)
        self.assertEqual(tuple(out.shape), (total_targets * 2 * num_tasks, dim))
        self.assertEqual(tokenizer.group_size, 2 * num_tasks)
        for i in range(total_targets):
            for k in range(num_tasks):
                base = i * 2 * num_tasks + 2 * k
                torch.testing.assert_close(out[base], candidates[i])
                torch.testing.assert_close(out[base + 1], tokenizer._task_tokens[k])

    def test_replica_gradients_sum_back_into_the_candidate(self) -> None:
        """Replicas are an ``expand``, so K task channels share one input.

        A ``repeat`` would give the same forward values and a K-times
        smaller candidate gradient.
        """
        num_tasks, dim = 4, 5
        tokenizer = OneRankTokenizer(num_tasks=num_tasks, embedding_dim=dim)
        candidates = torch.randn(2, dim, requires_grad=True)

        out = tokenizer(candidates)
        # Weight each replica distinctly so an averaging bug cannot cancel.
        weights = torch.arange(1, out.size(0) + 1, dtype=out.dtype).unsqueeze(-1)
        (out * weights).sum().backward()

        want = torch.zeros_like(candidates)
        for i in range(2):
            for k in range(num_tasks):
                want[i] += float(i * 2 * num_tasks + 2 * k + 1)
        torch.testing.assert_close(candidates.grad, want)

    def test_task_tokens_are_shared_across_candidates(self) -> None:
        """``K`` parameters total, not ``K`` per candidate."""
        tokenizer = OneRankTokenizer(num_tasks=7, embedding_dim=16)
        self.assertEqual(
            [(n, tuple(p.shape)) for n, p in tokenizer.named_parameters()],
            [("_task_tokens", (7, 16))],
        )

    def test_non_positive_num_tasks_raises(self) -> None:
        """There is no group layout without at least one task token."""
        for bad in (0, -1):
            with self.assertRaisesRegex(ValueError, "num_tasks must be positive"):
                OneRankTokenizer(num_tasks=bad, embedding_dim=8)


class OneRankSTULayerTest(unittest.TestCase):
    """Verify the construction-time contract of the OneRank STU layer."""

    _STU_KWARGS = {
        "embedding_dim": 16,
        "num_heads": 2,
        "hidden_dim": 8,
        "attention_dim": 8,
    }

    def _build(self, **overrides: object) -> OneRankSTULayer:
        kwargs = dict(self._STU_KWARGS)
        kwargs.update(overrides)
        return OneRankSTULayer(group_size=kwargs.pop("group_size", 4), **kwargs)

    def test_always_uses_its_own_func_tensor(self) -> None:
        """``uses_arbitrary_mask`` gates the NFUNC path in ``STUStack``."""
        layer = self._build()
        self.assertTrue(layer.uses_arbitrary_mask)

    def test_cache_signature_cannot_collide_with_sla(self) -> None:
        """The cross-layer func-tensor cache is keyed on this string.

        A collision with plain SLA in a mixed stack would silently reuse
        the wrong mask.
        """
        layer = self._build(group_size=14, num_heads=4)
        self.assertEqual(layer.attn_func_static_sig, "onerank:14:4")
        self.assertNotEqual(
            layer.attn_func_static_sig,
            self._build(group_size=14, num_heads=2).attn_func_static_sig,
        )

    @parameterized.expand(
        [
            ("sla_k1", {"sla_k1": 2}, "sla_k1"),
            ("sla_k2", {"sla_k2": 2}, "sla_k2"),
            ("contextual", {"contextual_seq_len": 4}, "contextual_seq_len"),
            ("max_attn_len", {"max_attn_len": 8}, "max_attn_len"),
            ("non_causal", {"causal": False}, "causal"),
            ("odd_group", {"group_size": 3}, "group_size"),
        ]
    )
    def test_unsupported_options_are_rejected(
        self, _name: str, overrides: dict, message: str
    ) -> None:
        """Each of these would need a third column interval in NFUNC=3.

        Rejecting at construction time rather than producing a mask that
        quietly drops one of the two requirements.
        """
        with self.assertRaisesRegex(ValueError, message):
            self._build(**overrides)

    def test_truncation_and_kv_cache_are_unsupported(self) -> None:
        """Both paths would drop the mask; fail loudly instead."""
        layer = self._build()
        x = torch.zeros(6, 16)
        offsets = torch.tensor([0, 6], dtype=torch.int32)
        num_targets = torch.tensor([4], dtype=torch.int32)
        with self.assertRaises(NotImplementedError):
            layer.truncate_input(
                x, offsets, max_seq_len=6, num_targets=num_targets, truncate_tail_len=2
            )
        with self.assertRaises(NotImplementedError):
            layer.cached_forward(x, num_targets)


if __name__ == "__main__":
    unittest.main()
