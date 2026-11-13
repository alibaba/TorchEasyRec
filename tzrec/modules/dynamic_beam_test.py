# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import unittest
from typing import Any, Dict, List, Tuple

import torch
from parameterized import parameterized

from tzrec.modules.dynamic_beam import (
    build_sid_prefix_masks,
    capped_beam_widths,
    dynamic_beam_search,
)
from tzrec.utils.test_util import (
    create_tiny_causal_lm,
    flash_attn_unavailable,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)


def _decode(
    lm,
    ids,
    pairs,
    width=8,
    beam_widths=None,
    attention_mask=None,
    final_topk_per_prefix=False,
    allowed_prefixes=None,
    initial_cache=None,
):
    """Run the kernel over ``pairs`` of inclusive per-level (lo, hi) band edges.

    Defaults to a FLAT schedule: the kernel is policy-free, so only the cases
    about scheduling spell one out. The kernel takes already-capped widths, so
    the helper caps them the way the model does.
    """
    if beam_widths is None:
        beam_widths = [width] * len(pairs)
    return dynamic_beam_search(
        lm,
        lm.get_input_embeddings()(ids),
        torch.ones_like(ids) if attention_mask is None else attention_mask,
        capped_widths=capped_beam_widths(beam_widths, pairs, final_topk_per_prefix),
        bands=pairs,
        final_topk_per_prefix=final_topk_per_prefix,
        allowed_prefixes=allowed_prefixes,
        initial_cache=initial_cache,
    )


class _RowSpy:
    """Backbone wrapper recording the row count of every forward call.

    The kernel calls the model itself, so the per-level beam width -- otherwise
    a local inside the kernel -- becomes observable.
    """

    def __init__(self, lm) -> None:
        self.rows: List[int] = []
        self._lm = lm

    def get_input_embeddings(self):
        return self._lm.get_input_embeddings()

    def __call__(self, **kwargs: Any) -> Any:
        first = kwargs.get("input_ids")
        if first is None:
            first = kwargs["inputs_embeds"]
        self.rows.append(first.shape[0])
        return self._lm(**kwargs)


def _bruteforce_scores(lm, input_ids, attention_mask, pairs):
    """Exact cumulative log-prob of every band combination, by full recompute."""
    ref: Dict[Tuple[int, ...], float] = {}
    with torch.no_grad():
        for combo in itertools.product(*[range(lo, hi + 1) for lo, hi in pairs]):
            seq, mask, total = input_ids, attention_mask, 0.0
            for tok in combo:
                logits = lm(input_ids=seq, attention_mask=mask).logits[0, -1].float()
                total += float(torch.log_softmax(logits, dim=-1)[tok])
                seq = torch.cat([seq, torch.tensor([[tok]])], dim=1)
                mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
            ref[combo] = total
    return ref


class DynamicBeamSearchTest(unittest.TestCase):
    def test_prefix_masks_match_catalog_trie(self) -> None:
        pairs = [(20, 22), (5, 8), (25, 26)]
        allowed = torch.zeros(3, 4, 2, dtype=torch.bool)
        paths = [(0, 1, 0), (0, 3, 1), (2, 2, 1)]
        for path in paths:
            allowed[path] = True
        masks = build_sid_prefix_masks(allowed.flatten(), pairs)
        for depth, mask in enumerate(masks, start=1):
            shape = allowed.shape[:depth]
            actual = {tuple(row) for row in mask.reshape(shape).nonzero().tolist()}
            self.assertEqual(actual, {path[:depth] for path in paths})

    def test_prefix_constraint_prevents_early_pruning_of_only_legal_sid(self) -> None:
        pairs = [(20, 22), (5, 7), (25, 27)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        unmasked = _decode(lm, ids, pairs, beam_widths=[1, 1, 1])[0]
        legal = ((int(unmasked[0]) - 20 + 1) % 3, 1, 0)
        allowed = torch.zeros(3, 3, 3, dtype=torch.bool)
        allowed[legal] = True
        self.assertNotEqual(int(unmasked[0]), legal[0] + 20)
        constrained = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[1, 1, 1],
            allowed_prefixes=build_sid_prefix_masks(allowed.flatten(), pairs),
        )
        self.assertEqual(constrained.tolist(), [[legal[0] + 20, 6, 25]])

    def test_prefix_constraint_preserves_full_vocab_score_order(self) -> None:
        pairs = [(20, 21), (5, 7), (25, 26)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        allowed = torch.zeros(2, 3, 2, dtype=torch.bool)
        paths = [(0, 0, 0), (0, 2, 1), (1, 1, 0), (1, 1, 1)]
        for path in paths:
            allowed[path] = True
        ref = _bruteforce_scores(lm, ids, torch.ones_like(ids), pairs)
        expected = [
            tuple(local + lo for local, (lo, _) in zip(path, pairs)) for path in paths
        ]
        expected.sort(key=ref.__getitem__, reverse=True)
        output = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[2, 6, 2],
            final_topk_per_prefix=True,
            allowed_prefixes=build_sid_prefix_masks(allowed.flatten(), pairs),
        )
        self.assertEqual(
            [tuple(row) for row in output[: len(paths)].tolist()], expected
        )

    def test_single_level_prefix_constraint(self) -> None:
        pairs = [(20, 23)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        allowed = torch.tensor([False, False, True, False])
        output = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[1],
            allowed_prefixes=build_sid_prefix_masks(allowed, pairs),
        )
        self.assertEqual(output.tolist(), [[22]])

    def test_full_catalog_preserves_unconstrained_order(self) -> None:
        pairs = [(20, 21), (5, 7), (25, 26)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        expected = _decode(
            lm, ids, pairs, beam_widths=[2, 4, 2], final_topk_per_prefix=True
        )
        actual = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[2, 4, 2],
            final_topk_per_prefix=True,
            allowed_prefixes=build_sid_prefix_masks(
                torch.ones(12, dtype=torch.bool), pairs
            ),
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_short_catalog_does_not_create_extra_valid_sids(self) -> None:
        pairs = [(20, 22), (5, 7), (25, 27)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        allowed = torch.zeros(3, 3, 3, dtype=torch.bool)
        paths = [(0, 1, 0), (0, 2, 1), (2, 2, 1)]
        for path in paths:
            allowed[path] = True
        output = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[3, 9, 3],
            final_topk_per_prefix=True,
            allowed_prefixes=build_sid_prefix_masks(allowed.flatten(), pairs),
        )
        local = output - torch.tensor([lo for lo, _ in pairs])
        valid = local[allowed[local[:, 0], local[:, 1], local[:, 2]]]
        self.assertEqual(len(valid), len(paths))
        self.assertEqual({tuple(row) for row in valid.tolist()}, set(paths))

    @parameterized.expand(
        [
            [[20, 20, 200]],
            [[50, 50, 200]],
            [[100, 100, 200]],
            [[50, 100, 200]],
            [[10, 10, 200]],
            [[20, 50, 200]],
            [[20, 20, 20]],
            [[50, 50, 50]],
            [[100, 100, 100]],
            [[20, 50, 100]],
        ],
        name_func=parameterized_name_func,
    )
    def test_final_prefix_topk_capacity(self, widths) -> None:
        pairs = [(0, 255), (256, 511), (512, 767)]
        capped = capped_beam_widths(widths, pairs, True)
        self.assertEqual(capped, [widths[0], widths[1], widths[1] * widths[2]])
        self.assertGreaterEqual(capped[-1], 200)

    def test_final_prefix_topk_matches_exhaustive_reference(self) -> None:
        pairs = [(20, 21), (5, 7), (25, 26)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        ref = _bruteforce_scores(lm, ids, torch.ones_like(ids), pairs)
        by_parent = {}
        for seq, score in ref.items():
            parent = seq[:-1]
            if parent not in by_parent or score > ref[by_parent[parent]]:
                by_parent[parent] = seq
        expected = sorted(by_parent.values(), key=ref.__getitem__, reverse=True)
        out = _decode(lm, ids, pairs, beam_widths=[2, 6, 1], final_topk_per_prefix=True)
        self.assertEqual([tuple(row) for row in out.tolist()], expected)

    def test_final_prefix_topk_band_cap(self) -> None:
        pairs = [(20, 21), (22, 24), (25, 26)]
        self.assertEqual(capped_beam_widths([2, 6, 200], pairs, True), [2, 6, 12])

    def test_prefix_constraint_is_applied_before_final_local_topk(self) -> None:
        pairs = [(20, 21), (5, 7), (25, 26)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        ref = _bruteforce_scores(lm, ids, torch.ones_like(ids), pairs)
        allowed = torch.zeros(2, 3, 2, dtype=torch.bool)
        expected = []
        for first in range(2):
            for second in range(3):
                choices = [(first + 20, second + 5, last + 25) for last in range(2)]
                worst = min(choices, key=ref.__getitem__)
                allowed[first, second, worst[-1] - 25] = True
                expected.append(worst)
        expected.sort(key=ref.__getitem__, reverse=True)
        output = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[2, 6, 1],
            final_topk_per_prefix=True,
            allowed_prefixes=build_sid_prefix_masks(allowed.flatten(), pairs),
        )
        self.assertEqual([tuple(row) for row in output.tolist()], expected)

    def test_final_prefix_topk_keeps_global_top200_when_local_k_is_200(self) -> None:
        pairs = [(0, 255), (256, 511), (512, 767)]
        lm = create_tiny_causal_lm(vocab_size=768)
        ids = torch.tensor([[5, 6, 7, 8]])
        expected = _decode(lm, ids, pairs, beam_widths=[20, 20, 200])
        actual = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[20, 20, 200],
            final_topk_per_prefix=True,
        )
        self.assertEqual(tuple(actual.shape), (4000, 3))
        self.assertTrue(torch.equal(actual[:200], expected))

    def test_final_prefix_topk_batch_matches_solo(self) -> None:
        pairs = [(20, 21), (22, 24), (25, 28)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8], [0, 9, 10, 11]])
        mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        out = _decode(
            lm,
            ids,
            pairs,
            beam_widths=[2, 4, 2],
            attention_mask=mask,
            final_topk_per_prefix=True,
        ).reshape(2, 8, 3)
        for index, solo in enumerate([ids[:1], ids[1:, 1:]]):
            expected = _decode(
                lm, solo, pairs, beam_widths=[2, 4, 2], final_topk_per_prefix=True
            )
            self.assertTrue(torch.equal(out[index], expected))

    @parameterized.expand(
        [
            # every request satisfiable -> the schedule is honoured verbatim
            [[4, 8, 16], [(20, 27), (28, 34), (35, 40)], [4, 8, 16]],
            # capped by band x surviving prefixes, not by what was asked
            [[6, 12], [(20, 21), (22, 24)], [2, 6]],
            # a width-1 band collapses level 0 and bounds everything after it
            [[4, 8], [(20, 20), (21, 23)], [1, 3]],
            # a flat (non-doubling) schedule is just as valid to the kernel
            [[3, 3, 3], [(20, 27), (28, 34), (35, 40)], [3, 3, 3]],
        ],
        name_func=parameterized_name_func,
    )
    def test_width_schedule(self, beam_widths, pairs, expected_widths) -> None:
        spy = _RowSpy(create_tiny_causal_lm(vocab_size=48))
        ids = torch.tensor([[5, 6, 7, 8]])
        out = _decode(spy, ids, pairs, beam_widths=beam_widths)
        # calls: [prompt (1 row)] + one per level>0, each carrying widths[j-1] rows
        widths = spy.rows[1:] + [out.shape[0]]
        self.assertEqual(spy.rows[0], 1)
        self.assertEqual(widths, expected_widths)
        self.assertEqual(tuple(out.shape), (expected_widths[-1], len(pairs)))

    def test_ragged_batch_rows_match_solo_runs(self) -> None:
        # every row of a ragged batch must decode exactly as if run alone.
        pairs = [(20, 21), (22, 24), (25, 28)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8], [0, 9, 10, 11]])
        am = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        out = _decode(lm, ids, pairs, attention_mask=am)
        width = out.shape[0] // 2
        solos = [torch.tensor([[5, 6, 7, 8]]), torch.tensor([[9, 10, 11]])]
        for i, solo_ids in enumerate(solos):
            solo = _decode(lm, solo_ids, pairs)
            self.assertTrue(torch.equal(out[i * width : (i + 1) * width], solo))

    def test_exhaustive_matches_bruteforce_topk(self) -> None:
        # widths [2, 6, 12] over 2*3*2 = 12 combinations -> no pruning at any
        # level. Disjoint, non-monotonic bands also verify that each level uses
        # its own bounds rather than assuming one contiguous SID layout.
        pairs = [(20, 21), (5, 7), (25, 26)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        am = torch.ones_like(ids)
        out = _decode(lm, ids, pairs, width=12)
        got = [tuple(r) for r in out.tolist()]
        ref = _bruteforce_scores(lm, ids, am, pairs)
        self.assertEqual(set(got), set(ref))
        self.assertEqual(len(got), len(ref))
        scores = [ref[c] for c in got]
        self.assertTrue(
            all(scores[i] >= scores[i + 1] - 1e-4 for i in range(len(scores) - 1))
        )
        self.assertEqual(got[0], max(ref, key=lambda combo: ref[combo]))


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
@unittest.skipIf(*flash_attn_unavailable)
class DynamicBeamSearchFlashAttentionTest(unittest.TestCase):
    @parameterized.expand(
        [
            ["qwen2", [50, 50, 50]],
            ["qwen3", [50, 50, 50]],
            ["qwen2", [20, 20, 200]],
            ["qwen3", [20, 20, 200]],
        ],
        name_func=parameterized_name_func,
    )
    def test_prefix_constrained_top200_matches_shared_fa2_cache(
        self, model_type: str, beam_widths: List[int]
    ) -> None:
        from tzrec.modules.flash_attention_beam_cache import (
            create_flash_attention_beam_cache,
        )

        device = torch.device("cuda")
        lm = create_tiny_causal_lm(
            768,
            model_type=model_type,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device)
        ids = torch.tensor([[5, 6, 7, 8], [0, 9, 10, 11]], device=device)
        attention_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], device=device)
        pairs = [(0, 255), (256, 511), (512, 767)]
        allowed = torch.zeros(256, 256, 256, dtype=torch.bool, device=device)
        allowed[::2, ::3, ::5] = True
        prefixes = build_sid_prefix_masks(allowed.flatten(), pairs)
        options = dict(
            beam_widths=beam_widths,
            attention_mask=attention_mask,
            final_topk_per_prefix=True,
            allowed_prefixes=prefixes,
        )
        ordinary = _decode(lm, ids, pairs, **options)
        shared = _decode(
            lm,
            ids,
            pairs,
            initial_cache=create_flash_attention_beam_cache(lm),
            **options,
        )
        self.assertTrue(torch.equal(shared, ordinary))
        capacity = beam_widths[1] * beam_widths[2]
        self.assertEqual(tuple(shared.shape), (2 * capacity, 3))
        top200 = shared.reshape(2, capacity, 3)[:, :200]
        local = top200 - torch.tensor([lo for lo, _ in pairs], device=device)
        self.assertTrue(allowed[local[..., 0], local[..., 1], local[..., 2]].all())

    @parameterized.expand(
        [["qwen2"], ["qwen3"]],
        name_func=parameterized_name_func,
    )
    def test_ragged_batch_matches_solo_dynamic_cache(self, model_type: str) -> None:
        device = torch.device("cuda")
        lm = create_tiny_causal_lm(
            30,
            model_type=model_type,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device)
        pairs = [(20, 21), (22, 24), (25, 28)]
        ids = torch.tensor(
            [[5, 6, 7, 8], [0, 9, 10, 11]],
            device=device,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1], [0, 1, 1, 1]],
            device=device,
        )

        out = _decode(lm, ids, pairs, attention_mask=attention_mask)
        width = out.shape[0] // 2
        solos = [
            torch.tensor([[5, 6, 7, 8]], device=device),
            torch.tensor([[9, 10, 11]], device=device),
        ]
        for index, solo_ids in enumerate(solos):
            solo = _decode(lm, solo_ids, pairs)
            torch.testing.assert_close(out[index * width : (index + 1) * width], solo)


if __name__ == "__main__":
    unittest.main()
