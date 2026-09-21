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

"""Unit tests for ``tzrec.modules.gr.onerank_sd`` (CPU only).

``_varlen_single_query_attn`` (the CUDA fast path) is pinned on GPU in the
``JaggedCrossAttentionVarlenDispatchTest``; everything else here pins the
reference math, whose values must stay identical to the historical
implementation.
"""

import unittest

import torch

from tzrec.modules.gr.onerank_sd import (
    JaggedCrossAttention,
    OneRankSituationDiscernment,
    _jagged_softmax,
)
from tzrec.ops import Kernel
from tzrec.utils.test_util import mark_ci_scope

_NUM_CANDIDATES = [2, 4]
_TOTAL = sum(_NUM_CANDIDATES)


@mark_ci_scope("h20", "gpu")
class JaggedCrossAttentionTest(unittest.TestCase):
    """Value-level tests for the single-query jagged MHCA of SD.

    Shape / finiteness / "candidates don't tie" assertions cannot catch
    the two failure modes that actually matter for a jagged single-query
    MHCA: a query pooling over the *wrong request's* candidates
    (segment-locality leak), or a softmax denominator that normalizes
    globally instead of per segment (that only rescales ``h_k``).  Both
    are pinned here with identity projections and a zeroed attention
    scale: every logit is zero, so the exact expected output is the
    *segment-local* uniform mean of the values.
    """

    _DIM = 6

    def _identity_attention(self) -> JaggedCrossAttention:
        torch.manual_seed(0)
        module = JaggedCrossAttention(embedding_dim=self._DIM, num_heads=1)
        for linear in (
            module._q_proj,
            module._k_proj,
            module._v_proj,
            module._out_proj,
        ):
            torch.nn.init.eye_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
        # logits = (q_rows * k).sum(-1) * scale -> all zero: uniform
        # attention within each segment, no scale effects to reason about.
        module._attn_scale = 0.0
        module.eval()
        return module

    def test_pools_over_each_segments_own_rows_only(self) -> None:
        """The output is the segment-local uniform mean of the values.

        A locality leak (query seeing another request's rows) or a global
        softmax denominator would both change these means.
        """
        module = self._identity_attention()
        torch.manual_seed(1)
        pool = torch.randn(6, self._DIM)  # two requests of 2 and 4 rows
        query = torch.zeros(2, self._DIM)
        lengths = torch.tensor([2, 4])

        out = module(query, pool, lengths)

        torch.testing.assert_close(out[0], pool[:2].mean(dim=0))
        torch.testing.assert_close(out[1], pool[2:].mean(dim=0))
        # The two requests are independent: changing one request's pool
        # leaves the other's context bit-identical.
        perturbed = pool.clone()
        perturbed[2:] += 100.0
        out_perturbed = module(query, perturbed, lengths)
        torch.testing.assert_close(out_perturbed[0], out[0])

    def test_empty_segment_gets_only_the_output_bias(self) -> None:
        """The documented empty-segment contract."""
        module = self._identity_attention()
        bias = torch.arange(1.0, self._DIM + 1.0)
        module._out_proj.bias.data = bias
        torch.manual_seed(2)
        pool = torch.randn(2, self._DIM)
        query = torch.zeros(2, self._DIM)

        out = module(query, pool, torch.tensor([0, 2]))

        torch.testing.assert_close(out[0], bias)
        torch.testing.assert_close(out[1], pool.mean(dim=0) + bias)

    def test_matches_dense_softmax_reference(self) -> None:
        """Random projections, two heads, live scale: the full math.

        The identity/zero-scale tests above never exercise the weighted
        attention -- every logit is zero, so a wrong
        ``(q_rows * k).sum(dim=-1)`` numerator, a head-order scramble in
        the multi-head collapse, or a per-segment max-shift regression in
        ``_jagged_softmax`` would all pass them.  This compares against a
        per-segment dense ``torch.softmax`` reference, ``num_heads=2``, so
        both head slices and the head-major reshape are value-pinned.
        """
        torch.manual_seed(5)
        module = JaggedCrossAttention(embedding_dim=self._DIM, num_heads=2)
        module.eval()
        lengths = torch.tensor([1, 3, 2])
        pool = torch.randn(int(lengths.sum()), self._DIM)
        query = torch.randn(lengths.size(0), self._DIM)

        out = module(query, pool, lengths)

        num_heads = module._num_heads
        head_dim = module._head_dim
        q = module._q_proj(query).view(-1, num_heads, head_dim)
        k = module._k_proj(pool).view(-1, num_heads, head_dim)
        v = module._v_proj(pool).view(-1, num_heads, head_dim)
        contexts = []
        offset = 0
        for request_idx, num in enumerate(lengths.tolist()):
            rows = slice(offset, offset + num)
            logits = (
                torch.einsum("hd,nhd->hn", q[request_idx], k[rows]) * module._attn_scale
            )
            attn = torch.softmax(logits, dim=-1)
            # Head-major collapse, matching the reference path's
            # ``(attn.unsqueeze(-1) * v).reshape(-1, num_heads * head_dim)``.
            contexts.append(torch.einsum("hn,nhd->hd", attn, v[rows]).reshape(-1))
            offset += num
        expected = module._out_proj(torch.stack(contexts))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


@mark_ci_scope("h20", "gpu")
class JaggedCrossAttentionVarlenDispatchTest(unittest.TestCase):
    """The CUDA varlen fast path against the reference math.

    The two paths must agree to kernel rounding on an eligible batch
    (CUDA + bf16 + no dropout + no empty segment), and an empty segment
    must stay on the reference path so the documented bias-only contract
    never depends on kernel-specific empty-KV behaviour.
    """

    _DIM = 16

    def _bf16_pair(self):
        """Two same-parameter modules: varlen-eligible and forced-reference.

        ``dropout_ratio`` above zero keeps the second module on the
        reference path even in eval mode, where ``F.dropout`` is a no-op
        -- so the only difference between the two runs is the code path.
        """
        torch.manual_seed(11)
        fast = JaggedCrossAttention(embedding_dim=self._DIM, num_heads=2)
        slow = JaggedCrossAttention(
            embedding_dim=self._DIM, num_heads=2, dropout_ratio=0.5
        )
        slow.load_state_dict(fast.state_dict())
        for module in (fast, slow):
            module.to(device="cuda", dtype=torch.bfloat16)
            module.eval()
        return fast, slow

    def test_varlen_path_matches_reference_on_bf16_cuda(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        fast, slow = self._bf16_pair()
        torch.manual_seed(12)
        lengths = torch.tensor([2, 4, 3], device="cuda")
        pool = torch.randn(
            int(lengths.sum()), self._DIM, device="cuda", dtype=torch.bfloat16
        )
        query = torch.randn(
            lengths.size(0), self._DIM, device="cuda", dtype=torch.bfloat16
        )

        torch.testing.assert_close(
            fast(query, pool, lengths),
            slow(query, pool, lengths),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_empty_segment_stays_on_reference_path(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        fast, slow = self._bf16_pair()
        torch.manual_seed(13)
        pool = torch.randn(2, self._DIM, device="cuda", dtype=torch.bfloat16)
        query = torch.randn(2, self._DIM, device="cuda", dtype=torch.bfloat16)
        lengths = torch.tensor([0, 2], device="cuda")

        fast_out = fast(query, pool, lengths)
        slow_out = slow(query, pool, lengths)
        # The guard routes the empty-segment batch to the reference path,
        # so the two paths agree AND the empty segment's row is exactly
        # the output projection's bias.
        torch.testing.assert_close(fast_out, slow_out)
        torch.testing.assert_close(
            fast_out[0].to(torch.float32),
            fast._out_proj.bias.to(torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )


class JaggedSoftmaxTest(unittest.TestCase):
    """Direct value-level pin of ``_jagged_softmax``.

    Every other use in these tests flows through attention paths that
    zero the logits first; non-uniform logits are what expose a
    per-segment max-shift or denominator regression.
    """

    def test_normalizes_each_segment_and_column_independently(self) -> None:
        torch.manual_seed(6)
        lengths = torch.tensor([1, 3, 2])
        # Two columns (heads) with independent random patterns, so a
        # column mix-up or a cross-segment normalization both fail loudly.
        logits = torch.randn(int(lengths.sum()), 2)

        weights = _jagged_softmax(logits, lengths)

        offset = 0
        for num in lengths.tolist():
            rows = slice(offset, offset + num)
            torch.testing.assert_close(
                weights[rows], torch.softmax(logits[rows], dim=0)
            )
            offset += num


class OneRankSituationDiscernmentTest(unittest.TestCase):
    """Value-level pins of the per-task wiring in SD.

    ``forward`` drives three parallel per-task lists (query projection,
    pool channel, attention) with the same ``task_idx`` and stacks the
    results in task order.  A permutation of any of the three leaves
    every shape/finite assertion in the model-level tests green while
    silently mis-ordering ``z_k`` against ``r^i_k`` in the scorer.  The
    tests here pin the wiring with values: one hand-computable
    (identity projections, uniform attention), one exhaustive over the
    per-task modules.
    """

    _DIM = 6
    _NUM_TASKS = 3

    def _identity_sd(self) -> OneRankSituationDiscernment:
        torch.manual_seed(0)
        module = OneRankSituationDiscernment(
            embedding_dim=self._DIM,
            num_tasks=self._NUM_TASKS,
            contextual_feature_dim=self._DIM,
            num_heads=2,
        )
        for linear in module._query_projs:
            torch.nn.init.eye_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
        for attention in module._attentions:
            for linear in (
                attention._q_proj,
                attention._k_proj,
                attention._v_proj,
                attention._out_proj,
            ):
                torch.nn.init.eye_(linear.weight)
                torch.nn.init.zeros_(linear.bias)
            # Zero logits: the expected pooling is the segment-local
            # uniform mean, and the (LayerNorm-ed) query cannot matter.
            attention._attn_scale = 0.0
        # The default TRITON LayerNorm is CUDA-only; the reference
        # math here is kernel-independent.
        module.set_kernel(Kernel.PYTORCH)
        module.eval()
        return module

    def test_output_channel_k_pools_task_channel_k(self) -> None:
        """``forward(...)[:, k]`` is the mean of task channel ``k`` rows."""
        module = self._identity_sd()
        torch.manual_seed(7)
        task_embeddings = torch.randn(_TOTAL, self._NUM_TASKS, self._DIM)
        contextual = torch.randn(len(_NUM_CANDIDATES), self._DIM)

        out = module(contextual, task_embeddings, torch.tensor(_NUM_CANDIDATES))

        offset = 0
        for request_idx, num in enumerate(_NUM_CANDIDATES):
            rows = slice(offset, offset + num)
            for task_idx in range(self._NUM_TASKS):
                torch.testing.assert_close(
                    out[request_idx, task_idx],
                    task_embeddings[rows, task_idx].mean(dim=0),
                )
            offset += num

    def test_per_task_modules_stay_paired(self) -> None:
        """Query ``k``, pool channel ``k`` and attention ``k`` stay paired."""
        torch.manual_seed(8)
        module = OneRankSituationDiscernment(
            embedding_dim=self._DIM,
            num_tasks=self._NUM_TASKS,
            contextual_feature_dim=self._DIM,
            num_heads=2,
        )
        module.set_kernel(Kernel.PYTORCH)
        module.eval()
        torch.manual_seed(9)
        task_embeddings = torch.randn(_TOTAL, self._NUM_TASKS, self._DIM)
        contextual = torch.randn(len(_NUM_CANDIDATES), self._DIM)

        out = module(contextual, task_embeddings, torch.tensor(_NUM_CANDIDATES))

        for task_idx in range(self._NUM_TASKS):
            query = module._query_norms[task_idx](
                module._query_projs[task_idx](contextual)
            )
            expected = module._attentions[task_idx](
                query=query,
                pool=task_embeddings[:, task_idx, :].contiguous(),
                lengths=torch.tensor(_NUM_CANDIDATES),
            )
            torch.testing.assert_close(out[:, task_idx, :], expected)


if __name__ == "__main__":
    unittest.main()
