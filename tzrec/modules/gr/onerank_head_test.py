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

"""Unit tests for ``tzrec.modules.gr.onerank_head`` (CPU only).

The scorer variants (``scorer_type``) and ``task_bias_init`` exist because
of the constant-prediction plateau of the rank-1 scorer (see the model
docs); the tests here pin down the contract
each variant is supposed to keep:

* BILINEAR starts *exactly* where DOT_PRODUCT would (identity ``W_k``);
* ``task_bias_init`` shifts every (candidate, task) logit by ``b_k`` and
  nothing else;
* MLP discriminates a request's candidates straight from random init;
* Strategic Gradient Detachment zeroes the off-diagonal gradients of the
  cross-task attention and keeps the diagonal alive.
"""

import math
import unittest

import torch
from parameterized import parameterized

from tzrec.modules.gr.onerank_cross_task import (
    CASCADE,
    FULL,
    HYBRID,
    PARALLEL,
    OneRankCrossTaskAttention,
    build_cross_task_mask,
)
from tzrec.modules.gr.onerank_head import OneRankPredictionHead
from tzrec.modules.gr.onerank_jagged import jagged_softmax
from tzrec.modules.gr.onerank_sd import (
    JaggedCrossAttention,
    OneRankSituationDiscernment,
)
from tzrec.ops import Kernel
from tzrec.utils.test_util import (
    TestGraphType,
    create_test_module,
    mark_ci_scope,
)

_GRAPH_TYPES = [TestGraphType.NORMAL, TestGraphType.FX_TRACE]
_SCORER_TYPES = ["dot_product", "bilinear", "mlp"]

_EMBEDDING_DIM = 16
_NUM_HEADS = 2
_TASK_NAMES = ["is_click", "is_like", "is_comment"]
_NUM_TASKS = len(_TASK_NAMES)
_CONTEXTUAL_DIM = 8
# Ragged on purpose: two requests of 2 and 4 candidates.
_NUM_CANDIDATES = [2, 4]
_TOTAL = sum(_NUM_CANDIDATES)
_TASK_BIAS = [0.25, -0.5, 1.0]


def _inputs(seed: int = 7) -> tuple:
    torch.manual_seed(seed)
    task_embeddings = torch.randn(_TOTAL, _NUM_TASKS, _EMBEDDING_DIM)
    contextual = torch.randn(len(_NUM_CANDIDATES), _CONTEXTUAL_DIM)
    num_candidates = torch.tensor(_NUM_CANDIDATES)
    return task_embeddings, num_candidates, contextual


def _head(
    scorer_type: str = "dot_product",
    task_bias_init=None,
    with_sd: bool = True,
    with_cross_task: bool = True,
) -> OneRankPredictionHead:
    torch.manual_seed(11)
    head = OneRankPredictionHead(
        embedding_dim=_EMBEDDING_DIM,
        task_names=_TASK_NAMES,
        contextual_feature_dim=_CONTEXTUAL_DIM,
        situation_discernment=({"num_heads": _NUM_HEADS} if with_sd else None),
        cross_task_head=(
            {"num_heads": _NUM_HEADS, "ffn_hidden_dim": 8} if with_cross_task else None
        ),
        scorer_type=scorer_type,
        scorer_hidden_dim=12,
        task_bias_init=task_bias_init,
    )
    # The default kernel (TRITON) is CUDA-only; these tests run on CPU.
    head.set_kernel(Kernel.PYTORCH)
    return head


@mark_ci_scope("h20", "gpu")
class OneRankPredictionHeadTest(unittest.TestCase):
    """Tests for the scorer variants and ``task_bias_init``."""

    @parameterized.expand(
        [
            (f"{scorer}_{gt.name.lower()}", scorer, gt)
            for gt in _GRAPH_TYPES
            for scorer in _SCORER_TYPES
        ]
    )
    def test_scorer_variants_produce_discriminating_logits(
        self, name, scorer_type, graph_type
    ) -> None:
        """Every scorer emits one finite logit per (candidate, task).

        ``FX_TRACE`` matters in its own right: the ``output_size=`` argument
        added to ``repeat_interleave`` keeps the broadcast sync-free, which
        only holds if the static size survives symbolic tracing.
        """
        head = _head(scorer_type=scorer_type)
        module = create_test_module(head, graph_type)
        task_embeddings, num_candidates, contextual = _inputs()

        logits = module(task_embeddings, num_candidates, contextual)

        self.assertEqual(logits.size(), (_TOTAL, _NUM_TASKS))
        self.assertTrue(torch.isfinite(logits).all())
        # Candidates of one request must not tie under any scorer: the MLP
        # by construction, the rank-1 pair because two random projections
        # onto one direction tie with probability zero.
        for k in range(_NUM_TASKS):
            self.assertFalse(
                torch.allclose(logits[: _NUM_CANDIDATES[0], k], logits[0, k]),
                msg=f"{scorer_type} ties a request's candidates on task {k}",
            )

    def test_bilinear_starts_where_dot_product_does(self) -> None:
        """Identity ``W_k`` makes BILINEAR equal DOT_PRODUCT, exactly.

        This is the contract that lets the two be compared at step 0: only
        the *ability* to leave the rank-1 subspace changes, not the
        starting point.
        """
        dot = _head(scorer_type="dot_product")
        bilinear = _head(scorer_type="bilinear")
        # Shared submodules (SD, cross-task, bias) identical by copy, so
        # the only difference left is the scorer itself.
        bilinear.load_state_dict(dot.state_dict(), strict=False)
        torch.testing.assert_close(
            bilinear._bilinear_weight.detach(),
            torch.eye(_EMBEDDING_DIM).repeat(_NUM_TASKS, 1, 1),
        )

        task_embeddings, num_candidates, contextual = _inputs()
        torch.testing.assert_close(
            bilinear(task_embeddings, num_candidates, contextual),
            dot(task_embeddings, num_candidates, contextual),
        )

    def test_bilinear_weight_is_trainable_off_the_identity(self) -> None:
        """One step of SGD must be able to move ``W_k`` off the identity.

        The identity init is a starting point, not a frozen prior: if the
        gradient never reached ``W_k`` the variant would be a no-op.
        """
        bilinear = _head(scorer_type="bilinear")
        task_embeddings, num_candidates, contextual = _inputs()

        bilinear(task_embeddings, num_candidates, contextual).sum().backward()
        self.assertIsNotNone(bilinear._bilinear_weight.grad)
        self.assertGreater(bilinear._bilinear_weight.grad.abs().max().item(), 0.0)

    def test_mlp_scorer_has_one_mlp_per_task(self) -> None:
        """Each task owns its whole MLP; nothing is shared between tasks."""
        mlp = _head(scorer_type="mlp")
        self.assertEqual(len(mlp._task_mlps), _NUM_TASKS)
        # Different parameters per task -- the point of the per-task scorer.
        first = mlp._task_mlps[0][0].weight
        for k in range(1, _NUM_TASKS):
            self.assertFalse(
                torch.allclose(first, mlp._task_mlps[k][0].weight),
                msg=f"task {k} shares task 0's MLP weights",
            )
        # And the hidden dim follows scorer_hidden_dim.
        self.assertEqual(mlp._task_mlps[0][0].out_features, 12)

    def test_task_bias_init_shifts_every_logit_by_b_k(self) -> None:
        """``task_bias_init`` is a pure per-task logit shift.

        Everything else (pooling, attention, scorer) must be untouched:
        the difference of the two heads is exactly ``b_k`` broadcast over
        the request's candidates.
        """
        plain = _head(scorer_type="mlp")
        biased = _head(scorer_type="mlp", task_bias_init=_TASK_BIAS)
        # Same-seed construction makes every shared parameter identical;
        # load_state_dict would also copy the *zero* bias over, so re-apply
        # the configured one after it.
        biased.load_state_dict(plain.state_dict(), strict=False)
        with torch.no_grad():
            biased._task_bias.copy_(torch.tensor(_TASK_BIAS))

        task_embeddings, num_candidates, contextual = _inputs()
        # Broadcasting (total, K) + (K,) adds b_k to column k for every
        # candidate -- exactly the claimed semantics.
        torch.testing.assert_close(
            biased(task_embeddings, num_candidates, contextual),
            plain(task_embeddings, num_candidates, contextual)
            + torch.tensor(_TASK_BIAS),
        )
        torch.testing.assert_close(biased._task_bias.detach(), torch.tensor(_TASK_BIAS))

    def test_task_bias_defaults_to_zero(self) -> None:
        """No ``task_bias_init`` keeps the zeros of previous runs."""
        plain = _head()
        torch.testing.assert_close(plain._task_bias.detach(), torch.zeros(_NUM_TASKS))

    def test_bad_configs_are_rejected(self) -> None:
        """Config errors fail at construction, not at the first forward."""
        with self.assertRaisesRegex(ValueError, "unknown scorer_type"):
            _head(scorer_type="nonsense")
        with self.assertRaisesRegex(ValueError, "task_bias_init has"):
            _head(task_bias_init=[0.1, 0.2])

    def test_mean_pooling_fallback_matches_reference(self) -> None:
        """Without SD / cross-task the scorer is the plain mean pool.

        ``z_k = mean_i r^i_k`` per request, ``s^i_k = z_k . r^i_k /
        sqrt(D)``: a straight transcription of paper 2.4's degenerate case
        (ablations V5 and V3 both off).
        """
        head = _head(scorer_type="dot_product", with_sd=False, with_cross_task=False)
        task_embeddings, num_candidates, _ = _inputs(seed=3)

        got = head(task_embeddings, num_candidates)
        scale = 1.0 / math.sqrt(_EMBEDDING_DIM)
        want = torch.zeros(_TOTAL, _NUM_TASKS)
        start = 0
        for n in _NUM_CANDIDATES:
            segment = task_embeddings[start : start + n]
            z = segment.mean(dim=0)  # (K, D)
            want[start : start + n] = (z * segment).sum(dim=-1) * scale
            start += n
        torch.testing.assert_close(got, want)

    def test_situation_discernment_requires_contextual(self) -> None:
        """SD configured but no contextual embeddings -> loud failure."""
        head = _head()
        task_embeddings, num_candidates, _ = _inputs()
        with self.assertRaisesRegex(ValueError, "contextual"):
            head(task_embeddings, num_candidates)


@mark_ci_scope("h20", "gpu")
class OneRankCrossTaskDetachmentTest(unittest.TestCase):
    """Strategic Gradient Detachment semantics (paper 2.4).

    Under detachment the off-diagonal key/value source is a constant: a
    loss on task ``k``'s output must not write gradients into any other
    task's input vector, while task ``k``'s own channel stays trainable.
    """

    K = 3
    D = 8

    def _request_vectors(self, seed: int = 5) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(2, self.K, self.D)

    def _cross_task(self, gradient_detachment: bool) -> OneRankCrossTaskAttention:
        torch.manual_seed(9)
        cross_task = OneRankCrossTaskAttention(
            embedding_dim=self.D,
            task_names=_TASK_NAMES,
            gradient_detachment=gradient_detachment,
            num_heads=2,
            ffn_hidden_dim=8,
        )
        # The default kernel (TRITON) is CUDA-only; these tests run on CPU.
        cross_task.set_kernel(Kernel.PYTORCH)
        return cross_task

    def _loss_on_task(self, cross_task, request_vectors, k: int) -> torch.Tensor:
        """A loss on task ``k``'s output channel with non-constant gradient.

        A plain ``.sum()`` is annihilated by the LayerNorms on the way
        (a constant upstream gradient has zero projection off the LN
        output), which would make every attention-path assertion
        vacuous; random coefficients keep the path alive.
        """
        out = cross_task(request_vectors)
        torch.manual_seed(13)
        return (out[:, k] * torch.randn_like(out[:, k])).sum()

    def test_detachment_blocks_off_diagonal_gradients(self) -> None:
        """Loss on task 2 writes only into task 2's input channel.

        Under the default CASCADE mask task 2 *reads* tasks 0 and 1; the
        assertion is that reading stays read-only once
        ``gradient_detachment`` is on.
        """
        cross_task = self._cross_task(gradient_detachment=True)
        request_vectors = self._request_vectors().requires_grad_(True)

        self._loss_on_task(cross_task, request_vectors, 2).backward()
        grad = request_vectors.grad
        # Off-diagonal channels are constants under detachment.
        self.assertEqual(grad[:, :2].abs().max().item(), 0.0)
        # The own channel keeps learning.
        self.assertGreater(grad[:, 2].abs().max().item(), 0.0)

    def test_no_detachment_lets_off_diagonal_flow(self) -> None:
        """The switch is real: without detachment the gradient does flow.

        Same setup as the detached test except the flag, so the pair pins
        the exact semantic of the knob.
        """
        cross_task = self._cross_task(gradient_detachment=False)
        request_vectors = self._request_vectors().requires_grad_(True)

        self._loss_on_task(cross_task, request_vectors, 2).backward()
        grad = request_vectors.grad
        # CASCADE lets task 2 read 0 and 1; with detachment off, reading
        # also writes.
        self.assertGreater(grad[:, :2].abs().max().item(), 0.0)
        self.assertGreater(grad[:, 2].abs().max().item(), 0.0)

    def test_detached_and_undetached_forwards_agree(self) -> None:
        """Detachment is a gradient-path property only.

        The forward values must be identical, so toggling the flag never
        changes what the model predicts -- only what trains.
        """
        detached = self._cross_task(gradient_detachment=True)
        undetached = self._cross_task(gradient_detachment=False)
        undetached.load_state_dict(detached.state_dict())

        request_vectors = self._request_vectors()
        torch.testing.assert_close(
            detached(request_vectors), undetached(request_vectors)
        )

    def test_detachment_keeps_projection_weights_learning(self) -> None:
        """The shared k/v projections still learn from every pair.

        Strategic Gradient Detachment detaches ``h_j``, not the shared
        projections.  The forward is identical either way, so the k/v
        projection weight gradients must match the undetached run --
        including the off-diagonal (k, j) contributions -- while the
        input gradients must not (that is the part detachment cuts).
        """
        detached = self._cross_task(gradient_detachment=True)
        undetached = self._cross_task(gradient_detachment=False)
        undetached.load_state_dict(detached.state_dict())

        rv_detached = self._request_vectors().requires_grad_(True)
        rv_undetached = self._request_vectors().requires_grad_(True)
        self._loss_on_task(detached, rv_detached, 2).backward()
        self._loss_on_task(undetached, rv_undetached, 2).backward()

        torch.testing.assert_close(
            detached._k_proj.weight.grad, undetached._k_proj.weight.grad
        )
        torch.testing.assert_close(
            detached._v_proj.weight.grad, undetached._v_proj.weight.grad
        )
        # ... while h_j stays read-only under detachment: the undetached
        # run accumulates the off-diagonal reads, the detached one does
        # not.
        self.assertEqual(rv_detached.grad[:, :2].abs().max().item(), 0.0)
        self.assertGreater(rv_undetached.grad[:, :2].abs().max().item(), 0.0)


@mark_ci_scope("h20", "gpu")
class BuildCrossTaskMaskTest(unittest.TestCase):
    """Table-driven tests for ``build_cross_task_mask``.

    ``mask_type`` / ``hybrid_chain_task_names`` are user-facing proto
    fields, and every branch of the builder -- the four mask types, the
    HYBRID cascade loop (including a chain ordered against
    ``task_configs``), and all four error paths -- is pinned here without
    going through the attention module.
    """

    @parameterized.expand(
        [
            (
                "cascade_is_lower_triangle",
                CASCADE,
                None,
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, True],
                ],
            ),
            (
                "parallel_is_identity",
                PARALLEL,
                None,
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ],
            ),
            (
                "full_is_all_ones",
                FULL,
                None,
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
            ),
            (
                "hybrid_full_chain_equals_cascade",
                HYBRID,
                ["is_click", "is_like", "is_comment"],
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, True],
                ],
            ),
            (
                # The chain decides the cascade direction regardless of
                # task_configs order: is_comment is the innermost task, so
                # is_click may read it, and is_like stays identity.
                "hybrid_chain_outside_task_order",
                HYBRID,
                ["is_comment", "is_click"],
                [
                    [True, False, True],
                    [False, True, False],
                    [False, False, True],
                ],
            ),
        ]
    )
    def test_mask_table(self, name: str, mask_type: str, chain, expected) -> None:
        mask = build_cross_task_mask(mask_type, _TASK_NAMES, chain)
        self.assertTrue(
            torch.equal(mask, torch.tensor(expected, dtype=torch.bool)),
            msg=f"{name}: got\n{mask}",
        )

    @parameterized.expand(
        [
            ("chain_too_short", HYBRID, ["is_click"], "at least two entries"),
            ("chain_has_unknown_task", HYBRID, ["is_click", "nope"], "not task"),
            ("chain_has_duplicates", HYBRID, ["is_click", "is_click"], "duplicates"),
            ("unknown_mask_type", "ONERANK_MASK_NOPE", None, "unknown cross-task"),
        ]
    )
    def test_invalid_arguments_raise(
        self, name: str, mask_type: str, chain, message: str
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            build_cross_task_mask(mask_type, _TASK_NAMES, chain)


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
        ``jagged_softmax`` would all pass them.  This compares against a
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
            # Head-major collapse, matching the module's
            # ``(attn.unsqueeze(-1) * v).reshape(-1, num_heads * head_dim)``.
            contexts.append(torch.einsum("hn,nhd->hd", attn, v[rows]).reshape(-1))
            offset += num
        expected = module._out_proj(torch.stack(contexts))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


class JaggedSoftmaxTest(unittest.TestCase):
    """Direct value-level pin of ``jagged_softmax``.

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

        weights = jagged_softmax(logits, lengths)

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
