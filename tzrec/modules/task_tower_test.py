# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``tzrec.modules.task_tower`` (CPU only).

``OneRankPredictionHead`` moved here from ``tzrec.modules.gr.onerank_head``.
The scorer variants (``scorer_type``) and ``task_bias_init`` exist because
of the constant-prediction plateau of the rank-1 scorer (see the model
docs); the tests here pin down the contract each variant is supposed to
keep:

* BILINEAR starts *exactly* where DOT_PRODUCT would (identity ``W_k``);
* ``task_bias_init`` shifts every (candidate, task) logit by ``b_k`` and
  nothing else;
* MLP discriminates a request's candidates straight from random init.
"""

import itertools
import math
import unittest

import torch
from parameterized import parameterized

from tzrec.modules.task_tower import OneRankPredictionHead, TaskTower
from tzrec.ops import Kernel
from tzrec.protos import tower_pb2
from tzrec.protos.module_pb2 import MLP
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.test_util import (
    TestGraphType,
    create_test_module,
    mark_ci_scope,
)


class TaskTowerTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_task_tower(self, graph_type) -> None:
        task_cgf = tower_pb2.TaskTower(
            tower_name="is_click",
            label_name="is_clk",
            mlp=MLP(hidden_units=[12, 8, 4]),
            num_class=2,
        )
        task_cgf = config_to_kwargs(task_cgf)
        task_tower = TaskTower(32, task_cgf["num_class"], mlp=task_cgf["mlp"])
        task_tower = create_test_module(task_tower, graph_type)
        features = torch.randn(4, 32)
        output = task_tower(features)
        self.assertEqual(list(output.size()), [4, 2])


@mark_ci_scope("h20", "gpu")
class OneRankPredictionHeadTest(unittest.TestCase):
    """Tests for the scorer variants and ``task_bias_init``."""

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

    # A comprehension over itertools.product (not a nested comprehension):
    # the single ``for`` iterable is evaluated in the class scope -- nested
    # comprehension iterables would not see ``_SCORER_TYPES`` and friends --
    # while the loop variables stay comprehension-local, leaving no
    # class-level leftovers to clean up.
    _SCORER_CASES = [
        (f"{scorer}_{gt.name.lower()}", scorer, gt)
        for gt, scorer in itertools.product(_GRAPH_TYPES, _SCORER_TYPES)
    ]

    @classmethod
    def _inputs(cls, seed: int = 7) -> tuple:
        torch.manual_seed(seed)
        task_embeddings = torch.randn(cls._TOTAL, cls._NUM_TASKS, cls._EMBEDDING_DIM)
        contextual = torch.randn(len(cls._NUM_CANDIDATES), cls._CONTEXTUAL_DIM)
        num_candidates = torch.tensor(cls._NUM_CANDIDATES)
        return task_embeddings, num_candidates, contextual

    @classmethod
    def _head(
        cls,
        scorer_type: str = "dot_product",
        task_bias_init=None,
        with_sd: bool = True,
        with_cross_task: bool = True,
    ) -> OneRankPredictionHead:
        torch.manual_seed(11)
        head = OneRankPredictionHead(
            embedding_dim=cls._EMBEDDING_DIM,
            task_names=cls._TASK_NAMES,
            contextual_feature_dim=cls._CONTEXTUAL_DIM,
            situation_discernment=({"num_heads": cls._NUM_HEADS} if with_sd else None),
            cross_task_head=(
                {"num_heads": cls._NUM_HEADS, "ffn_hidden_dim": 8}
                if with_cross_task
                else None
            ),
            scorer_type=scorer_type,
            scorer_hidden_dim=12,
            task_bias_init=task_bias_init,
        )
        # The default kernel (TRITON) is CUDA-only; these tests run on CPU.
        head.set_kernel(Kernel.PYTORCH)
        return head

    @parameterized.expand(_SCORER_CASES)
    def test_scorer_variants_produce_discriminating_logits(
        self, name, scorer_type, graph_type
    ) -> None:
        """Every scorer emits one finite logit per (candidate, task).

        ``FX_TRACE`` matters in its own right: the ``output_size=`` argument
        added to ``repeat_interleave`` keeps the broadcast sync-free, which
        only holds if the static size survives symbolic tracing.
        """
        head = self._head(scorer_type=scorer_type)
        module = create_test_module(head, graph_type)
        task_embeddings, num_candidates, contextual = self._inputs()

        logits = module(task_embeddings, num_candidates, contextual)

        self.assertEqual(logits.size(), (self._TOTAL, self._NUM_TASKS))
        self.assertTrue(torch.isfinite(logits).all())
        # Candidates of one request must not tie under any scorer: the MLP
        # by construction, the rank-1 pair because two random projections
        # onto one direction tie with probability zero.
        for k in range(self._NUM_TASKS):
            self.assertFalse(
                torch.allclose(logits[: self._NUM_CANDIDATES[0], k], logits[0, k]),
                msg=f"{scorer_type} ties a request's candidates on task {k}",
            )

    def test_bilinear_starts_where_dot_product_does(self) -> None:
        """Identity ``W_k`` makes BILINEAR equal DOT_PRODUCT, exactly.

        This is the contract that lets the two be compared at step 0: only
        the *ability* to leave the rank-1 subspace changes, not the
        starting point.
        """
        dot = self._head(scorer_type="dot_product")
        bilinear = self._head(scorer_type="bilinear")
        # Shared submodules (SD, cross-task, bias) identical by copy, so
        # the only difference left is the scorer itself.
        bilinear.load_state_dict(dot.state_dict(), strict=False)
        torch.testing.assert_close(
            bilinear._bilinear_weight.detach(),
            torch.eye(self._EMBEDDING_DIM).repeat(self._NUM_TASKS, 1, 1),
        )

        task_embeddings, num_candidates, contextual = self._inputs()
        torch.testing.assert_close(
            bilinear(task_embeddings, num_candidates, contextual),
            dot(task_embeddings, num_candidates, contextual),
        )

    def test_bilinear_weight_is_trainable_off_the_identity(self) -> None:
        """One step of SGD must be able to move ``W_k`` off the identity.

        The identity init is a starting point, not a frozen prior: if the
        gradient never reached ``W_k`` the variant would be a no-op.
        """
        bilinear = self._head(scorer_type="bilinear")
        task_embeddings, num_candidates, contextual = self._inputs()

        bilinear(task_embeddings, num_candidates, contextual).sum().backward()
        self.assertIsNotNone(bilinear._bilinear_weight.grad)
        self.assertGreater(bilinear._bilinear_weight.grad.abs().max().item(), 0.0)

    def test_mlp_scorer_has_one_mlp_per_task(self) -> None:
        """Each task owns its whole MLP; nothing is shared between tasks."""
        mlp = self._head(scorer_type="mlp")
        self.assertEqual(len(mlp._task_mlps), self._NUM_TASKS)
        # Different parameters per task -- the point of the per-task scorer.
        first = mlp._task_mlps[0][0].weight
        for k in range(1, self._NUM_TASKS):
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
        plain = self._head(scorer_type="mlp")
        biased = self._head(scorer_type="mlp", task_bias_init=self._TASK_BIAS)
        # Same-seed construction makes every shared parameter identical;
        # load_state_dict would also copy the *zero* bias over, so re-apply
        # the configured one after it.
        biased.load_state_dict(plain.state_dict(), strict=False)
        with torch.no_grad():
            biased._task_bias.copy_(torch.tensor(self._TASK_BIAS))

        task_embeddings, num_candidates, contextual = self._inputs()
        # Broadcasting (total, K) + (K,) adds b_k to column k for every
        # candidate -- exactly the claimed semantics.
        torch.testing.assert_close(
            biased(task_embeddings, num_candidates, contextual),
            plain(task_embeddings, num_candidates, contextual)
            + torch.tensor(self._TASK_BIAS),
        )
        torch.testing.assert_close(
            biased._task_bias.detach(), torch.tensor(self._TASK_BIAS)
        )

    def test_task_bias_defaults_to_zero(self) -> None:
        """No ``task_bias_init`` keeps the zeros of previous runs."""
        plain = self._head()
        torch.testing.assert_close(
            plain._task_bias.detach(), torch.zeros(self._NUM_TASKS)
        )

    def test_bad_configs_are_rejected(self) -> None:
        """Config errors fail at construction, not at the first forward."""
        with self.assertRaisesRegex(ValueError, "unknown scorer_type"):
            self._head(scorer_type="nonsense")
        with self.assertRaisesRegex(ValueError, "task_bias_init has"):
            self._head(task_bias_init=[0.1, 0.2])

    def test_mean_pooling_fallback_matches_reference(self) -> None:
        """Without SD / cross-task the scorer is the plain mean pool.

        ``z_k = mean_i r^i_k`` per request, ``s^i_k = z_k . r^i_k /
        sqrt(D)``: a straight transcription of paper 2.4's degenerate case
        (ablations V5 and V3 both off).
        """
        head = self._head(
            scorer_type="dot_product", with_sd=False, with_cross_task=False
        )
        task_embeddings, num_candidates, _ = self._inputs(seed=3)

        got = head(task_embeddings, num_candidates)
        scale = 1.0 / math.sqrt(self._EMBEDDING_DIM)
        want = torch.zeros(self._TOTAL, self._NUM_TASKS)
        start = 0
        for n in self._NUM_CANDIDATES:
            segment = task_embeddings[start : start + n]
            z = segment.mean(dim=0)  # (K, D)
            want[start : start + n] = (z * segment).sum(dim=-1) * scale
            start += n
        torch.testing.assert_close(got, want)

    def test_situation_discernment_requires_contextual(self) -> None:
        """SD configured but no contextual embeddings -> loud failure."""
        head = self._head()
        task_embeddings, num_candidates, _ = self._inputs()
        with self.assertRaisesRegex(ValueError, "contextual"):
            head(task_embeddings, num_candidates)


if __name__ == "__main__":
    unittest.main()
