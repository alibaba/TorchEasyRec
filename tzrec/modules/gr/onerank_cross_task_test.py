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

"""Unit tests for ``tzrec.modules.gr.onerank_cross_task`` (CPU only)."""

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
from tzrec.ops import Kernel
from tzrec.utils.test_util import mark_ci_scope

_TASK_NAMES = ["is_click", "is_like", "is_comment"]


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


if __name__ == "__main__":
    unittest.main()
