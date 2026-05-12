# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Correctness tests for ResidualQuantized in-place commitment-loss refactor.

Verifies:
  1. _single_commitment_loss aggregated over layers == legacy loop form.
  2. forward() is numerically deterministic in eval mode and gradients flow.
  3. quants_trunc (returned) equals the raw accumulated sum before STE.
"""

import unittest

import torch
import torch.nn.functional as F

from tzrec.modules.sid_generation.residual_quantized import ResidualQuantized


def _legacy_compute_commitment_loss(module, x, quant_list):
    """Reference implementation of the original loop form (pre-refactor)."""
    loss_list = []
    for quant in quant_list:
        if module.commitment_loss_type == "cos":
            loss1 = (
                (1 - F.cosine_similarity(x, quant.detach(), dim=-1))
                .mean()
                * module.commitment_w1
            )
            if module.use_ema:
                loss2 = torch.tensor(0.0, device=x.device)
            else:
                loss2 = (
                    (1 - F.cosine_similarity(x.detach(), quant, dim=-1))
                    .mean()
                    * module.commitment_w2
                )
        else:
            loss1 = (
                (x - quant.detach()).pow(2.0).mean()
                * module.commitment_w1
            )
            if module.use_ema:
                loss2 = torch.tensor(0.0, device=x.device)
            else:
                loss2 = (
                    (x.detach() - quant).pow(2.0).mean()
                    * module.commitment_w2
                )
        loss_list.append(loss1 + loss2)
    return torch.mean(torch.stack(loss_list))


def _make_module(
    seed=0,
    n_layers=3,
    n_embed=64,
    embed_dim=32,
    use_ema=True,
    commitment_loss="l2",
):
    torch.manual_seed(seed)
    return ResidualQuantized(
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_embed=n_embed,
        forward_mode="ste",
        commitment_loss=commitment_loss,
        latent_weight=[1.0, 0.5],
        kmeans_init=False,
        use_ema=use_ema,
        use_sinkhorn=False,
        rotation_trick=False,
    )


class ResidualQuantizedCommitmentLossEquivTest(unittest.TestCase):
    """Unit-level equivalence test for the loss aggregation refactor."""

    def test_single_loss_aggregated_matches_legacy_l2_ema(self):
        """l2 + EMA: per-layer single_loss mean == legacy list-based mean."""
        module = _make_module(seed=1, use_ema=True, commitment_loss="l2")
        module.eval()
        torch.manual_seed(1)
        x = torch.randn(16, 32)
        quant_list = [torch.randn(16, 32) for _ in range(3)]

        per_layer = [module._single_commitment_loss(x, q) for q in quant_list]
        new_loss = torch.mean(torch.stack(per_layer))
        old_loss = _legacy_compute_commitment_loss(module, x, quant_list)

        torch.testing.assert_close(new_loss, old_loss, atol=0, rtol=0)

    def test_single_loss_aggregated_matches_legacy_l2_noema(self):
        """l2 + no-EMA: loss2 path must also match exactly."""
        module = _make_module(seed=2, use_ema=False, commitment_loss="l2")
        module.eval()
        torch.manual_seed(2)
        x = torch.randn(16, 32, requires_grad=True)
        quant_list = [torch.randn(16, 32, requires_grad=True) for _ in range(3)]

        per_layer = [module._single_commitment_loss(x, q) for q in quant_list]
        new_loss = torch.mean(torch.stack(per_layer))
        old_loss = _legacy_compute_commitment_loss(module, x, quant_list)

        torch.testing.assert_close(new_loss, old_loss, atol=0, rtol=0)

    def test_single_loss_aggregated_matches_legacy_cos_noema(self):
        """cos + no-EMA."""
        module = _make_module(seed=3, use_ema=False, commitment_loss="cos")
        module.eval()
        torch.manual_seed(3)
        x = torch.randn(16, 32)
        quant_list = [torch.randn(16, 32) for _ in range(3)]

        per_layer = [module._single_commitment_loss(x, q) for q in quant_list]
        new_loss = torch.mean(torch.stack(per_layer))
        old_loss = _legacy_compute_commitment_loss(module, x, quant_list)

        torch.testing.assert_close(new_loss, old_loss, atol=0, rtol=0)


class ResidualQuantizedForwardTest(unittest.TestCase):
    """End-to-end forward behaviour after the refactor."""

    def test_forward_eval_deterministic(self):
        """Two forwards on same input + same seed -> bit-level identical."""
        module = _make_module(seed=11)
        module.eval()
        x = torch.randn(32, 32)

        out1 = module(x)
        out2 = module(x)

        torch.testing.assert_close(
            out1.quantized_embeddings, out2.quantized_embeddings, atol=0, rtol=0
        )
        torch.testing.assert_close(
            out1.cluster_ids, out2.cluster_ids, atol=0, rtol=0
        )
        torch.testing.assert_close(
            out1.quantization_loss, out2.quantization_loss, atol=0, rtol=0
        )

    def test_forward_output_shapes(self):
        module = _make_module(seed=12, n_layers=3, embed_dim=16)
        module.eval()
        x = torch.randn(8, 16)
        out = module(x)
        self.assertEqual(out.cluster_ids.shape, (8, 3))
        self.assertEqual(out.quantized_embeddings.shape, (8, 16))
        self.assertEqual(out.quantization_loss.shape, ())
        self.assertTrue(torch.isfinite(out.quantization_loss))

    def test_eval_quants_trunc_equals_codebook_sum(self):
        """In eval mode quants_trunc must equal sum of codebook rows at ids."""
        module = _make_module(seed=13, n_layers=3, embed_dim=16, n_embed=32)
        module.eval()
        x = torch.randn(8, 16)
        out = module(x)

        manual = torch.zeros_like(x)
        for i, layer in enumerate(module.layers):
            manual = manual + layer.embedding(out.cluster_ids[:, i])

        torch.testing.assert_close(
            out.quantized_embeddings, manual, atol=1e-6, rtol=1e-5
        )

    def test_backward_flows_to_codebook(self):
        """Commitment loss (no-EMA) should propagate gradient to codebook."""
        module = _make_module(seed=14, use_ema=False, commitment_loss="l2")
        module.train()
        x = torch.randn(16, 32, requires_grad=False)
        out = module(x)
        out.quantization_loss.backward()

        any_grad = False
        for layer in module.layers:
            if layer.embedding.weight.grad is not None and \
                    layer.embedding.weight.grad.abs().sum() > 0:
                any_grad = True
                break
        self.assertTrue(any_grad, "codebook should receive gradient under no-EMA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
