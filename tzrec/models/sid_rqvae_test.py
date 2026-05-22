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

import unittest

import torch
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.models.sid_rqvae import SidRqvae
from tzrec.protos import model_pb2
from tzrec.protos.models import sid_model_pb2
from tzrec.utils.state_dict_util import init_parameters


def _make_batch(
    batch_size: int,
    input_dim: int,
    feature_name: str = "item_emb",
    extra_features: dict = None,
) -> Batch:
    """Create a minimal Batch with dense embedding features."""
    keys = [feature_name]
    tensors = [torch.randn(batch_size, input_dim)]
    if extra_features:
        for k, v in extra_features.items():
            keys.append(k)
            tensors.append(v)
    dense_feature = KeyedTensor.from_tensor_list(keys=keys, tensors=tensors)
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={},
        labels={},
    )


class SidRqvaeTest(unittest.TestCase):
    """Tests for SidRqvae model."""

    def _create_model(self, use_clip=False, input_dim=32, embed_dim=8, n_layers=2):
        """Helper to create a SidRqvae model with minimal config."""
        n_embed_str = ",".join(["16"] * n_layers)
        sid_rqvae_cfg = sid_model_pb2.SidRqvae(
            input_dim=input_dim,
            embed_dim=embed_dim,
            codebook=n_embed_str,
            forward_mode="ste",
            loss_type="mse",
            kmeans_init=False,
            embedding_feature_name="item_emb",
        )
        if use_clip:
            sid_rqvae_cfg.clip_config.CopyFrom(
                sid_model_pb2.ClipConfig(
                    clip_feature_name="image_emb",
                    is_clip_pair_feature_name="is_clip_pair",
                )
            )

        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            sid_rqvae=sid_rqvae_cfg,
        )
        model = SidRqvae(model_config=model_config, features=[], labels=[])
        init_parameters(model, device=torch.device("cpu"))
        return model

    def test_rqvae_train_mode(self) -> None:
        """Test SidRqvae in train mode: predict -> loss -> metric."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        model.init_loss()
        model.init_metric()

        batch = _make_batch(B, input_dim)
        predictions = model.predict(batch)

        # Train mode should return all fields
        self.assertIn("codes", predictions)
        self.assertIn("quantized", predictions)
        self.assertIn("x_hat", predictions)
        self.assertIn("reconstruction_loss", predictions)
        self.assertIn("quantization_loss", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)

        # Loss should return reconstruction_loss + quantization_loss
        losses = model.loss(predictions, batch)
        self.assertIn("reconstruction_loss", losses)
        self.assertIn("quantization_loss", losses)

        # Total loss should be a scalar and have grad
        total_loss = sum(losses.values())
        self.assertTrue(total_loss.requires_grad)

        # Metric update should not raise
        model.update_metric(predictions, batch, losses)
        metrics = model.compute_metric()
        self.assertIn("mse", metrics)
        self.assertIn("unique_sid_ratio", metrics)

    def test_rqvae_eval_mode(self) -> None:
        """Test SidRqvae in eval mode: predict returns all fields."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.eval()

        batch = _make_batch(B, input_dim)
        predictions = model.predict(batch)

        # Eval mode (not inference) should return all fields
        self.assertIn("codes", predictions)
        self.assertIn("quantized", predictions)
        self.assertIn("x_hat", predictions)
        self.assertIn("reconstruction_loss", predictions)
        self.assertIn("quantization_loss", predictions)

    def test_rqvae_inference_mode(self) -> None:
        """Test SidRqvae in inference mode: only codes returned."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.eval()
        model.set_is_inference(True)

        batch = _make_batch(B, input_dim)
        predictions = model.predict(batch)

        # Inference mode should only return codes
        self.assertIn("codes", predictions)
        self.assertNotIn("x_hat", predictions)
        self.assertNotIn("reconstruction_loss", predictions)

    def test_rqvae_clip_mode(self) -> None:
        """Test SidRqvae with CLIP mixed mode (mixed recon + clip batch)."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        # Build mixed batch: first half recon, second half clip.
        # With the explicit is_clip_pair column the actual tensor values
        # no longer matter — the flag column drives routing.
        item_emb = torch.randn(B, input_dim)
        image_emb = torch.randn(B, input_dim)
        is_clip_pair = torch.zeros(B, 1)
        is_clip_pair[B // 2 :] = 1.0  # clip rows

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[item_emb, image_emb, is_clip_pair],
                )
            },
            sparse_features={},
            labels={},
        )

        predictions = model.predict(batch)

        # Mixed mode should return recon_loss, clip_loss, commitment_loss
        self.assertIn("codes", predictions)
        self.assertIn("recon_loss", predictions)
        self.assertIn("clip_loss", predictions)
        self.assertIn("commitment_loss", predictions)
        self.assertIn("x_hat", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)

        # Loss should return all three
        losses = model.loss(predictions, batch)
        self.assertIn("recon_loss", losses)
        self.assertIn("clip_loss", losses)
        self.assertIn("commitment_loss", losses)

        total_loss = sum(losses.values())
        self.assertTrue(total_loss.requires_grad)

        # Backward should work
        total_loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_rqvae_clip_all_recon(self) -> None:
        """Test mixed mode with all-recon batch (edge case)."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        # All recon: is_clip_pair = 0 everywhere
        item_emb = torch.randn(B, input_dim)
        image_emb = torch.randn(B, input_dim)
        is_clip_pair = torch.zeros(B, 1)

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[item_emb, image_emb, is_clip_pair],
                )
            },
            sparse_features={},
            labels={},
        )

        predictions = model.predict(batch)
        model.loss(predictions, batch)

        # clip_loss should be 0 (no clip rows)
        self.assertEqual(predictions["clip_loss"].item(), 0.0)
        # recon_loss should be > 0
        self.assertGreater(predictions["recon_loss"].item(), 0.0)

    def test_rqvae_clip_all_clip(self) -> None:
        """Test mixed mode with all-clip batch (edge case)."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        # All clip: is_clip_pair = 1 everywhere
        item_emb = torch.randn(B, input_dim)
        image_emb = torch.randn(B, input_dim)
        is_clip_pair = torch.ones(B, 1)

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[item_emb, image_emb, is_clip_pair],
                )
            },
            sparse_features={},
            labels={},
        )

        predictions = model.predict(batch)
        model.loss(predictions, batch)

        # recon_loss should be 0 (no recon rows)
        self.assertEqual(predictions["recon_loss"].item(), 0.0)
        # clip_loss should be > 0
        self.assertGreater(predictions["clip_loss"].item(), 0.0)

    def test_rqvae_backward(self) -> None:
        """Test that backward pass works without errors."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        model.init_loss()

        batch = _make_batch(B, input_dim)
        predictions = model.predict(batch)
        losses = model.loss(predictions, batch)
        total_loss = sum(losses.values())
        total_loss.backward()

        # Encoder params should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_clip_mask_uses_flag_not_equality(self) -> None:
        """The is_clip_pair flag, not bit-exact equality, drives routing.

        Build a batch where ``image_emb == item_emb`` numerically but
        ``is_clip_pair=1``: row must route to the CLIP branch (under the
        old bit-exact logic it would have been silently relabeled recon).
        """
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        item_emb = torch.randn(B, input_dim)
        image_emb = item_emb.clone()  # bit-identical
        is_clip_pair = torch.ones(B, 1)  # but flagged as clip

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[item_emb, image_emb, is_clip_pair],
                )
            },
            sparse_features={},
            labels={},
        )

        predictions = model.predict(batch)
        # All rows flagged as clip -> recon_loss should be 0, clip_loss > 0
        self.assertEqual(predictions["recon_loss"].item(), 0.0)
        self.assertGreater(predictions["clip_loss"].item(), 0.0)

    def test_commitment_loss_l1_branch(self) -> None:
        """Verify the new commitment_loss='l1' branch runs end-to-end.

        Previously ``"l1"`` silently fell through to the L2 branch.
        """
        from tzrec.modules.sid_generation.residual_quantized import (
            ResidualQuantized,
        )

        torch.manual_seed(0)
        rq = ResidualQuantized(
            embed_dim=8,
            n_layers=2,
            n_embed=4,
            forward_mode="ste",
            commitment_loss="l1",
            kmeans_init=False,
            use_sinkhorn=False,
        )
        # Stub the codebook to known centroids so the result is reproducible.
        for layer in rq.layers:
            torch.nn.init.normal_(layer.embedding.weight, std=0.1)

        x = torch.randn(4, 8, requires_grad=True)
        out = rq(x)
        # Loss must be a finite scalar with gradient flowing back into x.
        self.assertTrue(torch.isfinite(out.quantization_loss))
        out.quantization_loss.backward()
        self.assertIsNotNone(x.grad)

    def test_sinkhorn_config_enabled_false(self) -> None:
        """``sinkhorn_config { enabled: false }`` must turn Sinkhorn off.

        Previously ``use_sinkhorn`` was hard-coded ``True`` and the proto
        block was honored only for iters/epsilon.
        """
        n_embed_str = ",".join(["16"] * 2)
        sid_rqvae_cfg = sid_model_pb2.SidRqvae(
            input_dim=32,
            embed_dim=8,
            codebook=n_embed_str,
            forward_mode="ste",
            loss_type="mse",
            kmeans_init=False,
            embedding_feature_name="item_emb",
        )
        sid_rqvae_cfg.sinkhorn_config.CopyFrom(
            sid_model_pb2.SinkhornConfig(enabled=False)
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            sid_rqvae=sid_rqvae_cfg,
        )
        model = SidRqvae(model_config=model_config, features=[], labels=[])
        init_parameters(model, device=torch.device("cpu"))

        for layer in model._rqvae.quantizer.layers:
            self.assertFalse(layer.use_sinkhorn)

    def test_sinkhorn_config_default_enabled(self) -> None:
        """Omitting ``sinkhorn_config`` preserves on-by-default behavior.

        Back-compat for legacy configs that never set the sub-config.
        """
        model = self._create_model()  # no sinkhorn_config set
        for layer in model._rqvae.quantizer.layers:
            self.assertTrue(layer.use_sinkhorn)

    def test_commitment_loss_invalid_raises(self) -> None:
        """ResidualQuantized rejects unknown commitment_loss spellings."""
        from tzrec.modules.sid_generation.residual_quantized import (
            ResidualQuantized,
        )

        with self.assertRaisesRegex(AssertionError, "commitment_loss"):
            ResidualQuantized(
                embed_dim=8,
                n_layers=2,
                n_embed=4,
                commitment_loss="bogus",
                use_sinkhorn=False,
            )


if __name__ == "__main__":
    unittest.main()
