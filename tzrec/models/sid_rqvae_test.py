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

import unittest

import torch
from parameterized import parameterized
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
        n_embed_list = [16] * n_layers
        sid_rqvae_cfg = sid_model_pb2.SidRqvae(
            input_dim=input_dim,
            embed_dim=embed_dim,
            codebook=n_embed_list,
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

        # SID models read the item-embedding dense feature directly from the
        # batch; they do not consume feature_groups, so none is set (which
        # keeps the config consistent with the empty ``features`` list).
        model_config = model_pb2.ModelConfig(
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

        # Mixed mode returns reconstruction_loss, clip_loss, quantization_loss
        self.assertIn("codes", predictions)
        self.assertIn("reconstruction_loss", predictions)
        self.assertIn("clip_loss", predictions)
        self.assertIn("quantization_loss", predictions)
        self.assertIn("x_hat", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)

        # Loss should return all three
        losses = model.loss(predictions, batch)
        self.assertIn("reconstruction_loss", losses)
        self.assertIn("clip_loss", losses)
        self.assertIn("quantization_loss", losses)

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
        # reconstruction_loss should be > 0
        self.assertGreater(predictions["reconstruction_loss"].item(), 0.0)

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

        # reconstruction_loss should be 0 (no recon rows)
        self.assertEqual(predictions["reconstruction_loss"].item(), 0.0)
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

    def test_latent_weight_wrong_length_raises(self) -> None:
        """latent_weight must be exactly [w1, w2]; a bad length fails fast."""
        for bad in ([1.0], [1.0, 0.5, 0.25]):
            cfg = sid_model_pb2.SidRqvae(
                input_dim=32,
                embed_dim=8,
                codebook=[16, 16],
                kmeans_init=False,
                latent_weight=bad,
            )
            model_config = model_pb2.ModelConfig(sid_rqvae=cfg)
            with self.assertRaisesRegex(ValueError, "latent_weight"):
                SidRqvae(model_config=model_config, features=[], labels=[])

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
        # All rows flagged as clip -> reconstruction_loss should be 0, clip_loss > 0
        self.assertEqual(predictions["reconstruction_loss"].item(), 0.0)
        self.assertGreater(predictions["clip_loss"].item(), 0.0)

    @parameterized.expand(
        [
            ("omitted", None, True),  # no sinkhorn_config -> on by default
            ("enabled_true", True, True),
            ("enabled_false", False, False),  # was hard-coded True before
        ]
    )
    def test_sinkhorn_config(self, _name, enabled, expect_use_sinkhorn) -> None:
        """``sinkhorn_config.enabled`` (or its omission) drives layer.use_sinkhorn."""
        cfg = sid_model_pb2.SidRqvae(
            input_dim=32,
            embed_dim=8,
            codebook=[16, 16],
            forward_mode="ste",
            loss_type="mse",
            kmeans_init=False,
            embedding_feature_name="item_emb",
        )
        if enabled is not None:
            cfg.sinkhorn_config.CopyFrom(sid_model_pb2.SinkhornConfig(enabled=enabled))
        model = SidRqvae(
            model_config=model_pb2.ModelConfig(sid_rqvae=cfg), features=[], labels=[]
        )
        init_parameters(model, device=torch.device("cpu"))
        for layer in model._quantizer.layers:
            self.assertEqual(layer.use_sinkhorn, expect_use_sinkhorn)

    @parameterized.expand([("mse",), ("l1",), ("cosine",)])
    def test_loss_type_recon_branch(self, loss_type) -> None:
        """Each loss_type recon branch runs end-to-end (grad flows)."""
        B, input_dim = 4, 32
        cfg = sid_model_pb2.SidRqvae(
            input_dim=input_dim,
            embed_dim=8,
            codebook=[16, 16],
            forward_mode="ste",
            loss_type=loss_type,
            kmeans_init=False,
            embedding_feature_name="item_emb",
        )
        model = SidRqvae(
            model_config=model_pb2.ModelConfig(sid_rqvae=cfg), features=[], labels=[]
        )
        init_parameters(model, device=torch.device("cpu"))
        model.train()
        model.init_loss()
        recon = model.predict(_make_batch(B, input_dim))["reconstruction_loss"]
        self.assertTrue(torch.isfinite(recon), f"{loss_type} recon not finite")
        recon.backward()  # grad must flow through the decoder

    def test_logit_scale_clamped_prevents_overflow(self) -> None:
        """A raw logit_scale far above ln(100) must not overflow.

        The clamp caps ``exp()`` so the CLIP loss and the parameter gradient
        stay finite; without it, ``exp(large)`` -> +Inf -> a NaN gradient that
        permanently corrupts the parameter.
        """
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()
        with torch.no_grad():
            model._logit_scale_self.fill_(100.0)
            model._logit_scale_cl.fill_(100.0)
            model._logit_scale.fill_(100.0)

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[
                        torch.randn(B, input_dim),
                        torch.randn(B, input_dim),
                        torch.ones(B, 1),
                    ],
                )
            },
            sparse_features={},
            labels={},
        )
        losses = model.loss(model.predict(batch), batch)
        self.assertTrue(torch.isfinite(losses["clip_loss"]))
        sum(losses.values()).backward()
        for p in (
            model._logit_scale_self,
            model._logit_scale_cl,
            model._logit_scale,
        ):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
