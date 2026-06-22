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
from tzrec.protos import loss_pb2, model_pb2
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


def _recon_loss_cfg(kind: str = "recon_l2_loss") -> loss_pb2.LossConfig:
    """A LossConfig whose sid_loss oneof is the given recon variant."""
    lc = loss_pb2.LossConfig()
    getattr(lc, kind).SetInParent()
    return lc


def _commitment_cfg(
    latent_weight=(1.0, 0.5), commitment_type="l2"
) -> loss_pb2.LossConfig:
    lc = loss_pb2.LossConfig()
    lc.commitment_loss.latent_weight.extend(latent_weight)
    lc.commitment_loss.commitment_type = commitment_type
    return lc


def _clip_cfg() -> loss_pb2.LossConfig:
    # The contrastive objective marker (empty); the paired-feature wiring lives
    # on the model proto (SidRqvae.clip_config), set in _create_model.
    lc = loss_pb2.LossConfig()
    lc.sid_clip_loss.SetInParent()
    return lc


class SidRqvaeTest(unittest.TestCase):
    """Tests for SidRqvae model."""

    def _create_model(
        self,
        use_clip=False,
        input_dim=32,
        embed_dim=8,
        n_layers=2,
        recon="recon_l2_loss",
    ):
        """Helper to create a SidRqvae model with config-driven losses."""
        n_embed_list = [16] * n_layers
        sid_rqvae_cfg = sid_model_pb2.SidRqvae(
            input_dim=input_dim,
            embed_dim=embed_dim,
            codebook=n_embed_list,
            forward_mode="ste",
            kmeans_init=False,
            embedding_feature_name="item_emb",
        )
        losses = [_recon_loss_cfg(recon), _commitment_cfg()]
        if use_clip:
            # structure on the model proto; objective marker in losses.
            sid_rqvae_cfg.clip_config.clip_feature_name = "image_emb"
            sid_rqvae_cfg.clip_config.is_clip_pair_feature_name = "is_clip_pair"
            losses.append(_clip_cfg())

        # SID models read the item-embedding dense feature directly from the
        # batch; they do not consume feature_groups, so none is set (which
        # keeps the config consistent with the empty ``features`` list).
        model_config = model_pb2.ModelConfig(sid_rqvae=sid_rqvae_cfg, losses=losses)
        model = SidRqvae(model_config=model_config, features=[], labels=[])
        init_parameters(model, device=torch.device("cpu"))
        return model

    def _clip_batch(self, B, input_dim, is_clip_pair):
        return Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[
                        torch.randn(B, input_dim),
                        torch.randn(B, input_dim),
                        is_clip_pair,
                    ],
                )
            },
            sparse_features={},
            labels={},
        )

    def test_rqvae_train_mode(self) -> None:
        """Test SidRqvae in train mode: predict -> loss -> metric."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        model.init_loss()
        model.init_metric()

        batch = _make_batch(B, input_dim)
        predictions = model.predict(batch)

        # predict() returns only the raw tensors the losses consume.
        self.assertIn("codes", predictions)
        self.assertIn("x_hat", predictions)
        self.assertIn("encoder_out", predictions)
        self.assertIn("latents", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)

        # loss() computes the configured recon + commitment terms.
        losses = model.loss(predictions, batch)
        self.assertIn("recon_l2_loss", losses)
        self.assertIn("commitment_loss", losses)

        total_loss = sum(losses.values())
        self.assertTrue(total_loss.requires_grad)

        model.update_metric(predictions, batch, losses)
        metrics = model.compute_metric()
        self.assertIn("mse", metrics)
        self.assertIn("unique_sid_ratio", metrics)

    def test_rqvae_eval_mode(self) -> None:
        """Test SidRqvae in eval mode: predict returns the recon fields."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.eval()

        predictions = model.predict(_make_batch(B, input_dim))

        # Eval mode (not inference) exposes x_hat for the metric + losses.
        self.assertIn("codes", predictions)
        self.assertIn("x_hat", predictions)
        self.assertIn("encoder_out", predictions)
        self.assertIn("latents", predictions)

    def test_rqvae_inference_mode(self) -> None:
        """Test SidRqvae in inference mode: only codes returned."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.eval()
        model.set_is_inference(True)

        predictions = model.predict(_make_batch(B, input_dim))
        self.assertIn("codes", predictions)
        self.assertNotIn("x_hat", predictions)
        self.assertNotIn("latents", predictions)

    def test_rqvae_clip_mode(self) -> None:
        """Test SidRqvae with CLIP mixed mode (mixed recon + clip batch)."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        is_clip_pair = torch.zeros(B, 1)
        is_clip_pair[B // 2 :] = 1.0  # second half clip
        batch = self._clip_batch(B, input_dim, is_clip_pair)

        predictions = model.predict(batch)
        self.assertIn("codes", predictions)
        self.assertIn("x_hat", predictions)
        self.assertIn("embed_a", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)

        losses = model.loss(predictions, batch)
        self.assertIn("recon_l2_loss", losses)
        self.assertIn("commitment_loss", losses)
        self.assertIn("sid_clip_loss", losses)

        total_loss = sum(losses.values())
        self.assertTrue(total_loss.requires_grad)
        total_loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_rqvae_clip_all_recon(self) -> None:
        """Mixed mode with all-recon batch: clip term 0, recon term > 0."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        batch = self._clip_batch(B, input_dim, torch.zeros(B, 1))
        losses = model.loss(model.predict(batch), batch)
        self.assertEqual(losses["sid_clip_loss"].item(), 0.0)
        self.assertGreater(losses["recon_l2_loss"].item(), 0.0)

    def test_rqvae_clip_all_clip(self) -> None:
        """Mixed mode with all-clip batch: recon term 0, clip term > 0."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        batch = self._clip_batch(B, input_dim, torch.ones(B, 1))
        losses = model.loss(model.predict(batch), batch)
        self.assertEqual(losses["recon_l2_loss"].item(), 0.0)
        self.assertGreater(losses["sid_clip_loss"].item(), 0.0)

    def test_rqvae_backward(self) -> None:
        """Test that backward pass works without errors."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        model.init_loss()

        batch = _make_batch(B, input_dim)
        losses = model.loss(model.predict(batch), batch)
        sum(losses.values()).backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_commitment_latent_weight_wrong_length_raises(self) -> None:
        """A commitment_loss with a bad latent_weight length fails in init_loss."""
        for bad in ([1.0], [1.0, 0.5, 0.25]):
            cfg = sid_model_pb2.SidRqvae(
                input_dim=32, embed_dim=8, codebook=[16, 16], kmeans_init=False
            )
            model_config = model_pb2.ModelConfig(
                sid_rqvae=cfg, losses=[_commitment_cfg(latent_weight=bad)]
            )
            model = SidRqvae(model_config=model_config, features=[], labels=[])
            with self.assertRaisesRegex(ValueError, "latent_weight"):
                model.init_loss()

    def test_clip_mask_uses_flag_not_equality(self) -> None:
        """The is_clip_pair flag, not bit-exact equality, drives routing.

        Build a batch where ``image_emb == item_emb`` numerically but
        ``is_clip_pair=1``: rows must route to the CLIP branch (under the old
        bit-exact logic they would have been silently relabeled recon).
        """
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_clip=True)
        model.train()
        model.init_loss()

        item_emb = torch.randn(B, input_dim)
        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb", "is_clip_pair"],
                    tensors=[item_emb, item_emb.clone(), torch.ones(B, 1)],
                )
            },
            sparse_features={},
            labels={},
        )
        losses = model.loss(model.predict(batch), batch)
        self.assertEqual(losses["recon_l2_loss"].item(), 0.0)
        self.assertGreater(losses["sid_clip_loss"].item(), 0.0)

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

    @parameterized.expand(
        [
            ("recon_l2_loss",),
            ("recon_l1_loss",),
            ("recon_cosine_loss",),
        ]
    )
    def test_recon_loss_variant_branch(self, recon) -> None:
        """Each recon variant runs end-to-end (grad flows through the decoder)."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, recon=recon)
        model.train()
        model.init_loss()
        losses = model.loss(
            model.predict(_make_batch(B, input_dim)), _make_batch(B, input_dim)
        )
        recon_loss = losses[recon]
        self.assertTrue(torch.isfinite(recon_loss), f"{recon} not finite")
        recon_loss.backward()  # grad must flow through the decoder

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

        batch = self._clip_batch(B, input_dim, torch.ones(B, 1))
        losses = model.loss(model.predict(batch), batch)
        self.assertTrue(torch.isfinite(losses["sid_clip_loss"]))
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
