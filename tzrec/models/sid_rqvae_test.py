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
from tzrec.features.feature import create_features
from tzrec.models.model import ScriptWrapper
from tzrec.models.sid_rqvae import SidRqvae
from tzrec.protos import feature_pb2, loss_pb2, model_pb2
from tzrec.protos.models import sid_model_pb2
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.state_dict_util import init_parameters


def _features_and_groups(
    input_dim: int, use_contrastive: bool = False, pair_emb_dim: int = None, flag_dim=1
):
    """Real raw features + feature groups for a SID model.

    Mirrors how every other model test wires inputs: ``create_features`` builds
    the ``BaseFeature`` objects and ``feature_groups`` (consumed by the model's
    :class:`EmbeddingGroup`) name the main ``deep`` group — plus, for the
    contrastive path, the paired group and the per-row pair-flag group.
    ``pair_emb_dim`` (default: match ``input_dim``) sizes the paired group and
    ``flag_dim`` (default 1) sizes the pair-flag group, so a test can
    deliberately mismatch either.
    """

    def _raw(name: str, dim: int) -> feature_pb2.FeatureConfig:
        return feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name=name, value_dim=dim)
        )

    def _deep(group_name: str, feature_name: str) -> model_pb2.FeatureGroupConfig:
        return model_pb2.FeatureGroupConfig(
            group_name=group_name,
            feature_names=[feature_name],
            group_type=model_pb2.FeatureGroupType.DEEP,
        )

    feature_cfgs = [_raw("item_emb", input_dim)]
    groups = [_deep("deep", "item_emb")]
    if use_contrastive:
        feature_cfgs += [
            _raw("pair_emb", pair_emb_dim if pair_emb_dim is not None else input_dim),
            _raw("is_pair", flag_dim),
        ]
        groups += [_deep("pair", "pair_emb"), _deep("pair_flag", "is_pair")]
    return create_features(feature_cfgs), groups


def _make_batch(batch_size: int, input_dim: int) -> Batch:
    """Create a minimal Batch with the ``item_emb`` dense feature."""
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["item_emb"], tensors=[torch.randn(batch_size, input_dim)]
    )
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={},
        labels={},
    )


def _recon_loss_cfg(recon_type: str = "l2") -> loss_pb2.LossConfig:
    """A LossConfig with a recon_loss term of the given recon_type."""
    lc = loss_pb2.LossConfig()
    lc.recon_loss.recon_type = recon_type
    return lc


def _commitment_cfg(
    latent_weight=(1.0, 0.5), commitment_type="l2"
) -> loss_pb2.LossConfig:
    lc = loss_pb2.LossConfig()
    lc.commitment_loss.latent_weight.extend(latent_weight)
    lc.commitment_loss.commitment_type = commitment_type
    return lc


def _contrastive_cfg() -> loss_pb2.LossConfig:
    # The contrastive objective marker (empty); the paired-feature wiring lives
    # on the model proto (SidRqvae.contrastive_config), set in _create_model.
    lc = loss_pb2.LossConfig()
    lc.contrastive_loss.SetInParent()
    return lc


class SidRqvaeTest(unittest.TestCase):
    """Tests for SidRqvae model."""

    def _create_model(
        self,
        use_contrastive=False,
        input_dim=32,
        embed_dim=8,
        n_layers=2,
        recon="l2",
        candidate_output=False,
    ):
        """Helper to create a SidRqvae model with config-driven losses."""
        n_embed_list = [16] * n_layers
        sid_rqvae_cfg = sid_model_pb2.SidRqvae(
            embed_dim=embed_dim,
            codebook=n_embed_list,
            forward_mode="ste",
            kmeans_init=False,
        )
        losses = [_recon_loss_cfg(recon), _commitment_cfg()]
        if use_contrastive:
            sid_rqvae_cfg.contrastive_config.pair_feature_group = "pair"
            sid_rqvae_cfg.contrastive_config.pair_flag_feature_group = "pair_flag"
            losses.append(_contrastive_cfg())
        if candidate_output:
            sid_rqvae_cfg.candidate_output_config.enabled = True
            sid_rqvae_cfg.candidate_output_config.topk = 3

        # Real features + feature_groups: input_dim is derived from the group.
        features, feature_groups = _features_and_groups(input_dim, use_contrastive)
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups, sid_rqvae=sid_rqvae_cfg, losses=losses
        )
        model = SidRqvae(model_config=model_config, features=features, labels=[])
        init_parameters(model, device=torch.device("cpu"))
        return model

    def _contrastive_batch(self, B, input_dim, is_pair):
        return Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "pair_emb", "is_pair"],
                    tensors=[
                        torch.randn(B, input_dim),
                        torch.randn(B, input_dim),
                        is_pair,
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
        self.assertIn("recon_loss", losses)
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
        self.assertNotIn("candidate_codes", predictions)

    def test_rqvae_export_trace_keeps_candidate_output(self) -> None:
        """Export wrapper traces candidate outputs after inference is set."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, candidate_output=True)
        model.eval()
        model.set_is_inference(True)

        wrapper = ScriptWrapper(model).eval()
        data = {"item_emb.values": torch.randn(B, input_dim)}

        predictions = wrapper(data)
        traced = symbolic_trace(wrapper)
        traced_predictions = traced(data)

        self.assertEqual(
            set(predictions.keys()),
            {"codes", "candidate_codes", "candidate_scores"},
        )
        self.assertEqual(set(traced_predictions.keys()), set(predictions.keys()))
        self.assertEqual(traced_predictions["candidate_codes"].shape, (B, 3, 2))
        self.assertEqual(traced_predictions["candidate_scores"].shape, (B, 3))
        torch.testing.assert_close(
            traced_predictions["candidate_codes"][:, 0, :],
            traced_predictions["codes"],
        )

    def test_rqvae_candidate_scores_sorted(self) -> None:
        """Candidate scores come back sorted best-first, minimum at slot 0."""
        B, input_dim = 4, 32
        model = self._create_model(
            input_dim=input_dim,
            candidate_output=True,
        )
        model.eval()
        model.set_is_inference(True)

        preds = model.predict(_make_batch(B, input_dim))
        self.assertEqual(
            set(preds.keys()), {"codes", "candidate_codes", "candidate_scores"}
        )
        self.assertEqual(preds["candidate_codes"].shape, (B, 3, 2))
        self.assertEqual(preds["candidate_scores"].shape, (B, 3))
        scores = preds["candidate_scores"]
        # Slot 0 is the best-scored (nearest) neighbor: minimum score, sorted.
        self.assertTrue(bool(torch.all(scores[:, 0] == scores.min(dim=1).values)))
        self.assertTrue(bool(torch.all(scores[:, :-1] <= scores[:, 1:])))

    def test_candidate_output_topk_exceeds_codebook_raises(self) -> None:
        """Reject topk above the last-layer codebook size at construction."""
        features, feature_groups = _features_and_groups(32)
        cfg = sid_model_pb2.SidRqvae(embed_dim=8, codebook=[16, 16], kmeans_init=False)
        cfg.candidate_output_config.enabled = True
        cfg.candidate_output_config.topk = 17
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups, sid_rqvae=cfg
        )
        with self.assertRaisesRegex(ValueError, "topk must be in"):
            SidRqvae(model_config=model_config, features=features, labels=[])

    def test_rqvae_contrastive_mode(self) -> None:
        """Test SidRqvae with the mixed recon + contrastive path."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim, use_contrastive=True)
        model.train()
        model.init_loss()

        is_pair = torch.zeros(B, 1)
        is_pair[B // 2 :] = 1.0  # second half are contrastive pairs
        batch = self._contrastive_batch(B, input_dim, is_pair)

        predictions = model.predict(batch)
        self.assertIn("codes", predictions)
        self.assertIn("x_hat", predictions)
        self.assertIn("embed_a", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)

        losses = model.loss(predictions, batch)
        self.assertIn("recon_loss", losses)
        self.assertIn("commitment_loss", losses)
        self.assertIn("contrastive_loss", losses)

        total_loss = sum(losses.values())
        self.assertTrue(total_loss.requires_grad)
        total_loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_rqvae_contrastive_all_recon(self) -> None:
        """Mixed mode, all-recon batch: contrastive term 0, recon term > 0."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_contrastive=True)
        model.train()
        model.init_loss()

        batch = self._contrastive_batch(B, input_dim, torch.zeros(B, 1))
        losses = model.loss(model.predict(batch), batch)
        self.assertEqual(losses["contrastive_loss"].item(), 0.0)
        self.assertGreater(losses["recon_loss"].item(), 0.0)

    def test_rqvae_contrastive_all_pair(self) -> None:
        """Mixed mode, all-pair batch: recon term 0, contrastive term > 0."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_contrastive=True)
        model.train()
        model.init_loss()

        batch = self._contrastive_batch(B, input_dim, torch.ones(B, 1))
        losses = model.loss(model.predict(batch), batch)
        self.assertEqual(losses["recon_loss"].item(), 0.0)
        self.assertGreater(losses["contrastive_loss"].item(), 0.0)

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
        features, feature_groups = _features_and_groups(32)
        for bad in ([1.0], [1.0, 0.5, 0.25]):
            cfg = sid_model_pb2.SidRqvae(
                embed_dim=8, codebook=[16, 16], kmeans_init=False
            )
            model_config = model_pb2.ModelConfig(
                feature_groups=feature_groups,
                sid_rqvae=cfg,
                losses=[_commitment_cfg(latent_weight=bad)],
            )
            model = SidRqvae(model_config=model_config, features=features, labels=[])
            with self.assertRaisesRegex(ValueError, "latent_weight"):
                model.init_loss()

    def test_pair_feature_group_dim_mismatch_raises(self) -> None:
        """A paired group whose dim != the main group fails fast at init.

        The paired feature is encoded by the same encoder as the main input, so
        a dim mismatch would otherwise crash with an opaque matmul shape error
        on the first contrastive forward — not at construction.
        """
        features, feature_groups = _features_and_groups(
            32, use_contrastive=True, pair_emb_dim=16
        )
        cfg = sid_model_pb2.SidRqvae(embed_dim=8, codebook=[16, 16], kmeans_init=False)
        cfg.contrastive_config.pair_feature_group = "pair"
        cfg.contrastive_config.pair_flag_feature_group = "pair_flag"
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups, sid_rqvae=cfg, losses=[_contrastive_cfg()]
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            SidRqvae(model_config=model_config, features=features, labels=[])

    def test_pair_flag_group_must_be_dim_1(self) -> None:
        """A pair-flag group with dim != 1 fails fast (would mis-route rows)."""
        features, feature_groups = _features_and_groups(
            32, use_contrastive=True, flag_dim=3
        )
        cfg = sid_model_pb2.SidRqvae(embed_dim=8, codebook=[16, 16], kmeans_init=False)
        cfg.contrastive_config.pair_feature_group = "pair"
        cfg.contrastive_config.pair_flag_feature_group = "pair_flag"
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups, sid_rqvae=cfg, losses=[_contrastive_cfg()]
        )
        with self.assertRaisesRegex(ValueError, "dim-1 raw flag"):
            SidRqvae(model_config=model_config, features=features, labels=[])

    def test_contrastive_group_missing_raises(self) -> None:
        """A typo'd contrastive group name fails fast at init, not on forward."""
        features, feature_groups = _features_and_groups(32, use_contrastive=True)
        cfg = sid_model_pb2.SidRqvae(embed_dim=8, codebook=[16, 16], kmeans_init=False)
        cfg.contrastive_config.pair_feature_group = "pair"
        cfg.contrastive_config.pair_flag_feature_group = "pair_flagTYPO"
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups, sid_rqvae=cfg, losses=[_contrastive_cfg()]
        )
        with self.assertRaisesRegex(ValueError, "not in model_config.feature_groups"):
            SidRqvae(model_config=model_config, features=features, labels=[])

    def test_eval_metric_masks_contrastive_pair_rows(self) -> None:
        """Contrastive eval mse/rel_loss score only the non-pair (recon) rows.

        Training masks the recon loss to non-pair rows; update_metric must apply
        the same ``recon_mask`` so the eval metric stays comparable (pair rows,
        which the decoder is not trained to reconstruct, must not dilute it).
        """
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim, use_contrastive=True)
        model.eval()
        model.init_metric()

        # All-pair batch: recon_mask selects zero rows, so mse observes none.
        all_pair = self._contrastive_batch(B, input_dim, torch.ones(B, 1))
        model.update_metric(model.predict(all_pair), all_pair)
        self.assertEqual(model._metric_modules["mse"].total.item(), 0.0)

        # A recon (non-pair) batch then contributes rows.
        all_recon = self._contrastive_batch(B, input_dim, torch.zeros(B, 1))
        model.update_metric(model.predict(all_recon), all_recon)
        self.assertGreater(model._metric_modules["mse"].total.item(), 0.0)

    def test_pair_flag_drives_routing_not_equality(self) -> None:
        """The is_pair flag, not bit-exact equality, drives routing.

        Build a batch where ``pair_emb == item_emb`` numerically but
        ``is_pair=1``: rows must route to the contrastive branch (under the old
        bit-exact logic they would have been silently relabeled recon).
        """
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, use_contrastive=True)
        model.train()
        model.init_loss()

        item_emb = torch.randn(B, input_dim)
        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "pair_emb", "is_pair"],
                    tensors=[item_emb, item_emb.clone(), torch.ones(B, 1)],
                )
            },
            sparse_features={},
            labels={},
        )
        losses = model.loss(model.predict(batch), batch)
        self.assertEqual(losses["recon_loss"].item(), 0.0)
        self.assertGreater(losses["contrastive_loss"].item(), 0.0)

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
            embed_dim=8,
            codebook=[16, 16],
            forward_mode="ste",
            kmeans_init=False,
        )
        if enabled is not None:
            cfg.sinkhorn_config.CopyFrom(sid_model_pb2.SinkhornConfig(enabled=enabled))
        features, feature_groups = _features_and_groups(32)
        model = SidRqvae(
            model_config=model_pb2.ModelConfig(
                feature_groups=feature_groups, sid_rqvae=cfg
            ),
            features=features,
            labels=[],
        )
        init_parameters(model, device=torch.device("cpu"))
        for layer in model._quantizer.layers:
            self.assertEqual(layer.use_sinkhorn, expect_use_sinkhorn)

    @parameterized.expand([("l2",), ("l1",), ("cos",)])
    def test_recon_type_branch(self, recon_type) -> None:
        """Each recon_type runs end-to-end (grad flows through the decoder)."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim, recon=recon_type)
        model.train()
        model.init_loss()
        losses = model.loss(
            model.predict(_make_batch(B, input_dim)), _make_batch(B, input_dim)
        )
        recon = losses["recon_loss"]
        self.assertTrue(torch.isfinite(recon), f"{recon_type} not finite")
        recon.backward()  # grad must flow through the decoder

    def test_logit_scale_clamped_prevents_overflow(self) -> None:
        """A raw logit_scale far above ln(100) must not overflow.

        The clamp caps ``exp()`` so the contrastive loss and the parameter
        gradient stay finite; without it, ``exp(large)`` -> +Inf -> a NaN
        gradient that permanently corrupts the parameter.
        """
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim, use_contrastive=True)
        model.train()
        model.init_loss()
        # The temperatures live on the contrastive module that owns the clamp.
        contrastive = model._loss_modules["contrastive_loss"]
        scales = (
            contrastive.logit_scale_self,
            contrastive.logit_scale_cl,
            contrastive.logit_scale_ori,
        )
        with torch.no_grad():
            for p in scales:
                p.fill_(100.0)

        batch = self._contrastive_batch(B, input_dim, torch.ones(B, 1))
        losses = model.loss(model.predict(batch), batch)
        self.assertTrue(torch.isfinite(losses["contrastive_loss"]))
        sum(losses.values()).backward()
        for p in scales:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
