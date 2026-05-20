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
                sid_model_pb2.ClipConfig(clip_feature_name="image_emb")
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

        # Build mixed batch: first half recon (image_emb == item_emb),
        # second half clip (image_emb != item_emb)
        item_emb = torch.randn(B, input_dim)
        image_emb = item_emb.clone()
        image_emb[B // 2 :] = torch.randn(B - B // 2, input_dim)  # clip rows

        extra = {"image_emb": image_emb}
        batch = _make_batch(B, input_dim, extra_features=extra)
        # Override item_emb in batch to match our crafted tensor
        batch.dense_features[BASE_DATA_GROUP] = KeyedTensor.from_tensor_list(
            keys=["item_emb", "image_emb"],
            tensors=[item_emb, image_emb],
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

        # All recon: image_emb == item_emb
        item_emb = torch.randn(B, input_dim)
        image_emb = item_emb.clone()

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb"],
                    tensors=[item_emb, image_emb],
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

        # All clip: image_emb != item_emb
        item_emb = torch.randn(B, input_dim)
        image_emb = torch.randn(B, input_dim)

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["item_emb", "image_emb"],
                    tensors=[item_emb, image_emb],
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


if __name__ == "__main__":
    unittest.main()
