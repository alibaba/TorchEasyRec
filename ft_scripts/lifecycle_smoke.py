"""End-to-end lifecycle smoke for SidRqkmeans + SidRqvae (clip and no-clip).

For each model:
  1. construct via proto config
  2. run ``init_loss`` + ``init_metric``
  3. drive a few training-mode predict/loss/backward/update_train_metric steps
  4. (SidRqkmeans only) call ``on_train_end`` to fit FAISS
  5. flip to eval and run a predict + update_metric
  6. flip to inference via ``set_is_inference`` and run a codes-only predict
  7. assert the result shapes / non-emptiness make sense at every stage

This is a hand-driven approximation of the lifecycle that ``tzrec.main``
runs in production; the goal is to surface any AttributeError /
ShapeError / dtype mismatch that the existing unit tests miss.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402
from google.protobuf.struct_pb2 import Struct  # noqa: E402
from torchrec import KeyedTensor  # noqa: E402

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch  # noqa: E402
from tzrec.models.sid_rqkmeans import SidRqkmeans  # noqa: E402
from tzrec.models.sid_rqvae import SidRqvae  # noqa: E402
from tzrec.protos import model_pb2  # noqa: E402
from tzrec.protos.models import sid_model_pb2  # noqa: E402
from tzrec.utils.state_dict_util import init_parameters  # noqa: E402


def _make_batch(
    batch_size: int,
    input_dim: int,
    extra: dict | None = None,
) -> Batch:
    keys = ["item_emb"]
    tensors = [torch.randn(batch_size, input_dim)]
    if extra:
        for k, v in extra.items():
            keys.append(k)
            tensors.append(v)
    kt = KeyedTensor.from_tensor_list(keys=keys, tensors=tensors)
    return Batch(
        dense_features={BASE_DATA_GROUP: kt},
        sparse_features={},
        labels={},
    )


# ---------------------------------------------------------------------------
# SidRqkmeans
# ---------------------------------------------------------------------------

def _build_sid_rqkmeans(input_dim: int, n_layers: int) -> SidRqkmeans:
    codebook = ",".join(["16"] * n_layers)
    faiss_kwargs = Struct()
    faiss_kwargs.update({"niter": 5, "verbose": False, "seed": 1234})
    cfg = sid_model_pb2.SidRqkmeans(
        input_dim=input_dim,
        codebook=codebook,
        normalize_residuals=False,
        faiss_kmeans_kwargs=faiss_kwargs,
        embedding_feature_name="item_emb",
    )
    mc = model_pb2.ModelConfig(
        feature_groups=[
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            )
        ],
        sid_rqkmeans=cfg,
    )
    m = SidRqkmeans(model_config=mc, features=[], labels=[])
    init_parameters(m, device=torch.device("cpu"))
    return m


def lifecycle_rqkmeans() -> None:
    print("\n=== SidRqkmeans lifecycle ===")
    B, D, L = 64, 32, 2
    model = _build_sid_rqkmeans(input_dim=D, n_layers=L)
    model.init_loss()
    model.init_metric()

    print("[train] buffering 8 batches…")
    model.train()
    for step in range(8):
        batch = _make_batch(B, D)
        preds = model.predict(batch)
        losses = model.loss(preds, batch)
        sum(losses.values()).backward()
        model.update_train_metric(preds, batch)
    train_metrics = model.compute_train_metric()
    assert all(torch.isfinite(v).all() for v in train_metrics.values()), (
        f"non-finite train metric: {train_metrics}"
    )
    print(f"[train] buffer={len(model._offline_buffer)} chunks; metrics OK")

    print("[on_train_end] fitting FAISS…")
    model.on_train_end()
    assert model._offline_buffer == [], "buffer not cleared"
    assert all(
        bool(l._is_initialized.item()) for l in model._rqkmeans.quantizer.layers
    ), "centroids not initialised"

    print("[eval]")
    model.eval()
    batch = _make_batch(B, D)
    preds = model.predict(batch)
    losses = model.loss(preds, batch)
    model.update_metric(preds, batch, losses)
    eval_metrics = model.compute_metric()
    codes = preds["codes"]
    assert codes.shape == (B, L) and (codes >= 0).all() and (codes < 16).all()
    assert all(torch.isfinite(v).all() for v in eval_metrics.values())
    print(f"[eval] codes shape={tuple(codes.shape)}; metrics OK")

    print("[infer]")
    model.set_is_inference(True)
    preds = model.predict(batch)
    assert "codes" in preds
    assert "input_embedding" not in preds, "infer should not return aux fields"
    print(f"[infer] keys={sorted(preds)} codes shape={tuple(preds['codes'].shape)}")


# ---------------------------------------------------------------------------
# SidRqvae (no-clip)
# ---------------------------------------------------------------------------

def _build_sid_rqvae_no_clip(input_dim: int, n_layers: int) -> SidRqvae:
    codebook = ",".join(["16"] * n_layers)
    cfg = sid_model_pb2.SidRqvae(
        input_dim=input_dim,
        embed_dim=8,
        codebook=codebook,
        forward_mode="ste",
        loss_type="mse",
        kmeans_init=False,
        embedding_feature_name="item_emb",
    )
    mc = model_pb2.ModelConfig(
        feature_groups=[
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            )
        ],
        sid_rqvae=cfg,
    )
    m = SidRqvae(model_config=mc, features=[], labels=[])
    init_parameters(m, device=torch.device("cpu"))
    return m


def lifecycle_rqvae_no_clip() -> None:
    print("\n=== SidRqvae (no clip) lifecycle ===")
    B, D, L = 8, 32, 2
    model = _build_sid_rqvae_no_clip(input_dim=D, n_layers=L)
    model.init_loss()
    model.init_metric()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("[train] 8 steps with backward + optim")
    model.train()
    first_loss = last_loss = None
    for step in range(8):
        batch = _make_batch(B, D)
        preds = model.predict(batch)
        losses = model.loss(preds, batch)
        total = sum(losses.values())
        optim.zero_grad()
        total.backward()
        optim.step()
        model.update_train_metric(preds, batch)
        if step == 0:
            first_loss = float(total.detach())
        last_loss = float(total.detach())
    train_metrics = model.compute_train_metric()
    assert all(torch.isfinite(v).all() for v in train_metrics.values())
    print(f"[train] loss {first_loss:.4f} -> {last_loss:.4f}; metrics OK")

    print("[on_train_end] no-op for SidRqvae")
    model.on_train_end()

    print("[eval]")
    model.eval()
    batch = _make_batch(B, D)
    preds = model.predict(batch)
    losses = model.loss(preds, batch)
    model.update_metric(preds, batch, losses)
    eval_metrics = model.compute_metric()
    assert "codes" in preds and preds["codes"].shape == (B, L)
    assert "x_hat" in preds and preds["x_hat"].shape == (B, D)
    assert all(torch.isfinite(v).all() for v in eval_metrics.values())
    print(f"[eval] codes shape={tuple(preds['codes'].shape)}; metrics OK")

    print("[infer]")
    model.set_is_inference(True)
    preds = model.predict(batch)
    assert "codes" in preds
    assert "x_hat" not in preds, "infer should not return x_hat"
    print(f"[infer] keys={sorted(preds)}")


# ---------------------------------------------------------------------------
# SidRqvae (CLIP)
# ---------------------------------------------------------------------------

def _build_sid_rqvae_clip(input_dim: int, n_layers: int) -> SidRqvae:
    codebook = ",".join(["16"] * n_layers)
    cfg = sid_model_pb2.SidRqvae(
        input_dim=input_dim,
        embed_dim=8,
        codebook=codebook,
        forward_mode="ste",
        loss_type="mse",
        kmeans_init=False,
        embedding_feature_name="item_emb",
    )
    cfg.clip_config.CopyFrom(
        sid_model_pb2.ClipConfig(clip_feature_name="image_emb")
    )
    mc = model_pb2.ModelConfig(
        feature_groups=[
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            )
        ],
        sid_rqvae=cfg,
    )
    m = SidRqvae(model_config=mc, features=[], labels=[])
    init_parameters(m, device=torch.device("cpu"))
    return m


def _mixed_batch(B: int, D: int) -> Batch:
    """Half recon (image_emb == item_emb), half clip (different)."""
    item = torch.randn(B, D)
    image = item.clone()
    image[B // 2:] = torch.randn(B - B // 2, D)
    kt = KeyedTensor.from_tensor_list(
        keys=["item_emb", "image_emb"],
        tensors=[item, image],
    )
    return Batch(
        dense_features={BASE_DATA_GROUP: kt},
        sparse_features={},
        labels={},
    )


def lifecycle_rqvae_clip() -> None:
    print("\n=== SidRqvae (CLIP mixed) lifecycle ===")
    B, D, L = 8, 32, 2
    model = _build_sid_rqvae_clip(input_dim=D, n_layers=L)
    model.init_loss()
    model.init_metric()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("[train] 8 mixed steps with backward + optim")
    model.train()
    first_loss = last_loss = None
    for step in range(8):
        batch = _mixed_batch(B, D)
        preds = model.predict(batch)
        losses = model.loss(preds, batch)
        total = sum(losses.values())
        optim.zero_grad()
        total.backward()
        optim.step()
        model.update_train_metric(preds, batch)
        if step == 0:
            first_loss = float(total.detach())
        last_loss = float(total.detach())
    train_metrics = model.compute_train_metric()
    assert all(torch.isfinite(v).all() for v in train_metrics.values())
    print(f"[train] loss {first_loss:.4f} -> {last_loss:.4f}; metrics OK")

    print("[eval]")
    model.eval()
    batch = _mixed_batch(B, D)
    preds = model.predict(batch)
    losses = model.loss(preds, batch)
    model.update_metric(preds, batch, losses)
    eval_metrics = model.compute_metric()
    for key in ("codes", "x_hat", "recon_loss", "clip_loss", "commitment_loss"):
        assert key in preds, f"missing {key!r} in eval preds: {sorted(preds)}"
    assert preds["codes"].shape == (B, L)
    assert all(torch.isfinite(v).all() for v in eval_metrics.values())
    print(f"[eval] keys={sorted(preds)}; codes shape={tuple(preds['codes'].shape)}")

    print("[infer] (clip-aware)")
    model.set_is_inference(True)
    # In inference _predict_mixed shortcuts; image_emb need not be present.
    item = torch.randn(B, D)
    kt = KeyedTensor.from_tensor_list(keys=["item_emb"], tensors=[item])
    infer_batch = Batch(
        dense_features={BASE_DATA_GROUP: kt},
        sparse_features={},
        labels={},
    )
    preds = model.predict(infer_batch)
    assert "codes" in preds and preds["codes"].shape == (B, L)
    assert "x_hat" not in preds, "infer should not return x_hat"
    print(f"[infer] keys={sorted(preds)}")


def main() -> None:
    lifecycle_rqkmeans()
    lifecycle_rqvae_no_clip()
    lifecycle_rqvae_clip()
    print("\n=== ALL LIFECYCLES OK ===")


if __name__ == "__main__":
    sys.exit(main())
