"""Smoke test for RQ-VAE / RQ-KMeans on the maxcompute CSV samples.

Loads /workspace/fangtinglin/dataset_maxcompute/forge_contrastive_item_embedding.csv,
parses the JSON-string embedding columns into float32 tensors, and runs:

  1) RQKMeans   (online MiniBatch + offline FAISS)
  2) ResidualQuantized in RQ-VAE style (Sinkhorn off, EMA optional)

It runs deterministically (manual seed) so the same script can be checked out
against both code_style and onerec_kmeans branches to verify numerical parity
(or document any intentional divergence).
"""

import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

CSV_PATH = (
    "/workspace/fangtinglin/dataset_maxcompute/"
    "forge_contrastive_item_embedding.csv"
)


_EMB_RE = re.compile(r"\[(?:[^\[\]]*)\]")


def _parse_line(line: str) -> tuple[np.ndarray, np.ndarray] | None:
    # CSV layout per row:
    #   id1,id2,[<json list>],[<json list>],is_contrastive
    # The two bracketed fields are not properly CSV-quoted, but each
    # contains no nested brackets, so a regex captures them cleanly.
    matches = _EMB_RE.findall(line)
    if len(matches) != 2:
        return None
    try:
        e1 = np.asarray(json.loads(matches[0]), dtype=np.float32)
        e2 = np.asarray(json.loads(matches[1]), dtype=np.float32)
    except json.JSONDecodeError:
        return None
    return e1, e2


def load_embeddings(limit: int | None = None) -> torch.Tensor:
    rows1 = []
    rows2 = []
    with open(CSV_PATH) as f:
        header = next(f)
        assert "item1_embedding" in header, header
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            parsed = _parse_line(line)
            if parsed is None:
                continue
            e1, e2 = parsed
            rows1.append(e1)
            rows2.append(e2)
    emb1 = np.stack(rows1)
    emb2 = np.stack(rows2)
    all_emb = np.concatenate([emb1, emb2], axis=0)
    return torch.from_numpy(all_emb)  # (2N, D), float32


def test_rqkmeans_offline(x: torch.Tensor) -> dict:
    from tzrec.modules.sid_generation.residual_kmeans import RQKMeans

    torch.manual_seed(0)
    np.random.seed(0)
    model = RQKMeans(
        embed_dim=x.shape[1],
        n_layers=3,
        n_embed=64,
        normalize_residuals=True,
        faiss_kmeans_kwargs={
            "niter": 20,
            "verbose": False,
            "seed": 1234,
            "spherical": False,
        },
    )

    t0 = time.time()
    model.train_offline(x, verbose=False)
    dt = time.time() - t0

    model.eval()
    codes = model.get_codes(x)
    return {
        "mode": "offline_faiss",
        "elapsed_s": round(dt, 3),
        "codes_shape": tuple(codes.shape),
        "codes_min": int(codes.min()),
        "codes_max": int(codes.max()),
        "first8": codes[:8].tolist(),
        "uniq_layer0": int(codes[:, 0].unique().numel()),
        "uniq_layer1": int(codes[:, 1].unique().numel()),
        "uniq_layer2": int(codes[:, 2].unique().numel()),
    }


def test_rqvae(x: torch.Tensor) -> dict:
    from tzrec.modules.sid_generation.rqvae import RQVAE

    torch.manual_seed(0)
    model = RQVAE(
        input_dim=x.shape[1],
        embed_dim=64,
        hidden_dims=[256, 128],
        n_layers=3,
        n_embed=[64, 64, 64],
        forward_mode="ste",
        kmeans_init=False,
        use_ema=False,
        use_sinkhorn=False,
    )
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    B = 512
    losses = []
    for epoch in range(2):
        for start in range(0, x.shape[0], B):
            out = model.forward_rqvae(x[start : start + B])
            loss = out["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
    dt = time.time() - t0

    model.eval()
    with torch.no_grad():
        out = model.forward_rqvae(x[:1024])
    return {
        "elapsed_s": round(dt, 3),
        "first_loss": round(losses[0], 5),
        "last_loss": round(losses[-1], 5),
        "first8_codes": out["codes"][:8].tolist(),
        "recon_loss_eval": round(float(out["loss"]), 5),
    }


def main() -> None:
    print(f"[smoke] loading CSV {CSV_PATH}")
    x = load_embeddings(limit=int(os.environ.get("SMOKE_ROWS", "2000")))
    print(f"[smoke] loaded x shape={tuple(x.shape)} dtype={x.dtype}")

    print("\n[smoke] === RQKMeans offline_faiss ===")
    r2 = test_rqkmeans_offline(x)
    print(json.dumps(r2, indent=2))

    print("\n[smoke] === RQVAE 2-epoch train ===")
    r3 = test_rqvae(x)
    print(json.dumps(r3, indent=2))

    print("\n[smoke] DONE")


if __name__ == "__main__":
    sys.exit(main())
