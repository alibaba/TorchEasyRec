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

"""SidRqkmeans: SID generation model using residual K-Means.

Training is FAISS-only: ``predict`` collects embeddings into a CPU
buffer; the actual FAISS fit is triggered ONCE after the train_eval
loop ends, via the :meth:`BaseModel.on_train_end` lifecycle hook
(``tzrec.main`` calls ``_model.on_train_end()`` unconditionally).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torchmetrics
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.sid.kmeans import recon_diagnostics
from tzrec.modules.sid.residual_kmeans_quantizer import (
    ResidualKMeansQuantizer,
)
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


def _coerce_proto_numbers(d: Dict) -> Dict:
    """Coerce float-typed integers back to int.

    ``google.protobuf.Struct.number_value`` is always float, but most
    ``faiss.Kmeans`` kwargs (``niter``, ``seed``, ``nredo``, ...) require
    Python ``int``. This helper converts any float that is an exact
    integer to ``int`` for downstream consumption.
    """
    out: Dict = {}
    for k, v in d.items():
        if isinstance(v, float) and v.is_integer():
            out[k] = int(v)
        else:
            out[k] = v
    return out


class SidRqkmeans(BaseSidModel):
    """SID generation model using residual K-Means (FAISS-only).

    No gradient-based training. The codebook is built once at the end
    of the train_eval loop via a single FAISS K-Means pass over the
    embeddings collected during training.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)

        cfg = self._model_config  # SidRqkmeans proto message

        # config_to_kwargs returns Struct numbers as floats (it is
        # MessageToDict under the hood), so _coerce_proto_numbers restores
        # the ints faiss.Kmeans expects (niter, seed, nredo, ...).
        self._faiss_kwargs = (
            _coerce_proto_numbers(config_util.config_to_kwargs(cfg.faiss_kmeans_kwargs))
            if cfg.HasField("faiss_kmeans_kwargs")
            else {}
        )

        self._quantizer = ResidualKMeansQuantizer(
            embed_dim=self._input_dim,
            n_layers=self._n_layers,
            n_embed=self._n_embed_list,
            normalize_residuals=self._normalize_residuals,
            faiss_kmeans_kwargs=self._faiss_kwargs,
        )

        # Per-rank reservoir cap. FAISS K-Means only ever consumes
        # K * max_points_per_centroid points (it subsamples internally), so
        # buffering the full corpus is wasted memory. We reservoir-sample to
        # that target instead, split across ranks so the gathered set on
        # rank0 is ~train_sample_size and FAISS does no further subsampling.
        # Use the LARGEST per-layer K so non-uniform codebooks (e.g.
        # [256, 512, 1024]) still feed their biggest layer enough points.
        k = max(self._n_embed_list)
        max_ppc = int(self._faiss_kwargs.get("max_points_per_centroid", 256))
        global_target = (
            cfg.train_sample_size if cfg.train_sample_size > 0 else k * max_ppc
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._sample_cap = max(1, -(-global_target // world_size))  # ceil div

        # Bounded host-resident reservoir (allocated lazily on first batch,
        # once the embedding dim/device is known). ``_n_filled`` slots hold
        # data; ``_n_seen`` is the running count for the sampling probability.
        self._reservoir: Optional[torch.Tensor] = None
        self._n_filled = 0
        self._n_seen = 0

        # KMeans has no learnable parameters (centroids use register_buffer).
        # Add dummy param to keep optimizer/DDP happy.
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    @torch.no_grad()
    def _reservoir_add(self, x: torch.Tensor) -> None:
        """Add a batch to the bounded reservoir (Vitter's Algorithm R).

        Keeps a uniform random ``self._sample_cap`` subset of every embedding
        seen so far in O(cap) host memory, in a single streaming pass.

        Args:
            x (Tensor): a batch of embeddings, shape (B, D); copied to host.
        """
        x = x.detach().to("cpu", dtype=torch.float32)
        cap = self._sample_cap
        if self._reservoir is None:
            self._reservoir = torch.empty(cap, x.shape[1], dtype=torch.float32)

        # Phase 1: fill empty slots first.
        if self._n_filled < cap:
            take = min(x.shape[0], cap - self._n_filled)
            self._reservoir[self._n_filled : self._n_filled + take] = x[:take]
            self._n_filled += take
            self._n_seen += take
            x = x[take:]
            if x.shape[0] == 0:
                return

        # Phase 2: replacement. Row j (0-indexed in x) is the
        # (n_seen + j)-th item seen; it enters the reservoir with prob
        # cap / (n_seen + j + 1), displacing a uniformly-random slot.
        r = x.shape[0]
        pos = self._n_seen + torch.arange(r)
        accept = torch.rand(r) < (cap / (pos + 1).to(torch.float64))
        idx = accept.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            slots = torch.randint(0, cap, (idx.numel(),))
            # Intra-batch slot collisions resolve last-write-wins; the bias is
            # O(B/cap) per step and negligible for codebook fitting.
            self._reservoir[slots] = x[idx]
        self._n_seen += r

    def _reservoir_sample(self) -> torch.Tensor:
        """Return the filled portion of the reservoir, shape (n_filled, D)."""
        if self._reservoir is None or self._n_filled == 0:
            return torch.empty(0, self._input_dim, dtype=torch.float32)
        return self._reservoir[: self._n_filled]

    def _reset_reservoir(self) -> None:
        """Drop the reservoir after the FAISS fit to free host memory."""
        self._reservoir = None
        self._n_filled = 0
        self._n_seen = 0

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Training: buffer embeddings only (codes are dummy until FAISS fits).
        Eval/inference (after ``on_train_end``): real predict + lookup.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        embedding = self._extract_feature(batch)

        # Training: reservoir-sample into a bounded host buffer for the
        # end-of-loop FAISS fit, and return dummy codes — the codebook does
        # not exist yet. The reservoir caps memory at _sample_cap rows
        # regardless of corpus size (FAISS only consumes a subset anyway).
        if self.is_train:
            self._reservoir_add(embedding)
            B = embedding.shape[0]
            return {
                "codes": torch.zeros(
                    B, self._n_layers, dtype=torch.long, device=embedding.device
                )
            }

        codes, quantized = self._quantizer(embedding)

        predictions: Dict[str, torch.Tensor] = {
            "codes": codes,
        }

        if self.is_eval:
            predictions["quantized"] = quantized
            predictions["input_embedding"] = embedding

        return predictions

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model.

        Returns zero loss to keep TrainWrapper backward happy.
        _dummy_param * 0.0 ensures a compute graph exists so DDP
        does not complain about unused parameters.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.

        Return:
            losses (dict): a dict of loss tensor.
        """
        return {"dummy_loss": self._dummy_param.sum() * 0.0}

    def init_metric(self) -> None:
        """Initialize metric modules (shared eval metrics + rel_loss).

        Only eval metrics are registered. During training ``predict``
        returns dummy zero codes (the codebook does not exist yet), so
        any train-time metric would be either NaN or trivially constant;
        the inherited no-op ``update_train_metric`` keeps the train path
        empty (``compute_train_metric`` then returns an empty dict, which
        the framework already tolerates).
        """
        super().init_metric()
        self._metric_modules["rel_loss"] = torchmetrics.MeanMetric()

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
            losses (dict, optional): a dict of loss.
        """
        if "input_embedding" in predictions:
            _, rel = recon_diagnostics(
                predictions["input_embedding"],
                predictions["quantized"],
            )
            # MeanSquaredError aggregates (preds, target) itself; rel_loss has
            # no torchmetrics equivalent so it stays a MeanMetric.
            self._metric_modules["mse"].update(
                predictions["quantized"], predictions["input_embedding"]
            )
            self._metric_modules["rel_loss"].update(rel)

        self._metric_modules["unique_sid_ratio"].update(predictions["codes"])

    @torch.no_grad()
    def on_train_end(self) -> None:
        """Trigger one-shot FAISS fit after the train_eval loop ends.

        Overrides :meth:`BaseModel.on_train_end`. Called unconditionally
        by ``tzrec.main.train_and_evaluate`` after the training loop exits.

        DDP behavior:
            - rank0: receive each rank's reservoir sample via gather_object,
              concat, run FAISS fit, then broadcast centroids to all ranks.
            - other ranks: ship their reservoir sample via gather_object
              (dst=0) and wait for the broadcast.

        No cross-rank empty-buffer handshake is needed: the dataset layer
        enforces ``num_files >= world_size`` (``tzrec.datasets.dataset``
        raises otherwise), so in synchronized training every rank receives
        at least one shard and reaches the gather with a non-empty sample.
        """
        is_ddp = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

        local = self._reservoir_sample()
        self._reset_reservoir()

        if is_ddp:
            # DDP path: every rank ships its reservoir sample to rank 0 via
            # gather_object. Each sample is bounded by _sample_cap, so the
            # gathered set on rank0 is ~train_sample_size and FAISS does no
            # further subsampling.
            rank = dist.get_rank()
            gathered: Optional[List[Optional[torch.Tensor]]] = (
                [None] * dist.get_world_size() if rank == 0 else None
            )
            dist.gather_object(local, gathered, dst=0)
            del local
            if rank == 0:
                assert gathered is not None
                full = torch.cat([g for g in gathered if g is not None], dim=0)
                del gathered
                logger.info(
                    "[SidRqkmeans.on_train_end] rank0 fitting FAISS "
                    "on %d samples (D=%d)." % (full.shape[0], full.shape[1])
                )
                self._quantizer.train_offline(full, verbose=True)
                del full
            # Broadcast centroids and set the init flag locally on every
            # rank. ``_is_initialized`` is a bool buffer and NCCL's bool
            # dtype support is inconsistent across versions, so we avoid
            # a separate broadcast for it — all ranks enter this block in
            # lockstep, so a local fill_() keeps state consistent.
            for layer in self._quantizer.layers:
                dist.broadcast(layer.centroids, src=0)
                layer._is_initialized.fill_(True)
            dist.barrier()
            return

        # Single-process path. Guard an empty sample with a plain local check
        # (no collective): on_train_end may be invoked without a training pass.
        if local.shape[0] == 0:
            logger.warning(
                "[SidRqkmeans.on_train_end] empty reservoir; skipping FAISS "
                "fit. Did the train_eval loop run?"
            )
            return

        logger.info(
            "[SidRqkmeans.on_train_end] fitting FAISS on %d samples (D=%d)."
            % (local.shape[0], local.shape[1])
        )
        self._quantizer.train_offline(local, verbose=True)
