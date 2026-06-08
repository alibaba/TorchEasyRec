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
    """Coerce whole-valued floats back to int.

    ``Struct.number_value`` is always float, but faiss.Kmeans kwargs
    (``niter``, ``seed``, ...) need ``int``.
    """
    return {
        k: int(v) if isinstance(v, float) and v.is_integer() else v
        for k, v in d.items()
    }


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

        # config_to_kwargs yields Struct numbers as floats; coerce back to int.
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

        self._init_reservoir()

        # KMeans has no learnable params; a dummy keeps the optimizer/DDP happy.
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _init_reservoir(self) -> None:
        """Set up the bounded host reservoir for the end-of-loop FAISS fit.

        Per-rank cap: target the points the FAISS fit will subsample to
        (``ResidualKMeansQuantizer.default_fit_sample_size``), split across
        ranks, rather than buffer the whole corpus.
        """
        target = self._model_config.train_sample_size
        global_target = (
            target if target > 0 else self._quantizer.default_fit_sample_size()
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._sample_cap = max(1, -(-global_target // world_size))  # ceil div

        # Allocated lazily on the first batch. _n_filled = used slots;
        # _n_seen = running count for the accept prob.
        self._reservoir: Optional[torch.Tensor] = None
        self._n_filled = 0
        self._n_seen = 0

    @torch.no_grad()
    def _reservoir_add(self, x: torch.Tensor) -> None:
        """Stream a batch into the reservoir (Vitter Algorithm R).

        Keeps a uniform ``_sample_cap`` sample of all embeddings seen, in
        O(cap) host memory.

        Args:
            x (Tensor): batch of embeddings, shape (B, D).
        """
        x = x.detach()
        cap = self._sample_cap
        if self._reservoir is None:
            self._reservoir = torch.empty(cap, x.shape[1], dtype=torch.float32)

        # Phase 1: fill empty slots first. Copy only the rows we keep to host.
        if self._n_filled < cap:
            take = min(x.shape[0], cap - self._n_filled)
            self._reservoir[self._n_filled : self._n_filled + take] = x[:take].to(
                "cpu", dtype=torch.float32
            )
            self._n_filled += take
            self._n_seen += take
            x = x[take:]
            if x.shape[0] == 0:
                return

        # Phase 2: row j enters with prob cap/(n_seen+j+1), displacing a random
        # slot. The accept decision needs only counts, so compute it on host and
        # copy ONLY accepted rows (in steady state, almost none) — avoiding the
        # whole-batch GPU->CPU copy. float64 keeps n_seen+j+1 exact past 2**24.
        r = x.shape[0]
        pos = self._n_seen + torch.arange(r)
        accept = torch.rand(r) < (cap / (pos + 1).to(torch.float64))
        idx = accept.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            slots = torch.randint(0, cap, (idx.numel(),))
            # Slot collisions are last-write-wins; O(B/cap) bias, negligible here.
            self._reservoir[slots] = x[idx.to(x.device)].to("cpu", dtype=torch.float32)
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

        # Training: just reservoir-sample for the end-of-loop FAISS fit and
        # return dummy codes — the codebook does not exist yet.
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

        Zero loss via ``_dummy_param * 0`` — gives TrainWrapper/DDP a compute
        graph despite there being no real trainable params.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.

        Return:
            losses (dict): a dict of loss tensor.
        """
        return {"dummy_loss": self._dummy_param.sum() * 0.0}

    def init_metric(self) -> None:
        """Register eval metrics (shared ``mse`` + ``rel_loss``).

        Train-time metrics are intentionally absent: ``predict`` returns dummy
        codes pre-fit, so the inherited no-op ``update_train_metric`` keeps the
        train path empty.
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
            # mse aggregates (preds, target) itself; rel_loss has no
            # torchmetrics equivalent, so it stays a MeanMetric.
            self._metric_modules["mse"].update(
                predictions["quantized"], predictions["input_embedding"]
            )
            self._metric_modules["rel_loss"].update(rel)

        self._metric_modules["unique_sid_ratio"].update(predictions["codes"])

    @torch.no_grad()
    def on_train_end(self) -> bool:
        """Fit the FAISS codebook once, after the train_eval loop exits.

        Overrides :meth:`BaseModel.on_train_end` (called unconditionally by
        ``tzrec.main``). DDP: every rank gather_objects its reservoir to rank0,
        which fits and broadcasts the centroids back.

        An empty reservoir only happens for a pathologically tiny corpus
        (rebalance splits rows across ``num_workers * world_size``); it then
        fails fast via the fit-status broadcast rather than hanging.

        Returns:
            is_ckpt_after_train (bool): ``True`` if the codebook was fitted
            (centroids changed → force a final checkpoint), ``False`` if the
            fit was skipped (empty reservoir).
        """
        is_ddp = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

        local = self._reservoir_sample()
        self._reset_reservoir()

        if is_ddp:
            # Each rank ships its (capped) reservoir to rank0, which fits.
            rank = dist.get_rank()
            gathered: Optional[List[Optional[torch.Tensor]]] = (
                [None] * dist.get_world_size() if rank == 0 else None
            )
            dist.gather_object(local, gathered, dst=0)
            del local
            fit_ok = True
            if rank == 0:
                assert gathered is not None
                try:
                    full = torch.cat([g for g in gathered if g is not None], dim=0)
                    del gathered
                    logger.info(
                        "[SidRqkmeans.on_train_end] rank0 fitting FAISS "
                        "on %d samples (D=%d)." % (full.shape[0], full.shape[1])
                    )
                    self._quantizer.train_offline(full, verbose=True)
                    del full
                except Exception:  # noqa: BLE001
                    # Don't raise yet — peers would hang on the broadcast below.
                    # Signal failure via the status flag so all ranks raise.
                    # logger.exception keeps the traceback so the rank0-only
                    # failure is diagnosable from the log.
                    fit_ok = False
                    logger.exception(
                        "[SidRqkmeans.on_train_end] rank0 FAISS fit failed"
                    )
            # Broadcast rank0's status (int, not bool — see NCCL note below) so
            # a rank0-only failure makes all ranks raise instead of deadlocking.
            status = torch.tensor(
                [1 if fit_ok else 0],
                device=self._quantizer.layers[0].centroids.device,
            )
            dist.broadcast(status, src=0)
            if int(status.item()) == 0:
                raise RuntimeError(
                    "[SidRqkmeans.on_train_end] FAISS fit failed on rank0; "
                    "see rank0 logs for the underlying error."
                )
            # Broadcast centroids; set the init flag locally (avoids
            # broadcasting a bool buffer — NCCL bool support is inconsistent).
            # All ranks are in lockstep, so a local mark_initialized_() agrees.
            for layer in self._quantizer.layers:
                dist.broadcast(layer.centroids, src=0)
                layer.mark_initialized_()
            dist.barrier()
            return True

        # Single-process: guard an empty reservoir with a plain local check.
        if local.shape[0] == 0:
            logger.warning(
                "[SidRqkmeans.on_train_end] empty reservoir; skipping FAISS "
                "fit. Did the train_eval loop run?"
            )
            return False

        logger.info(
            "[SidRqkmeans.on_train_end] fitting FAISS on %d samples (D=%d)."
            % (local.shape[0], local.shape[1])
        )
        self._quantizer.train_offline(local, verbose=True)
        return True
