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

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics
from google.protobuf.json_format import MessageToDict
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.sid_generation import RQKMeans
from tzrec.modules.sid_generation.kmeans import recon_diagnostics
from tzrec.protos.model_pb2 import ModelConfig
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

        self._faiss_kwargs = (
            _coerce_proto_numbers(MessageToDict(cfg.faiss_kmeans_kwargs))
            if cfg.HasField("faiss_kmeans_kwargs")
            else {}
        )

        self._rqkmeans = RQKMeans(
            embed_dim=cfg.input_dim,
            n_layers=self._n_layers,
            n_embed=self._n_embed_list,
            normalize_residuals=cfg.normalize_residuals,
            faiss_kmeans_kwargs=self._faiss_kwargs,
        )

        # CPU buffer for embeddings collected during training; FAISS
        # consumes it in on_train_end() at end-of-loop.
        self._offline_buffer: List[torch.Tensor] = []

        # KMeans has no learnable parameters (centroids use register_buffer).
        # Add dummy param to keep optimizer/DDP happy.
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

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

        # Training: buffer for the end-of-loop FAISS fit and return dummy
        # codes — the codebook does not exist yet.
        # TODO(perf): .cpu() is a synchronous D2H per step and the buffer
        # grows unbounded with steps. Rework to either (a) GPU-resident
        # buffer + bulk D2H in on_train_end with size cap, or (b) replace
        # the train pass with an inference_mode corpus walk launched from
        # on_train_end. Skipped here to avoid OOM-vs-refactor tradeoffs;
        # tracked separately.
        if self.is_train:
            self._offline_buffer.append(embedding.detach().cpu())
            B = embedding.shape[0]
            return {
                "codes": torch.zeros(
                    B, self._n_layers, dtype=torch.long, device=embedding.device
                )
            }

        result = self._rqkmeans(embedding)

        predictions: Dict[str, torch.Tensor] = {
            "codes": result["codes"],
        }

        if self.is_eval:
            predictions["quantized"] = result["quantized"]
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
            mse, rel = recon_diagnostics(
                predictions["input_embedding"],
                predictions["quantized"],
            )
            self._metric_modules["mse"].update(mse)
            self._metric_modules["rel_loss"].update(rel)

        self._update_unique_sid_ratio(predictions["codes"])

    @torch.no_grad()
    def on_train_end(self) -> None:
        """Trigger one-shot FAISS fit after the train_eval loop ends.

        Overrides :meth:`BaseModel.on_train_end`. Called unconditionally
        by ``tzrec.main.train_and_evaluate`` after the training loop
        exits. No-op when the buffer is empty.

        DDP behavior:
            - rank0: receive local buffers via gather_object, concat,
              run FAISS fit, then broadcast centroids to other ranks.
            - other ranks: ship local buffer via gather_object(dst=0)
              and wait for the broadcast.
        """
        is_ddp = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

        # A local-only empty check would deadlock: the empty rank returns
        # while peers block in gather_object below. OR the flag across
        # ranks and bail together if any rank is empty.
        local_empty = len(self._offline_buffer) == 0
        if is_ddp:
            # int32, not bool — NCCL bool support is version-dependent.
            flag = torch.tensor(
                int(local_empty),
                dtype=torch.int32,
                device=self._dummy_param.device,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            any_empty = bool(flag.item())
        else:
            any_empty = local_empty

        if any_empty:
            if (not is_ddp) or dist.get_rank() == 0:
                logger.warning(
                    "[SidRqkmeans.on_train_end] at least one rank has an "
                    "empty offline buffer; skipping FAISS fit on all ranks. "
                    "Did the train_eval loop run, and is the per-rank shard "
                    "non-empty?"
                )
            return

        if is_ddp:
            # DDP path: every rank ships its local buffer to rank 0 via
            # gather_object (variable-length pickle — fine for this one-
            # shot, CPU-resident gather). Only rank 0 holds the corpus,
            # so peak memory is O(world_size) on rank 0 and O(1) elsewhere
            # (vs O(world_size²) for all_gather_object).
            local = torch.cat(self._offline_buffer, dim=0)
            del self._offline_buffer
            self._offline_buffer = []

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
                self._rqkmeans.train_offline(full, verbose=True)
                del full
            # Broadcast centroids and set the init flag locally on every
            # rank. ``_is_initialized`` is a bool buffer and NCCL's bool
            # dtype support is inconsistent across versions, so we avoid
            # a separate broadcast for it — all ranks enter this block in
            # lockstep, so a local fill_() keeps state consistent.
            for layer in self._rqkmeans.quantizer.layers:
                dist.broadcast(layer.centroids, src=0)
                layer._is_initialized.fill_(True)
            dist.barrier()
        else:
            # Single-process path: build the full numpy matrix directly
            # from the buffer list, popping each chunk after copy so the
            # transient memory high-water mark stays ~= final matrix size
            # (instead of 2× when going through torch.cat).
            N = sum(t.shape[0] for t in self._offline_buffer)
            D = self._offline_buffer[0].shape[1]
            logger.info(
                "[SidRqkmeans.on_train_end] fitting FAISS on "
                "%d samples (D=%d)." % (N, D)
            )
            full_np = np.empty((N, D), dtype=np.float32)
            offset = 0
            # Pop from the front; each popped tensor is released before
            # the next copy so cumulative torch memory shrinks monotonically.
            while self._offline_buffer:
                t = self._offline_buffer.pop(0)
                n = t.shape[0]
                # .float().numpy() returns a view sharing storage with
                # the fp32 tensor; the subsequent assignment copies into
                # full_np, after which ``t`` can be freed.
                full_np[offset : offset + n] = t.float().numpy()
                offset += n
                del t
            del self._offline_buffer
            self._offline_buffer = []

            # train_offline takes ownership of ``full_np`` (in-place
            # residual updates); drop our reference after the call.
            self._rqkmeans.train_offline(full_np, verbose=True)
            del full_np
