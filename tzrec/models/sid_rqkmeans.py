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

"""SidRqkmeans: SID generation model using residual Mini-Batch KMeans.

Two training backends:
  - 'online'        : centroids updated on the fly via train_step()
                      during predict() in training mode (default).
  - 'offline_faiss' : predict() only collects embeddings into a CPU
                      buffer; the actual FAISS fit is triggered ONCE
                      after the train_eval loop ends, via
                      `flush_offline_fit()` invoked by main.py.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchmetrics
from google.protobuf.json_format import MessageToDict
from torch import nn

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.modules.sid_generation import RQKMeans
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.logging_util import logger


def _parse_int_list(s: str) -> List[int]:
    """Parse comma-separated int string, e.g. '256,128' -> [256, 128]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _recon_loss(
    x: torch.Tensor, out: torch.Tensor, epsilon: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruction diagnostics aligned with OpenOneRec::ResKmeans.calc_loss.

    Args:
        x: ground-truth embedding, shape (B, D).
        out: quantized reconstruction, shape (B, D).
        epsilon: numerical stabilizer for rel_loss denominator.

    Returns:
        mse: ((out - x) ** 2).mean()
        rel_loss: (|x - out| / (max(|x|, |out|) + eps)).mean()
    """
    mse = ((out - x) ** 2).mean()
    rel = (
        torch.abs(x - out)
        / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)
    ).mean()
    return mse, rel


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


def _all_gather_concat(local: torch.Tensor) -> torch.Tensor:
    """All-gather variable-length tensors across DDP ranks and concat.

    Args:
        local: local tensor on the current rank, shape (n_local, D).

    Returns:
        Concatenated tensor (sum_n, D) on CPU.
    """
    world_size = dist.get_world_size()

    # 1) gather local sizes
    local_n = torch.tensor(
        [local.shape[0]], dtype=torch.long, device=local.device
    )
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    # 2) pad local to max_n then all_gather
    if local.shape[0] < max_n:
        pad = torch.zeros(
            max_n - local.shape[0],
            local.shape[1],
            dtype=local.dtype,
            device=local.device,
        )
        padded = torch.cat([local, pad], dim=0)
    else:
        padded = local
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # 3) trim padding and concat
    pieces = [g[:n].cpu() for g, n in zip(gathered, sizes) if n > 0]
    return torch.cat(pieces, dim=0) if pieces else local.cpu()


class SidRqkmeans(BaseModel):
    """SID generation model using residual Mini-Batch KMeans.

    No gradient-based training. Centroids are updated online via
    train_step() during the predict() call in training mode.

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
        self._embedding_feature_name = cfg.embedding_feature_name

        assert cfg.codebook, (
            "codebook must be set, e.g. '256,256,256'"
        )
        n_embed_list = _parse_int_list(cfg.codebook)
        n_layers = len(n_embed_list)

        # Resolve new fields with backward compatibility:
        # proto2 string default may return empty string if not set in
        # the textproto, so explicitly fallback to 'online'.
        self._train_mode = cfg.train_mode or "online"
        self._faiss_kwargs = (
            _coerce_proto_numbers(MessageToDict(cfg.faiss_kmeans_kwargs))
            if cfg.HasField("faiss_kmeans_kwargs")
            else {}
        )

        self._rqkmeans = RQKMeans(
            embed_dim=cfg.input_dim,
            n_layers=n_layers,
            n_embed=n_embed_list,
            normalize_residuals=cfg.normalize_residuals,
            init_buffer_size=cfg.init_buffer_size,
            train_mode=self._train_mode,
            faiss_kmeans_kwargs=self._faiss_kwargs,
        )

        # Offline mode: collect embeddings into a CPU buffer
        if self._train_mode == "offline_faiss":
            self._offline_buffer: List[torch.Tensor] = []

        # KMeans has no learnable parameters (centroids use register_buffer).
        # Add dummy param to keep optimizer/DDP happy.
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _extract_embedding(self, batch: Batch) -> torch.Tensor:
        """Extract item embedding from Batch.dense_features."""
        kt = batch.dense_features[BASE_DATA_GROUP]
        return kt[self._embedding_feature_name]

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        RQKMeans.forward() internally distinguishes training/eval:
          training: calls layer.train_step() to update centroids
          eval:     calls layer.predict() for assignment only

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        embedding = self._extract_embedding(batch)

        # Offline mode + training stage: only collect; skip _rqkmeans forward.
        if self._train_mode == "offline_faiss" and self.is_train:
            self._offline_buffer.append(embedding.detach().cpu())
            B = embedding.shape[0]
            n_layers = self._rqkmeans.quantizer.n_layers
            return {
                "codes": torch.zeros(
                    B, n_layers, dtype=torch.long, device=embedding.device
                )
            }

        # Online minibatch mode.
        result = self._rqkmeans(embedding)

        predictions: Dict[str, torch.Tensor] = {
            "codes": result["codes"],
        }

        if self.is_train or self.is_eval:
            predictions["quantized"] = result["quantized"]
            predictions["input_embedding"] = embedding

        return predictions

    def init_loss(self) -> None:
        """Initialize loss modules.

        KMeans has no gradient loss. Centroids are updated
        in predict() via train_step().
        """
        pass

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
        """Initialize metric modules."""
        # Eval metrics
        self._metric_modules["mse"] = torchmetrics.MeanMetric()
        self._metric_modules["rel_loss"] = torchmetrics.MeanMetric()
        self._metric_modules["unique_sid_ratio"] = torchmetrics.MeanMetric()

        # Train metrics (loss is dummy, only track mse/rel_loss + sid ratio)
        self._train_metric_modules["mse"] = torchmetrics.MeanMetric()
        self._train_metric_modules["rel_loss"] = torchmetrics.MeanMetric()
        self._train_metric_modules["unique_sid_ratio"] = torchmetrics.MeanMetric()

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
        """
        # Quantization MSE + rel_loss (skipped in offline_faiss train stage
        # where predictions carry only dummy codes without input_embedding).
        if "input_embedding" in predictions:
            mse, rel = _recon_loss(
                predictions["input_embedding"],
                predictions["quantized"],
            )
            self._train_metric_modules["mse"].update(mse)
            self._train_metric_modules["rel_loss"].update(rel)

        # Unique SID ratio
        codes = predictions["codes"]
        B = codes.shape[0]
        unique_sids = torch.unique(codes, dim=0).shape[0]
        self._train_metric_modules["unique_sid_ratio"].update(
            torch.tensor(unique_sids / B, device=codes.device)
        )

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
        codes = predictions["codes"]
        B = codes.shape[0]

        # Quantization MSE + rel_loss (aligned with OpenOneRec calc_loss)
        if "input_embedding" in predictions:
            mse, rel = _recon_loss(
                predictions["input_embedding"],
                predictions["quantized"],
            )
            self._metric_modules["mse"].update(mse)
            self._metric_modules["rel_loss"].update(rel)

        # Unique SID ratio
        unique_sids = torch.unique(codes, dim=0).shape[0]
        self._metric_modules["unique_sid_ratio"].update(
            torch.tensor(unique_sids / B, device=codes.device)
        )

    @torch.no_grad()
    def flush_offline_fit(self) -> None:
        """Trigger one-shot FAISS fit after the train_eval loop ends.

        Called by ``tzrec.main.train_and_evaluate`` after the training
        loop exits. No-op for ``train_mode='online'`` or when the
        offline buffer is empty.

        DDP behavior:
            - rank0: all_gather full embedding matrix, run FAISS fit,
              then broadcast (centroids + _is_initialized + _offline_locked)
              to all other ranks.
            - other ranks: send local buffer via all_gather, then wait
              for broadcast.
        """
        if self._train_mode != "offline_faiss":
            return
        if not getattr(self, "_offline_buffer", None):
            logger.warning(
                "[SidRqkmeans.flush_offline_fit] offline buffer is empty; "
                "skip FAISS fit. Did the train_eval loop run?"
            )
            return

        local = torch.cat(self._offline_buffer, dim=0)
        # Free buffer immediately after cat to reduce peak memory
        # (avoids holding both the list of tensors AND the cat result).
        del self._offline_buffer
        self._offline_buffer = []

        is_ddp = dist.is_available() and dist.is_initialized() \
            and dist.get_world_size() > 1

        if is_ddp:
            full = _all_gather_concat(local)
            del local  # no longer needed after gather
            rank = dist.get_rank()
            if rank == 0:
                logger.info(
                    "[SidRqkmeans.flush_offline_fit] rank0 fitting FAISS "
                    "on %d samples (D=%d)." % (full.shape[0], full.shape[1])
                )
                self._rqkmeans.train_offline(full, verbose=True)
            del full
            # Broadcast all codebook-related buffers (centroids + 2 guards).
            # Missing any one breaks rank consistency on subsequent forward.
            for layer in self._rqkmeans.quantizer.layers:
                dist.broadcast(layer.centroids, src=0)
                dist.broadcast(layer._is_initialized, src=0)
                dist.broadcast(layer._offline_locked, src=0)
            dist.barrier()
        else:
            logger.info(
                "[SidRqkmeans.flush_offline_fit] fitting FAISS on "
                "%d samples (D=%d)." % (local.shape[0], local.shape[1])
            )
            self._rqkmeans.train_offline(local, verbose=True)
            del local
