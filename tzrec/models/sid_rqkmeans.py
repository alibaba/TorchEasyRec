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
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.sid.kmeans_quantize import ReservoirSampler
from tzrec.modules.sid.residual_kmeans_quantizer import (
    ResidualKMeansQuantizer,
)
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.logging_util import logger


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

        # CPU-only: embeddings, reservoir, and FAISS fit all stay on the host,
        # so there are no device copies. Refuse to run when CUDA is visible.
        if torch.cuda.is_available():
            raise RuntimeError(
                "SidRqkmeans is CPU-only, but a CUDA device is visible. "
                'Run with CUDA_VISIBLE_DEVICES="" (or on a CPU-only host).'
            )

        # Single-process only: the fit runs over one process's local reservoir,
        # with no cross-rank gather. Fail fast before the (wasted) train pass.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError(
                "SidRqkmeans supports single-process training only "
                f"(world_size=1); got world_size={dist.get_world_size()}. "
                "Launch with --nproc-per-node=1."
            )

        cfg = self._model_config

        # Typed faiss kwargs: only the explicitly-set fields are forwarded, so
        # unset ones fall back to faiss's own defaults (no float->int coercion).
        self._faiss_kwargs = {
            f.name: v for f, v in cfg.faiss_kmeans_kwargs.ListFields()
        }

        self._quantizer = ResidualKMeansQuantizer(
            embed_dim=self._input_dim,
            n_layers=self._n_layers,
            n_embed=self._n_embed_list,
            normalize_residuals=self._normalize_residuals,
            faiss_kmeans_kwargs=self._faiss_kwargs,
        )

        # Bounded host reservoir for the end-of-loop fit: cap at
        # ``train_sample_size`` (when >0) else the fit's subsample size, rather
        # than buffer the whole corpus.
        target = self._model_config.train_sample_size
        cap = target if target > 0 else self._quantizer.default_fit_sample_size()
        # Fail fast: a cap below the largest codebook would only fail deep in
        # train_offline, after the whole training pass.
        max_k = max(self._n_embed_list)
        if cap < max_k:
            raise RuntimeError(
                f"reservoir cap ({cap}) < largest codebook size ({max_k}); set "
                f"train_sample_size >= {max_k} (or 0 for the default)."
            )
        self._reservoir = ReservoirSampler(cap, self._input_dim)

        # KMeans has no learnable params; a dummy keeps the optimizer/DDP happy.
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

        # Training: reservoir-sample only; codes are dummy until the fit.
        if self.is_train:
            self._reservoir.add(embedding)
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

        # Expose the centroid-sum reconstruction (``x_hat``) for update_metric
        # only once fitted — pre-fit it is all-zeros, so omitting it skips the
        # eval metrics. (Meaningful only with normalize_residuals=False.)
        if self.is_eval and self._quantizer.is_fitted:
            predictions["x_hat"] = quantized

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

    @torch.no_grad()
    def on_train_end(self) -> None:
        """Fit the FAISS codebook once, after the train_eval loop exits.

        Overrides :meth:`BaseModel.on_train_end` (called unconditionally by
        ``tzrec.main``). Single-process only (enforced by the world_size guard
        in ``__init__``): the fit runs on one process over its local reservoir,
        with no cross-rank gather/broadcast. The tail ``final=True`` checkpoint
        then persists the fitted codebook (SID runs with periodic checkpointing
        disabled, so that save is never deduped away).

        An empty reservoir only happens for a pathologically tiny corpus; the
        fit is then skipped.
        """
        # train_offline consumes its input; we hand it the reservoir buffer
        # directly (no copy) since nothing reads it after this — reset() drops
        # the sampler's reference and ``local`` is the last user of the storage.
        local = self._reservoir.sample()
        self._reservoir.reset()

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
