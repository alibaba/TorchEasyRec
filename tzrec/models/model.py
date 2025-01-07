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

# Copyright (c) Alibaba, Inc. and its affiliates.
from itertools import chain
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchmetrics
from torch import nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollectionInterface,
)

from tzrec.datasets.data_parser import DataParser
from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.load_class import get_register_class_meta

_MODEL_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_MODEL_CLASS_MAP)


class BaseModel(nn.Module, metaclass=_meta_cls):
    """TorchEasyRec base model.

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
        super().__init__(**kwargs)
        self._base_model_config = model_config
        self._model_type = model_config.WhichOneof("model")
        self._features = features
        self._labels = labels
        self._model_config = (
            getattr(model_config, self._model_type) if self._model_type else None
        )
        self._metric_modules = nn.ModuleDict()
        self._loss_modules = nn.ModuleDict()

        if sample_weights:
            self._sample_weights = sample_weights

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        raise NotImplementedError

    def init_loss(self) -> None:
        """Initialize loss modules."""
        raise NotImplementedError

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.

        Return:
            losses (dict): a dict of loss tensor.
        """
        raise NotImplementedError

    def init_metric(self) -> None:
        """Initialize metric modules."""
        raise NotImplementedError

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
        raise NotImplementedError

    def compute_metric(self) -> Dict[str, torch.Tensor]:
        """Compute metric.

        Return:
            metric_result (dict): a dict of metric result tensor.
        """
        metric_results = {}
        for metric_name, metric in self._metric_modules.items():
            metric_results[metric_name] = metric.compute()
            metric.reset()
        return metric_results

    def sparse_parameters(self) -> Iterable[nn.Parameter]:
        """Get an iterator over sparse parameters of the module."""
        q = Queue()
        q.put(self)
        parameters_list = []
        while not q.empty():
            m = q.get()
            if isinstance(m, EmbeddingBagCollectionInterface) or isinstance(
                m, EmbeddingCollectionInterface
            ):
                parameters_list.append(m.parameters())
            else:
                for child in m.children():
                    q.put(child)
        return chain.from_iterable(parameters_list)

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model."""
        return self.predict(batch)

    def _init_loss_metric_impl(self, loss_cfg: LossConfig, suffix: str = "") -> None:
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        self._metric_modules[loss_name] = torchmetrics.MeanMetric()

    def _update_loss_metric_impl(
        self,
        losses: Dict[str, torch.Tensor],
        batch: Batch,
        label_name: str,
        loss_cfg: LossConfig,
        suffix: str = "",
    ) -> None:
        label = batch.labels[label_name]
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        loss = losses[loss_name]
        self._metric_modules[loss_name].update(loss, loss.new_tensor(label.size(0)))


TRAIN_OUT_TYPE = Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Batch]
TRAIN_FWD_TYPE = Tuple[torch.Tensor, TRAIN_OUT_TYPE]


class TrainWrapper(nn.Module):
    """Model train wrapper for pipeline."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.model = module
        self.model.init_loss()
        self.model.init_metric()

    def forward(self, batch: Batch) -> TRAIN_FWD_TYPE:
        """Predict and compute loss.

        Args:
            batch (Batch): input batch data.

        Return:
            total_loss (Tensor): total loss.
            losses (dict): a dict of loss tensor.
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
        """
        predictions = self.model.predict(batch)
        losses = self.model.loss(predictions, batch)
        total_loss = torch.stack(list(losses.values())).sum()

        losses = {k: v.detach() for k, v in losses.items()}
        predictions = {k: v.detach() for k, v in predictions.items()}
        return total_loss, (losses, predictions, batch)


class ScriptWrapper(nn.Module):
    """Model inference wrapper for jit.script."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.model = module
        self._features = self.model._features
        self._data_parser = DataParser(self._features)

    def get_batch(
        self,
        data: Dict[str, torch.Tensor],
        # pyre-ignore [9]
        device: torch.device = "cpu",
    ) -> Batch:
        """Get batch."""
        batch = self._data_parser.to_batch(data)
        batch = batch.to(device, non_blocking=True)
        return batch

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        # pyre-ignore [9]
        device: torch.device = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            data (dict): a dict of input data for Batch.
            device (torch.device): inference device.

        Return:
            predictions (dict): a dict of predicted result.
        """
        batch = self.get_batch(data, device)
        return self.model.predict(batch)


class ExportWrapperAOT(ScriptWrapper):
    """Model inference wrapper for aot export."""

    # pyre-ignore [14]
    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            data (dict): a dict of input data for Batch.

        Return:
            predictions (dict): a dict of predicted result.
        """
        batch = self._data_parser.to_batch(data)
        batch = batch.to(torch.device("cuda"), non_blocking=True)
        return self.model.predict(batch)
