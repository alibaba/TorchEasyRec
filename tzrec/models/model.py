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
from collections import OrderedDict
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchmetrics
from torch import nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollectionInterface,
)

from tzrec.constant import TRAGET_REPEAT_INTERLEAVE_KEY
from tzrec.datasets.data_parser import DataParser
from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.modules.utils import BaseModule
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.model_pb2 import FeatureGroupConfig, ModelConfig
from tzrec.utils.load_class import get_register_class_meta

_MODEL_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_MODEL_CLASS_MAP)


class BaseModel(BaseModule, metaclass=_meta_cls):
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

        self._train_metric_modules = nn.ModuleDict()

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

    def compute_train_metric(self) -> Dict[str, torch.Tensor]:
        """Compute train metric."""
        metric_results = {}
        for metric_name, metric in self._train_metric_modules.items():
            metric_results[metric_name] = metric.compute()
        return metric_results

    def sparse_parameters(
        self,
    ) -> Tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
        """Get an iterator over sparse parameters of the module."""
        q = Queue()
        q.put(self)
        trainable_parameters_list = []
        frozen_parameters_list = []
        while not q.empty():
            m = q.get()
            if isinstance(m, EmbeddingBagCollectionInterface):
                frozen_names = {
                    f".{t.name}.weight"
                    for t in m.embedding_bag_configs()
                    # pyre-ignore [16]
                    if not t.trainable
                }
                for name, param in m.named_parameters():
                    frozen = any(map(lambda x: name.endswith(x), frozen_names))
                    if frozen:
                        frozen_parameters_list.append(param)
                    else:
                        trainable_parameters_list.append(param)
            elif isinstance(m, EmbeddingCollectionInterface):
                frozen_names = {
                    f".{t.name}.weight"
                    for t in m.embedding_configs()
                    # pyre-ignore [16]
                    if not t.trainable
                }
                for name, param in m.named_parameters():
                    frozen = any(map(lambda x: name.endswith(x), frozen_names))
                    if frozen:
                        frozen_parameters_list.append(param)
                    else:
                        trainable_parameters_list.append(param)
            else:
                for child in m.children():
                    q.put(child)
        return trainable_parameters_list, frozen_parameters_list

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
        label: torch.Tensor,
        loss_cfg: LossConfig,
        suffix: str = "",
    ) -> None:
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        loss = losses[loss_name]
        self._metric_modules[loss_name].update(loss, loss.new_tensor(label.size(0)))

    def get_features_in_feature_groups(
        self, feature_groups: List[FeatureGroupConfig]
    ) -> List[BaseFeature]:
        """Select features order by feature groups."""
        name_to_feature = {x.name: x for x in self._features}
        grouped_features = OrderedDict()
        for feature_group in feature_groups:
            for x in feature_group.feature_names:
                grouped_features[x] = name_to_feature[x]
            for sequence_group in feature_group.sequence_groups:
                for x in sequence_group.feature_names:
                    grouped_features[x] = name_to_feature[x]
        return list(grouped_features.values())


TRAIN_OUT_TYPE = Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Batch]
TRAIN_FWD_TYPE = Tuple[torch.Tensor, TRAIN_OUT_TYPE]


class TrainWrapper(BaseModule):
    """Model train wrapper for pipeline."""

    def __init__(
        self,
        module: nn.Module,
        device: Optional[torch.device] = None,
        mixed_precision: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = module
        self.model.init_loss()
        self.model.init_metric()
        self._device = device
        self._device_type = "cpu"
        if device is not None:
            self._device_type = device.type
        if mixed_precision is None or len(mixed_precision) == 0:
            self._mixed_dtype = None
        elif mixed_precision == "FP16":
            self._mixed_dtype = torch.float16
        elif mixed_precision == "BF16":
            self._mixed_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"mixed_precision should be FP16 or BF16, but got [{mixed_precision}]"
            )

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
        with torch.amp.autocast(
            device_type=self._device_type,
            dtype=self._mixed_dtype,
            enabled=self._mixed_dtype is not None,
        ):
            predictions = self.model.predict(batch)
            losses = self.model.loss(predictions, batch)
            total_loss = torch.stack(list(losses.values())).sum()

        losses = {k: v.detach() for k, v in losses.items()}
        predictions = {k: v.detach() for k, v in predictions.items()}
        return total_loss, (losses, predictions, batch)


class PredictWrapper(BaseModule):
    """Model predict wrapper for pipeline."""

    def __init__(
        self,
        module: nn.Module,
        device: Optional[torch.device] = None,
        mixed_precision: Optional[str] = None,
        output_cols: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = module
        self._device = device
        self._device_type = "cpu"
        if device is not None:
            self._device_type = device.type
        if mixed_precision is None or len(mixed_precision) == 0:
            self._mixed_dtype = None
        elif mixed_precision == "FP16":
            self._mixed_dtype = torch.float16
        elif mixed_precision == "BF16":
            self._mixed_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"mixed_precision should be FP16 or BF16, but got [{mixed_precision}]"
            )
        self._output_cols = output_cols

    def forward(
        self, batch: Batch
    ) -> Tuple[None, Tuple[Dict[str, torch.Tensor], Batch]]:
        """Predict.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
        """
        with torch.amp.autocast(
            device_type=self._device_type,
            dtype=self._mixed_dtype,
            enabled=self._mixed_dtype is not None,
        ):
            predictions = self.model.predict(batch)
            if self._output_cols is not None:
                result = dict()
                for c in self._output_cols:
                    result[c] = predictions[c].to("cpu", non_blocking=True)
                if TRAGET_REPEAT_INTERLEAVE_KEY in predictions:
                    result[TRAGET_REPEAT_INTERLEAVE_KEY] = predictions[
                        TRAGET_REPEAT_INTERLEAVE_KEY
                    ].to("cpu", non_blocking=True)
            else:
                result = {
                    k: v.to("cpu", non_blocking=True) for k, v in predictions.items()
                }
        return None, (result, batch)


class ScriptWrapper(BaseModule):
    """Model inference wrapper for jit.script."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.model = module
        self._features = self.model._features
        self._data_parser = DataParser(
            self._features,
            sampler_type=str(module.sampler_type)
            if hasattr(module, "sampler_type")
            else None,
        )

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


class CudaExportWrapper(ScriptWrapper):
    """Model inference wrapper for cuda export(aot/trt)."""

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
