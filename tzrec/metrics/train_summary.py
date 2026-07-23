# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.protos.summary_pb2 import ModelSummaries

BASE_DATA_GROUP = "__BASE__"


def resolve_pred_tensor(
    predictions: Dict[str, torch.Tensor], pred_name: str
) -> torch.Tensor:
    if pred_name not in predictions:
        raise ValueError(
            f'summaries pred_name "{pred_name}" not found in prediction_dict keys: '
            f"{sorted(predictions.keys())}"
        )
    pred = predictions[pred_name].float()
    if pred_name == "logits" or pred_name.startswith("logits"):
        pred = torch.sigmoid(pred)
    if pred.dim() > 1:
        if pred.dim() == 2 and pred.size(-1) == 2:
            pred = pred[:, 1]
        else:
            pred = pred.reshape(-1)
    return pred.reshape(-1)


def _feature_values(batch: Any, feature_name: str) -> torch.Tensor:
    if BASE_DATA_GROUP in batch.sparse_features:
        sparse = batch.sparse_features[BASE_DATA_GROUP].to_dict()
        if feature_name in sparse:
            return sparse[feature_name].to_padded_dense(1)[:, 0]
    if BASE_DATA_GROUP in batch.dense_features:
        dense = batch.dense_features[BASE_DATA_GROUP].to_dict()
        if feature_name in dense:
            return dense[feature_name].values().reshape(-1)
    raise ValueError(
        f'summaries feature_name "{feature_name}" not found in batch features'
    )


def build_feature_mask(
    batch: Any, feature_name: str, feature_value: str
) -> torch.Tensor:
    feat = _feature_values(batch, feature_name)
    try:
        target = float(feature_value)
        if feat.is_floating_point() or feat.dtype in (torch.int32, torch.int64):
            return feat.float() == target
    except (TypeError, ValueError):
        pass
    # string / id features: compare as int when possible
    try:
        return feat.long() == int(float(feature_value))
    except (TypeError, ValueError):
        return feat == feat.new_tensor(feature_value)


class _RunningMeanState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_sum", torch.zeros(1), persistent=False)
        self.register_buffer("_count", torch.zeros(1), persistent=False)

    def update(self, values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        values = values.reshape(-1).float()
        if mask is not None:
            values = values[mask.reshape(-1)]
        if values.numel() == 0:
            return
        self._sum += values.sum()
        self._count += values.numel()

    def compute_mean(self) -> Optional[torch.Tensor]:
        if self._count.item() == 0:
            return None
        return self._sum / self._count

    def reset(self) -> None:
        self._sum.zero_()
        self._count.zero_()


class TrainSummaryModule(nn.Module):
    """Accumulate training summaries between log steps."""

    def __init__(self, summary_cfg: ModelSummaries) -> None:
        super().__init__()
        self._summary_cfg = summary_cfg
        self._state = _RunningMeanState()
        self._label_state = _RunningMeanState()

    @property
    def summary_cfg(self) -> ModelSummaries:
        """Configured summary proto for this module."""
        return self._summary_cfg

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Any,
        label_name: str,
    ) -> None:
        summary_type = self._summary_cfg.WhichOneof("summary")
        if summary_type == "pcoc":
            cfg = self._summary_cfg.pcoc
            pred_name = cfg.pred_name or "probs"
            preds = resolve_pred_tensor(predictions, pred_name)
            label = batch.labels[label_name].float().reshape(-1)
            self._state.update(preds)
            self._label_state.update(label)
        elif summary_type == "scalars":
            cfg = self._summary_cfg.scalars
            pred_name = cfg.pred_name or "probs"
            preds = resolve_pred_tensor(predictions, pred_name)
            if cfg.HasField("feature_name") and cfg.HasField("feature_value"):
                mask = build_feature_mask(batch, cfg.feature_name, cfg.feature_value)
                self._state.update(preds, mask)
            else:
                self._state.update(preds)
        else:
            raise ValueError(f"unsupported summary type: {summary_type}")

    def compute(self) -> Dict[str, torch.Tensor]:
        summary_type = self._summary_cfg.WhichOneof("summary")
        result: Dict[str, torch.Tensor] = {}
        if summary_type == "pcoc":
            mean_pred = self._state.compute_mean()
            mean_label = self._label_state.compute_mean()
            if mean_pred is None or mean_label is None:
                return result
            epsilon = self._summary_cfg.pcoc.epsilon
            result["summary/predicted_ctr"] = mean_pred
            result["summary/observed_ctr"] = mean_label
            result["summary/pcoc"] = mean_pred / (mean_label + epsilon)
        elif summary_type == "scalars":
            mean_pred = self._state.compute_mean()
            if mean_pred is not None:
                name = self._summary_cfg.scalars.name
                result[f"summary/{name}"] = mean_pred
        self.reset()
        return result

    def reset(self) -> None:
        self._state.reset()
        self._label_state.reset()


def build_train_summary_modules(
    summaries_set: List[ModelSummaries],
) -> nn.ModuleList:
    return nn.ModuleList([TrainSummaryModule(cfg) for cfg in summaries_set])
