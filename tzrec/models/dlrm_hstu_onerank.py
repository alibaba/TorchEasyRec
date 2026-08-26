# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Optional

import torch
from torch.autograd.profiler import record_function

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.onerank_listwise_loss import OneRankListwiseLoss
from tzrec.models.dlrm_hstu import (
    DlrmHSTU,
    _fx_avg_batch_size,
    _fx_flip_tensor_dict,
)
from tzrec.models.rank_model import TARGET_REPEAT_INTERLEAVE_KEY, RankModel
from tzrec.modules.gr.onerank_head import OneRankPredictionHead
from tzrec.modules.gr.onerank_tokenizer import OneRankHSTUTransducer
from tzrec.ops.utils import set_static_max_seq_lens
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.config_util import config_to_kwargs

# `torch.fx.wrap` registers by name in the *calling* module's globals, so the
# decorators in dlrm_hstu.py do not cover the calls made from here.
torch.fx.wrap(_fx_flip_tensor_dict)
torch.fx.wrap(_fx_avg_batch_size)

# Loss types whose `_output_to_prediction_impl` branch publishes a
# `logits_<task>` entry, which is what the list-wise term scores over.
_LOGIT_LOSS_TYPES = ("binary_cross_entropy", "binary_focal_loss")


def _listwise_loss_name(task_name: str) -> str:
    return f"listwise_infonce_{task_name}"


class DlrmHSTUOneRank(DlrmHSTU):
    """OneRank-style multi-task DLRM HSTU model.

    Differs from :class:`DlrmHSTU` in three places:

    1. every candidate is expanded into ``2K`` tokens (a candidate replica
       and a task token per task) and attention runs under a task-private
       mask, so the trunk emits one representation ``r^i_k`` per
       ``(candidate, task)`` pair rather than a single per-candidate vector
       (paper 2.1 / 2.2, see
       :mod:`tzrec.modules.gr.onerank_tokenizer`);
    2. scoring is a per-task inner product instead of ``FusionMTLTower``'s
       shared MLP (paper 2.4, see :mod:`tzrec.modules.gr.onerank_head`);
    3. the candidate-side ``_item_embedding_mlp`` is gone -- the candidate
       features already enter through the HSTU input preprocessor, and the
       task token's own channel replaces the fused ``user * item`` product.

    Everything else -- labels, bitmask decoding, point-wise losses, metrics
    -- is inherited unchanged, so the two models are directly comparable on
    the same ``fusion_mtl_tower.task_configs``.

    Requires ``kernel: CUTLASS`` (or ``PYTORCH``) plus bf16/fp16 mixed
    precision: the task-private mask is expressed as an NFUNC func tensor,
    which the Triton attention kernel does not implement.

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
        # Call grandparent RankModel.__init__ directly to skip DlrmHSTU's
        # dlrm_hstu-specific model-type assertion.
        RankModel.__init__(
            self, model_config, features, labels, sample_weights, **kwargs
        )
        assert model_config.WhichOneof("model") == "dlrm_hstu_onerank", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.DlrmHSTUOneRank)
        self._init()

    def _init(self) -> None:
        super()._init()
        # The group inflation makes the real sequence longer than
        # `max_seq_len`, and `autotune_max_seq_len` falls back to the largest
        # configured bucket once the runtime length exceeds all of them --
        # silently autotuning every jagged kernel for a shorter sequence.
        set_static_max_seq_lens([self._inflated_max_seq_len()])

    def _num_tasks(self) -> int:
        """Number of task tokens ``K``, one per task tower."""
        return len(self._task_configs)

    def _inflated_max_seq_len(self) -> int:
        """``max_seq_len`` after the candidate-group expansion.

        Used as the attention-output scaling divisor.  Leaving it at the
        un-inflated ``max_seq_len`` would quietly change the normalization
        scale relative to the ``DlrmHSTU`` baseline and make attention
        magnitudes incomparable.
        """
        onerank = self._model_config.onerank
        return (
            self._model_config.max_seq_len
            + onerank.max_num_candidates * 2 * self._num_tasks()
        )

    def _build_transducer(
        self, contextual_feature_dim: int, max_contextual_seq_len: int
    ) -> torch.nn.Module:
        num_tasks = self._num_tasks()
        if num_tasks == 0:
            raise ValueError(
                "dlrm_hstu_onerank requires at least one "
                "fusion_mtl_tower.task_configs entry: the number of task "
                "tokens K is taken from it."
            )
        for task_cfg in self._task_configs:
            if task_cfg.num_class > 1:
                raise ValueError(
                    f"dlrm_hstu_onerank scores each task with a single inner "
                    f"product, so num_class > 1 is not supported; task "
                    f"'{task_cfg.task_name}' has num_class={task_cfg.num_class}."
                )
        if self._model_config.onerank.max_num_candidates == 0:
            raise ValueError("onerank.max_num_candidates must be > 0.")
        return OneRankHSTUTransducer(
            num_task_tokens=num_tasks,
            uih_embedding_dim=self.embedding_group.group_total_dim("uih"),
            target_embedding_dim=self.embedding_group.group_total_dim("candidate"),
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=self._contextual_group_name,
            scaling_seqlen=self._inflated_max_seq_len(),
            **config_to_kwargs(self._model_config.hstu),
            return_full_embeddings=False,
        )

    def _contextual_token_dim(self) -> int:
        """Flattened width of the contextual token ``s`` fed to the head.

        Equals ``max_contextual_seq_len * contextual_feature_dim`` for both
        branches of ``DlrmHSTU._init``, which is the layout of
        ``grouped_features[contextual_group_name]``.
        """
        return sum(self.embedding_group.group_dims(self._contextual_group_name))

    def _build_output_modules(self, stu_embedding_dim: int) -> None:
        onerank = self._model_config.onerank
        situation_discernment = None
        if onerank.HasField("situation_discernment"):
            situation_discernment = config_to_kwargs(onerank.situation_discernment)
        cross_task_head = None
        if onerank.HasField("cross_task_head"):
            cross_task_head = config_to_kwargs(onerank.cross_task_head)
        self._onerank_head: torch.nn.Module = OneRankPredictionHead(
            embedding_dim=stu_embedding_dim,
            task_names=[task_cfg.task_name for task_cfg in self._task_configs],
            contextual_feature_dim=self._contextual_token_dim(),
            situation_discernment=situation_discernment,
            cross_task_head=cross_task_head,
        )

    def init_loss(self) -> None:
        """Initialize loss modules.

        Adds the list-wise InfoNCE terms on top of the inherited per-task
        point-wise losses; the total is
        ``sum_k alpha_k * L_list_k + sum_k weight_k * L_point_k``.
        """
        super().init_loss()
        task_cfgs = {cfg.task_name: cfg for cfg in self._task_configs}
        seen = set()
        for listwise_cfg in self._model_config.onerank.listwise_losses:
            task_name = listwise_cfg.task_name
            if task_name in seen:
                raise ValueError(
                    f"onerank.listwise_losses has more than one entry for task "
                    f"'{task_name}'."
                )
            seen.add(task_name)
            task_cfg = task_cfgs.get(task_name)
            if task_cfg is None:
                raise ValueError(
                    f"onerank.listwise_losses references task '{task_name}', "
                    f"which is not in fusion_mtl_tower.task_configs "
                    f"({sorted(task_cfgs)})."
                )
            if not any(
                loss_cfg.WhichOneof("loss") in _LOGIT_LOSS_TYPES
                for loss_cfg in task_cfg.losses
            ):
                raise ValueError(
                    f"onerank.listwise_losses on task '{task_name}' needs that "
                    f"task to also carry one of {_LOGIT_LOSS_TYPES}: the "
                    f"list-wise term scores over the `logits_{task_name}` "
                    f"prediction those losses publish."
                )
            self._loss_modules[_listwise_loss_name(task_name)] = OneRankListwiseLoss(
                temperature_init=listwise_cfg.temperature_init,
                learnable_temperature=listwise_cfg.learnable_temperature,
            )

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = super().loss(predictions, batch)
        listwise_cfgs = self._model_config.onerank.listwise_losses
        if len(listwise_cfgs) == 0:
            return losses

        task_cfgs = {cfg.task_name: cfg for cfg in self._task_configs}
        # Un-flipped candidate counts, matching the request order of both
        # `logits_<task>` (flipped back in `predict`) and the jagged labels.
        lengths = predictions[TARGET_REPEAT_INTERLEAVE_KEY]
        loss_weight = None
        if self._model_config.enable_global_average_loss:
            # The term is a mean over this rank's requests; rescale by the
            # local/global request-count ratio so that DDP's cross-rank
            # gradient average comes out as a global mean on a ragged batch.
            loss_weight = lengths.size(0) / _fx_avg_batch_size(lengths)

        for listwise_cfg in listwise_cfgs:
            task_name = listwise_cfg.task_name
            loss_name = _listwise_loss_name(task_name)
            losses[loss_name] = (
                self._loss_modules[loss_name](
                    predictions[f"logits_{task_name}"],
                    self._get_label(batch, task_cfgs[task_name]),
                    lengths,
                    loss_weight,
                )
                * listwise_cfg.alpha
            )
        return losses

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        with record_function("## preprocess ##"):
            grouped_features = self.build_input(batch)

        # Capture num_targets before the descending-timestamp flip below, so the
        # output split key stays in the original (un-flipped) request order.
        num_targets = grouped_features["candidate.sequence_length"]

        if not self._model_config.sequence_timestamp_is_ascending:
            # if timestamp of sequence is descending,
            # we should reverse all features
            grouped_features = _fx_flip_tensor_dict(grouped_features)

        with record_function("## user_forward ##"):
            task_embeddings, _ = self._hstu_transducer(grouped_features)

        with record_function("## onerank_head ##"):
            # Post-flip lengths and contextual rows: `task_embeddings` is
            # jagged in the same (possibly reversed) request order the
            # transducer just consumed.
            scores = self._onerank_head(
                task_embeddings=task_embeddings,
                num_candidates=grouped_features["candidate.sequence_length"],
                contextual_embeddings=grouped_features[self._contextual_group_name],
            )

        mt_preds: Dict[str, torch.Tensor] = {}
        for i, task_cfg in enumerate(self._task_configs):
            # Keep the trailing class dim so `_output_to_prediction_impl`
            # sees the same `(N, 1)` shape FusionMTLTower produces.
            mt_preds[task_cfg.task_name] = scores[:, i : i + 1]

        if not self._model_config.sequence_timestamp_is_ascending:
            # if timestamp of sequence is descending,
            # we should reverse predictions back to input order
            mt_preds = _fx_flip_tensor_dict(mt_preds)

        predictions = {}
        for task_cfg in self._task_configs:
            task_name = task_cfg.task_name
            for loss_cfg in task_cfg.losses:
                predictions.update(
                    self._output_to_prediction_impl(
                        mt_preds[task_name],
                        loss_cfg,
                        num_class=task_cfg.num_class,
                        suffix=f"_{task_name}",
                    )
                )
        predictions[TARGET_REPEAT_INTERLEAVE_KEY] = num_targets

        return predictions
