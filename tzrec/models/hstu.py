# Copyright (c) 2024-2025, Alibaba Group;
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
import torch.nn.functional as F
from torch import nn

from tzrec.datasets.utils import (
    BASE_DATA_GROUP,
    CAND_POS_LENGTHS,
    HARD_NEG_INDICES,
    Batch,
)
from tzrec.features.feature import (
    BaseFeature,
    create_features,
    project_grouped_sequence_feature_to_scalar,
)
from tzrec.models.match_model import MatchModel, MatchTowerWoEG
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.gr.hstu_transducer import HSTUMatchEncoder
from tzrec.modules.mlp import MLP
from tzrec.ops.utils import set_static_max_seq_lens
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import match_model_pb2
from tzrec.utils.config_util import config_to_kwargs


class HSTUUserTower(MatchTowerWoEG):
    """HSTU user tower (reusable beyond match — produces a user embedding).

    Holds an `HSTUMatchEncoder` (shared HSTU encode pipeline + last-position
    extraction) and forwards `grouped_features` through. Returns the
    last-position embedding per user — consumed by `MatchModel.sim` for
    training scoring and ANN retrieval at inference time.

    Args:
        tower_config (HSTUUserTower): user tower config with HSTU settings.
        output_dim (int): user output embedding dimension.
        similarity (Similarity): similarity method config.
        feature_groups (list): feature group configs the tower consumes. The
            primary uih group (named by `tower_config.input`) is required;
            an optional `contextual` group is detected by name; auxiliary
            `uih_action` / `uih_watchtime` / `uih_timestamp` groups (read at
            forward time by the input preprocessor + positional encoder)
            should also be passed when configured so that
            `TowerWoEGWrapper` can rebuild the EmbeddingGroup with all
            consumed keys at export time.
        features (list): list of features for every group in `feature_groups`.
        embedding_group (EmbeddingGroup): shared embedding group used to
            resolve per-group dims at construction time (not stored).
    """

    def __init__(
        self,
        tower_config: tower_pb2.HSTUUserTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        feature_groups: List[model_pb2.FeatureGroupConfig],
        features: List[BaseFeature],
        embedding_group: EmbeddingGroup,
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_groups, features)

        contextual_feature_group = next(
            (
                feature_group
                for feature_group in feature_groups
                if feature_group.group_name == "contextual"
            ),
            None,
        )
        contextual_feature_dim = 0
        max_contextual_seq_len = 0
        contextual_group_name = "contextual"
        if contextual_feature_group is not None:
            if contextual_feature_group.group_type == model_pb2.SEQUENCE:
                contextual_group_name = "contextual.query"
            contextual_dims = embedding_group.group_dims(contextual_group_name)
            if len(set(contextual_dims)) > 1:
                raise ValueError(
                    "output_dim of features in contextual feature_group "
                    f"must be same, but now {set(contextual_dims)}."
                )
            contextual_feature_dim = contextual_dims[0]
            max_contextual_seq_len = len(contextual_dims)

        self._hstu_encoder: HSTUMatchEncoder = HSTUMatchEncoder(
            uih_embedding_dim=embedding_group.group_total_dim(
                f"{tower_config.input}.sequence"
            ),
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=contextual_group_name,
            scaling_seqlen=tower_config.max_seq_len,
            is_inference=False,
            **config_to_kwargs(tower_config.hstu),
        )
        if self._output_dim > 0:
            self.output = nn.Linear(tower_config.hstu.stu.embedding_dim, output_dim)

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the user tower.

        Args:
            grouped_features: dictionary of embedded features from EmbeddingGroup.

        Returns:
            user embeddings of shape (B, D), last-position embedding per user.
        """
        user_emb = self._hstu_encoder(grouped_features)
        if self._output_dim > 0:
            user_emb = self.output(user_emb)
        if self._similarity == simi_pb2.Similarity.COSINE:
            user_emb = F.normalize(user_emb, p=2.0, dim=-1, eps=1e-6)
        return user_emb


class HSTUMatchItemTower(MatchTowerWoEG):
    """HSTU Match model item tower.

    Optional MLP projection of the candidate embedding. Applies L2
    normalization only when similarity method is COSINE.

    Args:
        tower_config (Tower): item tower config (reuses DSSM's `Tower`
            proto); `tower_config.mlp` is optional — when unset no projection
            is applied (caller must size candidate embedding_dim to match
            output_dim).
        output_dim (int): item output embedding dimension.
        similarity (Similarity): similarity method config.
        feature_groups (list): single-element list `[candidate_feature_group]`.
        features (list): candidate features.
        embedding_group (EmbeddingGroup): shared embedding group used to
            resolve candidate dims at construction time (not stored).
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        feature_groups: List[model_pb2.FeatureGroupConfig],
        features: List[BaseFeature],
        embedding_group: EmbeddingGroup,
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_groups, features)
        # Item tower's primary group is candidate, not uih (the latter is what
        # tower_config.input names on the user-tower proto). Use the item-side
        # tower_config.input here, which equals feature_groups[0].group_name.
        self._group_name = tower_config.input
        # Training view: candidate group is JAGGED_SEQUENCE; embedding_group
        # emits the per-row jagged tensor at `{group_name}.sequence`. The
        # export-time scalar view (see `set_is_inference(True)`) swaps this
        # to `{group_name}.query` -- one row per item, no positive-set
        # grouping container.
        self._cand_key = f"{self._group_name}.sequence"
        candidate_dims = embedding_group.group_dims(self._cand_key)
        candidate_total_dim = sum(candidate_dims)
        if tower_config.HasField("mlp"):
            self.mlp: torch.nn.Module = MLP(
                in_features=candidate_total_dim,
                **config_to_kwargs(tower_config.mlp),
            )
            mlp_out_dim = self.mlp.output_dim()
        else:
            # No MLP: candidate embedding flows straight to the output linear.
            self.mlp: torch.nn.Module = torch.nn.Identity()
            mlp_out_dim = candidate_total_dim
        if self._output_dim > 0:
            self.output = nn.Linear(mlp_out_dim, output_dim)

    def set_is_inference(self, is_inference: bool) -> None:
        """Switch the tower into the scalar item-export view (one-way).

        Training stays at `candidate.sequence` (jagged sub-feature). Export
        copies the tower (`copy.deepcopy`), calls this with `True`, and the
        wrapper picks up the swapped `_features` / `_feature_groups` /
        `_cand_key` for serving. Idempotent: re-entering after the swap is a
        no-op. `set_is_inference(False)` does NOT rebuild the training view;
        callers must export from a copy so the original training tower is
        preserved.

        Note: `MatchTowerWoEG` derives from `nn.Module`, not `BaseModule`,
        so this method doesn't call `super().set_is_inference()`. The
        `_is_inference` attribute is propagated to sub-modules separately by
        `ScriptWrapper.set_is_inference()` via `recursive_setattr` during
        export.
        """
        if not is_inference:
            return
        if self._cand_key.endswith(".query"):
            return  # already swapped (idempotent)

        # Project each grouped sequence sub-feature into a scalar export
        # FeatureConfig, then rebuild as top-level scalar features.
        scalar_configs = [
            project_grouped_sequence_feature_to_scalar(f) for f in self._features
        ]
        source = self._features[0]
        scalar_features = create_features(
            scalar_configs,
            fg_mode=source.fg_mode,
            neg_fields=None,
            fg_encoded_multival_sep=source._fg_encoded_multival_sep,
            force_base_data_group=any(
                f.data_group == BASE_DATA_GROUP for f in self._features
            ),
        )
        self._features = scalar_features
        self._feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name=self._group_name,
                feature_names=[f.name for f in scalar_features],
                group_type=model_pb2.JAGGED_SEQUENCE,
            )
        ]
        self._cand_key = f"{self._group_name}.query"

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the item tower.

        Args:
            grouped_features: dictionary of embedded features from EmbeddingGroup.

        Returns:
            item embeddings of shape (sum_candidates, D).
        """
        cand_emb = grouped_features[self._cand_key]
        item_emb = self.mlp(cand_emb)
        if self._output_dim > 0:
            item_emb = self.output(item_emb)
        if self._similarity == simi_pb2.Similarity.COSINE:
            item_emb = F.normalize(item_emb, p=2.0, dim=-1, eps=1e-6)
        return item_emb


class HSTUMatch(MatchModel):
    """HSTU Match model for two-tower retrieval.

    Uses modern STUStack for user sequence encoding with native jagged sequences.
    User tower processes UIH through UIHPreprocessor + STU. Item tower
    projects and normalizes candidate embeddings. Similarity via dot product.

    Feature groups (model_config.feature_groups):
        - `user_tower.input` (JAGGED_SEQUENCE, required): user interaction
          history. Conventionally named "uih".
        - `item_tower.input` (JAGGED_SEQUENCE, required): candidate items
          (positives + appended negs after the sampler's block-(B-1) suffix
          combine). Conventionally named "candidate".
        - "contextual" (optional, DEEP/SEQUENCE): user contextual features.
        - "uih_action" / "uih_watchtime" / "uih_timestamp" (optional,
          JAGGED_SEQUENCE): auxiliary uih sub-features consumed by
          UIHPreprocessor's action_encoder and the HSTU positional
          encoder's time bias. Required when `uih_preprocessor.action_encoder`
          is configured.

    User tower returns the last-position UIH embedding per user; it is compared
    against candidate embeddings via the configured similarity at both train and
    inference time.

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
        assert model_config.WhichOneof("model") == "hstu_match", (
            "invalid model config: %s" % model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, match_model_pb2.HSTUMatch)
        assert not self._in_batch_negative, (
            "HSTUMatch does not support in_batch_negative — multi-positive rows "
            "(Q queries from B users) make the B×B in-batch path ill-defined. "
            "Use a NegativeSampler/HardNegativeSampler instead."
        )

        user_tower_cfg = self._model_config.user_tower
        item_tower_cfg = self._model_config.item_tower
        set_static_max_seq_lens([user_tower_cfg.max_seq_len])

        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )

        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}
        candidate_feature_group = name_to_feature_group[item_tower_cfg.input]
        # User tower consumes every feature group except the candidate. That
        # includes the primary uih group, optional contextual, and any
        # auxiliary uih_* groups (uih_action / uih_watchtime / uih_timestamp)
        # the input preprocessor / positional encoder reads at forward time.
        # TowerWoEGWrapper rebuilds the EmbeddingGroup from the tower's
        # `_feature_groups` + `_features` at export, so every group whose
        # `{group_name}.sequence` key is read must be declared here.
        user_feature_groups = [
            feature_group
            for feature_group in model_config.feature_groups
            if feature_group.group_name != item_tower_cfg.input
        ]
        user_features = self.get_features_in_feature_groups(user_feature_groups)
        candidate_features = self.get_features_in_feature_groups(
            [candidate_feature_group]
        )

        self.user_tower = HSTUUserTower(
            tower_config=user_tower_cfg,
            output_dim=self._model_config.output_dim,
            similarity=self._model_config.similarity,
            feature_groups=user_feature_groups,
            features=user_features,
            embedding_group=self.embedding_group,
        )

        self.item_tower = HSTUMatchItemTower(
            tower_config=item_tower_cfg,
            output_dim=self._model_config.output_dim,
            similarity=self._model_config.similarity,
            feature_groups=[candidate_feature_group],
            features=candidate_features,
            embedding_group=self.embedding_group,
        )

        self._temperature = self._model_config.temperature

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Candidate sequence carries `[pos_i] * (B-1) + [pos_{B-1}, simple, hard]`,
        so after the item tower runs once over all values, the resulting
        `item_emb` is laid out as `[pos(Q), simple(M), hard(H)]` — exactly the
        layout `MatchModel.sim` / `_sim_with_sampler` expects with `query_emb`
        of shape `(Q, D)` substituted for the usual `(B, D)` user embedding.
        """
        grouped_features = self.embedding_group(batch)

        user_emb = self.user_tower(grouped_features)
        item_emb = self.item_tower(grouped_features)

        pos_lengths_t = batch.additional_infos.get(CAND_POS_LENGTHS, None)
        assert pos_lengths_t is not None, (
            "HSTUMatch requires a sampler that produces per-row positive "
            "lengths (CAND_POS_LENGTHS). Check that negative_sampler is "
            "configured with attr_fields matching the candidate "
            "sequence_id_feature's name."
        )
        pos_lengths = pos_lengths_t.to(item_emb.device, dtype=torch.long)
        # Repeat each user embedding K_i times so query embeddings are aligned
        # with their positives in the candidate sequence.
        query_emb = torch.repeat_interleave(user_emb, pos_lengths, dim=0)

        hard_neg_indices = batch.additional_infos.get(HARD_NEG_INDICES, None)
        ui_sim = self.sim(query_emb, item_emb, hard_neg_indices) / self._temperature
        return {"similarity": ui_sim}

    def _get_label(self, batch: Batch) -> torch.Tensor:
        """Read the candidate-side jagged label."""
        return batch.jagged_labels[self._label_name].values().to(torch.int64)

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss; reads the jagged label rather than a scalar bare label."""
        losses = {}
        label = self._get_label(batch)
        for loss_cfg in self._base_model_config.losses:
            losses.update(self._loss_impl(predictions, batch, label, loss_cfg))
        losses.update(self._loss_collection)
        return losses

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state; reads the jagged label."""
        label = self._get_label(batch)
        for metric_cfg in self._base_model_config.metrics:
            self._update_metric_impl(predictions, batch, label, metric_cfg)
        if losses is not None:
            for loss_cfg in self._base_model_config.losses:
                self._update_loss_metric_impl(losses, batch, label, loss_cfg)

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state; reads the jagged label."""
        label = self._get_label(batch)
        for metric_cfg in self._base_model_config.train_metrics:
            self._update_train_metric_impl(predictions, batch, label, metric_cfg)
