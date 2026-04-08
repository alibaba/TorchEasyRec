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

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTowerWoEG
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.gr.positional_encoder import HSTUPositionalEncoder
from tzrec.modules.gr.postprocessors import (
    OutputPostprocessor,
    create_output_postprocessor,
)
from tzrec.modules.gr.preprocessors import (
    InputPreprocessor,
    create_input_preprocessor,
)
from tzrec.modules.gr.stu import STU, STULayer, STUStack
from tzrec.modules.norm import LayerNorm
from tzrec.modules.utils import init_linear_xavier_weights_zero_bias
from tzrec.ops.utils import set_static_max_seq_lens
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import match_model_pb2
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.fx_util import fx_int_item

torch.fx.wrap(fx_int_item)


@torch.fx.wrap
def _jagged_candidate_sim(
    user_emb: torch.Tensor, item_emb: torch.Tensor
) -> torch.Tensor:
    """Compute per-user similarity for JAGGED_SEQUENCE candidates.

    Each user has the same number of candidates (1 pos + num_neg). The item
    embeddings are organized as: [pos_1, neg_1_1, ..., neg_1_k, pos_2, ...].

    Args:
        user_emb: (B, D) user embeddings.
        item_emb: (B * (1 + num_neg), D) candidate embeddings.

    Returns:
        similarity (B, 1 + num_neg), first column is positive.
    """
    batch_size = user_emb.size(0)
    num_cand = item_emb.size(0) // batch_size
    item_emb = item_emb.view(batch_size, num_cand, -1)
    return torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)


class HSTUMatchUserTower(MatchTowerWoEG):
    """HSTU Match model user tower using modern STU module.

    Processes UIH (User Interaction History) sequences through UIHPreprocessor,
    HSTUPositionalEncoder, and STUStack to produce user embeddings. During training,
    returns one embedding per UIH position (autoregressive). During inference, returns
    the last position embedding per user for ANN retrieval.

    Args:
        tower_config (HSTUMatchTower): user tower config with HSTU settings.
        output_dim (int): user output embedding dimension (stu.embedding_dim).
        similarity (Similarity): similarity method config.
        feature_group (FeatureGroupConfig): uih feature group config.
        feature_group_dims (list): per-feature embedding dims in the group.
        features (list): list of features.
        model_config (ModelConfig): full model config.
        contextual_feature_dim (int): dimension of each contextual feature.
        max_contextual_seq_len (int): number of contextual features.
        contextual_group_name (str): contextual group name in grouped features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.HSTUMatchTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        feature_group_dims: List[int],
        features: List[BaseFeature],
        model_config: ModelConfig,
        contextual_feature_dim: int = 0,
        max_contextual_seq_len: int = 0,
        contextual_group_name: str = "contextual",
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        self._pass_grouped_features = True
        hstu_cfg = tower_config.hstu
        uih_dim = sum(feature_group_dims)
        stu_dim = hstu_cfg.stu.embedding_dim

        # Preprocessor: projects UIH, handles optional contextual/actions
        self._input_preprocessor: InputPreprocessor = create_input_preprocessor(
            hstu_cfg.input_preprocessor,
            uih_embedding_dim=uih_dim,
            output_embedding_dim=stu_dim,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=contextual_group_name,
        )

        # Positional encoder
        pos_kwargs = config_to_kwargs(hstu_cfg.positional_encoder)
        self._positional_encoder: HSTUPositionalEncoder = HSTUPositionalEncoder(
            embedding_dim=stu_dim,
            contextual_seq_len=self._input_preprocessor.contextual_seq_len(),
            **pos_kwargs,
        )

        # STU stack
        stu_kwargs = config_to_kwargs(hstu_cfg.stu)
        self._stu_module: STU = STUStack(
            stu_list=[STULayer(**stu_kwargs) for _ in range(hstu_cfg.attn_num_layers)],
        )

        # Output postprocessor (L2 norm or layer norm)
        self._output_postprocessor: OutputPostprocessor = create_output_postprocessor(
            hstu_cfg.output_postprocessor,
            embedding_dim=stu_dim,
        )

        self._input_dropout_ratio: float = hstu_cfg.input_dropout_ratio

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the user tower.

        Args:
            grouped_features: dictionary of embedded features from EmbeddingGroup.

        Returns:
            user embeddings of shape (B, D), last position embedding per user.
        """
        # 1. Preprocess: project UIH + optional contextual/actions
        (
            max_seq_len,
            total_uih_len,
            _,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
        ) = self._input_preprocessor(grouped_features)

        # 2. Positional encoding
        seq_embeddings = self._positional_encoder(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
        )

        # 3. Input dropout + STU
        seq_embeddings = F.dropout(
            seq_embeddings, p=self._input_dropout_ratio, training=self.training
        )
        seq_embeddings = self._stu_module(
            x=seq_embeddings,
            x_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )

        # 4. Output postprocessor
        user_emb = self._output_postprocessor(
            seq_embeddings=seq_embeddings,
            seq_timestamps=seq_timestamps,
        )

        # Extract last position embedding per user → (B, D)
        # Assumes all sequences are non-empty (guaranteed by EmbeddingGroup).
        user_emb = user_emb[seq_offsets[1:] - 1]

        return user_emb


class HSTUMatchItemTower(MatchTowerWoEG):
    """HSTU Match model item tower.

    Projects candidate embeddings to STU embedding dimension. Applies L2
    normalization only when similarity method is COSINE.

    Args:
        tower_config (HSTUMatchTower): tower config.
        output_dim (int): item output embedding dimension (stu.embedding_dim).
        similarity (Similarity): similarity method config.
        feature_group (FeatureGroupConfig): candidate feature group config.
        feature_group_dims (list): per-feature embedding dims in the group.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.HSTUMatchTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        feature_group_dims: List[int],
        features: List[BaseFeature],
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        # Override _group_name: parent sets it from tower_config.input ("uih"),
        # but item tower needs to read from the candidate feature group.
        self._group_name = feature_group.group_name
        self._pass_grouped_features = True
        cand_dim = sum(feature_group_dims)
        self._item_projection: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(cand_dim, output_dim),
            LayerNorm(output_dim),
        ).apply(init_linear_xavier_weights_zero_bias)

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the item tower.

        Args:
            grouped_features: dictionary of embedded features from EmbeddingGroup.

        Returns:
            item embeddings of shape (sum_candidates, D).
        """
        cand_emb = grouped_features[f"{self._group_name}.sequence"]
        item_emb = self._item_projection(cand_emb)
        if self._similarity == simi_pb2.Similarity.COSINE:
            item_emb = F.normalize(item_emb, p=2.0, dim=-1, eps=1e-6)
        return item_emb


class HSTUMatch(MatchModel):
    """HSTU Match model for two-tower retrieval.

    Uses modern STUStack for user sequence encoding with native jagged sequences.
    User tower processes UIH through UIHPreprocessor + STU. Item tower
    projects and normalizes candidate embeddings. Similarity via dot product.

    Feature groups:
        - "uih" (JAGGED_SEQUENCE): user interaction history
        - "candidate" (JAGGED_SEQUENCE): candidate items (pos + neg)
        - "contextual" (optional, DEEP/SEQUENCE): user contextual features

    Training: autoregressive — each UIH position produces a user embedding,
    compared against candidates via dot product similarity.
    Inference: last UIH position → user embedding for ANN retrieval.

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

        tower_cfg = self._model_config.hstu_tower
        set_static_max_seq_lens([tower_cfg.max_seq_len])

        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )

        stu_dim = tower_cfg.hstu.stu.embedding_dim

        # Resolve feature groups
        name_to_fg = {x.group_name: x for x in model_config.feature_groups}
        uih_fg = name_to_fg[tower_cfg.input]
        cand_fg = name_to_fg.get("candidate")
        assert cand_fg is not None, "HSTUMatch requires a 'candidate' feature group."

        uih_features = self.get_features_in_feature_groups([uih_fg])
        cand_features = self.get_features_in_feature_groups([cand_fg])

        uih_dims = self.embedding_group.group_dims(tower_cfg.input + ".sequence")
        cand_dims = self.embedding_group.group_dims("candidate.sequence")

        # Optional contextual features
        contextual_feature_dim = 0
        max_contextual_seq_len = 0
        contextual_group_name = "contextual"
        if "contextual" in name_to_fg:
            ctx_group_type = self.embedding_group.group_type("contextual")
            if ctx_group_type == model_pb2.SEQUENCE:
                contextual_group_name = "contextual.query"
            elif ctx_group_type == model_pb2.DEEP:
                contextual_group_name = "contextual"
            ctx_dims = self.embedding_group.group_dims(contextual_group_name)
            if len(set(ctx_dims)) > 1:
                raise ValueError(
                    "output_dim of features in contextual features_group "
                    f"must be same, but now {set(ctx_dims)}."
                )
            contextual_feature_dim = ctx_dims[0]
            max_contextual_seq_len = len(ctx_dims)

        self.user_tower = HSTUMatchUserTower(
            tower_config=tower_cfg,
            output_dim=stu_dim,
            similarity=self._model_config.similarity,
            feature_group=uih_fg,
            feature_group_dims=uih_dims,
            features=uih_features,
            model_config=model_config,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=max_contextual_seq_len,
            contextual_group_name=contextual_group_name,
        )

        self.item_tower = HSTUMatchItemTower(
            tower_config=tower_cfg,
            output_dim=stu_dim,
            similarity=self._model_config.similarity,
            feature_group=cand_fg,
            feature_group_dims=cand_dims,
            features=cand_features,
        )

        self._temperature = self._model_config.temperature

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result with 'similarity' key.
        """
        grouped_features = self.embedding_group(batch)

        user_emb = self.user_tower(grouped_features)
        item_emb = self.item_tower(grouped_features)

        ui_sim = _jagged_candidate_sim(user_emb, item_emb) / self._temperature
        return {"similarity": ui_sim}
