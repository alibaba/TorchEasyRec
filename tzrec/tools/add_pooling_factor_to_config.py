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

"""Fill each embedding feature's pooling_factor and unique_id_factor from data.

Reads batches of the pipeline config's train input through the training
dataloader (FG included, no GPU) and writes ``embedding_constraints`` on every
measured feature of a copy of the config:

  pooling_factor    average number of raw id positions one sample looks up in
                    this feature, duplicates included, at least 1: total raw
                    ids over all samples read, divided by the actual sample
                    count, partial batches included. The planner keeps one
                    entry per feature of a table, for input, output and
                    communication estimates.
  unique_id_factor  maximum across global batches of the table's joint distinct
                    ID count divided by the actual batch sample count. All
                    samples and sharing features participate in the union.
                    Each member receives the same table-level factor.

Both values are measured in one pass. A feature missing from the sampled
batches is left unchanged. A shared table's union is only measured when
every member feature appears in the batch - a missing member is not an
empty one, and no partial union is written as the table's measurement.

Tables are resolved from the pipeline configuration's feature groups, the
same resolution model construction uses; no model is built. Any model that
follows the standard feature_groups semantics is supported without importing
its class; a custom model that decides table sharing in its own constructor
is outside this mapping. Managed-collision (ZCH) tables are resolved as
tables; the deduplicated count is over raw ids and does not measure ZCH's
post-collision embedding-row working set.

Usage: python -m tzrec.tools.add_pooling_factor_to_config
    --pipeline_config_path pipeline.config --output_path pipeline_measured.config
    --batch_size 4096 [--dataset_type OdpsDataset] [--input_path odps://...]
    [--num_batches 100]
"""

import argparse
import copy
import math
import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from tzrec.datasets.dataset import create_dataloader
from tzrec.features.feature import BaseFeature
from tzrec.main import _create_features
from tzrec.modules.embedding import (
    build_embedding_bag_configs,
    build_embedding_collection_configs,
    build_embedding_group_mappings,
)
from tzrec.protos import model_pb2
from tzrec.protos.data_pb2 import DatasetType
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


class AddPoolingFactorToConfig(object):
    """Measure and fill pooling_factor and unique_id_factor for embeddings.

    Args:
        pipeline_config_path (str): pipeline config path; its train input is
            read and its model config decides the physical tables.
        output_path (str): path to write the filled config to.
        num_batches (int): number of global batches to sample.
        dataset_type (str, optional): dataset name used only for measurement.
        input_path (str, optional): input path used only for measurement.
        batch_size (int, optional): measurement batch size, preferably the target
            global batch size. Unset overrides use the pipeline config values.
    """

    def __init__(
        self,
        pipeline_config_path: str,
        output_path: str,
        num_batches: int,
        dataset_type: Optional[str] = None,
        input_path: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self._dataset_type = (
            DatasetType.Value(dataset_type) if dataset_type is not None else None
        )
        self._input_path = input_path
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size
        if num_batches < 1:
            raise ValueError(f"num_batches must be at least 1, got {num_batches}")
        if os.path.realpath(output_path) == os.path.realpath(pipeline_config_path):
            raise ValueError(
                "output_path must differ from pipeline_config_path; the tool "
                "writes a new config and never modifies its input"
            )
        self._pipeline_config_path = pipeline_config_path
        self._output_path = output_path
        self._num_batches = num_batches

    def run(self) -> None:
        """Measure the train input and write the filled config."""
        cfg = config_util.load_pipeline_config(self._pipeline_config_path)
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        tables = self.get_embedding_table_features(cfg, features)
        factors = self.measure(cfg, features, tables)
        filled = self.fill(features, factors)
        config_util.save_message(cfg, self._output_path)
        measured_pooling = sum(1 for p, _ in factors.values() if p)
        measured_unique = sum(1 for _, u in factors.values() if u)
        logger.info(
            f"wrote pooling_factor for {measured_pooling} features and "
            f"unique_id_factor for {measured_unique} to {self._output_path}, "
            f"{len(filled)} features filled"
        )

    @staticmethod
    def get_embedding_table_features(
        cfg: EasyRecConfig, features: List[BaseFeature]
    ) -> List[Tuple[str, List[str]]]:
        """(collection-qualified table, feature names) of every embedding table.

        Calls the configuration rules model construction uses - group
        classification, table naming, suffixes, ZCH separation, feature
        sharing - and extracts each table's identity and input features from
        the built configs. The feature groups are copied first: the mapping
        helpers may supplement a nested sequence group's name, and the config
        the tool saves must not carry that.

        Args:
            cfg (EasyRecConfig): the pipeline config.
            features (list): the config's features.

        Returns:
            (table, feature names) pairs, the table identified by data group,
            collection, embedding dim and name; feature names as they appear
            in a batch (grouped sequence features carry their sequence
            prefix).
        """
        model_oneof = cfg.model_config.WhichOneof("model")
        model_cfg = (
            getattr(cfg.model_config, model_oneof) if model_oneof is not None else None
        )
        if isinstance(model_cfg, str):
            model_cfg = None
        feature_groups = []
        for group in cfg.model_config.feature_groups:
            group_copy = model_pb2.FeatureGroupConfig()
            group_copy.CopyFrom(group)
            feature_groups.append(group_copy)
        impl_key_to_feat_groups, impl_key_to_seq_groups, _ = (
            build_embedding_group_mappings(features, feature_groups)
        )
        tables: List[Tuple[str, List[str]]] = []
        for impl_key, groups in impl_key_to_feat_groups.items():
            table_configs = build_embedding_bag_configs(
                features,
                groups,
                getattr(model_cfg, "wide_embedding_dim", None),
                getattr(model_cfg, "wide_init_fn", None),
            )
            for kind, configs in (
                ("ebc", table_configs.ebc),
                ("mc_ebc", table_configs.mc_ebc),
                ("ebc_user", table_configs.ebc_user),
                ("mc_ebc_user", table_configs.mc_ebc_user),
            ):
                for name, emb_bag_config in configs.items():
                    tables.append(
                        (
                            f"{impl_key}.{kind}.{name}",
                            list(emb_bag_config.feature_names),
                        )
                    )
        for impl_key, groups in impl_key_to_seq_groups.items():
            ec_table_configs = build_embedding_collection_configs(features, groups)
            for kind, dim_to_configs in (
                ("ec", ec_table_configs.ec),
                ("mc_ec", ec_table_configs.mc_ec),
                ("ec_user", ec_table_configs.ec_user),
                ("mc_ec_user", ec_table_configs.mc_ec_user),
            ):
                for embedding_dim, configs in dim_to_configs.items():
                    for name, emb_config in configs.items():
                        tables.append(
                            (
                                f"{impl_key}.{kind}.{embedding_dim}.{name}",
                                list(emb_config.feature_names),
                            )
                        )
        return tables

    def measure(
        self,
        cfg: EasyRecConfig,
        features: List[BaseFeature],
        tables: List[Tuple[str, List[str]]],
    ) -> Dict[str, Tuple[float, float]]:
        """Sample batches and measure both factors.

        Args:
            cfg (EasyRecConfig): the pipeline config to measure.
            features (list): the config's features.
            tables (List[Tuple[str, List[str]]]): physical tables and their
                features, from :meth:`get_embedding_table_features`.

        Returns:
            feature name -> (pooling_factor, unique_id_factor); the second is
            0.0 for features whose tables were never looked up.
        """
        data_config = copy.deepcopy(cfg.data_config)
        if self._dataset_type is not None:
            data_config.dataset_type = cast(DatasetType, self._dataset_type)
        if self._batch_size is not None:
            data_config.batch_size = self._batch_size
        data_config.drop_remainder = False
        data_config.min_batch_size = 0
        dataloader = create_dataloader(
            data_config, features, self._input_path or cfg.train_input_path
        )
        raw_ids: Dict[str, int] = {}
        table_max: Dict[str, float] = {}
        samples = 0
        got = 0
        for batch in dataloader:
            got += 1
            per_feature: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for kjt in batch.sparse_features.values():
                for key in kjt.keys():
                    jt = kjt[key]
                    per_feature[key] = (
                        jt.values().detach().cpu().numpy(),
                        jt.lengths().detach().cpu().numpy(),
                    )
            batch_samples = 0
            if getattr(batch, "labels", None):
                batch_samples = int(next(iter(batch.labels.values())).shape[0])
            elif per_feature:
                batch_samples = int(next(iter(per_feature.values()))[1].shape[0])
            samples += batch_samples
            for key, (values, _) in per_feature.items():
                raw_ids[key] = raw_ids.get(key, 0) + values.size
            for table, names in tables:
                # a missing member is not an empty one: no partial unions
                if any(name not in per_feature for name in names):
                    continue
                feats = [per_feature[name] for name in names]
                if not batch_samples or any(
                    lengths.shape[0] != batch_samples for _, lengths in feats
                ):
                    # not sample-aligned, e.g. a tiled user view
                    continue
                union = (
                    np.unique(np.concatenate([values for values, _ in feats])).size
                    / batch_samples
                )
                table_max[table] = (
                    union if table not in table_max else max(table_max[table], union)
                )
            if got >= self._num_batches:
                break
        pooling: Dict[str, float] = (
            {
                name: max(1.0, math.ceil(count / samples * 10000) / 10000)
                for name, count in raw_ids.items()
            }
            if samples
            else {}
        )
        unique: Dict[str, float] = {}
        for table, names in tables:
            value = table_max.get(table, 0)
            for name in names:
                if value > 0:
                    unique[name] = max(unique.get(name, 0.0), float(value))
        factors = {
            name: (pooling.get(name, 0.0), unique.get(name, 0.0))
            for name in set(pooling) | set(unique)
        }
        logger.info(
            f"measured {len(factors)} features of {len(tables)} tables from "
            f"{got} batches, {samples} samples"
        )
        return factors

    def fill(
        self, features: List[BaseFeature], factors: Dict[str, Tuple[float, float]]
    ) -> List[str]:
        """Write measured factors into the config, in place.

        The features hold the config's own messages, so setting through them
        fills the config that is saved, and same-named sequence sub-features
        of different sequences stay apart.

        Args:
            features (list): the config's features.
            factors (Dict[str, Tuple[float, float]]): feature name ->
                (pooling_factor, unique_id_factor), keyed like measure()
                returns them.

        Returns:
            the names of the features that got a factor.
        """
        filled = []
        for feature in features:
            pooling_factor, unique_id_factor = factors.get(feature.name, (0.0, 0.0))
            if not pooling_factor and not unique_id_factor:
                continue
            if pooling_factor:
                feature.config.embedding_constraints.pooling_factor = pooling_factor
            if unique_id_factor:
                feature.config.embedding_constraints.unique_id_factor = unique_id_factor
            filled.append(feature.name)
        return filled


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset_type", choices=DatasetType.keys())
    parser.add_argument("--input_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_batches", type=int, default=100)
    args = parser.parse_args()
    tool = AddPoolingFactorToConfig(
        args.pipeline_config_path,
        args.output_path,
        args.num_batches,
        dataset_type=args.dataset_type,
        input_path=args.input_path,
        batch_size=args.batch_size,
    )
    tool.run()


if __name__ == "__main__":
    main()
