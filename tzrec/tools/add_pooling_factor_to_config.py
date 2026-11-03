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
  unique_id_factor  largest number of distinct ids one sample looks up in a
                    table through all the features that share it: the
                    per-sample union, measured jointly, never the max of
                    independently measured features. The table's conservative
                    cache working-set estimate is batch_size * the factor.
                    The same table-level value is written on each member
                    feature; a feature feeding several physical tables keeps
                    the maximum of their measured factors.

Both values are measured in one pass. Features missing from the sampled
batches are left unset; the constraint builder falls back to the raw id
factors for them, so a shared table is never silently underestimated.

Usage: python -m tzrec.tools.add_pooling_factor_to_config
    --pipeline_config_path pipeline.config --output_path pipeline_measured.config
    [--num_batches 100]
"""

import argparse
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from tzrec.datasets.dataset import create_dataloader
from tzrec.main import _create_features, _create_model
from tzrec.modules.embedding import EmbeddingGroupImpl, SequenceEmbeddingGroupImpl
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


class AddPoolingFactorToConfig(object):
    """Measure and fill pooling_factor and unique_id_factor for embeddings.

    Args:
        pipeline_config_path (str): pipeline config path; its train input is
            read and its model config decides the physical tables.
        output_path (str): path to write the filled config to.
        num_batches (int): number of batches to sample.
    """

    def __init__(
        self, pipeline_config_path: str, output_path: str, num_batches: int
    ) -> None:
        self._pipeline_config_path = pipeline_config_path
        self._output_path = output_path
        self._num_batches = num_batches

    def run(self) -> None:
        """Measure the train input and write the filled config."""
        cfg = config_util.load_pipeline_config(self._pipeline_config_path)
        model = self.build_model(cfg)
        tables = self.table_features(model)
        factors = self.measure(cfg, tables)
        filled = self.fill(cfg, factors)
        config_util.save_message(cfg, self._output_path)
        logger.info(
            f"wrote pooling_factor for "
            f"{sum(1 for p, _ in factors.values() if p)} features and "
            f"unique_id_factor for {sum(1 for _, u in factors.values() if u)} "
            f"to {self._output_path}, {len(filled)} features filled"
        )

    @staticmethod
    def build_model(cfg: EasyRecConfig) -> Any:
        """Build the config's model, meta device, for its table mapping."""
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        return _create_model(
            cfg.model_config,
            features,
            list(cfg.data_config.label_fields),
            sample_weights=list(cfg.data_config.sample_weight_fields),
        )

    @staticmethod
    def table_features(model: Any) -> List[Tuple[str, List[str]]]:
        """Physical tables of the model and the features that feed each.

        Tables are identified by collection and name, so embedding-name
        suffixes, wide renaming and input-tile user collections keep apart
        tables that a bare embedding_name would merge.

        Returns:
            (table, feature names) pairs; feature names as they appear in a
            batch (grouped sequence features carry their sequence prefix).
        """
        tables = []
        for m in model.modules():
            if isinstance(m, EmbeddingGroupImpl):
                for attr in ("ebc", "ebc_user"):
                    ebc = getattr(m, attr, None)
                    if ebc is None:
                        continue
                    for config in ebc.embedding_bag_configs():
                        tables.append(
                            (f"{attr}.{config.name}", list(config.feature_names))
                        )
            elif isinstance(m, SequenceEmbeddingGroupImpl):
                for attr in ("ec_dict", "ec_dict_user"):
                    ec_dict = getattr(m, attr, None)
                    if not ec_dict:
                        continue
                    for ec in ec_dict.values():
                        for config in ec.embedding_configs():
                            tables.append(
                                (f"{attr}.{config.name}", list(config.feature_names))
                            )
        return tables

    def measure(
        self, cfg: EasyRecConfig, tables: List[Tuple[str, List[str]]]
    ) -> Dict[str, Tuple[float, float]]:
        """Sample batches and measure both factors.

        Args:
            cfg (EasyRecConfig): the pipeline config to measure.
            tables (List[Tuple[str, List[str]]]): physical tables and their
                features, from :meth:`table_features`.

        Returns:
            feature name -> (pooling_factor, unique_id_factor); the second is
            0.0 for features whose tables were never looked up.
        """
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        dataloader = create_dataloader(cfg.data_config, features, cfg.train_input_path)
        raw_ids: Dict[str, int] = {}
        table_max: Dict[str, int] = {}
        samples = 0
        got = 0
        for batch in dataloader:
            got += 1
            per_feature: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for kjt in batch.sparse_features.values():
                for key in kjt.keys():
                    if key in per_feature:
                        # input-tile duplicates read the same column
                        continue
                    per_feature[key] = (
                        kjt[key].values().detach().cpu().numpy().astype(np.int64),
                        kjt[key].lengths().detach().cpu().numpy(),
                    )
            if per_feature:
                samples += int(next(iter(per_feature.values()))[1].shape[0])
            elif getattr(batch, "labels", None):
                samples += int(next(iter(batch.labels.values())).shape[0])
            for key, (values, _) in per_feature.items():
                raw_ids[key] = raw_ids.get(key, 0) + values.size
            for table, names in tables:
                feats = [per_feature[n] for n in names if n in per_feature]
                if feats:
                    table_max[table] = max(
                        table_max.get(table, 0), _max_distinct_union(feats)
                    )
            if got >= self._num_batches:
                break
        pooling = (
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
        self, cfg: EasyRecConfig, factors: Dict[str, Tuple[float, float]]
    ) -> List[str]:
        """Write measured factors into the config, in place.

        Args:
            cfg (EasyRecConfig): the config to fill, modified in place.
            factors (Dict[str, Tuple[float, float]]): feature name ->
                (pooling_factor, unique_id_factor), keyed like measure()
                returns them.

        Returns:
            the names of the features that got a factor.
        """
        names = {
            f.config.feature_name: f.name
            for f in _create_features(list(cfg.feature_configs), cfg.data_config)
        }
        filled = []
        for feat_config in cfg.feature_configs:
            oneof = feat_config.WhichOneof("feature")
            if oneof is None:
                continue
            if oneof.startswith("sequence_"):
                sub_configs = list(getattr(feat_config, oneof).features)
            else:
                sub_configs = [feat_config]
            for sub_config in sub_configs:
                sub_oneof = sub_config.WhichOneof("feature")
                if sub_oneof is None:
                    continue
                f = getattr(sub_config, sub_oneof)
                pooling_factor, unique_id_factor = factors.get(
                    names.get(f.feature_name, f.feature_name), (0.0, 0.0)
                )
                if not pooling_factor and not unique_id_factor:
                    continue
                if pooling_factor:
                    f.embedding_constraints.pooling_factor = pooling_factor
                if unique_id_factor:
                    f.embedding_constraints.unique_id_factor = unique_id_factor
                filled.append(f.feature_name)
        return filled


def _max_distinct_union(feats: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """Largest per-sample number of distinct ids across the given features.

    The ids of all features are laid out in one padded row per sample and
    sorted; the number of value changes in the filled part of a row is its
    distinct count. Padding is a value above every id, so it sorts last.

    Args:
        feats (List[Tuple[np.ndarray, np.ndarray]]): (ids, lengths per sample)
            of every feature of one table.

    Returns:
        the maximum distinct count over the samples.
    """
    num_samples = int(feats[0][1].shape[0])
    row_total = np.zeros(num_samples, dtype=np.int64)
    max_id = 0
    for values, lengths in feats:
        row_total += lengths
        if values.size:
            max_id = max(max_id, int(values.max()))
    width = int(row_total.max()) if num_samples else 0
    if width == 0:
        return 0
    rows = np.full((num_samples, width), max_id + 1, dtype=np.int64)
    cursor = np.zeros(num_samples, dtype=np.int64)
    for values, lengths in feats:
        if not values.size:
            continue
        starts = np.cumsum(lengths) - lengths
        row_idx = np.repeat(np.arange(num_samples), lengths)
        col_idx = np.arange(values.size) - np.repeat(starts, lengths) + cursor[row_idx]
        rows[row_idx, col_idx] = values
        cursor += lengths
    sorted_rows = np.sort(rows, axis=1)
    changes = np.ones(sorted_rows.shape, dtype=bool)
    changes[:, 1:] = sorted_rows[:, 1:] != sorted_rows[:, :-1]
    distinct = (changes & (sorted_rows != max_id + 1)).sum(axis=1)
    return int(distinct.max())


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_batches", type=int, default=100)
    args = parser.parse_args()
    tool = AddPoolingFactorToConfig(
        args.pipeline_config_path, args.output_path, args.num_batches
    )
    tool.run()


if __name__ == "__main__":
    main()
