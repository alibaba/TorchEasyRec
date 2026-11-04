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
batches are left unset; the constraint builder aggregates only configured
values and leaves a table-level value unset when no member carries one.

Tables are resolved from the pipeline configuration's feature groups, the
same resolution model construction uses; no model is built. Any model that
follows the standard feature_groups semantics is supported without importing
its class; a custom model that decides table sharing in its own constructor
is outside this mapping. Managed-collision (ZCH) tables are resolved as
tables; the deduplicated count is over raw ids and does not measure ZCH's
post-collision embedding-row working set.

Usage: python -m tzrec.tools.add_pooling_factor_to_config
    --pipeline_config_path pipeline.config --output_path pipeline_measured.config
    [--num_batches 100]
"""

import argparse
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from tzrec.datasets.dataset import create_dataloader
from tzrec.main import _create_features
from tzrec.modules.embedding import resolve_embedding_tables
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
        tables = self.table_features(cfg, features)
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
    def table_features(
        cfg: EasyRecConfig, features: List[Any]
    ) -> List[Tuple[str, List[str]]]:
        """Physical tables of the config and the features that feed each.

        Resolution walks the pipeline configuration's feature groups exactly
        like model construction - wide renaming, embedding-name suffixes,
        input-tile user collections, managed-collision (ZCH) collections and
        data-group separation included - without building any module.

        Args:
            cfg (EasyRecConfig): the pipeline config.
            features (list): the config's features.

        Returns:
            (table, feature names) pairs, the table identified by data
            group, collection, embedding dim and name; feature names as
            they appear in a batch (grouped sequence features carry their
            sequence prefix).
        """
        model_oneof = cfg.model_config.WhichOneof("model")
        model_cfg = (
            getattr(cfg.model_config, model_oneof) if model_oneof is not None else None
        )
        if isinstance(model_cfg, str):
            model_cfg = None
        return resolve_embedding_tables(
            features,
            list(cfg.model_config.feature_groups),
            wide_embedding_dim=getattr(model_cfg, "wide_embedding_dim", None),
            wide_init_fn=getattr(model_cfg, "wide_init_fn", None),
        )

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
                    jt = kjt[key]
                    per_feature[key] = (
                        jt.values().detach().cpu().numpy(),
                        jt.lengths().detach().cpu().numpy(),
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
        filled = []
        # the features hold the config's own messages, so setting through them
        # fills the config in place, and same-named sequence sub-features of
        # different sequences stay apart
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        for feature in features:
            pooling_factor, unique_id_factor = factors.get(feature.name, (0.0, 0.0))
            if not pooling_factor and not unique_id_factor:
                continue
            if pooling_factor:
                feature.config.embedding_constraints.pooling_factor = pooling_factor
            if unique_id_factor:
                feature.config.embedding_constraints.unique_id_factor = unique_id_factor
            filled.append(feature.config.feature_name)
        return filled


def _max_distinct_union(feats: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """Largest per-sample number of distinct ids across the given features.

    The ids of all features are concatenated with the index of their sample
    and lexsorted by (sample, id); a position starts a new distinct run when
    either index changes, and the runs are counted per sample. Cost and memory
    scale with the number of ids, not with the longest sample, and every int64
    id value is safe.

    Args:
        feats (List[Tuple[np.ndarray, np.ndarray]]): (ids, lengths per sample)
            of every feature of one table.

    Returns:
        the maximum distinct count over the samples.
    """
    num_samples = int(feats[0][1].shape[0])
    ids = np.concatenate([values for values, _ in feats])
    if ids.size == 0:
        return 0
    rows = np.concatenate(
        [np.repeat(np.arange(num_samples), lengths) for _, lengths in feats]
    )
    order = np.lexsort((ids, rows))
    ids, rows = ids[order], rows[order]
    starts = np.ones(ids.size, dtype=bool)
    starts[1:] = (rows[1:] != rows[:-1]) | (ids[1:] != ids[:-1])
    return int(np.bincount(rows[starts], minlength=num_samples).max())


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
