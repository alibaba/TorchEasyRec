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

"""Fill each embedding feature's pooling_factor, measured from training data.

Reads a number of batches of the pipeline config's train input through the
same dataloader as training (FG included, no model) and writes
``embedding_constraints { pooling_factor: N }`` on every measured feature of a
copy of the config. The planner sizes a table's inputs, outputs and dynamicemb
GPU caches from ``batch_size * pooling_factor``; features that do not set it
are assumed to look up one id per sample.

dynamicemb features are counted after deduplication, the way dynamicemb looks
them up; other tables count every id. The value written is the largest
per-batch count over the sampled batches, divided by ``batch_size``, rounded
up. Features sharing a table sum their factors at plan time.

Usage: python -m tzrec.tools.add_pooling_factor_to_config
    --pipeline_config_path pipeline.config --output_path pipeline_measured.config
    [--num_batches 100]
"""

import argparse
import math
from typing import Dict, List

import torch

from tzrec.datasets.dataset import create_dataloader
from tzrec.main import _create_features
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


class AddPoolingFactorToConfig(object):
    """Measure and fill pooling_factor for embedding features.

    Args:
        pipeline_config_path (str): pipeline config path; its train input is read.
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
        factors = self.measure(cfg)
        filled = self.fill(cfg, factors)
        config_util.save_message(cfg, self._output_path)
        logger.info(
            f"wrote pooling_factor for {len(filled)} features to {self._output_path}"
        )

    def measure(self, cfg: EasyRecConfig) -> Dict[str, float]:
        """Largest ids per batch of every sparse feature, over sampled batches.

        Args:
            cfg (EasyRecConfig): the pipeline config to measure.

        Returns:
            feature name (as it appears in a batch, with the sequence
            prefix of grouped sub-features) -> measured pooling_factor,
            the largest per-batch count divided by batch_size, rounded up.
        """
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        dynamicemb = {
            f.name: f.config.HasField("dynamicemb")
            for f in features
            if hasattr(f.config, "dynamicemb")
        }
        dataloader = create_dataloader(cfg.data_config, features, cfg.train_input_path)
        counts: Dict[str, int] = {}
        got = 0
        for batch in dataloader:
            got += 1
            for kjt in batch.sparse_features.values():
                for name in kjt.keys():
                    ids = kjt[name].values()
                    if dynamicemb.get(name, False):
                        n = int(torch.unique(ids).numel())
                    else:
                        n = int(ids.numel())
                    counts[name] = max(counts.get(name, 0), n)
            if got >= self._num_batches:
                break
        batch_size = cfg.data_config.batch_size
        factors = {
            name: math.ceil(count / batch_size * 10000) / 10000
            for name, count in counts.items()
        }
        logger.info(
            f"measured {len(factors)} sparse features "
            f"from {got} batches of batch_size {batch_size}"
        )
        return factors

    def fill(self, cfg: EasyRecConfig, factors: Dict[str, float]) -> List[str]:
        """Write measured factors into the config, in place.

        Args:
            cfg (EasyRecConfig): the config to fill, modified in place.
            factors (Dict[str, float]): feature name -> measured factor, keyed
                like measure() returns them.

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
                factor = factors.get(names.get(f.feature_name, f.feature_name))
                if factor is None:
                    continue
                f.embedding_constraints.pooling_factor = factor
                filled.append(f.feature_name)
        return filled


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
