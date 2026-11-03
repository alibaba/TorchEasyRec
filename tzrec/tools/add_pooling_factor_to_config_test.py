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

import os
import unittest

import pyarrow as pa
import pyarrow.dataset as ds

from tzrec.protos import data_pb2, pipeline_pb2
from tzrec.tools.add_pooling_factor_to_config import AddPoolingFactorToConfig
from tzrec.utils import config_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.test_util import make_test_dir, mark_ci_scope


def _write_parquet(test_dir, num_rows):
    t = pa.Table.from_arrays(
        [
            pa.array(["7"] * num_rows),
            pa.array(["2\x033"] * num_rows),
            pa.array([["1", "2", "3"]] * num_rows, type=pa.list_(pa.string())),
            pa.array([1] * num_rows),
        ],
        names=["id_a", "tag_b", "click_seq__seq_id", "label"],
    )
    ds.write_dataset(t, test_dir, format="parquet")


def _config(train_input_path):
    cfg = pipeline_pb2.EasyRecConfig()
    cfg.train_input_path = train_input_path
    cfg.data_config.batch_size = 4
    cfg.data_config.dataset_type = data_pb2.DatasetType.ParquetDataset
    cfg.data_config.fg_mode = data_pb2.FgMode.FG_NONE
    cfg.data_config.label_fields.append("label")
    cfg.feature_configs.add().id_feature.SetInParent()
    cfg.feature_configs[0].id_feature.feature_name = "id_a"
    cfg.feature_configs[0].id_feature.embedding_dim = 8
    cfg.feature_configs[0].id_feature.num_buckets = 100
    cfg.feature_configs.add().id_feature.SetInParent()
    cfg.feature_configs[1].id_feature.feature_name = "tag_b"
    cfg.feature_configs[1].id_feature.embedding_dim = 8
    cfg.feature_configs[1].id_feature.num_buckets = 100
    seq = cfg.feature_configs.add().sequence_feature
    seq.sequence_name = "click_seq"
    seq.sequence_length = 50
    sub = seq.features.add()
    sub.id_feature.SetInParent()
    sub.id_feature.feature_name = "seq_id"
    sub.id_feature.embedding_dim = 8
    sub.id_feature.num_buckets = 100
    return cfg


class AddPoolingFactorToConfigTest(unittest.TestCase):
    """Measuring a train input and filling embedding_constraints from it."""

    def test_measures_ids_per_sample_and_fills_the_config(self):
        test_dir = make_test_dir()
        _write_parquet(test_dir, 8)
        cfg = _config(os.path.join(test_dir, "*.parquet"))
        tool = AddPoolingFactorToConfig("", "", num_batches=2)
        factors = tool.measure(cfg)
        self.assertAlmostEqual(factors["id_a"], 1.0)
        self.assertAlmostEqual(factors["tag_b"], 2.0)
        self.assertAlmostEqual(factors["click_seq__seq_id"], 3.0)

        filled = tool.fill(cfg, factors)
        self.assertEqual(set(filled), {"id_a", "tag_b", "seq_id"})
        self.assertEqual(
            cfg.feature_configs[0].id_feature.embedding_constraints.pooling_factor,
            1.0,
        )
        self.assertAlmostEqual(
            cfg.feature_configs[1].id_feature.embedding_constraints.pooling_factor,
            2.0,
        )

        # sequence sub-features are looked up under their batch name
        self.assertEqual(tool.fill(cfg, {"click_seq__seq_id": 50.0}), ["seq_id"])
        self.assertAlmostEqual(
            cfg.feature_configs[2]
            .sequence_feature.features[0]
            .id_feature.embedding_constraints.pooling_factor,
            50.0,
        )
        # features without a measured factor keep what they had
        self.assertEqual(
            cfg.feature_configs[0].id_feature.embedding_constraints.pooling_factor,
            1.0,
        )

    def test_run_writes_a_loadable_config_with_measured_factors(self):
        test_dir = make_test_dir()
        _write_parquet(test_dir, 8)
        in_path = os.path.join(test_dir, "pipeline.config")
        out_path = os.path.join(test_dir, "measured.config")
        cfg = _config(os.path.join(test_dir, "*.parquet"))
        config_util.save_message(cfg, in_path)
        AddPoolingFactorToConfig(in_path, out_path, num_batches=2).run()
        out = config_util.load_pipeline_config(out_path)
        self.assertAlmostEqual(
            out.feature_configs[1].id_feature.embedding_constraints.pooling_factor,
            2.0,
            places=5,
        )


@unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
@mark_ci_scope("gpu")
class AddPoolingFactorDynamicEmbTest(unittest.TestCase):
    """dynamicemb features count the ids of a batch after deduplication."""

    def test_dynamicemb_features_count_unique_ids(self):
        test_dir = make_test_dir()
        # every sample repeats one id, so a batch holds a single unique id
        _write_parquet(test_dir, 8)
        cfg = _config(os.path.join(test_dir, "*.parquet"))
        cfg.feature_configs[0].id_feature.dynamicemb.max_capacity = 100_000
        factors = AddPoolingFactorToConfig("", "", num_batches=2).measure(cfg)
        self.assertAlmostEqual(factors["id_a"], 0.25)


if __name__ == "__main__":
    unittest.main()
