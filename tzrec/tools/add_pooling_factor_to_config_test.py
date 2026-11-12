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

import json
import os
import unittest

import pyarrow as pa
import pyarrow.dataset as ds
from parameterized import parameterized

from tzrec.protos import data_pb2, model_pb2, pipeline_pb2
from tzrec.tools.add_pooling_factor_to_config import AddPoolingFactorToConfig
from tzrec.utils import config_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.test_util import make_test_dir, mark_ci_scope, parameterized_name_func


def _write_parquet(test_dir, num_rows=10):
    r"""Per-sample columns; ``\x03`` separates the ids of one sample."""
    t = pa.Table.from_arrays(
        [
            pa.array(["7"] * num_rows),
            pa.array(["1\x032\x033"] * num_rows),
            pa.array(["5" if i % 2 == 0 else "" for i in range(num_rows)]),
            pa.array(["100\x03200"] * num_rows),
            pa.array(["200\x03300"] * num_rows),
            pa.array(["1\x032"] * num_rows),
            pa.array(["3\x034"] * num_rows),
            pa.array(["1\x032\x032"] * num_rows),
            pa.array(["1\x035000"] * num_rows),
            pa.array([["1", "2", "3"]] * num_rows, type=pa.list_(pa.string())),
            pa.array([1] * num_rows),
        ],
        names=[
            "id_a",
            "multi_b",
            "sparse_c",
            "shared_a",
            "shared_b",
            "dis_a",
            "dis_b",
            "t1",
            "t2",
            "click_seq__seq_id",
            "label",
        ],
    )
    ds.write_dataset(t, test_dir, format="parquet")


def _config(train_input_path, with_model=True):
    cfg = pipeline_pb2.EasyRecConfig()
    cfg.train_input_path = train_input_path
    cfg.data_config.batch_size = 4
    cfg.data_config.dataset_type = data_pb2.DatasetType.ParquetDataset
    cfg.data_config.fg_mode = data_pb2.FgMode.FG_NONE
    cfg.data_config.label_fields.append("label")

    def _id(name, embedding_name="", dim=8):
        f = cfg.feature_configs.add().id_feature
        f.feature_name = name
        f.embedding_dim = dim
        f.num_buckets = 100
        if embedding_name:
            f.embedding_name = embedding_name

    _id("id_a")
    _id("multi_b")
    _id("sparse_c")
    _id("shared_a", "shared", dim=4)
    _id("shared_b", "shared", dim=4)
    _id("dis_a", "disjoint", dim=4)
    _id("dis_b", "disjoint", dim=4)
    _id("t1", "item", dim=4)
    _id("t2", "item2", dim=4)
    seq = cfg.feature_configs.add().sequence_feature
    seq.sequence_name = "click_seq"
    seq.sequence_length = 50
    sub = seq.features.add()
    sub.id_feature.feature_name = "seq_id"
    sub.id_feature.embedding_dim = 8
    sub.id_feature.num_buckets = 100
    if with_model:
        group = cfg.model_config.feature_groups.add()
        group.group_name = "deep"
        group.feature_names.extend(
            [
                "id_a",
                "multi_b",
                "sparse_c",
                "shared_a",
                "shared_b",
                "dis_a",
                "dis_b",
                "t1",
                "t2",
            ]
        )
        group.group_type = model_pb2.FeatureGroupType.DEEP
        mt = cfg.model_config.multi_tower
        mt.SetInParent()
        tower = mt.towers.add()
        tower.input = "deep"
        tower.mlp.hidden_units.extend([8])
        mt.final.hidden_units.extend([8])
    return cfg


class AddPoolingFactorToConfigTest(unittest.TestCase):
    """Measuring a train input and filling embedding_constraints from it."""

    @staticmethod
    def _measure(test_dir, tables, **kwargs):
        from tzrec.main import _create_features

        cfg = _config(os.path.join(test_dir, "*.parquet"), **kwargs)
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        tool = AddPoolingFactorToConfig("in.config", "out.config", num_batches=10)
        return cfg, tool.measure(cfg, features, tables)

    def test_measures_raw_average_and_joint_union(self):
        test_dir = make_test_dir()
        _write_parquet(test_dir)
        # hand-resolved tables: t1 feeds two physical tables
        cfg, factors = self._measure(
            test_dir,
            [
                ("ebc.id_a_emb", ["id_a"]),
                ("ebc.multi_b_emb", ["multi_b"]),
                ("ebc.sparse_c_emb", ["sparse_c"]),
                ("ebc.shared", ["shared_a", "shared_b"]),
                ("ebc.disjoint", ["dis_a", "dis_b"]),
                ("ebc.item", ["t1"]),
                ("ebc.itemx", ["t1", "t2"]),
            ],
        )
        # raw average per sample, partial batches counted: 10 rows in
        # batches of 4 give 10 samples, so multi_b is 30/10, not 30/12
        self.assertEqual(factors["id_a"], (1.0, 0.5))
        self.assertEqual(factors["multi_b"], (3.0, 1.5))
        # a feature with ids in half the samples clamps to 1
        self.assertEqual(factors["sparse_c"], (1.0, 0.5))
        # The tail batch has three joint IDs across two samples.
        self.assertEqual(factors["shared_a"], (2.0, 1.5))
        self.assertEqual(factors["shared_b"], (2.0, 1.5))
        # disjoint ids of two features union to their sum
        self.assertEqual(factors["dis_a"], (2.0, 2.0))
        self.assertEqual(factors["dis_b"], (2.0, 2.0))
        # a feature in two physical tables keeps the larger table's value
        self.assertEqual(factors["t1"], (3.0, 1.5))
        self.assertEqual(factors["t2"], (2.0, 1.5))

    @parameterized.expand(
        [("tail_maximum", False, 0.125), ("first_batch_maximum", True, 1.125)],
        name_func=parameterized_name_func,
    )
    def test_global_batches_and_pooling_average(self, _, reverse, first_factor):
        # eight samples bring one id and two bring nine, so the average is
        # 26/10 = 2.6: neither the per-batch maximum (9) nor a division by
        # padded batch counts (26/12) may replace it
        test_dir = make_test_dir()
        num_rows = 10
        ids = ["7"] * 8 + ["1\x032\x033\x034\x035\x036\x037\x038\x039"] * 2
        if reverse:
            ids.reverse()
        t = pa.Table.from_arrays(
            [
                pa.array(ids),
                pa.array([1] * num_rows),
            ],
            names=["id_a", "label"],
        )
        ds.write_dataset(t, test_dir, format="parquet")
        cfg = pipeline_pb2.EasyRecConfig()
        cfg.train_input_path = os.path.join(test_dir, "*.parquet")
        cfg.data_config.batch_size = 4
        cfg.data_config.drop_remainder = True
        cfg.data_config.min_batch_size = 4
        cfg.data_config.dataset_type = data_pb2.DatasetType.ParquetDataset
        cfg.data_config.fg_mode = data_pb2.FgMode.FG_NONE
        cfg.data_config.label_fields.append("label")
        f = cfg.feature_configs.add().id_feature
        f.feature_name = "id_a"
        f.embedding_dim = 8
        f.num_buckets = 100
        from tzrec.main import _create_features

        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        factors = AddPoolingFactorToConfig(
            "in.config", "out.config", num_batches=10
        ).measure(cfg, features, [("ebc.id_a_emb", ["id_a"])])
        self.assertEqual(factors["id_a"], (2.6, 2.25 if reverse else 4.5))
        self.assertEqual(cfg.data_config.batch_size, 4)
        cfg.data_config.batch_size = 8
        tool = AddPoolingFactorToConfig("in.config", "out.config", 1)
        self.assertEqual(
            tool.measure(cfg, features, [("t", ["id_a"])])["id_a"],
            (3.0 if reverse else 1.0, first_factor),
        )
        self.assertEqual(cfg.data_config.batch_size, 8)
        self.assertTrue(cfg.data_config.drop_remainder)
        self.assertEqual(cfg.data_config.min_batch_size, 4)
        tool = AddPoolingFactorToConfig("in.config", "out.config", 2)
        self.assertEqual(
            tool.measure(cfg, features, [("t", ["id_a"])])["id_a"],
            (2.6, 1.125 if reverse else 4.5),
        )

    def test_incomplete_shared_table_has_no_unique_id_estimate(self):
        # a missing member is not an empty one: its table keeps no union
        # measurement, while the member that did appear is still measured
        test_dir = make_test_dir()
        _write_parquet(test_dir)
        cfg, factors = self._measure(
            test_dir,
            [("ebc.partial", ["id_a", "absent_from_batches"])],
        )
        self.assertEqual(factors["id_a"], (1.0, 0.0))

    def test_shared_table_union_handles_empty_samples(self):
        # an observed feature with no ids is a member like any other: the
        # union is measured over every sample
        test_dir = make_test_dir()
        _write_parquet(test_dir)
        cfg, factors = self._measure(
            test_dir,
            [("ebc.sparse_union", ["id_a", "sparse_c"])],
        )
        # id_a holds 7 in every sample, sparse_c adds 5 in the even ones
        self.assertEqual(factors["id_a"], (1.0, 1.0))
        self.assertEqual(factors["sparse_c"], (1.0, 1.0))

    def test_table_feature_mapping_preserves_sharing_and_separation(self):
        cfg = _config("unused.parquet")
        from tzrec.datasets.utils import BASE_DATA_GROUP
        from tzrec.main import _create_features

        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        table_list = AddPoolingFactorToConfig.get_embedding_table_features(
            cfg, features
        )
        tables = dict(table_list)
        self.assertEqual(tables[f"{BASE_DATA_GROUP}.ebc.id_a_emb"], ["id_a"])
        self.assertEqual(
            tables[f"{BASE_DATA_GROUP}.ebc.shared"], ["shared_a", "shared_b"]
        )
        # similarly named tables stay apart
        self.assertEqual(tables[f"{BASE_DATA_GROUP}.ebc.item"], ["t1"])
        self.assertEqual(tables[f"{BASE_DATA_GROUP}.ebc.item2"], ["t2"])
        self.assertNotIn(f"{BASE_DATA_GROUP}.ebc.seq_id", tables)

    def test_get_embedding_table_features_keeps_the_config_unchanged(self):
        # the mapping helpers supplement a nested sequence group's name; the
        # config the tool saves must not carry that
        cfg = _config("unused.parquet", with_model=False)
        from tzrec.main import _create_features

        group = cfg.model_config.feature_groups.add()
        group.group_name = "deep"
        group.feature_names.append("id_a")
        group.sequence_groups.add().feature_names.extend(["click_seq__seq_id"])
        group.sequence_encoders.add().simple_attention.SetInParent()
        before = str(cfg)
        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        AddPoolingFactorToConfig.get_embedding_table_features(cfg, features)
        self.assertEqual(str(cfg), before)
        self.assertFalse(
            cfg.model_config.feature_groups[-1]
            .sequence_groups[0]
            .HasField("group_name")
        )

    def test_fill_writes_both_factors_and_keeps_sequence_identity(self):
        # a normal feature and two sequence sub-features that share the bare
        # name each keep their own measured factors, and the writes land on
        # the config the tool saves
        test_dir = make_test_dir()
        cfg = _config(os.path.join(test_dir, "*.parquet"), with_model=False)
        seq = cfg.feature_configs.add().sequence_feature
        seq.sequence_name = "buy_seq"
        seq.sequence_length = 50
        sub = seq.features.add()
        sub.id_feature.feature_name = "seq_id"
        sub.id_feature.embedding_dim = 8
        sub.id_feature.num_buckets = 100
        from tzrec.main import _create_features

        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        tool = AddPoolingFactorToConfig("in.config", "out.config", num_batches=10)
        filled = tool.fill(
            features,
            {
                "id_a": (1.0, 1.0),
                "shared_a": (2.0, 3.0),
                "click_seq__seq_id": (3.0, 3.0),
                "buy_seq__seq_id": (7.0, 9.0),
            },
        )
        self.assertEqual(
            set(filled),
            {"id_a", "shared_a", "click_seq__seq_id", "buy_seq__seq_id"},
        )
        self.assertEqual(
            cfg.feature_configs[0].id_feature.embedding_constraints.pooling_factor,
            1.0,
        )
        self.assertEqual(
            cfg.feature_configs[3].id_feature.embedding_constraints.unique_id_factor,
            3.0,
        )
        self.assertAlmostEqual(
            cfg.feature_configs[9]
            .sequence_feature.features[0]
            .id_feature.embedding_constraints.pooling_factor,
            3.0,
        )
        self.assertAlmostEqual(
            cfg.feature_configs[10]
            .sequence_feature.features[0]
            .id_feature.embedding_constraints.pooling_factor,
            7.0,
        )
        self.assertAlmostEqual(
            cfg.feature_configs[10]
            .sequence_feature.features[0]
            .id_feature.embedding_constraints.unique_id_factor,
            9.0,
        )

    def test_invalid_tool_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_batches"):
            AddPoolingFactorToConfig("in.config", "out.config", 0)
        with self.assertRaisesRegex(ValueError, "output_path"):
            AddPoolingFactorToConfig("dir/pipeline.config", "dir/pipeline.config", 5)

    @parameterized.expand([(False,), (True,)], name_func=parameterized_name_func)
    def test_run_measures_and_writes_a_loadable_config(self, edit):
        test_dir = make_test_dir()
        _write_parquet(test_dir)
        in_path = os.path.join(test_dir, "pipeline.config")
        out_path = os.path.join(test_dir, "measured.config")
        cfg = _config("unused.parquet" if edit else os.path.join(test_dir, "*.parquet"))
        config_util.save_message(cfg, in_path)
        before = config_util.load_pipeline_config(in_path)
        edits = (
            {
                "data_config.batch_size": 8,
                "train_input_path": os.path.join(test_dir, "*.parquet"),
                "feature_configs[0].id_feature.embedding_dim": 16,
            }
            if edit
            else None
        )
        AddPoolingFactorToConfig(
            in_path,
            out_path,
            num_batches=1 if edit else 10,
            edit_config_json=json.dumps(edits) if edit else None,
        ).run()
        out = config_util.load_pipeline_config(out_path)
        self.assertEqual(config_util.load_pipeline_config(in_path), before)
        if edits:
            config_util.edit_config(before, edits)
        self.assertEqual(out.train_input_path, before.train_input_path)
        # measured factors for a normal and a shared feature, and for the
        # two features of similarly named tables
        self.assertEqual(
            out.feature_configs[0].id_feature.embedding_constraints.pooling_factor,
            1.0,
        )
        self.assertEqual(
            out.feature_configs[3].id_feature.embedding_constraints.unique_id_factor,
            0.375 if edit else 1.5,
        )
        self.assertEqual(
            out.feature_configs[7].id_feature.embedding_constraints.unique_id_factor,
            0.25 if edit else 1.0,
        )
        self.assertEqual(
            out.feature_configs[8].id_feature.embedding_constraints.unique_id_factor,
            0.25 if edit else 1.0,
        )
        # nothing else about the configuration changed
        for field in ("model_config", "data_config"):
            self.assertEqual(str(getattr(out, field)), str(getattr(before, field)))

        def _clear_constraints(feature_config):
            oneof = feature_config.WhichOneof("feature")
            f = getattr(feature_config, oneof) if oneof is not None else None
            if f is None:
                return
            if f.DESCRIPTOR.fields_by_name.get("embedding_constraints"):
                f.ClearField("embedding_constraints")
            if f.DESCRIPTOR.fields_by_name.get("features"):
                for sub in f.features:
                    _clear_constraints(sub)

        for got, was in zip(out.feature_configs, before.feature_configs):
            _clear_constraints(got)
            _clear_constraints(was)
            self.assertEqual(str(got), str(was))

    @unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
    @mark_ci_scope("gpu")
    def test_dynamicemb_counts_raw_positions_and_distinct_union(self):
        test_dir = make_test_dir()
        num_rows = 10
        t = pa.Table.from_arrays(
            [
                pa.array(["7\x037\x038"] * num_rows),
                pa.array([1] * num_rows),
            ],
            names=["id_a", "label"],
        )
        ds.write_dataset(t, test_dir, format="parquet")
        cfg = pipeline_pb2.EasyRecConfig()
        cfg.train_input_path = os.path.join(test_dir, "*.parquet")
        cfg.data_config.batch_size = 4
        cfg.data_config.dataset_type = data_pb2.DatasetType.ParquetDataset
        cfg.data_config.fg_mode = data_pb2.FgMode.FG_NONE
        cfg.data_config.label_fields.append("label")
        f = cfg.feature_configs.add().id_feature
        f.feature_name = "id_a"
        f.embedding_dim = 8
        f.dynamicemb.max_capacity = 100_000
        from tzrec.main import _create_features

        features = _create_features(list(cfg.feature_configs), cfg.data_config)
        factors = AddPoolingFactorToConfig(
            "in.config", "out.config", num_batches=10
        ).measure(cfg, features, [("ebc.id_a_emb", ["id_a"])])
        # raw positions include the duplicate 7; the union does not
        self.assertEqual(factors["id_a"], (3.0, 1.0))


if __name__ == "__main__":
    unittest.main()
