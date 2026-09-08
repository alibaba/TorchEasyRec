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
import torch
from parameterized import param, parameterized
from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
    jt_is_equal,
    kjt_is_equal,
)

from tzrec.datasets.data_parser import DataParser
from tzrec.features.feature import FgMode, create_features
from tzrec.protos import feature_pb2
from tzrec.utils.test_util import parameterized_name_func


class DataParserTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("INPUT_TILE", None)
        os.environ.pop("INPUT_TILE_3_ONLINE", None)

    def test_nofg(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b", embedding_dim=8, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_b", value_dim=2)
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_a",
                                embedding_dim=16,
                                num_buckets=100,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="int_a", fg_encoded_default_value="0"
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs)

        data_parser = DataParser(features=features, labels=["label"])

        data = data_parser.parse(
            input_data={
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(["4\x035", "", "6"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                "label": pa.array([0, 0, 1], pa.int32()),
            }
        )

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5, 6], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected_int_a_values = torch.tensor([[7], [8], [9]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3, 1, 0], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16], [17], [0]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["int_b.values"], expected_int_b_values)
        torch.testing.assert_close(
            data["click_seq__cat_a.values"], expected_seq_cat_a_values
        )
        torch.testing.assert_close(
            data["click_seq__cat_a.lengths"], expected_seq_cat_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__int_a.values"], expected_seq_int_a_values
        )
        torch.testing.assert_close(
            data["click_seq__int_a.lengths"], expected_seq_int_a_seq_lengths
        )
        torch.testing.assert_close(data["label"], expected_label)

        expected_dense_feat = KeyedTensor(
            keys=["int_a", "int_b"],
            length_per_key=[1, 2],
            values=torch.tensor(
                [[7, 27, 37], [8, 28, 38], [9, 29, 39]], dtype=torch.float32
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "tag_b", "click_seq__cat_a"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13]),
            lengths=torch.tensor([1, 1, 1, 2, 0, 1, 3, 1, 0], dtype=torch.int32),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        batch = data_parser.to_batch(data)
        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )
        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    def test_list_label(self):
        """A label stored as a list column is parsed into a jagged label."""
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
        ]
        features = create_features(feature_cfgs)

        data_parser = DataParser(features=features, labels=["label", "list_label"])

        data = data_parser.parse(
            input_data={
                "cat_a": pa.array([1, 2, 3]),
                "label": pa.array([0, 0, 1], pa.int32()),
                "list_label": pa.array([[4], [5], [6]], pa.list_(pa.int32())),
            }
        )
        batch = data_parser.to_batch(data)

        torch.testing.assert_close(
            batch.labels["label"], torch.tensor([0, 0, 1], dtype=torch.int64)
        )
        self.assertNotIn("list_label", batch.labels)
        torch.testing.assert_close(
            batch.jagged_labels["list_label"].values(),
            torch.tensor([4, 5, 6], dtype=torch.int64),
        )

    def test_fg_encoded_id_with_weight(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a1",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                    fg_encoded_default_value="0",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b",
                    embedding_dim=8,
                    num_buckets=1000,
                    weighted=True,
                    use_weight=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b1",
                    embedding_dim=8,
                    num_buckets=1000,
                    weighted=True,
                    use_weight=True,
                    fg_encoded_default_value="0",
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_b", value_dim=2)
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_a", embedding_dim=16, num_buckets=100
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="int_a", fg_encoded_default_value="0"
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs)

        data_parser = DataParser(features=features, labels=["label"])

        data = data_parser.parse(
            input_data={
                "cat_a": pa.array(
                    [{1: 2.0}, {2: 1.0}, {3: 3.5}],
                    type=pa.map_(pa.int64(), pa.float32()),
                ),
                "cat_a1": pa.array(
                    [{1: 2.0}, None, {3: 3.5}], type=pa.map_(pa.int64(), pa.float32())
                ),
                "tag_b": pa.array(["4:2.3\x035:2.4", "", "6:2.5"]),
                "tag_b1": pa.array(["4:2.3\x035:2.4", "", "6:2.5"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                "label": pa.array([0, 0, 1], pa.int32()),
            }
        )

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_cat_a_weights = torch.tensor([2.0, 1.0, 3.5], dtype=torch.float32)
        expected_cat_a1_values = torch.tensor([1, 0, 3], dtype=torch.int64)
        expected_cat_a1_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_cat_a1_weights = torch.tensor([2.0, 1.0, 3.5], dtype=torch.float32)
        expected_tag_b_values = torch.tensor([4, 5, 6], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected_tag_b_weights = torch.tensor([2.3, 2.4, 2.5], dtype=torch.float32)
        expected_tag_b1_values = torch.tensor([4, 5, 0, 6], dtype=torch.int64)
        expected_tag_b1_lengths = torch.tensor([2, 1, 1], dtype=torch.int32)
        expected_tag_b1_weights = torch.tensor(
            [2.3, 2.4, 1.0, 2.5], dtype=torch.float32
        )
        expected_int_a_values = torch.tensor([[7], [8], [9]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3, 1, 0], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16], [17], [0]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["cat_a.weights"], expected_cat_a_weights)
        torch.testing.assert_close(data["cat_a1.values"], expected_cat_a1_values)
        torch.testing.assert_close(data["cat_a1.lengths"], expected_cat_a1_lengths)
        torch.testing.assert_close(data["cat_a1.weights"], expected_cat_a1_weights)
        torch.testing.assert_close(data["tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["tag_b.weights"], expected_tag_b_weights)
        torch.testing.assert_close(data["tag_b1.values"], expected_tag_b1_values)
        torch.testing.assert_close(data["tag_b1.lengths"], expected_tag_b1_lengths)
        torch.testing.assert_close(data["tag_b1.weights"], expected_tag_b1_weights)
        torch.testing.assert_close(data["int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["int_b.values"], expected_int_b_values)
        torch.testing.assert_close(
            data["click_seq__cat_a.values"], expected_seq_cat_a_values
        )
        torch.testing.assert_close(
            data["click_seq__cat_a.lengths"], expected_seq_cat_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__int_a.values"], expected_seq_int_a_values
        )
        torch.testing.assert_close(
            data["click_seq__int_a.lengths"], expected_seq_int_a_seq_lengths
        )
        torch.testing.assert_close(data["label"], expected_label)

        expected_dense_feat = KeyedTensor(
            keys=["int_a", "int_b"],
            length_per_key=[1, 2],
            values=torch.tensor(
                [[7, 27, 37], [8, 28, 38], [9, 29, 39]], dtype=torch.float32
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_a1", "tag_b", "tag_b1", "click_seq__cat_a"],
            values=torch.tensor(
                [1, 2, 3, 1, 0, 3, 4, 5, 6, 4, 5, 0, 6, 10, 11, 12, 13]
            ),
            lengths=torch.tensor(
                [1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 1, 3, 1, 0], dtype=torch.int32
            ),
            weights=torch.tensor(
                [
                    2.0,
                    1.0,
                    3.5,
                    2.0,
                    1.0,
                    3.5,
                    2.3,
                    2.4,
                    2.5,
                    2.3,
                    2.4,
                    1.0,
                    2.5,
                    1,
                    1,
                    1,
                    1,
                ],
                dtype=torch.float32,
            ),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        batch = data_parser.to_batch(data)
        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )
        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    def _create_test_fg_feature_cfgs(
        self,
        tag_b_weighted=False,
        tag_b_seq=False,
        with_const=False,
        with_stub_feat=False,
        tag_b_use_weight=True,
        tag_b_pooling="sum",
    ):
        seq_sub_feas = [
            feature_pb2.SeqFeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="f_cat_a",
                    expression="item:cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                )
            ),
            feature_pb2.SeqFeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="f_int_a", expression="item:int_a"
                )
            ),
        ]
        if tag_b_seq:
            seq_sub_feas.append(
                feature_pb2.SeqFeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="f_tag_b",
                        expression="item:tag_b",
                        embedding_dim=8,
                        num_buckets=1000,
                        value_dim=0,
                    )
                ),
            )
        if with_stub_feat:
            seq_sub_feas.append(
                feature_pb2.SeqFeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="f_cat_a_stub",
                        expression="item:cat_a",
                        stub_type=True,
                    )
                )
            )
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="f_cat_a",
                    expression="item:cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="f_tag_b",
                    expression="user:tag_b",
                    embedding_dim=8,
                    num_buckets=1000,
                    weighted=tag_b_weighted,
                    use_weight=tag_b_use_weight,
                    pooling=tag_b_pooling,
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="f_int_a", expression="user:int_a"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="f_int_b", expression="item:int_b", value_dim=2
                )
            ),
            feature_pb2.FeatureConfig(
                lookup_feature=feature_pb2.LookupFeature(
                    feature_name="f_lookup_a", map="user:map_a", key="item:cat_a"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=seq_sub_feas,
                )
            ),
        ]
        if with_const:
            feature_cfgs.append(
                feature_pb2.FeatureConfig(
                    lookup_feature=feature_pb2.LookupFeature(
                        feature_name="f_lookup_b", map="user:map_a", key="const:1"
                    )
                ),
            )
        if with_stub_feat:
            feature_cfgs.append(
                feature_pb2.FeatureConfig(
                    combo_feature=feature_pb2.ComboFeature(
                        feature_name="f_combo_a",
                        expression=["feature:click_seq__f_cat_a_stub", "item:cat_a"],
                        hash_bucket_size=100,
                    )
                ),
            )
        return feature_cfgs

    @parameterized.expand(
        [
            [FgMode.FG_NORMAL, False, False],
            [FgMode.FG_DAG, False, False],
            [FgMode.FG_NORMAL, True, False],
            [FgMode.FG_DAG, True, False],
            [FgMode.FG_NORMAL, False, True],
            [FgMode.FG_DAG, False, True],
        ]
    )
    def test_fg(self, fg_mode, weigted_id, complex_type=False):
        feature_cfgs = self._create_test_fg_feature_cfgs(
            tag_b_weighted=weigted_id, tag_b_seq=True
        )
        features = create_features(feature_cfgs, fg_mode=fg_mode)
        data_parser = DataParser(features=features, labels=["label"])
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)),
            [
                "cat_a",
                "click_seq__cat_a",
                "click_seq__int_a",
                "click_seq__tag_b",
                "int_a",
                "int_b",
                "map_a",
                "tag_b",
            ],
        )
        if complex_type:
            input_data = {
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(
                    [{4: 0.1, 5: 0.2}, {}, {6: 0.3}],
                    type=pa.map_(pa.int64(), pa.float32()),
                )
                if weigted_id
                else pa.array([[4, 5], [], [6]], type=pa.list_(pa.int64())),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(
                    [[27, 37], [28, 38], [29, 39]], type=pa.list_(pa.float32())
                ),
                "map_a": pa.array(
                    [{"1": 0.1, "3": 0.2}, {}, {"1": 0.1, "3": 0.2}],
                    type=pa.map_(pa.string(), pa.float32()),
                ),
                "click_seq__cat_a": pa.array(
                    [[10, 11, 12], [13], []], type=pa.list_(pa.int64())
                ),
                "click_seq__int_a": pa.array(
                    [[14, 15, 16], [17], []], type=pa.list_(pa.float32())
                ),
                "click_seq__tag_b": pa.array(
                    [[["17", "18"], ["19"], ["20", "21"]], [["22"]], [[]]],
                    type=pa.list_(pa.list_(pa.string())),
                ),
                "label": pa.array([0, 0, 1], pa.int32()),
                "__SAMPLE_MASK__": pa.array([True, False, False]),
            }
        else:
            input_data = {
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(["4:0.1\x1d5:0.2", "", "6:0.3"])
                if weigted_id
                else pa.array(["4\x1d5", "", "6"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                "click_seq__tag_b": pa.array(["17\x1d18;19;20\x1d21", "22", ""]),
                "label": pa.array([0, 0, 1], pa.int32()),
                "__SAMPLE_MASK__": pa.array([True, False, False]),
            }

        data = data_parser.parse(input_data=input_data)

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5, 6], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected_tag_b_weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        expected_int_a_values = torch.tensor([[7], [8], [9]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_lookup_a_values = torch.tensor(
            [[0.1], [0.0], [0.2]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12, 13, 0], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16], [17], [0]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_tag_b_values = torch.tensor(
            [17, 18, 19, 20, 21, 22, 0], dtype=torch.int64
        )
        expected_seq_tag_b_key_lengths = torch.tensor(
            [2, 1, 2, 1, 1], dtype=torch.int32
        )
        expected_seq_tag_b_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["f_cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["f_cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["f_tag_b.lengths"], expected_tag_b_lengths)
        if weigted_id:
            tag_b_idx = torch.argsort(data["f_tag_b.values"][:2])
            tag_b_values = torch.cat(
                [data["f_tag_b.values"][tag_b_idx], data["f_tag_b.values"][2:]]
            )
            tag_b_weights = torch.cat(
                [data["f_tag_b.weights"][tag_b_idx], data["f_tag_b.weights"][2:]]
            )
            torch.testing.assert_close(tag_b_values, expected_tag_b_values)
            torch.testing.assert_close(tag_b_weights, expected_tag_b_weights)
        else:
            torch.testing.assert_close(data["f_tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["f_int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["f_int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["f_lookup_a.values"], expected_lookup_a_values)
        torch.testing.assert_close(
            data["click_seq__f_cat_a.values"], expected_seq_cat_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_cat_a.lengths"], expected_seq_cat_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.values"], expected_seq_int_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.lengths"], expected_seq_int_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.values"], expected_seq_tag_b_values
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.key_lengths"], expected_seq_tag_b_key_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.lengths"], expected_seq_tag_b_seq_lengths
        )
        torch.testing.assert_close(data["label"], expected_label)

        batch = data_parser.to_batch(data)

        expected_dense_feat = KeyedTensor(
            keys=["f_int_a", "f_int_b", "f_lookup_a"],
            length_per_key=[1, 2, 1],
            values=torch.tensor(
                [[7, 27, 37, 0.1], [8, 28, 38, 0.0], [9, 29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        if weigted_id:
            expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
                keys=["f_cat_a", "f_tag_b", "click_seq__f_cat_a", "click_seq__f_tag_b"],
                values=torch.tensor(
                    [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 0, 17, 18, 19, 20, 21, 22, 0]
                ),
                lengths=torch.tensor(
                    [1, 1, 1, 2, 0, 1, 3, 1, 1, 5, 1, 1], dtype=torch.int32
                ),
                weights=torch.tensor(
                    [
                        1.0,
                        1.0,
                        1.0,
                        0.1,
                        0.2,
                        0.3,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                ),
            )
            sparse_feat = batch.sparse_features["__BASE__"]
            sparse_feat_values = sparse_feat.values()
            sparse_feat_weights = sparse_feat.weights()
            tag_b_idx = torch.argsort(sparse_feat.values()[3:5]) + 3
            sparse_feat_values = torch.cat(
                [
                    sparse_feat_values[:3],
                    sparse_feat_values[tag_b_idx],
                    sparse_feat_values[5:],
                ]
            )
            sparse_feat_weights = torch.cat(
                [
                    sparse_feat_weights[:3],
                    sparse_feat_weights[tag_b_idx],
                    sparse_feat_weights[5:],
                ]
            )
            sparse_feat = KeyedJaggedTensor(
                keys=sparse_feat.keys(),
                values=sparse_feat_values,
                lengths=sparse_feat.lengths(),
                weights=sparse_feat_weights,
            )
            self.assertTrue(kjt_is_equal(sparse_feat, expected_sparse_feat))
        else:
            expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
                keys=["f_cat_a", "f_tag_b", "click_seq__f_cat_a", "click_seq__f_tag_b"],
                values=torch.tensor(
                    [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 0, 17, 18, 19, 20, 21, 22, 0]
                ),
                lengths=torch.tensor(
                    [1, 1, 1, 2, 0, 1, 3, 1, 1, 5, 1, 1], dtype=torch.int32
                ),
            )
            self.assertTrue(
                kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
            )
        expected_seq_mulval_lengths_user = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__f_tag_b"],
            values=torch.tensor([2, 1, 2, 1, 1], dtype=torch.int32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )

        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )

        self.assertTrue(
            kjt_is_equal(
                batch.sequence_mulval_lengths["__BASE__"],
                expected_seq_mulval_lengths_user,
            )
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__f_int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    def test_fg_encoded_weighted_id_mean_pooling(self):
        # mean pooling in a weighted data group is carried by the per-sample
        # weights, normalized to sum to one per bag, so that pooling stays a
        # plain weighted sum in every lookup kernel.
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_a",
                    embedding_dim=8,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                    pooling="mean",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b",
                    embedding_dim=8,
                    num_buckets=100,
                    pooling="mean",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_c",
                    embedding_dim=8,
                    num_buckets=100,
                )
            ),
        ]
        features = create_features(feature_cfgs)
        data_parser = DataParser(features=features)
        data = data_parser.parse(
            input_data={
                # the last row has an all-zero weight bag, its weight sum is
                # guarded to one so that normalization does not divide by zero
                "tag_a": pa.array(["1:1.0\x032:3.0", "", "3:2.0", "4:0.0\x035:0.0"]),
                "tag_b": pa.array(["1\x032", "", "3", "4"]),
                "tag_c": pa.array(["1\x032", "", "3", "4"]),
            }
        )
        # raw weights are kept in the parsed inputs, normalization is in to_batch
        torch.testing.assert_close(
            data["tag_a.weights"], torch.tensor([1.0, 3.0, 2.0, 0.0, 0.0])
        )

        batch = data_parser.to_batch(data)
        weights = batch.sparse_features["__BASE__"].weights()
        # tag_a: weighted mean -> w / sum(w); tag_b: plain mean -> 1 / length;
        # tag_c: sum pooling -> untouched ones
        torch.testing.assert_close(
            weights,
            torch.tensor(
                # fmt: off
                [
                    0.25,
                    0.75,
                    1.0,
                    0.0,
                    0.0,  # tag_a  w / sum(w), 0/0 -> 0
                    0.5,
                    0.5,
                    1.0,
                    1.0,  # tag_b  1 / length
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # tag_c  untouched ones
                ],
                # fmt: on
            ),
        )

    def _weighted_input_tile_features(self):
        """A weighted mean, a plain mean and a sum feature, user/user/item side."""
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="u_w",
                    expression="user:u_w",
                    embedding_dim=8,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                    pooling="mean",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="u_m",
                    expression="user:u_m",
                    embedding_dim=8,
                    num_buckets=100,
                    pooling="mean",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="i_s",
                    expression="item:i_s",
                    embedding_dim=8,
                    num_buckets=100,
                )
            ),
        ]
        return create_features(feature_cfgs)

    @staticmethod
    def _weighted_input_tile_input_data():
        # the same user in every row, so tiling row 0 is equivalent to the batch
        return {
            "u_w": pa.array(["4:1.0\x035:3.0"] * 3),
            "u_m": pa.array(["1\x032\x033"] * 3),
            "i_s": pa.array(["7", "8", "9"]),
        }

    def test_weighted_id_mean_pooling_input_tile(self):
        # INPUT_TILE=2: user weights are normalized on the untiled bag and only
        # then tiled, so each tiled copy keeps weights summing to one.
        os.environ["INPUT_TILE"] = "2"
        os.environ["INPUT_TILE_3_ONLINE"] = "1"
        data_parser = DataParser(features=self._weighted_input_tile_features())
        data = data_parser.parse(input_data=self._weighted_input_tile_input_data())
        batch = data_parser.to_batch(data)

        sparse_feature = batch.sparse_features["__BASE__"]
        self.assertEqual(sparse_feature.keys(), ["u_w", "u_m", "i_s"])
        torch.testing.assert_close(
            sparse_feature.weights(),
            torch.tensor(
                # fmt: off
                [
                    0.25,
                    0.75,
                    0.25,
                    0.75,
                    0.25,
                    0.75,  # u_w  w / sum(w), tiled
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    1 / 3,
                    1 / 3,  # u_m  1 / length, tiled
                    1.0,
                    1.0,
                    1.0,  # i_s  untouched ones
                ],
                # fmt: on
            ),
        )

    def test_weighted_id_mean_pooling_input_tile_emb(self):
        # INPUT_TILE=3: user features stay at batch_size=1 in their own kjt, the
        # weights are split across the user and the item collection.
        os.environ["INPUT_TILE"] = "3"
        os.environ["INPUT_TILE_3_ONLINE"] = "1"
        data_parser = DataParser(features=self._weighted_input_tile_features())
        data = data_parser.parse(input_data=self._weighted_input_tile_input_data())
        batch = data_parser.to_batch(data)

        user_feature = batch.sparse_features["__BASE___user"]
        self.assertEqual(user_feature.keys(), ["u_w", "u_m"])
        torch.testing.assert_close(
            user_feature.weights(),
            torch.tensor([0.25, 0.75, 1 / 3, 1 / 3, 1 / 3]),
        )
        item_feature = batch.sparse_features["__BASE__"]
        self.assertEqual(item_feature.keys(), ["i_s"])
        torch.testing.assert_close(
            item_feature.weights(), torch.tensor([1.0, 1.0, 1.0])
        )

    @parameterized.expand(
        [
            param("fg_normal", fg_mode=FgMode.FG_NORMAL),
            param("fg_dag", fg_mode=FgMode.FG_DAG),
            param("fg_none", fg_mode=FgMode.FG_NONE),
        ],
        name_func=parameterized_name_func,
    )
    def test_fg_weighted_id_without_use_weight(self, name, fg_mode):
        feature_cfgs = self._create_test_fg_feature_cfgs(
            tag_b_weighted=True, tag_b_use_weight=False
        )
        features = create_features(feature_cfgs, fg_mode=fg_mode)
        data_parser = DataParser(features=features)
        self.assertEqual(len(data_parser.has_weight_keys["__BASE__"]), 0)

        if fg_mode == FgMode.FG_NONE:
            input_data = {
                "f_cat_a": pa.array(["1", "2", "3"]),
                "f_tag_b": pa.array(["4:0.1\x035:0.2", "", "6:0.3"]),
                "f_int_a": pa.array([7, 8, 9], pa.float32()),
                "f_int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                "f_lookup_a": pa.array([0.1, 0.0, 0.2], pa.float32()),
                "click_seq__f_cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__f_int_a": pa.array(["14;15;16", "17", ""]),
            }
        else:
            input_data = {
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(["4:0.1\x1d5:0.2", "", "6:0.3"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
            }

        data = data_parser.parse(input_data=input_data)
        self.assertNotIn("f_tag_b.weights", data)
        torch.testing.assert_close(
            data["f_tag_b.lengths"], torch.tensor([2, 0, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            torch.sort(data["f_tag_b.values"][:2]).values,
            torch.tensor([4, 5], dtype=torch.int64),
        )

        batch = data_parser.to_batch(data)
        self.assertIsNone(batch.sparse_features["__BASE__"].weights_or_none())

    def test_fg_with_const(self):
        feature_cfgs = self._create_test_fg_feature_cfgs(
            tag_b_seq=True, with_const=True
        )
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        data_parser = DataParser(features=features, labels=["label"])
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)),
            [
                "cat_a",
                "click_seq__cat_a",
                "click_seq__int_a",
                "click_seq__tag_b",
                "int_a",
                "int_b",
                "map_a",
                "tag_b",
            ],
        )

        data = data_parser.parse(
            input_data={
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(["4\x1d5", "", "6"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                "click_seq__tag_b": pa.array(["17\x1d18;19;20\x1d21", "22", ""]),
                "label": pa.array([0, 0, 1], pa.int32()),
                "__SAMPLE_MASK__": pa.array([True, False, False]),
            }
        )
        batch = data_parser.to_batch(data)

        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        expected_dense_feat = KeyedTensor(
            keys=["f_int_a", "f_int_b", "f_lookup_a", "f_lookup_b"],
            length_per_key=[1, 2, 1],
            values=torch.tensor(
                [[7, 27, 37, 0.1, 0.1], [8, 28, 38, 0.0, 0.0], [9, 29, 39, 0.2, 0.1]],
                dtype=torch.float32,
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["f_cat_a", "f_tag_b", "click_seq__f_cat_a", "click_seq__f_tag_b"],
            values=torch.tensor(
                [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 0, 17, 18, 19, 20, 21, 22, 0]
            ),
            lengths=torch.tensor(
                [1, 1, 1, 2, 0, 1, 3, 1, 1, 5, 1, 1], dtype=torch.int32
            ),
        )
        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )
        expected_seq_mulval_lengths_user = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__f_tag_b"],
            values=torch.tensor([2, 1, 2, 1, 1], dtype=torch.int32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )

        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )

        self.assertTrue(
            kjt_is_equal(
                batch.sequence_mulval_lengths["__BASE__"],
                expected_seq_mulval_lengths_user,
            )
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__f_int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    def test_fg_with_stub_feature(self):
        feature_cfgs = self._create_test_fg_feature_cfgs(
            tag_b_seq=True, with_stub_feat=True
        )
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        data_parser = DataParser(features=features, labels=["label"])
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)),
            [
                "cat_a",
                "click_seq__cat_a",
                "click_seq__int_a",
                "click_seq__tag_b",
                "int_a",
                "int_b",
                "map_a",
                "tag_b",
            ],
        )

        data = data_parser.parse(
            input_data={
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(["4\x1d5", "", "6"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                "click_seq__tag_b": pa.array(["17\x1d18;19;20\x1d21", "22", ""]),
                "label": pa.array([0, 0, 1], pa.int32()),
                "__SAMPLE_MASK__": pa.array([True, False, False]),
            }
        )
        batch = data_parser.to_batch(data)

        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        expected_dense_feat = KeyedTensor(
            keys=["f_int_a", "f_int_b", "f_lookup_a"],
            length_per_key=[1, 2, 1],
            values=torch.tensor(
                [[7, 27, 37, 0.1], [8, 28, 38, 0.0], [9, 29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "f_cat_a",
                "f_tag_b",
                "click_seq__f_cat_a",
                "click_seq__f_tag_b",
                "f_combo_a",
            ],
            values=torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    10,
                    11,
                    12,
                    13,
                    0,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    0,
                    93,
                    23,
                    25,
                    38,
                    12,
                ]
            ),
            lengths=torch.tensor(
                [1, 1, 1, 2, 0, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1], dtype=torch.int32
            ),
        )
        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )
        expected_seq_mulval_lengths_user = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__f_tag_b"],
            values=torch.tensor([2, 1, 2, 1, 1], dtype=torch.int32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )

        self.assertTrue(
            kjt_is_equal(
                batch.sequence_mulval_lengths["__BASE__"],
                expected_seq_mulval_lengths_user,
            )
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__f_int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    @parameterized.expand(
        [
            param("fg_dag", fg_mode=FgMode.FG_DAG),
            param("fg_bucketize", fg_mode=FgMode.FG_BUCKETIZE),
        ],
        name_func=parameterized_name_func,
    )
    def test_fg_seq_bool_mask_with_expr_mask(self, _, fg_mode):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    sequence_length=10,
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            expr_feature=feature_pb2.ExprFeature(
                                feature_name="is_same_cate",
                                expression="cate == tgt_cate",
                                variables=["item:cate", "item:tgt_cate"],
                                sequence_fields=["cate"],
                                fg_value_type="int64",
                                stub_type=True,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            bool_mask_feature=feature_pb2.BoolMaskFeature(
                                feature_name="masked_iid",
                                expression=[
                                    "item:iid",
                                    "feature:click_seq__is_same_cate",
                                ],
                                embedding_dim=16,
                                num_buckets=1000,
                            )
                        ),
                    ],
                )
            )
        ]
        features = create_features(feature_cfgs, fg_mode=fg_mode)
        data_parser = DataParser(features=features)
        self.assertEqual(data_parser.sparse_keys["__BASE__"], ["click_seq__masked_iid"])
        if fg_mode == FgMode.FG_DAG:
            expected_input_names = ["click_seq__cate", "click_seq__iid", "tgt_cate"]
            input_data = {
                "click_seq__cate": pa.array(["1;2;3;2", "1;1", "2"]),
                "click_seq__iid": pa.array(["11;12;13;14", "21;22", "31"]),
                "tgt_cate": pa.array([2, 1, 2]),
            }
        else:
            # fg does not output the mask, it is a dag intermediate result, we
            # only need the masked ids before bucketize.
            expected_input_names = ["click_seq__masked_iid"]
            input_data = {
                "click_seq__masked_iid": pa.array([["12", "14"], ["21", "22"], ["31"]])
            }
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)), expected_input_names
        )

        data = data_parser.parse(input_data=input_data)
        batch = data_parser.to_batch(data)

        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__masked_iid"],
            values=torch.tensor([12, 14, 21, 22, 31]),
            lengths=torch.tensor([2, 2, 1], dtype=torch.int32),
        )
        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )

    def test_fg_bucketize_only(self):
        feature_cfgs = self._create_test_fg_feature_cfgs(tag_b_seq=True)
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_BUCKETIZE)
        data_parser = DataParser(features=features, labels=["label"])
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)),
            [
                "click_seq__f_cat_a",
                "click_seq__f_int_a",
                "click_seq__f_tag_b",
                "f_cat_a",
                "f_int_a",
                "f_int_b",
                "f_lookup_a",
                "f_tag_b",
            ],
        )

        data = data_parser.parse(
            input_data={
                "f_cat_a": pa.array([["1"], ["2"], ["3"]]),
                "f_tag_b": pa.array([["4", "5"], [], ["6"]]),
                "f_int_a": pa.array([7, 8, 9], pa.float32()),
                "f_int_b": pa.array(
                    [[27, 37], [28, 38], [29, 39]], type=pa.list_(pa.float32())
                ),
                "f_lookup_a": pa.array([0.1, 0.0, 0.2], type=pa.float32()),
                "click_seq__f_cat_a": pa.array([["10", "11", "12"], ["13"], ["0"]]),
                "click_seq__f_int_a": pa.array(
                    [[14, 15, 16], [17], [0]], type=pa.list_(pa.float32())
                ),
                "click_seq__f_tag_b": pa.array(
                    [[["17", "18"], ["19"], ["20", "21"]], [["22"]], [["0"]]]
                ),
                "label": pa.array([0, 0, 1], pa.int32()),
            }
        )

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5, 6], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected_int_a_values = torch.tensor([[7], [8], [9]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_lookup_a_values = torch.tensor(
            [[0.1], [0.0], [0.2]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12, 13, 0], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16], [17], [0]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_tag_b_values = torch.tensor(
            [17, 18, 19, 20, 21, 22, 0], dtype=torch.int64
        )
        expected_seq_tag_b_key_lengths = torch.tensor(
            [2, 1, 2, 1, 1], dtype=torch.int32
        )
        expected_seq_tag_b_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["f_cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["f_cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["f_tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["f_tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["f_int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["f_int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["f_lookup_a.values"], expected_lookup_a_values)
        torch.testing.assert_close(
            data["click_seq__f_cat_a.values"], expected_seq_cat_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_cat_a.lengths"], expected_seq_cat_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.values"], expected_seq_int_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.lengths"], expected_seq_int_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.values"], expected_seq_tag_b_values
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.key_lengths"], expected_seq_tag_b_key_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.lengths"], expected_seq_tag_b_seq_lengths
        )
        torch.testing.assert_close(data["label"], expected_label)

    def test_fg_bucketize_only_with_stub_feat(self):
        feature_cfgs = self._create_test_fg_feature_cfgs(with_stub_feat=True)
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_BUCKETIZE)
        data_parser = DataParser(features=features, labels=["label"])
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)),
            [
                "click_seq__f_cat_a",
                "click_seq__f_int_a",
                "f_cat_a",
                "f_combo_a",
                "f_int_a",
                "f_int_b",
                "f_lookup_a",
                "f_tag_b",
            ],
        )

    @parameterized.expand(
        [
            [
                {
                    "f_cat_a": pa.array([1, 2, 3]),
                    "f_tag_b": pa.array(["4\x035", "", "6"]),
                    "f_int_a": pa.array([7, 8, 9], pa.float32()),
                    "f_int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                    "f_lookup_a": pa.array([0.1, 0.0, 0.2]),
                    "click_seq__f_cat_a": pa.array(["10;11;12", "13", "0"]),
                    "click_seq__f_int_a": pa.array(["14;15;16", "17", "0"]),
                    "click_seq__f_tag_b": pa.array(["17\x0318;19;20\x0321", "22", "0"]),
                    "label": pa.array([0, 0, 1], pa.int32()),
                },
                FgMode.FG_NONE,
            ],
            [
                {
                    "cat_a": pa.array([1, 2, 3]),
                    "tag_b": pa.array(["4\x1d5", "", "6"]),
                    "int_a": pa.array([7, 8, 9], pa.float32()),
                    "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                    "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                    "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                    "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                    "click_seq__tag_b": pa.array(["17\x1d18;19;20\x1d21", "22", ""]),
                    "label": pa.array([0, 0, 1], pa.int32()),
                },
                FgMode.FG_DAG,
            ],
        ]
    )
    def test_input_tile(self, input_data, fg_mode):
        os.environ["INPUT_TILE"] = "2"
        feature_cfgs = self._create_test_fg_feature_cfgs(tag_b_seq=True)
        features = create_features(feature_cfgs, fg_mode=fg_mode)

        data_parser = DataParser(features=features, labels=["label"])

        data = data_parser.parse(input_data=input_data)

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5, 6], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected_int_a_values = torch.tensor([[7], [8], [9]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_lookup_a_values = torch.tensor(
            [[0.1], [0.0], [0.2]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12, 13, 0], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16], [17], [0]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_tag_b_values = torch.tensor(
            [17, 18, 19, 20, 21, 22, 0], dtype=torch.int64
        )
        expected_seq_tag_b_key_lengths = torch.tensor(
            [2, 1, 2, 1, 1], dtype=torch.int32
        )
        expected_seq_tag_b_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["f_cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["f_cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["f_tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["f_tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["f_int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["f_int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["f_lookup_a.values"], expected_lookup_a_values)
        torch.testing.assert_close(
            data["click_seq__f_cat_a.values"], expected_seq_cat_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_cat_a.lengths"], expected_seq_cat_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.values"], expected_seq_int_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.lengths"], expected_seq_int_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.values"], expected_seq_tag_b_values
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.key_lengths"], expected_seq_tag_b_key_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.lengths"], expected_seq_tag_b_seq_lengths
        )
        torch.testing.assert_close(data["label"], expected_label)

        expected_dense_feat_user = KeyedTensor(
            keys=["f_int_a"],
            length_per_key=[1],
            values=torch.tensor(
                [[7], [8], [9]],
                dtype=torch.float32,
            ),
        )
        expected_dense_feat = KeyedTensor(
            keys=["f_int_b", "f_lookup_a"],
            length_per_key=[2, 1],
            values=torch.tensor(
                [[27, 37, 0.1], [28, 38, 0.0], [29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["f_cat_a", "f_tag_b", "click_seq__f_cat_a", "click_seq__f_tag_b"],
            values=torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    10,
                    11,
                    12,
                    13,
                    0,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    0,
                ]
            ),
            lengths=torch.tensor(
                [1, 1, 1, 2, 0, 1, 3, 1, 1, 5, 1, 1], dtype=torch.int32
            ),
        )
        expected_seq_mulval_lengths = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__f_tag_b"],
            values=torch.tensor([2, 1, 2, 1, 1], dtype=torch.int32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        batch = data_parser.to_batch(data)
        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )
        torch.testing.assert_close(
            batch.dense_features["__BASE___user"].values(),
            expected_dense_feat_user.values(),
        )
        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )
        self.assertTrue(
            kjt_is_equal(
                batch.sequence_mulval_lengths["__BASE__"], expected_seq_mulval_lengths
            )
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__f_int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    @parameterized.expand(
        [
            [
                {
                    "f_cat_a": pa.array([1, 2, 3]),
                    "f_tag_b": pa.array(["4\x035", "", "6"]),
                    "f_int_a": pa.array([7, 8, 9], pa.float32()),
                    "f_int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                    "f_lookup_a": pa.array([0.1, 0.0, 0.2]),
                    "click_seq__f_cat_a": pa.array(["10;11;12", "13", "0"]),
                    "click_seq__f_int_a": pa.array(["14;15;16", "17", "0"]),
                    "click_seq__f_tag_b": pa.array(["17\x0318;19;20\x0321", "22", "0"]),
                    "label": pa.array([0, 0, 1], pa.int32()),
                },
                FgMode.FG_NONE,
            ],
            [
                {
                    "cat_a": pa.array([1, 2, 3]),
                    "tag_b": pa.array(["4\x1d5", "", "6"]),
                    "int_a": pa.array([7, 8, 9], pa.float32()),
                    "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                    "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                    "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                    "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                    "click_seq__tag_b": pa.array(["17\x1d18;19;20\x1d21", "22", ""]),
                    "label": pa.array([0, 0, 1], pa.int32()),
                },
                FgMode.FG_DAG,
            ],
        ]
    )
    def test_input_tile_emb(self, input_data, fg_mode):
        os.environ["INPUT_TILE"] = "3"
        feature_cfgs = self._create_test_fg_feature_cfgs(tag_b_seq=True)
        features = create_features(feature_cfgs, fg_mode=fg_mode)

        data_parser = DataParser(features=features, labels=["label"])

        data = data_parser.parse(input_data=input_data)

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5, 6], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        expected_int_a_values = torch.tensor([[7], [8], [9]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_lookup_a_values = torch.tensor(
            [[0.1], [0.0], [0.2]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12, 13, 0], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16], [17], [0]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_seq_tag_b_values = torch.tensor(
            [17, 18, 19, 20, 21, 22, 0], dtype=torch.int64
        )
        expected_seq_tag_b_key_lengths = torch.tensor(
            [2, 1, 2, 1, 1], dtype=torch.int32
        )
        expected_seq_tag_b_seq_lengths = torch.tensor([3, 1, 1], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["f_cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["f_cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["f_tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["f_tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["f_int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["f_int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["f_lookup_a.values"], expected_lookup_a_values)
        torch.testing.assert_close(
            data["click_seq__f_cat_a.values"], expected_seq_cat_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_cat_a.lengths"], expected_seq_cat_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.values"], expected_seq_int_a_values
        )
        torch.testing.assert_close(
            data["click_seq__f_int_a.lengths"], expected_seq_int_a_seq_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.values"], expected_seq_tag_b_values
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.key_lengths"], expected_seq_tag_b_key_lengths
        )
        torch.testing.assert_close(
            data["click_seq__f_tag_b.lengths"], expected_seq_tag_b_seq_lengths
        )
        torch.testing.assert_close(data["label"], expected_label)

        expected_dense_feat_user = KeyedTensor(
            keys=["f_int_a"],
            length_per_key=[1],
            values=torch.tensor(
                [[7], [8], [9]],
                dtype=torch.float32,
            ),
        )
        expected_dense_feat = KeyedTensor(
            keys=["f_int_b", "f_lookup_a"],
            length_per_key=[2, 1],
            values=torch.tensor(
                [[27, 37, 0.1], [28, 38, 0.0], [29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["f_cat_a"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([1, 1, 1], dtype=torch.int32),
        )
        expected_seq_mulval_lengths_user = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__f_tag_b"],
            values=torch.tensor([2, 1, 2, 1, 1], dtype=torch.int32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        expected_sparse_feat_user = KeyedJaggedTensor.from_lengths_sync(
            keys=["f_tag_b", "click_seq__f_cat_a", "click_seq__f_tag_b"],
            values=torch.tensor(
                [4, 5, 6, 10, 11, 12, 13, 0, 17, 18, 19, 20, 21, 22, 0]
            ),
            lengths=torch.tensor([2, 0, 1, 3, 1, 1, 5, 1, 1], dtype=torch.int32),
        )

        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16], [17], [0]], dtype=torch.float32),
            lengths=torch.tensor([3, 1, 1], dtype=torch.int32),
        )
        batch = data_parser.to_batch(data)

        torch.testing.assert_close(
            batch.dense_features["__BASE__"].values(), expected_dense_feat.values()
        )
        torch.testing.assert_close(
            batch.dense_features["__BASE___user"].values(),
            expected_dense_feat_user.values(),
        )

        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE__"], expected_sparse_feat)
        )
        self.assertTrue(
            kjt_is_equal(
                batch.sequence_mulval_lengths["__BASE___user"],
                expected_seq_mulval_lengths_user,
            )
        )
        self.assertTrue(
            kjt_is_equal(
                batch.sparse_features["__BASE___user"], expected_sparse_feat_user
            )
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__f_int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)

    def test_dump_parsed_inputs(self):
        feature_cfgs = self._create_test_fg_feature_cfgs(
            tag_b_weighted=True, tag_b_seq=True
        )
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        data_parser = DataParser(features=features, labels=["label"])

        input_data = {
            "cat_a": pa.array([1, 2, 3]),
            "tag_b": pa.array(["4:0.1\x1d5:0.2", "", "6:0.3"]),
            "int_a": pa.array([7, 8, 9], pa.float32()),
            "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
            "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
            "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
            "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
            "click_seq__tag_b": pa.array(["17\x1d18;19;20\x1d21", "22", ""]),
            "label": pa.array([0, 0, 1], pa.int32()),
            "__SAMPLE_MASK__": pa.array([True, False, False]),
        }
        data = data_parser.parse(input_data=input_data)
        dump_str = data_parser.dump_parsed_inputs(data)
        self.assertEqual(
            dump_str,
            pa.array(
                [
                    "f_cat_a:1 | f_tag_b:4:0.1,5:0.2 | f_int_a:7.0 | f_int_b:27.0,37.0 | f_lookup_a:0.1 | click_seq__f_cat_a:10;11;12 | click_seq__f_int_a:14.0;15.0;16.0 | click_seq__f_tag_b:17,18;19;20,21",  # NOQA
                    "f_cat_a:2 | f_tag_b: | f_int_a:8.0 | f_int_b:28.0,38.0 | f_lookup_a:0.0 | click_seq__f_cat_a:13 | click_seq__f_int_a:17.0 | click_seq__f_tag_b:22",  # NOQA
                    "f_cat_a:3 | f_tag_b:6:0.3 | f_int_a:9.0 | f_int_b:29.0,39.0 | f_lookup_a:0.2 | click_seq__f_cat_a:0 | click_seq__f_int_a:0.0 | click_seq__f_tag_b:0",  # NOQA
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
