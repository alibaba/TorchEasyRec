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
from parameterized import parameterized
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


class DataParserTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("INPUT_TILE", None)

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

    def test_fg_encoded_id_with_weight(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a1",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    fg_encoded_default_value="0",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b",
                    embedding_dim=8,
                    num_buckets=1000,
                    weighted=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b1",
                    embedding_dim=8,
                    num_buckets=1000,
                    weighted=True,
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
                "cat_a__values": pa.array([1, 2, 3]),
                "cat_a__weights": pa.array([2.0, 1.0, 3.5]),
                "cat_a1__values": pa.array([1, None, 3]),
                "cat_a1__weights": pa.array([2.0, None, 3.5]),
                "tag_b__values": pa.array(["4\x035", "", "6"]),
                "tag_b__weights": pa.array(["2.3\x032.4", "", "2.5"]),
                "tag_b1__values": pa.array(["4\x035", "", "6"]),
                "tag_b1__weights": pa.array(["2.3\x032.4", "", "2.5"]),
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

    def _create_test_fg_feature_cfgs(self, tag_b_weighted=False):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    expression="item:cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="tag_b",
                    expression="user:tag_b",
                    embedding_dim=8,
                    num_buckets=1000,
                    weighted=tag_b_weighted,
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="int_a", expression="user:int_a"
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="int_b", expression="item:int_b", value_dim=2
                )
            ),
            feature_pb2.FeatureConfig(
                lookup_feature=feature_pb2.LookupFeature(
                    feature_name="lookup_a", map="user:map_a", key="item:cat_a"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_a",
                                expression="item:cat_a",
                                embedding_dim=16,
                                num_buckets=100,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="int_a", expression="item:int_a"
                            )
                        ),
                    ],
                )
            ),
        ]
        return feature_cfgs

    @parameterized.expand(
        [
            [FgMode.FG_NORMAL, False],
            [FgMode.FG_DAG, False],
            [FgMode.FG_NORMAL, True],
            [FgMode.FG_DAG, True],
        ]
    )
    def test_fg(self, fg_mode, weigted_id):
        feature_cfgs = self._create_test_fg_feature_cfgs(tag_b_weighted=weigted_id)
        features = create_features(feature_cfgs, fg_mode=fg_mode)
        data_parser = DataParser(features=features, labels=["label"])
        self.assertEqual(
            sorted(list(data_parser.feature_input_names)),
            [
                "cat_a",
                "click_seq__cat_a",
                "click_seq__int_a",
                "int_a",
                "int_b",
                "map_a",
                "tag_b",
            ],
        )

        data = data_parser.parse(
            input_data={
                "cat_a": pa.array([1, 2, 3]),
                "tag_b": pa.array(["4:0.1\x1d5:0.2", "", "6:0.3"])
                if weigted_id
                else pa.array(["4\x1d5", "", "6"]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array(["27\x1d37", "28\x1d38", "29\x1d39"]),
                "map_a": pa.array(["1:0.1\x1d3:0.2", "", "1:0.1\x1d3:0.2"]),
                "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
                "label": pa.array([0, 0, 1], pa.int32()),
                "__SAMPLE_MASK__": pa.array([True, False, False]),
            }
        )

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
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["tag_b.lengths"], expected_tag_b_lengths)
        if weigted_id:
            torch.testing.assert_close(data["tag_b.weights"], expected_tag_b_weights)
        torch.testing.assert_close(data["int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["lookup_a.values"], expected_lookup_a_values)
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
            keys=["int_a", "int_b", "lookup_a"],
            length_per_key=[1, 2, 1],
            values=torch.tensor(
                [[7, 27, 37, 0.1], [8, 28, 38, 0.0], [9, 29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        if weigted_id:
            expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
                keys=["cat_a", "tag_b", "click_seq__cat_a"],
                values=torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 0]),
                lengths=torch.tensor([1, 1, 1, 2, 0, 1, 3, 1, 1], dtype=torch.int32),
                weights=torch.tensor(
                    [1.0, 1.0, 1.0, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0]
                ),
            )
        else:
            expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
                keys=["cat_a", "tag_b", "click_seq__cat_a"],
                values=torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 0]),
                lengths=torch.tensor([1, 1, 1, 2, 0, 1, 3, 1, 1], dtype=torch.int32),
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

    def test_fg_bucketize_only(self):
        feature_cfgs = self._create_test_fg_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_BUCKETIZE)
        data_parser = DataParser(features=features, labels=["label"])
        data = data_parser.parse(
            input_data={
                "cat_a": pa.array([["1"], ["2"], ["3"]]),
                "tag_b": pa.array([["4", "5"], [], ["6"]]),
                "int_a": pa.array([7, 8, 9], pa.float32()),
                "int_b": pa.array([[27, 37], [28, 38], [29, 39]]),
                "lookup_a": pa.array([0.1, 0.0, 0.2], type=pa.float32()),
                "click_seq__cat_a": pa.array([["10", "11", "12"], ["13"], ["0"]]),
                "click_seq__int_a": pa.array([["14", "15", "16"], ["17"], ["0"]]),
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
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["lookup_a.values"], expected_lookup_a_values)
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

    @parameterized.expand(
        [
            [
                {
                    "cat_a": pa.array([1, 2, 3]),
                    "tag_b": pa.array(["4\x035", "", "6"]),
                    "int_a": pa.array([7, 8, 9], pa.float32()),
                    "int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                    "lookup_a": pa.array([0.1, 0.0, 0.2]),
                    "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                    "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
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
                    "label": pa.array([0, 0, 1], pa.int32()),
                },
                FgMode.FG_DAG,
            ],
        ]
    )
    def test_input_tile(self, input_data, fg_mode):
        os.environ["INPUT_TILE"] = "2"
        feature_cfgs = self._create_test_fg_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=fg_mode)

        data_parser = DataParser(features=features, labels=["label"])

        data = data_parser.parse(input_data=input_data)

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2], dtype=torch.int32)
        expected_int_a_values = torch.tensor([[7]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_lookup_a_values = torch.tensor(
            [[0.1], [0.0], [0.2]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["lookup_a.values"], expected_lookup_a_values)
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

        expected_dense_feat_user = KeyedTensor(
            keys=["int_a"],
            length_per_key=[1],
            values=torch.tensor(
                [[7]],
                dtype=torch.float32,
            ),
        )
        expected_dense_feat = KeyedTensor(
            keys=["int_b", "lookup_a"],
            length_per_key=[2, 1],
            values=torch.tensor(
                [[27, 37, 0.1], [28, 38, 0.0], [29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "tag_b", "click_seq__cat_a"],
            values=torch.tensor(
                [1, 2, 3, 4, 5, 4, 5, 4, 5, 10, 11, 12, 10, 11, 12, 10, 11, 12]
            ),
            lengths=torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32),
        )
        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16]], dtype=torch.float32),
            lengths=torch.tensor([3], dtype=torch.int32),
        )
        batch = data_parser.to_batch(data)
        torch.testing.assert_close(
            batch.dense_features["__BASE___item"].values(), expected_dense_feat.values()
        )
        torch.testing.assert_close(
            batch.dense_features["__BASE___user"].values(),
            expected_dense_feat_user.values(),
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

    @parameterized.expand(
        [
            [
                {
                    "cat_a": pa.array([1, 2, 3]),
                    "tag_b": pa.array(["4\x035", "", "6"]),
                    "int_a": pa.array([7, 8, 9], pa.float32()),
                    "int_b": pa.array(["27\x0337", "28\x0338", "29\x0339"]),
                    "lookup_a": pa.array([0.1, 0.0, 0.2]),
                    "click_seq__cat_a": pa.array(["10;11;12", "13", ""]),
                    "click_seq__int_a": pa.array(["14;15;16", "17", ""]),
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
                    "label": pa.array([0, 0, 1], pa.int32()),
                },
                FgMode.FG_DAG,
            ],
        ]
    )
    def test_input_tile_emb(self, input_data, fg_mode):
        os.environ["INPUT_TILE"] = "3"
        feature_cfgs = self._create_test_fg_feature_cfgs()
        features = create_features(feature_cfgs, fg_mode=fg_mode)

        data_parser = DataParser(features=features, labels=["label"])

        data = data_parser.parse(input_data=input_data)

        expected_cat_a_values = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected_cat_a_lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        expected_tag_b_values = torch.tensor([4, 5], dtype=torch.int64)
        expected_tag_b_lengths = torch.tensor([2], dtype=torch.int32)
        expected_int_a_values = torch.tensor([[7]], dtype=torch.float32)
        expected_int_b_values = torch.tensor(
            [[27, 37], [28, 38], [29, 39]], dtype=torch.float32
        )
        expected_lookup_a_values = torch.tensor(
            [[0.1], [0.0], [0.2]], dtype=torch.float32
        )
        expected_seq_cat_a_values = torch.tensor([10, 11, 12], dtype=torch.int64)
        expected_seq_cat_a_seq_lengths = torch.tensor([3], dtype=torch.int32)
        expected_seq_int_a_values = torch.tensor(
            [[14], [15], [16]], dtype=torch.float32
        )
        expected_seq_int_a_seq_lengths = torch.tensor([3], dtype=torch.int32)
        expected_label = torch.tensor([0, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(data["cat_a.values"], expected_cat_a_values)
        torch.testing.assert_close(data["cat_a.lengths"], expected_cat_a_lengths)
        torch.testing.assert_close(data["tag_b.values"], expected_tag_b_values)
        torch.testing.assert_close(data["tag_b.lengths"], expected_tag_b_lengths)
        torch.testing.assert_close(data["int_a.values"], expected_int_a_values)
        torch.testing.assert_close(data["int_b.values"], expected_int_b_values)
        torch.testing.assert_close(data["lookup_a.values"], expected_lookup_a_values)
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

        expected_dense_feat_user = KeyedTensor(
            keys=["int_a"],
            length_per_key=[1],
            values=torch.tensor(
                [[7]],
                dtype=torch.float32,
            ),
        )
        expected_dense_feat = KeyedTensor(
            keys=["int_b", "lookup_a"],
            length_per_key=[2, 1],
            values=torch.tensor(
                [[27, 37, 0.1], [28, 38, 0.0], [29, 39, 0.2]],
                dtype=torch.float32,
            ),
        )
        expected_sparse_feat = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([1, 1, 1], dtype=torch.int32),
        )

        expected_sparse_feat_user = KeyedJaggedTensor.from_lengths_sync(
            keys=["tag_b", "click_seq__cat_a"],
            values=torch.tensor([4, 5, 10, 11, 12]),
            lengths=torch.tensor([2, 3], dtype=torch.int32),
        )

        expected_seq_dense_feat = JaggedTensor(
            values=torch.tensor([[14], [15], [16]], dtype=torch.float32),
            lengths=torch.tensor([3], dtype=torch.int32),
        )
        batch = data_parser.to_batch(data)

        torch.testing.assert_close(
            batch.dense_features["__BASE___item"].values(), expected_dense_feat.values()
        )
        torch.testing.assert_close(
            batch.dense_features["__BASE___user"].values(),
            expected_dense_feat_user.values(),
        )

        self.assertTrue(
            kjt_is_equal(batch.sparse_features["__BASE___item"], expected_sparse_feat)
        )
        self.assertTrue(
            kjt_is_equal(
                batch.sparse_features["__BASE___user"], expected_sparse_feat_user
            )
        )
        self.assertTrue(
            jt_is_equal(
                batch.sequence_dense_features["click_seq__int_a"],
                expected_seq_dense_feat,
            )
        )
        torch.testing.assert_close(batch.labels["label"], expected_label)


if __name__ == "__main__":
    unittest.main()
