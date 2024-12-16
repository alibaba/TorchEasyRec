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


import unittest

import numpy as np
import pyarrow as pa
from parameterized import parameterized
from torchrec.modules.embedding_configs import EmbeddingConfig

from tzrec.features import sequence_feature as sequence_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class SequenceIdFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                ["1;2", "", "3", "4\x035;6", None],
                "",
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 2, 1],
                [2, 0, 1, 2, 0],
            ],
            [
                [[[1], [2]], [], [[3]], [[4, 5], [6]], None],
                "",
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 2, 1],
                [2, 0, 1, 2, 0],
            ],
            [
                [[1, 2], [], [3], [4, 6], None],
                "",
                [1, 2, 3, 4, 6],
                [1, 1, 1, 1, 1],
                [2, 0, 1, 2, 0],
            ],
            [
                ["1;2", "", "3", "4\x035;6", None],
                "0",
                [1, 2, 0, 3, 4, 5, 6, 0],
                [1, 1, 1, 1, 2, 1, 1],
                [2, 1, 1, 2, 1],
            ],
            [
                [[[1], [2]], [], [[3]], [[4, 5], [6]], None],
                "0",
                [1, 2, 0, 3, 4, 5, 6, 0],
                [1, 1, 1, 1, 2, 1, 1],
                [2, 1, 1, 2, 1],
            ],
            [
                [[1, 2], [], [3], [4, 6], None],
                "0",
                [1, 2, 0, 3, 4, 6, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [2, 1, 1, 2, 1],
            ],
        ],
    )
    def test_fg_encoded_sequence_id_feature(
        self,
        input_feat,
        default_value,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                fg_encoded_default_value=default_value,
            )
        )
        seq_feat = sequence_feature_lib.SequenceIdFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
        )
        input_data = {"click_50_seq__id_feat": pa.array(input_feat)}
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__id_feat"])

        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))
        np.testing.assert_allclose(
            parsed_feat.seq_lengths, np.array(expected_seq_lengths)
        )

    def test_fg_encoded_simple_sequence_id_feature(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.SequenceIdFeature(
                feature_name="click_50_seq_id",
                sequence_delim=";",
                sequence_length=50,
                embedding_dim=16,
            )
        )
        seq_feat = sequence_feature_lib.SequenceIdFeature(
            seq_feat_cfg,
        )
        input_data = {"click_50_seq_id": pa.array(["1;2", "", "3", "4\x035;6"])}
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq_id"])

        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_id")
        np.testing.assert_allclose(parsed_feat.values, np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1, 1, 1, 2, 1]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 0, 1, 2]))

    @parameterized.expand(
        [
            ["", [33, 44, 66, 26, 66], [2, 1, 1, 1], [2, 1, 1]],
            ["xyz", [33, 44, 66, 13, 66], [2, 1, 1, 1], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_id_feature_with_hash_bucket_size(
        self, default_value, expected_values, expected_lengths, expected_seq_lengths
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                expression="user:id_str",
                default_value=default_value,
            )
        )
        seq_feat = sequence_feature_lib.SequenceIdFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__id_str"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="click_50_seq__id_feat_emb",
            feature_names=["click_50_seq__id_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq__id_str": pa.array(["abc\x1defg;hij", "", "hij"])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    @parameterized.expand(
        [
            ["", [33, 44, 66, 26, 66], [2, 1, 1, 1], [2, 1, 1]],
            ["xyz", [33, 44, 66, 13, 66], [2, 1, 1, 1], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_simple_sequence_id_feature_with_hash_bucket_size(
        self, default_value, expected_values, expected_lengths, expected_seq_lengths
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.SequenceIdFeature(
                feature_name="click_50_seq_id_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                expression="user:click_50_seq_id_str",
                sequence_delim=";",
                sequence_length=50,
                default_value=default_value,
            )
        )
        seq_feat = sequence_feature_lib.SequenceIdFeature(
            seq_feat_cfg,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq_id_str"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="click_50_seq_id_feat_emb",
            feature_names=["click_50_seq_id_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq_id_str": pa.array(["abc\x1defg;hij", "", "hij"])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    @parameterized.expand(
        [
            [
                "",
                ["1;2", "", "3", "4\0355;6"],
                [1, 2, 0, 3, 4, 5, 6],
                [1, 1, 1, 1, 2, 1],
                [2, 1, 1, 2],
            ],
            [
                "",
                [["1", "2"], None, ["3"], ["4\0355", "6"]],
                [1, 2, 0, 3, 4, 5, 6],
                [1, 1, 1, 1, 2, 1],
                [2, 1, 1, 2],
            ],
            [
                "",
                [[1, 2], [], [3], [4, 6]],
                [1, 2, 0, 3, 4, 6],
                [1, 1, 1, 1, 1, 1],
                [2, 1, 1, 2],
            ],
            [
                "0",
                ["1;2", "", "3", "4\0355;6"],
                [1, 2, 0, 3, 4, 5, 6],
                [1, 1, 1, 1, 2, 1],
                [2, 1, 1, 2],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_id_feature_with_num_buckets(
        self,
        default_value,
        input_feat,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                num_buckets=10,
                embedding_dim=16,
                expression="user:id_str",
                default_value=default_value,
            )
        )
        seq_feat = sequence_feature_lib.SequenceIdFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__id_str"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=10,
            embedding_dim=16,
            name="click_50_seq__id_feat_emb",
            feature_names=["click_50_seq__id_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq__id_str": pa.array(input_feat)}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    def test_sequence_id_feature_with_vocab_list(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                vocab_list=["a", "b", "c"],
                embedding_dim=16,
                expression="user:id_str",
                default_value="0",
            )
        )
        seq_feat = sequence_feature_lib.SequenceIdFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim="|",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        input_data = {"click_50_seq__id_str": pa.array(["c||a|b|b|", "", "a|b||c"])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        self.assertTrue(
            np.allclose(parsed_feat.values, np.array([4, 0, 2, 3, 3, 0, 0, 2, 3, 0, 4]))
        )
        self.assertTrue(
            np.allclose(
                parsed_feat.lengths, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            )
        )
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([6, 1, 4]))

    # TODO(hongsheng.jhs): add max sequence length tests.


class SequenceRawFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                ["0.1;0.2", "", "0.3", "0.4;0.5", None],
                "0",
                1,
                [[0.1], [0.2], [0.0], [0.3], [0.4], [0.5], [0.0]],
                [2, 1, 1, 2, 1],
            ],
            [
                [
                    "0.1\x031.1;0.2\x031.2",
                    "",
                    "0.3\x031.3",
                    "0.4\x031.4;0.5\x031.5",
                    None,
                ],
                "0\x030",
                2,
                [
                    [0.1, 1.1],
                    [0.2, 1.2],
                    [0.0, 0.0],
                    [0.3, 1.3],
                    [0.4, 1.4],
                    [0.5, 1.5],
                    [0.0, 0.0],
                ],
                [2, 1, 1, 2, 1],
            ],
            [
                [[0.1, 0.2], [], [0.3], [0.4, 0.5], None],
                "0",
                1,
                [[0.1], [0.2], [0.0], [0.3], [0.4], [0.5], [0.0]],
                [2, 1, 1, 2, 1],
            ],
            [
                [
                    [[0.1, 1.1], [0.2, 1.2]],
                    [],
                    [[0.3, 1.3]],
                    [[0.4, 1.4], [0.5, 1.5]],
                    None,
                ],
                "0\x030",
                2,
                [
                    [0.1, 1.1],
                    [0.2, 1.2],
                    [0.0, 0.0],
                    [0.3, 1.3],
                    [0.4, 1.4],
                    [0.5, 1.5],
                    [0.0, 0.0],
                ],
                [2, 1, 1, 2, 1],
            ],
            [
                ["0.1;0.2", "0.0", "0.3", "0.4;0.5", "0.0"],
                "",
                1,
                [[0.1], [0.2], [0.0], [0.3], [0.4], [0.5], [0.0]],
                [2, 1, 1, 2, 1],
            ],
            [
                [
                    "0.1\x031.1;0.2\x031.2",
                    "0.0\x030.0",
                    "0.3\x031.3",
                    "0.4\x031.4;0.5\x031.5",
                    "0.0\x030.0",
                ],
                "",
                2,
                [
                    [0.1, 1.1],
                    [0.2, 1.2],
                    [0.0, 0.0],
                    [0.3, 1.3],
                    [0.4, 1.4],
                    [0.5, 1.5],
                    [0.0, 0.0],
                ],
                [2, 1, 1, 2, 1],
            ],
            [
                [[0.1, 0.2], [0.0], [0.3], [0.4, 0.5], [0.0]],
                "",
                1,
                [[0.1], [0.2], [0.0], [0.3], [0.4], [0.5], [0.0]],
                [2, 1, 1, 2, 1],
            ],
            [
                [
                    [[0.1, 1.1], [0.2, 1.2]],
                    [[0.0, 0.0]],
                    [[0.3, 1.3]],
                    [[0.4, 1.4], [0.5, 1.5]],
                    [[0.0, 0.0]],
                ],
                "",
                2,
                [
                    [0.1, 1.1],
                    [0.2, 1.2],
                    [0.0, 0.0],
                    [0.3, 1.3],
                    [0.4, 1.4],
                    [0.5, 1.5],
                    [0.0, 0.0],
                ],
                [2, 1, 1, 2, 1],
            ],
        ]
    )
    def test_fg_encoded_sequence_raw_feature(
        self,
        input_feat,
        default_value,
        value_dim,
        expected_values,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                value_dim=value_dim,
                fg_encoded_default_value=default_value,
            )
        )
        seq_feat = sequence_feature_lib.SequenceRawFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
        )
        input_data = {"click_50_seq__raw_feat": pa.array(input_feat)}
        self.assertEqual(seq_feat.output_dim, value_dim)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__raw_feat"])

        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__raw_feat")
        self.assertTrue(np.allclose(parsed_feat.values, np.array(expected_values)))
        np.testing.assert_allclose(
            parsed_feat.seq_lengths, np.array(expected_seq_lengths)
        )

    def test_fg_encoded_simple_sequence_raw_feature(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_raw_feature=feature_pb2.SequenceRawFeature(
                feature_name="click_50_seq_raw",
                sequence_delim=";",
                sequence_length=50,
                fg_encoded_default_value="0",
            )
        )
        seq_feat = sequence_feature_lib.SequenceRawFeature(
            seq_feat_cfg,
        )
        input_data = {"click_50_seq_raw": pa.array(["0.1;0.2", "", "0.3", "0.4;0.5"])}
        self.assertEqual(seq_feat.output_dim, 1)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["click_50_seq_raw"])

        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_raw")
        self.assertTrue(
            np.allclose(
                parsed_feat.values, np.array([[0.1], [0.2], [0.0], [0.3], [0.4], [0.5]])
            )
        )
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1, 1, 2]))

    @parameterized.expand(
        [
            [["0.1;0.2", "", "0.3"], "", 1, [[0.1], [0.2], [0.3]], [2, 0, 1]],
            [["0.1;0.2", "", "0.3"], "0.0", 1, [[0.1], [0.2], [0.0], [0.3]], [2, 1, 1]],
            [[[0.1, 0.2], None, [0.3]], "", 1, [[0.1], [0.2], [0.3]], [2, 0, 1]],
            [
                ["0.1\x1d1.1;0.2\x1d1.2", "", "0.3\x1d1.3"],
                "",
                2,
                [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
                [2, 0, 1],
            ],
            [
                ["0.1\x1d1.1;0.2\x1d1.2", "", "0.3\x1d1.3"],
                "0.0\x1d0.0",
                2,
                [[0.1, 1.1], [0.2, 1.2], [0.0, 0.0], [0.3, 1.3]],
                [2, 1, 1],
            ],
            [
                [["0.1\x1d1.1", "0.2\x1d1.2"], None, ["0.3\x1d1.3"]],
                "",
                2,
                [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
                [2, 0, 1],
            ],
            [
                [[[0.1, 1.1], [0.2, 1.2]], None, [[0.3, 1.3]]],
                "",
                2,
                [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
                [2, 0, 1],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_sequence_raw_feature_dense(
        self,
        input_feat,
        default_value,
        value_dim,
        expected_values,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                expression="user:raw_input",
                default_value=default_value,
                value_dim=value_dim,
            )
        )
        seq_feat = sequence_feature_lib.SequenceRawFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, value_dim)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__raw_input"])
        self.assertEqual(seq_feat.emb_config, None)

        input_data = {"click_50_seq__raw_input": pa.array(input_feat)}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    @parameterized.expand(
        [
            [["0.1;0.2", "", "0.3"], "0.0", 1, [[0.1], [0.2], [0.0], [0.3]], [2, 1, 1]],
            [
                ["0.1\x1d1.1;0.2\x1d1.2", "", "0.3\x1d1.3"],
                "",
                2,
                [[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
                [2, 0, 1],
            ],
        ]
    )
    def test_simple_sequence_sequence_raw_feature_dense(
        self,
        input_feat,
        default_value,
        value_dim,
        expected_values,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_raw_feature=feature_pb2.SequenceRawFeature(
                feature_name="click_50_seq_raw_feat",
                expression="user:click_50_seq_raw_input",
                default_value=default_value,
                value_dim=value_dim,
                sequence_delim=";",
                sequence_length=50,
            )
        )
        seq_feat = sequence_feature_lib.SequenceRawFeature(
            seq_feat_cfg,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, value_dim)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["click_50_seq_raw_input"])
        self.assertEqual(seq_feat.emb_config, None)

        input_data = {"click_50_seq_raw_input": pa.array(input_feat)}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    @parameterized.expand(
        [
            [["0.1;0.2", "", "0.3"], "", 1, [1, 2, 3], [2, 0, 1]],
            [["0.1;0.2", "", "0.3"], "0.0", 1, [1, 2, 0, 3], [2, 1, 1]],
            [
                ["0.1\x1d1.1;0.2\x1d1.2", "", "0.3\x1d1.3"],
                "",
                2,
                [1, 3, 2, 3, 3, 3],
                [2, 0, 1],
            ],
            [
                ["0.1\x1d1.1;0.2\x1d1.2", "", "0.3\x1d1.3"],
                "0.0\x1d0.0",
                2,
                [1, 3, 2, 3, 0, 0, 3, 3],
                [2, 1, 1],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_sequence_raw_feature_with_boundaries(
        self,
        input_feat,
        default_value,
        value_dim,
        expected_values,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                embedding_dim=16,
                expression="user:raw_input",
                boundaries=[0.05, 0.15, 0.25],
                default_value=default_value,
                value_dim=value_dim,
            )
        )
        seq_feat = sequence_feature_lib.SequenceRawFeature(
            seq_feat_cfg,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__raw_input"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="click_50_seq__raw_feat_emb",
            feature_names=["click_50_seq__raw_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq__raw_input": pa.array(input_feat)}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    # TODO(hongsheng.jhs): add normalizer tests.


if __name__ == "__main__":
    unittest.main()
