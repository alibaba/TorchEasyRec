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
from parameterized import param, parameterized
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)

from tzrec.features import overlap_feature as overlap_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class OverlapFeatureTest(unittest.TestCase):
    def test_fg_encoded_overlap_feature_dense(self):
        overlap_feat_cfg = feature_pb2.FeatureConfig(
            overlap_feature=feature_pb2.OverlapFeature(
                feature_name="overlap_feat",
            )
        )
        overlap_feat = overlap_feature_lib.OverlapFeature(overlap_feat_cfg)
        self.assertEqual(overlap_feat.output_dim, 1)
        self.assertEqual(overlap_feat.is_sparse, False)
        self.assertEqual(overlap_feat.inputs, ["overlap_feat"])
        input_data = {"overlap_feat": pa.array(["0.0", "0.2", "0.3"])}
        parsed_feat = overlap_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "overlap_feat")
        self.assertTrue(
            np.allclose(parsed_feat.values, np.array([[0.0], [0.2], [0.3]]))
        )

    def test_fg_encoded_overlap_feature_sparse(self):
        overlap_feat_cfg = feature_pb2.FeatureConfig(
            overlap_feature=feature_pb2.OverlapFeature(
                feature_name="overlap_feat",
                embedding_dim=16,
                boundaries=[0.1, 0.2, 0.3],
            )
        )
        overlap_feat = overlap_feature_lib.OverlapFeature(overlap_feat_cfg)
        self.assertEqual(overlap_feat.output_dim, 16)
        self.assertEqual(overlap_feat.is_sparse, True)
        self.assertEqual(overlap_feat.inputs, ["overlap_feat"])
        input_data = {"overlap_feat": pa.array([1, 2, 0, 3])}
        parsed_feat = overlap_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "overlap_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([1, 2, 0, 3]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1, 1, 1, 1]))

    @parameterized.expand(
        [
            [
                ["abc\x1defg", "abc\x1defg", "", None],
                ["abc\x1dqwe\x1drty\x1duio", "abc\x1defg\x1drty", "", None],
                "query_common_ratio",
                [[0.5], [1.0], [0.0], [0.0]],
            ],
            [
                [["abc", "efg"], ["abc", "efg"], [], None],
                [["abc", "qwe", "rty", "uio"], ["abc", "efg", "rty"], [], None],
                "query_common_ratio",
                [[0.5], [1.0], [0.0], [0.0]],
            ],
            [
                ["abc\x1defg", "abc\x1defg", "", None],
                ["abc\x1dqwe\x1drty\x1duio", "abc", "", None],
                "title_common_ratio",
                [[0.25], [1.0], [0.0], [0.0]],
            ],
            [
                ["abc\x1defg", "abc\x1defg", "", None],
                ["abc\x1dqwe\x1drty\x1duio", "abc\x1defg\x1drty", "", None],
                "is_contain",
                [[0.0], [1.0], [0.0], [0.0]],
            ],
            [
                ["abc\x1defg", "abc\x1defg", "", None],
                ["abc\x1defg\x1drty", "abc\x1defg", "", None],
                "is_equal",
                [[0.0], [1.0], [0.0], [0.0]],
            ],
            [
                [["abc", "defg"], ["abc", "efg"], [], None],
                [["abc", "qwe", "rty", "uio"], ["abc", "efg", "rty"], [], None],
                "query_common_ratio",
                [[0.5], [1.0], [0.0], [0.0]],
            ],
        ]
    )
    def test_overlap_feature_dense(
        self, query_input, title_input, method, expected_values
    ):
        overlap_feat_cfg = feature_pb2.FeatureConfig(
            overlap_feature=feature_pb2.OverlapFeature(
                feature_name="overlap_feat",
                query="user:query",
                title="item:title",
                method=method,
            )
        )
        overlap_feat = overlap_feature_lib.OverlapFeature(
            overlap_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(overlap_feat.output_dim, 1)
        self.assertEqual(overlap_feat.is_sparse, False)
        self.assertEqual(overlap_feat.inputs, ["query", "title"])
        self.assertEqual(overlap_feat.emb_bag_config, None)
        self.assertEqual(overlap_feat.emb_config, None)
        input_data = {"query": pa.array(query_input), "title": pa.array(title_input)}
        parsed_feat = overlap_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "overlap_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [
                ["abc\x1defg", "abc\x1defg", "", None],
                ["abc\x1dqwe\x1drty\x1duio", "abc\x1defg\x1drty", "", None],
                "query_common_ratio",
                [1, 2, 0, 0],
                [1, 1, 1, 1],
            ],
        ]
    )
    def test_overlap_feature_with_boundaries(
        self, query_input, title_input, method, expected_values, expected_lengths
    ):
        overlap_feat_cfg = feature_pb2.FeatureConfig(
            overlap_feature=feature_pb2.OverlapFeature(
                feature_name="overlap_feat",
                query="user:query",
                title="item:title",
                method=method,
                embedding_dim=16,
                boundaries=[0.25, 0.75],
            )
        )
        overlap_feat = overlap_feature_lib.OverlapFeature(
            overlap_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(overlap_feat.output_dim, 16)
        self.assertEqual(overlap_feat.is_sparse, True)
        self.assertEqual(overlap_feat.inputs, ["query", "title"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=3,
            embedding_dim=16,
            name="overlap_feat_emb",
            feature_names=["overlap_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(
            repr(overlap_feat.emb_bag_config), repr(expected_emb_bag_config)
        )
        expected_emb_config = EmbeddingConfig(
            num_embeddings=3,
            embedding_dim=16,
            name="overlap_feat_emb",
            feature_names=["overlap_feat"],
        )
        self.assertEqual(repr(overlap_feat.emb_config), repr(expected_emb_config))
        input_data = {"query": pa.array(query_input), "title": pa.array(title_input)}
        parsed_feat = overlap_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "overlap_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_overlap_feature_with_invalid_method(self):
        overlap_feat_cfg = feature_pb2.FeatureConfig(
            overlap_feature=feature_pb2.OverlapFeature(
                feature_name="overlap_feat",
                query="user:query",
                title="item:title",
                method="is_common",
            )
        )
        overlap_feat = overlap_feature_lib.OverlapFeature(overlap_feat_cfg)
        with self.assertRaisesRegex(ValueError, "invalid method"):
            overlap_feat.fg_json()

    @parameterized.expand(
        [
            ["is_equal", 0.0],
            ["is_contain", 1.0],
            ["index_of", 1.0],
            ["query_common_ratio", 1.0],
            ["title_common_ratio", 0.5],
            ["proximity_min_dist", 1.0],
            ["proximity_max_dist", 1.0],
            ["proximity_avg_dist", 1.0],
            ["proximity_min_cover", 2.0],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_overlap_feature_methods(self, method, expected_value):
        # every method in FG_OVERLAP_METHODS should be honored by fg, an entry
        # fg does not know would make it return the default value for all rows
        overlap_feat_cfg = feature_pb2.FeatureConfig(
            overlap_feature=feature_pb2.OverlapFeature(
                feature_name="overlap_feat",
                query="user:query",
                title="item:title",
                method=method,
            )
        )
        overlap_feat = overlap_feature_lib.OverlapFeature(
            overlap_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        input_data = {
            "query": pa.array([["b", "c"]]),
            "title": pa.array([["a", "b", "c", "d"]]),
        }
        parsed_feat = overlap_feat.parse(input_data)
        np.testing.assert_allclose(parsed_feat.values, np.array([[expected_value]]))


class SequenceOverlapFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("item_side_seq", title="item:title", sequence_fields=[]),
            param("user_side_seq", title="user:title", sequence_fields=["title"]),
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_simple_sequence_overlap_feature_dense(self, name, title, sequence_fields):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_overlap_feature=feature_pb2.OverlapFeature(
                feature_name="click_50_seq_overlap_feat",
                sequence_delim=";",
                sequence_length=50,
                query="user:query",
                title=title,
                sequence_fields=sequence_fields,
                method="query_common_ratio",
            )
        )
        seq_feat = overlap_feature_lib.OverlapFeature(
            seq_feat_cfg, is_sequence=True, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(seq_feat.output_dim, 1)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["query", "title"])
        self.assertEqual(seq_feat.sequence_input_names, ["title"])
        self.assertEqual(
            seq_feat.fg_json()[0].get("sequence_fields"), sequence_fields or None
        )

        input_data = {
            "query": pa.array(["abc\x1defg", "abc\x1defg"]),
            "title": pa.array(["abc\x1drty;abc\x1defg", "qwe"]),
        }
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_overlap_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([[0.5], [1.0], [0.0]]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1]))


if __name__ == "__main__":
    unittest.main()
