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
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)

from tzrec.features import overlap_feature as overlap_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


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


if __name__ == "__main__":
    unittest.main()
