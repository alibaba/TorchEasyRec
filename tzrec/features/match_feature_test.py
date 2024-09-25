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
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType

from tzrec.features import match_feature as match_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class MatchFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [["1\x032", "", "3"], [1, 2, 0, 3], [2, 1, 1]],
            [[1, 2, None, 3], [1, 2, 0, 3], [1, 1, 1, 1]],
        ]
    )
    def test_fg_encoded_match_feature(
        self, input_feat, expected_values, expected_lengths
    ):
        match_feat_cfg = feature_pb2.FeatureConfig(
            match_feature=feature_pb2.MatchFeature(
                feature_name="match_feat",
                num_buckets=10,
                embedding_dim=16,
                fg_encoded_default_value="0",
            )
        )
        match_feat = match_feature_lib.MatchFeature(match_feat_cfg)
        self.assertEqual(match_feat.output_dim, 16)
        self.assertEqual(match_feat.is_sparse, True)
        self.assertEqual(match_feat.inputs, ["match_feat"])

        input_data = {"match_feat": pa.array(input_feat)}
        parsed_feat = match_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "match_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_match_feature_dense(self):
        match_feat_cfg = feature_pb2.FeatureConfig(
            match_feature=feature_pb2.MatchFeature(
                feature_name="match_feat",
                nested_map="user:match_cate_brand",
                pkey="item:cate",
                skey="item:brand",
                default_value="0",
            )
        )
        match_feat = match_feature_lib.MatchFeature(
            match_feat_cfg, fg_mode=FgMode.NORMAL
        )
        self.assertEqual(match_feat.output_dim, 1)
        self.assertEqual(match_feat.is_sparse, False)
        self.assertEqual(match_feat.inputs, ["match_cate_brand", "cate", "brand"])
        self.assertEqual(match_feat.emb_bag_config, None)
        self.assertEqual(match_feat.emb_config, None)

        input_data = {
            "match_cate_brand": pa.array(
                [
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "",
                    "",
                ]
            ),
            "cate": pa.array(["ca", "ca", "", "ca", None]),
            "brand": pa.array(["ba", "", "", "ba", None]),
        }
        parsed_feat = match_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "match_feat")
        self.assertTrue(
            np.allclose(parsed_feat.values, np.array([[1], [0], [0], [0], [0]]))
        )

    @parameterized.expand(
        [
            ["0", [2, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            ["", [2], [1, 0, 0, 0, 0]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_match_feature_with_boundary(
        self, default_value, expected_values, expected_lengths
    ):
        match_feat_cfg = feature_pb2.FeatureConfig(
            match_feature=feature_pb2.MatchFeature(
                feature_name="match_feat",
                boundaries=[-0.5, 0.5, 1.5, 2.5],
                embedding_dim=16,
                nested_map="user:match_cate_brand",
                pkey="item:cate",
                skey="item:brand",
                default_value=default_value,
            )
        )
        match_feat = match_feature_lib.MatchFeature(
            match_feat_cfg, fg_mode=FgMode.NORMAL
        )
        self.assertEqual(match_feat.output_dim, 16)
        self.assertEqual(match_feat.is_sparse, True)
        self.assertEqual(match_feat.inputs, ["match_cate_brand", "cate", "brand"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=5,
            embedding_dim=16,
            name="match_feat_emb",
            feature_names=["match_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(match_feat.emb_bag_config), repr(expected_emb_bag_config))

        input_data = {
            "match_cate_brand": pa.array(
                [
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "",
                    "",
                ]
            ),
            "cate": pa.array(["ca", "ca", "", "ca", None]),
            "brand": pa.array(["ba", "", "", "ba", None]),
        }
        parsed_feat = match_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "match_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", [1], [1, 0, 0, 0, 0]],
            ["0", [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_match_feature_with_num_buckets(
        self, default_value, expected_values, expected_lengths
    ):
        match_feat_cfg = feature_pb2.FeatureConfig(
            match_feature=feature_pb2.MatchFeature(
                feature_name="match_feat",
                num_buckets=10,
                embedding_dim=16,
                nested_map="user:match_cate_brand",
                pkey="item:cate",
                skey="item:brand",
                default_value=default_value,
            )
        )
        match_feat = match_feature_lib.MatchFeature(
            match_feat_cfg, fg_mode=FgMode.NORMAL
        )
        self.assertEqual(match_feat.output_dim, 16)
        self.assertEqual(match_feat.is_sparse, True)
        self.assertEqual(match_feat.inputs, ["match_cate_brand", "cate", "brand"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=10,
            embedding_dim=16,
            name="match_feat_emb",
            feature_names=["match_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(match_feat.emb_bag_config), repr(expected_emb_bag_config))

        input_data = {
            "match_cate_brand": pa.array(
                [
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "",
                    "",
                ]
            ),
            "cate": pa.array(["ca", "ca", "", "ca", None]),
            "brand": pa.array(["ba", "", "", "ba", None]),
        }
        parsed_feat = match_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "match_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", [71], [1, 0, 0, 0, 0]],
            ["z", [71, 54, 54, 54, 54], [1, 1, 1, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_match_feature_with_hash_bucket_size(
        self, default_value, expected_values, expected_lengths
    ):
        match_feat_cfg = feature_pb2.FeatureConfig(
            match_feature=feature_pb2.MatchFeature(
                feature_name="match_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                nested_map="user:match_cate_brand",
                pkey="item:cate",
                skey="item:brand",
                default_value=default_value,
            )
        )
        match_feat = match_feature_lib.MatchFeature(
            match_feat_cfg, fg_mode=FgMode.NORMAL
        )
        self.assertEqual(match_feat.output_dim, 16)
        self.assertEqual(match_feat.is_sparse, True)
        self.assertEqual(match_feat.inputs, ["match_cate_brand", "cate", "brand"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="match_feat_emb",
            feature_names=["match_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(match_feat.emb_bag_config), repr(expected_emb_bag_config))

        input_data = {
            "match_cate_brand": pa.array(
                [
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "ca^ba:1,bb:2|cb^ba:3,bb:4",
                    "",
                    "",
                ]
            ),
            "cate": pa.array(["ca", "ca", "", "ca", None]),
            "brand": pa.array(["ba", "", "", "ba", None]),
        }
        parsed_feat = match_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "match_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
