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


import unittest

import numpy as np
import pyarrow as pa
from parameterized import parameterized
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType

from tzrec.features import combine_feature as combine_feature_lib
from tzrec.protos import feature_pb2


class CombineFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [["1\x032", "", "3"], [1, 2, 0, 3], [2, 1, 1]],
            [[1, 2, None, 3], [1, 2, 0, 3], [1, 1, 1, 1]],
        ]
    )
    def test_fg_encoded_combine_feature(
        self, input_feat, expected_values, expected_lengths
    ):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=10,
                embedding_dim=16,
                fg_encoded_default_value="0",
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(combine_feat_cfg)
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)
        self.assertEqual(combine_feat.inputs, ["combine_feat"])

        input_data = {"combine_feat": pa.array(input_feat)}
        parsed_feat = combine_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combine_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_combine_feature_dense(self):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                expression="user:combine_input",
                default_value="0",
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(combine_feat_cfg)
        self.assertEqual(combine_feat.output_dim, 1)
        self.assertEqual(combine_feat.is_sparse, False)
        self.assertEqual(combine_feat.emb_bag_config, None)
        self.assertEqual(combine_feat.emb_config, None)

    def test_combine_feature_with_boundary(self):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                boundaries=[-0.5, 0.5, 1.5, 2.5],
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(combine_feat_cfg)
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)
        self.assertEqual(combine_feat.num_embeddings, 5)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=5,
            embedding_dim=16,
            name="combine_feat_emb",
            feature_names=["combine_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(
            repr(combine_feat.emb_bag_config), repr(expected_emb_bag_config)
        )

    def test_combine_feature_with_num_buckets(self):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=100,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(combine_feat_cfg)
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)
        self.assertEqual(combine_feat.num_embeddings, 100)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="combine_feat_emb",
            feature_names=["combine_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(
            repr(combine_feat.emb_bag_config), repr(expected_emb_bag_config)
        )

        # Verify fg_json value_type is int64 for num_buckets
        fg_jsons = combine_feat._fg_json()
        self.assertEqual(fg_jsons[0]["value_type"], "int64")

    def test_combine_feature_with_value_map(self):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=100,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                value_map={"click": 1.0, "buy": 2.0, "cart": 3.0},
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(combine_feat_cfg)
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)

        # Verify value_map is included in fg_json
        fg_jsons = combine_feat._fg_json()
        self.assertEqual(fg_jsons[0]["feature_type"], "combine_feature")
        self.assertIn("value_map", fg_jsons[0])
        self.assertEqual(
            fg_jsons[0]["value_map"], {"click": 1.0, "buy": 2.0, "cart": 3.0}
        )
        self.assertEqual(fg_jsons[0]["num_buckets"], 100)
        self.assertEqual(fg_jsons[0]["value_type"], "int64")

    def test_sequence_combine_feature_with_value_map(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=100,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                value_map={"click": 1.0, "buy": 2.0},
            )
        )
        seq_feat = combine_feature_lib.CombineFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.is_sequence, True)
        self.assertEqual(seq_feat.name, "click_50_seq__combine_feat")
        self.assertEqual(seq_feat.value_dim, 1)

    def test_fg_json_output(self):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=100,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                value_map={"a": 1.0, "b": 2.0},
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(combine_feat_cfg)

        fg_jsons = combine_feat._fg_json()
        self.assertEqual(len(fg_jsons), 1)
        fg_cfg = fg_jsons[0]
        self.assertEqual(fg_cfg["feature_type"], "combine_feature")
        self.assertEqual(fg_cfg["feature_name"], "combine_feat")
        self.assertEqual(fg_cfg["expression"], "user:combine_input")
        self.assertEqual(fg_cfg["num_buckets"], 100)
        self.assertEqual(fg_cfg["value_map"], {"a": 1.0, "b": 2.0})
        self.assertEqual(fg_cfg["value_type"], "int64")
        self.assertEqual(fg_cfg["value_dim"], 1)

    def test_fg_json_sequence_prefix(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=100,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                sequence_length=50,
                sequence_delim=";",
            )
        )
        seq_feat = combine_feature_lib.CombineFeature(
            seq_feat_cfg,
            is_sequence=True,
        )

        # fg_json() should prefix "sequence_" for standalone sequence
        fg_jsons = seq_feat.fg_json()
        self.assertEqual(fg_jsons[0]["feature_type"], "sequence_combine_feature")
        self.assertEqual(fg_jsons[0]["sequence_delim"], ";")
        self.assertEqual(fg_jsons[0]["sequence_length"], 50)


if __name__ == "__main__":
    unittest.main()
