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
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)

from tzrec.features import combine_feature as combine_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class CombineFeatureTest(unittest.TestCase):
    def test_fg_encoded_combine_feature(self):
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

        input_data = {"combine_feat": pa.array([1, 2, None, 3])}
        parsed_feat = combine_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combine_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([1, 2, 0, 3]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1, 1, 1, 1]))

    @parameterized.expand(
        [
            ["sum"],
            ["mean"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_combine_feature_dense(self, combiner):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                expression="user:combine_input",
                default_value="0",
                combiner=combiner,
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(
            combine_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(combine_feat.output_dim, 1)
        self.assertEqual(combine_feat.is_sparse, False)
        self.assertEqual(combine_feat.inputs, ["combine_input"])
        self.assertEqual(combine_feat.emb_bag_config, None)
        self.assertEqual(combine_feat.emb_config, None)

        input_data = {
            "combine_input": pa.array(["1.0\x1d2.0", "3.0", "", None]),
        }
        parsed_feat = combine_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combine_feat")

    @parameterized.expand(
        [
            ["sum"],
            ["mean"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_combine_feature_with_boundary(self, combiner):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                boundaries=[-0.5, 0.5, 1.5, 2.5],
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                combiner=combiner,
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(
            combine_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)
        self.assertEqual(combine_feat.num_embeddings, 5)
        self.assertEqual(combine_feat.inputs, ["combine_input"])
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

        input_data = {
            "combine_input": pa.array(["1.0\x1d2.0", "0.0", "", None]),
        }
        parsed_feat = combine_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combine_feat")

    @parameterized.expand(
        [
            ["sum"],
            ["mean"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_combine_feature_with_num_buckets(self, combiner):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=10,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                combiner=combiner,
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(
            combine_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)
        self.assertEqual(combine_feat.num_embeddings, 10)
        self.assertEqual(combine_feat.inputs, ["combine_input"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=10,
            embedding_dim=16,
            name="combine_feat_emb",
            feature_names=["combine_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(
            repr(combine_feat.emb_bag_config), repr(expected_emb_bag_config)
        )

        input_data = {
            "combine_input": pa.array(["1\x1d2", "3", "", None]),
        }
        parsed_feat = combine_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combine_feat")

    @parameterized.expand(
        [
            ["sum"],
            ["mean"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_combine_feature_with_value_map(self, combiner):
        combine_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=10,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                value_map={"tag1": 1.0, "tag2": 2.0, "tag3": 3.0},
                combiner=combiner,
            )
        )
        combine_feat = combine_feature_lib.CombineFeature(
            combine_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(combine_feat.output_dim, 16)
        self.assertEqual(combine_feat.is_sparse, True)
        self.assertEqual(combine_feat.inputs, ["combine_input"])

        input_data = {
            "combine_input": pa.array(["tag1\x1dtag2", "tag3", "", None]),
        }
        parsed_feat = combine_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combine_feat")

    @parameterized.expand(
        [
            ["sum"],
            ["mean"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_combine_feature_with_value_map(self, combiner):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            combine_feature=feature_pb2.CombineFeature(
                feature_name="combine_feat",
                num_buckets=10,
                embedding_dim=16,
                expression="user:combine_input",
                default_value="0",
                value_map={"tag1": 1.0, "tag2": 2.0},
                combiner=combiner,
            )
        )
        seq_feat = combine_feature_lib.CombineFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.is_sequence, True)
        self.assertEqual(seq_feat.name, "click_50_seq__combine_feat")
        self.assertEqual(seq_feat.value_dim, 1)

        input_data = {
            "click_50_seq__combine_input": pa.array(["tag1\x1dtag2;tag1", "tag2", ""]),
        }
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__combine_feat")

    @parameterized.expand(
        [
            ["", [1, 0, 0], [1, 1, 1], [1, 1, 1]],
            ["0", [1, 0, 0], [1, 1, 1], [1, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_simple_sequence_combine_feature(
        self,
        default_value,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_combine_feature=feature_pb2.CombineFeature(
                feature_name="seq_combine_feat",
                num_buckets=10,
                embedding_dim=16,
                expression="user:seq_combine_input",
                sequence_delim=";",
                sequence_length=50,
                default_value=default_value,
            )
        )
        seq_feat = combine_feature_lib.CombineFeature(
            seq_feat_cfg,
            is_sequence=True,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["seq_combine_input"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=10,
            embedding_dim=16,
            name="seq_combine_feat_emb",
            feature_names=["seq_combine_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"seq_combine_input": pa.array(["1", "", None])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "seq_combine_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )


if __name__ == "__main__":
    unittest.main()
