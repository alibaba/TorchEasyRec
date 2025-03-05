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

from tzrec.features import tokenize_feature as tokenize_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class TokenizeFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [["1\x032", "", None, "3"], [1, 2, 3], [2, 0, 0, 1]],
            [[[1, 2], None, None, [3]], [1, 2, 3], [2, 0, 0, 1]],
        ]
    )
    def test_fg_encoded_tokenize_feature(
        self, input_feat, expected_values, expected_lengths
    ):
        token_feat_cfg = feature_pb2.FeatureConfig(
            tokenize_feature=feature_pb2.TokenizeFeature(
                feature_name="token_feat",
                embedding_dim=16,
                vocab_file="data/test/tokenizer.json",
            )
        )
        token_feat = tokenize_feature_lib.TokenizeFeature(token_feat_cfg)
        self.assertEqual(token_feat.output_dim, 16)
        self.assertEqual(token_feat.is_sparse, True)
        self.assertEqual(token_feat.inputs, ["token_feat"])

        input_data = {"token_feat": pa.array(input_feat)}
        parsed_feat = token_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "token_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                "",
                ["abc efg", "", "hij"],
                False,
                [19758, 299, 16054, 73, 1944],
                [3, 0, 2],
            ],
            [
                "xyz",
                ["abc efg", "", "hij"],
                False,
                [19758, 299, 16054, 35609, 73, 1944],
                [3, 1, 2],
            ],
            [
                "",
                ["ABC efg", "", "HIJ"],
                True,
                [19758, 299, 16054, 73, 1944],
                [3, 0, 2],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_tokenize_feature(
        self,
        default_value,
        input_data,
        use_text_norm,
        expected_values,
        expected_lengths,
    ):
        token_feat_cfg = feature_pb2.FeatureConfig(
            tokenize_feature=feature_pb2.TokenizeFeature(
                feature_name="token_feat",
                vocab_file="data/test/tokenizer.json",
                embedding_dim=16,
                expression="user:token_input",
                default_value=default_value,
            )
        )
        if use_text_norm:
            text_norm = feature_pb2.TextNormalizer(
                norm_options=[feature_pb2.TEXT_UPPER2LOWER]
            )
            token_feat_cfg.tokenize_feature.text_normalizer.CopyFrom(text_norm)
        token_feat = tokenize_feature_lib.TokenizeFeature(
            token_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(token_feat.inputs, ["token_input"])

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=50277,
            embedding_dim=16,
            name="token_feat_emb",
            feature_names=["token_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(token_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=50277,
            embedding_dim=16,
            name="token_feat_emb",
            feature_names=["token_feat"],
        )
        self.assertEqual(repr(token_feat.emb_config), repr(expected_emb_config))

        input_data = {"token_input": pa.array(input_data)}
        parsed_feat = token_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "token_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                "",
                ["abc efg", "", "hij"],
                [703, 75, 3, 15, 89, 122, 7102, 354],
                [6, 0, 2],
            ],
            [
                "xyz",
                ["abc efg", "", "hij"],
                [703, 75, 3, 15, 89, 122, 3, 226, 63, 172, 7102, 354],
                [6, 4, 2],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_tokenize_feature_sentencepiece(
        self,
        default_value,
        input_data,
        expected_values,
        expected_lengths,
    ):
        token_feat_cfg = feature_pb2.FeatureConfig(
            tokenize_feature=feature_pb2.TokenizeFeature(
                feature_name="token_feat",
                vocab_file="data/test/spiece.model",
                embedding_dim=16,
                tokenizer_type="sentencepiece",
                expression="user:token_input",
                default_value=default_value,
            )
        )
        token_feat = tokenize_feature_lib.TokenizeFeature(
            token_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(token_feat.inputs, ["token_input"])

        input_data = {"token_input": pa.array(input_data)}
        parsed_feat = token_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "token_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
