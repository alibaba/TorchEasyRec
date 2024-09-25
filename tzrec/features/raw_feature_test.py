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

from tzrec.features import raw_feature as raw_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


class RawFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [[None, "0.2", "0.3"], "0.1", [[0.1], [0.2], [0.3]], 1],
            [[None, 2, 3], "1", [[1], [2], [3]], 1],
            [[None, 0.2, 0.3], "0", [[0.0], [0.2], [0.3]], 1],
            [
                ["0.1\x030.4", "0.2\x030.5", None, "0.3\x030.6"],
                "0.0\x030.0",
                [[0.1, 0.4], [0.2, 0.5], [0.0, 0.0], [0.3, 0.6]],
                2,
            ],
        ]
    )
    def test_fg_encoded_raw_feature_dense(
        self, input_feat, default_value, expected_values, value_dim
    ):
        raw_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                value_dim=value_dim,
                fg_encoded_default_value=default_value,
            )
        )
        raw_feat = raw_feature_lib.RawFeature(raw_feat_cfg)
        self.assertEqual(raw_feat.output_dim, value_dim)
        self.assertEqual(raw_feat.is_sparse, False)
        self.assertEqual(raw_feat.inputs, ["raw_feat"])

        input_data = {"raw_feat": pa.array(input_feat)}
        parsed_feat = raw_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [["1", "", None, "3"], "", [1, 3], [1, 0, 0, 1]],
            [[1, 2, None, 3], "0", [1, 2, 0, 3], [1, 1, 1, 1]],
        ]
    )
    def test_fg_encoded_raw_feature_sparse(
        self, input_feat, default_value, expected_values, expected_lengths
    ):
        raw_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                embedding_dim=16,
                boundaries=[0.1, 0.2, 0.3],
                fg_encoded_default_value=default_value,
            )
        )
        raw_feat = raw_feature_lib.RawFeature(raw_feat_cfg)
        self.assertEqual(raw_feat.output_dim, 16)
        self.assertEqual(raw_feat.is_sparse, True)
        self.assertEqual(raw_feat.inputs, ["raw_feat"])

        input_data = {"raw_feat": pa.array(input_feat)}
        parsed_feat = raw_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                ["0.202392938293829442", "", "0.3", None],
                "0.1",
                "",
                [[0.202392938293829442], [0.1], [0.3], [0.1]],
                1,
                [0.1],
            ],
            [[0.2, 0.1, 0.3, None], "0.1", "", [[0.2], [0.1], [0.3], [0.1]], 1, [0.1]],
            [
                [0.202392938293829442, 0.13902309283232323, 0.39203902392939293, None],
                "0.1",
                "",
                [
                    [0.202392938293829442],
                    [0.13902309283232323],
                    [0.39203902392939293],
                    [0.1],
                ],
                1,
                [0.1],
            ],
            [
                ["0.2\x1d0.5", "", "0.3\x1d0.6"],
                "0.1\x1d0.4",
                "",
                [[0.2, 0.5], [0.1, 0.4], [0.3, 0.6]],
                2,
                [0.1, 0.4],
            ],
            [
                [["0.2", "0.5"], None, ["0.3", "0.6"]],
                "0.1\x1d0.4",
                "",
                [[0.2, 0.5], [0.1, 0.4], [0.3, 0.6]],
                2,
                [0.1, 0.4],
            ],
            [
                ["0.1", "", "10"],
                "0.01",
                "method=log10,threshold=0.05,default=-5",
                [[-1], [0.01], [1]],
                1,
                [0.01],
            ],
            [
                [0.2, 0.1, 0.3],
                "0.1",
                "method=zscore,mean=0.1,standard_deviation=10.0",
                [[0.01], [0.0], [0.02]],
                1,
                [0.1],
            ],
            [
                ["0.2\x1d0.5", "", "0.3\x1d0.6"],
                "0.1\x1d0.4",
                "method=minmax,min=0.1,max=0.6",
                [[0.2, 0.8], [0.1, 0.4], [0.4, 1.0]],
                2,
                [0.1, 0.4],
            ],
        ]
    )
    def test_raw_feature_dense(
        self,
        input_feat,
        default_value,
        normalizer,
        expected_values,
        value_dim,
        expected_fg_default,
    ):
        raw_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                expression="user:raw_input",
                default_value=default_value,
                normalizer=normalizer,
                value_dim=value_dim,
            )
        )
        raw_feat = raw_feature_lib.RawFeature(raw_feat_cfg, fg_mode=FgMode.NORMAL)
        np.testing.assert_allclose(
            raw_feat.fg_encoded_default_value(), expected_fg_default
        )
        self.assertEqual(raw_feat.output_dim, value_dim)
        self.assertEqual(raw_feat.is_sparse, False)
        self.assertEqual(raw_feat.inputs, ["raw_input"])
        self.assertEqual(raw_feat.emb_bag_config, None)
        self.assertEqual(raw_feat.emb_config, None)

        input_data = {"raw_input": pa.array(input_feat)}
        parsed_feat = raw_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [[0.2, 0.1, 0.3, None], "0.1", "", [2, 1, 3, 1], [1, 1, 1, 1], 1, 1],
            [["0.2", "", "0.3"], "", "", [2, 3], [1, 0, 1], 1, None],
            [
                ["0.2\x1d0.5", "", "0.3\x1d0.6"],
                "0.1\x1d0.4",
                "",
                [2, 3, 1, 3, 3, 3],
                [2, 2, 2],
                2,
                [1, 3],
            ],
            [
                [["0.2", "0.5"], [], ["0.3", "0.6"]],
                "0.1\x1d0.4",
                "",
                [2, 3, 1, 3, 3, 3],
                [2, 2, 2],
                2,
                [1, 3],
            ],
            [
                ["0.1", "", "10"],
                "0.01",
                "method=log10,threshold=0.05,default=-5",
                [0, 0, 3],
                [1, 1, 1],
                1,
                [0],
            ],
        ]
    )
    def test_raw_feature_with_boundaries(
        self,
        input_feat,
        default_value,
        normalizer,
        expected_values,
        expected_lengths,
        value_dim,
        expected_fg_default,
    ):
        raw_feat_cfg = feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="raw_feat",
                embedding_dim=16,
                boundaries=[0.05, 0.15, 0.25],
                expression="user:raw_input",
                default_value=default_value,
                normalizer=normalizer,
                value_dim=value_dim,
            )
        )
        raw_feat = raw_feature_lib.RawFeature(raw_feat_cfg, fg_mode=FgMode.NORMAL)
        fg_default = raw_feat.fg_encoded_default_value()
        if expected_fg_default:
            np.testing.assert_allclose(fg_default, expected_fg_default)
        else:
            self.assertEqual(fg_default, expected_fg_default)
        self.assertEqual(raw_feat.output_dim, 16)
        self.assertEqual(raw_feat.is_sparse, True)
        self.assertEqual(raw_feat.inputs, ["raw_input"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="raw_feat_emb",
            feature_names=["raw_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(raw_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="raw_feat_emb",
            feature_names=["raw_feat"],
        )
        self.assertEqual(repr(raw_feat.emb_config), repr(expected_emb_config))

        input_data = {"raw_input": pa.array(input_feat)}
        parsed_feat = raw_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "raw_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
