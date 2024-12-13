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

from tzrec.features import expr_feature as expr_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


class ExprFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [[None, "0.2", "0.3"], "0.1", [[0.1], [0.2], [0.3]]],
            [[None, 2, 3], "1", [[1], [2], [3]]],
            [[None, 0.2, 0.3], "0", [[0.0], [0.2], [0.3]]],
        ]
    )
    def test_fg_encoded_expr_feature_dense(
        self, input_feat, default_value, expected_values
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                fg_encoded_default_value=default_value,
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(expr_feat_cfg)
        self.assertEqual(expr_feat.output_dim, 1)
        self.assertEqual(expr_feat.is_sparse, False)
        self.assertEqual(expr_feat.inputs, ["expr_feat"])

        input_data = {"expr_feat": pa.array(input_feat)}
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [["1", "", None, "3"], "", [1, 3], [1, 0, 0, 1]],
            [[1, 2, None, 3], "0", [1, 2, 0, 3], [1, 1, 1, 1]],
        ]
    )
    def test_fg_encoded_expr_feature_sparse(
        self, input_feat, default_value, expected_values, expected_lengths
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                embedding_dim=16,
                boundaries=[0.1, 0.2, 0.3],
                fg_encoded_default_value=default_value,
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(expr_feat_cfg)
        self.assertEqual(expr_feat.output_dim, 16)
        self.assertEqual(expr_feat.is_sparse, True)
        self.assertEqual(expr_feat.inputs, ["expr_feat"])

        input_data = {"expr_feat": pa.array(input_feat)}
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                ["0.2", "", "0.3", None],
                [0.2, 0.2, 0.2, 0.2],
                "0.1",
                [[0.4], [0.1], [0.5], [0.1]],
            ],
            [
                [0.2, 0.1, 0.3, None],
                [0.2, 0.2, 0.2, 0.2],
                "0.1",
                [[0.4], [0.3], [0.5], [0.1]],
            ],
            [[0.2], [0.2, 0.2, 0.2, 0.2], "0.1", [[0.4], [0.4], [0.4], [0.4]]],
            [[0.2, 0.1, 0.3, None], [0.2], "0.1", [[0.4], [0.3], [0.5], [0.1]]],
        ]
    )
    def test_expr_feature_dense(
        self, input_feat_a, input_feat_b, default_value, expected_values
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                expression="a+b",
                variables=["user:a", "item:b"],
                default_value=default_value,
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(expr_feat.output_dim, 1)
        self.assertEqual(expr_feat.is_sparse, False)
        self.assertEqual(expr_feat.inputs, ["a", "b"])
        self.assertEqual(expr_feat.emb_bag_config, None)
        self.assertEqual(expr_feat.emb_config, None)

        input_data = {"a": pa.array(input_feat_a), "b": pa.array(input_feat_b)}
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [
                [0.2, 0.1, 0.3, None],
                [0.2, 0.2, 0.2, 0.2],
                "0.1",
                [3, 2, 3, 0],
                [1, 1, 1, 1],
            ],
            [["0.2", "", "0.3"], [0.2, 0.2, 0.2], "", [3, 3], [1, 0, 1]],
            [[0.2, 0.1, 0.3, None], [0.2], "0.1", [3, 2, 3, 0], [1, 1, 1, 1]],
            [[0.2], [0.2, 0.2, 0.2, 0.2], "0.1", [3, 3, 3, 3], [1, 1, 1, 1]],
        ]
    )
    def test_expr_feature_with_boundaries(
        self,
        input_feat_a,
        input_feat_b,
        default_value,
        expected_values,
        expected_lengths,
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                embedding_dim=16,
                boundaries=[0.15, 0.25, 0.35],
                expression="a+b",
                variables=["user:a", "item:b"],
                default_value=default_value,
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(expr_feat.output_dim, 16)
        self.assertEqual(expr_feat.is_sparse, True)
        self.assertEqual(expr_feat.inputs, ["a", "b"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="expr_feat_emb",
            feature_names=["expr_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(expr_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="expr_feat_emb",
            feature_names=["expr_feat"],
        )
        self.assertEqual(repr(expr_feat.emb_config), repr(expected_emb_config))

        input_data = {"a": pa.array(input_feat_a), "b": pa.array(input_feat_b)}
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                [[0.2, 0.3], [0.1, 0.2], [0.3, 0.4], []],
                [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]],
                "0.1",
                [2, 1, 2, 1],
                [1, 1, 1, 1],
            ],
            [
                ["0.2\x1d0.3", "", "0.3\x1d0.4"],
                ["0.2\x1d0.2", "0.2\x1d0.2", "0.2\x1d0.2"],
                "",
                [1, 2],
                [1, 0, 1],
            ],
        ]
    )
    def test_expr_feature_dot(
        self,
        input_feat_a,
        input_feat_b,
        default_value,
        expected_values,
        expected_lengths,
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                embedding_dim=16,
                boundaries=[0.05, 0.10, 0.15],
                expression="dot(a,b)",
                variables=["user:a", "item:b"],
                default_value=default_value,
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(expr_feat.output_dim, 16)
        self.assertEqual(expr_feat.is_sparse, True)
        self.assertEqual(expr_feat.inputs, ["a", "b"])

        input_data = {"a": pa.array(input_feat_a), "b": pa.array(input_feat_b)}
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
