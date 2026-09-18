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

from tzrec.features import expr_feature as expr_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


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
                [2, 1, 2, 2],
                [1, 1, 1, 1],
                None,
            ],
            [
                ["0.2\x1d0.2", "", "0.3\x1d0.4"],
                ["0.2\x1d0.2", "0.2\x1d0.2", "0.2\x1d0.2"],
                "",
                [1, 2],
                [1, 0, 1],
                None,
            ],
            [
                ["0.2,0.2", "", "0.3,0.4"],
                ["0.2,0.2", "0.2,0.2", "0.2,0.2"],
                "",
                [1, 2],
                [1, 0, 1],
                ",",
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
        sep,
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
        if sep is not None:
            expr_feat_cfg.expr_feature.separator = sep
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

    @parameterized.expand(
        [
            [{"num_buckets": 100}, "int64", [10, 20, 30, 0]],
            [{"hash_bucket_size": 100}, "float", [39, 63, 46, 0]],
        ]
    )
    def test_expr_feature_with_id_bucketizer(
        self, bucketizer, expected_value_type, expected_values
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                embedding_dim=16,
                expression="a*10",
                variables=["user:a"],
                default_value="0",
                **bucketizer,
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(expr_feat.output_dim, 16)
        self.assertEqual(expr_feat.is_sparse, True)
        self.assertEqual(expr_feat.inputs, ["a"])
        self.assertEqual(expr_feat.fg_json()[0]["value_type"], expected_value_type)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="expr_feat_emb",
            feature_names=["expr_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(expr_feat.emb_bag_config), repr(expected_emb_bag_config))

        input_data = {"a": pa.array([1.0, 2.0, 3.0, None])}
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1, 1, 1, 1]))

    def test_expr_feature_with_int_value_type(self):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                expression="a==b",
                variables=["user:a", "item:b"],
                fg_value_type="int64",
                value_dim=4,
                default_value="0",
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(expr_feat.output_dim, 4)
        self.assertEqual(expr_feat.is_sparse, False)
        self.assertEqual(expr_feat.fg_json()[0]["value_type"], "int64")

        input_data = {
            "a": pa.array(["1\x1d2\x1d3\x1d2", "1\x1d1\x1d1\x1d1"]),
            "b": pa.array([2, 1]),
        }
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "expr_feat")
        np.testing.assert_allclose(
            parsed_feat.values, np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
        )

    def test_expr_feature_int_value_type_with_boundaries(self):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="expr_feat",
                embedding_dim=16,
                boundaries=[0.5],
                expression="a==b",
                variables=["user:a", "item:b"],
                fg_value_type="int64",
            )
        )
        with self.assertRaises(AssertionError):
            expr_feature_lib.ExprFeature(
                expr_feat_cfg, fg_mode=FgMode.FG_NORMAL
            ).fg_json()

    @parameterized.expand(
        [
            param(
                "int_scalar",
                request_time=[10, 20],
                event_time=["2|3", "4"],
                expected_values=[[8], [7], [16]],
                expected_seq_lengths=[2, 1],
            ),
            param(
                "str_scalar",
                request_time=["10", "20"],
                event_time=["2|3", "4"],
                expected_values=[[8], [7], [16]],
                expected_seq_lengths=[2, 1],
            ),
            param(
                "with_null",
                request_time=[10, None, 5],
                event_time=["2|3", "4", None],
                expected_values=[[8], [7], [0], [0]],
                expected_seq_lengths=[2, 1, 1],
            ),
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_expr_feature_dense(
        self, name, request_time, event_time, expected_values, expected_seq_lengths
    ):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="ts_diff",
                expression="request_time - event_time",
                variables=["user:request_time", "user:event_time"],
                sequence_fields=["event_time"],
                default_value="0",
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim="|",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(expr_feat.output_dim, 1)
        self.assertEqual(expr_feat.is_sparse, False)
        self.assertEqual(expr_feat.inputs, ["request_time", "click_50_seq__event_time"])
        self.assertEqual(expr_feat.emb_config, None)

        input_data = {
            "request_time": pa.array(request_time),
            "click_50_seq__event_time": pa.array(event_time),
        }
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__ts_diff")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(
            parsed_feat.seq_lengths, np.array(expected_seq_lengths)
        )

    def test_sequence_expr_feature_with_boundaries(self):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            expr_feature=feature_pb2.ExprFeature(
                feature_name="ts_diff",
                embedding_dim=16,
                boundaries=[5, 10, 15],
                expression="request_time - event_time",
                variables=["user:request_time", "user:event_time"],
                sequence_fields=["event_time"],
                default_value="0",
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim="|",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(expr_feat.output_dim, 16)
        self.assertEqual(expr_feat.is_sparse, True)
        self.assertEqual(expr_feat.inputs, ["request_time", "click_50_seq__event_time"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="click_50_seq__ts_diff_emb",
            feature_names=["click_50_seq__ts_diff"],
        )
        self.assertEqual(repr(expr_feat.emb_config), repr(expected_emb_config))

        input_data = {
            "request_time": pa.array([10, 20, None]),
            "click_50_seq__event_time": pa.array(["2|3", "4", "1"]),
        }
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__ts_diff")
        np.testing.assert_allclose(parsed_feat.values, np.array([1, 1, 3, 0]))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array([1, 1, 1, 1]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1, 1]))

    @parameterized.expand(
        [
            param(
                "item_side_seq",
                variables=["user:request_time", "item:event_time"],
                sequence_fields=[],
            ),
            param(
                "user_side_seq",
                variables=["user:request_time", "user:event_time"],
                sequence_fields=["event_time"],
            ),
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_simple_sequence_expr_feature_dense(self, name, variables, sequence_fields):
        expr_feat_cfg = feature_pb2.FeatureConfig(
            sequence_expr_feature=feature_pb2.ExprFeature(
                feature_name="click_50_seq_ts_diff",
                sequence_delim="|",
                sequence_length=50,
                expression="request_time - event_time",
                variables=variables,
                sequence_fields=sequence_fields,
                default_value="0",
            )
        )
        expr_feat = expr_feature_lib.ExprFeature(
            expr_feat_cfg, is_sequence=True, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(expr_feat.output_dim, 1)
        self.assertEqual(expr_feat.is_sparse, False)
        self.assertEqual(expr_feat.inputs, ["request_time", "event_time"])
        self.assertEqual(expr_feat.sequence_input_names, ["event_time"])
        self.assertEqual(expr_feat.emb_config, None)
        self.assertEqual(
            expr_feat.fg_json()[0].get("sequence_fields"), sequence_fields or None
        )

        input_data = {
            "request_time": pa.array([10, None, 5]),
            "event_time": pa.array(["2|3", "4", None]),
        }
        parsed_feat = expr_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_ts_diff")
        np.testing.assert_allclose(parsed_feat.values, np.array([[8], [7], [0], [0]]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1, 1]))


if __name__ == "__main__":
    unittest.main()
