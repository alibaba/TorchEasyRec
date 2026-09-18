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

from tzrec.features import kv_dot_product as kv_dot_product_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class KvDotProductTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                pa.array(["a:0.5|b:0.5", "a|b|c", "a|b", None]),
                pa.array(["a:0.5|b:0.5", "a|b", "", ""]),
                "0.1",
                [[0.5], [2.0], [0.1], [0.1]],
            ],
            [
                pa.array([["a:0.5", "b:0.5"], ["a", "b", "c"], ["a", "b"], None]),
                pa.array([["a:0.5", "b:0.5"], ["a", "b"], [], []]),
                "0.1",
                [[0.5], [2.0], [0.1], [0.1]],
            ],
            [
                pa.array(
                    [
                        {"a": 0.5, "b": 0.5},
                        {"a": 1, "b": 1, "c": 1},
                        {"a": 1, "b": 1},
                        None,
                    ],
                    type=pa.map_(pa.string(), pa.float32()),
                ),
                pa.array(
                    [{"a": 0.5, "b": 0.5}, {"a": 1, "b": 1}, {}, {}],
                    type=pa.map_(pa.string(), pa.float32()),
                ),
                "0.1",
                [[0.5], [2.0], [0.1], [0.1]],
            ],
        ]
    )
    def test_kv_dot_product_dense(
        self, input_feat_q, input_feat_d, default_value, expected_values
    ):
        kdp_feat_cfg = feature_pb2.FeatureConfig(
            kv_dot_product=feature_pb2.KvDotProduct(
                feature_name="kdp_feat",
                query="user:q",
                document="item:d",
                default_value=default_value,
                separator="|",
            )
        )
        kdp_feat = kv_dot_product_lib.KvDotProduct(
            kdp_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(kdp_feat.output_dim, 1)
        self.assertEqual(kdp_feat.is_sparse, False)
        self.assertEqual(kdp_feat.inputs, ["q", "d"])
        self.assertEqual(kdp_feat.emb_bag_config, None)
        self.assertEqual(kdp_feat.emb_config, None)

        input_data = {"q": input_feat_q, "d": input_feat_d}
        parsed_feat = kdp_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "kdp_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [
                pa.array(["a:0.5|b:0.5", "a|b|c", "a|b", None]),
                pa.array(["a:0.5|b:0.5", "a|b", "", ""]),
                "0.1",
                [2, 3, 1, 1],
                [1, 1, 1, 1],
            ],
            [
                pa.array([["a:0.5", "b:0.5"], ["a", "b", "c"], ["a", "b"], None]),
                pa.array([["a:0.5", "b:0.5"], ["a", "b"], [], []]),
                "0.1",
                [2, 3, 1, 1],
                [1, 1, 1, 1],
            ],
            [
                pa.array(
                    [
                        {"a": 0.5, "b": 0.5},
                        {"a": 1, "b": 1, "c": 1},
                        {"a": 1, "b": 1},
                        None,
                    ],
                    type=pa.map_(pa.string(), pa.float32()),
                ),
                pa.array(
                    [{"a": 0.5, "b": 0.5}, {"a": 1, "b": 1}, {}, {}],
                    type=pa.map_(pa.string(), pa.float32()),
                ),
                "",
                [2, 3, 0, 0],
                [1, 1, 1, 1],
            ],
        ]
    )
    def test_kv_dot_product_with_boundaries(
        self,
        input_feat_q,
        input_feat_d,
        default_value,
        expected_values,
        expected_lengths,
    ):
        kdp_feat_cfg = feature_pb2.FeatureConfig(
            kv_dot_product=feature_pb2.KvDotProduct(
                feature_name="kdp_feat",
                query="user:q",
                document="item:d",
                default_value=default_value,
                boundaries=[0.05, 0.15, 1.0],
                embedding_dim=16,
                separator="|",
            )
        )
        kdp_feat = kv_dot_product_lib.KvDotProduct(
            kdp_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(kdp_feat.output_dim, 16)
        self.assertEqual(kdp_feat.is_sparse, True)
        self.assertEqual(kdp_feat.inputs, ["q", "d"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="kdp_feat_emb",
            feature_names=["kdp_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(kdp_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4,
            embedding_dim=16,
            name="kdp_feat_emb",
            feature_names=["kdp_feat"],
        )
        self.assertEqual(repr(kdp_feat.emb_config), repr(expected_emb_config))

        input_data = {"q": input_feat_q, "d": input_feat_d}
        parsed_feat = kdp_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "kdp_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_kv_dot_product_with_kv_delimiter_and_normalizer(self):
        kdp_feat_cfg = feature_pb2.FeatureConfig(
            kv_dot_product=feature_pb2.KvDotProduct(
                feature_name="kdp_feat",
                query="user:q",
                document="item:d",
                default_value="0",
                separator="|",
                kv_delimiter="=",
                normalizer="method=log10,threshold=1e-10,default=-10",
            )
        )
        kdp_feat = kv_dot_product_lib.KvDotProduct(
            kdp_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        fg_cfg = kdp_feat.fg_json()[0]
        self.assertEqual(fg_cfg["kv_delimiter"], "=")
        self.assertEqual(
            fg_cfg["normalizer"], "method=log10,threshold=1e-10,default=-10"
        )

        input_data = {"q": pa.array(["a=2|b=3"]), "d": pa.array(["a=2|b=2"])}
        parsed_feat = kdp_feat.parse(input_data)
        np.testing.assert_allclose(parsed_feat.values, np.array([[1.0]]))

    def test_kv_dot_product_with_invalid_kv_delimiter(self):
        kdp_feat_cfg = feature_pb2.FeatureConfig(
            kv_dot_product=feature_pb2.KvDotProduct(
                feature_name="kdp_feat",
                query="user:q",
                document="item:d",
                kv_delimiter="::",
            )
        )
        kdp_feat = kv_dot_product_lib.KvDotProduct(kdp_feat_cfg)
        with self.assertRaisesRegex(ValueError, "invalid kv_delimiter"):
            kdp_feat.fg_json()


class SequenceKvDotProductTest(unittest.TestCase):
    @parameterized.expand(
        [
            param("item_side_seq", document="item:d", sequence_fields=[]),
            param("user_side_seq", document="user:d", sequence_fields=["d"]),
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_simple_sequence_kv_dot_product_dense(
        self, name, document, sequence_fields
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_kv_dot_product=feature_pb2.KvDotProduct(
                feature_name="click_50_seq_kdp_feat",
                sequence_delim=";",
                sequence_length=50,
                query="user:q",
                document=document,
                sequence_fields=sequence_fields,
                separator="|",
                default_value="0.1",
            )
        )
        seq_feat = kv_dot_product_lib.KvDotProduct(
            seq_feat_cfg, is_sequence=True, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(seq_feat.output_dim, 1)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["q", "d"])
        self.assertEqual(seq_feat.sequence_input_names, ["d"])
        self.assertEqual(
            seq_feat.fg_json()[0].get("sequence_fields"), sequence_fields or None
        )

        input_data = {
            "q": pa.array(["a:0.5|b:0.5", "a|b|c"]),
            "d": pa.array(["a:0.5|b:0.5;a|b", "a|b"]),
        }
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_kdp_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([[0.5], [1.0], [2.0]]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1]))


if __name__ == "__main__":
    unittest.main()
