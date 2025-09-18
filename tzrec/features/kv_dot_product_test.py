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

from tzrec.features import kv_dot_product as kv_dot_product_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


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


if __name__ == "__main__":
    unittest.main()
