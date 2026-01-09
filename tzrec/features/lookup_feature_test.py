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

from tzrec.features import lookup_feature as lookup_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class LookupFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [["1\x032", "", "3"], [1, 2, 0, 3], [2, 1, 1], "0"],
            [[1, 2, None, 3], [1, 2, 0, 3], [1, 1, 1, 1], "0"],
            [[1, 2, 0, 3], [1, 2, 0, 3], [1, 1, 1, 1], None],
        ]
    )
    def test_fg_encoded_lookup_feature(
        self, input_feat, expected_values, expected_lengths, fg_encoded_default
    ):
        lookup_feat_cfg = feature_pb2.FeatureConfig(
            lookup_feature=feature_pb2.LookupFeature(
                feature_name="lookup_feat",
                map="user:kv_cate",
                key="item:cate",
                num_buckets=10,
                embedding_dim=16,
            )
        )
        if fg_encoded_default:
            lookup_feat_cfg.lookup_feature.fg_encoded_default_value = fg_encoded_default
        lookup_feat = lookup_feature_lib.LookupFeature(lookup_feat_cfg)

        if fg_encoded_default is None:
            np.testing.assert_allclose(lookup_feat.fg_encoded_default_value(), [0])
        self.assertEqual(lookup_feat.output_dim, 16)
        self.assertEqual(lookup_feat.is_sparse, True)
        self.assertEqual(lookup_feat.inputs, ["lookup_feat"])

        input_data = {"lookup_feat": pa.array(input_feat)}
        parsed_feat = lookup_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "lookup_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [["1\x032", "", "3\x034"], [[1, 2], [0, 0], [3, 4]], "0\x030"],
            [[[1, 2], [], [3, 4]], [[1, 2], [0, 0], [3, 4]], "0\x030"],
            [[[1, 2], [0, 0], [3, 4]], [[1, 2], [0, 0], [3, 4]], None],
        ]
    )
    def test_fg_encoded_lookup_dense_feature(
        self, input_feat, expected_values, fg_encoded_default
    ):
        lookup_feat_cfg = feature_pb2.FeatureConfig(
            lookup_feature=feature_pb2.LookupFeature(
                feature_name="lookup_feat",
                map="user:kv_cate",
                key="item:cate",
                value_dim=2,
            )
        )
        if fg_encoded_default:
            lookup_feat_cfg.lookup_feature.fg_encoded_default_value = fg_encoded_default
        lookup_feat = lookup_feature_lib.LookupFeature(lookup_feat_cfg)

        if fg_encoded_default is None:
            np.testing.assert_allclose(lookup_feat.fg_encoded_default_value(), [0, 0])
        self.assertEqual(lookup_feat.output_dim, 2)
        self.assertEqual(lookup_feat.is_sparse, False)
        self.assertEqual(lookup_feat.inputs, ["lookup_feat"])

        input_data = {"lookup_feat": pa.array(input_feat)}
        parsed_feat = lookup_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "lookup_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            [
                pa.array(
                    [
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "",
                        "",
                    ]
                ),
                pa.array(["ca\x1dcb", "ca", "ca\x1dcd", "cd", "", "ca", ""]),
                1,
                "sum",
                [[3], [1], [1], [0], [0], [0], [0]],
                0,
            ],
            [
                pa.array(
                    [
                        {"ca": 1, "cb": 2},
                        {"ca": 1, "cb": 2},
                        {"ca": 1, "cb": 2},
                        {"ca": 1, "cb": 2},
                        {"ca": 1, "cb": 2},
                        {},
                        None,
                    ],
                    type=pa.map_(pa.string(), pa.int64()),
                ),
                pa.array(
                    [["ca", "cb"], ["ca"], ["ca", "cd"], ["cd"], [], ["ca"], None]
                ),
                1,
                "sum",
                [[3], [1], [1], [0], [0], [0], [0]],
                0,
            ],
            [
                pa.array(
                    [
                        {},
                        None,
                        {},
                        None,
                        None,
                        None,
                        None,
                    ],
                    type=pa.map_(pa.string(), pa.int64()),
                ),
                pa.array(
                    [["ca", "cb"], ["ca"], ["ca", "cd"], ["cd"], [], ["ca"], None]
                ),
                1,
                "sum",
                [[0], [0], [0], [0], [0], [0], [0]],
                0,
            ],
            [
                pa.array(
                    [
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "ca:1\x1dcb:2",
                        "",
                        "",
                    ]
                ),
                pa.array(["ca\x1dcb", "ca", "ca\x1dcd", "cd", "", "ca", ""]),
                1,
                "mean",
                [[1.5], [1], [1], [0], [0], [0], [0]],
                0,
            ],
            [
                pa.array(
                    [
                        "ca:1,2\x1dcb:3,4",
                        "ca:1,2\x1dcb:3,4",
                        "ca:1,2\x1dcb:3,4",
                        "",
                        "ca:1,2\x1dcb:3,4",
                    ]
                ),
                pa.array(["ca", "cd", "", "ca", ""]),
                2,
                "",
                [[1, 2], [0, 0], [0, 0], [0, 0], [0, 0]],
                [0, 0],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_lookup_feature_dense(
        self,
        kv_data,
        key_data,
        value_dim,
        combiner,
        expected_values,
        expected_fg_encoded_default,
    ):
        lookup_feat_cfg = feature_pb2.FeatureConfig(
            lookup_feature=feature_pb2.LookupFeature(
                feature_name="lookup_feat",
                map="user:kv_cate",
                key="item:cate",
                default_value="0",
                combiner=combiner,
                value_dim=value_dim,
            )
        )
        lookup_feat = lookup_feature_lib.LookupFeature(
            lookup_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(
            lookup_feat.fg_encoded_default_value(), expected_fg_encoded_default
        )
        self.assertEqual(lookup_feat.output_dim, value_dim)
        self.assertEqual(lookup_feat.is_sparse, False)
        self.assertEqual(lookup_feat.inputs, ["kv_cate", "cate"])
        self.assertEqual(lookup_feat.emb_bag_config, None)
        self.assertEqual(lookup_feat.emb_config, None)

        input_data = {
            "kv_cate": kv_data,
            "cate": key_data,
        }
        parsed_feat = lookup_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "lookup_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))

    @parameterized.expand(
        [
            ["sum", [4, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
            ["mean", [3, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
            ["", [2, 3, 2, 2, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_lookup_feature_with_boundary(
        self, combiner, expected_values, expected_lengths
    ):
        lookup_feat_cfg = feature_pb2.FeatureConfig(
            lookup_feature=feature_pb2.LookupFeature(
                feature_name="lookup_feat",
                boundaries=[-0.5, 0.5, 1.5, 2.5],
                embedding_dim=16,
                map="user:kv_cate",
                key="item:cate",
                combiner=combiner,
                default_value="0",
            )
        )
        lookup_feat = lookup_feature_lib.LookupFeature(
            lookup_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(lookup_feat.output_dim, 16)
        self.assertEqual(lookup_feat.is_sparse, True)
        self.assertEqual(lookup_feat.inputs, ["kv_cate", "cate"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=5,
            embedding_dim=16,
            name="lookup_feat_emb",
            feature_names=["lookup_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(
            repr(lookup_feat.emb_bag_config), repr(expected_emb_bag_config)
        )

        input_data = {
            "kv_cate": pa.array(
                [
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "",
                    "",
                ]
            ),
            "cate": pa.array(["ca\x1dcb", "ca", "ca\x1dcd", "cd", "", "ca", ""]),
        }
        parsed_feat = lookup_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "lookup_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", [1, 2, 1, 1], [2, 1, 1, 0, 0, 0, 0]],
            ["0", [1, 2, 1, 1, 0, 0, 0, 0], [2, 1, 1, 1, 1, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_lookup_feature_with_num_buckets(
        self, default_value, expected_values, expected_lengths
    ):
        lookup_feat_cfg = feature_pb2.FeatureConfig(
            lookup_feature=feature_pb2.LookupFeature(
                feature_name="lookup_feat",
                num_buckets=10,
                embedding_dim=16,
                map="user:kv_cate",
                key="item:cate",
                pooling="mean",
                default_value=default_value,
            )
        )
        lookup_feat = lookup_feature_lib.LookupFeature(
            lookup_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(lookup_feat.output_dim, 16)
        self.assertEqual(lookup_feat.is_sparse, True)
        self.assertEqual(lookup_feat.inputs, ["kv_cate", "cate"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=10,
            embedding_dim=16,
            name="lookup_feat_emb",
            feature_names=["lookup_feat"],
            pooling=PoolingType.MEAN,
        )
        self.assertEqual(
            repr(lookup_feat.emb_bag_config), repr(expected_emb_bag_config)
        )

        input_data = {
            "kv_cate": pa.array(
                [
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "ca:1\x1dcb:2",
                    "",
                    None,
                ]
            ),
            "cate": pa.array(["ca\x1dcb", "ca", "ca\x1dcd", "cd", None, "ca", ""]),
        }
        parsed_feat = lookup_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "lookup_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", [35, 30, 35, 35], [2, 1, 1, 0, 0, 0, 0]],
            ["z", [35, 30, 35, 35, 54, 54, 54, 54], [2, 1, 1, 1, 1, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_lookup_feature_with_hash_bucket_size(
        self, default_value, expected_values, expected_lengths
    ):
        lookup_feat_cfg = feature_pb2.FeatureConfig(
            lookup_feature=feature_pb2.LookupFeature(
                feature_name="lookup_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                map="user:kv_cate",
                key="item:cate",
                default_value=default_value,
            )
        )
        lookup_feat = lookup_feature_lib.LookupFeature(
            lookup_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(lookup_feat.output_dim, 16)
        self.assertEqual(lookup_feat.is_sparse, True)
        self.assertEqual(lookup_feat.inputs, ["kv_cate", "cate"])
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="lookup_feat_emb",
            feature_names=["lookup_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(
            repr(lookup_feat.emb_bag_config), repr(expected_emb_bag_config)
        )

        input_data = {
            "kv_cate": pa.array(
                [
                    "ca:x\x1dcb:y",
                    "ca:x\x1dcb:y",
                    "ca:x\x1dcb:y",
                    "ca:x\x1dcb:y",
                    "ca:x\x1dcb:y",
                    "",
                    "",
                ]
            ),
            "cate": pa.array(["ca\x1dcb", "ca", "ca\x1dcd", "cd", "", "ca", ""]),
        }
        parsed_feat = lookup_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "lookup_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
