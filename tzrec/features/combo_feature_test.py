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
    PoolingType,
)

from tzrec.features import combo_feature as combo_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class ComboFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [[["1\x032", "", "3"], [1, 2, 3], [2, 0, 1]], [[1, 2, 3], [1, 2, 3], [1, 1, 1]]]
    )
    def test_fg_encoded_combo_feature(
        self, input_feat, expected_values, expected_lengths
    ):
        combo_feat_cfg = feature_pb2.FeatureConfig(
            combo_feature=feature_pb2.ComboFeature(
                feature_name="combo_feat", embedding_dim=16
            )
        )
        combo_feat = combo_feature_lib.ComboFeature(combo_feat_cfg)
        self.assertEqual(combo_feat.output_dim, 16)
        self.assertEqual(combo_feat.is_sparse, True)
        self.assertEqual(combo_feat.inputs, ["combo_feat"])

        input_data = {"combo_feat": pa.array(input_feat)}
        parsed_feat = combo_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combo_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                ["ua\x1dub", "uc\x1dud", "uee\x1duf", "ug", "uh", "uj", "", None, ""],
                ["ia\x1dib", "ic", "", "id\x1die", "if", None, "ig\1dih", "ij", ""],
                "",
                [37, 90, 5, 97, 60, 56, 40, 57, 55],
                [4, 2, 0, 2, 1, 0, 0, 0, 0],
            ],
            [
                [
                    ["ua", "ub"],
                    ["uc", "ud"],
                    ["uee", "uf"],
                    ["ug"],
                    ["uh"],
                    ["uj"],
                    [],
                    None,
                    [],
                ],
                [
                    ["ia", "ib"],
                    ["ic"],
                    [],
                    ["id", "ie"],
                    ["if"],
                    None,
                    ["ig", "ih"],
                    ["ij"],
                    [],
                ],
                "",
                [37, 90, 5, 97, 60, 56, 40, 57, 55],
                [4, 2, 0, 2, 1, 0, 0, 0, 0],
            ],
            [
                ["ua\x1dub", "uc\x1dud", "uee\x1duf", "ug", "uh", "uj", "", None, ""],
                ["ia\x1dib", "ic", "", "id\x1die", "if", None, "ig\1dih", "ij", ""],
                "xyz",
                [37, 90, 5, 97, 60, 56, 13, 40, 57, 55, 13, 13, 13, 13],
                [4, 2, 1, 2, 1, 1, 1, 1, 1],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_combo_feature_with_hash_bucket_size(
        self, uid_data, iid_data, default_value, expected_values, expected_lengths
    ):
        combo_feat_cfg = feature_pb2.FeatureConfig(
            combo_feature=feature_pb2.ComboFeature(
                feature_name="combo_feat",
                embedding_dim=16,
                hash_bucket_size=100,
                expression=["user:uid_str", "item:iid_str"],
                default_value=default_value,
            )
        )
        combo_feat = combo_feature_lib.ComboFeature(
            combo_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(combo_feat.output_dim, 16)
        self.assertEqual(combo_feat.is_sparse, True)
        self.assertEqual(combo_feat.inputs, ["uid_str", "iid_str"])

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="combo_feat_emb",
            feature_names=["combo_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(combo_feat.emb_bag_config), repr(expected_emb_bag_config))

        input_data = {
            "uid_str": pa.array(uid_data),
            "iid_str": pa.array(iid_data),
        }
        parsed_feat = combo_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combo_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [
                "",
                ["ua_ia", "ua_ib", "ub_ia"],
                [2, 3, 4, 1, 2, 4, 2, 3, 3],
                [4, 2, 0, 2, 1, 0, 0, 0, 0],
            ],
            [
                "xyz",
                ["ua_ia", "ua_ib", "ub_ia"],
                [2, 3, 4, 1, 2, 4, 0, 2, 3, 3, 0, 0, 0, 0],
                [4, 2, 1, 2, 1, 1, 1, 1, 1],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_combo_feature_with_vocab_list(
        self, default_value, vocab_list, expected_values, expected_lengths
    ):
        combo_feat_cfg = feature_pb2.FeatureConfig(
            combo_feature=feature_pb2.ComboFeature(
                feature_name="combo_feat",
                embedding_dim=16,
                vocab_list=vocab_list,
                expression=["user:uid_str", "item:iid_str"],
                default_value=default_value,
            )
        )
        combo_feat = combo_feature_lib.ComboFeature(
            combo_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(combo_feat.output_dim, 16)
        self.assertEqual(combo_feat.is_sparse, True)
        self.assertEqual(combo_feat.inputs, ["uid_str", "iid_str"])

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=5,
            embedding_dim=16,
            name="combo_feat_emb",
            feature_names=["combo_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(combo_feat.emb_bag_config), repr(expected_emb_bag_config))

        input_data = {
            "uid_str": pa.array(
                ["ua\x1dub", "ua\x1dub", "ua\x1dub", "ua", "ua", "ua", None, "", ""]
            ),
            "iid_str": pa.array(
                ["ia\x1dib", "ia", "", "ia\x1dib", "ib", "", "ia\1dib", "ia", None]
            ),
        }
        parsed_feat = combo_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "combo_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
