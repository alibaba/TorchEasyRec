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

from tzrec.features import bool_mask_feature as bool_mask_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


class BoolMaskFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [[["1,2,3,4"], ["true,false,true,false"]], "0", [1, 3], [1, 1]],
            [
                [["1", "2", "3", "4"], ["true", "false", "true", "false"]],
                "0",
                [1, 3],
                [1, 1],
            ],
            [[1, 2, None], "3", [1, 3], [1, 1]],
        ]
    )
    def test_fg_encoded_bool_mask_feature_with_num_buckets(
        self, input_feat, default_value, expected_values, expected_lengths
    ):
        bool_mask_feature = feature_pb2.BoolMaskFeature(
            feature_name="bool_mask_feat",
            embedding_dim=16,
            expression=["user:itemid", "user:mask"],
            default_value=default_value,
            num_buckets=10,
            separator=",",
        )
        bool_mask_feat_cfg = feature_pb2.FeatureConfig(
            bool_mask_feature=bool_mask_feature
        )
        bool_feat = bool_mask_feature_lib.BoolMaskFeature(
            bool_mask_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(bool_feat.output_dim, 16)
        self.assertEqual(bool_feat.is_sparse, True)
        self.assertEqual(bool_feat.inputs, ["itemid", "mask"])

        input_data = {
            "itemid": pa.array(input_feat[0]),
            "mask": pa.array(input_feat[1]),
        }
        parsed_feat = bool_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "bool_mask_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            [[["1,2,3,4"], ["true,false,true,false"]], "0", [1, 3], [5, 1]],
            [
                [["1", "2", "3", "4"], ["true", "false", "true", "false"]],
                "0",
                [5, 3],
                [1, 1],
            ],
            [[1, 2, None], "3", [5, 3], [1, 1]],
        ]
    )
    def test_fg_encoded_bool_mask_feature_with_hash_bucket_size(
        self, input_feat, default_value, expected_values, expected_lengths
    ):
        bool_mask_feature = feature_pb2.BoolMaskFeature(
            feature_name="bool_mask_feat",
            embedding_dim=16,
            expression=["user:itemid", "user:mask"],
            default_value=default_value,
            hash_bucket_size=10,
            separator=",",
        )
        bool_mask_feat_cfg = feature_pb2.FeatureConfig(
            bool_mask_feature=bool_mask_feature
        )
        bool_feat = bool_mask_feature_lib.BoolMaskFeature(
            bool_mask_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(bool_feat.output_dim, 16)
        self.assertEqual(bool_feat.is_sparse, True)
        self.assertEqual(bool_feat.inputs, ["itemid", "mask"])

        input_data = {
            "itemid": pa.array(input_feat[0]),
            "mask": pa.array(input_feat[1]),
        }
        parsed_feat = bool_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "bool_mask_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


if __name__ == "__main__":
    unittest.main()
