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
from google.protobuf import struct_pb2

from tzrec.features import custom_feature as custom_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


class CustomFeatureTest(unittest.TestCase):
    def test_edit_distance(self):
        custom_feat_cfg = feature_pb2.FeatureConfig(
            custom_feature=feature_pb2.CustomFeature(
                feature_name="custom_feat",
                operator_name="EditDistance",
                operator_lib_file="pyfg/lib/libedit_distance.so",
                expression=["user:query", "item:title"],
                operator_params=struct_pb2.Struct(
                    fields={"encoding": struct_pb2.Value(string_value="utf-8")}
                ),
            )
        )
        custom_feat = custom_feature_lib.CustomFeature(
            custom_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(custom_feat.output_dim, 1)
        self.assertEqual(custom_feat.is_sparse, False)
        self.assertEqual(custom_feat.inputs, ["query", "title"])

        input_data = {
            "query": pa.array(["裙子"]),
            "title": pa.array(["连衣裙"]),
        }
        parsed_feat = custom_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "custom_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([[3]]))

    def test_edit_distance_with_boundary(self):
        custom_feat_cfg = feature_pb2.FeatureConfig(
            custom_feature=feature_pb2.CustomFeature(
                feature_name="custom_feat",
                operator_name="EditDistance",
                operator_lib_file="pyfg/lib/libedit_distance.so",
                expression=["user:query", "item:title"],
                operator_params=struct_pb2.Struct(
                    fields={"encoding": struct_pb2.Value(string_value="utf-8")}
                ),
                boundaries=[0, 5, 10],
                embedding_dim=16,
            )
        )
        custom_feat = custom_feature_lib.CustomFeature(
            custom_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(custom_feat.output_dim, 16)
        self.assertEqual(custom_feat.is_sparse, True)
        self.assertEqual(custom_feat.inputs, ["query", "title"])

        input_data = {
            "query": pa.array(["裙子"]),
            "title": pa.array(["连衣裙"]),
        }
        parsed_feat = custom_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "custom_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([1]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1]))


if __name__ == "__main__":
    unittest.main()
