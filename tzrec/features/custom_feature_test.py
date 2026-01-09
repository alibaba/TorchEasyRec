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
from parameterized import parameterized

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


class SequenceCustomFeatureTest(unittest.TestCase):
    def test_sequence_expr_feature_dense(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            custom_feature=feature_pb2.CustomFeature(
                feature_name="custom_feat",
                expression=["user:cur_time", "user:clk_time_seq"],
                operator_name="SeqExpr",
                operator_lib_file="pyfg/lib/libseq_expr.so",
                operator_params=struct_pb2.Struct(
                    fields={
                        "formula": struct_pb2.Value(
                            string_value="cur_time-clk_time_seq"
                        )
                    }
                ),
            )
        )
        seq_feat = custom_feature_lib.CustomFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim="|",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 1)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["cur_time", "clk_time_seq"])
        self.assertEqual(seq_feat.emb_config, None)

        input_data = {
            "cur_time": pa.array(["10"]),
            "clk_time_seq": pa.array(["2|3", "4"]),
        }
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__custom_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([[8], [7], [6]]))
        self.assertTrue(np.allclose(parsed_feat.seq_lengths, np.array([2, 1])))

    def test_simple_sequence_expr_feature_dense(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_custom_feature=feature_pb2.CustomFeature(
                feature_name="custom_feat",
                sequence_delim="|",
                sequence_length=50,
                expression=["user:cur_time", "user:clk_time_seq"],
                operator_name="SeqExpr",
                operator_lib_file="pyfg/lib/libseq_expr.so",
                operator_params=struct_pb2.Struct(
                    fields={
                        "formula": struct_pb2.Value(
                            string_value="cur_time-clk_time_seq"
                        )
                    }
                ),
            )
        )
        seq_feat = custom_feature_lib.CustomFeature(
            seq_feat_cfg,
            is_sequence=True,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 1)
        self.assertEqual(seq_feat.is_sparse, False)
        self.assertEqual(seq_feat.inputs, ["cur_time", "clk_time_seq"])
        self.assertEqual(seq_feat.emb_config, None)

        input_data = {
            "cur_time": pa.array(["10", None, None]),
            "clk_time_seq": pa.array(["2|3", "4", None]),
        }
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "custom_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([[8], [7], [0], [0]]))
        self.assertTrue(np.allclose(parsed_feat.seq_lengths, np.array([2, 1, 1])))

    @parameterized.expand(
        [
            [["item:ilng", "item:ilat", "user:ulng", "user:ulat"], []],
            [["user:ilng", "user:ilat", "user:ulng", "user:ulat"], ["ilng", "ilat"]],
        ]
    )
    def test_sequence_expr_feature_sparse(self, expression, sequence_fields):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            custom_feature=feature_pb2.CustomFeature(
                feature_name="custom_feat",
                expression=expression,
                operator_name="SeqExpr",
                operator_lib_file="pyfg/lib/libseq_expr.so",
                operator_params=struct_pb2.Struct(
                    fields={
                        "formula": struct_pb2.Value(string_value="spherical_distance")
                    }
                ),
                boundaries=[0, 150, 1500],
                embedding_dim=16,
                sequence_fields=sequence_fields,
            )
        )
        seq_feat = custom_feature_lib.CustomFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim="|",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(
            seq_feat.inputs,
            [
                "click_50_seq__ilng",
                "click_50_seq__ilat",
                "ulng",
                "ulat",
            ],
        )

        input_data = {
            "click_50_seq__ilng": pa.array(["113.728|116.4074", "121.4737", None]),
            "click_50_seq__ilat": pa.array(["23.002|39.9042", "31.2304", None]),
            "ulng": pa.array(["113.1057", "116.4074", None]),
            "ulat": pa.array(["22.5614", "39.9042", None]),
        }
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__custom_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([1, 3, 2, 1]))
        self.assertTrue(np.allclose(parsed_feat.seq_lengths, np.array([2, 1, 1])))


if __name__ == "__main__":
    unittest.main()
