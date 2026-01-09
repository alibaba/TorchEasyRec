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


import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

import numpy as np
import pyarrow as pa
from parameterized import parameterized

from tzrec.features import (
    combo_feature,
    id_feature,
    lookup_feature,
    match_feature,
    raw_feature,
)
from tzrec.features import feature as feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2


class FeatureTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @parameterized.expand(
        [
            [
                pa.array(["1;2;3", "", None, "4"]),
                [1, 2, 3, 4],
                [3, 0, 0, 1],
                None,
                None,
                None,
            ],
            [
                pa.array([["1", "2", "3"], [], None, ["4"]]),
                [1, 2, 3, 4],
                [3, 0, 0, 1],
                None,
                False,
                None,
            ],
            [pa.array([1, 2, None, 4]), [1, 2, 4], [1, 1, 0, 1], None, None, None],
            [
                pa.array(["1;2;3", "", None, "4"]),
                [1, 2, 3, 0, 0, 4],
                [3, 1, 1, 1],
                [0],
                False,
                None,
            ],
            [
                pa.array([["1", "2", "3"], [], None, ["4"]]),
                [1, 2, 3, 0, 0, 4],
                [3, 1, 1, 1],
                [0],
                False,
                None,
            ],
            [pa.array([1, 2, None, 4]), [1, 2, 0, 4], [1, 1, 1, 1], [0], None, None],
            [
                pa.array(["1:0.1;2:0.2;3:0.3", "", None, "4:0.4"]),
                [1, 2, 3, 4],
                [3, 0, 0, 1],
                None,
                True,
                [0.1, 0.2, 0.3, 0.4],
            ],
            [
                pa.array([["1:0.1", "2:0.2", "3:0.3"], [], None, ["4:0.4"]]),
                [1, 2, 3, 4],
                [3, 0, 0, 1],
                None,
                True,
                [0.1, 0.2, 0.3, 0.4],
            ],
            [
                pa.array(
                    [{1: 0.1}, {2: 0.2}, None, {4: 0.4}],
                    type=pa.map_(pa.int64(), pa.float32()),
                ),
                [1, 2, 4],
                [1, 1, 0, 1],
                None,
                True,
                [0.1, 0.2, 0.4],
            ],
            [
                pa.array(["1:0.1;2:0.2;3:0.3", "", None, "4:0.4"]),
                [1, 2, 3, 0, 0, 4],
                [3, 1, 1, 1],
                [0],
                True,
                [0.1, 0.2, 0.3, 1.0, 1.0, 0.4],
            ],
            [
                pa.array([["1:0.1", "2:0.2", "3:0.3"], [], None, ["4:0.4"]]),
                [1, 2, 3, 0, 0, 4],
                [3, 1, 1, 1],
                [0],
                True,
                [0.1, 0.2, 0.3, 1.0, 1.0, 0.4],
            ],
            [
                pa.array(
                    [{1: 0.1}, {2: 0.2}, None, {4: 0.4}],
                    type=pa.map_(pa.int64(), pa.float32()),
                ),
                [1, 2, 0, 4],
                [1, 1, 1, 1],
                [0],
                True,
                [0.1, 0.2, 1.0, 0.4],
            ],
        ]
    )
    def test_parse_fg_encoded_sparse_feature_impl(
        self,
        input_feat,
        expected_values,
        expected_lengths,
        default_value,
        is_weighted,
        expected_weights,
    ):
        feat = input_feat
        tag_data = feature_lib._parse_fg_encoded_sparse_feature_impl(
            "tag",
            feat,
            default_value=default_value,
            multival_sep=";",
            is_weighted=is_weighted,
        )
        np.testing.assert_allclose(tag_data.values, np.array(expected_values))
        np.testing.assert_allclose(tag_data.lengths, np.array(expected_lengths))
        if is_weighted:
            np.testing.assert_allclose(tag_data.weights, np.array(expected_weights))

    @parameterized.expand(
        [
            [
                ["1:2", "0:0", "3:1", None],
                [[1.0, 2.0], [0.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
                [0, 0],
            ],
            [
                [["1", "2"], ["0", "0"], ["3", "1"], None],
                [[1.0, 2.0], [0.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
                [0, 0],
            ],
            [[1, 3, 5, None], [[1.0], [3.0], [5.0], [0.0]], [0]],
            [
                [25304793, 16777217, 16777215, None],
                [[25304792], [16777216], [16777215], [0.0]],
                [0],
            ],
            [
                ["1:2", "0:0", "3:1", "0:0"],
                [[1.0, 2.0], [0.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
                None,
            ],
            [
                [["1", "2"], ["0", "0"], ["3", "1"], ["0", "0"]],
                [[1.0, 2.0], [0.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
                None,
            ],
            [[1, 3, 5, 0], [[1.0], [3.0], [5.0], [0.0]], None],
        ]
    )
    def test_parse_fg_encoded_dense_feature_impl(
        self, input_feat, expected_values, default_value
    ):
        feat = pa.array(input_feat)
        dense_data = feature_lib._parse_fg_encoded_dense_feature_impl(
            name="dense", feat=feat, multival_sep=":", default_value=default_value
        )
        self.assertTrue(np.all(dense_data.values == np.array(expected_values)))

    def test_create_features(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="cat_a")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                combo_feature=feature_pb2.ComboFeature(feature_name="combo_c")
            ),
            feature_pb2.FeatureConfig(
                lookup_feature=feature_pb2.LookupFeature(feature_name="lookup_d")
            ),
            feature_pb2.FeatureConfig(
                match_feature=feature_pb2.MatchFeature(feature_name="match_e")
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(feature_name="cat_a")
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(feature_name="int_a")
                        ),
                    ],
                )
            ),
        ]
        features = feature_lib.create_features(feature_cfgs)
        self.assertTrue(isinstance(features[0], id_feature.IdFeature))
        self.assertTrue(isinstance(features[1], raw_feature.RawFeature))
        self.assertTrue(isinstance(features[2], combo_feature.ComboFeature))
        self.assertTrue(isinstance(features[3], lookup_feature.LookupFeature))
        self.assertTrue(isinstance(features[4], match_feature.MatchFeature))
        self.assertTrue(isinstance(features[5], id_feature.IdFeature))
        self.assertTrue(isinstance(features[6], raw_feature.RawFeature))

    def _create_test_feature_cfgs(self):
        return [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", expression="item:cat_a", num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="int_a", expression="item:int_a", boundaries=[1, 2, 3]
                )
            ),
            feature_pb2.FeatureConfig(
                combo_feature=feature_pb2.ComboFeature(
                    feature_name="combo_c",
                    expression=["user:combo_uc", "item:combo_ic"],
                    hash_bucket_size=1000,
                )
            ),
            feature_pb2.FeatureConfig(
                lookup_feature=feature_pb2.LookupFeature(
                    feature_name="lookup_d",
                    map="user:map_d",
                    key="item:key_d",
                    vocab_list=["a", "b", "c"],
                )
            ),
            feature_pb2.FeatureConfig(
                match_feature=feature_pb2.MatchFeature(
                    feature_name="match_e",
                    nested_map="user:nested_map",
                    pkey="item:key_e",
                    skey="item:key_f",
                    vocab_dict={"e": 2, "f": 3, "g": 4},
                )
            ),
            feature_pb2.FeatureConfig(
                expr_feature=feature_pb2.ExprFeature(
                    feature_name="expr_f",
                    expression="int_g+int_h",
                    variables=["item:int_g", "item:int_h"],
                    boundaries=[4, 5, 6],
                )
            ),
            feature_pb2.FeatureConfig(
                tokenize_feature=feature_pb2.TokenizeFeature(
                    feature_name="token_g",
                    expression="item:token_g",
                    vocab_file="data/test/tokenizer.json",
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.IdFeature(
                    feature_name="click_seq_cat_simple",
                    expression="item:click_seq_cat_simple",
                    sequence_length=50,
                    sequence_delim=";",
                    vocab_file="data/test/id_vocab_list_0",
                    default_bucketize_value=0,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.RawFeature(
                    feature_name="click_seq_int_simple",
                    expression="user:click_seq_int_simple",
                    sequence_length=50,
                    sequence_delim=";",
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    sequence_length=50,
                    sequence_delim=";",
                    sequence_pk="user:click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_a",
                                expression="item:cat_a",
                                hash_bucket_size=10,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="int_a",
                                expression="item:int_a",
                                boundaries=[7, 8, 9],
                            )
                        ),
                    ],
                )
            ),
        ]

    @parameterized.expand([[False], [True]])
    def test_create_fg_json(self, with_asset_dir=False):
        asset_dir = None
        token_file = "data/test/tokenizer.json"
        vocab_file = "data/test/id_vocab_list_0"
        if with_asset_dir:
            self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
            asset_dir = self.test_dir
            token_file = "tokenizer_b2faab7921bbfb593973632993ca4c85.json"
            vocab_file = "id_vocab_list_0_583794bd44eb2c6d83336c71258521e8"
        feature_cfgs = self._create_test_feature_cfgs()
        features = feature_lib.create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        fg_json = feature_lib.create_fg_json(features, asset_dir=asset_dir)
        self.maxDiff = None
        self.assertEqual(
            fg_json,
            {
                "features": [
                    {
                        "feature_type": "id_feature",
                        "feature_name": "cat_a",
                        "default_value": "",
                        "expression": "item:cat_a",
                        "value_type": "string",
                        "need_prefix": False,
                        "value_dim": 0,
                        "num_buckets": 100,
                    },
                    {
                        "feature_type": "raw_feature",
                        "feature_name": "int_a",
                        "default_value": "0",
                        "expression": "item:int_a",
                        "value_type": "float",
                        "boundaries": [1.0, 2.0, 3.0],
                    },
                    {
                        "feature_type": "combo_feature",
                        "feature_name": "combo_c",
                        "default_value": "",
                        "expression": ["user:combo_uc", "item:combo_ic"],
                        "value_type": "string",
                        "need_prefix": False,
                        "value_dim": 0,
                        "hash_bucket_size": 1000,
                    },
                    {
                        "feature_type": "lookup_feature",
                        "feature_name": "lookup_d",
                        "map": "user:map_d",
                        "key": "item:key_d",
                        "default_value": "0",
                        "value_type": "string",
                        "needDiscrete": True,
                        "needKey": False,
                        "combiner": "",
                        "value_dim": 1,
                        "vocab_list": ["0", "<OOV>", "a", "b", "c"],
                        "default_bucketize_value": 1,
                    },
                    {
                        "feature_type": "match_feature",
                        "feature_name": "match_e",
                        "user": "user:nested_map",
                        "category": "item:key_e",
                        "item": "item:key_f",
                        "matchType": "hit",
                        "default_value": "0",
                        "value_type": "string",
                        "needDiscrete": True,
                        "show_category": False,
                        "show_item": False,
                        "value_dim": 1,
                        "vocab_dict": OrderedDict(
                            [("e", 2), ("f", 3), ("g", 4), ("0", 0)]
                        ),
                        "default_bucketize_value": 1,
                    },
                    {
                        "feature_name": "expr_f",
                        "feature_type": "expr_feature",
                        "expression": "int_g+int_h",
                        "variables": ["item:int_g", "item:int_h"],
                        "default_value": "0",
                        "value_type": "float",
                        "boundaries": [4.0, 5.0, 6.0],
                    },
                    {
                        "feature_name": "token_g",
                        "feature_type": "tokenize_feature",
                        "expression": "item:token_g",
                        "output_delim": "\x03",
                        "output_type": "word_id",
                        "tokenizer_type": "bpe",
                        "vocab_file": token_file,
                        "default_value": "",
                    },
                    {
                        "feature_name": "click_seq_cat_simple",
                        "feature_type": "sequence_id_feature",
                        "sequence_delim": ";",
                        "sequence_length": 50,
                        "expression": "item:click_seq_cat_simple",
                        "default_value": "0",
                        "need_prefix": False,
                        "value_type": "string",
                        "value_dim": 1,
                        "vocab_file": vocab_file,
                        "default_bucketize_value": 0,
                    },
                    {
                        "feature_name": "click_seq_int_simple",
                        "feature_type": "sequence_raw_feature",
                        "sequence_delim": ";",
                        "sequence_length": 50,
                        "expression": "user:click_seq_int_simple",
                        "default_value": "0",
                        "value_type": "float",
                    },
                    {
                        "sequence_name": "click_seq",
                        "sequence_length": 50,
                        "sequence_delim": ";",
                        "sequence_pk": "user:click_seq",
                        "features": [
                            {
                                "feature_type": "id_feature",
                                "feature_name": "cat_a",
                                "default_value": "0",
                                "expression": "item:cat_a",
                                "value_type": "string",
                                "need_prefix": False,
                                "value_dim": 1,
                                "hash_bucket_size": 10,
                            },
                            {
                                "feature_type": "raw_feature",
                                "feature_name": "int_a",
                                "default_value": "0",
                                "expression": "item:int_a",
                                "value_type": "float",
                                "boundaries": [7.0, 8.0, 9.0],
                            },
                        ],
                    },
                ]
            },
        )
        if with_asset_dir:
            self.assertTrue(os.path.exists(os.path.join(asset_dir, token_file)))

    @parameterized.expand([[False], [True]])
    def test_create_fg_json_remove_bucketizer(self, with_asset_dir=False):
        asset_dir = None
        token_file = "data/test/tokenizer.json"
        if with_asset_dir:
            self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
            asset_dir = self.test_dir
            token_file = "tokenizer_b2faab7921bbfb593973632993ca4c85.json"
        feature_cfgs = self._create_test_feature_cfgs()
        features = feature_lib.create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        fg_json = feature_lib.create_fg_json(
            features, asset_dir=asset_dir, remove_bucketizer=True
        )
        self.maxDiff = None
        self.assertEqual(
            fg_json,
            {
                "features": [
                    {
                        "feature_type": "id_feature",
                        "feature_name": "cat_a",
                        "default_value": "",
                        "expression": "item:cat_a",
                        "value_type": "string",
                        "need_prefix": False,
                        "value_dim": 0,
                    },
                    {
                        "feature_type": "raw_feature",
                        "feature_name": "int_a",
                        "default_value": "0",
                        "expression": "item:int_a",
                        "value_type": "float",
                    },
                    {
                        "feature_type": "combo_feature",
                        "feature_name": "combo_c",
                        "default_value": "",
                        "expression": ["user:combo_uc", "item:combo_ic"],
                        "value_type": "string",
                        "need_prefix": False,
                        "value_dim": 0,
                    },
                    {
                        "feature_type": "lookup_feature",
                        "feature_name": "lookup_d",
                        "map": "user:map_d",
                        "key": "item:key_d",
                        "default_value": "0",
                        "value_type": "string",
                        "needDiscrete": True,
                        "needKey": False,
                        "combiner": "",
                        "value_dim": 1,
                        "default_bucketize_value": 1,
                    },
                    {
                        "feature_type": "match_feature",
                        "feature_name": "match_e",
                        "user": "user:nested_map",
                        "category": "item:key_e",
                        "item": "item:key_f",
                        "matchType": "hit",
                        "default_value": "0",
                        "value_type": "string",
                        "needDiscrete": True,
                        "show_category": False,
                        "show_item": False,
                        "value_dim": 1,
                        "default_bucketize_value": 1,
                    },
                    {
                        "feature_name": "expr_f",
                        "feature_type": "expr_feature",
                        "expression": "int_g+int_h",
                        "variables": ["item:int_g", "item:int_h"],
                        "default_value": "0",
                        "value_type": "float",
                    },
                    {
                        "feature_name": "token_g",
                        "feature_type": "tokenize_feature",
                        "expression": "item:token_g",
                        "output_delim": "\x03",
                        "output_type": "word_id",
                        "tokenizer_type": "bpe",
                        "vocab_file": token_file,
                        "default_value": "",
                    },
                    {
                        "feature_name": "click_seq_cat_simple",
                        "feature_type": "sequence_id_feature",
                        "sequence_delim": ";",
                        "sequence_length": 50,
                        "expression": "item:click_seq_cat_simple",
                        "default_value": "0",
                        "need_prefix": False,
                        "value_type": "string",
                        "value_dim": 1,
                        "default_bucketize_value": 0,
                    },
                    {
                        "feature_name": "click_seq_int_simple",
                        "feature_type": "sequence_raw_feature",
                        "sequence_delim": ";",
                        "sequence_length": 50,
                        "expression": "user:click_seq_int_simple",
                        "default_value": "0",
                        "value_type": "float",
                    },
                    {
                        "sequence_name": "click_seq",
                        "sequence_length": 50,
                        "sequence_delim": ";",
                        "sequence_pk": "user:click_seq",
                        "features": [
                            {
                                "feature_type": "id_feature",
                                "feature_name": "cat_a",
                                "default_value": "0",
                                "expression": "item:cat_a",
                                "value_type": "string",
                                "need_prefix": False,
                                "value_dim": 1,
                            },
                            {
                                "feature_type": "raw_feature",
                                "feature_name": "int_a",
                                "default_value": "0",
                                "expression": "item:int_a",
                                "value_type": "float",
                            },
                        ],
                    },
                ]
            },
        )
        if with_asset_dir:
            self.assertTrue(os.path.exists(os.path.join(asset_dir, token_file)))

    @parameterized.expand([[False], [True]])
    def test_create_feauture_configs(self, with_asset_dir=False):
        feature_cfgs = self._create_test_feature_cfgs()
        features = feature_lib.create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)

        asset_dir = None
        token_file = "data/test/tokenizer.json"
        vocab_file = "data/test/id_vocab_list_0"
        if with_asset_dir:
            self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
            asset_dir = self.test_dir
            token_file = "tokenizer_b2faab7921bbfb593973632993ca4c85.json"
            vocab_file = "id_vocab_list_0_583794bd44eb2c6d83336c71258521e8"

        again_feature_cfgs = feature_lib.create_feature_configs(
            features, asset_dir=asset_dir
        )

        if with_asset_dir:
            feature_cfgs[6].tokenize_feature.vocab_file = token_file
            feature_cfgs[6].tokenize_feature.asset_dir = asset_dir
            feature_cfgs[7].sequence_id_feature.vocab_file = vocab_file
            feature_cfgs[7].sequence_id_feature.asset_dir = asset_dir
            self.assertTrue(os.path.exists(os.path.join(asset_dir, token_file)))
        self.assertEqual(repr(feature_cfgs), repr(again_feature_cfgs))


if __name__ == "__main__":
    unittest.main()
