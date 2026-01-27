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
import unittest
from functools import partial

import numpy as np
import pyarrow as pa
import pyfg
import torch
from parameterized import parameterized
from torch import nn
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)

from tzrec.datasets.utils import C_SAMPLE_MASK
from tzrec.features import id_feature as id_feature_lib
from tzrec.features.feature import FgMode
from tzrec.protos import feature_pb2
from tzrec.utils import test_util


class IdFeatureTest(unittest.TestCase):
    def tearDown(self):
        if "USE_FARM_HASH_TO_BUCKETIZE" in os.environ:
            os.environ.pop("USE_FARM_HASH_TO_BUCKETIZE")
            pyfg.unset_env("USE_FARM_HASH_TO_BUCKETIZE")

    @parameterized.expand(
        [
            [["1\x032", "", None, "3"], "", [1, 2, 3], [2, 0, 0, 1]],
            [["1\x032", "", None, "3"], "0", [1, 2, 0, 0, 3], [2, 1, 1, 1]],
            [[1, 2, None, 3], "", [1, 2, 3], [1, 1, 0, 1]],
            [[1, 2, None, 3], "0", [1, 2, 0, 3], [1, 1, 1, 1]],
        ]
    )
    def test_fg_encoded_id_feature(
        self, input_feat, default_value, expected_values, expected_lengths
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                fg_encoded_default_value=default_value,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        self.assertEqual(id_feat.output_dim, 16)
        self.assertEqual(id_feat.is_sparse, True)
        self.assertEqual(id_feat.inputs, ["id_feat"])

        input_data = {"id_feat": pa.array(input_feat)}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_init_fn_id_feature(self):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                num_buckets=100,
                init_fn="nn.init.uniform_,b=0.01",
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
            init_fn=partial(nn.init.uniform_, b=0.01),
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            init_fn=partial(nn.init.uniform_, b=0.01),
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

    @parameterized.expand(
        [
            ["lambda x: probabilistic_threshold_filter(x,0.05)"],
            ["lambda x: (x > 10, 10)"],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_zch_id_feature(self, threshold_filtering_func):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                zch=feature_pb2.ZeroCollisionHash(
                    zch_size=100,
                    eviction_interval=5,
                    distance_lfu=feature_pb2.DistanceLFU_EvictionPolicy(
                        decay_exponent=1.0,
                    ),
                    threshold_filtering_func=threshold_filtering_func,
                ),
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))
        mc_module = id_feat.mc_module(torch.device("meta"))
        self.assertEqual(mc_module._zch_size, 100)
        self.assertEqual(mc_module._eviction_interval, 5)
        self.assertTrue(
            mc_module._eviction_policy._threshold_filtering_func is not None
        )

    @parameterized.expand(
        [
            [pa.array(["1:1.0", "2:1.5\x033:2.0", "4:2.5"])],
            [
                pa.array(
                    [["1:1.0"], ["2:1.5", "3:2.0"], ["4:2.5"]],
                    type=pa.list_(pa.string()),
                )
            ],
            [
                pa.array(
                    [{1: 1.0}, {2: 1.5, 3: 2.0}, {4: 2.5}],
                    type=pa.map_(pa.int64(), pa.float32()),
                )
            ],
            [
                pa.array(
                    [{"1": 1.0}, {"2": 1.5, "3": 2.0}, {"4": 2.5}],
                    type=pa.map_(pa.string(), pa.float32()),
                )
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_fg_encoded_with_weighted(self, inputs):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cate",
                hash_bucket_size=10,
                embedding_dim=16,
                expression="item:cate",
                weighted=True,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        self.assertEqual(id_feat.inputs[0], "cate")

        parsed_feat = id_feat.parse({"cate": inputs})
        expected_values = [1, 2, 3, 4]
        expected_lengths = [1, 2, 1]
        expected_weights = [1.0, 1.5, 2.0, 2.5]
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))
        np.testing.assert_allclose(parsed_feat.weights, np.array(expected_weights))

    def test_fg_encoded_id_feature_with_mask(self):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                use_mask=True,
                fg_encoded_default_value="",
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg)
        self.assertEqual(id_feat.output_dim, 16)
        self.assertEqual(id_feat.is_sparse, True)
        self.assertEqual(id_feat.inputs, ["id_feat"])

        input_data = {
            "id_feat": pa.array(["1\x032", "", None, "3"]),
            C_SAMPLE_MASK: pa.array([True, False, False, False]),
        }
        np.random.seed(42)
        parsed_feat = id_feat.parse(input_data, is_training=True)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array([3]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([0, 0, 0, 1]))

    @parameterized.expand(
        [
            [pa.array(["123:0.5", "1391:0.3", None, "12:0.9\035123:0.21", ""])],
            [
                pa.array(
                    [{"123": 0.5}, {"1391": 0.3}, None, {"12": 0.9, "123": 0.21}, {}],
                    type=pa.map_(pa.string(), pa.float32()),
                )
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_weighted(self, inputs):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cate",
                num_buckets=100000,
                embedding_dim=16,
                expression="item:cate",
                weighted=True,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)
        self.assertEqual(id_feat.inputs, ["cate"])

        parsed_feat = id_feat.parse({"cate": inputs})
        self.assertEqual(parsed_feat.name, "cate")

        tag_idx = np.argsort(parsed_feat.values[2:])
        parsed_values = np.concatenate(
            [parsed_feat.values[:2], parsed_feat.values[2 + tag_idx]]
        )
        parsed_weights = np.concatenate(
            [parsed_feat.weights[:2], parsed_feat.weights[2 + tag_idx]]
        )
        np.testing.assert_allclose(parsed_values, np.array([123, 1391, 12, 123]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([1, 1, 0, 2, 0]))
        self.assertTrue(np.allclose(parsed_weights, np.array([0.5, 0.3, 0.9, 0.21])))

    @parameterized.expand(
        [
            ["", ["abc\x1defg", None, "hij"], [33, 44, 66], [2, 0, 1], [], False],
            [
                "xyz",
                ["abc\x1defg", None, "hij"],
                [33, 44, 13, 66],
                [2, 1, 1],
                13,
                False,
            ],
            [
                "xyz",
                [["abc", "efg"], None, ["hij"]],
                [33, 44, 13, 66],
                [2, 1, 1],
                13,
                False,
            ],
            ["", [1, 2, None, 3], [95, 70, 13], [1, 1, 0, 1], [], False],
            ["4", [1, 2, None, 3], [95, 70, 56, 13], [1, 1, 1, 1], 56, False],
            ["", [1, 2, None, 3], [49, 59, 21], [1, 1, 0, 1], [], True],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_hash_bucket_size(
        self,
        default_value,
        input_data,
        expected_values,
        expected_lengths,
        expected_fg_default,
        use_farm_hash=False,
    ):
        if use_farm_hash:
            os.environ["USE_FARM_HASH_TO_BUCKETIZE"] = "true"
            pyfg.set_env("USE_FARM_HASH_TO_BUCKETIZE", "true")
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                expression="user:id_input",
                default_value=default_value,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)
        self.assertEqual(id_feat.inputs, ["id_input"])

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        fg_default = id_feat.fg_encoded_default_value()
        if expected_fg_default:
            np.testing.assert_allclose(fg_default, expected_fg_default)
        else:
            self.assertEqual(fg_default, expected_fg_default)
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_input": pa.array(input_data)}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", ["abc", "efg"], None, [2, 3, 1], [2, 0, 1]],
            ["xyz", ["abc", "efg"], None, [2, 3, 0, 1], [2, 1, 1]],
            ["abc", ["abc", "efg"], None, [2, 3, 2, 1], [2, 1, 1]],
            ["", ["xyz", "abc", "efg"], 0, [1, 2, 0], [2, 0, 1]],
            ["xyz", ["xyz", "abc", "efg"], 0, [1, 2, 0, 0], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_vocab_list(
        self,
        default_value,
        vocab_list,
        default_bucketize_value,
        expected_values,
        expected_lengths,
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                vocab_list=vocab_list,
                expression="user:id_str",
                pooling="mean",
                default_value=default_value,
            )
        )
        if default_bucketize_value is not None:
            id_feat_cfg.id_feature.default_bucketize_value = default_bucketize_value

        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=4 if default_bucketize_value is None else 3,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.MEAN,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=4 if default_bucketize_value is None else 3,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_str": pa.array(["abc\x1defg", "", "hij"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", {"abc": 2, "efg": 2}, None, [2, 2, 1], [2, 0, 1]],
            ["xyz", {"abc": 2, "efg": 2}, None, [2, 2, 0, 1], [2, 1, 1]],
            ["abc", {"abc": 2, "efg": 2}, None, [0, 2, 0, 1], [2, 1, 1]],
            ["", {"abc": 1, "efg": 1}, 0, [1, 1, 0], [2, 0, 1]],
            ["xyz", {"xyz": 0, "abc": 1, "efg": 1}, 0, [1, 1, 0, 0], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_vocab_dict(
        self,
        default_value,
        vocab_dict,
        default_bucketize_value,
        expected_values,
        expected_lengths,
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                vocab_dict=vocab_dict,
                expression="user:id_str",
                pooling="mean",
                default_value=default_value,
            )
        )
        if default_bucketize_value is not None:
            id_feat_cfg.id_feature.default_bucketize_value = default_bucketize_value

        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=3 if default_bucketize_value is None else 2,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.MEAN,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=3 if default_bucketize_value is None else 2,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_str": pa.array(["abc\x1defg", "", "hij"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [["", [0, 1, 2], [2, 0, 1]], ["3", [0, 1, 3, 2], [2, 1, 1]]],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_num_buckets(
        self, default_value, expected_values, expected_lengths
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                num_buckets=100,
                expression="user:id_int",
                default_value=default_value,
            )
        )
        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.SUM,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_int": pa.array(["0\x1d1", "", "2"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            ["", "data/test/id_vocab_list_0", 4, [2, 3, 1], [2, 0, 1]],
            ["xyz", "data/test/id_vocab_list_1", 4, [2, 3, 0, 1], [2, 1, 1]],
            ["", "data/test/id_vocab_dict_2", 3, [2, 2, 1], [2, 0, 1]],
            ["xyz", "data/test/id_vocab_dict_3", 3, [2, 2, 0, 1], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_id_feature_with_vocab_file(
        self,
        default_value,
        vocab_file,
        expected_num_embeddings,
        expected_values,
        expected_lengths,
    ):
        id_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                vocab_file=vocab_file,
                default_bucketize_value=1,
                expression="user:id_str",
                pooling="mean",
                default_value=default_value,
            )
        )

        id_feat = id_feature_lib.IdFeature(id_feat_cfg, fg_mode=FgMode.FG_NORMAL)

        expected_emb_bag_config = EmbeddingBagConfig(
            num_embeddings=expected_num_embeddings,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
            pooling=PoolingType.MEAN,
        )
        self.assertEqual(repr(id_feat.emb_bag_config), repr(expected_emb_bag_config))
        expected_emb_config = EmbeddingConfig(
            num_embeddings=expected_num_embeddings,
            embedding_dim=16,
            name="id_feat_emb",
            feature_names=["id_feat"],
        )
        self.assertEqual(repr(id_feat.emb_config), repr(expected_emb_config))

        input_data = {"id_str": pa.array(["abc\x1defg", "", "hij"])}
        parsed_feat = id_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))


class SequenceIdFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [
                ["1;2", "", "3", "4\x035;6", None],
                "",
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 2, 1],
                [2, 0, 1, 2, 0],
            ],
            [
                [[[1], [2]], [], [[3]], [[4, 5], [6]], None],
                "",
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 2, 1],
                [2, 0, 1, 2, 0],
            ],
            [
                [[1, 2], [], [3], [4, 6], None],
                "",
                [1, 2, 3, 4, 6],
                [1, 1, 1, 1, 1],
                [2, 0, 1, 2, 0],
            ],
            [
                ["1;2", "", "3", "4\x035;6", None],
                "0",
                [1, 2, 0, 3, 4, 5, 6, 0],
                [1, 1, 1, 1, 2, 1, 1],
                [2, 1, 1, 2, 1],
            ],
            [
                [[[1], [2]], [], [[3]], [[4, 5], [6]], None],
                "0",
                [1, 2, 0, 3, 4, 5, 6, 0],
                [1, 1, 1, 1, 2, 1, 1],
                [2, 1, 1, 2, 1],
            ],
            [
                [[1, 2], [], [3], [4, 6], None],
                "0",
                [1, 2, 0, 3, 4, 6, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [2, 1, 1, 2, 1],
            ],
        ],
    )
    def test_fg_encoded_sequence_id_feature(
        self,
        input_feat,
        default_value,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                embedding_dim=16,
                fg_encoded_default_value=default_value,
            )
        )
        seq_feat = id_feature_lib.IdFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
        )
        input_data = {"click_50_seq__id_feat": pa.array(input_feat)}
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__id_feat"])

        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array(expected_lengths))
        np.testing.assert_allclose(
            parsed_feat.seq_lengths, np.array(expected_seq_lengths)
        )

    def test_fg_encoded_simple_sequence_id_feature(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.IdFeature(
                feature_name="click_50_seq_id",
                sequence_delim=";",
                sequence_length=50,
                embedding_dim=16,
            )
        )
        seq_feat = id_feature_lib.IdFeature(
            seq_feat_cfg,
            is_sequence=True,
        )
        input_data = {"click_50_seq_id": pa.array(["1;2", "", "3", "4\x035;6"])}
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq_id"])

        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_id")
        np.testing.assert_allclose(parsed_feat.values, np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array([1, 1, 1, 2, 1]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 0, 1, 2]))

    @parameterized.expand(
        [
            ["", 0, [33, 44, 66, 26, 66], [2, 1, 1, 1], [2, 1, 1]],
            ["xyz", 0, [33, 44, 66, 13, 66], [2, 1, 1, 1], [2, 1, 1]],
            ["", 1, [33, 66, 26, 66], [1, 1, 1, 1], [2, 1, 1]],
            ["xyz", 1, [33, 66, 13, 66], [1, 1, 1, 1], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_id_feature_with_hash_bucket_size(
        self,
        default_value,
        value_dim,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                expression="user:id_str",
                default_value=default_value,
                value_dim=value_dim,
            )
        )
        seq_feat = id_feature_lib.IdFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__id_str"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="click_50_seq__id_feat_emb",
            feature_names=["click_50_seq__id_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq__id_str": pa.array(["abc\x1defg;hij", "", "hij"])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    @parameterized.expand(
        [
            ["", 0, [33, 44, 66, 26, 66], [2, 1, 1, 1], [2, 1, 1]],
            ["xyz", 0, [33, 44, 66, 13, 66], [2, 1, 1, 1], [2, 1, 1]],
            ["", 1, [33, 66, 26, 66], [1, 1, 1, 1], [2, 1, 1]],
            ["xyz", 1, [33, 66, 13, 66], [1, 1, 1, 1], [2, 1, 1]],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_simple_sequence_id_feature_with_hash_bucket_size(
        self,
        default_value,
        value_dim,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.IdFeature(
                feature_name="click_50_seq_id_feat",
                hash_bucket_size=100,
                embedding_dim=16,
                expression="user:click_50_seq_id_str",
                sequence_delim=";",
                sequence_length=50,
                default_value=default_value,
                value_dim=value_dim,
            )
        )
        seq_feat = id_feature_lib.IdFeature(
            seq_feat_cfg,
            is_sequence=True,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq_id_str"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=100,
            embedding_dim=16,
            name="click_50_seq_id_feat_emb",
            feature_names=["click_50_seq_id_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq_id_str": pa.array(["abc\x1defg;hij", "", "hij"])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq_id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    @parameterized.expand(
        [
            [
                "",
                0,
                ["1;2", "", "3", "4\0355;6"],
                [1, 2, 0, 3, 4, 5, 6],
                [1, 1, 1, 1, 2, 1],
                [2, 1, 1, 2],
            ],
            [
                "",
                0,
                [["1", "2"], None, ["3"], ["4\0355", "6"]],
                [1, 2, 0, 3, 4, 5, 6],
                [1, 1, 1, 1, 2, 1],
                [2, 1, 1, 2],
            ],
            [
                "",
                0,
                [[1, 2], [], [3], [4, 6]],
                [1, 2, 0, 3, 4, 6],
                [1, 1, 1, 1, 1, 1],
                [2, 1, 1, 2],
            ],
            [
                "0",
                0,
                ["1;2", "", "3", "4\0355;6"],
                [1, 2, 0, 3, 4, 5, 6],
                [1, 1, 1, 1, 2, 1],
                [2, 1, 1, 2],
            ],
            [
                "0",
                1,
                ["1;2", "", "3", "4\0355;6"],
                [1, 2, 0, 3, 4, 6],
                [1, 1, 1, 1, 1, 1],
                [2, 1, 1, 2],
            ],
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_sequence_id_feature_with_num_buckets(
        self,
        default_value,
        value_dim,
        input_feat,
        expected_values,
        expected_lengths,
        expected_seq_lengths,
    ):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                num_buckets=10,
                embedding_dim=16,
                expression="user:id_str",
                default_value=default_value,
                value_dim=value_dim,
            )
        )
        seq_feat = id_feature_lib.IdFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        self.assertEqual(seq_feat.output_dim, 16)
        self.assertEqual(seq_feat.is_sparse, True)
        self.assertEqual(seq_feat.inputs, ["click_50_seq__id_str"])
        expected_emb_config = EmbeddingConfig(
            num_embeddings=10,
            embedding_dim=16,
            name="click_50_seq__id_feat_emb",
            feature_names=["click_50_seq__id_feat"],
        )
        self.assertEqual(repr(seq_feat.emb_config), repr(expected_emb_config))

        input_data = {"click_50_seq__id_str": pa.array(input_feat)}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array(expected_lengths))
        self.assertTrue(
            np.allclose(parsed_feat.seq_lengths, np.array(expected_seq_lengths))
        )

    def test_sequence_id_feature_with_vocab_list(self):
        seq_feat_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="id_feat",
                vocab_list=["a", "b", "c"],
                embedding_dim=16,
                expression="user:id_str",
                default_value="0",
            )
        )
        seq_feat = id_feature_lib.IdFeature(
            seq_feat_cfg,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim="|",
            sequence_length=50,
            fg_mode=FgMode.FG_NORMAL,
        )
        input_data = {"click_50_seq__id_str": pa.array(["c||a|b|b|", "", "a|b||c"])}
        parsed_feat = seq_feat.parse(input_data)
        self.assertEqual(parsed_feat.name, "click_50_seq__id_feat")
        self.assertTrue(
            np.allclose(parsed_feat.values, np.array([4, 0, 2, 3, 3, 0, 0, 2, 3, 0, 4]))
        )
        self.assertTrue(
            np.allclose(
                parsed_feat.key_lengths, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            )
        )
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([6, 1, 4]))

    # TODO(hongsheng.jhs): add max sequence length tests.


if __name__ == "__main__":
    unittest.main()
