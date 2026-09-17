# Copyright (c) 2026, Alibaba Group;
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
import pyfg
from parameterized import param, parameterized

from tzrec.features import regex_replace_feature as regex_replace_feature_lib
from tzrec.features.feature import FgMode, create_features, create_fg_json
from tzrec.protos import feature_pb2
from tzrec.utils import test_util

# `<|endoftext|>` is an added token of data/test/tokenizer.json
_EOS = "<|endoftext|>"
_EOS_ID = 0


class RegexReplaceFeatureTest(unittest.TestCase):
    @parameterized.expand(
        [
            [["1\x032", "", None, "3"], [1, 2, 3], [2, 0, 0, 1]],
            [[[1, 2], None, None, [3]], [1, 2, 3], [2, 0, 0, 1]],
        ]
    )
    def test_fg_encoded_regex_replace_feature(
        self, input_feat, expected_values, expected_lengths
    ):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="regex_feat",
                embedding_dim=16,
                hash_bucket_size=100,
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(regex_feat_cfg)
        self.assertEqual(regex_feat.output_dim, 16)
        self.assertEqual(regex_feat.is_sparse, True)
        self.assertEqual(regex_feat.inputs, ["regex_feat"])

        parsed_feat = regex_feat.parse({"regex_feat": pa.array(input_feat)})
        self.assertEqual(parsed_feat.name, "regex_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    @parameterized.expand(
        [
            param(
                "replace_all",
                regex_pattern=["\\|"],
                replacement=" ",
                inputs=["中华|人民|共和国", "abc", None],
                vocab_list=["中华 人民 共和国", "abc"],
                expected_values=[2, 3],
            ),
            param(
                "replace_first",
                regex_pattern=["\\|"],
                replacement=" ",
                replace_all=False,
                inputs=["中华|人民|共和国", "abc", None],
                vocab_list=["中华 人民|共和国", "abc"],
                expected_values=[2, 3],
            ),
            param(
                "multi_pattern",
                regex_pattern=["\\|", "#", "\\(.*\\)"],
                replacement="",
                inputs=["a|b#c(d)", "abc", None],
                vocab_list=["abc"],
                expected_values=[2, 2],
            ),
            param(
                "icase",
                regex_pattern=["abc"],
                replacement="x",
                icase=True,
                inputs=["ABCd", "abcd", None],
                vocab_list=["xd"],
                expected_values=[2, 2],
            ),
            param(
                "truncate_and_append_eos",
                regex_pattern=["(?s)^(.{0,3}).*$"],
                replacement="\\1" + _EOS,
                replace_all=False,
                inputs=["abcdef", "中华人民共和国", None],
                vocab_list=["abc" + _EOS, "中华人" + _EOS],
                expected_values=[2, 3],
            ),
            param(
                "default_value_not_replaced",
                regex_pattern=["\\|"],
                replacement=" ",
                inputs=["a|b", None],
                default_value="x|y",
                vocab_list=["a b"],
                expected_values=[2, 0],
                expected_lengths=[1, 1],
            ),
        ],
        name_func=test_util.parameterized_name_func,
    )
    def test_regex_replace_feature(
        self,
        name,
        regex_pattern,
        replacement,
        inputs,
        vocab_list,
        expected_values,
        expected_lengths=(1, 1, 0),
        default_value="",
        replace_all=True,
        icase=False,
    ):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="regex_feat",
                embedding_dim=16,
                expression="item:title",
                regex_pattern=regex_pattern,
                replacement=replacement,
                replace_all=replace_all,
                icase=icase,
                default_value=default_value,
                vocab_list=vocab_list,
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(
            regex_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(regex_feat.output_dim, 16)
        self.assertEqual(regex_feat.is_sparse, True)
        self.assertEqual(regex_feat.inputs, ["title"])
        self.assertEqual(regex_feat.num_embeddings, len(vocab_list) + 2)

        parsed_feat = regex_feat.parse({"title": pa.array(inputs)})
        self.assertEqual(parsed_feat.name, "regex_feat")
        np.testing.assert_allclose(parsed_feat.values, np.array(expected_values))
        np.testing.assert_allclose(parsed_feat.lengths, np.array(expected_lengths))

    def test_regex_replace_feature_with_num_buckets(self):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="regex_feat",
                embedding_dim=16,
                expression="item:title",
                regex_pattern=["[^0-9]"],
                replacement="",
                num_buckets=100,
                default_value="0",
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(
            regex_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(regex_feat.num_embeddings, 100)

        parsed_feat = regex_feat.parse({"title": pa.array(["id7", "id42", None])})
        np.testing.assert_allclose(parsed_feat.values, np.array([7, 42, 0]))

    def test_regex_replace_feature_with_hash_bucket_size(self):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="regex_feat",
                embedding_dim=16,
                expression="item:title",
                regex_pattern=["\\|"],
                replacement=" ",
                hash_bucket_size=100,
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(
            regex_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(regex_feat.num_embeddings, 100)

        # "a|b" and "a b" hash to the same bucket after the replacement
        parsed_feat = regex_feat.parse({"title": pa.array(["a|b", "a b", "c|d"])})
        values = parsed_feat.values.tolist()
        self.assertEqual(values[0], values[1])
        self.assertNotEqual(values[0], values[2])
        self.assertTrue(all(0 <= v < 100 for v in values))

    def test_regex_replace_feature_with_multival_input(self):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="regex_feat",
                embedding_dim=16,
                expression="item:title",
                regex_pattern=["\\|"],
                replacement=" ",
                value_dim=0,
                vocab_list=["a b", "c d"],
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(
            regex_feat_cfg, fg_mode=FgMode.FG_NORMAL
        )
        self.assertEqual(regex_feat.fg_json()[0]["value_dim"], 0)

        parsed_feat = regex_feat.parse({"title": pa.array([["a|b", "c|d"], ["c|d"]])})
        np.testing.assert_allclose(parsed_feat.values, np.array([2, 3, 3]))
        np.testing.assert_allclose(parsed_feat.lengths, np.array([2, 1]))

    def test_regex_replace_feature_without_regex_pattern(self):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="regex_feat",
                embedding_dim=16,
                expression="item:title",
                replacement="x",
                hash_bucket_size=100,
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(regex_feat_cfg)
        with self.assertRaises(ValueError):
            regex_feat.fg_json()

    def test_tokenize_truncated_text_with_eos(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                regex_replace_feature=feature_pb2.RegexReplaceFeature(
                    feature_name="title_eos",
                    expression="item:title",
                    regex_pattern=["(?s)^(.{0,8}).*$"],
                    replacement="\\1" + _EOS,
                    replace_all=False,
                    stub_type=True,
                )
            ),
            feature_pb2.FeatureConfig(
                tokenize_feature=feature_pb2.TokenizeFeature(
                    feature_name="title_token",
                    expression="feature:title_eos",
                    embedding_dim=16,
                    vocab_file="data/test/tokenizer.json",
                    tokens_as_sequence=True,
                )
            ),
        ]
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        # pyre-ignore [16]
        fg_handler = pyfg.FgArrowHandler(create_fg_json(features), 1)
        fg_output, status = fg_handler.process_arrow(
            {"title": pa.array(["abc efg hij klm", "hij", None])}
        )
        self.assertTrue(status.ok(), status.message())

        feat_data = fg_output["title_token"]
        np.testing.assert_allclose(
            feat_data.np_values,
            np.array([19758, 299, 16054, 209, _EOS_ID, 73, 1944, _EOS_ID, 17]),
        )
        np.testing.assert_allclose(feat_data.np_lengths, np.array([5, 3, 1]))


class SequenceRegexReplaceFeatureTest(unittest.TestCase):
    def test_sequence_regex_replace_feature(self):
        regex_feat_cfg = feature_pb2.FeatureConfig(
            sequence_regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="click_50_seq_title",
                embedding_dim=16,
                expression="item:titles",
                regex_pattern=["\\|"],
                replacement=" ",
                sequence_delim=";",
                sequence_length=50,
                vocab_list=["a b", "c d"],
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(
            regex_feat_cfg, fg_mode=FgMode.FG_NORMAL, is_sequence=True
        )
        self.assertEqual(regex_feat.is_sequence, True)
        self.assertEqual(regex_feat.inputs, ["titles"])
        # fg has no sequence_regex_replace_feature, is_sequence activates it
        fg_cfg = regex_feat.fg_json()[0]
        self.assertEqual(fg_cfg["feature_type"], "regex_replace_feature")
        self.assertEqual(fg_cfg["is_sequence"], True)
        self.assertEqual(fg_cfg["sequence_delim"], ";")
        self.assertEqual(fg_cfg["sequence_length"], 50)

        parsed_feat = regex_feat.parse({"titles": pa.array(["a|b;c|d", "c|d"])})
        np.testing.assert_allclose(parsed_feat.values, np.array([2, 3, 3]))
        np.testing.assert_allclose(parsed_feat.key_lengths, np.array([1, 1, 1]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1]))

    def test_grouped_sequence_regex_replace_feature(self):
        regex_feat_cfg = feature_pb2.SeqFeatureConfig(
            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                feature_name="title_clean",
                embedding_dim=16,
                expression="item:title",
                regex_pattern=["\\|"],
                replacement=" ",
                vocab_list=["a b", "c d"],
            )
        )
        regex_feat = regex_replace_feature_lib.RegexReplaceFeature(
            regex_feat_cfg,
            fg_mode=FgMode.FG_NORMAL,
            is_sequence=True,
            sequence_name="click_50_seq",
            sequence_delim=";",
            sequence_length=50,
        )
        self.assertEqual(regex_feat.inputs, ["click_50_seq__title"])
        fg_cfg = regex_feat.fg_json()[0]
        self.assertEqual(fg_cfg["feature_type"], "regex_replace_feature")
        self.assertEqual(fg_cfg["is_sequence"], True)
        self.assertNotIn("sequence_delim", fg_cfg)

        parsed_feat = regex_feat.parse(
            {"click_50_seq__title": pa.array(["a|b;c|d", "c|d"])}
        )
        self.assertEqual(parsed_feat.name, "click_50_seq__title_clean")
        np.testing.assert_allclose(parsed_feat.values, np.array([2, 3, 3]))
        np.testing.assert_allclose(parsed_feat.seq_lengths, np.array([2, 1]))

    def test_tokenize_truncated_sequence_text_with_eos(self):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_50_seq",
                    sequence_length=50,
                    sequence_delim=";",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            regex_replace_feature=feature_pb2.RegexReplaceFeature(
                                feature_name="title_eos",
                                expression="item:title",
                                regex_pattern=["(?s)^(.{0,8}).*$"],
                                replacement="\\1" + _EOS,
                                replace_all=False,
                                stub_type=True,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            tokenize_feature=feature_pb2.TokenizeFeature(
                                feature_name="title_token",
                                expression="feature:title_eos",
                                sequence_fields=["title_eos"],
                                embedding_dim=16,
                                vocab_file="data/test/tokenizer.json",
                            )
                        ),
                    ],
                )
            )
        ]
        features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
        # pyre-ignore [16]
        fg_handler = pyfg.FgArrowHandler(create_fg_json(features), 1)
        fg_output, status = fg_handler.process_arrow(
            {
                "click_50_seq": pa.array(["a;b"]),
                "click_50_seq__title": pa.array(["abc efg hij;hij"]),
            }
        )
        self.assertTrue(status.ok(), status.message())

        feat_data = fg_output["click_50_seq__title_token"]
        np.testing.assert_allclose(
            feat_data.np_values,
            np.array([19758, 299, 16054, 209, _EOS_ID, 73, 1944, _EOS_ID]),
        )
        np.testing.assert_allclose(feat_data.np_key_lengths, np.array([5, 3]))
        np.testing.assert_allclose(feat_data.np_lengths, np.array([2]))


if __name__ == "__main__":
    unittest.main()
