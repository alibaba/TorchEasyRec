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

import pyarrow as pa
from google.protobuf import text_format
from parameterized import parameterized

from tzrec.features.feature import FgMode, create_features
from tzrec.protos import feature_pb2
from tzrec.utils.test_util import parameterized_name_func


def _feature(text, fg_mode=FgMode.FG_NONE):
    fc = feature_pb2.FeatureConfig()
    text_format.Merge(f"sequence_sid_feature {{ {text} }}", fc)
    return create_features([fc], fg_mode=fg_mode)[0]


_BASE = (
    'feature_name: "user_sequence" expression: "user:user_sequence" '
    "codebook: 4 codebook: 4 codebook: 4"
)


class SidFeatureTest(unittest.TestCase):
    def test_dispatch_and_defaults(self) -> None:
        f = _feature(_BASE)
        self.assertEqual(type(f).__name__, "SidFeature")
        # the oneof FIELD name is what makes it a sequence, not the message
        self.assertTrue(f.is_sequence)
        self.assertFalse(f.is_sparse)
        self.assertEqual(f.name, "user_sequence")
        self.assertEqual(f.value_dim, 1)
        self.assertEqual(f.output_dim, 1)
        self.assertEqual(f.side_inputs, [("user", "user_sequence")])
        self.assertEqual(f.prefix_text, "")
        self.assertEqual(f.suffix_text, "")
        self.assertEqual(f.codebook, [4, 4, 4])
        self.assertEqual(f.num_levels, 3)
        self.assertEqual(f.sid_vocab_size, 12)
        self.assertEqual(f.level_offsets, [0, 4, 8])

    def test_prompt_text_round_trips(self) -> None:
        f = _feature(f'{_BASE} prefix_text: "History: " suffix_text: "."')
        self.assertEqual(f.prefix_text, "History: ")
        self.assertEqual(f.suffix_text, ".")

    @parameterized.expand(
        [[FgMode.FG_NORMAL], [FgMode.FG_DAG], [FgMode.FG_BUCKETIZE]],
        name_func=parameterized_name_func,
    )
    def test_rejects_every_fg_mode_but_none(self, fg_mode) -> None:
        # SID codes come from an offline generation model; there is no pyfg
        # counterpart, so anything but FG_NONE must fail loudly, not mis-parse.
        with self.assertRaisesRegex(ValueError, "FG_NONE only"):
            _feature(_BASE, fg_mode=fg_mode)

    def test_parse_folds_in_the_level_offsets(self) -> None:
        # offsets [0, 4, 8]: level j's 0-based code k becomes flat index k + off[j],
        # which is also the atom index -- no bridging shift anywhere.
        f = _feature(_BASE)
        parsed = f.parse({"user_sequence": pa.array([[0, 1, 2, 1, 2, 3], [0, 0, 0]])})
        self.assertEqual(
            parsed.values.flatten().tolist(), [0, 5, 10, 1, 6, 11, 0, 4, 8]
        )
        self.assertEqual(parsed.seq_lengths.tolist(), [6, 3])

    def test_parse_rejects_out_of_range_and_partial_items(self) -> None:
        f = _feature(_BASE)
        with self.assertRaisesRegex(ValueError, "local 0-based"):
            f.parse({"user_sequence": pa.array([[0, 1, 4]])})  # 4 == codebook[2]
        with self.assertRaisesRegex(ValueError, "local 0-based"):
            f.parse({"user_sequence": pa.array([[-1, 1, 2]])})
        with self.assertRaisesRegex(ValueError, "whole 3-level items"):
            f.parse({"user_sequence": pa.array([[0, 1]])})

    def test_rejects_a_bad_codebook(self) -> None:
        for bad, msg in (("", "non-empty"), ("codebook: 4 codebook: 0", "positive")):
            with self.subTest(bad=bad):
                base = 'feature_name: "s" expression: "user:s" ' + bad
                with self.assertRaisesRegex(ValueError, msg):
                    _ = _feature(base).codebook

    def test_no_embedding_table(self) -> None:
        f = _feature(_BASE)
        self.assertFalse(f.has_embedding)
        self.assertIsNone(f.emb_config)
        with self.assertRaisesRegex(RuntimeError, "no .*embedding table"):
            _ = f.num_embeddings

    def test_no_fg_representation(self) -> None:
        f = _feature(_BASE)
        with self.assertRaisesRegex(RuntimeError, "no fg representation"):
            f.fg_json()


if __name__ == "__main__":
    unittest.main()
