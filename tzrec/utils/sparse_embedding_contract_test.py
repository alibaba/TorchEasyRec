# Copyright (c) 2025, Alibaba Group;
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

from tzrec.utils.sparse_embedding_contract import (
    build_sparse_embedding_name_map,
    resolve_sparse_embedding_name,
    sparse_embedding_role_from_state_key,
)


class SparseEmbeddingContractTest(unittest.TestCase):
    def test_single_collection_keeps_table_name(self):
        names = build_sparse_embedding_name_map(
            [("ec", "sequence_emb"), ("ebc", "user_emb")]
        )
        self.assertEqual(names[("ec", "sequence_emb")], "sequence_emb")
        self.assertEqual(names[("ebc", "user_emb")], "user_emb")

    def test_cross_collection_collision_uses_role_and_numeric_suffix(self):
        names = build_sparse_embedding_name_map(
            [
                ("ec", "shared"),
                ("ebc", "shared"),
                ("ec", "shared__ec"),
            ]
        )
        self.assertEqual(names[("ec", "shared")], "shared__ec_1")
        self.assertEqual(names[("ebc", "shared")], "shared__ebc")
        self.assertEqual(names[("ec", "shared__ec")], "shared__ec")

    def test_resolver_requires_role_for_ambiguous_table(self):
        names = build_sparse_embedding_name_map([("ec", "shared"), ("ebc", "shared")])
        with self.assertRaisesRegex(ValueError, "multiple collection"):
            resolve_sparse_embedding_name(names, "shared", None)
        self.assertEqual(
            resolve_sparse_embedding_name(names, "shared", "ec"), "shared__ec"
        )

    def test_state_key_role(self):
        self.assertEqual(
            sparse_embedding_role_from_state_key(
                "model.ebc.embedding_bags.user_emb.weight"
            ),
            "ebc",
        )
        self.assertEqual(
            sparse_embedding_role_from_state_key(
                "model.ec.embeddings.sequence_emb.weight"
            ),
            "ec",
        )
        self.assertIsNone(sparse_embedding_role_from_state_key("model.unknown.weight"))


if __name__ == "__main__":
    unittest.main()
