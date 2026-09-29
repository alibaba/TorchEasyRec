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

import io
import os
import shutil
import unittest
from unittest import mock

import numpy as np
from parameterized import parameterized

from tzrec.utils import dynamic_embedding_export, quant_util
from tzrec.utils.dynamic_embedding_export import get_dynamic_embedding_export
from tzrec.utils.npz_util import savez_streaming
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class DynamicEmbeddingExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir("dynamic_embedding_export_")
        self.addCleanup(shutil.rmtree, self.test_dir)
        self.module = "model.model.embedding_group.emb_impls.group.ebc"
        self.table = "model.embedding_group.emb_impls.group.ebc.embedding_bags.table"

    def _write_shard(
        self,
        rank: int,
        keys: np.ndarray,
        values: np.ndarray,
        world_size: int = 1,
        module: str = "",
    ) -> str:
        directory = os.path.join(self.test_dir, "dynamicemb", module or self.module)
        os.makedirs(directory, exist_ok=True)
        for name, array in (
            ("keys", keys.astype(np.int64)),
            ("values", values.astype(np.float32)),
            ("scores", (keys + 100).astype(np.int64)),
        ):
            array.tofile(
                os.path.join(
                    directory, f"table_emb_{name}.rank_{rank}.world_size_{world_size}"
                )
            )
        return directory

    @parameterized.expand(
        [("fp32", ""), ("int8", "QUint8RowwiseF16")],
        name_func=parameterized_name_func,
    )
    def test_npz_and_callback_match_across_shards(self, name, quant_format) -> None:
        keys = np.arange(6, dtype=np.int64)
        values = np.arange(12, dtype=np.float32).reshape(6, 2) - 4
        self._write_shard(1, keys[3:], values[3:], world_size=2)
        directory = self._write_shard(0, keys[:3], values[:3], world_size=2)
        for field in ("emb_opt_values", "counter_keys", "counter_frequencies"):
            with open(
                os.path.join(directory, f"table_{field}.rank_0.world_size_2"), "wb"
            ):
                pass
        callback = mock.Mock()
        with mock.patch("builtins.open", side_effect=AssertionError("Payload read")):
            arrays, metadata = get_dynamic_embedding_export(
                self.test_dir,
                {self.table: 2},
                32,
                quant_format=quant_format,
                on_values=callback,
            )
        self.assertEqual(metadata[self.table]["shape"], [6, 2])
        self.assertEqual(metadata[self.table]["key_name"], f"{self.table}.keys")
        self.assertEqual(metadata[self.table]["score_dtype"], "int64")
        callback.assert_not_called()
        path = os.path.join(self.test_dir, "export.npz")
        savez_streaming(path, arrays)
        exported_values = values
        if quant_format:
            exported_values = quant_util.distributed_quantize_embeddings(
                values, 2, self.table, quant_format
            )
            self.assertEqual(metadata[self.table]["storage_shape"], [6, 6])
        self.assertEqual(metadata[self.table]["memory"], exported_values.nbytes)
        with np.load(path) as actual:
            np.testing.assert_array_equal(actual[f"{self.table}.keys"], keys)
            np.testing.assert_array_equal(actual[f"{self.table}.scores"], keys + 100)
            np.testing.assert_array_equal(
                actual[f"{self.table}.values"], exported_values
            )
        calls = callback.call_args_list
        self.assertEqual([len(call.args[1]) for call in calls], [2, 1, 2, 1])
        for call in calls:
            self.assertEqual(call.args[0], self.table)
            self.assertLessEqual(call.args[1].size * (8 + 2 * 4), 32)
        np.testing.assert_array_equal(
            np.concatenate([call.args[1] for call in calls]), keys
        )
        np.testing.assert_array_equal(
            np.concatenate([call.args[2] for call in calls]), exported_values
        )

    def test_rank_assignment_and_numeric_order(self) -> None:
        for rank in reversed(range(12)):
            self._write_shard(
                rank, np.array([rank]), np.full((1, 2), rank), world_size=12
            )
        arrays, metadata = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 16, rank=0, world_size=2
        )
        np.testing.assert_array_equal(
            np.concatenate(list(arrays[f"{self.table}.keys"].chunks)),
            [0, 2, 4, 6, 8, 10],
        )
        self.assertEqual(metadata[self.table]["shape"], [6, 2])
        arrays, metadata = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 16, rank=15, world_size=16
        )
        self.assertEqual(arrays, {})
        self.assertEqual(metadata, {})

    @parameterized.expand(
        [("fp32", ""), ("int8", "QUint8RowwiseF16")],
        name_func=parameterized_name_func,
    )
    def test_empty_table(self, name, quant_format) -> None:
        self._write_shard(0, np.empty(0), np.empty((0, 2)))
        callback = mock.Mock()
        arrays, metadata = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 16, quant_format, callback
        )
        path = os.path.join(self.test_dir, "empty.npz")
        savez_streaming(path, arrays)
        with np.load(path) as actual:
            self.assertEqual(actual[f"{self.table}.keys"].shape, (0,))
            self.assertEqual(
                actual[f"{self.table}.values"].shape,
                (0, 6 if quant_format else 2),
            )
        self.assertEqual(metadata[self.table]["memory"], 0)
        callback.assert_not_called()

    @parameterized.expand(
        [("ebc", "embedding_bags"), ("ec_dict.sequence", "embeddings")],
        name_func=parameterized_name_func,
    )
    def test_module_name_mapping_and_physical_alias(self, module, collection) -> None:
        prefix = "model.model.embedding_group.emb_impls.group."
        directory = self._write_shard(
            0,
            np.array([-1, 42]),
            np.ones((2, 2)),
            module=prefix + module,
        )
        user_module = module.replace("ebc", "ebc_user").replace(
            "ec_dict", "ec_dict_user"
        )
        os.symlink(
            os.path.abspath(directory),
            os.path.join(self.test_dir, "dynamicemb", prefix + user_module),
        )
        table = f"model.embedding_group.emb_impls.group.{module}.{collection}.table"
        arrays, metadata = get_dynamic_embedding_export(self.test_dir, {table: 2}, 32)
        np.testing.assert_array_equal(
            np.concatenate(list(arrays[f"{table}.keys"].chunks)), [-1, 42]
        )
        self.assertEqual(metadata[table]["shape"], [2, 2])

    def test_unselected_table_is_skipped(self) -> None:
        self._write_shard(0, np.array([1]), np.ones((1, 2)))
        arrays, metadata = get_dynamic_embedding_export(self.test_dir, {}, 16)
        self.assertEqual(arrays, {})
        self.assertEqual(metadata, {})

    @parameterized.expand(
        [
            ("keys", b"a", "key file size"),
            ("scores", b"", "key/score row mismatch"),
            ("values", b"", "value row mismatch"),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_file_sizes_fail_before_iteration(
        self, field, payload, error
    ) -> None:
        directory = self._write_shard(0, np.array([1]), np.ones((1, 2)))
        with open(
            os.path.join(directory, f"table_emb_{field}.rank_0.world_size_1"), "wb"
        ) as stream:
            stream.write(payload)
        with self.assertRaisesRegex(ValueError, error):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 16)

    def test_missing_score_file(self) -> None:
        directory = self._write_shard(0, np.array([1]), np.ones((1, 2)))
        os.unlink(os.path.join(directory, "table_emb_scores.rank_0.world_size_1"))
        with self.assertRaises(FileNotFoundError):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 16)

    def test_inconsistent_checkpoint_world_sizes(self) -> None:
        self._write_shard(0, np.array([1]), np.ones((1, 2)), world_size=1)
        self._write_shard(1, np.array([2]), np.ones((1, 2)), world_size=2)
        with self.assertRaisesRegex(ValueError, "inconsistent checkpoint world_size"):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 16)

    def test_missing_checkpoint_rank(self) -> None:
        self._write_shard(0, np.array([1]), np.ones((1, 2)), world_size=2)
        with self.assertRaisesRegex(ValueError, "incomplete.*checkpoint shard ranks"):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 16)

    def test_duplicate_checkpoint_rank(self) -> None:
        directory = self._write_shard(0, np.array([1]), np.ones((1, 2)))
        shutil.copytree(directory, directory + "_user")
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint shard ranks"):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 16)

    def test_short_reads_and_post_stat_truncation(self) -> None:
        directory = self._write_shard(0, np.array([1, 2]), np.ones((2, 2)))
        original_open = open

        class ShortReadFile(io.BytesIO):
            def __init__(self, data, name):
                super().__init__(data)
                self.name = name

            def readinto(self, buffer):
                return super().readinto(buffer[:3])

        def short_open(path, mode):
            with original_open(path, mode) as stream:
                return ShortReadFile(stream.read(), path)

        arrays, _ = get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 32)
        with mock.patch.object(dynamic_embedding_export, "open", short_open):
            chunks = list(arrays[f"{self.table}.values"].chunks)
        np.testing.assert_array_equal(np.concatenate(chunks), np.ones((2, 2)))

        arrays, _ = get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 32)
        with open(
            os.path.join(directory, "table_emb_values.rank_0.world_size_1"), "wb"
        ):
            pass
        with self.assertRaisesRegex(ValueError, "Unexpected EOF"):
            list(arrays[f"{self.table}.values"].chunks)

    def test_callback_failure_stops_reading(self) -> None:
        self._write_shard(0, np.arange(4), np.ones((4, 2)))
        callback = mock.Mock(side_effect=RuntimeError("upload failed"))
        arrays, _ = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 16, on_values=callback
        )
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            list(arrays[f"{self.table}.values"].chunks)
        callback.assert_called_once()

    @parameterized.expand(
        [("zero", 0), ("negative", -1), ("sub_row", 15)],
        name_func=parameterized_name_func,
    )
    def test_invalid_read_budget(self, name, budget) -> None:
        self._write_shard(0, np.array([1]), np.ones((1, 2)))
        with self.assertRaises(ValueError):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, budget)

    def test_quantization_rejects_non_finite_values(self) -> None:
        self._write_shard(0, np.array([1]), np.array([[np.inf, 1]]))
        callback = mock.Mock()
        arrays, _ = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 16, "QUint8RowwiseF16", callback
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            list(arrays[f"{self.table}.values"].chunks)
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
