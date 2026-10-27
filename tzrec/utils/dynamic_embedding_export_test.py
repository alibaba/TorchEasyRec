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
import json
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

    @parameterized.expand(
        [("npz_only", False), ("upload", True)],
        name_func=parameterized_name_func,
    )
    def test_key_files_are_reread_only_for_upload(self, _name, upload) -> None:
        directory = self._write_shard(0, np.arange(3), np.ones((3, 2)))
        arrays, _ = get_dynamic_embedding_export(
            self.test_dir,
            {self.table: 2},
            16,
            on_values=mock.Mock() if upload else None,
        )
        with mock.patch.object(dynamic_embedding_export, "open", wraps=open) as reads:
            savez_streaming(os.path.join(self.test_dir, "export.npz"), arrays)
        key_file = os.path.join(directory, "table_emb_keys.rank_0.world_size_1")
        self.assertEqual(
            sum(call.args[0] == key_file for call in reads.call_args_list),
            2 if upload else 1,
        )

    def test_checkpoint_shards_in_numeric_rank_order(self) -> None:
        for rank in reversed(range(12)):
            self._write_shard(
                rank, np.array([rank]), np.full((1, 2), rank), world_size=12
            )
        arrays, metadata = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 16
        )
        np.testing.assert_array_equal(
            np.concatenate(list(arrays[f"{self.table}.keys"].chunks)),
            np.arange(12),
        )
        self.assertEqual(metadata[self.table]["shape"], [12, 2])

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

    @parameterized.expand([("values",), ("scores",)], name_func=parameterized_name_func)
    def test_missing_checkpoint_file(self, field) -> None:
        directory = self._write_shard(0, np.array([1]), np.ones((1, 2)))
        path = os.path.join(directory, f"table_emb_{field}.rank_0.world_size_1")
        os.unlink(path)
        with self.assertRaises(FileNotFoundError) as error:
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 16)
        self.assertIn(self.table, str(error.exception))
        self.assertIn(path, str(error.exception))

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
        [
            ("keys", "keys"),
            ("scores", "scores"),
            ("values", "keys"),
            ("values", "values"),
        ],
        name_func=parameterized_name_func,
    )
    def test_file_growth_during_iteration(self, array_field, file_field) -> None:
        directory = self._write_shard(0, np.array([1, 2]), np.ones((2, 2)))
        arrays, _ = get_dynamic_embedding_export(
            self.test_dir,
            {self.table: 2},
            16,
            on_values=mock.Mock()
            if array_field == "values" and file_field == "keys"
            else None,
        )
        chunks = arrays[f"{self.table}.{array_field}"].chunks
        next(chunks)
        path = os.path.join(directory, f"table_emb_{file_field}.rank_0.world_size_1")
        with open(path, "ab") as stream:
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "file grew during export"):
            list(chunks)

    @parameterized.expand(
        [
            ("zero_dimension", 0, 1, 0, "Invalid embedding dimension"),
            ("negative_dimension", 0, 1, -1, "Invalid embedding dimension"),
            ("invalid_checkpoint_rank", 1, 1, 2, "Invalid checkpoint rank"),
            ("zero_checkpoint_world_size", 0, 0, 2, "Invalid checkpoint rank"),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_checkpoint_contract(
        self, _name, rank, world_size, dimension, error
    ) -> None:
        self._write_shard(rank, np.array([1]), np.ones((1, 2)), world_size=world_size)
        with self.assertRaisesRegex(ValueError, error):
            get_dynamic_embedding_export(self.test_dir, {self.table: dimension}, 16)

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


def _has_rocksdb() -> bool:
    try:
        import tzrec_rocksdb  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_rocksdb(), "tzrec_rocksdb is not installed")
class NativeCheckpointExportTest(unittest.TestCase):
    """Export streams RocksDB native checkpoints like dynamicemb dump files."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir("dynamic_embedding_export_native_")
        self.addCleanup(shutil.rmtree, self.test_dir)
        self.namespace = "model.embedding_group.emb_impls.group.ebc.table"
        self.table = "model.embedding_group.emb_impls.group.ebc.embedding_bags.table"

    def _write_rank(self, rank, world, keys, values, dtype):
        import torch
        from tzrec_rocksdb import BackendConfig, open_session
        from tzrec_rocksdb.checkpoint import create_checkpoint

        root = os.path.join(self.test_dir, f"db{rank}")
        session = open_session(
            BackendConfig(root, 8 << 20, 8 << 20, 1 << 20, io_threads=0),
            os.path.join(root, "work"),
        )
        dim, state_dim = values.shape[1], 2
        elem = torch.empty(0, dtype=dtype).element_size()
        layout = {
            "dtype": str(dtype).replace("torch.", ""),
            "dim": dim,
            "state_dim": state_dim,
            "opt_type": "adam",
            "record_bytes": 8 + (dim + state_dim) * elem,
        }
        ns = session.namespace_ids([self.namespace])[self.namespace]
        session.set_table_meta(self.namespace, layout)
        rows = torch.zeros(len(keys), dim + state_dim, dtype=dtype)
        rows[:, :dim] = torch.from_numpy(values).to(dtype)
        session.db.write(
            np.array(
                [[ns, dim * elem, state_dim * elem, dim * elem, 0]], dtype=np.int64
            ),
            np.zeros(len(keys), dtype=np.int64),
            keys.astype(np.int64).view(np.uint64),
            rows.view(torch.uint8).numpy(),
            (keys + 100).astype(np.int64),
            False,
        )
        native = os.path.join(root, "ckpt")
        create_checkpoint(session, native)
        session.close()
        os.makedirs(os.path.join(self.test_dir, "rocksdb"), exist_ok=True)
        manifest = {
            "rank": rank,
            "world_size": world,
            "native_checkpoint": native,
            "groups": [
                {"namespaces": [self.namespace], "layouts": {self.namespace: layout}}
            ],
        }
        with open(
            os.path.join(self.test_dir, "rocksdb", f"rank_{rank}.json"), "w"
        ) as f:
            json.dump(manifest, f)

    @parameterized.expand(
        [("fp32", "float32"), ("bf16", "bfloat16")], name_func=parameterized_name_func
    )
    def test_native_npz_and_callback(self, _name, dtype_name) -> None:
        import torch

        dtype = getattr(torch, dtype_name)
        keys = np.array([-3, 0, 5, 7, 2**62, 9], dtype=np.int64)
        values = (np.arange(12, dtype=np.float32).reshape(6, 2) - 4) / 2
        self._write_rank(1, 2, keys[3:], values[3:], dtype)
        self._write_rank(0, 2, keys[:3], values[:3], dtype)
        callback = mock.Mock()
        arrays, metadata = get_dynamic_embedding_export(
            self.test_dir, {self.table: 2}, 32, on_values=callback
        )
        self.assertEqual(metadata[self.table]["shape"], [6, 2])
        callback.assert_not_called()
        path = os.path.join(self.test_dir, "export.npz")
        savez_streaming(path, arrays)
        # RocksDB stores ids in unsigned key order within each rank.
        order = np.concatenate(
            [
                np.argsort(keys[:3].view(np.uint64)),
                3 + np.argsort(keys[3:].view(np.uint64)),
            ]
        )
        with np.load(path) as actual:
            np.testing.assert_array_equal(actual[f"{self.table}.keys"], keys[order])
            np.testing.assert_array_equal(
                actual[f"{self.table}.scores"], keys[order] + 100
            )
            np.testing.assert_array_equal(actual[f"{self.table}.values"], values[order])
            self.assertEqual(actual[f"{self.table}.values"].dtype, np.float32)
        sizes = [len(call.args[1]) for call in callback.call_args_list]
        self.assertEqual(sum(sizes), 6)
        self.assertLessEqual(max(sizes), 32 // (8 + 2 * 4))

    def test_missing_rank_rejected(self) -> None:
        import torch

        self._write_rank(
            0, 2, np.arange(3), np.zeros((3, 2), np.float32), torch.float32
        )
        with self.assertRaisesRegex(ValueError, "ranks"):
            get_dynamic_embedding_export(self.test_dir, {self.table: 2}, 1 << 20)


if __name__ == "__main__":
    unittest.main()
