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

import os
import shutil
import struct
import unittest
import zipfile
from unittest import mock

import numpy as np
from parameterized import parameterized

from tzrec.utils import npz_util
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class NpzUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _path(self, name: str = "out.npz") -> str:
        return os.path.join(self.test_dir, name)

    @parameterized.expand(
        [
            ("float32_2d", np.float32, (3, 4)),
            ("int64_1d", np.int64, (5,)),
            ("uint8_2d", np.uint8, (2, 4)),
            ("float64_1d", np.float64, (7,)),
        ],
        name_func=parameterized_name_func,
    )
    def test_savez_streaming_roundtrip(self, _name, dtype, shape):
        arrays = {
            "model.ebc.user_id_emb": np.arange(np.prod(shape), dtype=dtype).reshape(
                shape
            ),
            "user_id_emb.keys": np.arange(shape[0], dtype=np.int64),
            "user_id_emb.scores": np.arange(shape[0], dtype=np.int64),
        }
        path = self._path()
        npz_util.savez_streaming(path, arrays)

        loaded = np.load(path)
        for k, arr in arrays.items():
            got = loaded[f"{k}.npy"]
            self.assertEqual(got.dtype, arr.dtype)
            self.assertEqual(got.shape, arr.shape)
            np.testing.assert_array_equal(got, arr)

    def test_interchangeable_with_np_savez(self):
        arrays = {
            "a": np.arange(12, dtype=np.float32).reshape(3, 4),
            "b": np.array([1, 2, 3], dtype=np.int64),
        }
        stream_path = self._path("stream.npz")
        savez_path = self._path("savez.npz")
        npz_util.savez_streaming(stream_path, arrays)
        np.savez(savez_path, **arrays)

        stream_loaded = {k: v for k, v in np.load(stream_path).items()}
        savez_loaded = {k: v for k, v in np.load(savez_path).items()}
        self.assertEqual(set(stream_loaded), set(savez_loaded))
        for k in stream_loaded:
            np.testing.assert_array_equal(stream_loaded[k], savez_loaded[k])

    def test_uses_streaming_data_descriptor_and_zip64(self):
        # gp_flag bit 3 (data descriptor) => zipfile did not seek back to patch
        # the local header; version_needed 45 => zip64, required for large shards.
        path = self._path()
        npz_util.savez_streaming(path, {"a": np.zeros(4, dtype=np.float32)})
        with open(path, "rb") as f:
            self.assertEqual(f.read(4), b"PK\x03\x04")
            version_needed = struct.unpack("<H", f.read(2))[0]
            gp_flag = struct.unpack("<H", f.read(2))[0]
        self.assertTrue(gp_flag & 0x08, f"bit-3 not set: gp_flag=0x{gp_flag:04x}")
        self.assertEqual(version_needed, 45)

    def test_large_entry_zip64_path(self):
        # Lower the zip64 threshold so a small array exercises the >2GiB
        # central-directory/EOCD64 machinery without allocating gigabytes.
        arr = np.arange(1000, dtype=np.uint8)
        path = self._path()
        with mock.patch("zipfile.ZIP64_LIMIT", 64):
            npz_util.savez_streaming(path, {"big": arr})
        np.testing.assert_array_equal(np.load(path)["big.npy"], arr)
        with zipfile.ZipFile(path) as zf:
            self.assertIsNone(zf.testzip())

    def test_part_file_not_left_on_success(self):
        path = self._path()
        npz_util.savez_streaming(path, {"a": np.zeros(2, dtype=np.float32)})
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(f"{path}.part"))

    def test_overwrite_existing(self):
        path = self._path()
        npz_util.savez_streaming(path, {"a": np.zeros(2, dtype=np.float32)})
        new_arr = np.array([5.0, 6.0], dtype=np.float32)
        npz_util.savez_streaming(path, {"a": new_arr})
        np.testing.assert_array_equal(np.load(path)["a.npy"], new_arr)

    def test_empty_mapping(self):
        path = self._path()
        npz_util.savez_streaming(path, {})
        with zipfile.ZipFile(path) as zf:
            self.assertEqual(zf.namelist(), [])
            self.assertIsNone(zf.testzip())


if __name__ == "__main__":
    unittest.main()
