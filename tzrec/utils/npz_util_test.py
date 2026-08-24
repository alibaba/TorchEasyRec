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

import glob
import io
import os
import shutil
import struct
import subprocess
import unittest
import zipfile
import zlib
from unittest import mock

import numpy as np
from parameterized import parameterized

from tzrec.utils import npz_util
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

_UNZIP = shutil.which("unzip")


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
        self.assertEqual(glob.glob(f"{path}.part.*"), [])

    def test_part_file_removed_on_serialize_failure(self):
        path = self._path()

        class _Unserializable:
            def __array__(self, dtype=None, copy=None):
                raise ValueError("cannot serialize")

        with self.assertRaises(ValueError):
            npz_util.savez_streaming(path, {"a": _Unserializable()})
        self.assertFalse(os.path.exists(path))
        self.assertEqual(glob.glob(f"{path}.part.*"), [])

    def test_part_file_removed_on_rename_failure(self):
        path = self._path()
        with mock.patch("os.replace", side_effect=OSError("simulated rename failure")):
            with self.assertRaises(OSError):
                npz_util.savez_streaming(path, {"a": np.zeros(2, dtype=np.float32)})
        self.assertFalse(os.path.exists(path))
        self.assertEqual(glob.glob(f"{path}.part.*"), [])

    def _central_directory_read(self, path):
        """Read every entry locating it through the central directory only.

        Mirrors how libzip / the TorchRecProcessor npz reader consumes the
        archive: the central directory is the sole source of truth, and the
        local headers (zeroed crc/sizes for bit-3 streaming entries) are never
        trusted for entry location or integrity. Returns {name: payload}.
        """
        with open(path, "rb") as f:
            blob = f.read()
        eocd = blob.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd, 0)
        count, cd_size, cd_off = struct.unpack_from("<HII", blob, eocd + 10)
        if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
            loc = eocd - 20
            self.assertEqual(blob[loc : loc + 4], b"PK\x06\x07")
            eocd64 = struct.unpack_from("<Q", blob, loc + 8)[0]
            self.assertEqual(blob[eocd64 : eocd64 + 4], b"PK\x06\x06")
            count = struct.unpack_from("<Q", blob, eocd64 + 32)[0]
            cd_size, cd_off = struct.unpack_from("<QQ", blob, eocd64 + 40)
        entries = {}
        pos = cd_off
        for _ in range(count):
            self.assertEqual(blob[pos : pos + 4], b"PK\x01\x02")
            crc, csize, usize = struct.unpack_from("<III", blob, pos + 16)
            nlen, elen, comment_len = struct.unpack_from("<HHH", blob, pos + 28)
            lho = struct.unpack_from("<I", blob, pos + 42)[0]
            name = blob[pos + 46 : pos + 46 + nlen].decode()
            extra = blob[pos + 46 + nlen : pos + 46 + nlen + elen]
            e = 0
            while e + 4 <= len(extra):
                tag, sz = struct.unpack_from("<HH", extra, e)
                if tag == 0x0001:
                    # only fields set to 0xFFFFFFFF appear, in fixed order
                    vals = iter(struct.unpack_from(f"<{sz // 8}Q", extra, e + 4))
                    if usize == 0xFFFFFFFF:
                        usize = next(vals)
                    if csize == 0xFFFFFFFF:
                        csize = next(vals)
                    if lho == 0xFFFFFFFF:
                        lho = next(vals)
                    break
                e += 4 + sz
            self.assertEqual(blob[lho : lho + 4], b"PK\x03\x04")
            lnlen, lelen = struct.unpack_from("<HH", blob, lho + 26)
            data = blob[lho + 30 + lnlen + lelen : lho + 30 + lnlen + lelen + csize]
            self.assertEqual(zlib.crc32(data), crc, name)
            self.assertEqual(len(data), usize, name)
            entries[name] = data
            pos += 46 + nlen + elen + comment_len
        self.assertEqual(pos, cd_off + cd_size)
        return entries

    def test_readable_via_central_directory(self):
        # np.load / libzip locate entries through the central directory and
        # ignore the (zeroed, bit-3 deferred) local-header crc/sizes; verify
        # the payload reached that way is intact and matches the input.
        arrays = {"a": np.arange(12, dtype=np.float32).reshape(3, 4)}
        path = self._path()
        npz_util.savez_streaming(path, arrays)
        entries = self._central_directory_read(path)
        self.assertEqual(set(entries), {"a.npy"})
        np.testing.assert_array_equal(
            np.load(io.BytesIO(entries["a.npy"])), arrays["a"]
        )

    def test_readable_via_central_directory_zip64(self):
        arr = np.arange(1000, dtype=np.uint8)
        path = self._path()
        with mock.patch("zipfile.ZIP64_LIMIT", 64):
            npz_util.savez_streaming(path, {"big": arr})
        entries = self._central_directory_read(path)
        np.testing.assert_array_equal(np.load(io.BytesIO(entries["big.npy"])), arr)

    @unittest.skipIf(_UNZIP is None, "unzip not installed")
    def test_external_zip_reader_accepts_streaming_output(self):
        # Serving reads the npz with libzip, not CPython zipfile. An
        # independent zip reader validates the same properties libzip relies
        # on -- that entries carrying bit-3 data descriptors and ZIP64
        # extension fields are located via the central directory and pass a
        # full crc32 check.
        path = self._path()
        arrays = {
            "small": np.zeros(2, dtype=np.float32),
            "big": np.arange(1000, dtype=np.uint8),
        }
        with mock.patch("zipfile.ZIP64_LIMIT", 64):
            npz_util.savez_streaming(path, arrays)
        assert _UNZIP is not None
        result = subprocess.run([_UNZIP, "-t", path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("No errors detected", result.stdout)

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
