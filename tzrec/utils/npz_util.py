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

"""Streaming npz writer for seek-broken distributed filesystems.

``np.savez`` writes a zip and then seeks back to patch each entry's local file
header with its crc32/sizes. On an OSS-mounted filesystem that seek breaks, so
export today stages a local temp file and moves it into place -- which can
overflow local disk for large dynamic embedding shards.

This module writes an npz directly to the target path with no seek on the
output file and no local temp file, by forcing CPython ``zipfile`` into
streaming mode: each entry carries a bit-3 data descriptor (crc/sizes after the
payload) and the central directory is appended at the end. The output is
readable by ``np.load`` and by libzip (the TorchRecProcessor npz reader), which
locate entries through the central directory and ignore the local header
crc/sizes.
"""

import errno
import os
import zipfile
from typing import Any, Mapping

import numpy as np


class _Unseekable:
    """Writable binary stream wrapper that forces zipfile into streaming mode.

    CPython ``ZipFile`` probes seekability on open in ``"w"`` mode by calling
    ``tell()`` then ``seek()``; raising ``ESPIPE`` from ``seek`` flips it into
    unseekable mode, after which it emits bit-3 data descriptors instead of
    seeking back to rewrite local headers.

    Attributes:
        _fp: The underlying writable binary file object.
        _n: Running count of bytes written, returned from ``tell`` so the
            counter stays correct whether or not zipfile wraps this object in
            its own ``_Tellable`` adapter.
    """

    def __init__(self, fp: Any) -> None:
        self._fp = fp
        self._n = 0

    def write(self, data: bytes) -> int:
        """Write *data* through to the underlying stream and count bytes."""
        n = self._fp.write(data)
        self._n += len(data) if n is None else n
        return n

    def flush(self) -> None:
        """Flush the underlying stream."""
        self._fp.flush()

    def seekable(self) -> bool:
        """Report the stream as unseekable so zipfile uses data descriptors."""
        return False

    def seek(self, *args: Any, **kwargs: Any) -> int:
        """Reject seeks; zipfile catches this and switches to streaming mode."""
        raise OSError(errno.ESPIPE, "unseekable output stream")

    def tell(self) -> int:
        """Return the running write offset without touching the stream."""
        return self._n

    def close(self) -> None:
        """Close the underlying stream."""
        self._fp.close()


def savez_streaming(path: str, arrays: Mapping[str, Any]) -> None:
    """Write *arrays* (name -> array-like) to *path* as an npz, fully streaming.

    Entry naming and ``.npy`` payload match ``np.savez`` (entry ``"<name>.npy"``,
    ``ZIP_STORED``, C-order little-endian), so the output is interchangeable with
    ``np.savez`` for readers that go through the central directory (``np.load``,
    libzip). Unlike ``np.savez`` it never seeks the output file and uses no
    local temp file, making it safe on OSS-mounted filesystems and immune to
    local-disk overflow on large shards.

    A ``<path>.part`` file is streamed first and renamed into place on success,
    so a crashed export leaves a ``.part`` file that the serving shard-discovery
    regex (anchored on ``.npz$``) never matches.

    Args:
        path: Destination ``.npz`` path on the (possibly seek-broken) DFS mount.
        arrays: Mapping of entry name to array-like value (numpy or torch
            tensor); names must not carry a ``.npy`` suffix, it is appended.
    """
    part = f"{path}.part"
    with open(part, "wb") as raw:
        fp = _Unseekable(raw)
        with zipfile.ZipFile(fp, "w", allowZip64=True) as zf:
            for name, arr in arrays.items():
                # force_zip64 keeps entries > ZIP64_LIMIT from raising and
                # matches np.savez, which forces zip64 on every entry.
                with zf.open(f"{name}.npy", "w", force_zip64=True) as ent:
                    np.save(ent, arr)
    os.rename(part, path)
