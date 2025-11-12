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

import builtins
import glob as glob_module
import os
import shutil
import time
import traceback

import fsspec
from tensorboard.compat import tensorflow_stub
from tensorboard.compat.tensorflow_stub.io import gfile

_original_open = builtins.open
_original_makedirs = os.makedirs
_original_listdir = os.listdir
_original_remove = os.remove
_original_isdir = os.path.isdir
_original_exists = os.path.exists
_original_copy = shutil.copy
_original_glob = glob_module.glob

_CACHED_FSSPEC_FILESYSTEMS = {}


def url_to_fs(path):
    """Convert urlpath to filesystem and relative path."""
    protocol = None
    rpath = path
    if isinstance(path, str):
        protocol, rpath = fsspec.core.split_protocol(path)
    if protocol is None:
        return None, rpath
    elif protocol in _CACHED_FSSPEC_FILESYSTEMS:
        return _CACHED_FSSPEC_FILESYSTEMS[protocol], rpath
    else:
        fs, _ = fsspec.core.url_to_fs(path)
        _CACHED_FSSPEC_FILESYSTEMS[protocol] = fs
        return fs, rpath


def _patched_open(path, mode="r", *args, **kwargs):
    fs, _ = url_to_fs(path)
    if fs is not None:
        return fs.open(path, mode, *args, **kwargs)
    else:
        return _original_open(path, mode, *args, **kwargs)


def _patched_makedirs(path, mode=0o777, exist_ok=False):
    fs, _ = url_to_fs(path)
    if fs is not None:
        return fs.makedirs(path, exist_ok=exist_ok)
    else:
        return _original_makedirs(path, mode=mode, exist_ok=exist_ok)


def _patched_isdir(path):
    fs, _ = url_to_fs(path)
    if fs is not None:
        return fs.isdir(path)
    else:
        return _original_isdir(path)


def _patched_listdir(path):
    fs, _ = url_to_fs(path)
    if fs is not None:
        return fs.ls(path, detail=False)
    else:
        return _original_listdir(path)


def _patched_remove(path):
    fs, _ = url_to_fs(path)
    if fs is not None:
        return fs.rm(path)
    else:
        return _original_remove(path)


def _patched_exists(path):
    fs, _ = url_to_fs(path)
    if fs is not None:
        return fs.exists(path, check_dir=True)
    else:
        return _original_exists(path)


def _patched_copy(src, dst, *args, **kwargs):
    src_fs, _ = url_to_fs(src)
    dst_fs, _ = url_to_fs(dst)
    if src_fs is not None or dst_fs is not None:
        with src_fs.open(src, "rb") as fsrc:
            with dst_fs.open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
        return dst
    else:
        return _original_copy(src, dst, *args, **kwargs)


def _patched_glob(pattern, *args, **kwargs):
    fs, _ = url_to_fs(pattern)
    if fs is not None:
        return fs.glob(pattern, *args, **kwargs)
    else:
        return _original_glob(pattern, *args, **kwargs)


def apply_monkeypatch():
    """Apply fsspec-backed monkeypatches to builtins/os/shutil."""
    builtins.open = _patched_open
    os.makedirs = _patched_makedirs
    os.path.isdir = _patched_isdir
    os.listdir = _patched_listdir
    os.remove = _patched_remove
    os.path.exists = _patched_exists
    shutil.copy = _patched_copy
    glob_module.glob = _patched_glob


def remove_monkeypatch():
    """Restore original functions."""
    builtins.open = _original_open
    os.makedirs = _original_makedirs
    os.path.isdir = _original_isdir
    os.listdir = _original_listdir
    os.remove = _original_remove
    os.path.exists = _original_exists
    shutil.copy = _original_copy
    glob_module.glob = _original_glob


class PanguGFile(object):
    """Provide tensorboard filesystem access to pangu dfs."""

    def __init__(self):
        from pangudfs_client.high_level_client import PanguClient

        self.pangu_client = PanguClient()
        self._max_retry_time = 10

    def _format_path(self, filename):
        if isinstance(filename, bytes):
            filename = filename.decode()
        return filename

    def exists(self, filename):
        """Determines whether a path exists or not."""
        return self.pangu_client.status(self._format_path(filename), check_dir=True)

    def join(self, path, *paths):
        """Join paths with a slash."""
        return "/".join((path,) + paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        pass

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text
        """
        # Always convert to bytes for writing
        if binary_mode:
            if not isinstance(file_content, bytes):
                raise TypeError("File content type must be bytes")
        else:
            file_content = tensorflow_stub.compat.as_bytes(file_content)

        from pangudfs_client.common.exception import PanguException

        # cause hdfs client not stable ,add retry
        retry_times = 0
        while retry_times < self._max_retry_time:
            retry_times = retry_times + 1
            try:
                with self.pangu_client.buffer_append(
                    pangu_path=self._format_path(filename)
                ) as writer:
                    writer.append(file_content)
                return
            except PanguException as e:
                if e.pangu_err_no == 16:
                    break
                else:
                    traceback.print_exc()
                    time.sleep(10)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        return []

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        return self.pangu_client.is_dir(self._format_path(dirname))

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        return self.pangu_client.list(self._format_path(dirname))

    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        pass

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by ContentLength from S3,
        # but we convert to .length

        if not self.exists(self._format_path(filename)):
            file_status = self.pangu_client.status(self._format_path(filename))
            return gfile.StatData(file_status.pangu_file_stat.file_size)
        else:
            raise tensorflow_stub.errors.NotFoundError(
                None, None, "Could not find file"
            )


def register_external_filesystem():
    """Register user-defined filesystems."""
    use_fsspec = int(os.environ.get("USE_FSSPEC", "1")) == 1
    try:
        from pangudfs_client.high_level_client.extern.fsspec import PanguDfs
        from tensorboard.compat.tensorflow_stub.io import gfile

        gfile.register_filesystem("dfs", PanguGFile())
        fsspec.register_implementation("dfs", PanguDfs)
        use_fsspec = True
    except ImportError:
        pass

    if use_fsspec:
        apply_monkeypatch()
