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
import os
import shutil
from urllib.parse import urlparse

import fsspec

_original_open = builtins.open
_original_listdir = os.listdir
_original_remove = os.remove
_original_exists = os.path.exists
_original_copy = shutil.copy


def _is_remote_path(path):
    """Detect URI scheme like s3://, gs://, etc."""
    if not isinstance(path, str):
        return False
    parsed = urlparse(path)
    return bool(parsed.scheme) and len(parsed.scheme) > 1


def _get_fs_and_path(path):
    """Return (filesystem_instance, stripped_path) for URI or local fs."""
    return fsspec.core.url_to_fs(path)


def _patched_open(path, mode="r", *args, **kwargs):
    if _is_remote_path(path):
        return fsspec.open(path, mode, **kwargs).__enter__()
    else:
        return _original_open(path, mode, *args, **kwargs)


def _patched_listdir(path):
    if _is_remote_path(path):
        fs, _, rpath = _get_fs_and_path(path)
        return [os.path.basename(p) for p in fs.ls(rpath)]
    else:
        return _original_listdir(path)


def _patched_remove(path):
    if _is_remote_path(path):
        fs, _, rpath = _get_fs_and_path(path)
        return fs.rm(rpath)
    else:
        return _original_remove(path)


def _patched_exists(path):
    if _is_remote_path(path):
        fs, _, rpath = _get_fs_and_path(path)
        return fs.exists(rpath)
    else:
        return _original_exists(path)


def _patched_copy(src, dst, *args, **kwargs):
    if _is_remote_path(src) or _is_remote_path(dst):
        # Always use fsspec filesystem copy
        # src and dst can each have their own fs
        src_fs, _, src_path = _get_fs_and_path(src)
        dst_fs, _, dst_path = _get_fs_and_path(dst)

        # Read from src
        with src_fs.open(src_path, "rb") as fsrc:
            # Write to dst
            with dst_fs.open(dst_path, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
        return dst
    else:
        # Local copy
        return _original_copy(src, dst, *args, **kwargs)


def apply_monkeypatch():
    """Apply fsspec-backed monkeypatches to builtins/os/shutil."""
    builtins.open = _patched_open
    os.listdir = _patched_listdir
    os.remove = _patched_remove
    os.path.exists = _patched_exists
    shutil.copy = _patched_copy


def remove_monkeypatch():
    """Restore original functions."""
    builtins.open = _original_open
    os.listdir = _original_listdir
    os.remove = _original_remove
    os.path.exists = _original_exists
    shutil.copy = _original_copy
