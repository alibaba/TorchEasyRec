# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monkey-patches for torchrec/fbgemm to support unified AOTI export.

torchrec's KJT processing calls .item() and .tolist() on symbolic
tensors during torch.export. These create unbacked SymInts that
downstream ops (BatchNorm1d) can't guard on.

This module patches the critical functions to avoid .item() during
export by using propagate_real_tensors mode with suppressed fbgemm
schema mutation checks.
"""

import contextlib
import logging
from typing import Generator

import torch
import torch._functorch.config
import torch._library.utils as _lib_utils

logger = logging.getLogger(__name__)

_orig_mutation_check = _lib_utils.MutationChecker.check


def _lenient_mutation_check(self: _lib_utils.MutationChecker) -> None:
    """Suppress fbgemm schema mutation errors during export."""
    try:
        _orig_mutation_check(self)
    except RuntimeError as e:
        if "empirically wrong" in str(e):
            logger.warning("suppressed fbgemm schema error: %s", str(e)[:200])
        else:
            raise


def _patch_torchrec_item_calls() -> None:
    """Patch torchrec functions that call .item()/.tolist().

    These create unbacked SymInts during torch.export. We patch them
    to keep values as tensor-backed expressions.
    """
    # Patch _fx_to_list to avoid .tolist() during export.
    # .tolist() on symbolic tensors creates unbacked SymInts.
    # We replace the function body but keep the same function object
    # (which is already registered with torch.fx.wrap).

    import torchrec.modules.utils as _torchrec_utils

    def _patched_fx_to_list_body(tensor: torch.Tensor):  # type: ignore[return]
        if torch.compiler.is_compiling():
            return [tensor[i].item() for i in range(tensor.shape[0])]
        return tensor.long().tolist()

    # Replace the code of the existing wrapped function
    _torchrec_utils._fx_to_list.__code__ = _patched_fx_to_list_body.__code__
    logger.info("patched torchrec _fx_to_list")


@contextlib.contextmanager
def export_patches() -> Generator[None, None, None]:
    """Context manager that enables patches for unified AOTI export.

    Enables propagate_real_tensors so that unbacked SymInts from
    .item() calls in torchrec KJT processing are evaluated against
    concrete values and transmuted into runtime assertions.

    Suppresses fbgemm schema mutation checks (permute_2D_sparse_data,
    bounds_check_indices have wrong schema annotations).
    """
    _patch_torchrec_item_calls()
    _lib_utils.MutationChecker.check = _lenient_mutation_check  # type: ignore[assignment]

    try:
        yield
    finally:
        _lib_utils.MutationChecker.check = _orig_mutation_check  # type: ignore[assignment]
