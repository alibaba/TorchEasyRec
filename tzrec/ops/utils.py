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

import copy
import os
from typing import List

import torch

from tzrec.utils.logging_util import logger


def enable_tma() -> bool:
    flag = os.environ.get("ENABLE_TMA", "0") == "1"
    if flag:
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 9:
                import triton
                from packaging import version

                if version.parse(triton.__version__) >= version.parse("3.5.0"):
                    return True
                else:
                    logger.warning("triton version lower than 3.5.0, we disable TMA.")
            else:
                logger.warning("device capability lower than 9.0, we disable TMA.")
        else:
            logger.warning("CUDA not available, we disable TMA.")
    return False


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def prev_power_of_2(n: int) -> int:
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    n = n >> 1
    return n


STATIC_MAX_SEQ_LENS: List[int] = []
USE_RUNTIME_MAX_SEQ_LEN: bool = False


def set_static_max_seq_lens(max_seq_lens: List[int]) -> None:
    global STATIC_MAX_SEQ_LENS
    STATIC_MAX_SEQ_LENS = copy.deepcopy(max_seq_lens)
    STATIC_MAX_SEQ_LENS.sort()


def set_use_runtime_max_seq_len(use_runtime_max_seq_len: bool) -> None:
    global USE_RUNTIME_MAX_SEQ_LEN
    USE_RUNTIME_MAX_SEQ_LEN = use_runtime_max_seq_len


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    global USE_RUNTIME_MAX_SEQ_LEN

    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if not torch.jit.is_scripting() and torch.compiler.is_compiling():
                torch._check(max_len >= runtime_max_seq_len)
            if max_len >= runtime_max_seq_len:
                return max_len
        max_len = STATIC_MAX_SEQ_LENS[-1]
        return max_len
