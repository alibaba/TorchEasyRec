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

from enum import Enum, unique
from typing import Optional

import torch


@unique
class Kernel(Enum):
    """TorchEasyRec kernel."""

    TRITON = "TRITON"
    PYTORCH = "PYTORCH"
    CUTLASS = "CUTLASS"


_is_ppu_arch_cached: Optional[bool] = None


def is_ppu_arch() -> bool:
    """Return True if a CUDA device is an Alibaba PPU (alixpu) accelerator."""
    global _is_ppu_arch_cached
    if _is_ppu_arch_cached is None:
        try:
            _is_ppu_arch_cached = torch.cuda.is_available() and any(
                "PPU" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            )
        except Exception:
            _is_ppu_arch_cached = False
    return _is_ppu_arch_cached
