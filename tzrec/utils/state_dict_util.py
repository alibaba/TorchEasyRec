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

import torch
from torch import nn
from torchrec.modules.mc_modules import MCHManagedCollisionModule


def fix_mch_state(model: nn.Module) -> None:
    """Rebuild rank-local _output_segments_tensor of mc modules.

    The buffer can be in two bad states by the time we reach export:
      * a meta tensor (model was built on a meta device and no value was
        ever materialized);
      * an all-zeros tensor (init_parameters() materialized it via
        torch.zeros_like, and PartialLoadPlanner now skips loading this
        rank-specific buffer from checkpoint).

    In both cases the only safe value is the locally-known segments
    [_output_global_offset, _output_global_offset + _zch_size], which is
    sufficient for `validate_state()` because it checks that both ends are
    present in the segments tensor.
    """
    for _, m in model.named_modules():
        if not isinstance(m, MCHManagedCollisionModule):
            continue
        # pyre-ignore [16]
        buf = m._buffers["_output_segments_tensor"]
        needs_rebuild = buf.is_meta or bool(torch.all(buf == 0).item())
        if not needs_rebuild:
            continue
        output_segments = [
            m._output_global_offset,
            m._output_global_offset + m._zch_size,
        ]
        m._buffers["_output_segments_tensor"] = torch.tensor(
            output_segments + [-1] * (1025 - len(output_segments)),
            dtype=torch.int64,
            device=m._current_iter_tensor.device,
        )


def init_parameters(module: nn.Module, device: torch.device) -> None:
    """Init param for model with meta device type."""

    @torch.no_grad()
    def init_parameters(module: nn.Module) -> None:
        # Allocate parameters and buffers if over 'meta' device.
        has_meta_param = False
        for name, param in module._parameters.items():
            if isinstance(param, torch.Tensor) and param.device.type == "meta":
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param, device=device),
                    requires_grad=param.requires_grad,
                )
                has_meta_param = True
        for name, buffer in module._buffers.items():
            if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
                module._buffers[name] = torch.zeros_like(buffer, device=device)

        # Init parameters if at least one parameter is over 'meta' device.
        if has_meta_param and hasattr(module, "reset_parameters"):
            module.reset_parameters()

    module.apply(init_parameters)
