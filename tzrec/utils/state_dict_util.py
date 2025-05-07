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

from tzrec.utils.logging_util import logger


def validate_state(model: nn.Module) -> None:
    """For mc modules, we should update and sort mch buffers."""
    validate_log_flag = True
    for _, m in model.named_modules():
        if hasattr(m, "validate_state"):
            if validate_log_flag:
                logger.info("validate states...")
                validate_log_flag = False
            m.validate_state()
            if isinstance(m, MCHManagedCollisionModule):
                # fix output_segments_tensor is a meta tensor.
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
