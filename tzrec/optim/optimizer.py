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
import logging
import os
from typing import Any, Callable, Optional, Union

import torch
from fbgemm_gpu import split_table_batched_embeddings_ops_training
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    SplitState,
)
from torch import nn
from torch.amp import GradScaler
from torchrec.optim import KeyedOptimizer, OptimizerWrapper


class TZRecOptimizer(OptimizerWrapper):
    """TorchEasyRec optimizer wrapper.

    For gradient accumulate / gradient scaler etc.

    Args:
        optimizer (KeyedOptimizer): optimizer to wrap.
        grad_scaler (Optional[GradScaler]): gradient scaler.
        gradient_accumulation_steps (int): gradient accumulate steps.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        grad_scaler: Optional[GradScaler] = None,
        gradient_accumulation_steps: int = 0,
    ) -> None:
        super().__init__(optimizer)
        self._step = 0
        self._grad_scaler = grad_scaler
        self._gradient_accumulation_steps = gradient_accumulation_steps

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        if (
            self._gradient_accumulation_steps <= 1
            or self._step % self._gradient_accumulation_steps == 0
        ):
            self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        """Step."""
        self._step += 1
        if (
            self._gradient_accumulation_steps <= 1
            or self._step % self._gradient_accumulation_steps == 0
        ):
            if self._grad_scaler is not None:
                self._grad_scaler.step(self._optimizer)
                # pyre-ignore [16]
                self._grad_scaler.update()
            else:
                self._optimizer.step(closure=closure)


# The Adagrad optimizer in TensorFlow includes the parameter
# `initial_accumulator_value`, with a default value of 0.1.
# Here, we patch the fbgemm embedding optimizer state split helper
# to support `momentum1` (Adagrad) with the specified initial value.
def apply_split_helper(
    persistent_state_fn: Callable[[str, torch.Tensor], None],
    set_attr_fn: Callable[
        [str, Union[torch.Tensor, list[int], list[EmbeddingLocation]]], None
    ],
    current_device: torch.device,
    use_cpu: bool,
    feature_table_map: list[int],
    split: SplitState,
    prefix: str,
    dtype: type[torch.dtype],
    enforce_hbm: bool = False,
    make_dev_param: bool = False,
    dev_reshape: Optional[tuple[int, ...]] = None,
    uvm_tensors_log: Optional[list[str]] = None,
    uvm_host_mapped: bool = False,
) -> None:
    """Patch for state split helper of FBGEMM SplitTableBatchedEmbeddingBagsCodegen."""
    # Adagrad of tensorflow has param initial_accumulator_value with default value 0.1
    momentum1_init_value_str = os.environ.get("FBGEMM_MOMENTUM1_STATE_INIT_VALUE", None)
    init_value = 0.0
    use_init_value = (
        momentum1_init_value_str is not None
        and prefix == "momentum1"
        and dtype.is_floating_point
    )
    if use_init_value:
        init_value = float(momentum1_init_value_str)

    set_attr_fn(f"{prefix}_physical_placements", split.placements)
    set_attr_fn(f"{prefix}_physical_offsets", split.offsets)

    offsets = [split.offsets[t] for t in feature_table_map]
    placements = [split.placements[t] for t in feature_table_map]
    persistent_state_fn(
        f"{prefix}_offsets",
        torch.tensor(offsets, device=current_device, dtype=torch.int64),
    )
    persistent_state_fn(
        f"{prefix}_placements",
        torch.tensor(placements, device=current_device, dtype=torch.int32),
    )
    if split.dev_size > 0:
        dev_buffer = (
            torch.zeros(
                split.dev_size,
                device=current_device,
                # pyre-fixme[6]
                dtype=dtype,
            )
            if not use_init_value
            else torch.full(
                (split.dev_size,),
                init_value,
                device=current_device,
                # pyre-fixme[6]
                dtype=dtype,
            )
        )
        dev_buffer = (
            dev_buffer.view(*dev_reshape) if dev_reshape is not None else dev_buffer
        )
    else:
        # pyre-fixme[6]
        dev_buffer = torch.empty(0, device=current_device, dtype=dtype)
    if make_dev_param:
        set_attr_fn(f"{prefix}_dev", nn.Parameter(dev_buffer))
    else:
        persistent_state_fn(f"{prefix}_dev", dev_buffer)
    if split.host_size > 0:
        if dtype == torch.uint8:
            persistent_state_fn(
                f"{prefix}_host",
                torch.zeros(
                    split.host_size,
                    device=current_device,
                    # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                    #  3rd param but got `Type[Type[torch._dtype]]`.
                    dtype=dtype,
                ),
            )
        else:
            set_attr_fn(
                f"{prefix}_host",
                nn.Parameter(
                    torch.zeros(
                        split.host_size,
                        device=current_device,
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                        #  for 3rd param but got `Type[Type[torch._dtype]]`.
                        dtype=dtype,
                    )
                    if not use_init_value
                    else torch.full(
                        (split.host_size,),
                        init_value,
                        device=current_device,
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                        #  for 3rd param but got `Type[Type[torch._dtype]]`.
                        dtype=dtype,
                    )
                ),
            )
        if uvm_tensors_log is not None:
            uvm_tensors_log.append(f"{prefix}_host")
    else:
        persistent_state_fn(
            f"{prefix}_host",
            # pyre-fixme[6]: For 3rd param expected `dtype` but got `Type[dtype]`.
            torch.empty(0, device=current_device, dtype=dtype),
        )
    if split.uvm_size > 0:
        assert not use_cpu
        if enforce_hbm:
            logging.info("Enforce hbm for the cache location")
            persistent_state_fn(
                f"{prefix}_uvm",
                torch.zeros(
                    split.uvm_size,
                    device=current_device,
                    # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                    #  3rd param but got `Type[Type[torch._dtype]]`.
                    dtype=dtype,
                )
                if not use_init_value
                else torch.full(
                    (split.uvm_size,),
                    init_value,
                    device=current_device,
                    # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]` for
                    #  3rd param but got `Type[Type[torch._dtype]]`.
                    dtype=dtype,
                ),
            )
        else:
            persistent_state_fn(
                f"{prefix}_uvm",
                torch.zeros(
                    (split.uvm_size,),
                    out=torch.ops.fbgemm.new_unified_tensor(
                        # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                        #  for 3rd param but got `Type[Type[torch._dtype]]`.
                        torch.zeros(1, device=current_device, dtype=dtype)
                        if not use_init_value
                        else torch.full(
                            1, init_value, device=current_device, dtype=dtype
                        ),
                        [split.uvm_size],
                        is_host_mapped=uvm_host_mapped,
                    ),
                ),
            )
            if uvm_tensors_log is not None:
                uvm_tensors_log.append(f"{prefix}_uvm")
    else:
        persistent_state_fn(
            f"{prefix}_uvm",
            # pyre-fixme[6]: For 3rd param expected `dtype` but got `Type[dtype]`.
            torch.empty(0, device=current_device, dtype=dtype),
        )


split_table_batched_embeddings_ops_training.apply_split_helper = apply_split_helper
