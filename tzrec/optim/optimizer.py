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
from typing import Any, Callable, Iterable, Optional, Union

import torch
from fbgemm_gpu import split_table_batched_embeddings_ops_training
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    SplitState,
)
from torch import nn
from torch.amp import GradScaler
from torch.optim.optimizer import Optimizer
from torchrec.distributed.utils import _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE
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


class FTRL(Optimizer):
    """Placeholder for the dynamicemb FTRL sparse embedding optimizer.

    FBGEMM has no FTRL kernel, so torchrec ships no FTRL wrapper to reuse. Like
    torchrec's own placeholders this class never runs: it names the optimizer so
    that the sharding plan can resolve it, and the update happens inside the
    dynamicemb table.

    Args:
        params (Iterable[nn.Parameter]): parameters to attach the optimizer to.
        **kwargs: fused params, forwarded to the dynamicemb table.
    """

    def __init__(self, params: Iterable[nn.Parameter], **kwargs: Any) -> None:
        self._params = params
        self._kwargs = kwargs

    # pyrefly: ignore[bad-override]  # matches torchrec's placeholder optimizers
    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        """Step, never reached, the dynamicemb table applies the update."""
        raise NotImplementedError


def register_ftrl_emb_opt_type() -> None:
    """Map :class:`FTRL` to dynamicemb's optimizer type in torchrec's table.

    torchrec derives an embedding table's fused `optimizer` param from the
    in-backward optimizer class, and does so unconditionally on the
    EmbeddingCollection path, so FTRL has to be registered in that table rather
    than injected into the fused params. Must be called before planning.

    Raises:
        RuntimeError: dynamicemb is missing or predates its FTRL support.
    """
    try:
        from dynamicemb import DynamicEmbOptimType
    except ImportError as e:
        raise RuntimeError(
            "sparse ftrl_optimizer requires dynamicemb >= "
            "0.1.0+20260920.9643985; FBGEMM has no FTRL embedding kernel. "
            "Please reinstall dynamicemb, see docs/source/feature/dynamicemb.md."
        ) from e
    _OPTIMIZER_CLASS_TO_EMB_OPT_TYPE[FTRL] = DynamicEmbOptimType.FTRL


# The Adagrad optimizer in TensorFlow includes the parameter
# `initial_accumulator_value`, with a default value of 0.1.
# Here, we patch the fbgemm embedding optimizer state split helper
# to support `momentum1` (Adagrad) with the specified initial value.
_sparse_init_accumulator_value = 0.0


def set_sparse_init_accumulator_value(value: float) -> None:
    """Record the accumulator initial value for embedding tables built later.

    Takes effect at table build time, in the ``apply_split_helper`` patch below and
    in ``dynamicemb_util``'s plan-time fused params, so it must be set before
    planning; FBGEMM TBE has no such kwarg, hence this module-level switch. Used
    by Adagrad and, on dynamicemb tables, by FTRL for its squared-gradient
    accumulator.

    Args:
        value: accumulator initial value, 0.0 for optimizers without one.
    """
    global _sparse_init_accumulator_value
    _sparse_init_accumulator_value = value


def sparse_init_accumulator_value() -> float:
    """Sparse accumulator initial value, 0.0 when not configured."""
    return _sparse_init_accumulator_value


def apply_split_helper(
    persistent_state_fn: Callable[[str, torch.Tensor], None],
    set_attr_fn: Callable[
        [str, Union[torch.Tensor, list[int], list[EmbeddingLocation]]], None
    ],
    current_device: torch.device,
    use_cpu: bool,
    feature_table_map: list[int],
    # pyrefly: ignore[not-a-type]  # fbgemm's SplitState resolves to a NamedTuple fallback
    split: SplitState,
    prefix: str,
    dtype: torch.dtype,
    enforce_hbm: bool = False,
    make_dev_param: bool = False,
    dev_reshape: Optional[tuple[int, ...]] = None,
    uvm_tensors_log: Optional[list[str]] = None,
    uvm_host_mapped: bool = False,
    make_persistent: bool = False,
    preallocated_host_buffer: Optional[torch.Tensor] = None,
) -> None:
    """Patch for state split helper of FBGEMM SplitTableBatchedEmbeddingBagsCodegen."""
    init_value = sparse_init_accumulator_value()
    use_init_value = (
        init_value != 0.0 and prefix == "momentum1" and dtype.is_floating_point
    )

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
                dtype=dtype,
            )
            if not use_init_value
            else torch.full(
                (split.dev_size,),
                init_value,
                device=current_device,
                dtype=dtype,
            )
        )
        dev_buffer = (
            dev_buffer.view(*dev_reshape) if dev_reshape is not None else dev_buffer
        )
    else:
        dev_buffer = torch.empty(0, device=current_device, dtype=dtype)
    if make_dev_param:
        set_attr_fn(f"{prefix}_dev", nn.Parameter(dev_buffer))
    else:
        persistent_state_fn(f"{prefix}_dev", dev_buffer)
    if split.host_size > 0:
        if preallocated_host_buffer is not None:
            assert preallocated_host_buffer.numel() == split.host_size, (
                f"preallocated_host_buffer size mismatch for '{prefix}_host': "
                f"expected {split.host_size}, got {preallocated_host_buffer.numel()}"
            )
            assert preallocated_host_buffer.is_contiguous(), (
                f"preallocated_host_buffer for '{prefix}_host' must be contiguous"
            )
            assert preallocated_host_buffer.dim() == 1, (
                f"preallocated_host_buffer for '{prefix}_host' must be 1D, got "
                f"{preallocated_host_buffer.dim()}D with shape "
                f"{preallocated_host_buffer.shape}"
            )
            assert preallocated_host_buffer.dtype == dtype, (
                f"preallocated_host_buffer dtype mismatch for '{prefix}_host': "
                f"expected {dtype}, got {preallocated_host_buffer.dtype}"
            )
            assert preallocated_host_buffer.device == current_device, (
                f"preallocated_host_buffer device mismatch for '{prefix}_host': "
                f"expected {current_device}, got {preallocated_host_buffer.device}"
            )
            host_buffer = preallocated_host_buffer
            if use_init_value:
                host_buffer.fill_(init_value)
        else:
            host_buffer = (
                torch.zeros(
                    split.host_size,
                    device=current_device,
                    dtype=dtype,
                )
                if not use_init_value
                else torch.full(
                    (split.host_size,),
                    init_value,
                    device=current_device,
                    dtype=dtype,
                )
            )
        if dtype == torch.uint8:
            persistent_state_fn(f"{prefix}_host", host_buffer)
        # For fused TBE on MTIA, always set tensor in persistent states and not
        # nn.Parameter. This only applies to Split TBE and not QR TBE.
        elif make_persistent:
            assert not make_dev_param, "MTIA does not support OptimizerType.NONE."
            persistent_state_fn(f"{prefix}_host", host_buffer)
        else:
            set_attr_fn(f"{prefix}_host", nn.Parameter(host_buffer))
        if uvm_tensors_log is not None:
            uvm_tensors_log.append(f"{prefix}_host")
    else:
        persistent_state_fn(
            f"{prefix}_host",
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
                    dtype=dtype,
                )
                if not use_init_value
                else torch.full(
                    (split.uvm_size,),
                    init_value,
                    device=current_device,
                    dtype=dtype,
                ),
            )
        else:
            persistent_state_fn(
                f"{prefix}_uvm",
                torch.zeros(
                    (split.uvm_size,),
                    out=torch.ops.fbgemm.new_unified_tensor(
                        torch.zeros(1, device=current_device, dtype=dtype)
                        if not use_init_value
                        else torch.full(
                            (1,), init_value, device=current_device, dtype=dtype
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
            torch.empty(0, device=current_device, dtype=dtype),
        )


# fbgemm annotates its own dtype parameter as `type[torch.dtype]`, but callers pass a
# dtype instance; this replacement keeps the accurate annotation.
# pyrefly: ignore[bad-assignment]
split_table_batched_embeddings_ops_training.apply_split_helper = apply_split_helper
