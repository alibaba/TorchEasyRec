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

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torchrec import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.fx import symbolic_trace as _symbolic_trace

# Modules whose forward FX cannot record -- they branch on tensor values or
# turn them into Python ints -- so tracing keeps them opaque and TorchScript
# compiles them whole. Matched by class name.
UNTRACEABLE_MODULES = ["ComputeJTDictToKJT", "PromptAssembler", "HoleKeyBuilder"]


@torch.fx.wrap
def _restore_unweighted_kjt(
    source: KeyedJaggedTensor, permuted: KeyedJaggedTensor
) -> KeyedJaggedTensor:
    """Preserve absent weights after a TorchRec feature permutation.

    FBGEMM CUDA can return an undefined Tensor instead of None for absent
    weights. Python converts it to None, but native TorchScript retains it and
    fails when another permutation consumes it.

    Mutates and returns ``permuted`` without copying it. ``source`` and
    ``permuted`` may be the same object on identity-permutation paths.
    """
    if source.weights_or_none() is None:
        permuted._weights = None
    return permuted


def symbolic_trace(
    # pyre-ignore[24]
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Dict[str, Any]] = None,
    leaf_modules: Optional[List[str]] = None,
) -> torch.fx.GraphModule:
    """Symbolic tracing API.

    Given an `nn.Module` or function instance `root`, this function will return a
    `GraphModule` constructed by recording operations seen while tracing through `root`.

    `concrete_args` allows you to partially specialize your function, whether it's to
    remove control flow or data structures.

    Inserts absent-weight guards after FX-wrapped TorchRec KJT permutations.
    This post-processing is idempotent when tracing an already guarded graph.

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and
            converted into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized
        leaf_modules (Optional[List[str]]): modules do not trace

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    # Resolve private TorchRec helpers only when tracing.
    from torchrec.modules.mc_modules import _mcc_lazy_init_inplace
    from torchrec.quant.embedding_modules import _permute_kjt

    _leaf_modules = list(UNTRACEABLE_MODULES)
    if leaf_modules:
        _leaf_modules.extend(leaf_modules)
    gm = _symbolic_trace(root, concrete_args, _leaf_modules)
    inserted = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in (
            _mcc_lazy_init_inplace,
            _permute_kjt,
        ):
            continue
        source = node.args[0] if node.args else node.kwargs["features"]
        # Split exporters can trace an already guarded graph again.
        if len(node.users) == 1:
            user = next(iter(node.users))
            if user.target == _restore_unweighted_kjt and user.args == (source, node):
                continue
        with gm.graph.inserting_after(node):
            restored = gm.graph.call_function(
                _restore_unweighted_kjt, args=(source, node)
            )
        # Keep the guard opaque when FX traces the generated module again.
        restored.meta["is_wrapped"] = True
        node.replace_all_uses_with(restored)
        # Replace-all also rewrites the guard's input; restore it to avoid a cycle.
        restored.args = (source, node)
        inserted = True
    if inserted:
        gm.graph.lint()
        gm.recompile()
    return gm


@torch.fx.wrap
def fx_get_label(
    labels: Dict[str, torch.Tensor],
    jagged_labels: Dict[str, JaggedTensor],
    label_name: str,
) -> torch.Tensor:
    """Fx trace wrapper for reading a label that may be stored as a list column."""
    if label_name in labels:
        return labels[label_name]
    return jagged_labels[label_name].values()


@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    """Fx trace wrapper for arange."""
    return torch.arange(len, device=device)


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    """Unwrap optional tensor for trace."""
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def fx_int_item(x: torch.Tensor) -> int:
    """Fx trace wrapper for `int(x.item())`."""
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        int_item = x.item()
        torch._check(int_item > 0)
        torch._check(int_item <= 2**31 - 1)
    else:
        int_item = int(x.item())
    # pyre-ignore[7]
    return int_item


@torch.fx.wrap
def fx_numel(x: torch.Tensor) -> int:
    """Fx trace wrapper for x.numel()."""
    total_len = x.numel()
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(total_len >= 0)
        torch._check(total_len <= 2**31 - 1)
    return total_len


@torch.fx.wrap
def fx_avg_counts(lengths: torch.Tensor) -> torch.Tensor:
    """Fx trace wrapper for the DDP-averaged ``(requests, candidates)`` counts.

    Used to rescale a local-batch mean loss into a global-batch mean so
    DDP's cross-rank gradient average stays unbiased on ragged batches. A
    loss that reduces over requests takes the first element and one that
    reduces over candidates the second; both travel in one all-reduce
    because a scalar collective costs a rank synchronization, not
    bandwidth.

    Args:
        lengths (torch.Tensor): ``(B,)`` candidates per request.

    Returns:
        torch.Tensor: ``(2,)`` cross-rank mean of ``B`` and ``sum(lengths)``.
    """
    counts = torch.stack(
        [
            torch.tensor(lengths.size(0), dtype=torch.float, device=lengths.device),
            lengths.sum().to(torch.float),
        ]
    )
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.AVG)
    return counts


@torch.fx.wrap
def fx_size0_max1(x: torch.Tensor) -> int:
    """Fx trace wrapper for max(x.size(0), 1).

    The inline ``max()`` compares a traced ``size()`` proxy in a bool context,
    which raises ``TraceError`` under torchrec's train-pipeline FX rewrite.
    """
    return max(x.size(0), 1)


@torch.fx.wrap
def fx_flip_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reverse every tensor in a dictionary along its first dimension."""
    flipped_tensor_dict = {}
    for key, value in tensor_dict.items():
        flipped_tensor_dict[key] = torch.flip(value, [0])
    return flipped_tensor_dict


@torch.fx.wrap
def fx_mark_keyed_tensor(name: str, x: KeyedTensor, is_dense: bool = False) -> None:
    """Mark a KeyedTensor in fx.graph.

    Used in EmbeddingGroup for split sparse part model when export.
    KeyedTensor.values() will be sparse part output and dense part input.
    If ``is_dense`` is true, split exporters keep the node in the dense graph
    instead of treating it as sparse-model output.
    """
    return


@torch.fx.wrap
def fx_mark_tensor(
    name: str, x: torch.Tensor, keys: Optional[List[str]] = None
) -> None:
    """Mark a Tensor in fx.graph.

    Used in EmbeddingGroup for split sparse part model when export.
    Tensor will be sparse part output and dense part input.
    """
    return


@torch.fx.wrap
def fx_mark_seq_tensor(
    seq_name: str,
    x: torch.Tensor,
    keys: Optional[List[str]] = None,
    max_seq_len: Optional[int] = None,
    is_jagged_seq: bool = False,
) -> None:
    """Mark a Sequence Tensor in fx.graph.

    Used in EmbeddingGroup for split sparse part model when export.
    Tensor will be sparse part output and dense part input.
    """
    return


@torch.fx.wrap
def fx_mark_seq_len(seq_name: str, x: torch.Tensor) -> None:
    """Mark a sequence length Tensor in fx.graph."""
    return


@torch.fx.wrap
def fx_mark_seq_ec_jt(seq_name: str, x: JaggedTensor) -> None:
    """Mark a query or sequence embedding collection output."""
    return
