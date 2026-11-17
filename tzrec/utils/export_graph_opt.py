# Copyright (c) 2026, Alibaba Group;
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
import operator
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_linear_bn_eval, fuse_linear_bn_weights
from torchrec.sparse.jagged_tensor import _all_keys_used_once

from tzrec.datasets.data_parser import _tile_size
from tzrec.modules.utils import Transpose
from tzrec.utils.logging_util import logger


@torch.fx.wrap
def _tzrec_eval_bn_dice(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    alpha: torch.Tensor,
    channel_last: bool = False,
) -> torch.Tensor:
    """Standard PyTorch fallback and TorchScript IR anchor for fused BN + Dice."""
    x_f = x.float()
    s = scale.float()
    b = shift.float()
    a = alpha.float()
    if channel_last:
        norm_x = x_f * s + b
        p = torch.sigmoid(norm_x)
        return ((p + a * (1.0 - p)) * x_f).to(dtype=x.dtype)
    if x_f.dim() == 3:
        norm_x = x_f * s.unsqueeze(1) + b.unsqueeze(1)
        p = torch.sigmoid(norm_x)
        a_3d = a.unsqueeze(1)
        return ((p + a_3d * (1.0 - p)) * x_f).to(dtype=x.dtype)
    norm_x = x_f * s + b
    p = torch.sigmoid(norm_x)
    return ((p + a * (1.0 - p)) * x_f).to(dtype=x.dtype)


@torch.fx.wrap
def _tzrec_eval_bn(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    channel_last: bool = False,
) -> torch.Tensor:
    """Standard PyTorch fallback and TorchScript IR anchor for standalone eval BN1d."""
    x_f = x.float()
    s = scale.float()
    b = shift.float()
    if channel_last or x_f.dim() == 2:
        return (x_f * s + b).to(dtype=x.dtype)
    return (x_f * s.unsqueeze(1) + b.unsqueeze(1)).to(dtype=x.dtype)


@torch.fx.wrap
def _infer_tile_size(user_ref: torch.Tensor, item_ref: torch.Tensor) -> int:
    """Derive user-feature tile factor from TensorImpl dim-0 without .item()."""
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        if user_ref.size(0) == item_ref.size(0):
            return 1
        return item_ref.size(0)
    return 1 if user_ref.size(0) == item_ref.size(0) else item_ref.size(0)


@dataclass
class _OptContext:
    """Mutable optimization state isolated from the incoming GraphModule."""

    orig_gm: torch.fx.GraphModule
    graph: torch.fx.Graph
    new_submodules: Dict[str, nn.Module] = field(default_factory=dict)
    new_buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    used_names: Set[str] = field(default_factory=set)
    counter: int = 0

    @classmethod
    def from_graph_module(cls, gm: torch.fx.GraphModule) -> "_OptContext":
        """Create an isolated optimization context from ``gm``."""
        opt_graph = copy.deepcopy(gm.graph)
        used = set(dir(gm)) | set(gm.__dict__.keys())
        used.update(gm._modules.keys())
        used.update(gm._parameters.keys())
        used.update(gm._buffers.keys())
        for node in opt_graph.nodes:
            if node.op in ("call_module", "get_attr") and isinstance(node.target, str):
                used.add(node.target.split(".", 1)[0])
        return cls(orig_gm=gm, graph=opt_graph, used_names=used)

    def alloc_name(self, prefix: str) -> str:
        """Allocate a collision-free top-level attribute name."""
        while True:
            candidate = f"{prefix}_{self.counter}"
            self.counter += 1
            if candidate not in self.used_names:
                self.used_names.add(candidate)
                return candidate

    def get_module(self, target: str) -> Optional[nn.Module]:
        """Resolve a submodule from newly created modules or ``orig_gm``."""
        if target in self.new_submodules:
            return self.new_submodules[target]
        try:
            return self.orig_gm.get_submodule(target)
        except AttributeError:
            return None

    def get_attr(self, target: str) -> Any:
        """Resolve an attribute from newly created buffers or ``orig_gm``."""
        if target in self.new_buffers:
            return self.new_buffers[target]
        module_path, _, attr_name = target.rpartition(".")
        parent: Any = (
            self.orig_gm
            if module_path == ""
            else self.orig_gm.get_submodule(module_path)
        )
        return getattr(parent, attr_name)


def _clone_module_eval_nograd(mod: nn.Module) -> nn.Module:
    """Shallow-clone ``mod`` in eval/no-grad without copying tensor storage."""
    cloned = copy.copy(mod)
    cloned._parameters = {
        k: (None if v is None else nn.Parameter(v.detach(), requires_grad=False))
        for k, v in mod._parameters.items()
    }
    cloned._buffers = {
        k: (None if v is None else (v.detach() if v.requires_grad else v))
        for k, v in mod._buffers.items()
    }
    cloned._modules = {
        k: (None if v is None else _clone_module_eval_nograd(v))
        for k, v in mod._modules.items()
    }
    if hasattr(mod, "_non_persistent_buffers_set"):
        cloned._non_persistent_buffers_set = set(mod._non_persistent_buffers_set)
    cloned.training = False
    return cloned


def _add_module_by_dotted_path(
    root: nn.Module, dotted_path: str, module: nn.Module
) -> None:
    """Attach ``module`` at ``dotted_path`` under ``root``."""
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        child = parent._modules.get(part)
        if child is None:
            child = nn.Module()
            parent.add_module(part, child)
        parent = child
    parent.add_module(parts[-1], module)


def _ensure_container_by_dotted_path(root: nn.Module, dotted_path: str) -> nn.Module:
    """Ensure empty container modules exist down to ``dotted_path``."""
    if dotted_path == "":
        return root
    parent = root
    for part in dotted_path.split("."):
        child = parent._modules.get(part)
        if child is None:
            child = nn.Module()
            parent.add_module(part, child)
        parent = child
    return parent


def _is_constant_one(val: Any) -> bool:
    """Whether ``val`` is numeric scalar 1 or 1.0 (excluding bool)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and val == 1.0


def _match_transpose_12(
    node: Any, get_mod: Callable[[str], Optional[nn.Module]]
) -> Optional[torch.fx.Node]:
    """Return input FX node if ``node`` swaps dims 1 and 2, else None."""
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "call_method" and node.target == "transpose":
        x_arg = node.args[0] if len(node.args) >= 1 else node.kwargs.get("input")
        dim0 = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim0")
        dim1 = node.args[2] if len(node.args) >= 3 else node.kwargs.get("dim1")
        if (
            isinstance(x_arg, torch.fx.Node)
            and type(dim0) is int
            and type(dim1) is int
            and {dim0, dim1} == {1, 2}
        ):
            return x_arg
        return None
    if node.op == "call_function" and node.target is torch.transpose:
        x_arg = node.args[0] if len(node.args) >= 1 else node.kwargs.get("input")
        dim0 = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim0")
        dim1 = node.args[2] if len(node.args) >= 3 else node.kwargs.get("dim1")
        if (
            isinstance(x_arg, torch.fx.Node)
            and type(dim0) is int
            and type(dim1) is int
            and {dim0, dim1} == {1, 2}
        ):
            return x_arg
        return None
    if node.op == "call_module" and isinstance(node.target, str):
        mod = get_mod(node.target)
        if isinstance(mod, Transpose) and {mod.dim0, mod.dim1} == {1, 2}:
            x_arg = node.args[0] if len(node.args) >= 1 else node.kwargs.get("x")
            if isinstance(x_arg, torch.fx.Node):
                return x_arg
    return None


def _is_valid_eval_bn1d(mod: Any, expected_features: Optional[int] = None) -> bool:
    """Whether ``mod`` is an eval BatchNorm1d with valid 1D running stats."""
    if type(mod) is not nn.BatchNorm1d:
        return False
    if mod.training or not mod.track_running_stats:
        return False
    rm = mod.running_mean
    rv = mod.running_var
    if rm is None or rv is None or rm.is_meta or rv.is_meta:
        return False
    if rm.dim() != 1 or rv.dim() != 1:
        return False
    num_c = rm.size(0)
    if num_c <= 0 or rv.size(0) != num_c:
        return False
    if expected_features is not None and num_c != expected_features:
        return False
    if mod.weight is not None and (
        mod.weight.is_meta or mod.weight.dim() != 1 or mod.weight.size(0) != num_c
    ):
        return False
    if mod.bias is not None and (
        mod.bias.is_meta or mod.bias.dim() != 1 or mod.bias.size(0) != num_c
    ):
        return False
    return True


def _compute_eval_bn_scale_shift(
    bn_mod: nn.BatchNorm1d,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute contiguous 1D FP32 scale and shift buffers from eval BN1d."""
    with torch.no_grad():
        assert bn_mod.running_mean is not None and bn_mod.running_var is not None
        rm = bn_mod.running_mean.detach().float()
        rv = bn_mod.running_var.detach().float()
        invstd = torch.rsqrt(rv + bn_mod.eps)
        scale = (
            bn_mod.weight.detach().float() * invstd
            if bn_mod.weight is not None
            else invstd
        )
        shift = (
            bn_mod.bias.detach().float() - rm * scale
            if bn_mod.bias is not None
            else -rm * scale
        )
        device = rm.device
        scale = scale.to(device=device, dtype=torch.float32).contiguous()
        shift = shift.to(device=device, dtype=torch.float32).contiguous()
        scale.requires_grad_(False)
        shift.requires_grad_(False)
    return scale, shift


def _fuse_linear_bn_eval_safe(lin_mod: nn.Linear, bn_mod: nn.BatchNorm1d) -> nn.Linear:
    """Fuse eval Linear and BatchNorm1d (supporting both affine=True and False)."""
    if bn_mod.weight is not None and bn_mod.bias is not None:
        fused = fuse_linear_bn_eval(lin_mod, bn_mod)
    else:
        assert bn_mod.running_mean is not None and bn_mod.running_var is not None
        fused = copy.deepcopy(lin_mod)
        bn_w = (
            bn_mod.weight
            if bn_mod.weight is not None
            else torch.ones_like(bn_mod.running_var)
        )
        bn_b = (
            bn_mod.bias
            if bn_mod.bias is not None
            else torch.zeros_like(bn_mod.running_mean)
        )
        fused.weight, fused.bias = fuse_linear_bn_weights(
            fused.weight,
            fused.bias,
            bn_mod.running_mean,
            bn_mod.running_var,
            bn_mod.eps,
            bn_w,
            bn_b,
        )
    fused.eval()
    fused.requires_grad_(False)
    return fused


def _is_sigmoid_node(
    node: torch.fx.Node,
    bn_node: torch.fx.Node,
    get_mod: Callable[[str], Optional[nn.Module]],
) -> bool:
    """Whether ``node`` applies sigmoid directly to ``bn_node``."""
    if node.op == "call_method" and node.target == "sigmoid":
        return node.args == (bn_node,) and not node.kwargs
    if node.op == "call_function" and node.target in (torch.sigmoid, F.sigmoid):
        return (node.args == (bn_node,) and not node.kwargs) or (
            not node.args and node.kwargs == {"input": bn_node}
        )
    if node.op == "call_module" and isinstance(node.target, str):
        mod = get_mod(node.target)
        return type(mod) is nn.Sigmoid and node.args == (bn_node,) and not node.kwargs
    return False


def _is_one_minus_node(node: torch.fx.Node, sig_node: torch.fx.Node) -> bool:
    """Whether ``node`` computes ``1 - sig_node``."""
    if node.kwargs or len(node.args) != 2:
        return False
    if node.op == "call_function" and node.target in (operator.sub, torch.sub):
        return _is_constant_one(node.args[0]) and node.args[1] is sig_node
    if (node.op == "call_function" and node.target is torch.rsub) or (
        node.op == "call_method" and node.target == "rsub"
    ):
        return node.args[0] is sig_node and _is_constant_one(node.args[1])
    return False


def _binary_operands(node: Any, op_kind: str) -> Optional[Tuple[Any, Any]]:
    """Return ``(lhs, rhs)`` if ``node`` is a binary ``mul`` or ``add``."""
    if not isinstance(node, torch.fx.Node) or node.kwargs or len(node.args) != 2:
        return None
    if op_kind == "mul":
        is_match = (
            node.op == "call_function" and node.target in (operator.mul, torch.mul)
        ) or (node.op == "call_method" and node.target == "mul")
        return (node.args[0], node.args[1]) if is_match else None
    if op_kind == "add":
        is_match = (
            node.op == "call_function" and node.target in (operator.add, torch.add)
        ) or (node.op == "call_method" and node.target == "add")
        return (node.args[0], node.args[1]) if is_match else None
    return None


def _match_commutative_pair(
    node: Any, op_kind: str, known: torch.fx.Node
) -> Tuple[bool, Any]:
    """Match a commutative binary node containing ``known`` and return other arg."""
    operands = _binary_operands(node, op_kind)
    if operands is None:
        return False, None
    lhs, rhs = operands
    if lhs is known:
        return True, rhs
    if rhs is known:
        return True, lhs
    return False, None


def _extract_alpha_node(
    ctx: _OptContext, node: Any, expected_c: int
) -> Optional[Tuple[torch.Tensor, bool, List[torch.fx.Node]]]:
    """Extract Dice ``alpha`` tensor (1D `[C]`) and unsqueeze-node cleanup list."""
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "get_attr" and isinstance(node.target, str):
        tensor = ctx.get_attr(node.target)
        if (
            isinstance(tensor, torch.Tensor)
            and not tensor.is_meta
            and tensor.numel() == expected_c
            and (
                tensor.dim() == 1
                or (tensor.dim() == 2 and 1 in tensor.shape)
                or (tensor.dim() == 3 and tensor.shape[1] == expected_c)
            )
        ):
            alpha_1d = (
                tensor.detach()
                .to(device=tensor.device, dtype=torch.float32)
                .contiguous()
                .view(expected_c)
            )
            alpha_1d.requires_grad_(False)
            return alpha_1d, tensor.dim() > 1, []
        return None

    inner: Any = None
    dim_arg: Any = None
    if node.op == "call_method" and node.target == "unsqueeze":
        inner = node.args[0] if len(node.args) >= 1 else None
        dim_arg = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim")
    elif node.op == "call_function" and node.target is torch.unsqueeze:
        inner = node.args[0] if len(node.args) >= 1 else node.kwargs.get("input")
        dim_arg = node.args[1] if len(node.args) >= 2 else node.kwargs.get("dim")

    if (
        type(dim_arg) is int
        and dim_arg == 1
        and isinstance(inner, torch.fx.Node)
        and inner.op == "get_attr"
        and isinstance(inner.target, str)
    ):
        tensor = ctx.get_attr(inner.target)
        if (
            isinstance(tensor, torch.Tensor)
            and not tensor.is_meta
            and tensor.dim() == 1
            and tensor.numel() == expected_c
        ):
            alpha_1d = (
                tensor.detach()
                .to(device=tensor.device, dtype=torch.float32)
                .contiguous()
                .view(expected_c)
            )
            alpha_1d.requires_grad_(False)
            return alpha_1d, True, [node]
    return None


@dataclass
class _DiceMatch:
    """Matched BatchNorm1d + Dice subgraph."""

    dice_out: torch.fx.Node
    alpha_1d: torch.Tensor
    alpha_was_unsqueezed: bool
    x_user_nodes: Set[torch.fx.Node]
    nodes_to_erase: List[torch.fx.Node]
    optional_erase_nodes: List[torch.fx.Node]


def _match_three_factor_product(
    ctx: _OptContext,
    term_node: Any,
    x_bn: torch.fx.Node,
    one_minus_xp: torch.fx.Node,
    expected_c: int,
) -> Optional[
    Tuple[
        torch.Tensor,
        bool,
        List[torch.fx.Node],
        List[torch.fx.Node],
        torch.fx.Node,
    ]
]:
    """Match ``alpha * (1 - xp) * x_bn`` under any association/commutation."""
    outer_ops = _binary_operands(term_node, "mul")
    if outer_ops is None or len(term_node.users) != 1:
        return None
    lhs, rhs = outer_ops
    candidates: List[
        Tuple[
            List[Any],
            List[torch.fx.Node],
            Dict[torch.fx.Node, torch.fx.Node],
        ]
    ] = []
    for sub_node, other in ((lhs, rhs), (rhs, lhs)):
        sub_ops = _binary_operands(sub_node, "mul")
        if sub_ops is not None and len(sub_node.users) == 1:
            sub_l, sub_r = sub_ops
            owner_map: Dict[torch.fx.Node, torch.fx.Node] = {}
            if isinstance(sub_l, torch.fx.Node):
                owner_map[sub_l] = sub_node
            if isinstance(sub_r, torch.fx.Node):
                owner_map[sub_r] = sub_node
            if isinstance(other, torch.fx.Node):
                owner_map[other] = term_node
            candidates.append(([sub_l, sub_r, other], [term_node, sub_node], owner_map))

    for leaves, mul_nodes, owner_map in candidates:
        x_idx = [i for i, leaf in enumerate(leaves) if leaf is x_bn]
        omp_idx = [i for i, leaf in enumerate(leaves) if leaf is one_minus_xp]
        if len(x_idx) != 1 or len(omp_idx) != 1 or x_idx[0] == omp_idx[0]:
            continue
        alpha_idx = (set((0, 1, 2)) - {x_idx[0], omp_idx[0]}).pop()
        alpha_info = _extract_alpha_node(ctx, leaves[alpha_idx], expected_c)
        if alpha_info is None:
            continue
        alpha_1d, was_unsqueezed, opt_nodes = alpha_info
        return alpha_1d, was_unsqueezed, mul_nodes, opt_nodes, owner_map[x_bn]
    return None


def _match_dice_from_bn(
    ctx: _OptContext, bn_node: torch.fx.Node, expected_c: int
) -> Optional[_DiceMatch]:
    """Match a 2D or 3D Dice activation subgraph anchored at ``bn_node``."""
    if len(bn_node.args) != 1 or bn_node.kwargs:
        return None
    x_bn = bn_node.args[0]
    if not isinstance(x_bn, torch.fx.Node) or len(bn_node.users) != 1:
        return None
    sig_node = next(iter(bn_node.users))
    if not _is_sigmoid_node(sig_node, bn_node, ctx.get_module):
        return None
    if len(sig_node.users) != 2:
        return None

    sig_users = list(sig_node.users)
    if _is_one_minus_node(sig_users[0], sig_node):
        one_minus_xp, sig_other = sig_users[0], sig_users[1]
    elif _is_one_minus_node(sig_users[1], sig_node):
        one_minus_xp, sig_other = sig_users[1], sig_users[0]
    else:
        return None

    if len(one_minus_xp.users) != 1:
        return None

    # Form A: alpha * (1 - xp) * x + xp * x
    ok_term2, term2_other = _match_commutative_pair(sig_other, "mul", sig_node)
    if ok_term2 and term2_other is x_bn and len(sig_other.users) == 1:
        dice_out = next(iter(sig_other.users))
        ok_add, term1 = _match_commutative_pair(dice_out, "add", sig_other)
        if ok_add:
            prod = _match_three_factor_product(
                ctx, term1, x_bn, one_minus_xp, expected_c
            )
            if prod is not None:
                alpha_1d, was_unsqueezed, mul_nodes, opt_nodes, x_term1_user = prod
                return _DiceMatch(
                    dice_out=dice_out,
                    alpha_1d=alpha_1d,
                    alpha_was_unsqueezed=was_unsqueezed,
                    x_user_nodes={bn_node, sig_other, x_term1_user},
                    nodes_to_erase=[
                        dice_out,
                        *mul_nodes,
                        sig_other,
                        one_minus_xp,
                        sig_node,
                        bn_node,
                    ],
                    optional_erase_nodes=opt_nodes,
                )

    # Form B: (xp + alpha * (1 - xp)) * x
    ok_gate, term_a = _match_commutative_pair(sig_other, "add", sig_node)
    if ok_gate and len(sig_other.users) == 1 and isinstance(term_a, torch.fx.Node):
        if len(term_a.users) == 1:
            ok_alpha_mul, alpha_leaf = _match_commutative_pair(
                term_a, "mul", one_minus_xp
            )
            if ok_alpha_mul:
                alpha_info = _extract_alpha_node(ctx, alpha_leaf, expected_c)
                if alpha_info is not None:
                    dice_out = next(iter(sig_other.users))
                    ok_out, out_other = _match_commutative_pair(
                        dice_out, "mul", sig_other
                    )
                    if ok_out and out_other is x_bn:
                        alpha_1d, was_unsqueezed, opt_nodes = alpha_info
                        return _DiceMatch(
                            dice_out=dice_out,
                            alpha_1d=alpha_1d,
                            alpha_was_unsqueezed=was_unsqueezed,
                            x_user_nodes={bn_node, dice_out},
                            nodes_to_erase=[
                                dice_out,
                                sig_other,
                                term_a,
                                one_minus_xp,
                                sig_node,
                                bn_node,
                            ],
                            optional_erase_nodes=opt_nodes,
                        )
    return None


def _pass1_eliminate_eval_identities(ctx: _OptContext) -> None:
    """Pass 1: Remove eval Dropout/Identity modules and dead _all_keys_used_once."""
    for node in list(ctx.graph.nodes):
        if node.op == "call_module" and isinstance(node.target, str):
            mod = ctx.get_module(node.target)
            if (
                isinstance(mod, (nn.Dropout, nn.Dropout1d, nn.Identity))
                and not mod.training
            ):
                x_arg = node.args[0] if node.args else node.kwargs.get("input")
                if x_arg is not None:
                    node.replace_all_uses_with(x_arg)
                    ctx.graph.erase_node(node)
        elif node.op == "call_function" and node.target is F.dropout:
            training_arg = (
                node.args[2]
                if len(node.args) >= 3
                else node.kwargs.get("training", True)
            )
            if training_arg is False:
                x_arg = node.args[0] if node.args else node.kwargs.get("input")
                if x_arg is not None:
                    node.replace_all_uses_with(x_arg)
                    ctx.graph.erase_node(node)
        elif (
            node.op == "call_function"
            and (
                node.target is _all_keys_used_once
                or getattr(node.target, "__name__", "") == "_all_keys_used_once"
            )
            and len(node.users) == 0
        ):
            ctx.graph.erase_node(node)


def _pass2_fuse_linear_bn(ctx: _OptContext) -> None:
    """Pass 2: Algebraically fuse 2D and 3D eval Linear + BatchNorm1d."""
    erased: Set[torch.fx.Node] = set()
    for lin_node in list(ctx.graph.nodes):
        if lin_node in erased or lin_node.op != "call_module":
            continue
        if not isinstance(lin_node.target, str) or len(lin_node.users) != 1:
            continue
        lin_mod = ctx.get_module(lin_node.target)
        if type(lin_mod) is not nn.Linear or lin_mod.training or lin_mod.weight.is_meta:
            continue

        user = next(iter(lin_node.users))
        bn_node: Optional[torch.fx.Node] = None
        t1_node: Optional[torch.fx.Node] = None
        t2_node: Optional[torch.fx.Node] = None
        replace_target: Optional[torch.fx.Node] = None

        if (
            _match_transpose_12(user, ctx.get_module) is lin_node
            and len(user.users) == 1
        ):
            t1_node = user
            cand_bn = next(iter(t1_node.users))
            if cand_bn.op == "call_module" and len(cand_bn.users) == 1:
                cand_t2 = next(iter(cand_bn.users))
                if _match_transpose_12(cand_t2, ctx.get_module) is cand_bn:
                    bn_node = cand_bn
                    t2_node = cand_t2
                    replace_target = t2_node
        elif user.op == "call_module":
            bn_node = user
            replace_target = bn_node

        if (
            bn_node is None
            or replace_target is None
            or not isinstance(bn_node.target, str)
        ):
            continue
        if bn_node.kwargs or len(bn_node.args) != 1:
            continue
        bn_mod = ctx.get_module(bn_node.target)
        if not _is_valid_eval_bn1d(bn_mod, expected_features=lin_mod.out_features):
            continue
        bn_mod = cast(nn.BatchNorm1d, bn_mod)
        assert bn_mod.running_mean is not None
        if bn_mod.running_mean.device != lin_mod.weight.device:
            continue

        fused_lin = _fuse_linear_bn_eval_safe(lin_mod, bn_mod)
        fused_name = ctx.alloc_name("_opt_fused_linear")
        ctx.new_submodules[fused_name] = fused_lin
        lin_node.target = fused_name
        replace_target.replace_all_uses_with(lin_node)

        for dead in (t2_node, bn_node, t1_node):
            if dead is not None:
                ctx.graph.erase_node(dead)
                erased.add(dead)


def _emit_std_dice(
    graph: torch.fx.Graph,
    x_in: torch.fx.Node,
    scale_node: torch.fx.Node,
    shift_node: torch.fx.Node,
    alpha_node: torch.fx.Node,
    unsqueeze_1d: bool,
) -> torch.fx.Node:
    """Emit standard ATen FP32-computed Dice expression without custom anchors."""
    if unsqueeze_1d:
        scale_node = graph.call_method("unsqueeze", args=(scale_node, 1))
        shift_node = graph.call_method("unsqueeze", args=(shift_node, 1))
        alpha_node = graph.call_method("unsqueeze", args=(alpha_node, 1))
    x_f = graph.call_method("float", args=(x_in,))
    scaled = graph.call_function(operator.mul, args=(x_f, scale_node))
    norm_x = graph.call_function(operator.add, args=(scaled, shift_node))
    p = graph.call_function(torch.sigmoid, args=(norm_x,))
    one_minus_p = graph.call_function(operator.sub, args=(1.0, p))
    alpha_term = graph.call_function(operator.mul, args=(alpha_node, one_minus_p))
    gate = graph.call_function(operator.add, args=(p, alpha_term))
    out_f = graph.call_function(operator.mul, args=(gate, x_f))
    return graph.call_method("type_as", args=(out_f, x_in))


def _pass3_fuse_bn_dice(ctx: _OptContext, emit_jit_anchors: bool) -> None:
    """Pass 3: Rewrite eval BatchNorm1d + Dice into _tzrec_eval_bn_dice or ATen."""
    erased: Set[torch.fx.Node] = set()
    for bn_node in list(ctx.graph.nodes):
        if bn_node in erased or bn_node.op != "call_module":
            continue
        if not isinstance(bn_node.target, str):
            continue
        bn_mod = ctx.get_module(bn_node.target)
        if not _is_valid_eval_bn1d(bn_mod):
            continue
        bn_mod = cast(nn.BatchNorm1d, bn_mod)
        assert bn_mod.running_mean is not None
        num_c = bn_mod.running_mean.size(0)

        match = _match_dice_from_bn(ctx, bn_node, num_c)
        if match is None:
            continue

        x_bn = cast(torch.fx.Node, bn_node.args[0])
        t1_node: Optional[torch.fx.Node] = None
        t2_node: Optional[torch.fx.Node] = None
        pre_t1 = _match_transpose_12(x_bn, ctx.get_module)
        channel_last = False
        x_in = x_bn
        replace_target = match.dice_out

        if (
            pre_t1 is not None
            and set(x_bn.users.keys()) == match.x_user_nodes
            and len(match.dice_out.users) == 1
        ):
            cand_t2 = next(iter(match.dice_out.users))
            if _match_transpose_12(cand_t2, ctx.get_module) is match.dice_out:
                channel_last = True
                t1_node = x_bn
                t2_node = cand_t2
                x_in = pre_t1
                replace_target = cand_t2

        scale, shift = _compute_eval_bn_scale_shift(bn_mod)
        alpha_1d = match.alpha_1d.to(device=scale.device, dtype=torch.float32)

        scale_name = ctx.alloc_name("_opt_dice_scale")
        shift_name = ctx.alloc_name("_opt_dice_shift")
        alpha_name = ctx.alloc_name("_opt_dice_alpha")
        ctx.new_buffers[scale_name] = scale
        ctx.new_buffers[shift_name] = shift
        ctx.new_buffers[alpha_name] = alpha_1d

        with ctx.graph.inserting_before(replace_target):
            scale_node = ctx.graph.get_attr(scale_name)
            shift_node = ctx.graph.get_attr(shift_name)
            alpha_node = ctx.graph.get_attr(alpha_name)
            if emit_jit_anchors:
                out_node = ctx.graph.call_function(
                    _tzrec_eval_bn_dice,
                    args=(
                        x_in,
                        scale_node,
                        shift_node,
                        alpha_node,
                        channel_last,
                    ),
                )
            else:
                out_node = _emit_std_dice(
                    ctx.graph,
                    x_in,
                    scale_node,
                    shift_node,
                    alpha_node,
                    unsqueeze_1d=(not channel_last and match.alpha_was_unsqueezed),
                )

        replace_target.replace_all_uses_with(out_node)
        dead_order: List[torch.fx.Node] = []
        if t2_node is not None:
            dead_order.append(t2_node)
        dead_order.extend(match.nodes_to_erase)
        if t1_node is not None:
            dead_order.append(t1_node)
        for dead in dead_order:
            if dead not in erased and len(dead.users) == 0:
                ctx.graph.erase_node(dead)
                erased.add(dead)
        for opt_dead in match.optional_erase_nodes:
            if opt_dead not in erased and len(opt_dead.users) == 0:
                ctx.graph.erase_node(opt_dead)
                erased.add(opt_dead)


def _pass4_rewrite_standalone_bn(ctx: _OptContext, emit_jit_anchors: bool) -> None:
    """Pass 4: Rewrite remaining eval BatchNorm1d into _tzrec_eval_bn or ATen."""
    erased: Set[torch.fx.Node] = set()
    for bn_node in list(ctx.graph.nodes):
        if bn_node in erased or bn_node.op != "call_module":
            continue
        if not isinstance(bn_node.target, str):
            continue
        if bn_node.kwargs or len(bn_node.args) != 1:
            continue
        x_bn = bn_node.args[0]
        if not isinstance(x_bn, torch.fx.Node):
            continue
        bn_mod = ctx.get_module(bn_node.target)
        if not _is_valid_eval_bn1d(bn_mod):
            continue
        bn_mod = cast(nn.BatchNorm1d, bn_mod)

        t1_node: Optional[torch.fx.Node] = None
        t2_node: Optional[torch.fx.Node] = None
        channel_last = False
        x_in = x_bn
        replace_target = bn_node

        pre_t1 = _match_transpose_12(x_bn, ctx.get_module)
        if pre_t1 is not None and len(x_bn.users) == 1 and len(bn_node.users) == 1:
            cand_t2 = next(iter(bn_node.users))
            if _match_transpose_12(cand_t2, ctx.get_module) is bn_node:
                channel_last = True
                t1_node = x_bn
                t2_node = cand_t2
                x_in = pre_t1
                replace_target = cand_t2

        scale, shift = _compute_eval_bn_scale_shift(bn_mod)
        scale_name = ctx.alloc_name("_opt_bn_scale")
        shift_name = ctx.alloc_name("_opt_bn_shift")
        ctx.new_buffers[scale_name] = scale
        ctx.new_buffers[shift_name] = shift

        with ctx.graph.inserting_before(replace_target):
            scale_node = ctx.graph.get_attr(scale_name)
            shift_node = ctx.graph.get_attr(shift_name)
            if emit_jit_anchors:
                out_node = ctx.graph.call_function(
                    _tzrec_eval_bn,
                    args=(x_in, scale_node, shift_node, channel_last),
                )
            else:
                x_work = (
                    x_in
                    if channel_last
                    else ctx.graph.call_method("transpose", args=(x_in, 1, -1))
                )
                x_f = ctx.graph.call_method("float", args=(x_work,))
                scaled = ctx.graph.call_function(operator.mul, args=(x_f, scale_node))
                norm_x = ctx.graph.call_function(
                    operator.add, args=(scaled, shift_node)
                )
                if not channel_last:
                    norm_x = ctx.graph.call_method("transpose", args=(norm_x, 1, -1))
                out_node = ctx.graph.call_method("type_as", args=(norm_x, x_in))

        replace_target.replace_all_uses_with(out_node)
        for dead in (t2_node, bn_node, t1_node):
            if dead is not None and dead not in erased and len(dead.users) == 0:
                ctx.graph.erase_node(dead)
                erased.add(dead)


def _pass5_replace_tile_size(
    ctx: _OptContext,
    user_input_keys: Optional[Set[str]],
    item_input_keys: Optional[Set[str]],
) -> None:
    """Pass 5: Replace _tile_size(batch_size) with _infer_tile_size(u, i)."""
    if not user_input_keys or not item_input_keys:
        return

    placeholders = [n for n in ctx.graph.nodes if n.op == "placeholder"]
    if not placeholders:
        return
    input_placeholder = placeholders[0]

    user_ref: Optional[torch.fx.Node] = None
    item_ref: Optional[torch.fx.Node] = None
    tile_nodes: List[torch.fx.Node] = []
    node_order: Dict[torch.fx.Node, int] = {}

    for idx, node in enumerate(ctx.graph.nodes):
        node_order[node] = idx
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) == 2
            and not node.kwargs
            and node.args[0] is input_placeholder
            and isinstance(node.args[1], str)
        ):
            key = node.args[1]
            if user_ref is None and key in user_input_keys:
                user_ref = node
            if item_ref is None and key in item_input_keys:
                item_ref = node
        elif (
            node.op == "call_function"
            and (
                node.target is _tile_size
                or getattr(node.target, "__name__", "") == "_tile_size"
            )
            and len(node.args) == 1
            and not node.kwargs
        ):
            bs_node = node.args[0]
            if (
                isinstance(bs_node, torch.fx.Node)
                and bs_node.op == "call_function"
                and bs_node.target is operator.getitem
                and bs_node.args == (input_placeholder, "batch_size")
            ):
                tile_nodes.append(node)

    if not tile_nodes or user_ref is None or item_ref is None:
        return

    first_tile = tile_nodes[0]
    first_tile_idx = node_order[first_tile]
    # DataParser reads batch_size before feature keys; hoist top-level refs if needed.
    if node_order[user_ref] > first_tile_idx:
        first_tile.prepend(user_ref)
    if node_order[item_ref] > first_tile_idx:
        first_tile.prepend(item_ref)

    for tile_node in tile_nodes:
        with ctx.graph.inserting_before(tile_node):
            infer_node = ctx.graph.call_function(
                _infer_tile_size, args=(user_ref, item_ref)
            )
        tile_node.replace_all_uses_with(infer_node)
        ctx.graph.erase_node(tile_node)


def _build_isolated_root(ctx: _OptContext) -> nn.Module:
    """Build an isolated eval/no-grad root module owning only live FX references."""
    root = nn.Module()
    call_module_paths: List[str] = []
    seen_modules: Set[str] = set()
    get_attr_paths: List[str] = []
    seen_attrs: Set[str] = set()

    for node in ctx.graph.nodes:
        if node.op == "call_module" and isinstance(node.target, str):
            if node.target not in seen_modules:
                seen_modules.add(node.target)
                call_module_paths.append(node.target)
        elif node.op == "get_attr" and isinstance(node.target, str):
            if node.target not in seen_attrs:
                seen_attrs.add(node.target)
                get_attr_paths.append(node.target)

    for path in sorted(call_module_paths, key=lambda p: p.count(".")):
        src_mod = ctx.get_module(path)
        if src_mod is None:
            raise AttributeError(f"Missing FX submodule target [{path}]")
        mod_obj = (
            src_mod
            if path in ctx.new_submodules
            else _clone_module_eval_nograd(src_mod)
        )
        _add_module_by_dotted_path(root, path, mod_obj)

    for path in get_attr_paths:
        module_path, _, attr_name = path.rpartition(".")
        if module_path in seen_modules:
            continue
        parent = _ensure_container_by_dotted_path(root, module_path)
        if hasattr(parent, attr_name):
            continue
        obj = ctx.get_attr(path)
        if isinstance(obj, nn.Parameter):
            parent.register_parameter(
                attr_name, nn.Parameter(obj.detach(), requires_grad=False)
            )
        elif isinstance(obj, torch.Tensor):
            buf = obj.detach() if obj.requires_grad else obj
            parent.register_buffer(attr_name, buf)
        else:
            setattr(parent, attr_name, obj)

    return root


def _pass6_finalize_graph_module(ctx: _OptContext) -> torch.fx.GraphModule:
    """Pass 6: Eliminate dead code, prune unused state, and freeze grad/eval flags."""
    tmp_root = _build_isolated_root(ctx)
    tmp_gm = torch.fx.GraphModule(tmp_root, ctx.graph)
    tmp_gm.graph.eliminate_dead_code()

    final_root = _build_isolated_root(ctx)
    opt_gm = torch.fx.GraphModule(final_root, ctx.graph)
    for param in opt_gm.parameters():
        param.requires_grad_(False)
    for buf in opt_gm.buffers():
        buf.requires_grad_(False)
    opt_gm.eval()
    opt_gm.graph.lint()
    opt_gm.recompile()
    return opt_gm


def optimize_fx_for_export(
    gm: torch.fx.GraphModule,
    emit_jit_anchors: bool = True,
    user_input_keys: Optional[Set[str]] = None,
    item_input_keys: Optional[Set[str]] = None,
) -> torch.fx.GraphModule:
    """Optimize a traced FX GraphModule for serving export without mutating ``gm``.

    Args:
        gm: Traced ``torch.fx.GraphModule`` to optimize. Left unmutated.
        emit_jit_anchors: Whether to emit ``_tzrec_eval_bn_dice`` /
            ``_tzrec_eval_bn`` anchor calls for TorchScript export. Set False
            for TRT / AOTI paths to emit pure standard PyTorch/ATen ops.
        user_input_keys: Confirmed user-side top-level input dict keys (dim-0 is
            1 online, B offline) for zero-sync ``_infer_tile_size`` rewriting.
        item_input_keys: Confirmed item-side top-level input dict keys (dim-0 is
            B online and offline) for zero-sync ``_infer_tile_size`` rewriting.

    Returns:
        An isolated, eval-mode, non-trainable ``torch.fx.GraphModule`` (or the
        original ``gm`` if ``TZREC_DISABLE_EXPORT_GRAPH_OPT=1``).
    """
    if os.environ.get("TZREC_DISABLE_EXPORT_GRAPH_OPT", "0") == "1":
        logger.info("TZREC_DISABLE_EXPORT_GRAPH_OPT=1, skipping FX export graph opt.")
        return gm
    if not isinstance(gm, torch.fx.GraphModule):
        return gm

    ctx = _OptContext.from_graph_module(gm)
    _pass1_eliminate_eval_identities(ctx)
    _pass2_fuse_linear_bn(ctx)
    _pass3_fuse_bn_dice(ctx, emit_jit_anchors=emit_jit_anchors)
    _pass4_rewrite_standalone_bn(ctx, emit_jit_anchors=emit_jit_anchors)
    _pass5_replace_tile_size(ctx, user_input_keys, item_input_keys)
    return _pass6_finalize_graph_module(ctx)
