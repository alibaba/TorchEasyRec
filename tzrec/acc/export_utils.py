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

import os
from typing import Dict, Tuple

import torch
import torch._prims_common as prims_utils
import torch.nn.functional as F
from torch import nn
from torch._decomp import decomposition_table, register_decomposition
from torch._prims_common.wrappers import out_wrapper
from torch.export import Dim

from tzrec.acc.utils import is_trt
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger

# add new aten._softmax decomposition which is supported by dynamo
aten = torch._ops.ops.aten
if aten._softmax.default in decomposition_table:
    del decomposition_table[aten._softmax.default]
    del decomposition_table[aten._softmax.out]


# pyre-ignore [56]
@register_decomposition(aten._softmax)
@out_wrapper()
def _softmax(x: torch.Tensor, dim: int, half_to_float: bool) -> torch.Tensor:
    # eager softmax returns a contiguous tensor. Ensure that decomp also returns
    # a contiguous tensor.
    x = x.contiguous()
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = prims_utils.elementwise_dtypes(
        x, type_promotion_kind=prims_utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    x_max = torch.max(x, dim, keepdim=True).values
    unnormalized = torch.exp(x - x_max)
    result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
    if not half_to_float:
        result = result.to(result_dtype)
    return result


# now dynamo generate sym_int5 = sym_sum(sym_int1, sym_int2, sym_int3) op
# instead of sym_int4 = sym_int1 + sym_int2; sym_int5 = sym_int4 + sym_int3.
# patch _check_graph_module of Verifier temporarily to add sym_sum into allowed ops
# def _check_graph_module(self, gm: torch.fx.GraphModule) -> None:
#     def _allowed_getattr_types() -> Tuple[Type[Any], ...]:
#         ret = self.allowed_getattr_types()
#         assert not any(t is object for t in ret)
#         return ret

#     def _check_valid_op(op) -> None:
#         def _allowed_builtin_ops() -> List:
#             ret = self.allowed_builtin_ops()
#             assert all(inspect.isbuiltin(op) for op in ret)
#             return ret

#         def _allowed_op_types() -> Tuple[Type[Any], ...]:
#             ret = self.allowed_op_types()
#             assert not any(t is object for t in ret)
#             return ret

#         # TODO Remove this allowlist.
#         _allowed_torch_functions = (
#             torch.autograd.grad_mode.set_grad_enabled,
#             torch.sym_sum,
#             torch.sym_int,
#             torch.sym_float,
#             torch.sym_ite,
#             torch.sym_max,
#             torch.sym_min,
#             torch.sym_not,
#             torch.sym_sqrt,
#             # TODO (tmanlaibaatar)
#             # Predispatch export is able to contain autograd ops.
#             # These will be modeled as HOO later
#             torch._C._set_grad_enabled,
#             torch.amp.autocast_mode._enter_autocast,
#             torch.amp.autocast_mode._exit_autocast,
#             torch.fx.experimental.symbolic_shapes.cast_symbool_to_symint_guardless,
#         )

#         if not isinstance(op, _allowed_op_types()):
#             if op not in _allowed_builtin_ops() and op not in _allowed_torch_functions:          # NOQA
#                 raise SpecViolationError(
#                     f"Operator '{op}' is not an allowed operator type: {_allowed_op_types()}\n"  # NOQA
#                     f"Valid builtin ops: {_allowed_builtin_ops()}"
#                     f"Valid torch functions: {_allowed_torch_functions}"
#                 )

#         if isinstance(op, OpOverload):
#             # All ops functional
#             # TODO (tmanlaibaatar) more proper way is needed here
#             if self.dialect != "TRAINING" and not is_functional(op):
#                 raise SpecViolationError(f"operator '{op}' is not functional")
#         self.check_valid_op(op)

#     for mod in gm.modules():
#         if not isinstance(mod, torch.fx.GraphModule):
#             continue

#         mod.graph.lint()
#         for node in mod.graph.nodes:
#             # TODO(T140410192): should have fake tensor for all dialects
#             if node.op in {"call_module", "call_method"}:
#                 raise SpecViolationError(
#                     f"call_module is not valid: got a class '{node.target}' ",
#                 )

#             elif node.op == "call_function":
#                 _check_val(node)

#                 _check_valid_op(node.target)

#             elif node.op == "get_attr":
#                 if not isinstance(node.target, str):
#                     raise SpecViolationError(
#                         f"Expected get_attr target to be string, but got {type(node.target)}"  # NOQA
#                     )

#                 attr = getattr_recursive(mod, node.target)
#                 if isinstance(attr, torch.nn.Module):

#                     def _is_type(name, ty):
#                         return isinstance(getattr(attr, name, None), ty)  # NOQA

#                     if type(attr).__name__ == "LoweredBackendModule":
#                         if (
#                             _is_type("backend_id", str)
#                             and _is_type("processed_bytes", bytes)
#                             and _is_type("compile_specs", list)
#                             and hasattr(attr, "original_module")
#                         ):
#                             continue
#                         else:
#                             backend_id = getattr(attr, "backend_id", None)
#                             processed_bytes = getattr(attr, "processed_bytes", None)
#                             compile_specs = getattr(attr, "compile_specs", None)
#                             raise SpecViolationError(
#                                 f"Invalid get_attr type {type(attr)}. \n"
#                                 f"LoweredBackendModule fields: "
#                                 f"backend_id(str) : {type(backend_id)}, "
#                                 f"processed_bytes(bytes) : {type(processed_bytes)}, "
#                                 f"compile_specs(list) : {type(compile_specs)}"
#                             )

#                 if not isinstance(attr, _allowed_getattr_types()):
#                     raise SpecViolationError(
#                         f"Invalid get_attr type {type(attr)}. \n"
#                         f"Valid get_attr types: {_allowed_getattr_types()}"
#                     )

#             elif node.op == "placeholder":
#                 _check_val(node)
#             # TODO(zhxchen17)
#             # elif node.op == "output":
#             #     _check_flattened_outputs()

#     self.check_additional(gm)


# Verifier._check_graph_module = _check_graph_module


def export_pm(
    model: nn.Module, data: Dict[str, torch.Tensor], save_dir: str
) -> Tuple[torch.export.ExportedProgram, Dict[str, torch.Tensor]]:
    """Export a PyTorch model and its parameters.

    Args:
        model (nn.Module): The PyTorch model to export.
        data (Dict[str, torch.Tensor]): containing the model's input tensors.
        save_dir (str): The directory where the model should be saved.

    Returns:
        Tuple[torch.export.ExportedProgram, Dict[str, torch.Tensor]]:
        The exported program and its input data.
    """
    gm = symbolic_trace(model)
    with open(os.path.join(save_dir, "gm.code"), "w") as f:
        f.write(gm.code)

    gm = gm.cuda()

    batch = Dim("batch")
    dynamic_shapes = {}
    for key in data:
        if key == "hard_neg_indices":
            dynamic_shapes[key] = {}
            continue
        # .lengths
        if key.endswith(".lengths"):
            # user feats
            if key.split(".")[0] in model._data_parser.user_feats:
                assert data[key].shape[0] == 1
                logger.info(
                    "uniq user length fea %s length=%s" % (key, data[key].shape)
                )
                dynamic_shapes[key] = {}
            else:
                logger.info("batch length fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {0: batch}
        elif key == "batch_size":
            dynamic_shapes[key] = {}

        # dense values
        elif key.split(".")[0] in model._data_parser.dense_keys_list:
            # user feats
            if key.split(".")[0] in model._data_parser.user_feats:
                assert data[key].shape[0] == 1
                logger.info("uniq user dense_fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {}
            else:
                logger.info("batch dense_fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {0: batch}

        # seq_dense or sparse
        else:
            if data[key].shape[0] < 2:
                data[key] = F.pad(
                    data[key],
                    [0, 0] * (len(data[key].shape) - 1) + [0, 2],
                    mode="constant",
                )
                data[key.split(".")[0] + ".lengths"][0] = data[key].shape[0]
            logger.info("sparse or seq dense fea=%s shape=%s" % (key, data[key].shape))
            tmp_val_dim = Dim(key.replace(".", "__") + "__batch", min=0)
            dynamic_shapes[key] = {0: tmp_val_dim}

        # trt need contiguous format
        if is_trt():
            data[key] = data[key].contiguous()

    logger.info("dynamic shapes=%s" % dynamic_shapes)
    exported_pg = torch.export.export(
        gm, args=(data,), dynamic_shapes=(dynamic_shapes,)
    )

    export_path = os.path.join(save_dir, "exported_pg.py")
    with open(export_path, "w") as fout:
        fout.write(str(exported_pg))

    exported_pg.module()(data)

    return (exported_pg, data)
