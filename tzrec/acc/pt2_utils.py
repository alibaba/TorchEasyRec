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

    # get dense keys list
    dense_keys_list = []
    for _, keys in model._data_parser.dense_keys.items():
        dense_keys_list.extend(keys)

    batch = Dim("batch", min=1, max=499999999)
    batch_tile = Dim("batch_tile", min=1, max=499999999)
    dynamic_shapes = {}
    for key in data:
        if key == "hard_neg_indices":
            dynamic_shapes[key] = {}
            continue
        # .lengths
        if key.endswith(".lengths"):
            # user feats
            if key.split(".")[0] in model._data_parser.user_feats:
                logger.info(
                    "tile user length fea %s length=%s" % (key, data[key].shape)
                )
                dynamic_shapes[key] = {0: batch_tile}
            else:
                logger.info("batch length fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {0: batch}
        elif key == "batch_size":
            dynamic_shapes[key] = {}

        # dense values
        elif key.split(".")[0] in dense_keys_list:
            # user feats
            if key.split(".")[0] in model._data_parser.user_feats:
                logger.info("tile user dense_fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {0: batch_tile}
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
                if key.split(".")[0] + ".key_lengths" in data:
                    data[key.split(".")[0] + ".key_lengths"][0] = data[key].shape[0]
                else:
                    data[key.split(".")[0] + ".lengths"][0] = data[key].shape[0]
            logger.info("sparse or seq dense fea=%s shape=%s" % (key, data[key].shape))
            tmp_val_dim = Dim(key.replace(".", "__") + "__batch", min=0, max=999999993)
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
