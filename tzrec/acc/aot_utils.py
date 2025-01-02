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

import functools
import os
from typing import Dict

import torch
import torch._prims_common as prims_utils
import torch.nn.functional as F
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torch._decomp import decomposition_table, register_decomposition
from torch._prims_common.wrappers import out_wrapper
from torch.export import Dim

from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger

# skip default bound check which is not allow by aot
if "ENABLE_AOT" in os.environ:
    # pyre-ignore [8]
    IntNBitTableBatchedEmbeddingBagsCodegen.__init__ = functools.partialmethod(
        IntNBitTableBatchedEmbeddingBagsCodegen.__init__,
        bounds_check_mode=BoundsCheckMode.NONE,
    )


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


def export_model_aot(
    model: nn.Module, data: Dict[str, torch.Tensor], save_dir: str
) -> torch.export.ExportedProgram:
    """Export aot model.

    Args:
        model (nn.Module): the model
        data (Dict[str, torch.Tensor]): the test data
        save_dir (str): model save dir
    """
    gm = symbolic_trace(model)
    with open(os.path.join(save_dir, "gm.code"), "w") as f:
        f.write(gm.code)
    # with open(os.path.join(save_dir, "gm.graph"), "w") as f:
    #     f.write(gm.graph.print_tabular())

    gm = gm.cuda()

    print(gm)

    batch = Dim("batch")
    dynamic_shapes = {}
    for key in data:
        if key.endswith(".lengths"):
            if data[key].shape[0] == 1:
                logger.info("uniq user sparse fea %s length=1" % key)
                dynamic_shapes[key] = {}
            else:
                dynamic_shapes[key] = {0: batch}
        elif key == "batch_size":
            dynamic_shapes[key] = {}
        elif data[key].dtype == torch.float32 and "__" not in key:
            if data[key].shape[0] == 1:
                logger.info("uniq user dense_fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {}
            else:
                logger.info("dense_fea=%s shape=%s" % (key, data[key].shape))
                dynamic_shapes[key] = {0: batch}
        elif (
            data[key].dtype == torch.float32 and "__" in key and data[key].shape[0] == 1
        ):
            logger.info("uniq seq_dense_fea=%s shape=%s" % (key, data[key].shape))
            dynamic_shapes[key] = {}
        else:
            tmp_val_dim = Dim(key.replace(".", "__") + "__batch", min=0)
            # to handle torch.export 0/1 specialization problem
            if data[key].shape[0] < 2:
                data[key] = F.pad(
                    data[key],
                    [0, 2] + [0, 0] * (len(data[key].shape) - 1),
                    mode="constant",
                )
            dynamic_shapes[key] = {0: tmp_val_dim}

    exported_gm = torch.export.export(
        gm, args=(data,), dynamic_shapes=(dynamic_shapes,)
    )
    print(exported_gm)

    export_path = os.path.join(save_dir, "exported_gm.code")
    with open(export_path, "w") as fout:
        fout.write(str(exported_gm))

    exported_gm_path = os.path.join(save_dir, "debug_exported_gm.py")
    with open(exported_gm_path, "w") as fout:
        fout.write(str(exported_gm))

    exported_gm.module()(data)

    return exported_gm
