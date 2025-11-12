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
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from tzrec.acc.utils import get_max_export_batch_size, is_debug_trt
from tzrec.models.model import ScriptWrapper
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger
from torchrec.sparse.jagged_tensor import KeyedTensor
from torch.export import Dim

import inspect
# cpu image has no torch_tensorrt
has_tensorrt = False
try:
    import torch_tensorrt

    has_tensorrt = True
except Exception:
    pass


def trt_convert(
    exp_program: torch.export.ExportedProgram,
    inputs: Optional[Sequence[Sequence[Any]]],
) -> torch.fx.GraphModule:
    """Convert model use trt.

    Args:
        exp_program (torch.export.ExportedProgram): Source exported program
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): inputs

    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """
    logger.info("trt convert start...")
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    enabled_precisions = {torch.float32}

    # Workspace size for TensorRT
    workspace_size = 2 << 30

    # Maximum number of TRT Engines
    # (Lower value allows more graph segmentation)
    min_block_size = 2

    #  use script model , unsupported the inputs : dict
    if is_debug_trt():
        with torch_tensorrt.logging.graphs():
            optimized_model = torch_tensorrt.dynamo.compile(
                exp_program,
                inputs,
                # pyre-ignore [6]
                enabled_precisions=enabled_precisions,
                workspace_size=workspace_size,
                min_block_size=min_block_size,
                hardware_compatible=True,
                assume_dynamic_shape_support=True,
                # truncate_long_and_double=True,
                allow_shape_tensors=True,
            )

    else:
        optimized_model = torch_tensorrt.dynamo.compile(
            exp_program,
            inputs,
            # pyre-ignore [6]
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_size,
            min_block_size=min_block_size,
            hardware_compatible=True,
            assume_dynamic_shape_support=True,
            # truncate_long_and_double=True,
            allow_shape_tensors=True,
        )

    logger.info("trt convert end")
    return optimized_model


class ScriptWrapperList(ScriptWrapper):
    """Model inference wrapper for jit.script.

    ScriptWrapperList for trace the ScriptWrapperTRT(emb_trace_gpu, dense_layer_trt)
    and return a list of Tensor instead of a dict of Tensor
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)

    # pyre-ignore [15]
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        # pyre-ignore [9]
        device: torch.device = "cpu",
    ) -> List[torch.Tensor]:
        """Predict the model.

        Args:
            data (dict): a dict of input data for Batch.
            device (torch.device): inference device.

        Return:
            predictions (dict): a dict of predicted result.
        """
        batch = self.get_batch(data, device)
        return self.model.predict(batch)


class ScriptWrapperTRT(nn.Module):
    """Model inference wrapper for jit.script."""

    def __init__(self, 
                 embedding_group: nn.Module, 
                 dense: nn.Module,
                 output_keys
        ) -> None:
        super().__init__()
        self.embedding_group = embedding_group
        self.dense = dense
        self.output_keys = output_keys

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        # pyre-ignore [9]
        device: torch.device = "cuda:0",
    ) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            data (dict): a dict of input data for Batch.
            device (torch.device): inference device.

        Return:
            predictions (dict): a dict of predicted result.
        """
        emb_ebc, _ = self.embedding_group(data, device)
        o = self.dense(emb_ebc)
        outputs = {k: o[i] for i, k in enumerate(self.output_keys)}
        return outputs


def get_trt_max_seq_len() -> int:
    """Get trt max seq len.

    Returns:
        int: max_seq_len
    """
    return int(os.environ.get("TRT_MAX_SEQ_LEN", 100))


def export_model_trt(
    sparse_model: nn.Module,
    dense_model: nn.Module,
    data: Dict[str, torch.Tensor],
    output_keys: tuple,
    save_dir: str,
) -> None:
    """Export trt model.

    Args:
        model (nn.Module): the model
        data (Dict[str, torch.Tensor]): the test data
        save_dir (str): model save dir
    """
    emb_ebc, _ = sparse_model(data, "cuda:0")
    sparse_model_traced = symbolic_trace(sparse_model)
    
    with open(os.path.join(save_dir, "gm_sparse.code"), "w") as f:
        f.write(sparse_model_traced.code)

    sparse_model_scripted = torch.jit.script(sparse_model_traced)

    # dynamic shapes
    max_batch_size = get_max_export_batch_size()
    max_seq_len = get_trt_max_seq_len()
    batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    dynamic_shapes_list = []
    values_list_cuda = []
    key_list = []
    for i, k in enumerate(emb_ebc.keys()):
        v = emb_ebc[k].detach().to("cuda:0")
        dict_dy = {0: batch}
        if v.dim() == 3:
            # workaround -> 0/1 specialization
            if v.size(1) < 2:
                v = torch.zeros(v.size(0), 2, v.size(2), device="cuda:0", dtype=v.dtype)
            dict_dy[1] = torch.export.Dim("seq_len" + str(i), min=1, max=max_seq_len)

        if v.size(0) < 2:
            v = torch.zeros((2,) + v.size()[1:], device="cuda:0", dtype=v.dtype)
        values_list_cuda.append(v)
        dynamic_shapes_list.append(dict_dy)
        key_list.append(k)
    # convert dense
    # logger.info("dense res: %s", dense_model(emb_ebc))

    dense_layer = symbolic_trace(dense_model)
    dense_signature = inspect.signature(dense_model.forward)
    dense_arg_name = list(dense_signature.parameters.keys())[0]
    dynamic_shapes = {}
    for i, k in enumerate(key_list):
        dynamic_shapes[dense_arg_name] = {k: dynamic_shapes_list[i]}

    exp_program = torch.export.export(
        dense_layer,
        (emb_ebc, ),
        dynamic_shapes=dynamic_shapes
    )
    dense_layer_trt = trt_convert(exp_program, (emb_ebc,))
    # logger.info("dense trt res: %s", dense_layer_trt(emb_ebc))


    dense_layer_trt_traced = torch.jit.trace(
        dense_layer_trt, example_inputs=(emb_ebc,), strict=False
    )
    with open(os.path.join(save_dir, "gm_dense.code"), "w") as f:
        f.write(dense_layer_trt_traced.code)

    dense_layer_trt_scripted = torch.jit.script(dense_layer_trt_traced)
    # save combined_model
    combined_model = ScriptWrapperTRT(
        embedding_group=sparse_model_scripted, 
        dense=dense_layer_trt_scripted,
        output_keys=output_keys
    )
    result = combined_model(data, "cuda:0")
    logger.info("combined model result: %s", result)
    # combined_model = symbolic_trace(combined_model)
    # combined_model = torch.jit.trace(
    #     combined_model, example_inputs=data, strict=False
    # )
    scripted_model = torch.jit.script(combined_model)
    # pyre-ignore [16]
    scripted_model.save(os.path.join(save_dir, "scripted_model.pt"))

    with open(os.path.join(save_dir, "gm.code"), "w") as f:
        f.write(scripted_model.code)

    if is_debug_trt():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function("model_inference_dense"):
                dict_res = dense_model(emb_ebc)
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function("model_inference_dense_trt"):
                dict_res = dense_layer_trt(emb_ebc)
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        model_gpu_combined = torch.jit.load(
            os.path.join(save_dir, "scripted_model.pt"), map_location="cuda:0"
        )
        res = model_gpu_combined(data)
        logger.info("final res: %s", res)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function("model_inference_combined_trt"):
                dict_res = model_gpu_combined(data)
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    logger.info("trt convert success")
