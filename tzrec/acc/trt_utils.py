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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

# cpu image has no torch_tensorrt
try:
    import torch_tensorrt
except Exception:
    pass
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torchrec.fx import symbolic_trace

from tzrec.acc.utils import is_debug_trt
from tzrec.models.model import ScriptWrapper
from tzrec.utils.logging_util import logger


def trt_convert(
    module: nn.Module,
    # pyre-ignore [2]
    inputs: Optional[Sequence[Sequence[Any]]],
    # pyre-ignore [2]
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],
) -> torch.fx.GraphModule:
    """Convert model use trt.

    Args:
        module (nn.Module): Source module
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): inputs
        dynamic_shapes: dynamic shapes

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

    exp_program = torch.export.export(module, (inputs,), dynamic_shapes=dynamic_shapes)
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

    def __init__(self, embedding_group: nn.Module, dense: nn.Module) -> None:
        super().__init__()
        self.embedding_group = embedding_group
        self.dense = dense

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
        grouped_features = self.embedding_group(data, device)
        y = self.dense(grouped_features)
        return y


def get_trt_max_batch_size() -> int:
    """Get trt max batch size.

    Returns:
        int: max_batch_size
    """
    return int(os.environ.get("TRT_MAX_BATCH_SIZE", 2048))


def get_trt_max_seq_len() -> int:
    """Get trt max seq len.

    Returns:
        int: max_seq_len
    """
    return int(os.environ.get("TRT_MAX_SEQ_LEN", 100))


def export_model_trt(
    model: nn.Module, data: Dict[str, torch.Tensor], save_dir: str
) -> None:
    """Export trt model.

    Args:
        model (nn.Module): the model
        data (Dict[str, torch.Tensor]): the test data
        save_dir (str): model save dir
    """
    # ScriptWrapperList for trace the ScriptWrapperTRT(emb_trace_gpu, dense_layer_trt)
    emb_trace_gpu = ScriptWrapperList(model.model.embedding_group)
    emb_res = emb_trace_gpu(data, "cuda:0")
    emb_trace_gpu = symbolic_trace(emb_trace_gpu)
    emb_trace_gpu = torch.jit.script(emb_trace_gpu)

    # dynamic shapes
    max_batch_size = get_trt_max_batch_size()
    max_seq_len = get_trt_max_seq_len()
    batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    dynamic_shapes_list = []
    values_list_cuda = []
    for i, value in enumerate(emb_res):
        v = value.detach().to("cuda:0")
        values_list_cuda.append(v)
        dict_dy = {0: batch}
        if v.dim() == 3:
            dict_dy[1] = torch.export.Dim("seq_len" + str(i), min=1, max=max_seq_len)
        dynamic_shapes_list.append(dict_dy)

    # convert dense
    dense = model.model.dense
    logger.info("dense res: %s", dense(values_list_cuda))
    dense_layer = symbolic_trace(dense)
    dynamic_shapes = {"args": dynamic_shapes_list}
    dense_layer_trt = trt_convert(dense_layer, values_list_cuda, dynamic_shapes)
    dict_res = dense_layer_trt(values_list_cuda)
    logger.info("dense trt res: %s", dict_res)

    # save combined_model
    combined_model = ScriptWrapperTRT(emb_trace_gpu, dense_layer_trt)
    result = combined_model(data, "cuda:0")
    logger.info("combined model result: %s", result)
    # combined_model = symbolic_trace(combined_model)
    combined_model = torch.jit.trace(
        combined_model, example_inputs=(data,), strict=False
    )
    scripted_model = torch.jit.script(combined_model)
    # pyre-ignore [16]
    scripted_model.save(os.path.join(save_dir, "scripted_model.pt"))

    if is_debug_trt():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function("model_inference_dense"):
                dict_res = dense(values_list_cuda)
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            with record_function("model_inference_dense_trt"):
                dict_res = dense_layer_trt(values_list_cuda)
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
