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

# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS, DataType
from torchrec.quant import embedding_modules


def is_input_tile() -> bool:
    """Judge is input file or not."""
    input_tile = os.environ.get("INPUT_TILE")
    if input_tile and (input_tile[0] == "2" or input_tile[0] == "3"):
        return True
    return False


def is_input_tile_emb() -> bool:
    """Judge is input file or not.

    Embedding Split user/item
    """
    input_tile = os.environ.get("INPUT_TILE")
    if input_tile and input_tile[0] == "3":
        return True
    return False


def is_aot() -> bool:
    """Judge is inductor or not."""
    is_aot = os.environ.get("ENABLE_AOT")
    if is_aot and is_aot[0] == "1":
        return True
    else:
        return False


def is_trt() -> bool:
    """Judge is trt or not."""
    is_trt = os.environ.get("ENABLE_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


def is_cuda_export() -> bool:
    """Judge is trt/aot or not."""
    return is_trt() or is_aot()


def is_trt_predict(model_path: str) -> bool:
    """Judge is trt or not in predict."""
    with open(model_path + "/model_acc.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    is_trt = data.get("ENABLE_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


def is_debug_trt() -> bool:
    """Judge is debug trt or not.

    Embedding Split user/item
    """
    is_trt = os.environ.get("DEBUG_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


def is_quant() -> bool:
    """Judge is quant or not."""
    is_quant = os.environ.get("QUANT_EMB")
    if is_quant and is_quant[0] == "0":
        return False
    return True


def quant_dtype() -> torch.dtype:
    """Get embedding quant dtype."""
    str_to_dtype = {
        "FP32": torch.float,
        "FP16": torch.half,
        "INT8": torch.qint8,
        "INT4": torch.quint4x2,
        "INT2": torch.quint2x4,
    }
    quant_dtype_str = os.environ.get("QUANT_EMB", "INT8")
    if quant_dtype_str == "1":
        # for compatible
        quant_dtype_str = "INT8"
    if quant_dtype_str not in str_to_dtype:
        raise ValueError(
            f"Unknown QUANT_EMB: {quant_dtype_str},"
            f"available types: {list(str_to_dtype.keys())}"
        )
    else:
        return str_to_dtype[quant_dtype_str]


def write_mapping_file_for_input_tile(
    state_dict: Dict[str, torch.Tensor], remap_file_path: str
) -> None:
    r"""Mapping ebc params to ebc_user and ebc_item Updates the model's state.

    dictionary with adapted parameters for the input tile.

    Args:
        state_dict (Dict[str, torch.Tensor]): model state_dict
        remap_file_path (str) : store new_params_name\told_params_name\n
    """
    input_tile_mapping = {
        ".ebc_user.embedding_bags.": ".ebc.embedding_bags.",
        ".mc_ebc_user._embedding_module.": ".mc_ebc._embedding_module.",
        ".mc_ebc_user._managed_collision_collection.": ".mc_ebc._managed_collision_collection.",  # NOQA
        ".ec_list_user.": ".ec_list.",
        ".mc_ec_list_user.": ".mc_ec_list.",
    }

    remap_str = ""
    for key, _ in state_dict.items():
        for input_tile_key in input_tile_mapping:
            if input_tile_key in key:
                src_key = key.replace(
                    input_tile_key, input_tile_mapping[input_tile_key]
                )
                remap_str += key + "\t" + src_key + "\n"

    with open(remap_file_path, "w") as f:
        f.write(remap_str)


def export_acc_config() -> Dict[str, str]:
    """Export acc config for model online inference."""
    # use int64 sparse id as input
    acc_config = {"SPARSE_INT64": "1"}
    if "INPUT_TILE" in os.environ:
        acc_config["INPUT_TILE"] = os.environ["INPUT_TILE"]
    if "QUANT_EMB" in os.environ:
        acc_config["QUANT_EMB"] = os.environ["QUANT_EMB"]
    if "ENABLE_TRT" in os.environ:
        acc_config["ENABLE_TRT"] = os.environ["ENABLE_TRT"]
    if "ENABLE_AOT" in os.environ:
        acc_config["ENABLE_AOT"] = os.environ["ENABLE_AOT"]
    return acc_config


# fix fp32 quantize
def _quantize_state_dict(
    module: nn.Module,
    table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]],
    table_name_to_data_type: Dict[str, DataType],
    table_name_to_num_embeddings_post_pruning: Optional[Dict[str, int]] = None,
) -> torch.device:
    device = torch.device("cpu")
    if not table_name_to_num_embeddings_post_pruning:
        table_name_to_num_embeddings_post_pruning = {}

    for key, tensor in module.state_dict().items():
        # Extract table name from state dict key.
        # e.g. ebc.embedding_bags.t1.weight
        splits = key.split(".")
        assert splits[-1] == "weight"
        table_name = splits[-2]
        data_type = table_name_to_data_type[table_name]
        num_rows = tensor.shape[0]

        if table_name in table_name_to_num_embeddings_post_pruning:
            num_rows = table_name_to_num_embeddings_post_pruning[table_name]

        device = tensor.device
        num_bits = DATA_TYPE_NUM_BITS[data_type]

        if tensor.is_meta:
            quant_weight = torch.empty(
                (num_rows, (tensor.shape[1] * num_bits) // 8),
                device="meta",
                dtype=torch.uint8,
            )
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                scale_shift = torch.empty(
                    (num_rows, 4),
                    device="meta",
                    dtype=torch.uint8,
                )
            else:
                scale_shift = None
        else:
            if num_rows != tensor.shape[0]:
                tensor = tensor[:num_rows, :]
            if tensor.dtype == torch.float or tensor.dtype == torch.float16:
                if data_type == DataType.FP16:
                    if tensor.dtype == torch.float:
                        tensor = tensor.half()
                    quant_res = tensor.view(torch.uint8)
                elif data_type == DataType.FP32:
                    if tensor.dtype == torch.float16:
                        tensor = tensor.float()
                    quant_res = tensor.view(torch.uint8)
                else:
                    quant_res = (
                        torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                            tensor, num_bits
                        )
                    )
            else:
                raise Exception("Unsupported dtype: {tensor.dtype}")
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                quant_weight, scale_shift = (
                    quant_res[:, :-4],
                    quant_res[:, -4:],
                )
            else:
                quant_weight, scale_shift = quant_res, None
        table_name_to_quantized_weights[table_name] = (quant_weight, scale_shift)
    return device


# pyre-ignore [9]
embedding_modules.quantize_state_dict = _quantize_state_dict
