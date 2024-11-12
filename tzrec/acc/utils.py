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
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


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


def is_trt() -> bool:
    """Judge is trt or not."""
    is_trt = os.environ.get("ENABLE_TRT")
    if is_trt and is_trt[0] == "1":
        return True
    return False


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


def update_state_dict_for_input_tile(state_dict: Dict[str, torch.Tensor]) -> None:
    """Copy the ebc params to ebc_user and ebc_item Updates the model's state.

    dictionary with adapted parameters for the input tile.

    Args:
        state_dict (Dict[str, torch.Tensor]): model state_dict
    """
    input_tile_keys = [
        ".ebc_user.embedding_bags.",
        ".ebc_item.embedding_bags.",
    ]
    input_tile_keys_ec = [
        ".ec_list_user.",
        ".ec_list_item.",
    ]

    for key, dst_tensor in state_dict.items():
        for input_tile_key in input_tile_keys:
            if input_tile_key in key:
                src_key = key.replace(input_tile_key, ".ebc.embedding_bags.")
                src_tensor = state_dict[src_key]
                dst_tensor.copy_(src_tensor)

        for input_tile_key in input_tile_keys_ec:
            if input_tile_key in key:
                src_key = key.replace(input_tile_key, ".ec_list.")
                src_tensor = state_dict[src_key]
                dst_tensor.copy_(src_tensor)


def write_mapping_file_for_input_tile(
    state_dict: Dict[str, torch.Tensor], remap_file_path: str
) -> None:
    r"""Mapping ebc params to ebc_user and ebc_item Updates the model's state.

    dictionary with adapted parameters for the input tile.

    Args:
        state_dict (Dict[str, torch.Tensor]): model state_dict
        remap_file_path (str) : store new_params_name\told_params_name\n
    """
    input_tile_keys = [
        ".ebc_user.embedding_bags.",
        ".ebc_item.embedding_bags.",
    ]
    input_tile_keys_ec = [
        ".ec_list_user.",
        ".ec_list_item.",
    ]

    remap_str = ""
    for key, _ in state_dict.items():
        for input_tile_key in input_tile_keys:
            if input_tile_key in key:
                src_key = key.replace(input_tile_key, ".ebc.embedding_bags.")
                remap_str += key + "\t" + src_key + "\n"

        for input_tile_key in input_tile_keys_ec:
            if input_tile_key in key:
                src_key = key.replace(input_tile_key, ".ec_list.")
                remap_str += key + "\t" + src_key + "\n"

    with open(remap_file_path, "w") as f:
        f.write(remap_str)


def export_acc_config() -> Dict[str, str]:
    """Export acc config for model online inference."""
    acc_config = dict()
    if "INPUT_TILE" in os.environ:
        acc_config["INPUT_TILE"] = os.environ["INPUT_TILE"]
    if "QUANT_EMB" in os.environ:
        acc_config["QUANT_EMB"] = os.environ["QUANT_EMB"]
    if "ENABLE_TRT" in os.environ:
        acc_config["ENABLE_TRT"] = os.environ["ENABLE_TRT"]
    return acc_config


def dicts_are_equal(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
) -> bool:
    """Compare dict[str,torch.Tensor]."""
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False

    return True


def lists_are_equal(list1: List[torch.Tensor], list2: List[torch.Tensor]) -> bool:
    """Compare List[torch.Tensor]."""
    if len(list1) != len(list2):
        return False

    for i in range(len(list1)):
        if not torch.equal(list1[i], list2[i]):
            return False
    return True


def is_close(df1: pd.DataFrame, df2: pd.DataFrame, abs_tol: float) -> bool:
    """Compare DataFrame."""
    if df1.shape != df2.shape:
        return False
    abs_diff = np.abs(df1.values - df2.values)
    result = np.all(abs_diff <= abs_tol)
    # pyre-ignore [7]
    return result
