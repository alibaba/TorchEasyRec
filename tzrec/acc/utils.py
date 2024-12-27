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
from typing import Dict

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
    ]
    input_tile_keys_ec = [
        ".ec_list_user.",
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
    input_tile_mapping = {
        ".ebc_user.embedding_bags.": ".ebc.embedding_bags.",
        ".ebc_item.embedding_bags.": ".ebc.embedding_bags.",
        ".mc_ebc_user._embedding_module.": ".mc_ebc._embedding_module.",
        ".mc_ebc_item._embedding_module.": ".mc_ebc._embedding_module.",
        ".mc_ebc_user._managed_collision_collection.": ".mc_ebc._managed_collision_collection.",  # NOQA
        ".mc_ebc_item._managed_collision_collection.": ".mc_ebc._managed_collision_collection.",  # NOQA
        ".ec_list_user.": ".ec_list.",
        ".ec_list_item.": ".ec_list.",
        ".mc_ec_list_user.": ".mc_ec_list.",
        ".mc_ec_list_item.": ".mc_ec_list.",
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
    return acc_config
