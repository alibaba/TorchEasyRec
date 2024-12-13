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
import re
from typing import Any, Dict, List, Type

import numpy as np
from google.protobuf import json_format, text_format
from google.protobuf.message import Message

from tzrec.protos import pipeline_pb2
from tzrec.protos.data_pb2 import FgMode
from tzrec.utils.logging_util import logger


def load_pipeline_config(
    pipeline_config_path: str, allow_unknown_field: bool = False
) -> pipeline_pb2.EasyRecConfig:
    """Load pipeline config.

    Args:
        pipeline_config_path (str): path to pipeline_pb2.EasyRecConfig.
        allow_unknown_field (bool): skip over unknown field and keep
            parsing. Avoid to use this option if possible.

    Return:
        a object of pipeline_pb2.EasyRecConfig.
    """
    config = pipeline_pb2.EasyRecConfig()
    with open(pipeline_config_path) as f:
        if pipeline_config_path.endswith(".json"):
            json_format.Parse(
                f.read(), config, ignore_unknown_fields=allow_unknown_field
            )
        else:
            text_format.Merge(f.read(), config, allow_unknown_field=allow_unknown_field)
    # compatible for fg_encoded
    config.data_config.fg_mode = _get_compatible_fg_mode(config.data_config)
    return config


def save_message(message: Message, filepath: str) -> None:
    """Saves a proto message object to text file.

    Args:
        message: a proto message.
        filepath: save path.
    """
    directory, _ = os.path.split(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pbtxt = text_format.MessageToString(message, as_utf8=True)
    with open(filepath, "w") as f:
        f.write(pbtxt)


def config_to_kwargs(config: Message) -> Dict[str, Any]:
    """Convert a message to a config dict."""
    return json_format.MessageToDict(
        config, including_default_value_fields=True, preserving_proto_field_name=True
    )


def which_msg(config: Message, oneof_group: str) -> str:
    """Returns the name of the message that is set inside a oneof group."""
    return getattr(config, config.WhichOneof(oneof_group)).__class__.__name__


def _get_compatible_fg_mode(data_config: Message) -> FgMode:
    """Compat for fg_encoded."""
    if data_config.HasField("fg_encoded"):
        logger.warning(
            "data_config.fg_encoded will be deprecated, "
            "please use data_config.fg_mode."
        )
        if data_config.fg_encoded:
            fg_mode = FgMode.FG_NONE
        elif data_config.fg_threads > 0:
            fg_mode = FgMode.FG_DAG
        else:
            fg_mode = FgMode.FG_NORMAL
    else:
        fg_mode = data_config.fg_mode
    return fg_mode


# pyre-ignore [24]
def _get_basic_types() -> List[Type]:
    dtypes = [
        bool,
        int,
        str,
        float,
        type(""),
        np.float16,
        np.float32,
        np.float64,
        np.char,
        np.byte,
        np.uint8,
        np.int8,
        np.int16,
        np.uint16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
    ]

    return dtypes


def _dot_split_with_bracket(s: str) -> List[str]:
    # Pattern to match text within square brackets, including the dots
    bracket_pattern = re.compile(r"[^\.]*\[[^\]]*\][^\.]*")

    # Temporary dictionary to hold the bracketed strings
    placeholders: Dict[str, str] = {}

    # pyre-ignore [2, 3]
    def replacer(match):
        key = f"PLACEHOLDER{len(placeholders)}"
        placeholders[key] = match.group(0)
        return key

    # Replace bracketed text with placeholders
    temp_string = bracket_pattern.sub(replacer, s)

    parts = temp_string.split(".")
    final_parts = [
        part if part not in placeholders else placeholders[part] for part in parts
    ]
    return final_parts


def edit_config(pipeline_config: Message, edit_config_json: Dict[str, Any]) -> Message:
    """Update params specified by automl.

    Args:
        pipeline_config (EasyRecConfig): a object of pipeline_pb2.EasyRecConfig.
        edit_config_json (dict): edit config json

    Returns:
        edited a object of pipeline_pb2.EasyRecConfig.
    """

    # pyre-ignore [2, 3]
    def _type_convert(proto, val, parent=None):
        if type(val) is not type(proto):
            try:
                if isinstance(proto, bool):
                    assert val in ["True", "true", "False", "false"]
                    val = val in ["True", "true"]
                else:
                    val = type(proto)(val)
            except ValueError as ex:
                if parent is None:
                    raise ex
                assert isinstance(proto, int)
                val = getattr(parent, val)
                assert isinstance(val, int)
        return val

    # pyre-ignore [2, 3]
    def _get_attr(obj, attr, only_last=False):
        # only_last means we only return the last element in paths array
        attr_toks = [x.strip() for x in _dot_split_with_bracket(attr) if x != ""]
        paths = []
        objs = [obj]
        nobjs = []
        for key in attr_toks:
            # clear old paths to clear new paths
            paths = []
            for obj in objs:
                if "[" in key:
                    pos = key.find("[")
                    name, cond = key[:pos], key[pos + 1 :]
                    cond = cond[:-1]
                    update_objs = getattr(obj, name)
                    # select all update_objs
                    if cond == ":":
                        for tid, update_obj in enumerate(update_objs):
                            paths.append((obj, update_obj, None, tid))
                            nobjs.append(update_obj)
                        continue

                    # select by range update_objs[1:10]
                    if ":" in cond:
                        colon_pos = cond.find(":")
                        sid = cond[:colon_pos]
                        if len(sid) == 0:
                            sid = 0
                        else:
                            sid = int(sid)
                        eid = cond[(colon_pos + 1) :]
                        if len(eid) == 0:
                            eid = len(update_objs)
                        else:
                            eid = int(eid)
                        for tid, update_obj in enumerate(update_objs[sid:eid]):
                            paths.append((obj, update_obj, None, tid + sid))
                            nobjs.append(update_obj)
                        continue

                    # for simple index update_objs[0]
                    try:
                        obj_id = int(cond)
                        obj = update_objs[obj_id]
                        paths.append((obj, update_objs, None, obj_id))
                        nobjs.append(obj)
                        continue
                    except ValueError:
                        pass

                    # for complex conditions a[optimizer.lr=20]
                    op_func_map = {
                        ">=": lambda x, y: x >= y,
                        "<=": lambda x, y: x <= y,
                        "<": lambda x, y: x < y,
                        ">": lambda x, y: x > y,
                        "=": lambda x, y: x == y,
                    }
                    cond_key = None
                    cond_val = None
                    op_func = None
                    for op in [">=", "<=", ">", "<", "="]:
                        tmp_pos = cond.rfind(op)
                        if tmp_pos != -1:
                            cond_key = cond[:tmp_pos]
                            cond_val = cond[(tmp_pos + len(op)) :]
                            op_func = op_func_map[op]
                            break

                    assert cond_key is not None, "invalid cond: %s" % cond
                    assert cond_val is not None, "invalid cond: %s" % cond

                    for tid, update_obj in enumerate(update_objs):
                        tmp, tmp_parent, _, _ = _get_attr(
                            update_obj, cond_key, only_last=True
                        )

                        cond_val = _type_convert(tmp, cond_val, tmp_parent)

                        # pyre-ignore [29]
                        if op_func(tmp, cond_val):
                            obj_id = tid
                            paths.append((update_obj, update_objs, None, obj_id))
                            nobjs.append(update_obj)
                else:
                    sub_obj = getattr(obj, key)
                    paths.append((sub_obj, obj, key, -1))
                    nobjs.append(sub_obj)
            # exchange to prepare for parsing next token
            objs = nobjs
            nobjs = []
        if only_last:
            return paths[-1]
        else:
            return paths

    for param_keys in edit_config_json:
        # multiple keys/vals combination
        param_vals = edit_config_json[param_keys]
        param_vals = [x.strip() for x in str(param_vals).split(";")]
        param_keys = [x.strip() for x in str(param_keys).split(";")]
        for param_key, param_val in zip(param_keys, param_vals):
            update_obj = pipeline_config
            tmp_paths = _get_attr(update_obj, param_key)
            # update a set of objs
            for tmp_val, tmp_obj, tmp_name, tmp_id in tmp_paths:
                # list and dict are not basic types, must be handle separately
                basic_types = _get_basic_types()
                if type(tmp_val) in basic_types:
                    # simple type cast
                    tmp_val = _type_convert(tmp_val, param_val, tmp_obj)
                    if tmp_name is None:
                        tmp_obj[tmp_id] = tmp_val
                    else:
                        setattr(tmp_obj, tmp_name, tmp_val)
                elif "Scalar" in str(type(tmp_val)) and "ClearField" in dir(tmp_obj):
                    tmp_obj.ClearField(tmp_name)
                    text_format.Parse("%s:%s" % (tmp_name, param_val), tmp_obj)
                else:
                    tmp_val.Clear()
                    param_val = param_val.strip()
                    if param_val.startswith("{") and param_val.endswith("}"):
                        param_val = param_val[1:-1]
                    text_format.Parse(param_val, tmp_val)

    return pipeline_config
