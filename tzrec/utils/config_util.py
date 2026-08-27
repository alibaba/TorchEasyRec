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
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from google.protobuf import any_pb2, json_format, symbol_database, text_format
from google.protobuf.message import Message

from tzrec.protos import data_pb2, eval_pb2, export_pb2, pipeline_pb2, train_pb2
from tzrec.protos.data_pb2 import FgMode
from tzrec.utils.load_class import import_class
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
    with open(pipeline_config_path) as f:
        content = f.read()
    is_json = pipeline_config_path.endswith(".json")
    _preload_custom_model(content, is_json)

    config = pipeline_pb2.EasyRecConfig()
    if is_json:
        json_format.Parse(content, config, ignore_unknown_fields=allow_unknown_field)
    else:
        text_format.Merge(content, config, allow_unknown_field=allow_unknown_field)
    # compatible for fg_encoded
    config.data_config.fg_mode = _get_compatible_fg_mode(config.data_config)
    return config


def _preload_custom_model(content: str, is_json: bool) -> None:
    """Import the custom model module before the pipeline config is parsed.

    A custom model config is embedded in a google.protobuf.Any, whose type must
    be registered in the default descriptor pool before parsing. Read class_path
    with a pre-parse that tolerates the not-yet-registered Any, and import the
    module defining the model, which registers the descriptor on the way.

    Args:
        content (str): raw content of a pipeline config.
        is_json (bool): whether the content is in json format.
    """
    preload = pipeline_pb2.PreloadConfig()
    if is_json:
        json_format.Parse(content, preload, ignore_unknown_fields=True)
    else:
        text_format.Merge(content, preload, allow_unknown_field=True)
    class_path = preload.model_config.custom_model.class_path
    if class_path:
        import_class(class_path)


def unpack_any(any_config: any_pb2.Any) -> Optional[Message]:
    """Unpack a google.protobuf.Any into its concrete message.

    Args:
        any_config (Any): a google.protobuf.Any message.

    Return:
        the packed message, or None when the Any is not set.
    """
    type_name = any_config.TypeName()
    if not type_name:
        return None
    try:
        config_cls = symbol_database.Default().GetSymbol(type_name)
    except KeyError as e:
        raise ValueError(
            f"config type [{type_name}] is not registered, please make sure the "
            "module defining the model imports its generated *_pb2 module."
        ) from e
    config = config_cls()
    if not any_config.Unpack(config):
        raise ValueError(f"failed to unpack config of type [{type_name}].")
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
    # NOTE: typeshed ships protobuf 5.x stubs, where this argument was renamed to
    # always_print_fields_with_no_presence; we pin protobuf 4.x via grpcio-tools.
    return json_format.MessageToDict(
        config,
        # pyrefly: ignore[unexpected-keyword]
        including_default_value_fields=True,
        preserving_proto_field_name=True,
    )


def which_msg(config: Message, oneof_group: str) -> str:
    """Returns the name of the message that is set inside a oneof group."""
    return getattr(config, config.WhichOneof(oneof_group)).__class__.__name__


def use_dense_ema(
    config: Optional[Union[eval_pb2.EvalConfig, export_pb2.ExportConfig]],
    train_config: train_pb2.TrainConfig,
) -> bool:
    """Resolve whether evaluation or export should use Dense EMA parameters.

    Args:
        config: EvalConfig or ExportConfig containing the optional override,
            or None to use the training default.
        train_config: Training configuration providing the default.
    """
    if config is not None and config.HasField("use_dense_ema"):
        return bool(config.use_dense_ema)
    return train_config.dense_optimizer.HasField("ema")


def get_inference_batch_size(data_config: data_pb2.DataConfig) -> int:
    """Get the effective batch size for a non-training dataloader.

    Args:
        data_config: Data configuration containing batch-size settings.
    """
    if data_config.HasField("eval_batch_size"):
        return data_config.eval_batch_size
    return data_config.batch_size


def set_inference_batch_size(data_config: data_pb2.DataConfig, batch_size: int) -> None:
    """Set and synchronize the batch size used by inference paths.

    Args:
        data_config: Data configuration containing batch-size settings.
        batch_size: Batch size to use for non-training dataloaders.
    """
    data_config.batch_size = batch_size
    data_config.eval_batch_size = batch_size


def _get_compatible_fg_mode(data_config: data_pb2.DataConfig) -> FgMode:
    """Compat for fg_encoded."""
    if data_config.HasField("fg_encoded"):
        logger.warning(
            "data_config.fg_encoded will be deprecated, please use data_config.fg_mode."
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
