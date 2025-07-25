# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common util functions used by layers."""

from google.protobuf import struct_pb2
from google.protobuf.descriptor import FieldDescriptor


# is_proto_message 是一个用于检查 Protocol Buffer (PB) 对象的工具函数，它判断给定的字段是否是 PB 消息类型字段。该函数的设计主要用于处理 Protocol Buffer 对象的动态属性和类型检查，确保字段符合特定的消息类型。
def is_proto_message(pb_obj, field):
    if not hasattr(pb_obj, "DESCRIPTOR"):
        return False
    if field not in pb_obj.DESCRIPTOR.fields_by_name:
        return False
    field_type = pb_obj.DESCRIPTOR.fields_by_name[field].type
    return field_type == FieldDescriptor.TYPE_MESSAGE


# Parameter 类是一个用于封装参数的工具类，支持处理结构化参数和 Protocol Buffer (PB) 消息类型的参数。它提供了一些便捷的方法和属性，用于访问、修改和验证参数，同时支持嵌套结构和默认值处理。
class Parameter(object):
    def __init__(self, params, is_struct, l2_reg=None):
        self.params = params
        self.is_struct = is_struct
        self._l2_reg = l2_reg

    @staticmethod
    def make_from_pb(config):
        return Parameter(config, False)

    def get_pb_config(self):
        assert not self.is_struct, "Struct parameter can not convert to pb config"
        return self.params

    @property
    def l2_regularizer(self):
        return self._l2_reg

    @l2_regularizer.setter
    def l2_regularizer(self, value):
        self._l2_reg = value

    def __getattr__(self, key):
        if self.is_struct:
            if key not in self.params:
                return None
            value = self.params[key]
            if type(value) == struct_pb2.Struct:
                return Parameter(value, True, self._l2_reg)
            else:
                return value
        value = getattr(self.params, key)
        if is_proto_message(self.params, key):
            return Parameter(value, False, self._l2_reg)
        return value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def get_or_default(self, key, def_val):
        if self.is_struct:
            if key in self.params:
                if def_val is None:
                    return self.params[key]
                value = self.params[key]
                if type(value) == float:
                    return type(def_val)(value)
                return value
            return def_val
        else:  # pb message
            value = getattr(self.params, key, def_val)
            if hasattr(value, "__len__"):  # repeated
                return value if len(value) > 0 else def_val
            try:
                if self.params.HasField(key):
                    return value
            except ValueError:
                pass
            return def_val  # maybe not equal to the default value of msg field

    def check_required(self, keys):
        if not self.is_struct:
            return
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for key in keys:
            if key not in self.params:
                raise KeyError("%s must be set in params" % key)

    def has_field(self, key):
        if self.is_struct:
            return key in self.params
        else:
            return self.params.HasField(key)


# params_to_dict 函数，用于将 Parameter 对象转换为字典格式。
def params_to_dict(parameter):
    """Convert Parameter object to a dictionary."""

    def convert(param):
        if isinstance(param, Parameter):
            if param.is_struct:
                return {key: convert(value) for key, value in param.params.items()}
            else:  # PB message
                result = {}
                for field in param.params.DESCRIPTOR.fields:
                    key = field.name
                    value = getattr(param.params, key, None)
                    if value is not None:
                        if is_proto_message(param.params, key):
                            result[key] = convert(
                                Parameter(value, False, param.l2_regularizer)
                            )
                        elif isinstance(value, struct_pb2.Struct):
                            result[key] = convert(
                                Parameter(value, True, param.l2_regularizer)
                            )
                        else:
                            result[key] = value
                return result
        elif isinstance(param, struct_pb2.Struct):
            return {key: convert(value) for key, value in param.fields.items()}
        else:
            return param

    return convert(parameter)
