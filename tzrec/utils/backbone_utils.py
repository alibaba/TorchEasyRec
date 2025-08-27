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


def is_proto_message(pb_obj, field):
    """Check if a given field in a Protocol Buffer object is a message type field.

    This utility function is designed to handle Protocol Buffer object dynamic
    attributes and type checking, ensuring that fields conform to specific
    message types.

    Args:
        pb_obj: The Protocol Buffer object to inspect.
        field: The field name to check for message type.

    Returns:
        bool: True if the field is a Protocol Buffer message type, False otherwise.
    """
    if not hasattr(pb_obj, "DESCRIPTOR"):
        return False
    if field not in pb_obj.DESCRIPTOR.fields_by_name:
        return False
    field_type = pb_obj.DESCRIPTOR.fields_by_name[field].type
    return field_type == FieldDescriptor.TYPE_MESSAGE


class Parameter(object):
    """A utility class for encapsulating and managing parameters.

    This class supports handling both structured parameters and Protocol Buffer (PB)
    message type parameters. It provides convenient methods and properties for
    accessing, modifying, and validating parameters, while supporting nested
    structures and default value handling.

    Attributes:
        params: The parameter data (dict for struct or PB message object).
        is_struct: Boolean indicating if this is a struct-type parameter.
        _l2_reg: L2 regularization value for this parameter.
    """

    def __init__(self, params, is_struct, l2_reg=None):
        self.params = params
        self.is_struct = is_struct
        self._l2_reg = l2_reg

    @staticmethod
    def make_from_pb(config):
        """Create a Parameter instance from a Protocol Buffer configuration.

        Args:
            config: The Protocol Buffer configuration object.

        Returns:
            Parameter: A new Parameter instance with is_struct=False.
        """
        return Parameter(config, False)

    def get_pb_config(self):
        """Get the Protocol Buffer configuration object.

        Returns:
            The Protocol Buffer configuration object.

        Raises:
            AssertionError: If this Parameter instance is a struct type.
        """
        assert not self.is_struct, "Struct parameter can not convert to pb config"
        return self.params

    @property
    def l2_regularizer(self):
        """Get the L2 regularization value.

        Returns:
            The L2 regularization value or None if not set.
        """
        return self._l2_reg

    @l2_regularizer.setter
    def l2_regularizer(self, value):
        self._l2_reg = value

    def __getattr__(self, key):
        if self.is_struct:
            if key not in self.params:
                return None
            value = self.params[key]
            if isinstance(value, struct_pb2.Struct):
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
        """Get parameter value or return default if not present or empty.

        Args:
            key: The parameter key to retrieve.
            def_val: The default value to return if key is not found or empty.

        Returns:
            The parameter value if present and non-empty, otherwise def_val.
        """
        if self.is_struct:
            if key in self.params:
                if def_val is None:
                    return self.params[key]
                value = self.params[key]
                if isinstance(value, float):
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
        """Check that required keys are present in the struct parameters.

        Args:
            keys: A key name or list/tuple of key names to check for presence.

        Raises:
            KeyError: If any required key is missing from the struct parameters.
        """
        if not self.is_struct:
            return
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for key in keys:
            if key not in self.params:
                raise KeyError("%s must be set in params" % key)

    def has_field(self, key):
        """Check if the parameter has the specified field.

        Args:
            key: The field name to check.

        Returns:
            bool: True if the field exists, False otherwise.
        """
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


def infer_input_dim(input_dim, input_fn=None, input_slice=None):
    """推断经过变换后的输入维度.

    Args:
        input_dim: int 或 List[int]，原始输入维度
        input_fn: str，lambda表达式字符串
        input_slice: str，格式如'[1]'或'[0:2]'

    Returns:
        变换后的输入维度（int或list）
    """
    # 先处理input_slice
    if input_slice is not None:
        # 假定input_dim是list或tuple的各项维度
        # input_slice: '[1]', '[0]', '[0:2]'
        idx = eval(input_slice)
        # 支持单一索引和切片
        if isinstance(idx, int):
            input_dim = input_dim[idx]
        elif isinstance(idx, slice):
            input_dim = input_dim[idx]
        elif isinstance(idx, list):
            input_dim = [input_dim[i] for i in idx]
        else:
            raise ValueError(f"input_slice({input_slice})格式无法识别")

    # 再处理input_fn (只支持常见表达式)
    if input_fn is not None:
        # 仅支持有限的自动推断，比如sum、reshape等
        if "sum" in input_fn:
            # 提取dim和keepdim
            import re

            m = re.search(r"sum\(dim=(\d+)(?:, *keepdim=(True|False))?", input_fn)
            if m:
                dim = int(m.group(1))
                keepdim = (m.group(2) == "True") if m.group(2) is not None else False
                # input_dim 可以是int或tuple/list
                # 推导后维度
                if isinstance(input_dim, int):
                    raise ValueError("sum运算作用在多维张量上，int维度不够信息")
                new_dim = list(input_dim)
                if keepdim:
                    new_dim[dim] = 1
                else:
                    del new_dim[dim]
                if len(new_dim) == 1:
                    return new_dim[0]
                else:
                    return tuple(new_dim)

        elif "lambda x: [x]" in input_fn or input_fn.strip() == "lambda x: [x]":
            # 将输入打包成列表
            return [input_dim]
        # 其他lambda表达式很难推断，需要你补充更多分支
        else:
            # 不认识的表达式，保守返回原始input_dim
            return input_dim

    return input_dim
