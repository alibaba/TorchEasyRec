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


def is_proto_message(pb_obj, field) -> bool:
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
    """

    def __init__(self, params, is_struct):
        self.params = params
        self.is_struct = is_struct

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

    def __getattr__(self, key):
        if self.is_struct:
            if key not in self.params:
                return None
            value = self.params[key]
            if isinstance(value, struct_pb2.Struct):
                return Parameter(value, True)
            else:
                return value
        value = getattr(self.params, key)
        if is_proto_message(self.params, key):
            return Parameter(value, False)
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

    def check_required(self, keys) -> None:
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

    def has_field(self, key) -> bool:
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


def params_to_dict(parameter) -> dict:
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
                            result[key] = convert(Parameter(value, False))
                        elif isinstance(value, struct_pb2.Struct):
                            result[key] = convert(Parameter(value, True))
                        else:
                            result[key] = value
                return result
        elif isinstance(param, struct_pb2.Struct):
            return {key: convert(value) for key, value in param.fields.items()}
        else:
            return param

    return convert(parameter)
