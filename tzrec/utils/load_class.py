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
import pkgutil
import pydoc
import traceback
from abc import ABCMeta


def import_pkg(pkg_info, prefix_to_remove=None):
    """Import package.

    Args:
        pkg_info: pkgutil.ModuleInfo object
        prefix_to_remove: the package prefix to be removed
    """
    package_path = pkg_info[0].path
    if prefix_to_remove is not None:
        package_path = package_path.replace(prefix_to_remove, "")
    mod_name = pkg_info[1]

    if package_path.startswith("/"):
        # absolute path file, we should use relative import
        mod = pkg_info[0].find_module(mod_name)
        if mod is not None:
            # skip those test files
            if not mod_name.endswith("_test"):
                mod.load_module(pkg_info[1])
        else:
            raise Exception("import module %s failed" % (package_path + mod_name))
    else:
        # use similar import methods as the import keyword
        module_path = os.path.join(package_path, mod_name).replace("/", ".")
        # skip those test files
        if not mod_name.endswith("_test"):
            try:
                __import__(module_path)
            except Exception as e:
                raise ValueError(
                    "import module %s failed: %s" % (module_path, str(e))
                ) from e


def auto_import(user_path=None):
    """Auto import python files.

    So that register_xxx decorator will take effect.
    By default, we will import files in pre-defined directory and import all
    files recursively in user_dir

    Args:
        user_path: directory or file that store user-defined python code,
            by default we will only search file in current directory
    """
    # True False indicates import recursively or not
    pre_defined_dirs = [
        ("tzrec/models", False),
        ("tzrec/datasets", False),
        ("tzrec/features", False),
    ]

    curr_dir, _ = os.path.split(__file__)
    parent_dir = os.path.dirname(os.path.dirname(curr_dir))
    prefix_to_remove = None
    # dealing with when in sited-packages, remove parent directory prefix
    if parent_dir != "":
        for idx in range(len(pre_defined_dirs)):
            pre_defined_dirs[idx] = (
                os.path.join(parent_dir, pre_defined_dirs[idx][0]),
                pre_defined_dirs[idx][1],
            )
        prefix_to_remove = parent_dir + "/"

    if user_path is not None:
        if os.path.isdir(user_path):
            user_dir = user_path
        else:
            user_dir, _ = os.path.split(user_path)
        pre_defined_dirs.append((user_dir, True))

    for dir_path, recursive_import in pre_defined_dirs:
        for pkg_info in pkgutil.iter_modules([dir_path]):
            import_pkg(pkg_info, prefix_to_remove)

        if recursive_import:
            for root, dirs, _ in os.walk(dir_path):
                for subdir in dirs:
                    dirname = os.path.join(root, subdir)
                    for pkg_info in pkgutil.iter_modules([dirname]):
                        import_pkg(pkg_info, prefix_to_remove)


def register_class(class_map, class_name, cls):
    """Register a class into class_map.

    Args:
        class_map: class register map.
        class_name: name of the class.
        cls: a class.
    """
    assert class_name not in class_map or class_map[class_name] == cls, (
        f"confilict class {cls} , "
        f"{class_name} is already register to be {class_map[class_name]}"
    )
    class_map[class_name] = cls


def get_register_class_meta(class_map):
    """Get a meta class with registry.

    Args:
        class_map: class register map.

    Return:
        a meta class with registry.
    """

    class RegisterABCMeta(ABCMeta):
        def __new__(mcs, name, bases, attrs):
            newclass = super(RegisterABCMeta, mcs).__new__(mcs, name, bases, attrs)
            register_class(class_map, name, newclass)

            @classmethod
            def create_class(cls, name):
                if name in class_map:
                    return class_map[name]
                else:
                    raise Exception(
                        "Class %s is not registered. Available ones are %s"
                        % (name, list(class_map.keys()))
                    )

            newclass.create_class = create_class
            return newclass

    return RegisterABCMeta


def load_by_path(path):
    """Load functions or modules or classes.

    Args:
        path: path to modules or functions or classes,
            such as: torch.nn.ReLU

    Return:
        modules or functions or classes
    """
    path = path.strip()
    if path == "" or path is None:
        return None
    if "lambda" in path:
        return eval(path)
    components = path.split(".")
    if components[0] == "nn":
        components.insert(0, "torch")
    path = ".".join(components)
    try:
        return pydoc.locate(path)
    except pydoc.ErrorDuringImport:
        print("load %s failed: %s" % (path, traceback.format_exc()))
        return None


def load_torch_layer(name):
    """Load torch layer class.

    Args:
      name (str): Module class name, e.g. 'Linear' or 'YourCustomLayer'

    Return:
      (layer_class, is_customize)
      module_class: The class object (e.g., torch.nn.Linear)
      is_customize: True if loaded from custom namespace, False if from torch.nn
    """
    name = name.strip()
    if name == "" or name is None:
        return None

    path = "tzrec.modules." + name
    try:
        cls = pydoc.locate(path)
        if cls is not None:
            return cls, True
        path = "torch.nn." + name
        return pydoc.locate(path), False
    except pydoc.ErrorDuringImport:
        print("load torch layer %s failed" % name)
        import logging

        logging.error("load torch layer %s failed: %s" % (name, traceback.format_exc()))
        return None, False
