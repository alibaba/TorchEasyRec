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
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tzrec.utils import load_class
from tzrec.utils.load_class import (
    _auto_import_custom_models,
    auto_import_package,
    load_by_path,
)


class LoadClassTest(unittest.TestCase):
    def test_auto_import_package(self):
        package = SimpleNamespace(
            __name__="tzrec_custom.models", __path__=["custom/models"]
        )
        modules = [
            pkgutil.ModuleInfo(None, "tzrec_custom.models.rank", False),
            pkgutil.ModuleInfo(None, "tzrec_custom.models.rank_test", False),
            pkgutil.ModuleInfo(None, "tzrec_custom.models.match", False),
        ]
        with (
            mock.patch.object(
                load_class.importlib,
                "import_module",
                side_effect=[package, mock.Mock(), mock.Mock()],
            ) as import_module,
            mock.patch.object(
                load_class.pkgutil,
                "walk_packages",
                return_value=modules,
            ),
        ):
            auto_import_package("tzrec_custom.models")

        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            [
                "tzrec_custom.models",
                "tzrec_custom.models.rank",
                "tzrec_custom.models.match",
            ],
        )

    def test_default_custom_package_is_optional(self):
        error = ModuleNotFoundError(
            "No module named 'tzrec_custom'", name="tzrec_custom"
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("tzrec.utils.load_class.auto_import_package", side_effect=error),
        ):
            _auto_import_custom_models()

    def test_configured_custom_package_is_required(self):
        error = ModuleNotFoundError("No module named 'my_models'", name="my_models")
        with (
            mock.patch.dict(
                os.environ, {"TZREC_CUSTOM_PACKAGE": "my_models"}, clear=True
            ),
            mock.patch("tzrec.utils.load_class.auto_import_package", side_effect=error),
            self.assertRaises(ModuleNotFoundError),
        ):
            _auto_import_custom_models()

    def test_load_by_path(self):
        loaded_cls = load_by_path("nn.ReLU")
        self.assertEqual(loaded_cls, torch.nn.ReLU)
        loaded_cls = load_by_path("torch.nn.ReLU")
        self.assertEqual(loaded_cls, torch.nn.ReLU)
        loaded_cls = load_by_path("torch.nn.MyReLU")
        self.assertEqual(loaded_cls, None)
        loaded_cls = load_by_path("")
        self.assertEqual(loaded_cls, None)


if __name__ == "__main__":
    unittest.main()
