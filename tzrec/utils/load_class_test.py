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

import unittest

import torch

from tzrec.utils.load_class import load_by_path


class LoadClassTest(unittest.TestCase):
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
