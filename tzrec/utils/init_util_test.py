# Copyright (c) 2026, Alibaba Group;
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
import unittest
from functools import partial
from unittest import mock

import torch
from torch import nn

from tzrec.utils import init_util


class InitUtilTest(unittest.TestCase):
    def test_create_init_fn_swaps_trunc_normal(self):
        init_fn = init_util.create_init_fn("nn.init.trunc_normal_,mean=0.0,std=0.0125")
        self.assertIsInstance(init_fn, partial)
        self.assertIs(init_fn.func, init_util.trunc_normal_)
        self.assertEqual(init_fn.keywords, {"mean": 0.0, "std": 0.0125})

    def test_create_init_fn_keeps_other_init(self):
        init_fn = init_util.create_init_fn("nn.init.uniform_,a=-0.01,b=0.01")
        self.assertIsInstance(init_fn, partial)
        self.assertIs(init_fn.func, nn.init.uniform_)
        self.assertEqual(init_fn.keywords, {"a": -0.01, "b": 0.01})

    def test_trunc_normal_default_native(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("USE_INPLACE_TRUNC_NORMAL", None)
            with mock.patch.object(
                nn.init, "trunc_normal_", wraps=nn.init.trunc_normal_
            ) as mock_native:
                t = torch.empty(1024)
                init_util.trunc_normal_(t, mean=0.5, std=0.1, a=-1.0, b=1.0)
                mock_native.assert_called_once_with(
                    t, 0.5, 0.1, -1.0, 1.0, generator=None
                )

    def test_trunc_normal_env_inplace(self):
        with mock.patch.dict(os.environ, {"USE_INPLACE_TRUNC_NORMAL": "1"}):
            with mock.patch.object(
                init_util,
                "_inplace_trunc_normal_",
                wraps=init_util._inplace_trunc_normal_,
            ) as mock_inplace:
                t = torch.empty(1024)
                init_util.trunc_normal_(t, mean=0.5, std=0.1, a=-1.0, b=1.0)
                mock_inplace.assert_called_once_with(
                    t, 0.5, 0.1, -1.0, 1.0, generator=None
                )

    def test_trunc_normal_inplace_statistics(self):
        t = torch.empty(1000, 1000)
        with mock.patch.dict(os.environ, {"USE_INPLACE_TRUNC_NORMAL": "1"}):
            init_util.trunc_normal_(t, mean=0.5, std=0.0125)
        self.assertAlmostEqual(t.mean().item(), 0.5, delta=1e-4)
        self.assertAlmostEqual(t.std().item(), 0.0125, delta=1e-4)
        self.assertGreaterEqual(t.min().item(), -2.0)
        self.assertLessEqual(t.max().item(), 2.0)

    def test_trunc_normal_inplace_truncates(self):
        t = torch.empty(1000, 1000)
        with mock.patch.dict(os.environ, {"USE_INPLACE_TRUNC_NORMAL": "1"}):
            init_util.trunc_normal_(t, mean=0.0, std=1.0, a=-1.0, b=1.0)
        self.assertGreaterEqual(t.min().item(), -1.0)
        self.assertLessEqual(t.max().item(), 1.0)
        self.assertLess(t.std().item(), 0.6)

    def test_trunc_normal_inplace_reproducible(self):
        with mock.patch.dict(os.environ, {"USE_INPLACE_TRUNC_NORMAL": "1"}):
            t1 = init_util.trunc_normal_(
                torch.empty(1000),
                std=0.1,
                generator=torch.Generator().manual_seed(42),
            )
            t2 = init_util.trunc_normal_(
                torch.empty(1000),
                std=0.1,
                generator=torch.Generator().manual_seed(42),
            )
        torch.testing.assert_close(t1, t2, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
