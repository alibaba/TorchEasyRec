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


import unittest

import torch
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from tzrec.ops import Kernel
from tzrec.ops.layer_norm import (
    layer_norm,
    swish_layer_norm,
)
from tzrec.utils.test_util import gpu_unavailable


class LayerNormTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.sampled_from([4200000]),
        D=st.sampled_from([512]),
        is_swish=st.sampled_from([False]),
        dtype=st.sampled_from(
            [torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=1,
    )
    # pyre-ignore[2]
    def test_large_tensors(self, *args, **kwargs) -> None:
        self._test_layernorm(
            *args,
            **kwargs,
            ref_kernel=Kernel.TRITON,
            real_kernel=Kernel.TRITON,
            skip_comparisons=True,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=0, max_value=10000),
        D=st.integers(min_value=32, max_value=512),
        is_swish=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_ln(self, *args, **kwargs) -> None:
        self._test_layernorm(
            *args,
            **kwargs,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.TRITON,
        )

    def _test_layernorm(
        self,
        N: int,
        D: int,
        is_swish: bool,
        dtype: torch.dtype,
        ref_kernel: Kernel,
        real_kernel: Kernel,
        skip_comparisons: bool = False,
    ) -> None:
        N = N // 4 * 4
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .normal_(0.0, 1.0)
            .requires_grad_()
        )
        weight = (
            torch.empty((D,), device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias = (
            torch.empty((D,), device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        if is_swish:
            layer_norm_func = swish_layer_norm
        else:
            layer_norm_func = layer_norm
        # ref
        ref_out = layer_norm_func(x, weight, bias, eps=1e-6, kernel=ref_kernel)
        dout = torch.randn_like(ref_out) * 0.05
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_dw, weight.grad = weight.grad.detach().clone(), None
        ref_db, bias.grad = bias.grad.detach().clone(), None
        # opt
        x = x.detach().clone().requires_grad_()
        weight = weight.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()
        opt_out = layer_norm_func(x, weight, bias, eps=1e-6, kernel=real_kernel)
        dout = dout.detach().clone()
        opt_out.backward(dout)
        opt_dx, x.grad = x.grad.detach().clone(), None
        opt_dw, weight.grad = weight.grad.detach().clone(), None
        opt_db, bias.grad = bias.grad.detach().clone(), None
        torch.testing.assert_close(ref_out, opt_out)
        torch.testing.assert_close(ref_dx, opt_dx)
        torch.testing.assert_close(ref_dw, opt_dw)
        torch.testing.assert_close(ref_db, opt_db)


if __name__ == "__main__":
    unittest.main()
