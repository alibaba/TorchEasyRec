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

import copy
import unittest

import torch
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from tzrec.modules.norm import (
    LayerNorm,
    SwishLayerNorm,
)
from tzrec.ops import Kernel
from tzrec.utils.test_util import gpu_unavailable


class LayerNormTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=32, max_value=10000),
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
    def test_modules(self, *args, **kwargs) -> None:
        self._test_layer_norm_module(
            *args,
            **kwargs,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.TRITON,
        )

    def _test_layer_norm_module(
        self,
        N: int,
        D: int,
        is_swish: bool,
        dtype: torch.dtype,
        ref_kernel: Kernel,
        real_kernel: Kernel,
        skip_comparisons: bool = False,
    ) -> None:
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .normal_(0.0, 1.0)
            .requires_grad_()
        )
        # ref
        if is_swish:
            ref_layer = SwishLayerNorm(
                dim=D,
                eps=1e-6,
            ).to(device="cuda")
            ref_layer._kernel = ref_kernel
        else:
            ref_layer = LayerNorm(
                dim=D,
                eps=1e-6,
            ).to(device="cuda")
            ref_layer._kernel = ref_kernel
        opt_layer = copy.deepcopy(ref_layer)
        opt_layer._kernel = real_kernel

        ref_out = ref_layer(x)
        dout = torch.randn_like(ref_out) * 0.05
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_dw = ref_layer._weight.grad.detach().clone()
        ref_db = ref_layer._bias.grad.detach().clone()
        # opt
        x = x.detach().clone().requires_grad_()
        opt_out = opt_layer(x)
        dout = dout.detach().clone()
        opt_out.backward(dout)
        opt_dx, x.grad = x.grad.detach().clone(), None
        opt_dw = opt_layer._weight.grad.detach().clone()
        opt_db = opt_layer._bias.grad.detach().clone()
        torch.testing.assert_close(ref_out, opt_out)
        torch.testing.assert_close(
            ref_dx,
            opt_dx,
        )
        torch.testing.assert_close(
            ref_dw,
            opt_dw,
        )
        torch.testing.assert_close(
            ref_db,
            opt_db,
        )


if __name__ == "__main__":
    unittest.main()
