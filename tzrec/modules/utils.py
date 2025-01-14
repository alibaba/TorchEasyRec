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

import torch


def div_no_nan(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    rounding_mode=None,
    out=None,
) -> torch.Tensor:
    """Divides input by other and avoid division by zero.

    Args:
        input (Tensor): the dividend
        other (Tensor): the divisor
        rounding_mode (str, optional):
            Type of rounding applied to the result:
            - None: default behavior. Performs no rounding and, if both input and other
                are integer types, promotes the inputs to the default scalar type.
                Equivalent to true division in Python (the / operator) and NumPy’s
                np.true_divide.
            - "trunc": rounds the results of the division towards zero. Equivalent to
                C-style integer division.
            - "floor": rounds the results of the division down. Equivalent to floor
                division in Python (the // operator) and NumPy’s np.floor_divide.
        out (Tensor, optional): the output tensor.

    Return:
        out (Tensor): the output tensor.
    """
    # other = other.clamp_min(eps)
    return torch.nan_to_num(
        torch.div(input, other, rounding_mode=rounding_mode, out=out),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
