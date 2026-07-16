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

import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

from tzrec.utils import env_util


def _inplace_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Fill the tensor in place with a truncated normal via inverse CDF.

    Reimplements the torch<=2.11 torch.nn.init._no_grad_trunc_normal_,
    which allocates no tensor-sized temporaries. torch>=2.12 switched to
    rejection sampling (pytorch/pytorch#174997) whose boolean masks and
    resampling buffers are as large as the tensor itself. Like torch<=2.11,
    a uniform sample landing exactly on an interval endpoint clamps to a or
    b (~2**-24 probability per element in fp32, more in lower precision).

    Args:
        tensor (torch.Tensor): an n-dimensional tensor filled in place.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
        generator (torch.Generator, optional): the sampling generator.

    Returns:
        the input tensor.
    """

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1, generator=generator)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Drop-in replacement for nn.init.trunc_normal_ with an in-place fallback.

    nn.init.trunc_normal_ in torch>=2.12 materializes several temporaries
    as large as the tensor itself, which OOMs when torchrec initializes a
    very large embedding table (e.g. a 150M-row ZCH table) at sharding
    time. Set USE_INPLACE_TRUNC_NORMAL=1 to initialize with the in-place
    inverse-CDF implementation of torch<=2.11 instead, which allocates no
    temporaries.

    Args:
        tensor (torch.Tensor): an n-dimensional tensor filled in place.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
        generator (torch.Generator, optional): the sampling generator.

    Returns:
        the input tensor.
    """
    if env_util.use_inplace_trunc_normal():
        return _inplace_trunc_normal_(tensor, mean, std, a, b, generator=generator)
    return nn.init.trunc_normal_(tensor, mean, std, a, b, generator=generator)


def create_init_fn(init_fn_expr: str) -> Callable[..., torch.Tensor]:
    """Create an embedding init_fn from its config expression.

    Evaluates an init_fn expression such as "nn.init.uniform_,a=-0.01,b=0.01"
    into partial(nn.init.uniform_, a=-0.01, b=0.01). nn.init.trunc_normal_ is
    replaced by the env-switchable trunc_normal_ above so that very large
    embedding tables can be initialized without OOM at sharding time on
    torch>=2.12 by setting USE_INPLACE_TRUNC_NORMAL=1.

    Args:
        init_fn_expr (str): comma-separated init function expression.

    Returns:
        a callable that initializes a tensor in place.
    """
    init_fn: Callable[..., torch.Tensor] = eval(
        f"partial({init_fn_expr})", {"partial": partial, "nn": nn, "torch": torch}
    )
    if isinstance(init_fn, partial) and init_fn.func is nn.init.trunc_normal_:
        init_fn = partial(trunc_normal_, *init_fn.args, **init_fn.keywords)
    return init_fn
