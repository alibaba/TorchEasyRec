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

from typing import Any, Dict, Tuple, Type, Union

import torch
from torch.optim.optimizer import Optimizer
from torchrec.optim import optimizers, rowwise_adagrad

from tzrec.optim.lr_scheduler import BaseLR
from tzrec.protos import optimizer_pb2
from tzrec.utils.config_util import config_to_kwargs


def create_sparse_optimizer(
    optimizer_config: optimizer_pb2.SparseOptimizer,
) -> Tuple[Type[Optimizer], Dict[str, Any]]:
    """Create optimizer for embedding module.

    Args:
        optimizer_config (optimizer_pb2.SparseOptimizer): an instance of
            SparseOptimizer config.

    Returns:
        optimizer (Optimizer): an instance of Optimizer.
        optimizer_kwargs (dict): optimizer params.
    """
    optimizer_type = optimizer_config.WhichOneof("optimizer")
    oneof_optim_config = getattr(optimizer_config, optimizer_type)
    optimizer_kwargs = config_to_kwargs(oneof_optim_config)

    if optimizer_type == "sgd_optimizer":
        return optimizers.SGD, optimizer_kwargs
    elif optimizer_type == "adagrad_optimizer":
        return optimizers.Adagrad, optimizer_kwargs
    elif optimizer_type == "adam_optimizer":
        return optimizers.Adam, optimizer_kwargs
    elif optimizer_type == "lars_sgd_optimizer":
        return optimizers.LarsSGD, optimizer_kwargs
    elif optimizer_type == "lamb_optimizer":
        return optimizers.LAMB, optimizer_kwargs
    elif optimizer_type == "partial_rowwise_lamb_optimizer":
        return optimizers.PartialRowWiseLAMB, optimizer_kwargs
    elif optimizer_type == "partial_rowwise_adam_optimizer":
        return optimizers.PartialRowWiseAdam, optimizer_kwargs
    elif optimizer_type == "rowwise_adagrad_optimizer":
        return rowwise_adagrad.RowWiseAdagrad, optimizer_kwargs
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_dense_optimizer(
    optimizer_config: optimizer_pb2.DenseOptimizer,
) -> Tuple[Type[Optimizer], Dict[str, Any]]:
    """Create optimizer for dense module.

    Args:
        optimizer_config (optimizer_pb2.DenseOptimizer): an instance of
            DenseOptimizer config.

    Returns:
        optimizer (Optimizer): an instance of Optimizer.
        optimizer_kwargs (dict): optimizer params.
    """
    optimizer_type = optimizer_config.WhichOneof("optimizer")
    oneof_optim_config = getattr(optimizer_config, optimizer_type)
    optimizer_kwargs = config_to_kwargs(oneof_optim_config)

    if optimizer_type == "sgd_optimizer":
        return torch.optim.SGD, optimizer_kwargs
    elif optimizer_type == "adagrad_optimizer":
        return torch.optim.Adagrad, optimizer_kwargs
    elif optimizer_type == "adam_optimizer":
        beta1 = optimizer_kwargs.pop("beta1")
        beta2 = optimizer_kwargs.pop("beta2")
        optimizer_kwargs["betas"] = (beta1, beta2)
        return torch.optim.Adam, optimizer_kwargs
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    optimizer_config: Union[
        optimizer_pb2.SparseOptimizer, optimizer_pb2.DenseOptimizer
    ],
) -> BaseLR:
    """Create optimizer for dense module.

    Args:
        optimizer (Optimizer): an instance of Optimizer.
        optimizer_config (optimizer_pb2.SparseOptimizer|optimizer_pb2.DenseOptimizer):
            an instance of Optimizer config.

    Returns:
        lr (BaseLR): a lr scheduler.
    """
    lr_type = optimizer_config.WhichOneof("learning_rate")
    oneof_lr_config = getattr(optimizer_config, lr_type)
    lr_cls_name = oneof_lr_config.__class__.__name__
    lr_kwargs = config_to_kwargs(oneof_lr_config)
    lr_kwargs["optimizer"] = optimizer
    # pyre-ignore [16]
    return BaseLR.create_class(lr_cls_name)(**lr_kwargs)
