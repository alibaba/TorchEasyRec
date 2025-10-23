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

import re
from functools import partial
from typing import Any, Dict, List, Tuple, Type, Union

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import WeightDecayMode
from torch import nn
from torch.optim.optimizer import Optimizer
from torchrec.optim import optimizers, rowwise_adagrad
from torchrec.optim.keyed import KeyedOptimizerWrapper

from tzrec.optim.lr_scheduler import BaseLR
from tzrec.protos import optimizer_pb2
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.logging_util import logger


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
    if "weight_decay_mode" in optimizer_kwargs:
        optimizer_kwargs["weight_decay_mode"] = WeightDecayMode[
            optimizer_kwargs["weight_decay_mode"]
        ]

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
    optimizer_config: Union[optimizer_pb2.DenseOptimizer, optimizer_pb2.ParamOptimizer],
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
    elif optimizer_type == "adamw_optimizer":
        beta1 = optimizer_kwargs.pop("beta1")
        beta2 = optimizer_kwargs.pop("beta2")
        optimizer_kwargs["betas"] = (beta1, beta2)
        return torch.optim.AdamW, optimizer_kwargs
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_param_optimizer(
    optimizer_config: optimizer_pb2.DenseOptimizer,
) -> Tuple[List[Type[Optimizer]], List[Dict[str, Any]], List[str]]:
    """Create optimizer class for some param."""
    optimizers = []
    optimizer_kwargs = []
    optimizer_regex_patterns = []
    for param_optimizer in optimizer_config.param_optimizers:
        optim, kwargs = create_dense_optimizer(param_optimizer)
        optimizers.append(optim)
        optimizer_kwargs.append(kwargs)
        optimizer_regex_patterns.append(param_optimizer.regex_pattern)
    return optimizers, optimizer_kwargs, optimizer_regex_patterns


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


def create_param_schedulers(
    optimizers: List[Optimizer],
    optimizer_config: optimizer_pb2.DenseOptimizer,
    optim_indexs: List[int],
) -> List[BaseLR]:
    """Create optimizer for dense module.

    Args:
        optimizers (list[Optimizer]): an list instance of Optimizer.
        optimizer_config (optimizer_pb2.SparseOptimizer|optimizer_pb2.DenseOptimizer):
            an instance of Optimizer config.
        optim_indexs (list[int]): optimizers index in optimizer_config.param_optimizers

    Returns:
        lr (list(BaseLR)): a list lr scheduler.
    """
    lr_type = optimizer_config.WhichOneof("learning_rate")
    oneof_lr_config = getattr(optimizer_config, lr_type)
    lr_cls_name = oneof_lr_config.__class__.__name__
    lr_kwargs = config_to_kwargs(oneof_lr_config)

    param_lrs = []
    param_optim_configs = list(optimizer_config.param_optimizers)
    for optimizer, index in zip(optimizers, optim_indexs):
        param_optim_config = param_optim_configs[index]
        if param_optim_config.HasField("param_learning_rate"):
            param_lr_config = param_optim_config.param_learning_rate
            param_lr_type = param_lr_config.WhichOneof("learning_rate")
            param_oneof_lr_config = getattr(param_lr_config, param_lr_type)
            param_lr_cls_name = param_oneof_lr_config.__class__.__name__
            param_lr_kwargs = config_to_kwargs(param_oneof_lr_config)
        else:
            param_lr_cls_name = lr_cls_name
            param_lr_kwargs = lr_kwargs
        param_lr_kwargs["optimizer"] = optimizer
        param_lrs.append(BaseLR.create_class(param_lr_cls_name)(**param_lr_kwargs))
    return param_lrs


def group_param_by_regex_pattern(
    params: Dict[str, nn.Parameter], regex_patterns: list[str]
) -> Tuple[Dict[str, nn.Parameter], List[Dict[str, nn.Parameter]]]:
    """Group params by regex."""
    remaining_params = dict()
    param_optim_params = [dict() for _ in range(len(regex_patterns))]
    for name, param in params.items():
        logger.info("parameter name: {}".format(name))
        for i, regex_pattern in enumerate(regex_patterns):
            if re.fullmatch(re.compile(regex_pattern), name):
                param_optim_params[i][name] = param
                break
        else:
            remaining_params[name] = param
    return remaining_params, param_optim_params


def build_param_optimizers(
    param_optim_cls: List[Type[Optimizer]],
    param_optim_kwargs: List[Dict[str, any]],
    param_optim_params: List[Dict[str, nn.Parameter]],
) -> Tuple[List[KeyedOptimizerWrapper], List[int]]:
    """Build param optimizer."""
    param_optimizers = []
    valid_index = []
    for i in range(len(param_optim_kwargs)):
        if len(param_optim_params[i]) > 0:
            valid_index.append(i)
            param = param_optim_params[i]
            optim_cls = param_optim_cls[i]
            kwargs = param_optim_kwargs[i]

            optimizer = KeyedOptimizerWrapper(
                param,
                partial(
                    lambda params, optim_cls, kwargs: optim_cls(params, **kwargs),
                    optim_cls=optim_cls,
                    kwargs=kwargs,
                ),
            )
            param_optimizers.append(optimizer)
            logger.info(
                "param optimizer index: {}, params key:{}".format(i, list(param.keys()))
            )
    return param_optimizers, valid_index
