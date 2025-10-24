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
from torch import Tensor, nn
from torch.optim import Optimizer

from tzrec.optim import optimizer_builder
from tzrec.protos import optimizer_pb2


class OpimizerBuilderTest(unittest.TestCase):
    def test_create_part_optimizer(self):
        pattern1 = "model.dbmtl.task(.*)"
        pattern2 = "model.dbmtl.mmoe(.*)"
        config = optimizer_pb2.DenseOptimizer(
            adam_optimizer=optimizer_pb2.AdamOptimizer(lr=0.001),
            exponential_decay_learning_rate=optimizer_pb2.ExponentialDecayLR(
                decay_size=500
            ),
            part_optimizers=[
                optimizer_pb2.PartOptimizer(
                    adam_optimizer=optimizer_pb2.AdamOptimizer(lr=0.01),
                    regex_pattern=pattern1,
                    exponential_decay_learning_rate=optimizer_pb2.ExponentialDecayLR(
                        decay_size=1000
                    ),
                ),
                optimizer_pb2.PartOptimizer(
                    sgd_optimizer=optimizer_pb2.SGDOptimizer(lr=0.01),
                    regex_pattern=pattern2,
                    constant_learning_rate=optimizer_pb2.ConstantLR(),
                ),
            ],
        )
        clss, kwargs, regex_patterns = optimizer_builder.create_part_optimizer(config)
        self.assertEqual(clss, [torch.optim.Adam, torch.optim.SGD])
        self.assertEqual(regex_patterns, [pattern1, pattern2])

    def test_group_param_by_regex_pattern(self):
        params = {
            "dbmtl.task_tower": nn.Parameter(Tensor([1.0])),
            "dbmtl.task_mlp": nn.Parameter(Tensor([2.0])),
            "dbmtl.mmoe.expert": nn.Parameter(Tensor([2.5])),
            "dbmtl.mmoe.gate": nn.Parameter(Tensor([3.0])),
            "dbmtl.mask_net.mlp": nn.Parameter(Tensor([3.5])),
            "dbmtl.bottom_mlp": nn.Parameter(Tensor([4.0])),
        }
        patterns = ["dbmtl.task(.*)", "dbmtl.mmoe(.*)", "dbmtl.bottom_mlp"]
        remaining_params, part_optim_params = (
            optimizer_builder.group_param_by_regex_pattern(params, patterns)
        )
        self.assertEqual(sorted(remaining_params.keys()), ["dbmtl.mask_net.mlp"])
        self.assertEqual(
            sorted(part_optim_params[0].keys()), ["dbmtl.task_mlp", "dbmtl.task_tower"]
        )
        self.assertEqual(
            sorted(part_optim_params[1].keys()),
            ["dbmtl.mmoe.expert", "dbmtl.mmoe.gate"],
        )
        self.assertEqual(sorted(part_optim_params[2].keys()), ["dbmtl.bottom_mlp"])

    def test_create_part_optim_schedulers(self):
        pattern1 = "model.dbmtl.task(.*)"
        pattern2 = "model.dbmtl.mmoe(.*)"
        pattern3 = "model.dbmtl.bottom(.*)"
        config = optimizer_pb2.DenseOptimizer(
            adam_optimizer=optimizer_pb2.AdamOptimizer(lr=0.001),
            manual_step_learning_rate=optimizer_pb2.ManualStepLR(
                learning_rates=[0.001], schedule_sizes=[10]
            ),
            part_optimizers=[
                optimizer_pb2.PartOptimizer(
                    adam_optimizer=optimizer_pb2.AdamOptimizer(lr=0.01),
                    regex_pattern=pattern1,
                    constant_learning_rate=optimizer_pb2.ConstantLR(),
                ),
                optimizer_pb2.PartOptimizer(
                    sgd_optimizer=optimizer_pb2.SGDOptimizer(lr=0.01),
                    regex_pattern=pattern2,
                    exponential_decay_learning_rate=optimizer_pb2.ExponentialDecayLR(
                        decay_size=1000
                    ),
                ),
                optimizer_pb2.PartOptimizer(
                    adamw_optimizer=optimizer_pb2.AdamWOptimizer(lr=0.01),
                    regex_pattern=pattern3,
                ),
            ],
        )
        params = [Tensor([1.0])]
        kwarg = {"lr": 0.01}
        optimizers = [Optimizer(params, kwarg), Optimizer(params, kwarg)]
        optim_indexs = [0, 2]
        schedulers = optimizer_builder.create_part_optim_schedulers(
            optimizers, config, optim_indexs
        )
        scheduler_class = [x.__class__.__name__ for x in schedulers]
        self.assertEqual(scheduler_class, ["ConstantLR", "ManualStepLR"])

    def test_build_part_optimizers(self):
        param_optim_cls = [torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW]
        param_optim_kwargs = [
            {"lr": 0.01, "betas": (0.9, 0.999)},
            {"lr": 0.01, "momentum": 0.9},
            {"lr": 0.01, "betas": (0.9, 0.999)},
        ]
        optim_params = [
            {"tower": nn.Parameter(Tensor([1.0]))},
            {},
            {"mlp": nn.Parameter(Tensor([2.0]))},
        ]
        optimizers, indices = optimizer_builder.build_part_optimizers(
            param_optim_cls, param_optim_kwargs, optim_params
        )
        opt_cls = [opt._optimizer.__class__.__name__ for opt in optimizers]
        self.assertEqual(opt_cls, ["Adam", "AdamW"])


if __name__ == "__main__":
    unittest.main()
