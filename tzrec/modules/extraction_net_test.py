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
from parameterized import parameterized

from tzrec.modules.extraction_net import ExtractionNet
from tzrec.protos.module_pb2 import MLP, ExtractionNetwork
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.test_util import TestGraphType, create_test_module


class ExtractionNetTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
        ]
    )
    def test_extraction_net(self, graph_type, final_flag) -> None:
        extraction_networks_cfg = ExtractionNetwork(
            network_name="layer1",
            expert_num_per_task=3,
            share_num=4,
            task_expert_net=MLP(hidden_units=[12, 8, 4]),
            share_expert_net=MLP(hidden_units=[12, 8, 6, 4]),
        )
        extraction_networks_cfg = config_to_kwargs(extraction_networks_cfg)
        extraction = ExtractionNet(
            in_extraction_networks=[16, 15, 14],
            in_shared_expert=13,
            final_flag=final_flag,
            **extraction_networks_cfg,
        )
        extraction = create_test_module(extraction, graph_type)
        features = [torch.randn(4, 16), torch.randn(4, 15), torch.randn(4, 14)]
        shared_feature = torch.randn(4, 13)
        task_features, new_shared = extraction(features, shared_feature)
        self.assertEqual(len(task_features), 3)
        for task_feature in task_features:
            self.assertEqual(task_feature.size(), (4, 4))
        if final_flag:
            self.assertEqual(new_shared, None)
        else:
            self.assertEqual(new_shared.size(), (4, 4))


if __name__ == "__main__":
    unittest.main()
