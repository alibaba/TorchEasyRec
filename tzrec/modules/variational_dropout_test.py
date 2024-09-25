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
from collections import OrderedDict

import torch
from parameterized import parameterized

from tzrec.modules.variational_dropout import VariationalDropout
from tzrec.utils.test_util import TestGraphType, create_test_module


class VariationalDropoutTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_variational_dropout(self, graph_type) -> None:
        features_dim = OrderedDict({"user_id": 4, "item_id": 3, "play_time": 2})
        variational_dropout = VariationalDropout(features_dim, "deep")

        variational_dropout = create_test_module(variational_dropout, graph_type)
        variational_dropout.train()
        feature = torch.Tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91],
            ]
        )
        noisy_feature, feature_loss = variational_dropout(feature)
        self.assertEqual(noisy_feature.size(), (2, 9))

        variational_dropout.eval()
        noisy_feature, feature_loss = variational_dropout(feature)
        self.assertEqual(noisy_feature.size(), (2, 9))


if __name__ == "__main__":
    unittest.main()
