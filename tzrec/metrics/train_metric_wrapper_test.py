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
import torchmetrics

from tzrec.metrics.decay_auc import DecayAUC
from tzrec.metrics.train_metric_wrapper import TrainMetricWrapper


class TrainMetricWrapperTest(unittest.TestCase):
    def test_module_is_decay_auc(self):
        metric = DecayAUC(thresholds=10)
        metric_wrapper = TrainMetricWrapper(metric, decay_rate=0.9, decay_step=1)
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        target = torch.tensor([1, 0, 1, 0, 0, 0, 1, 1])
        target2 = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        metric_wrapper.update(preds, target)
        value = metric_wrapper.compute()
        torch.testing.assert_close(value, torch.tensor(0.5625))
        metric_wrapper.update(preds, target)
        value = metric_wrapper.compute()
        torch.testing.assert_close(value, torch.tensor(0.5625))
        metric_wrapper.update(preds, target2)
        value = metric_wrapper.compute()
        torch.testing.assert_close(value, torch.tensor(0.5437), rtol=1e-4, atol=1e-4)

    def test_module_is_mean_absolute_error(self):
        metric = torchmetrics.MeanAbsoluteError()
        metric_wrapper = TrainMetricWrapper(metric, decay_rate=0.9, decay_step=1)
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        target = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        metric_wrapper.update(preds, target)
        value = metric_wrapper.compute()
        torch.testing.assert_close(value, torch.tensor(0.1))
        target = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        metric_wrapper.update(preds, target)
        value = metric_wrapper.compute()
        torch.testing.assert_close(value, torch.tensor(0.11))

    def test_state_restore_and_reset(self):
        metric_wrapper = TrainMetricWrapper(
            torchmetrics.MeanAbsoluteError(), decay_rate=0.9, decay_step=1
        )
        preds = torch.tensor([0.1, 0.2])
        metric_wrapper.update(preds, torch.tensor([0.6, 0.7]))
        torch.testing.assert_close(metric_wrapper.compute(), torch.tensor(0.5))
        self.assertEqual(
            list(metric_wrapper.state_dict().keys()), ["_value", "_step_cnt"]
        )
        self.assertEqual(list(metric_wrapper.parameters()), [])

        # continue train restores the running metric of the interrupted job.
        restored = TrainMetricWrapper(
            torchmetrics.MeanAbsoluteError(), decay_rate=0.9, decay_step=1
        )
        restored.load_state_dict(metric_wrapper.state_dict())
        torch.testing.assert_close(restored.compute(), torch.tensor(0.5))
        restored.update(preds, torch.tensor([0.2, 0.3]))
        torch.testing.assert_close(restored.compute(), torch.tensor(0.46))

        # fine tune resets it, so the next value is not blended with the old one.
        restored.reset()
        torch.testing.assert_close(restored.compute(), torch.tensor(0.0))
        restored.update(preds, torch.tensor([0.2, 0.3]))
        torch.testing.assert_close(restored.compute(), torch.tensor(0.1))


if __name__ == "__main__":
    unittest.main()
