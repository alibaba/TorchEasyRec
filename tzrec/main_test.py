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

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from tzrec.main import _train_and_evaluate


class MainTest(unittest.TestCase):
    """Tests for tzrec.main orchestration."""

    def test_train_and_evaluate_closes_exporter_and_ckpt_on_exception(self) -> None:
        """A training exception must still drain the exporter and ckpt manager.

        Before wrapping the training body in try/finally, a raise inside the
        loop skipped the happy-path close() calls; the daemon export worker
        outlived the function (keeping the manager reachable so its finalizer
        never fired), leaked the protected checkpoint, and could publish late.
        """
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.return_value = False
        exporter = mock.Mock()
        model = mock.Mock()
        optimizer = mock.Mock()
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([object()])

        pipeline = mock.Mock()
        pipeline.progress.side_effect = RuntimeError("boom")

        train_config = SimpleNamespace(
            num_steps=1,
            num_epochs=0,
            save_checkpoints_steps=0,
            save_checkpoints_epochs=0,
            save_checkpoints_timestamp_interval=0,
            save_checkpoints_timestamps=[],
            save_checkpoints_timestamp_quorum=0,
            use_tensorboard=False,
            tensorboard_summaries=[],
            is_profiling=False,
            log_step_count_steps=1,
        )
        eval_config = SimpleNamespace()

        with tempfile.TemporaryDirectory() as model_dir:
            with (
                mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
                mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
                mock.patch(
                    "tzrec.main.OnlineDenseExportManager", return_value=exporter
                ),
            ):
                with self.assertRaises(RuntimeError):
                    _train_and_evaluate(
                        model=model,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader,
                        eval_dataloader=None,
                        lr_scheduler=[],
                        model_dir=model_dir,
                        train_config=train_config,
                        eval_config=eval_config,
                        ckpt_manager=ckpt_manager,
                    )
        self.assertTrue(exporter.close.called)
        self.assertTrue(ckpt_manager.close.called)


if __name__ == "__main__":
    unittest.main()
