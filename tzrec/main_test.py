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

"""Tests for tzrec.main orchestration and the train-loop step counter."""

import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import torch
from google.protobuf import text_format
from parameterized import parameterized

from tzrec.datasets.utils import RecordBatchTensor
from tzrec.main import _train_and_evaluate, export, predict, predict_checkpoint
from tzrec.optim.ema import DenseEMA
from tzrec.protos.data_pb2 import DataConfig
from tzrec.protos.eval_pb2 import EvalConfig
from tzrec.protos.export_pb2 import ExportConfig, ExportFormat
from tzrec.protos.optimizer_pb2 import DenseOptimizer, EMAConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import predict_util
from tzrec.utils.test_util import parameterized_name_func


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
            dense_optimizer=DenseOptimizer(),
        )
        eval_config = EvalConfig()

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

    def test_train_eval_uses_ema_and_restores_training_parameters(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        dense_ema = DenseEMA({"parameter": parameter}, decay=0.5)
        dense_ema.update()
        parameter.data.fill_(4.0)

        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        optimizer = mock.Mock()
        optimizer.params = {}
        optimizer.param_groups = []
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([object()])
        eval_dataloader = mock.Mock()
        batch = SimpleNamespace(checkpoint_info=None, data_timestamp=-1.0)
        pipeline = mock.Mock()
        pipeline.progress.return_value = (
            {"loss": torch.tensor(1.0)},
            {},
            batch,
        )
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.side_effect = [True, False, False]
        exporter = mock.Mock()
        train_config = SimpleNamespace(
            num_steps=1,
            num_epochs=0,
            save_checkpoints_steps=1,
            save_checkpoints_epochs=0,
            save_checkpoints_timestamp_interval=0,
            save_checkpoints_timestamps=[],
            save_checkpoints_timestamp_quorum=0,
            use_tensorboard=False,
            tensorboard_summaries=[],
            is_profiling=False,
            log_step_count_steps=10,
            dense_optimizer=DenseOptimizer(ema=EMAConfig()),
        )

        def assert_ema(*args, **kwargs):
            torch.testing.assert_close(parameter, torch.tensor([2.0]))

        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main._evaluate", side_effect=assert_ema),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                lr_scheduler=[],
                model_dir="unused",
                train_config=train_config,
                eval_config=EvalConfig(),
                ckpt_manager=ckpt_manager,
                dense_ema=dense_ema,
                export_config=ExportConfig(use_dense_ema=False),
            )

        torch.testing.assert_close(parameter, torch.tensor([4.0]))
        self.assertTrue(model.module.model.on_train_end.called)
        for call in exporter.maybe_export.call_args_list:
            self.assertIsNone(call.kwargs["dense_ema"])

    def test_hf_export_rejects_dense_ema(self) -> None:
        # dcp_to_hf reads <ckpt>/model unconditionally, so it would silently
        # ship raw weights where TorchScript export ships the EMA ones.
        with tempfile.TemporaryDirectory() as test_dir:
            config = EasyRecConfig()
            config.train_input_path = "unused"
            config.eval_input_path = "unused"
            config.model_dir = os.path.join(test_dir, "train")
            os.makedirs(config.model_dir)
            config.train_config.dense_optimizer.ema.CopyFrom(EMAConfig())
            config.export_config.export_format = ExportFormat.HF
            config_path = os.path.join(test_dir, "pipeline.config")
            with open(config_path, "w") as f:
                f.write(text_format.MessageToString(config))
            with self.assertRaisesRegex(ValueError, "Dense EMA"):
                export(config_path, os.path.join(test_dir, "export"))


class PredictionLifecycleTest(unittest.TestCase):
    """Tests for prediction lifecycle wiring."""

    def _batch(self, rows: int) -> mock.Mock:
        batch = mock.Mock()
        batch.reserves = RecordBatchTensor(pa.record_batch({"id": list(range(rows))}))
        batch.to_dict.return_value = {}
        batch.dummy = False
        return batch

    def _pipeline_config(self) -> EasyRecConfig:
        return EasyRecConfig(
            train_input_path="",
            eval_input_path="",
            model_dir="model",
            data_config=DataConfig(dataset_type=3, num_workers=1),
        )

    def _run_predict(
        self,
        dataloader: mock.Mock,
        writer: mock.Mock,
        model: mock.Mock,
        predict_threads: int = 1,
    ) -> None:
        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch(
                "tzrec.main.init_process_group",
                return_value=(torch.device("cpu"), "gloo"),
            ),
            mock.patch("tzrec.main.url_to_fs", return_value=(None, "model")),
            mock.patch(
                "tzrec.main.config_util.load_pipeline_config",
                return_value=self._pipeline_config(),
            ),
            mock.patch("tzrec.main.acc_utils.allow_tf32_for_export"),
            mock.patch("tzrec.main.acc_utils.is_trt_predict", return_value=False),
            mock.patch("tzrec.main.acc_utils.is_aot_predict", return_value=False),
            mock.patch(
                "tzrec.main.acc_utils.is_input_tile_predict", return_value=False
            ),
            mock.patch("tzrec.main._create_features", return_value=[]),
            mock.patch("tzrec.main.create_dataloader", return_value=dataloader),
            mock.patch("tzrec.main.create_writer", return_value=writer),
            mock.patch("tzrec.main.torch.jit.load", return_value=model),
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01),
        ):
            predict(
                "input",
                "output",
                "model",
                reserved_columns="id",
                writer_type="MockWriter",
                predict_threads=predict_threads,
            )

    def _run_predict_checkpoint(self, pipeline: mock.Mock, writer: mock.Mock) -> None:
        dataloader = mock.Mock()
        dataloader.dataset.sampled_batch_size = 2
        dataloader.get_iterator.return_value = iter([])
        model = mock.Mock()
        wrapped_model = mock.Mock()
        distributed_model = mock.Mock()
        distributed_model.device = torch.device("cpu")
        ckpt_manager = mock.Mock()
        planner = mock.Mock()
        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch(
                "tzrec.main.config_util.load_pipeline_config",
                return_value=self._pipeline_config(),
            ),
            mock.patch(
                "tzrec.main.init_process_group",
                return_value=(torch.device("cpu"), "gloo"),
            ),
            mock.patch("tzrec.main.acc_utils.allow_tf32"),
            mock.patch("tzrec.main._create_features", return_value=[]),
            mock.patch("tzrec.main.create_dataloader", return_value=dataloader),
            mock.patch("tzrec.main.create_writer", return_value=writer),
            mock.patch("tzrec.main._create_model", return_value=model),
            mock.patch("tzrec.main.PredictWrapper", return_value=wrapped_model),
            mock.patch(
                "tzrec.main.checkpoint_util.CheckpointManager",
                return_value=ckpt_manager,
            ),
            mock.patch("tzrec.main.create_planner", return_value=planner),
            mock.patch("tzrec.main.get_default_sharders", return_value=[]),
            mock.patch(
                "tzrec.main.DistributedModelParallel",
                return_value=distributed_model,
            ),
            mock.patch("tzrec.main.config_util.use_dense_ema", return_value=False),
            mock.patch("tzrec.main.PredictPipelineSparseDist", return_value=pipeline),
            mock.patch.object(predict_util, "_PREDICT_PIPELINE_POLL_INTERVAL", 0.01),
        ):
            predict_checkpoint(
                "pipeline.config",
                "input",
                "output",
                checkpoint_path="checkpoint",
                reserved_columns="id",
                writer_type="MockWriter",
                predict_steps=2,
            )

    @parameterized.expand([[1], [2]], name_func=parameterized_name_func)
    def test_predict_counts_first_write_and_commits_once(
        self, predict_threads: int
    ) -> None:
        batch_count = predict_threads + 2
        dataloader = mock.Mock()
        dataloader.get_iterator.return_value = iter(
            [self._batch(2) for _ in range(batch_count)]
        )
        writer = mock.Mock()
        model = mock.Mock(return_value={"score": torch.tensor([0.1, 0.2])})

        self._run_predict(dataloader, writer, model, predict_threads=predict_threads)

        self.assertEqual(writer.write.call_count, batch_count)
        writer.close.assert_called_once_with()

    def test_predict_background_failures_do_not_commit(self) -> None:
        for stage in ("forward", "writer"):
            with self.subTest(stage=stage):
                dataloader = mock.Mock()
                dataloader.get_iterator.return_value = iter(
                    [self._batch(2), self._batch(2)]
                )
                writer = mock.Mock()
                predictions = {"score": torch.tensor([0.1, 0.2])}
                model = mock.Mock(
                    side_effect=[predictions, predictions]
                    if stage == "writer"
                    else [predictions, RuntimeError("forward boom")]
                )
                if stage == "writer":
                    writer.write.side_effect = [None, RuntimeError("writer boom")]

                with self.assertRaises(
                    predict_util.PredictPipelineStageError
                ) as context:
                    self._run_predict(dataloader, writer, model)

                self.assertEqual(context.exception.failure.stage, stage)
                self.assertEqual(context.exception.failure.message, f"{stage} boom")
                writer.close.assert_not_called()

    def test_predict_checkpoint_commits_only_after_writer_success(self) -> None:
        for writer_fails in (False, True):
            with self.subTest(writer_fails=writer_fails):
                writer = mock.Mock()
                if writer_fails:
                    writer.write.side_effect = [
                        None,
                        RuntimeError("checkpoint writer boom"),
                    ]
                pipeline = mock.Mock()
                pipeline.progress.side_effect = [
                    ({"score": torch.tensor([0.1, 0.2])}, self._batch(2)),
                    ({"score": torch.tensor([0.1, 0.2])}, self._batch(2)),
                ]

                if writer_fails:
                    with self.assertRaises(
                        predict_util.PredictPipelineStageError
                    ) as context:
                        self._run_predict_checkpoint(pipeline, writer)
                    self.assertEqual(context.exception.failure.stage, "writer")
                    self.assertEqual(
                        context.exception.failure.message,
                        "checkpoint writer boom",
                    )
                    writer.close.assert_not_called()
                else:
                    self._run_predict_checkpoint(pipeline, writer)
                    self.assertEqual(writer.write.call_count, 2)
                    writer.close.assert_called_once_with()


class TrainStepCounterMultiPassTest(unittest.TestCase):
    """Guard the step-based (``num_steps``) multi-pass step counter.

    Mirrors ``tzrec/main.py``: with ``use_step`` set, ``step_iter = iter(
    range(num_steps))`` and the data-pass ``StopIteration`` handler does
    ``step_iter = itertools.chain([i_step], step_iter); i_step -= 1``.
    ``step_iter`` must be a *one-shot* iterator so the next data pass resumes
    at ``i_step`` (retried) then ``i_step + 1``; a bare ``range`` is
    re-iterable and would yield ``[i_step, 0, 1, ...]``, resetting the counter
    so ``model.ckpt-{step}`` collides across passes and ``DynamicEmbDump``
    then refuses to overwrite the existing ``dynamicemb`` dir.
    """

    def test_use_step_counter_monotonic_and_terminates_across_passes(self):
        num_steps = 10
        # Mirrors tzrec/main.py use_step branch (post-fix): a one-shot iterator.
        step_iter = iter(range(num_steps))

        # Dataloader raises StopIteration at this global step on pass 1.
        exhaust_steps = iter([6])
        next_exhaust = next(exhaust_steps, None)

        trained = []
        # For use_step, epoch_iter is infinite (itertools.count(0, 0)); the loop
        # exits only via the num_steps termination guard below.
        while True:
            for i_step in step_iter:
                if next_exhaust is not None and i_step == next_exhaust:
                    # StopIteration handler (tzrec/main.py): retry this step on
                    # the next pass; do not let the chain re-iterate from 0.
                    step_iter = itertools.chain([i_step], step_iter)
                    i_step -= 1
                    break
                trained.append(i_step)
            # Mirrors "if use_step and i_step >= num_steps - 1: break".
            if i_step >= num_steps - 1:
                break
            next_exhaust = next(exhaust_steps, None)

        # Exactly num_steps steps trained, strictly monotonic, despite the
        # mid-run data exhaustion and retry at step 6: no reset to 0 -> no
        # model.ckpt-{step} collision across passes.
        self.assertEqual(trained, list(range(num_steps)))


if __name__ == "__main__":
    unittest.main()
