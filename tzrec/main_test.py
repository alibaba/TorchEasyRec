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
import runpy
import shutil
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pyarrow as pa
import torch
from parameterized import param, parameterized

from tzrec.datasets.utils import Batch, RecordBatchTensor
from tzrec.main import (
    _create_model,
    _create_train_profiler,
    _train_and_evaluate,
    evaluate,
    predict,
    predict_checkpoint,
    train_and_evaluate,
)
from tzrec.models.model import BaseModel
from tzrec.optim.ema import DenseEMA
from tzrec.optim.lr_scheduler import ExponentialDecayLR, LinearDecayLR
from tzrec.protos.data_pb2 import DataConfig, DatasetType
from tzrec.protos.eval_pb2 import EvalConfig
from tzrec.protos.export_pb2 import ExportConfig
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.module_pb2 import MLP
from tzrec.protos.optimizer_pb2 import DenseOptimizer, EMAConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.train_pb2 import (
    DeltaEmbeddingDumpConfig,
    ProfilerConfig,
    TrainConfig,
)
from tzrec.utils import checkpoint_util, predict_util
from tzrec.utils.delta_embedding_dump import DumpDecision
from tzrec.utils.online_learning_util import OnlineLearningHandover
from tzrec.utils.runtime import ShardedModelRuntime
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


def _train_config(**overrides) -> TrainConfig:
    """A TrainConfig for driving ``_train_and_evaluate`` against mocks."""
    fields = dict(
        num_steps=1,
        save_checkpoints_steps=0,
        use_tensorboard=False,
        dense_optimizer=DenseOptimizer(),
    )
    fields.update(overrides)
    return TrainConfig(**fields)


class _FakeSummaryWriter:
    """SummaryWriter stand-in: a context manager that closes itself on exit."""

    def __init__(self) -> None:
        self.flush = mock.Mock()
        self.close = mock.Mock()

    def __enter__(self) -> "_FakeSummaryWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class MainTest(unittest.TestCase):
    """Tests for tzrec.main orchestration."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir)

    @parameterized.expand([(False,), (True,)], name_func=parameterized_name_func)
    def test_train_cli_ignore_restore_lr_scheduler(self, ignore):
        argv = ["tzrec.train_eval"] + (
            ["--ignore_restore_lr_scheduler"] if ignore else []
        )
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch("tzrec.main.train_and_evaluate", autospec=True) as train,
        ):
            runpy.run_module("tzrec.train_eval", run_name="__main__")
        train.assert_called_once()
        self.assertIs(train.call_args.kwargs["ignore_restore_lr_scheduler"], ignore)

    def test_export_cli_edit_config_json(self):
        edit = '{"export_config.dynamic_embedding.feature_store_config.version":"v1"}'
        argv = ["tzrec.export", "--edit_config_json", edit]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch("tzrec.main.export", autospec=True) as export,
        ):
            runpy.run_module("tzrec.export", run_name="__main__")
        export.assert_called_once()
        self.assertEqual(export.call_args.kwargs["edit_config_json"], edit)

    def test_create_custom_model(self) -> None:
        """A custom model receives its unpacked protobuf configuration."""
        model_config = ModelConfig()
        model_config.custom_model.class_path = "tzrec.models.model.BaseModel"
        model_config.custom_model.config.Pack(MLP(hidden_units=[32, 16]))

        model = _create_model(model_config, [], [])

        self.assertIsInstance(model, BaseModel)
        self.assertEqual(list(model._model_config.hidden_units), [32, 16])

    def test_create_custom_model_without_config(self) -> None:
        """A custom model may omit its protobuf configuration."""
        model_config = ModelConfig()
        model_config.custom_model.class_path = "tzrec.models.model.BaseModel"

        model = _create_model(model_config, [], [])

        self.assertIsNone(model._model_config)

    def test_create_custom_model_requires_base_model(self) -> None:
        """Reject custom classes that do not implement the model contract."""
        model_config = ModelConfig()
        model_config.custom_model.class_path = "torch.nn.Linear"
        model_config.custom_model.config.Pack(MLP(hidden_units=[32]))

        with self.assertRaisesRegex(ValueError, "must inherit BaseModel"):
            _create_model(model_config, [], [])

    def test_train_and_evaluate_closes_exporter_not_ckpt_on_exception(self) -> None:
        """A training exception releases loop-owned resources only.

        The exporter is released by the loop's trailing finally; the
        CheckpointManager is borrowed from the runtime the training entry
        opened, which owns its lifetime, so the loop must not close it.
        The collective finalize_publish must not run on an exception path.
        """
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.return_value = False
        exporter = mock.Mock()
        exporter.enabled = False
        model = mock.Mock()
        optimizer = mock.Mock()
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([object()])

        pipeline = mock.Mock()
        pipeline.progress.side_effect = RuntimeError("boom")

        train_config = _train_config(log_step_count_steps=1)
        eval_config = EvalConfig()

        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            with self.assertRaises(RuntimeError):
                _train_and_evaluate(
                    model=model,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    eval_dataloader=None,
                    lr_scheduler=[],
                    model_dir=self.test_dir,
                    train_config=train_config,
                    eval_config=eval_config,
                    runtime=ShardedModelRuntime(ckpt_manager, None),
                )
        exporter.close.assert_called_once()
        ckpt_manager.close.assert_not_called()
        # a collective; missing ranks would hang it on exception paths
        exporter.finalize_publish.assert_not_called()

    def _run_train_entry(
        self,
        manager: mock.Mock,
        train_loop: mock.Mock,
        dumper: Optional[mock.Mock] = None,
        dump_enabled: bool = True,
    ) -> None:
        """Run a fully mocked train_and_evaluate with the given train loop."""
        config = EasyRecConfig(
            model_dir=self.test_dir, train_input_path="train", eval_input_path=""
        )
        config.train_config.num_steps = 1
        if dump_enabled:
            config.train_config.delta_embedding_dump_config.CopyFrom(
                DeltaEmbeddingDumpConfig(paired_dump_interval_steps=1)
            )
        train_dataloader = mock.Mock()
        wrapped_model = mock.Mock()
        wrapped_model.model.sparse_parameters.return_value = ([], [])
        dmp_model = mock.Mock()
        dmp_model.named_parameters.return_value = []
        optimizer_builder = mock.Mock()
        optimizer_builder.create_sparse_optimizer.return_value = (torch.optim.SGD, {})
        optimizer_builder.create_dense_optimizer.return_value = (torch.optim.SGD, {})
        optimizer_builder.create_part_optimizer.return_value = (torch.optim.SGD, {}, [])
        optimizer_builder.group_param_by_regex_pattern.return_value = ({}, {})
        optimizer_builder.build_part_optimizers.return_value = ([], [])
        optimizer_builder.create_part_optim_schedulers.return_value = []
        patches = [
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch(
                "tzrec.main.config_util.load_pipeline_config", return_value=config
            ),
            mock.patch(
                "tzrec.main.init_process_group",
                return_value=(torch.device("cpu"), "gloo"),
            ),
            mock.patch("tzrec.main.acc_utils.allow_tf32"),
            mock.patch("tzrec.main._create_features", return_value=[]),
            mock.patch("tzrec.utils.runtime.CheckpointManager", return_value=manager),
            mock.patch("tzrec.main.create_dataloader", return_value=train_dataloader),
            mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
            mock.patch("tzrec.main.TrainWrapper", return_value=wrapped_model),
            mock.patch("tzrec.main.apply_optimizer_in_backward"),
            mock.patch("tzrec.utils.runtime.create_planner", return_value=mock.Mock()),
            mock.patch("tzrec.utils.runtime.get_default_sharders", return_value=[]),
            mock.patch("torch.distributed.barrier"),
            mock.patch("tzrec.main.config_util.save_message"),
            mock.patch(
                "tzrec.utils.runtime.DistributedModelParallel", return_value=dmp_model
            ),
            mock.patch("tzrec.main.DeltaEmbeddingDumper", return_value=dumper),
            mock.patch("tzrec.main.optimizer_builder", optimizer_builder),
            mock.patch("tzrec.main.KeyedOptimizerWrapper", return_value=mock.Mock()),
            mock.patch("tzrec.main.CombinedOptimizer", return_value=mock.Mock()),
            mock.patch("tzrec.main.TZRecOptimizer", return_value=mock.Mock()),
            mock.patch("tzrec.main._train_and_evaluate", side_effect=train_loop),
        ]
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            train_and_evaluate("pipeline.config")

    def test_train_entry_drains_uploader_and_closes_manager(self) -> None:
        """A normal return drains the started uploader and closes the manager.

        The no-work return of the loop takes this same path: the entry, not
        the loop, owns both, and the loop borrows the runtime's manager.
        """
        dumper = mock.Mock()
        manager = mock.Mock()
        manager.latest_checkpoint.return_value = (None, 0)
        train_loop = mock.Mock()

        self._run_train_entry(manager, train_loop, dumper=dumper)

        dumper.start.assert_called_once()
        dumper.close.assert_called_once_with()
        manager.close.assert_called_once()
        self.assertIs(train_loop.call_args.kwargs["runtime"].checkpoints, manager)

    @parameterized.expand(
        [
            param("abort_ok", abort_raises=False),
            param("abort_raises", abort_raises=True),
        ],
        name_func=parameterized_name_func,
    )
    def test_train_entry_failure_requests_non_draining_abort(
        self, _name: str, abort_raises: bool
    ) -> None:
        """A training failure asks the uploader to abort without draining.

        The restarted run re-dumps from the latest checkpoint, so pending
        in-memory deltas are dropped; a secondary abort failure is only
        logged and the original training error still propagates.
        """
        dumper = mock.Mock()
        if abort_raises:
            dumper.close.side_effect = RuntimeError("abort boom")
        manager = mock.Mock()
        manager.latest_checkpoint.return_value = (None, 0)

        with self.assertRaisesRegex(RuntimeError, "train boom"):
            self._run_train_entry(
                manager,
                mock.Mock(side_effect=RuntimeError("train boom")),
                dumper=dumper,
            )

        dumper.close.assert_called_once_with(raise_on_error=False, drain=False)
        manager.close.assert_called_once()

    @parameterized.expand(
        [
            param("rank_zero", rank="0"),
            param("non_zero_rank", rank="1"),
        ],
        name_func=parameterized_name_func,
    )
    def test_evaluate_summary_writer_by_rank(self, _name: str, rank: str) -> None:
        """The standalone evaluate entry scopes its writer by rank.

        Rank zero's writer reaches _evaluate and closes when the
        evaluation call exits; other ranks create no writer and
        evaluate without one.
        """
        config = EasyRecConfig(model_dir=self.test_dir, eval_input_path="eval")
        runtime = mock.Mock()
        runtime.checkpoints.latest_checkpoint.return_value = ("ckpt", 5)
        runtime_cls = mock.MagicMock()
        runtime_cls.open.return_value.__enter__.return_value = runtime
        runtime_cls.open.return_value.__exit__.return_value = False
        writer = _FakeSummaryWriter()
        writer_cls = mock.Mock(return_value=writer)
        with (
            mock.patch.dict(os.environ, {"RANK": rank, "LOCAL_RANK": rank}),
            mock.patch(
                "tzrec.main.config_util.load_pipeline_config", return_value=config
            ),
            mock.patch(
                "tzrec.main.init_process_group",
                return_value=(torch.device("cpu"), "gloo"),
            ),
            mock.patch("tzrec.main.acc_utils.allow_tf32"),
            mock.patch("tzrec.main._create_features", return_value=[]),
            mock.patch("tzrec.main.create_dataloader", return_value=mock.Mock()),
            mock.patch("tzrec.main._create_model", return_value=mock.Mock()),
            mock.patch("tzrec.main.TrainWrapper", return_value=mock.Mock()),
            mock.patch("tzrec.main.ShardedModelRuntime", runtime_cls),
            mock.patch("tzrec.main.config_util.use_dense_ema", return_value=False),
            mock.patch("tzrec.main.SummaryWriter", writer_cls),
            mock.patch("tzrec.main._evaluate") as run_eval,
        ):
            evaluate("pipeline.config")

        if rank == "0":
            writer_cls.assert_called_once_with(os.path.join(self.test_dir, "eval"))
            self.assertIs(run_eval.call_args.kwargs["eval_summary_writer"], writer)
            writer.close.assert_called_once()
        else:
            writer_cls.assert_not_called()
            self.assertIsNone(run_eval.call_args.kwargs["eval_summary_writer"])

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
        exporter.enabled = False
        train_config = _train_config(
            save_checkpoints_steps=1,
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
                runtime=ShardedModelRuntime(ckpt_manager, None),
                dense_ema=dense_ema,
                export_config=ExportConfig(use_dense_ema=False),
            )

        torch.testing.assert_close(parameter, torch.tensor([4.0]))
        self.assertTrue(model.module.model.on_train_end.called)
        for call in exporter.maybe_export.call_args_list:
            self.assertIsNone(call.kwargs["dense_ema"])

    @parameterized.expand(
        [
            ["no_dumper", None, False],
            [
                "local_dump_only",
                DeltaEmbeddingDumpConfig(paired_dump_interval_steps=1),
                True,
            ],
        ],
        name_func=parameterized_name_func,
    )
    def test_online_dense_export_requires_feature_store_upload(
        self,
        _name: str,
        dump_config: DeltaEmbeddingDumpConfig,
        has_dumper: bool,
    ) -> None:
        """Enabled export must fail fast unless the sparse delta is uploaded.

        Without a FeatureStore uploader there is no upload watermark to gate
        current.json on, so a local-only dump would publish dense ahead of the
        serving-side sparse state.
        """
        exporter = mock.Mock()
        exporter.enabled = True
        train_config = _train_config()
        if dump_config is not None:
            train_config.delta_embedding_dump_config.CopyFrom(dump_config)

        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            with self.assertRaisesRegex(RuntimeError, "feature_store_config"):
                _train_and_evaluate(
                    model=mock.Mock(),
                    optimizer=mock.Mock(),
                    train_dataloader=mock.Mock(),
                    eval_dataloader=None,
                    lr_scheduler=[],
                    model_dir="unused",
                    train_config=train_config,
                    eval_config=EvalConfig(),
                    runtime=ShardedModelRuntime(mock.Mock(), None),
                    delta_embedding_dumper=mock.Mock() if has_dumper else None,
                )

    def test_train_loop_resume_with_empty_first_pass(self) -> None:
        """An end-of-data resume trains the next full pass.

        The empty remainder of the resumed pass yields no batch for the
        optimizer-init peek, so the peek falls through to the next pass:
        the empty pass is accounted like the loop's StopIteration handler
        and the restore still runs with the optimizer.
        """
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.return_value = False
        exporter = mock.Mock()
        exporter.enabled = False
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        optimizer = mock.Mock()
        optimizer.params = {}
        optimizer.param_groups = []
        train_dataloader = mock.Mock()
        batches = [
            SimpleNamespace(checkpoint_info=None, data_timestamp=1.0),
            SimpleNamespace(checkpoint_info=None, data_timestamp=2.0),
        ]
        train_dataloader.get_iterator.side_effect = [iter([]), iter(batches)]
        pipeline = mock.Mock()
        pipeline.progress.side_effect = lambda it: (
            {"loss": torch.tensor(1.0)},
            {},
            next(it),
        )
        dataloader_state = {checkpoint_util.EPOCHS_COMPLETED: 1, "src:0": 7}
        train_config = _train_config(num_steps=1, log_step_count_steps=10)
        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                lr_scheduler=[],
                model_dir="unused",
                train_config=train_config,
                eval_config=EvalConfig(),
                runtime=ShardedModelRuntime(ckpt_manager, None),
                ckpt_path="model.ckpt-10",
                restore_from_model_dir=True,
                dataloader_state=dataloader_state,
            )
        # the peek consumed the empty pass and took its batch from the next
        self.assertEqual(train_dataloader.get_iterator.call_count, 2)
        self.assertIs(ckpt_manager.restore.call_args.args[2], optimizer)
        # the empty pass completed the resumed one; its positions are gone
        self.assertEqual(dataloader_state, {checkpoint_util.EPOCHS_COMPLETED: 2})

    def test_train_loop_empty_data_fails_the_restore_loudly(self) -> None:
        """A training dataset without any row cannot init the optimizer.

        Both the empty remainder and the next full pass yield nothing, so
        the restore fails with a clear error instead of hanging the loop.
        """
        ckpt_manager = mock.Mock()
        exporter = mock.Mock()
        exporter.enabled = False
        model = mock.Mock()
        optimizer = mock.Mock()
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.side_effect = [iter([]), iter([])]
        pipeline = mock.Mock()
        train_config = _train_config(num_steps=1, log_step_count_steps=10)
        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            with self.assertRaisesRegex(RuntimeError, "training data is empty"):
                _train_and_evaluate(
                    model=model,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    eval_dataloader=None,
                    lr_scheduler=[],
                    model_dir="unused",
                    train_config=train_config,
                    eval_config=EvalConfig(),
                    runtime=ShardedModelRuntime(ckpt_manager, None),
                    ckpt_path="model.ckpt-10",
                    restore_from_model_dir=True,
                    dataloader_state={checkpoint_util.EPOCHS_COMPLETED: 1},
                )
        ckpt_manager.restore.assert_not_called()

    def test_train_loop_empty_resume_trains_exactly_the_leftover_epochs(self) -> None:
        """An end-of-data resume under an epoch budget trains the exact rest.

        The empty remainder consumed by the peek accounts the resumed pass
        and burns its epoch iteration, so a job with one epoch of budget
        left trains exactly one full pass, not two.
        """
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.return_value = False
        exporter = mock.Mock()
        exporter.enabled = False
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        optimizer = mock.Mock()
        optimizer.params = {}
        optimizer.param_groups = []
        train_dataloader = mock.Mock()
        batches = [
            SimpleNamespace(checkpoint_info=None, data_timestamp=float(i))
            for i in range(3)
        ]
        train_dataloader.get_iterator.side_effect = [iter([]), iter(batches)]
        pipeline = mock.Mock()
        pipeline.progress.side_effect = lambda it: (
            {"loss": torch.tensor(1.0)},
            {},
            next(it),
        )
        dataloader_state = {checkpoint_util.EPOCHS_COMPLETED: 1, "src:0": 7}
        train_config = _train_config(num_steps=0, num_epochs=3, log_step_count_steps=10)
        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                lr_scheduler=[],
                model_dir="unused",
                train_config=train_config,
                eval_config=EvalConfig(),
                runtime=ShardedModelRuntime(ckpt_manager, None),
                ckpt_path="model.ckpt-10",
                restore_from_model_dir=True,
                dataloader_state=dataloader_state,
            )
        # the peek's iterator plus the one full pass; no extra epoch ran
        self.assertEqual(train_dataloader.get_iterator.call_count, 2)
        self.assertEqual(pipeline.progress.call_count, 5)
        self.assertIs(ckpt_manager.restore.call_args.args[2], optimizer)
        self.assertEqual(dataloader_state, {checkpoint_util.EPOCHS_COMPLETED: 3})

    def test_train_loop_empty_resume_at_budget_saves_nothing(self) -> None:
        """The empty remainder completes the last epoch: nothing left to do.

        Like the entry's restored-counter early return, the run exits
        before the restore, so the source checkpoint is not rewritten.
        """
        ckpt_manager = mock.Mock()
        exporter = mock.Mock()
        exporter.enabled = False
        model = mock.Mock()
        optimizer = mock.Mock()
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([])
        pipeline = mock.Mock()
        train_config = _train_config(num_steps=0, num_epochs=2, log_step_count_steps=10)
        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                lr_scheduler=[],
                model_dir="unused",
                train_config=train_config,
                eval_config=EvalConfig(),
                runtime=ShardedModelRuntime(ckpt_manager, None),
                ckpt_path="model.ckpt-10",
                restore_from_model_dir=True,
                dataloader_state={checkpoint_util.EPOCHS_COMPLETED: 1},
            )
        self.assertEqual(train_dataloader.get_iterator.call_count, 1)
        ckpt_manager.restore.assert_not_called()
        ckpt_manager.maybe_save.assert_not_called()

    def test_train_loop_pairs_dense_export_with_delta_dump(self) -> None:
        """The dump decision drives the export; the poll and final publish follow.

        Pins the per-step order maybe_dump -> maybe_export(force=PAIRED) ->
        poll_publish(fresh upload state), the final export at the step
        final_dump returns, and dumper.close() draining before the
        finalize_publish collective so the final watermark covers the final
        dump.
        """
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        optimizer = mock.Mock()
        optimizer.params = {}
        optimizer.param_groups = []
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([object()] * 3)
        batch = SimpleNamespace(checkpoint_info=None, data_timestamp=17.0)
        pipeline = mock.Mock()
        pipeline.progress.return_value = ({"loss": torch.tensor(1.0)}, {}, batch)
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.return_value = False
        probe = {"pk": "emb.t", "sk": 1, "crc32": "aa" * 4, "encoding": "fp32"}
        upload_states = [(0, []), (0, []), (1, [probe]), (2, [probe])]
        dumper = mock.Mock()
        dumper.maybe_dump.side_effect = [
            DumpDecision.NONE,
            DumpDecision.PAIRED,
            DumpDecision.SPARSE_ONLY,
        ]
        dumper.completed_upload_state.side_effect = list(upload_states)
        dumper.final_dump.return_value = 2
        exporter = mock.Mock()
        exporter.enabled = True
        parent = mock.Mock()
        parent.attach_mock(dumper, "dumper")
        parent.attach_mock(exporter, "exporter")
        train_config = _train_config(num_steps=3, log_step_count_steps=10)
        train_config.delta_embedding_dump_config.paired_dump_interval_steps = 1
        train_config.delta_embedding_dump_config.feature_store_config.project_name = (
            "project"
        )

        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                lr_scheduler=[],
                model_dir="unused",
                train_config=train_config,
                eval_config=EvalConfig(),
                runtime=ShardedModelRuntime(ckpt_manager, None),
                delta_embedding_dumper=dumper,
            )

        self.assertEqual(
            dumper.maybe_dump.call_args_list, [mock.call(0), mock.call(1), mock.call(2)]
        )
        self.assertEqual(
            exporter.maybe_export.call_args_list,
            [
                mock.call(0, 17.0, model, force=False, dense_ema=None),
                mock.call(1, 17.0, model, force=True, dense_ema=None),
                mock.call(2, 17.0, model, force=False, dense_ema=None),
                mock.call(2, 17.0, model, final=True, dense_ema=None),
            ],
        )
        self.assertEqual(
            exporter.poll_publish.call_args_list,
            [mock.call(state) for state in upload_states[:3]],
        )
        dumper.final_dump.assert_called_once_with(2)
        exporter.finalize_publish.assert_called_once_with(upload_states[3])
        tracked = {
            "dumper.maybe_dump",
            "exporter.maybe_export",
            "exporter.poll_publish",
            "dumper.final_dump",
            "dumper.close",
            "exporter.finalize_publish",
            "exporter.close",
        }
        self.assertEqual(
            [name for name, _, _ in parent.mock_calls if name in tracked],
            ["dumper.maybe_dump", "exporter.maybe_export", "exporter.poll_publish"] * 3
            + [
                "dumper.final_dump",
                "exporter.maybe_export",
                "dumper.close",
                "exporter.finalize_publish",
                "exporter.close",
            ],
        )

    def _train_lrs(
        self,
        name,
        *,
        ignore_lr=None,
        ignore_optimizer=False,
        ckpt_path=None,
        by_epoch=False,
        save_by_epoch=False,
        fine_tune=False,
    ):
        """Train against mocks and return the LR each batch used.

        A resume replays the reference sequence from the checkpoint on; with
        the schedule restore skipped it restarts from the beginning.
        """
        parameter = torch.nn.Parameter(torch.ones(1))
        parameter.grad = torch.ones_like(parameter)
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        optimizer.params = {"weight": parameter}
        scheduler = (
            ExponentialDecayLR(optimizer, 1, 0.5, by_epoch=True)
            if by_epoch
            else LinearDecayLR(optimizer, num_training_steps=10)
        )
        used_lrs = []
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        # a fine-tune keeps its own step budget, so it trains from batch 0
        skip_steps = (
            checkpoint_util._get_checkpoint_step(ckpt_path)
            if ckpt_path and not fine_tune
            else -1
        )
        # 3-batch passes; a resume rejoins mid-pass, so the first is short
        pass_sizes = itertools.chain([3 - (skip_steps + 1) % 3], itertools.repeat(3))
        loader = mock.Mock()
        loader.get_iterator.side_effect = lambda: iter(range(next(pass_sizes)))
        batch = SimpleNamespace(checkpoint_info=None, data_timestamp=-1.0)
        warming_up = ckpt_path is not None and not ignore_optimizer

        def progress(iterator):
            nonlocal warming_up
            next(iterator)
            if warming_up:
                warming_up = False
            else:
                used_lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            return {"loss": parameter.detach().sum()}, {}, batch

        pipeline = mock.Mock()
        pipeline.progress.side_effect = progress
        dataloader_state = (
            checkpoint_util.restore_dataloader_state(ckpt_path) if ckpt_path else None
        )
        if dataloader_state and fine_tune:
            # fine-tune checkpoints do not carry this job's epoch budget
            dataloader_state.pop(checkpoint_util.EPOCHS_COMPLETED, None)
        manager = checkpoint_util.CheckpointManager(os.path.join(self.test_dir, name))
        config = _train_config(
            num_steps=0 if by_epoch else 6,
            num_epochs=3 if by_epoch else 0,
            save_checkpoints_steps=1,
            save_checkpoints_epochs=1 if save_by_epoch else 0,
        )
        restore_kwargs = (
            {} if ignore_lr is None else {"ignore_restore_lr_scheduler": ignore_lr}
        )
        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch(
                "tzrec.main.OnlineDenseExportManager",
                return_value=mock.Mock(enabled=False),
            ),
            mock.patch("tzrec.main._log_train"),
            mock.patch("tzrec.utils.checkpoint_util.save_model"),
            mock.patch("tzrec.utils.checkpoint_util.restore_model") as restore_model,
        ):
            _train_and_evaluate(
                model,
                optimizer,
                loader,
                None,
                [scheduler],
                os.path.join(self.test_dir, name),
                config,
                EvalConfig(),
                ShardedModelRuntime(manager, None),
                ckpt_path=ckpt_path,
                skip_steps=skip_steps,
                ignore_restore_optimizer=ignore_optimizer,
                restore_from_model_dir=ckpt_path is not None and not fine_tune,
                dataloader_state=dataloader_state,
                **restore_kwargs,
            )
        if ckpt_path:
            self.assertEqual(restore_model.call_count, 1)
            self.assertIs(
                restore_model.call_args.args[2], None if ignore_optimizer else optimizer
            )
        else:
            restore_model.assert_not_called()
        return used_lrs

    @parameterized.expand(
        [
            ("default", None, False, False, False),
            ("weights_only", None, True, False, False),
            ("ignore", True, False, False, False),
            ("ignore_without_optimizer", True, True, False, False),
            ("no_saved_state", None, False, True, False),
            ("fine_tune", None, True, False, True),
        ],
        name_func=parameterized_name_func,
    )
    def test_batch_resume(self, name, ignore_lr, ignore, no_saved_state, fine_tune):
        reference = self._train_lrs("source")
        ckpt = os.path.join(self.test_dir, "source", "model.ckpt-3")
        if no_saved_state:
            # stands in for a checkpoint written before this existed
            os.remove(os.path.join(ckpt, checkpoint_util.LR_SCHEDULER_CKPT_FILENAME))
        else:
            self.assertTrue(
                os.path.exists(
                    os.path.join(ckpt, checkpoint_util.LR_SCHEDULER_CKPT_FILENAME)
                )
            )
        actual = self._train_lrs(
            "resumed",
            ignore_lr=ignore_lr,
            ignore_optimizer=ignore,
            ckpt_path=ckpt,
            fine_tune=fine_tune,
        )
        if fine_tune:
            expected = reference
        elif ignore_lr or no_saved_state:
            expected = reference[:2]
        else:
            # model.ckpt-3 follows batch index 3, i.e. 4 advances
            expected = reference[4:]
        torch.testing.assert_close(actual, expected)

    @parameterized.expand(
        [
            ("default", None, True, False),
            ("ignore", True, True, False),
            ("epoch_mid_pass", None, False, False),
            ("epoch_boundary_epoch_save", None, True, True),
        ],
        name_func=parameterized_name_func,
    )
    def test_epoch_resume(self, name, ignore_lr, boundary, save_by_epoch):
        reference = self._train_lrs(
            "source", by_epoch=True, save_by_epoch=save_by_epoch
        )
        step = 2 if boundary else 4
        actual = self._train_lrs(
            "resumed",
            ignore_lr=ignore_lr,
            ignore_optimizer=True,
            ckpt_path=os.path.join(self.test_dir, "source", f"model.ckpt-{step}"),
            by_epoch=True,
        )
        expected = reference[step + 1 :]
        if ignore_lr:
            # one epoch behind with decay_factor=0.5 ⇒ every rate doubles
            expected = [lr * 2 for lr in expected]
        torch.testing.assert_close(actual, expected)

    @parameterized.expand(
        [
            param("fresh", num_steps=50, ckpt_step=None, fine_tune=False, start=0),
            param("continue", num_steps=100, ckpt_step=49, fine_tune=False, start=50),
            param("completed", num_steps=50, ckpt_step=49, fine_tune=False, start=50),
            param("past_target", num_steps=50, ckpt_step=99, fine_tune=False, start=50),
            param("fine_tune", num_steps=50, ckpt_step=49, fine_tune=True, start=0),
        ],
        name_func=parameterized_name_func,
    )
    def test_step_resume(self, name, num_steps, ckpt_step, fine_tune, start):
        """Resuming a reached num_steps target is a no-op without checkpoint writes.

        Only a model_dir resume skips steps; a fine-tune checkpoint's step
        belongs to the source job, so it trains (and saves) the full budget.
        """
        model_dir = os.path.join(self.test_dir, "model")
        ckpt_path = None
        if ckpt_step is not None:
            ckpt_dir = os.path.join(self.test_dir, "source") if fine_tune else model_dir
            ckpt_path = os.path.join(ckpt_dir, f"model.ckpt-{ckpt_step}")
        restore_from_model_dir = ckpt_path is not None and not fine_tune
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        loader = mock.Mock()
        loader.get_iterator.return_value = iter(range(num_steps))
        batch = SimpleNamespace(checkpoint_info=None, data_timestamp=-1.0)

        def progress(iterator):
            next(iterator)
            return {"loss": torch.tensor(1.0)}, {}, batch

        pipeline = mock.Mock()
        pipeline.progress.side_effect = progress
        writes = (
            "save_model",
            "save_lr_schedulers",
            "save_dataloader_state",
            "save_meta",
            "save_success_marker",
        )
        with (
            mock.patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch(
                "tzrec.main.OnlineDenseExportManager",
                return_value=mock.Mock(enabled=False),
            ),
            mock.patch("tzrec.main._log_train") as log_train,
            mock.patch("tzrec.utils.checkpoint_util.restore_model") as restore_model,
            mock.patch.multiple(
                "tzrec.utils.checkpoint_util", **{w: mock.DEFAULT for w in writes}
            ) as write_mocks,
        ):
            _train_and_evaluate(
                model,
                mock.Mock(),
                loader,
                None,
                [],
                model_dir,
                _train_config(num_steps=num_steps, log_step_count_steps=1),
                EvalConfig(),
                ShardedModelRuntime(checkpoint_util.CheckpointManager(model_dir), None),
                skip_steps=ckpt_step if restore_from_model_dir else -1,
                ckpt_path=ckpt_path,
                ignore_restore_optimizer=True,
                restore_from_model_dir=restore_from_model_dir,
            )
        # the trailing _log_train call reports the final step after the loop
        trained_steps = [call.args[0] for call in log_train.call_args_list[:-1]]
        self.assertEqual(trained_steps, list(range(start, num_steps)))
        if start >= num_steps:
            for w in writes:
                write_mocks[w].assert_not_called()
            pipeline.progress.assert_not_called()
            restore_model.assert_not_called()
        else:
            self.assertEqual(restore_model.call_count, int(ckpt_path is not None))
            self.assertEqual(
                [call.args[0] for call in write_mocks["save_model"].call_args_list],
                [os.path.join(model_dir, f"model.ckpt-{num_steps - 1}")],
            )


class KafkaOffsetLogTest(unittest.TestCase):
    """Cumulative Kafka offset logging from trained batches."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir)

    @staticmethod
    def _batch(checkpoint_info):
        return SimpleNamespace(checkpoint_info=checkpoint_info, data_timestamp=-1.0)

    def _run(
        self,
        batches,
        log_step_count_steps,
        rank="0",
        local_rank="0",
        kafka_dataset=True,
        log_every_n_messages=None,
        initial_state=None,
    ):
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        loader = mock.Mock()
        loader.get_iterator.return_value = iter(batches)
        data_config = DataConfig(
            dataset_type=(
                DatasetType.KafkaDataset
                if kafka_dataset
                else DatasetType.ParquetDataset
            )
        )
        if log_every_n_messages is not None:
            data_config.kafka_log_every_n_messages = log_every_n_messages
        loader.dataset._data_config = data_config

        pipeline = mock.Mock()

        def progress(iterator):
            return {"loss": torch.tensor(1.0)}, {}, next(iterator)

        pipeline.progress.side_effect = progress
        writes = (
            "save_model",
            "save_lr_schedulers",
            "save_dataloader_state",
            "save_meta",
            "save_success_marker",
        )
        num_steps = len(batches)
        model_dir = os.path.join(self.test_dir, f"model-{rank}-{num_steps}")
        dataloader_state = dict(initial_state or {})
        with (
            mock.patch.dict(os.environ, {"RANK": rank, "LOCAL_RANK": local_rank}),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch(
                "tzrec.main.OnlineDenseExportManager",
                return_value=mock.Mock(enabled=False),
            ),
            mock.patch("tzrec.main._log_train"),
            mock.patch("tzrec.main.logger") as mock_logger,
            mock.patch.multiple(
                "tzrec.utils.checkpoint_util", **{w: mock.DEFAULT for w in writes}
            ),
        ):
            _train_and_evaluate(
                model,
                mock.Mock(),
                loader,
                None,
                [],
                model_dir,
                _train_config(
                    num_steps=num_steps,
                    log_step_count_steps=log_step_count_steps,
                ),
                EvalConfig(),
                ShardedModelRuntime(checkpoint_util.CheckpointManager(model_dir), None),
                dataloader_state=dataloader_state,
            )
        return (
            [
                call.args[0]
                for call in mock_logger.info.call_args_list
                if "KafkaTrain" in call.args[0]
            ],
            dataloader_state,
        )

    @parameterized.expand(
        [
            param(
                "final_snapshot",
                [
                    {"topic:0": 1000},
                    {"topic:0": 1400},
                    {"topic:0": 1600, "topic:1": 50},
                    {"topic:0": 1800, "topic:1": 70},
                ],
                [
                    "KafkaTrain[rank-3] step=0 cumulative_offset_deltas={'topic:0': 0}",
                    "KafkaTrain[rank-3] step=2 cumulative_offset_deltas="
                    "{'topic:0': 600, 'topic:1': 0}",
                    "KafkaTrain[rank-3] step=3 cumulative_offset_deltas="
                    "{'topic:0': 800, 'topic:1': 20}",
                ],
            ),
            param(
                "final_step_deduplicated",
                [
                    {"topic:0": 1000},
                    {"topic:0": 1400},
                    {"topic:0": 1600, "topic:1": 50},
                    {"topic:0": 1800, "topic:1": 70},
                    {"topic:0": 2000, "topic:1": 90},
                ],
                [
                    "KafkaTrain[rank-3] step=0 cumulative_offset_deltas={'topic:0': 0}",
                    "KafkaTrain[rank-3] step=2 cumulative_offset_deltas="
                    "{'topic:0': 600, 'topic:1': 0}",
                    "KafkaTrain[rank-3] step=4 cumulative_offset_deltas="
                    "{'topic:0': 1000, 'topic:1': 40}",
                ],
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_logs_cumulative_offsets(self, _name, checkpoint_infos, expected_logs):
        batches = [self._batch(info) for info in checkpoint_infos]
        logs, _ = self._run(
            batches,
            log_step_count_steps=2,
            rank="3",
            local_rank="3",
            log_every_n_messages=1000,
        )
        self.assertEqual(logs, expected_logs)

    def test_checkpoint_baseline(self):
        batches = [
            self._batch({"topic:0": 1200}),
            self._batch({"topic:0": 1500}),
        ]
        logs, _ = self._run(
            batches,
            log_step_count_steps=2,
            log_every_n_messages=1000,
            initial_state={"topic:0": 1099},
        )
        self.assertEqual(
            logs,
            [
                "KafkaTrain[rank-0] step=0 cumulative_offset_deltas={'topic:0': 0}",
                "KafkaTrain[rank-0] step=1 cumulative_offset_deltas={'topic:0': 300}",
            ],
        )

    @parameterized.expand(
        [
            param("unset", True, None),
            param("zero", True, 0),
            param("non_kafka", False, 1000),
        ],
        name_func=parameterized_name_func,
    )
    def test_logging_disabled_keeps_checkpoint_state(self, _name, kafka_dataset, value):
        batches = [
            self._batch({"topic:0": 100, "topic:1": 50}),
            self._batch({"topic:0": 200}),
        ]
        logs, state = self._run(
            batches,
            log_step_count_steps=2,
            kafka_dataset=kafka_dataset,
            log_every_n_messages=value,
        )
        self.assertEqual(logs, [])
        self.assertEqual(state, {"topic:0": 200, "topic:1": 50})


class PredictionLifecycleTest(unittest.TestCase):
    """Tests for prediction lifecycle wiring."""

    def _batch(self, rows: int) -> Batch:
        return Batch(
            reserves=RecordBatchTensor(pa.record_batch({"id": list(range(rows))}))
        )

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
            mock.patch(
                "tzrec.main.acc_utils.is_input_tile_predict", return_value=False
            ),
            mock.patch(
                "tzrec.main.acc_utils.is_distributed_embedding_predict",
                return_value=False,
            ),
            mock.patch("tzrec.main._create_features", return_value=[]),
            mock.patch("tzrec.main.create_dataloader", return_value=dataloader),
            mock.patch("tzrec.main.create_writer", return_value=writer),
            mock.patch.object(predict_util, "load_predict_model", return_value=model),
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
                "tzrec.utils.runtime.CheckpointManager",
                return_value=ckpt_manager,
            ),
            mock.patch("tzrec.utils.runtime.create_planner", return_value=planner),
            mock.patch("tzrec.utils.runtime.get_default_sharders", return_value=[]),
            mock.patch(
                "tzrec.utils.runtime.DistributedModelParallel",
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


class TrainProfilerTest(unittest.TestCase):
    """Tests for the configurable training profiler."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def _actions(self, prof, num_steps=16):
        return [prof.schedule(i) for i in range(num_steps)]

    def test_default_config_matches_legacy_profiler(self):
        prof = _create_train_profiler(TrainConfig().profiler_config, self.test_dir)
        legacy = torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1)
        self.assertEqual(self._actions(prof), [legacy(i) for i in range(16)])
        self.assertTrue(prof.record_shapes)
        self.assertTrue(prof.profile_memory)
        self.assertTrue(prof.with_stack)

    def test_later_window_and_low_overhead(self):
        config = ProfilerConfig(
            skip_first_steps=5,
            wait_steps=0,
            warmup_steps=2,
            active_steps=3,
            profile_memory=False,
            with_stack=False,
        )
        prof = _create_train_profiler(config, self.test_dir)
        A = torch.profiler.ProfilerAction
        self.assertEqual(
            self._actions(prof, 12),
            [A.NONE] * 5
            + [A.WARMUP] * 2
            + [A.RECORD] * 2
            + [A.RECORD_AND_SAVE]
            + [A.NONE] * 2,
        )
        self.assertFalse(prof.profile_memory)
        self.assertFalse(prof.with_stack)

    @parameterized.expand(
        [([], True), ([0, 3], True), ([0, 1], False)],
        name_func=parameterized_name_func,
    )
    def test_rank_filter(self, ranks, profiled):
        with mock.patch.dict(os.environ, {"RANK": "3"}):
            prof = _create_train_profiler(ProfilerConfig(ranks=ranks), self.test_dir)
        self.assertEqual(prof is not None, profiled)

    def test_trace_file_named_by_rank(self):
        config = ProfilerConfig(wait_steps=0, warmup_steps=1, active_steps=1)
        with mock.patch.dict(os.environ, {"RANK": "3"}):
            prof = _create_train_profiler(config, self.test_dir)
        prof.start()
        for _ in range(3):
            torch.ones(4).sum()
            prof.step()
        prof.stop()
        traces = os.listdir(self.test_dir)
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0].startswith("rank3_"), traces[0])


class OnlineLearningHandoverTest(unittest.TestCase):
    """Tests for the train loop retiring a stream lineage on hand-over."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def test_train_loop_retires_on_handover(self) -> None:
        """Hand-over leaves both loops, skips the epoch-end and final saves.

        The successor check runs only at the log cadence after poll_publish;
        once it passes, the trailing delta is still dumped, uploads drained
        and the final publish finalized.
        """
        model = mock.Mock()
        model.module.model.compute_train_metric.return_value = {}
        optimizer = mock.Mock()
        optimizer.params = {}
        optimizer.param_groups = []
        train_dataloader = mock.Mock()
        train_dataloader.get_iterator.return_value = iter([object()] * 10)
        batch = SimpleNamespace(checkpoint_info=None, data_timestamp=17.0)
        pipeline = mock.Mock()
        pipeline.progress.return_value = ({"loss": torch.tensor(1.0)}, {}, batch)
        ckpt_manager = mock.Mock()
        ckpt_manager.maybe_save.return_value = False
        dumper = mock.Mock()
        dumper.maybe_dump.return_value = DumpDecision.NONE
        dumper.final_dump.return_value = 2
        exporter = mock.Mock()
        exporter.enabled = True
        train_config = _train_config(num_steps=0, num_epochs=1, log_step_count_steps=2)
        train_config.delta_embedding_dump_config.paired_dump_interval_steps = 1
        train_config.delta_embedding_dump_config.feature_store_config.project_name = (
            "project"
        )
        env = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "ONLINE_LEARNING_SUCCESSOR_DIR": os.path.join(self.test_dir, "successor"),
        }

        with (
            mock.patch.dict(os.environ, env),
            mock.patch("tzrec.main.create_train_pipeline", return_value=pipeline),
            mock.patch("tzrec.main.OnlineDenseExportManager", return_value=exporter),
            mock.patch.object(
                OnlineLearningHandover, "update_and_check", side_effect=[False, True]
            ) as update_and_check,
        ):
            _train_and_evaluate(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                lr_scheduler=[],
                model_dir="unused",
                train_config=train_config,
                eval_config=EvalConfig(),
                runtime=ShardedModelRuntime(ckpt_manager, None),
                delta_embedding_dumper=dumper,
            )

        self.assertEqual(
            update_and_check.call_args_list,
            [mock.call(0, 17.0), mock.call(2, 17.0)],
        )
        self.assertEqual(pipeline.progress.call_count, 3)
        self.assertEqual(exporter.poll_publish.call_count, 3)
        for call in ckpt_manager.maybe_save.call_args_list:
            self.assertNotIn("epoch", call.kwargs)
            self.assertFalse(call.kwargs.get("final", False))
        self.assertTrue(model.module.model.on_train_end.called)
        dumper.final_dump.assert_called_once_with(2)
        dumper.close.assert_called_once()
        exporter.finalize_publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
