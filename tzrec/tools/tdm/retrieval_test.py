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

import multiprocessing
import os
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import torch
from parameterized import parameterized

from tzrec.datasets.utils import Batch, RecordBatchTensor
from tzrec.protos.data_pb2 import DataConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.sampler_pb2 import TDMSampler
from tzrec.tools.tdm import retrieval
from tzrec.utils import predict_util
from tzrec.utils.test_util import parameterized_name_func

_BATCH_SIZE = 2
_SAMPLE_PER_BATCH = 8


class _FailingSampler:
    _item_id_field = "item_id"

    def init(self, worker_id):
        raise ValueError("worker boom")


class _PassingSampler:
    _item_id_field = "item_id"

    def init(self, worker_id):
        return None

    def init_sampler(self, n_cluster):
        return None

    def get(self, input_data):
        return {"item_id": pa.array([2])}


class _PassingParser:
    def parse(self, input_data):
        return input_data

    def to_batch(self, output_data, force_no_tile=False):
        return Batch()


class _RetrievalSampler:
    """Sampler stub expanding every input batch to a fixed candidate count."""

    _item_id_field = "item_id"

    def init(self, worker_id):
        return None

    def init_cluster(self, num_client_per_rank):
        return None

    def launch_server(self):
        return None

    def init_sampler(self, n_cluster):
        return None

    def get(self, input_data):
        return {"item_id": pa.array(list(range(_SAMPLE_PER_BATCH)))}


class TDMRetrievalLifecycleTest(unittest.TestCase):
    @parameterized.expand([[1], [2], [4], [8]], name_func=parameterized_name_func)
    def test_normal_sentinel_fanout(self, worker_count):
        for downstream_count in (worker_count, 1):
            data_queue = queue.Queue()
            pred_queue = queue.Queue(maxsize=2 if downstream_count == 1 else 0)
            cancel_event = threading.Event()
            failure_queue = queue.Queue()
            input_drained_event = threading.Event()
            for _ in range(worker_count):
                data_queue.put((None, None, None))

            retrieval._forward_loop(
                data_queue,
                pred_queue,
                3,
                worker_count,
                downstream_count,
                mock.Mock(),
                input_drained_event,
                cancel_event,
                failure_queue,
            )

            self.assertEqual(pred_queue.qsize(), downstream_count)
            for _ in range(downstream_count):
                self.assertEqual(pred_queue.get_nowait(), (None, None))
            self.assertFalse(cancel_event.is_set())
            self.assertTrue(failure_queue.empty())
            self.assertTrue(input_drained_event.is_set())

    def test_data_worker_keeps_shared_storage_alive_until_drained(self):
        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()
        output_drained_event = multiprocessing.Event()
        cancel_event = multiprocessing.Event()
        failure_queue = multiprocessing.Queue()
        record_batch_t = RecordBatchTensor(pa.record_batch({"item_id": [1]}))
        in_queue.put((record_batch_t, pa.array([1])))
        in_queue.put((None, None))
        worker = multiprocessing.Process(
            target=retrieval._tdm_predict_data_worker,
            args=(
                _PassingSampler(),
                _PassingParser(),
                1,
                2,
                in_queue,
                out_queue,
                False,
                0,
                output_drained_event,
                cancel_event,
                failure_queue,
            ),
        )
        worker.start()

        batch, output_record_batch_t, node_ids = out_queue.get(timeout=2)
        self.assertIsInstance(batch, Batch)
        self.assertEqual(output_record_batch_t.get()["item_id"].to_pylist(), [1])
        self.assertEqual(node_ids.to_pylist(), [2])
        self.assertEqual(out_queue.get(timeout=1), (None, None, None))
        self.assertTrue(worker.is_alive())
        output_drained_event.set()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.exitcode, 0)
        self.assertFalse(cancel_event.is_set())
        self.assertTrue(failure_queue.empty())
        for data_queue in (in_queue, out_queue, failure_queue):
            data_queue.close()
            data_queue.join_thread()

    @parameterized.expand(
        [["worker"], ["forward"], ["writer"]],
        name_func=parameterized_name_func,
    )
    def test_stage_failure_is_propagated_without_commit(self, stage):
        cancel_event = threading.Event()
        failure_queue = queue.Queue()
        writer = mock.Mock()

        if stage == "worker":
            with self.assertRaisesRegex(ValueError, "worker boom"):
                retrieval._tdm_predict_data_worker(
                    _FailingSampler(),
                    mock.Mock(),
                    1,
                    2,
                    queue.Queue(),
                    queue.Queue(),
                    True,
                    7,
                    threading.Event(),
                    cancel_event,
                    failure_queue,
                )
        elif stage == "forward":
            data_queue = queue.Queue()
            data_queue.put((object(), object(), object()))
            retrieval._forward_loop(
                data_queue,
                queue.Queue(),
                3,
                1,
                1,
                mock.Mock(side_effect=ValueError("forward boom")),
                threading.Event(),
                cancel_event,
                failure_queue,
            )
        else:
            pred_queue = queue.Queue()
            pred_queue.put((object(), object()))
            retrieval._write_loop(
                pred_queue,
                mock.Mock(side_effect=ValueError("writer boom")),
                cancel_event,
                failure_queue,
            )

        self.assertTrue(cancel_event.is_set())
        with self.assertRaises(predict_util.PredictPipelineStageError) as error:
            predict_util.raise_background_failure(failure_queue)
        self.assertIn(f"{stage} boom", str(error.exception))
        self.assertIn("ValueError", str(error.exception))
        with self.assertRaises(predict_util.PredictPipelineStageError):
            predict_util.commit_prediction_output(
                writer, error.exception, 1, 1, torch.device("cpu")
            )
        writer.close.assert_not_called()


class TDMRetrievalTest(unittest.TestCase):
    """Tests for the ``tdm_retrieval`` queue topology, end to end."""

    def _batch(self):
        return SimpleNamespace(
            reserves=RecordBatchTensor(
                pa.record_batch(
                    {
                        "item_id": list(range(_BATCH_SIZE)),
                        "user_id": list(range(_BATCH_SIZE)),
                    }
                )
            )
        )

    def _pipeline_config(self):
        return EasyRecConfig(
            data_config=DataConfig(
                dataset_type=3,
                num_workers=1,
                tdm_sampler=TDMSampler(
                    item_input_path="item",
                    edge_input_path="edge",
                    predict_edge_input_path="predict_edge",
                    item_id_field="item_id",
                    layer_num_sample=[1, 2, 4, 8],
                ),
            )
        )

    def _run_retrieval(
        self, batch_count: int, writer: mock.Mock, num_worker_per_level: int
    ) -> None:
        def bounded(fn):
            """Fail fast instead of stalling for the production queue timeout."""

            def wrapper(*args, **kwargs):
                kwargs.setdefault("timeout", 20)
                return fn(*args, **kwargs)

            return wrapper

        dataloader = mock.Mock()
        dataloader.get_iterator.return_value = iter(
            [self._batch() for _ in range(batch_count)]
        )
        model = mock.Mock(return_value={"probs": torch.rand(_SAMPLE_PER_BATCH)})
        with (
            mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}),
            mock.patch(
                "tzrec.tools.tdm.retrieval.init_process_group",
                return_value=(torch.device("cpu"), "gloo"),
            ),
            mock.patch("tzrec.tools.tdm.retrieval.dist"),
            mock.patch(
                "tzrec.tools.tdm.retrieval.config_util.load_pipeline_config",
                return_value=self._pipeline_config(),
            ),
            mock.patch("tzrec.tools.tdm.retrieval._create_features", return_value=[]),
            mock.patch(
                "tzrec.tools.tdm.retrieval.create_dataloader", return_value=dataloader
            ),
            mock.patch("tzrec.tools.tdm.retrieval.create_writer", return_value=writer),
            mock.patch(
                "tzrec.tools.tdm.retrieval.TDMPredictSampler",
                return_value=_RetrievalSampler(),
            ),
            mock.patch(
                "tzrec.tools.tdm.retrieval.DataParser", return_value=_PassingParser()
            ),
            mock.patch("tzrec.tools.tdm.retrieval.torch.jit.load", return_value=model),
            mock.patch.object(
                predict_util,
                "queue_get_interruptibly",
                bounded(predict_util.queue_get_interruptibly),
            ),
            mock.patch.object(
                predict_util,
                "queue_put_interruptibly",
                bounded(predict_util.queue_put_interruptibly),
            ),
            mock.patch.object(
                predict_util,
                "wait_for_pipeline",
                bounded(predict_util.wait_for_pipeline),
            ),
        ):
            retrieval.tdm_retrieval(
                predict_input_path="input",
                predict_output_path="output",
                scripted_model_path="model",
                recall_num=1,
                n_cluster=2,
                reserved_columns="user_id",
                writer_type="MockWriter",
                num_worker_per_level=num_worker_per_level,
            )

    @parameterized.expand([[1], [4]], name_func=parameterized_name_func)
    def test_retrieval_writes_every_batch_and_commits_once(self, num_worker_per_level):
        writer = mock.Mock()
        batch_count = 3

        self._run_retrieval(batch_count, writer, num_worker_per_level)

        self.assertEqual(writer.write.call_count, batch_count)
        recall_ids = writer.write.call_args.args[0]["recall_ids"]
        self.assertEqual(len(recall_ids), _BATCH_SIZE)
        writer.close.assert_called_once_with()

    def test_empty_input_drains_without_commit(self):
        writer = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "empty"):
            self._run_retrieval(0, writer, num_worker_per_level=2)

        writer.write.assert_not_called()
        writer.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
