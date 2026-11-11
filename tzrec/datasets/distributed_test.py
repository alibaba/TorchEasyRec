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

import os
import shutil
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import torch
import torch.multiprocessing as mp
from parameterized import parameterized
from pyarrow import parquet
from torch import distributed as dist

from tzrec.datasets.csv_dataset import CsvReader
from tzrec.datasets.dataset import BaseDataset
from tzrec.datasets.kafka_dataset import KafkaReader
from tzrec.datasets.odps_dataset import OdpsReader
from tzrec.datasets.parquet_dataset import ParquetReader
from tzrec.datasets.sampler import _bootstrap
from tzrec.utils import dist_util
from tzrec.utils.test_util import make_test_dir, mark_ci_scope, parameterized_name_func


def _check_readers(test_dir, pg):
    case = unittest.TestCase()
    consumer = mock.Mock()
    consumer.list_topics.return_value = SimpleNamespace(
        topics={
            "topic": SimpleNamespace(
                error=None,
                partitions={
                    0: SimpleNamespace(leader=1),
                    1: SimpleNamespace(leader=0),
                },
            )
        }
    )
    with mock.patch("tzrec.datasets.kafka_dataset.Consumer", return_value=consumer):
        reader = KafkaReader(
            "kafka://broker:9092/topic?group.id=g1",
            1,
            input_fields=[pa.field("x", pa.int64())],
            kafka_partition_round_robin=True,
        )
    case.assertTrue(reader._kafka_partition_round_robin)
    case.assertEqual(reader._round_robin_partition_order, [1, 0])
    if pg is None or dist.get_rank(pg) == 0:
        consumer.list_topics.assert_called_once()
    else:
        consumer.list_topics.assert_not_called()

    parquet_reader = ParquetReader(os.path.join(test_dir, "*.parquet"), 1)
    case.assertIs(parquet_reader._pg, pg)
    case.assertEqual(sum(parquet_reader._num_rows), 4)
    csv_reader = CsvReader(os.path.join(test_dir, "*.csv"), 1)
    case.assertEqual(len(csv_reader._input_files), 1)

    def init_client(reader):
        reader._table_to_cli["table"] = SimpleNamespace(
            table=SimpleNamespace(
                schema=SimpleNamespace(
                    simple_columns=[SimpleNamespace(name="x", type="bigint")]
                )
            )
        )

    with (
        mock.patch(
            "tzrec.datasets.odps_dataset._create_odps_account", return_value=(None, "")
        ),
        mock.patch.object(OdpsReader, "_init_client", init_client),
        mock.patch(
            "tzrec.datasets.odps_dataset._create_read_session",
            return_value=("session", 4),
        ),
        mock.patch("tzrec.datasets.odps_dataset._refresh_sessions_daemon"),
    ):
        odps_reader = OdpsReader("odps://project/tables/table", 1)
    case.assertIs(odps_reader._pg, pg)
    case.assertEqual(odps_reader._sess_row_counts, {"session": 4})


def _dataset_worker(rank, backend, test_dir):
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    if backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    case = unittest.TestCase()
    # Reinitialize to check that the cache never returns a destroyed group.
    previous_pg = None
    for generation in range(2):
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{test_dir}/store-{generation}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=45),
        )
        try:
            pg = dist_util.get_dataset_process_group()
            case.assertIsNot(pg, dist.group.WORLD)
            case.assertIsNot(pg, previous_pg)
            case.assertEqual(dist.get_backend(pg), "gloo")
            case.assertIs(pg, dist_util.get_dataset_process_group())
            previous_pg = pg

            with (
                mock.patch.object(
                    dist, "broadcast_object_list", wraps=dist.broadcast_object_list
                ) as broadcast,
                mock.patch.object(
                    dist, "all_gather_object", wraps=dist.all_gather_object
                ) as gather,
                mock.patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce,
            ):
                _check_readers(test_dir, pg)
                _bootstrap(1, rank, 0)
                for collective in (broadcast, gather, reduce):
                    case.assertTrue(collective.called)
                    for call in collective.call_args_list:
                        case.assertIs(call.kwargs["group"], pg)

            if rank == 0:
                with (
                    mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}),
                    mock.patch.object(dist, "new_group") as new_group,
                    mock.patch.object(dist, "broadcast_object_list") as broadcast,
                    mock.patch.object(dist, "all_gather_object") as gather,
                    mock.patch.object(dist, "all_reduce") as reduce,
                ):
                    case.assertIsNone(dist_util.get_dataset_process_group())
                    _check_readers(test_dir, None)
                    _bootstrap(1, 0, 0)
                    case.assertEqual(BaseDataset.get_worker_info(None), (0, 1, 1))
                    for collective in (new_group, broadcast, gather, reduce):
                        collective.assert_not_called()

            # Rank 1 waits here while rank 0 builds the local warm-up readers.
            value = torch.tensor([rank + 1], device=device)
            dist.all_reduce(value)
            case.assertEqual(value.item(), 3)
            case.assertIs(dist_util.get_dataset_process_group(), pg)
        finally:
            dist.destroy_process_group()


@mark_ci_scope("gpu")
class DatasetProcessGroupTest(unittest.TestCase):
    @parameterized.expand([["gloo"], ["nccl"]], name_func=parameterized_name_func)
    def test_readers_and_rank_zero_warmup(self, backend):
        if backend == "nccl" and torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices")
        test_dir = os.path.abspath(make_test_dir())
        context = None
        try:
            parquet.write_table(
                pa.table({"x": [1, 2, 3, 4]}), os.path.join(test_dir, "data.parquet")
            )
            with open(os.path.join(test_dir, "data.csv"), "w") as f:
                f.write("x\n1\n2\n")
            context = mp.spawn(
                _dataset_worker, args=(backend, test_dir), nprocs=2, join=False
            )
            deadline = time.monotonic() + 120
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    self.fail("dataset communication timed out")
        finally:
            if context is not None:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
