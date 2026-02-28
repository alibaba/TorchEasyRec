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

import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from parameterized import parameterized
from pyarrow import parquet
from torch import distributed as dist
from torch.utils.data import DataLoader

from tzrec.datasets.parquet_dataset import ParquetDataset, ParquetReader, ParquetWriter
from tzrec.datasets.utils import CKPT_ROW_IDX, CKPT_SOURCE_ID
from tzrec.features.feature import create_features
from tzrec.protos import data_pb2, feature_pb2
from tzrec.utils import misc_util


class ParquetDatasetTest(unittest.TestCase):
    @parameterized.expand([[5000], [10000]])
    def test_parquet_dataset(self, num_rows):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="id_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="tag_b")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="raw_c")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="raw_d", value_dim=2)
            ),
        ]
        features = create_features(feature_cfgs)

        t = pa.Table.from_arrays(
            [
                pa.array(["unused"] * num_rows),
                pa.array(["1"] * num_rows),
                pa.array(["2\x033"] * num_rows),
                pa.array([4] * num_rows),
                pa.array([[5.0, 6.0]] * num_rows, type=pa.list_(pa.float32())),
                pa.array([0] * num_rows),
            ],
            names=["unused", "id_a", "tag_b", "raw_c", "raw_d", "label"],
        )
        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            ds.write_dataset(
                t,
                test_dir,
                format="parquet",
                max_rows_per_file=5000,
                max_rows_per_group=5000,
            )

            dataset = ParquetDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=4,
                    dataset_type=data_pb2.DatasetType.ParquetDataset,
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                ),
                features=features,
                input_path=f"{test_dir}/*.parquet",
            )
            self.assertEqual(len(dataset.input_fields), 5)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=None,
                num_workers=2,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            iterator = iter(dataloader)
            for _ in range(10):
                data = next(iterator)
            self.assertEqual(
                sorted(data.to_dict().keys()),
                [
                    "id_a.lengths",
                    "id_a.values",
                    "label",
                    "raw_c.values",
                    "raw_d.values",
                    "tag_b.lengths",
                    "tag_b.values",
                ],
            )


class ParquetReaderTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @parameterized.expand([[True], [False]])
    def test_parquet_reader(self, rebalance):
        def _reader_worker(rank, port):
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(2)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group(backend="gloo")
            reader = ParquetReader(
                os.path.join(self.test_dir, "*.parquet"),
                batch_size=8192,
                rebalance=rebalance,
            )
            total_cnt = 0
            for batch in reader.to_batches(rank, 2):
                total_cnt += len(batch["id_a"])
            if rank == 0:
                assert total_cnt == 3000 if rebalance else 5000
            else:
                assert total_cnt == 3000 if rebalance else 1000

        for i, num_rows in enumerate([5000, 1000]):
            t = pa.Table.from_arrays(
                [
                    pa.array(["unused"] * num_rows),
                    pa.array(["1"] * num_rows),
                    pa.array(["2\x033"] * num_rows),
                    pa.array([4] * num_rows),
                    pa.array([[5.0, 6.0]] * num_rows, type=pa.list_(pa.float32())),
                    pa.array([0] * num_rows),
                ],
                names=["unused", "id_a", "tag_b", "raw_c", "raw_d", "label"],
            )
            writer = parquet.ParquetWriter(
                os.path.join(self.test_dir, f"part-{i}.parquet"), schema=t.schema
            )
            writer.write_table(t)
            writer.close()

        world_size = 2
        port = misc_util.get_free_port()
        procs = []
        for rank in range(world_size):
            p = mp.Process(
                target=_reader_worker,
                args=(rank, port),
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"reader worker-{i} failed.")


class ParquetWriterTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parquet_writer(self):
        def _writer_worker(rank):
            os.environ["RANK"] = str(rank)
            writer = ParquetWriter(self.test_dir)
            for _ in range(5):
                writer.write(
                    {
                        "int_a": pa.array(np.random.randint(0, 100, 128)),
                        "float_b": pa.array(np.random.random(128)),
                    }
                )
            writer.close()

        world_size = 2
        procs = []
        for rank in range(world_size):
            p = mp.Process(
                target=_writer_worker,
                args=(rank,),
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"writer worker-{i} failed.")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "part-0.parquet")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "part-1.parquet")))


class ParquetReaderCheckpointTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_data(self, num_rows=1000):
        """Create test parquet data."""
        t = pa.Table.from_arrays(
            [
                pa.array(["1"] * num_rows),
                pa.array([i for i in range(num_rows)]),
            ],
            names=["id_a", "row_idx"],
        )
        writer = parquet.ParquetWriter(
            os.path.join(self.test_dir, "part-0.parquet"), schema=t.schema
        )
        writer.write_table(t)
        writer.close()

    def test_parquet_reader_checkpoint_injection(self):
        """Test that checkpoint metadata columns are injected."""
        self._create_test_data(num_rows=100)

        reader = ParquetReader(
            os.path.join(self.test_dir, "*.parquet"),
            batch_size=32,
            rebalance=True,
        )
        batch_count = 0
        for batch in reader.to_batches(worker_id=0, num_workers=1):
            # Check that checkpoint columns exist
            self.assertIn(CKPT_SOURCE_ID, batch)
            self.assertIn(CKPT_ROW_IDX, batch)

            # Check source_id format: {input_path}:{start}
            source_ids = batch[CKPT_SOURCE_ID].to_pylist()
            self.assertTrue(all(":" in sid for sid in source_ids))

            # Check row indices are monotonically increasing within batch
            row_idxs = batch[CKPT_ROW_IDX].to_pylist()
            self.assertEqual(row_idxs, sorted(row_idxs))

            batch_count += 1

        self.assertGreater(batch_count, 0)

    def test_parquet_reader_checkpoint_source_id_format(self):
        """Test that source_id format is {input_path}:{start}."""
        self._create_test_data(num_rows=100)

        input_path = os.path.join(self.test_dir, "*.parquet")
        reader = ParquetReader(
            input_path,
            batch_size=32,
            rebalance=True,
        )

        for batch in reader.to_batches(worker_id=0, num_workers=1):
            source_ids = batch[CKPT_SOURCE_ID].to_pylist()
            for source_id in source_ids:
                # Verify format: path:start
                self.assertIn(":", source_id)
                parts = source_id.rsplit(":", 1)
                self.assertEqual(len(parts), 2)
                # start should be numeric
                self.assertTrue(parts[1].isdigit())
            break  # Only need to check first batch

    def test_parquet_reader_resume_same_topology(self):
        """Test resume with same num_workers."""
        self._create_test_data(num_rows=100)

        input_path = os.path.join(self.test_dir, "*.parquet")

        # First read: get checkpoint at row 49
        reader1 = ParquetReader(input_path, batch_size=50, rebalance=True)
        first_batch = None
        for batch in reader1.to_batches(worker_id=0, num_workers=1):
            first_batch = batch
            break

        self.assertIsNotNone(first_batch)
        max_row_idx = max(first_batch[CKPT_ROW_IDX].to_pylist())
        source_id = first_batch[CKPT_SOURCE_ID].to_pylist()[0]

        # Create checkpoint state
        checkpoint_state = {source_id: max_row_idx}

        # Resume with same topology
        reader2 = ParquetReader(input_path, batch_size=50, rebalance=True)
        reader2.set_checkpoint_state(checkpoint_state)

        resumed_rows = 0
        for batch in reader2.to_batches(worker_id=0, num_workers=1):
            # All rows should be after the checkpoint
            row_idxs = batch[CKPT_ROW_IDX].to_pylist()
            self.assertTrue(all(idx > max_row_idx for idx in row_idxs))
            resumed_rows += len(batch["id_a"])

        # Should have read remaining rows
        self.assertEqual(resumed_rows, 50)  # 100 - 50 already consumed

    def test_parquet_reader_resume_changed_topology(self):
        """Test resume with different num_workers."""

        def _reader_worker(test_dir, rank, port, world_size, checkpoint_state=None):
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group(backend="gloo")

            input_path = os.path.join(test_dir, "*.parquet")
            reader = ParquetReader(input_path, batch_size=25, rebalance=True)
            if checkpoint_state:
                reader.set_checkpoint_state(checkpoint_state)

            total_rows = 0
            for batch in reader.to_batches(worker_id=rank, num_workers=world_size):
                total_rows += len(batch["id_a"])
            return total_rows

        # Create test data
        self._create_test_data(num_rows=100)
        input_path = os.path.join(self.test_dir, "*.parquet")

        # First read without checkpoint
        reader = ParquetReader(input_path, batch_size=50, rebalance=True)
        first_batch = None
        for batch in reader.to_batches(worker_id=0, num_workers=1):
            first_batch = batch
            break

        self.assertIsNotNone(first_batch)
        max_row_idx = max(first_batch[CKPT_ROW_IDX].to_pylist())
        source_id = first_batch[CKPT_SOURCE_ID].to_pylist()[0]
        checkpoint_state = {source_id: max_row_idx}

        # Resume with different topology (2 workers instead of 1)
        reader2 = ParquetReader(input_path, batch_size=25, rebalance=True)
        reader2.set_checkpoint_state(checkpoint_state)

        total_resumed = 0
        for batch in reader2.to_batches(worker_id=0, num_workers=2):
            total_resumed += len(batch["id_a"])

        # Worker 0 should get half of remaining rows (approximately)
        self.assertGreater(total_resumed, 0)
        self.assertLessEqual(total_resumed, 50)  # Can't be more than remaining


if __name__ == "__main__":
    unittest.main()
