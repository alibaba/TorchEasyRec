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
from tzrec.features.feature import create_features
from tzrec.protos import data_pb2, feature_pb2
from tzrec.utils import misc_util
from tzrec.utils.checkpoint_util import update_dataloder_state


class ParquetDatasetTest(unittest.TestCase):
    def _create_test_parquet_data(self, test_dir: str, num_rows: int = 1000) -> None:
        """Create test parquet data files."""
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
        ds.write_dataset(
            t,
            test_dir,
            format="parquet",
            max_rows_per_file=5000,
            max_rows_per_group=5000,
        )

    def _create_feature_cfgs(self):
        """Create feature configs for testing."""
        return [
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

    @parameterized.expand([[5000], [10000]])
    def test_parquet_dataset(self, num_rows):
        feature_cfgs = self._create_feature_cfgs()
        features = create_features(feature_cfgs)

        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._create_test_parquet_data(test_dir, num_rows)

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

    def test_parquet_dataset_checkpoint_metadata(self):
        """Test that checkpoint_info is present and has correct format."""
        feature_cfgs = self._create_feature_cfgs()
        features = create_features(feature_cfgs)

        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._create_test_parquet_data(test_dir, num_rows=1000)
            input_path = f"{test_dir}/*.parquet"

            dataset = ParquetDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=128,
                    dataset_type=data_pb2.DatasetType.ParquetDataset,
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                ),
                features=features,
                input_path=input_path,
            )
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=None,
                num_workers=2,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            iterator = iter(dataloader)
            batch = next(iterator)

            # Verify checkpoint_info is present and has correct format
            self.assertIsNotNone(batch.checkpoint_info)
            self.assertIsInstance(batch.checkpoint_info, dict)

            # Checkpoint keys should be in format "{input_path}:{start}"
            for key, value in batch.checkpoint_info.items():
                self.assertIn(":", key)
                parts = key.rsplit(":", 1)
                self.assertEqual(len(parts), 2)
                # start should be numeric
                self.assertTrue(parts[1].isdigit())
                # Value should be a non-negative integer
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)

    def test_parquet_dataset_checkpoint_resume(self):
        """Test checkpoint resume functionality."""
        feature_cfgs = self._create_feature_cfgs()
        features = create_features(feature_cfgs)

        with tempfile.TemporaryDirectory(prefix="tzrec_") as test_dir:
            self._create_test_parquet_data(test_dir, num_rows=1000)
            input_path = f"{test_dir}/*.parquet"

            # First, read some batches and capture checkpoint info
            dataset1 = ParquetDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=128,
                    dataset_type=data_pb2.DatasetType.ParquetDataset,
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                ),
                features=features,
                input_path=input_path,
            )
            dataloader1 = DataLoader(
                dataset=dataset1,
                batch_size=None,
                num_workers=2,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            iterator1 = iter(dataloader1)

            # Read first batch and capture checkpoint
            checkpoint_state_acc = {}
            batch1 = next(iterator1)
            first_checkpoint_state = batch1.checkpoint_info.copy()
            update_dataloder_state(checkpoint_state_acc, first_checkpoint_state)
            self.assertIsNotNone(first_checkpoint_state)
            self.assertGreater(len(first_checkpoint_state), 0)

            # Read more batches
            for _ in range(3):
                batch1 = next(iterator1)
                update_dataloder_state(
                    checkpoint_state_acc, batch1.checkpoint_info.copy()
                )
            del iterator1
            del dataloader1

            # Now create a new dataset with checkpoint state set
            dataset2 = ParquetDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=128,
                    dataset_type=data_pb2.DatasetType.ParquetDataset,
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                ),
                features=features,
                input_path=input_path,
            )
            dataloader2 = DataLoader(
                dataset=dataset2,
                batch_size=None,
                num_workers=2,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            # Set first batch checkpoint state to resume from saved offsets
            dataloader2.dataset.load_state_dict(first_checkpoint_state)
            iterator2 = iter(dataloader2)

            # Read batch from resumed dataset
            batch2 = next(iterator2)

            # The resumed batch should have offsets greater than the checkpoint
            for key, new_offset in batch2.checkpoint_info.items():
                if key in first_checkpoint_state:
                    # New offset should be greater than first checkpoint offset
                    self.assertGreater(new_offset, first_checkpoint_state[key])
                if key in checkpoint_state_acc:
                    # New offset should be less than or equal to acc checkpoint offset
                    self.assertLessEqual(new_offset, checkpoint_state_acc[key])


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


if __name__ == "__main__":
    unittest.main()
