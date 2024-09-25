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


import tempfile
import unittest
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pyarrow as pa
from graphlearn.python.nn.pytorch.data import utils
from parameterized import parameterized
from torch.utils.data import DataLoader

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader
from tzrec.datasets.utils import BASE_DATA_GROUP, NEG_DATA_GROUP
from tzrec.features.feature import BaseFeature, create_features
from tzrec.protos import data_pb2, feature_pb2, sampler_pb2


class _TestReader(BaseReader):
    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        input_fields: Optional[List[pa.Field]] = None,
    ) -> None:
        super().__init__(input_path, batch_size, selected_cols)
        self.input_fields = input_fields or []

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        for _ in range(100):
            input_data = {}
            for f in self.input_fields:
                if f.type == pa.int32():
                    data = np.random.randint(
                        100, size=(self._batch_size,), dtype=np.int32
                    )
                elif f.type == pa.int64():
                    data = np.random.randint(
                        100, size=(self._batch_size,), dtype=np.int64
                    )
                elif f.type == pa.float32():
                    data = np.random.rand(self._batch_size).astype(np.float32)
                elif f.type == pa.float64():
                    data = np.random.rand(self._batch_size)
                elif f.type == pa.string():
                    data = np.random.randint(100, size=(self._batch_size,)).astype(
                        np.str_
                    )
                else:
                    raise ValueError(f"Unknown input_type {f.input_type}")
                input_data[f.name] = pa.array(data)
            yield input_data


class _TestDataset(BaseDataset):
    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        input_fields: Optional[List[pa.Field]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_config, features, input_path, **kwargs)
        self._input_fields = input_fields
        self._reader = _TestReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names),
            self.input_fields,
        )


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self._temp_files = []

    def tearDown(self):
        for f in self._temp_files:
            f.close()
        utils.SERVER_LAUNCHED = False
        del utils.STATS_DICT
        utils.STATS_DICT = []

    def test_dataset(self):
        input_fields = [
            pa.field(name="int_a", type=pa.int64()),
            pa.field(name="float_b", type=pa.float64()),
            pa.field(name="str_c", type=pa.string()),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="str_c")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_b")
            ),
        ]
        features = create_features(feature_cfgs)

        dataloader = DataLoader(
            dataset=_TestDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=4,
                    dataset_type=data_pb2.DatasetType.OdpsDataset,
                    fg_encoded=True,
                    label_fields=["label"],
                ),
                features=features,
                input_path="",
                input_fields=input_fields,
            ),
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        self.assertEqual(batch.dense_features[BASE_DATA_GROUP].keys(), ["float_b"])
        self.assertEqual(batch.dense_features[BASE_DATA_GROUP].values().size(), (4, 1))
        self.assertEqual(
            batch.sparse_features[BASE_DATA_GROUP].keys(), ["int_a", "str_c"]
        )
        self.assertEqual(batch.sparse_features[BASE_DATA_GROUP].values().size(), (8,))
        self.assertEqual(batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (8,))
        self.assertEqual(batch.labels["label"].size(), (4,))

    @parameterized.expand([[False], [True]])
    def test_dataset_with_sampler(self, force_base_data_group):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            f.write(f"{i}\t{1}\t{i}:{i+1000}:{i+2000}\n")
        f.flush()

        input_fields = [
            pa.field(name="int_a", type=pa.int64()),
            pa.field(name="float_b", type=pa.float64()),
            pa.field(name="str_c", type=pa.string()),
            pa.field(name="int_d", type=pa.int64()),
            pa.field(name="float_d", type=pa.float64()),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="str_c")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_b")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_d")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_d")
            ),
        ]
        features = create_features(
            feature_cfgs,
            neg_fields=["int_a", "float_b", "str_c"],
            force_base_data_group=force_base_data_group,
        )

        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_encoded=True,
                label_fields=["label"],
                negative_sampler=sampler_pb2.NegativeSampler(
                    input_path=f.name,
                    num_sample=8,
                    attr_fields=["int_a", "float_b", "str_c"],
                    item_id_field="int_a",
                ),
                force_base_data_group=force_base_data_group,
            ),
            features=features,
            input_path="",
            input_fields=input_fields,
        )
        dataset.launch_sampler_cluster(2)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        if not force_base_data_group:
            self.assertEqual(batch.dense_features[BASE_DATA_GROUP].keys(), ["float_d"])
            self.assertEqual(
                batch.dense_features[BASE_DATA_GROUP].values().size(), (4, 1)
            )
            self.assertEqual(batch.sparse_features[BASE_DATA_GROUP].keys(), ["int_d"])
            self.assertEqual(
                batch.sparse_features[BASE_DATA_GROUP].values().size(), (4,)
            )
            self.assertEqual(
                batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (4,)
            )
            self.assertEqual(batch.dense_features[NEG_DATA_GROUP].keys(), ["float_b"])
            self.assertEqual(
                batch.dense_features[NEG_DATA_GROUP].values().size(), (12, 1)
            )
            self.assertEqual(
                batch.sparse_features[NEG_DATA_GROUP].keys(), ["int_a", "str_c"]
            )
            self.assertEqual(
                batch.sparse_features[NEG_DATA_GROUP].values().size(), (24,)
            )
            self.assertEqual(
                batch.sparse_features[NEG_DATA_GROUP].lengths().size(), (24,)
            )
        else:
            self.assertEqual(
                batch.dense_features[BASE_DATA_GROUP].keys(), ["float_b", "float_d"]
            )
            self.assertEqual(
                batch.dense_features[BASE_DATA_GROUP].values().size(), (12, 2)
            )
            self.assertEqual(
                batch.sparse_features[BASE_DATA_GROUP].keys(),
                ["int_a", "str_c", "int_d"],
            )
            self.assertEqual(
                batch.sparse_features[BASE_DATA_GROUP].values().size(), (28,)
            )
            self.assertEqual(
                batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (36,)
            )
        self.assertEqual(batch.labels["label"].size(), (4,))

    def test_dataset_with_sample_mask(self):
        input_fields = [
            pa.field(name="int_a", type=pa.int64()),
            pa.field(name="float_b", type=pa.float64()),
            pa.field(name="str_c", type=pa.string()),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="int_a", use_mask=True, fg_encoded_default_value=""
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="str_c")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_b")
            ),
        ]
        features = create_features(feature_cfgs)

        dataloader = DataLoader(
            dataset=_TestDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=32,
                    dataset_type=data_pb2.DatasetType.OdpsDataset,
                    fg_encoded=True,
                    label_fields=["label"],
                    sample_mask_prob=0.4,
                ),
                features=features,
                input_path="",
                mode=Mode.TRAIN,
                input_fields=input_fields,
            ),
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        data_dict = batch.to_dict()
        self.assertLess(len(data_dict["int_a.values"]), 32)
        self.assertEqual(len(data_dict["float_b.values"]), 32)

    def test_dataset_with_neg_sample_mask(self):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            f.write(f"{i}\t{1}\t{i}:{i+1000}:{i+2000}\n")
        f.flush()

        input_fields = [
            pa.field(name="int_a", type=pa.int64()),
            pa.field(name="float_b", type=pa.float64()),
            pa.field(name="str_c", type=pa.string()),
            pa.field(name="int_d", type=pa.int64()),
            pa.field(name="float_d", type=pa.float64()),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="str_c", use_mask=True, fg_encoded_default_value=""
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_b")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_d")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_d")
            ),
        ]
        features = create_features(
            feature_cfgs, neg_fields=["int_a", "float_b", "str_c"]
        )

        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=32,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_encoded=True,
                label_fields=["label"],
                negative_sample_mask_prob=0.4,
                negative_sampler=sampler_pb2.NegativeSampler(
                    input_path=f.name,
                    num_sample=32,
                    attr_fields=["int_a", "float_b", "str_c"],
                    item_id_field="int_a",
                ),
            ),
            features=features,
            input_path="",
            mode=Mode.TRAIN,
            input_fields=input_fields,
        )
        dataset.launch_sampler_cluster(2)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        data_dict = batch.to_dict()
        self.assertLess(len(data_dict["str_c.values"]), 64)
        self.assertGreater(len(data_dict["str_c.values"]), 32)
        self.assertEqual(len(data_dict["float_b.values"]), 64)

    @parameterized.expand([[True], [False]])
    def test_dataset_predict_mode(self, debug_level):
        input_fields = [
            pa.field(name="int_a", type=pa.int64()),
            pa.field(name="float_b", type=pa.float64()),
            pa.field(name="str_c", type=pa.string()),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="str_c")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_b")
            ),
        ]
        features = create_features(feature_cfgs)

        dataloader = DataLoader(
            dataset=_TestDataset(
                data_config=data_pb2.DataConfig(
                    batch_size=4,
                    dataset_type=data_pb2.DatasetType.OdpsDataset,
                    fg_encoded=True,
                    label_fields=[],
                ),
                features=features,
                input_path="",
                reserved_columns=["label"],
                mode=Mode.PREDICT,
                input_fields=input_fields,
                debug_level=debug_level,
            ),
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        if debug_level > 0:
            self.assertEqual(
                list(batch.reserves.get().column_names), ["label", "__features__"]
            )
        else:
            self.assertEqual(list(batch.reserves.get().column_names), ["label"])


if __name__ == "__main__":
    unittest.main()
