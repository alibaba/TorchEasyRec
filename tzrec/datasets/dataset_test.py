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


import math
import os
import tempfile
import unittest
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pyarrow as pa
from graphlearn.python.nn.pytorch.data import utils
from parameterized import parameterized
from torch.utils.data import DataLoader

from tzrec.constant import Mode
from tzrec.datasets.dataset import BaseDataset, BaseReader
from tzrec.datasets.utils import (
    BASE_DATA_GROUP,
    CAND_POS_LENGTHS,
    HARD_NEG_INDICES,
    NEG_DATA_GROUP,
)
from tzrec.features.feature import BaseFeature, create_features
from tzrec.protos import data_pb2, feature_pb2, sampler_pb2


class _TestReader(BaseReader):
    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        input_fields: Optional[List[pa.Field]] = None,
        sample_cost_field: Optional[str] = None,
        batch_cost_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_path,
            batch_size,
            selected_cols,
            sample_cost_field=sample_cost_field,
            batch_cost_size=batch_cost_size,
        )
        self.input_fields = input_fields or []

    def _reader(self) -> Iterator[pa.RecordBatch]:
        for _ in range(100):
            input_data = OrderedDict()
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
                elif f.type == pa.string():
                    data = np.random.randint(100, size=(self._batch_size,)).astype(
                        np.str_
                    )
                elif pa.types.is_list(f.type):
                    if f.type.value_type == pa.string():
                        data = [
                            [str(np.random.randint(100)) for _ in range(2)]
                            for _ in range(self._batch_size)
                        ]
                    elif f.type.value_type == pa.int64():
                        data = [
                            [int(np.random.randint(100)) for _ in range(2)]
                            for _ in range(self._batch_size)
                        ]
                    elif f.type.value_type == pa.int32():
                        data = [
                            [int(np.random.randint(100)) for _ in range(2)]
                            for _ in range(self._batch_size)
                        ]
                    elif f.type.value_type == pa.float32():
                        data = [
                            [float(np.random.rand()) for _ in range(2)]
                            for _ in range(self._batch_size)
                        ]
                    elif f.type.value_type == pa.float64():
                        data = [
                            [float(np.random.rand()) for _ in range(2)]
                            for _ in range(self._batch_size)
                        ]
                    else:
                        raise ValueError(f"Unsupported list type: {f.type}")

                elif pa.types.is_list(f.type) and pa.types.is_list(f.type.value_type):
                    if f.type.value_type.value_type == pa.string():
                        data = [
                            [
                                [str(np.random.randint(100)) for _ in range(2)],
                                [str(np.random.randint(100)) for _ in range(2)],
                            ]
                            for _ in range(self._batch_size)
                        ]
                    else:
                        raise ValueError(f"Unsupported nested list type: {f.type}")
                else:
                    raise ValueError(f"Unknown input_type {f.type}")

                input_data[f.name] = pa.array(data)

            if self._sample_cost_field is not None and len(self._sample_cost_field) > 0:
                input_data[self._sample_cost_field] = pa.array(
                    list(range(self._batch_size)), type=pa.int64()
                )
            yield pa.RecordBatch.from_arrays(
                list(input_data.values()), names=list(input_data.keys())
            )

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        yield from self._arrow_reader_iter(self._reader())


class _TestDataset(BaseDataset):
    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        input_fields: Optional[List[pa.Field]] = None,
        sample_cost_field: Optional[str] = None,
        batch_cost_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_config, features, input_path, **kwargs)
        self._input_fields = input_fields
        self._reader = _TestReader(
            input_path,
            self._batch_size,
            list(self._selected_input_names),
            self.input_fields,
            sample_cost_field=sample_cost_field,
            batch_cost_size=batch_cost_size,
        )


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self._temp_files = []

    def tearDown(self):
        os.environ.pop("INPUT_TILE", None)
        for f in self._temp_files:
            f.close()
        utils.SERVER_LAUNCHED = False
        del utils.STATS_DICT
        utils.STATS_DICT = []

    @parameterized.expand([[False], [True]])
    def test_dataset(self, need_shuffle):
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
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                    shuffle=need_shuffle,
                ),
                features=features,
                input_path="",
                input_fields=input_fields,
                mode=Mode.TRAIN,
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

    @parameterized.expand(
        [
            [False, Mode.EVAL, False],
            [True, Mode.EVAL, False],
            [False, Mode.PREDICT, False],
            [True, Mode.PREDICT, False],
            [False, Mode.PREDICT, True],
            [True, Mode.PREDICT, True],
        ]
    )
    def test_dataset_with_sampler(self, force_base_data_group, mode, input_tile):
        if input_tile:
            os.environ["INPUT_TILE"] = "2"
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            f.write(f"{i}\t{1}\t{i}:{i + 1000}:{i + 2000}\n")
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
                id_feature=feature_pb2.IdFeature(
                    feature_name="int_a", expression="item:int_a", num_buckets=2000
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="str_c", expression="item:str_c", num_buckets=3000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="float_b", expression="item:float_b"
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="int_d", expression="user:int_d", num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name="float_d", expression="user:float_d"
                )
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
                fg_mode=data_pb2.FgMode.FG_NORMAL,
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
            mode=mode,
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
        if input_tile:
            if not force_base_data_group:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP + "_user"].keys(), ["float_d"]
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP + "_user"].keys(), ["int_d"]
                )
                self.assertEqual(
                    batch.dense_features[NEG_DATA_GROUP].keys(), ["float_b"]
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].keys(), ["int_a", "str_c"]
                )
            else:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP + "_user"].keys(), ["float_d"]
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP + "_user"].keys(), ["int_d"]
                )
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].keys(), ["float_b"]
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].keys(), ["int_a", "str_c"]
                )
        else:
            if not force_base_data_group:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].keys(), ["float_d"]
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].keys(), ["int_d"]
                )
                self.assertEqual(
                    batch.dense_features[NEG_DATA_GROUP].keys(), ["float_b"]
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].keys(), ["int_a", "str_c"]
                )
            else:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].keys(), ["float_b", "float_d"]
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].keys(),
                    ["int_a", "str_c", "int_d"],
                )

        if mode == Mode.EVAL:
            if not force_base_data_group:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].values().size(), (4, 1)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].values().size(), (4,)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (4,)
                )
                self.assertEqual(
                    batch.dense_features[NEG_DATA_GROUP].values().size(), (12, 1)
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].values().size(), (24,)
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].lengths().size(), (24,)
                )
            else:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].values().size(), (12, 2)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].values().size(), (28,)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (36,)
                )
            self.assertEqual(batch.labels["label"].size(), (4,))
        elif input_tile:
            if not force_base_data_group:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP + "_user"].values().size(),
                    (4, 1),
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP + "_user"].values().size(),
                    (4,),
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP + "_user"].lengths().size(),
                    (4,),
                )
                self.assertEqual(
                    batch.dense_features[NEG_DATA_GROUP].values().size(), (4, 1)
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].values().size(), (8,)
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].lengths().size(), (8,)
                )
            else:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP + "_user"].values().size(),
                    (4, 1),
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP + "_user"].values().size(),
                    (4,),
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP + "_user"].lengths().size(),
                    (4,),
                )
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].values().size(), (4, 1)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].values().size(), (8,)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (8,)
                )
        else:
            if not force_base_data_group:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].values().size(), (4, 1)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].values().size(), (4,)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (4,)
                )
                self.assertEqual(
                    batch.dense_features[NEG_DATA_GROUP].values().size(), (4, 1)
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].values().size(), (8,)
                )
                self.assertEqual(
                    batch.sparse_features[NEG_DATA_GROUP].lengths().size(), (8,)
                )
            else:
                self.assertEqual(
                    batch.dense_features[BASE_DATA_GROUP].values().size(), (4, 2)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].values().size(), (12,)
                )
                self.assertEqual(
                    batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (12,)
                )

    @parameterized.expand(
        [
            [Mode.TRAIN],
            [Mode.EVAL],
        ]
    )
    def test_dataset_with_sampler_list_item_id(self, mode):
        """E2E: list-typed item_id positives through a real NegativeSampler.

        Schema declares `cand_seq__item_id` (grouped sequence sub-feature
        flattened name) as `pa.list_(pa.int64())` (multi-positive column).
        Exercises `build_sampler_input`'s list-pass-through + flatten,
        the dynamic-`expand_factor` path in the sampler, and
        `combine_negs_to_candidate_sequence`'s list-typed-negs
        normalization. Parameterized over Mode.TRAIN / Mode.EVAL so the
        eval-mode `num_eval_sample` path is also covered.
        """
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            f.write(f"{i}\t{1}\t{i}\n")
        f.flush()

        input_fields = [
            pa.field(name="cand_seq__item_id", type=pa.list_(pa.int64())),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="cand_seq",
                    sequence_length=10,
                    sequence_delim=";",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="item_id",
                                expression="item:item_id",
                                num_buckets=200,
                                embedding_dim=8,
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(
            feature_cfgs,
            neg_fields=["item_id"],
            force_base_data_group=True,
        )

        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=data_pb2.FgMode.FG_NORMAL,
                label_fields=["label"],
                negative_sampler=sampler_pb2.NegativeSampler(
                    input_path=f.name,
                    num_sample=8,
                    num_eval_sample=4,
                    attr_fields=["item_id"],  # bare; gets rewritten
                    item_id_field="cand_seq__item_id",  # qualified
                ),
                force_base_data_group=True,
            ),
            features=features,
            input_path="",
            input_fields=input_fields,
            mode=mode,
        )
        dataset.launch_sampler_cluster(2)

        # Multi-positive mode (item_id is sequence-positive in train):
        # launch_sampler_cluster strips the outer list so the sampler emits
        # scalar item_id negs, not 1-elem lists via the multival_sep round-trip.
        item_id_idx = dataset._sampler._attr_names.index("cand_seq__item_id")
        self.assertEqual(dataset._sampler._attr_types[item_id_idx], pa.int64())

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)

        # _TestReader yields 2 list elements per row; batch_size = 4 -> K_i = 2.
        pos_lengths = batch.additional_infos[CAND_POS_LENGTHS]
        self.assertEqual(pos_lengths.tolist(), [2, 2, 2, 2])

    @parameterized.expand(
        [
            [Mode.TRAIN],
            [Mode.EVAL],
        ]
    )
    def test_dataset_with_hard_negative_sampler_list_item_id(self, mode):
        """E2E: HardNegativeSampler with list-typed item_id (Q != B contract).

        Locks the contract documented in sampler.py:660 -- after
        `build_sampler_input` flattens multi-positive queries,
        `HARD_NEG_INDICES[:, 0]` indexes into the flat Q-length src_ids,
        not the original B-length batch rows.
        """
        # Item file: ids 0..99 + canonical attrs.
        item_f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(item_f)
        item_f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            item_f.write(f"{i}\t{1}\t{i}\n")
        item_f.flush()

        # User file: ids 0..99.
        user_f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(user_f)
        user_f.write("id:int64\tweight:float\n")
        for i in range(100):
            user_f.write(f"{i}\t{1}\n")
        user_f.flush()

        # Hard-neg edges: every user u has 2 hard-neg items so any
        # sampled src_id returns at least one hard-neg row.
        hard_neg_f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(hard_neg_f)
        hard_neg_f.write("userid:int64\titemid:int64\tweight:float\n")
        for u in range(100):
            for offset in (50, 51):
                hard_neg_f.write(f"{u}\t{(u + offset) % 100}\t{1}\n")
        hard_neg_f.flush()

        input_fields = [
            pa.field(name="user_id", type=pa.int64()),
            pa.field(name="cand_seq__item_id", type=pa.list_(pa.int64())),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    num_buckets=200,
                    embedding_dim=8,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="cand_seq",
                    sequence_length=10,
                    sequence_delim=";",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="item_id",
                                expression="item:item_id",
                                num_buckets=200,
                                embedding_dim=8,
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(
            feature_cfgs,
            neg_fields=["item_id"],
            force_base_data_group=True,
        )

        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=data_pb2.FgMode.FG_NORMAL,
                label_fields=["label"],
                hard_negative_sampler=sampler_pb2.HardNegativeSampler(
                    user_input_path=user_f.name,
                    item_input_path=item_f.name,
                    hard_neg_edge_input_path=hard_neg_f.name,
                    num_sample=8,
                    num_eval_sample=4,
                    num_hard_sample=2,
                    attr_fields=["item_id"],  # bare; gets rewritten
                    item_id_field="cand_seq__item_id",  # qualified
                    user_id_field="user_id",
                ),
                force_base_data_group=True,
            ),
            features=features,
            input_path="",
            input_fields=input_fields,
            mode=mode,
        )
        dataset.launch_sampler_cluster(2)

        # Multi-positive mode: outer list stripped so sampler emits scalar negs.
        item_id_idx = dataset._sampler._attr_names.index("cand_seq__item_id")
        self.assertEqual(dataset._sampler._attr_types[item_id_idx], pa.int64())

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)

        # _TestReader yields K_i = 2 per row; batch_size = 4 -> Q = 8.
        pos_lengths = batch.additional_infos[CAND_POS_LENGTHS]
        self.assertEqual(pos_lengths.tolist(), [2, 2, 2, 2])

        # HARD_NEG_INDICES[:, 0] indexes into the flat Q=8 src_ids (post-flatten),
        # not the B=4 batch rows. Seed gives every flat position a hard-neg edge.
        hard_neg_indices = batch.additional_infos[HARD_NEG_INDICES]
        self.assertEqual(set(hard_neg_indices[:, 0].tolist()), {0, 1, 2, 3, 4, 5, 6, 7})

    def test_launch_sampler_cluster_grouped_sequence_strip_and_rewrite(self):
        """Prefix-rewrite + outer-list strip for HSTUMatch-shaped configs.

        Bare ``attr_fields=["cat_key"]`` is rewritten to
        ``["click_seq__cat_key"]``; strip drops the multi-positive outer
        list (``list<list<int64>>`` -> ``list<int64>``). ``cat_map``, a
        parquet column not in ``attr_fields``, is left untouched.
        """
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            f.write(f"{i}\t1.0\t{i}\n")
        f.flush()

        input_fields = [
            pa.field(name="cat_map", type=pa.list_(pa.string())),
            pa.field(name="click_seq__cat_key", type=pa.list_(pa.list_(pa.int64()))),
            pa.field(name="label", type=pa.int32()),
        ]
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    sequence_length=10,
                    sequence_delim=";",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            lookup_feature=feature_pb2.LookupFeature(
                                feature_name="lookup_c",
                                map="item:cat_map",
                                key="item:cat_key",
                                sequence_fields=["cat_key"],
                                num_buckets=10,
                                embedding_dim=8,
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(
            feature_cfgs,
            fg_mode=data_pb2.FgMode.FG_NORMAL,
            neg_fields=["cat_key"],
            force_base_data_group=True,
        )
        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=data_pb2.FgMode.FG_NORMAL,
                label_fields=["label"],
                negative_sampler=sampler_pb2.NegativeSampler(
                    input_path=f.name,
                    num_sample=4,
                    attr_fields=["cat_key"],  # bare; prefix-rewritten at launch
                    item_id_field="click_seq__cat_key",
                ),
                force_base_data_group=True,
            ),
            features=features,
            input_path="",
            input_fields=input_fields,
            mode=Mode.TRAIN,
        )
        self.assertEqual(dataset._sampler_seq_delim, ";")
        self.assertEqual(dataset._sampler_seq_prefix, "click_seq__")

        dataset.launch_sampler_cluster(2)
        # Deep-copy guard: data_config not mutated by the rewrite.
        self.assertEqual(
            list(dataset._data_config.negative_sampler.attr_fields), ["cat_key"]
        )
        self.assertIn("click_seq__cat_key", dataset._sampler._attr_names)
        self.assertNotIn("cat_key", dataset._sampler._attr_names)
        cat_key_idx = dataset._sampler._attr_names.index("click_seq__cat_key")
        self.assertEqual(
            dataset._sampler._attr_types[cat_key_idx], pa.list_(pa.int64())
        )
        self.assertNotIn("cat_map", dataset._sampler._attr_names)

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
                    fg_mode=data_pb2.FgMode.FG_NONE,
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
            f.write(f"{i}\t{1}\t{i}:{i + 1000}:{i + 2000}\n")
        f.flush()

        input_fields = [
            pa.field(name="item_id", type=pa.int64()),
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
                fg_mode=data_pb2.FgMode.FG_NONE,
                label_fields=["label"],
                negative_sample_mask_prob=0.4,
                negative_sampler=sampler_pb2.NegativeSampler(
                    input_path=f.name,
                    num_sample=32,
                    attr_fields=["int_a", "float_b", "str_c"],
                    item_id_field="item_id",
                ),
            ),
            features=features,
            input_path="",
            mode=Mode.TRAIN,
            input_fields=input_fields,
        )
        self.assertEqual(
            sorted(list(dataset._selected_input_names)),
            ["float_b", "float_d", "int_a", "int_d", "item_id", "label", "str_c"],
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
                    fg_mode=data_pb2.FgMode.FG_NONE,
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

    def test_dataset_with_tdm_sampler_and_remain_ratio(self):
        node = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(node)
        node.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(63):
            node.write(f"{i}\t{1}\t{int(math.log(i + 1, 2))}:{i}:{i + 1000}:{i * 2}\n")
        node.flush()

        def _ancestor(code):
            ancs = []
            while True:
                code = int((code - 1) / 2)
                if code <= 0:
                    break
                ancs.append(code)
            return ancs

        edge = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(edge)
        edge.write("src_id:int64\tdst_id:int\tweight:float\n")
        for i in range(31, 63):
            for ancestor in _ancestor(i):
                edge.write(f"{i}\t{ancestor}\t{1.0}\n")
        edge.flush()

        def _childern(code):
            return [2 * code + 1, 2 * code + 2]

        predict_edge = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(predict_edge)
        predict_edge.write("src_id:int64\tdst_id:int\tweight:float\n")
        for i in range(7, 15):
            predict_edge.write(f"0\t{i}\t{1}\n")
        for i in range(7, 31):
            for child in _childern(i):
                predict_edge.write(f"{i}\t{child}\t{1}\n")
        predict_edge.flush()

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
        features = create_features(feature_cfgs)

        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=data_pb2.FgMode.FG_NONE,
                label_fields=["label"],
                tdm_sampler=sampler_pb2.TDMSampler(
                    item_input_path=node.name,
                    edge_input_path=edge.name,
                    predict_edge_input_path=predict_edge.name,
                    attr_fields=["tree_level", "int_a", "float_b", "str_c"],
                    item_id_field="int_a",
                    layer_num_sample=[0, 1, 1, 1, 1, 5],
                    field_delimiter=",",
                    remain_ratio=0.4,
                    probability_type="UNIFORM",
                ),
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
        self.assertEqual(
            batch.dense_features[BASE_DATA_GROUP].keys(), ["float_b", "float_d"]
        )
        self.assertEqual(batch.dense_features[BASE_DATA_GROUP].values().size(), (40, 2))
        self.assertEqual(
            batch.sparse_features[BASE_DATA_GROUP].keys(),
            ["int_a", "str_c", "int_d"],
        )
        self.assertEqual(batch.sparse_features[BASE_DATA_GROUP].values().size(), (120,))
        self.assertEqual(
            batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (120,)
        )
        self.assertEqual(batch.labels["label"].size(), (40,))

    def test_dataset_with_tdm_sample_mask(self):
        node = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(node)
        node.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(63):
            node.write(f"{i}\t{1}\t{int(math.log(i + 1, 2))}:{i}:{i + 1000}:{i * 2}\n")
        node.flush()

        def _ancestor(code):
            ancs = []
            while True:
                code = int((code - 1) / 2)
                if code <= 0:
                    break
                ancs.append(code)
            return ancs

        edge = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(edge)
        edge.write("src_id:int64\tdst_id:int\tweight:float\n")
        for i in range(31, 63):
            for ancestor in _ancestor(i):
                edge.write(f"{i}\t{ancestor}\t{1.0}\n")
        edge.flush()

        def _childern(code):
            return [2 * code + 1, 2 * code + 2]

        predict_edge = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(predict_edge)
        predict_edge.write("src_id:int64\tdst_id:int\tweight:float\n")
        for i in range(7, 15):
            predict_edge.write(f"0\t{i}\t{1}\n")
        for i in range(7, 31):
            for child in _childern(i):
                predict_edge.write(f"{i}\t{child}\t{1}\n")
        predict_edge.flush()

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
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(feature_name="int_d")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="float_d")
            ),
        ]
        features = create_features(feature_cfgs)

        dataset = _TestDataset(
            data_config=data_pb2.DataConfig(
                batch_size=4,
                dataset_type=data_pb2.DatasetType.OdpsDataset,
                fg_mode=data_pb2.FgMode.FG_NONE,
                label_fields=["label"],
                sample_mask_prob=0.8,
                negative_sample_mask_prob=0.3,
                tdm_sampler=sampler_pb2.TDMSampler(
                    item_input_path=node.name,
                    edge_input_path=edge.name,
                    predict_edge_input_path=predict_edge.name,
                    attr_fields=["tree_level", "int_a", "float_b", "str_c"],
                    item_id_field="int_a",
                    layer_num_sample=[0, 1, 1, 1, 1, 5],
                    field_delimiter=",",
                    remain_ratio=0.4,
                    probability_type="UNIFORM",
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
        # After expand_tdm_sample, every feature has length 40
        # (batch_size=4 expanded by TDM sampling to 40 rows). float_b has
        # no use_mask so it is never masked. int_a has use_mask=True so the
        # high sample_mask_prob / negative_sample_mask_prob should null out
        # most rows, shrinking its values length well below 40.
        self.assertEqual(len(data_dict["float_b.values"]), 40)
        self.assertLess(len(data_dict["int_a.values"]), 40)

    def test_dataset_with_list_type_not_null(self):
        input_fields = [
            pa.field(name="int_a", type=pa.int64()),
            pa.field(name="float_b", type=pa.float64()),
            pa.field(name="str_c", type=pa.string()),
            pa.field(name="label", type=pa.int32()),
            pa.field(name="list_str_d", type=pa.list_(pa.string()), nullable=False),
            pa.field(
                name="list_list_str_e",
                type=pa.list_(pa.field("list_str_e", pa.string(), nullable=False)),
                nullable=False,
            ),
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
                raw_feature=feature_pb2.RawFeature(feature_name="list_str_d")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="list_list_str_e")
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
        self.assertIn("list_str_d", batch.dense_features[BASE_DATA_GROUP].keys())
        self.assertEqual(
            batch.dense_features[BASE_DATA_GROUP]["list_str_d"].size(), (4, 1)
        )
        self.assertIn("list_list_str_e", batch.dense_features[BASE_DATA_GROUP].keys())
        self.assertEqual(
            batch.dense_features[BASE_DATA_GROUP]["list_list_str_e"].size(), (4, 1)
        )

    def test_dataset_with_batch_cost(self):
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
                    fg_mode=data_pb2.FgMode.FG_NONE,
                    label_fields=["label"],
                ),
                features=features,
                input_path="",
                input_fields=input_fields,
                mode=Mode.TRAIN,
                sample_cost_field="sample_cost",
                batch_cost_size=4,
            ),
            batch_size=None,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        self.assertEqual(batch.sparse_features[BASE_DATA_GROUP].lengths().size(), (6,))


if __name__ == "__main__":
    unittest.main()
