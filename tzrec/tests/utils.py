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

import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.dataset as ds
import torch

from tzrec.acc.utils import is_aot_predict, is_trt_predict
from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.datasets.odps_dataset import _type_pa_to_table
from tzrec.features.combo_feature import ComboFeature
from tzrec.features.custom_feature import CustomFeature
from tzrec.features.expr_feature import ExprFeature
from tzrec.features.feature import BaseFeature, FgMode, create_features
from tzrec.features.id_feature import IdFeature
from tzrec.features.lookup_feature import LookupFeature
from tzrec.features.match_feature import MatchFeature
from tzrec.features.raw_feature import RawFeature
from tzrec.features.sequence_feature import (
    SequenceCustomFeature,
    SequenceIdFeature,
    SequenceRawFeature,
)
from tzrec.features.tokenize_feature import TokenizeFeature
from tzrec.protos import data_pb2
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util, env_util, misc_util


def _create_random_id_data(
    size: Tuple[int],
    num_ids: Optional[int] = None,
    vocab_list: Optional[List[str]] = None,
) -> npt.NDArray:
    if vocab_list:
        data = np.random.choice(vocab_list, size=size)
    else:
        data = np.random.randint(
            num_ids or 10000,
            size=size,
            dtype=np.int64,
        )
    return data


def _get_nproc_per_node() -> int:
    """Set nproc_per_node."""
    n_proc = int(os.getenv("TEST_NPROC_PER_NODE", "2"))
    return n_proc


class MockInput:
    """Mock input data base class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        raise NotImplementedError


class IdMockInput(MockInput):
    """Mock sparse id input data class."""

    def __init__(
        self,
        name: str,
        is_multi: bool = False,
        num_ids: Optional[int] = None,
        vocab_list: Optional[List[str]] = None,
        multival_sep: str = chr(3),
    ) -> None:
        super().__init__(name)
        self.is_multi = is_multi
        self.num_ids = num_ids
        self.vocab_list = vocab_list
        self.multival_sep = multival_sep

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        if not self.is_multi:
            # int64
            num_valid_rows = (
                random.randint(num_rows // 2, num_rows) if has_null else num_rows
            )
            data = list(
                _create_random_id_data((num_valid_rows,), self.num_ids, self.vocab_list)
            )
            data = data + [None] * (num_rows - num_valid_rows)
            random.shuffle(data)
        else:
            # string
            num_multi_rows = random.randint(num_rows // 3, 2 * num_rows // 3)
            num_multi_id = 2
            data_multi = _create_random_id_data(
                (num_multi_rows, num_multi_id), self.num_ids, self.vocab_list
            ).astype(str)
            data_multi = list(map(lambda x: self.multival_sep.join(x), data_multi))
            num_single_rows = (
                random.randint(num_rows // 3, num_rows - num_multi_rows)
                if has_null
                else num_rows - num_multi_rows
            )
            data_single = list(
                _create_random_id_data(
                    (num_single_rows,), self.num_ids, self.vocab_list
                ).astype(str)
            )
            data = (
                data_multi
                + data_single
                + [None] * (num_rows - num_multi_rows - num_single_rows)
            )
            random.shuffle(data)
        return pa.array(data)


class HSTUIdMockInput(MockInput):
    """Mock sparse id input data class."""

    def __init__(
        self,
        name: str,
        is_multi: bool = False,
        num_ids: Optional[int] = None,
        vocab_list: Optional[List[str]] = None,
        multival_sep: str = chr(3),
    ) -> None:
        super().__init__(name)
        self.is_multi = is_multi
        self.num_ids = num_ids
        self.vocab_list = vocab_list
        self.multival_sep = multival_sep

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        # string
        # num_multi_rows = random.randint(num_rows // 3, 2 * num_rows // 3)
        num_multi_id = 3
        data_multi = _create_random_id_data(
            (num_rows, num_multi_id), self.num_ids, self.vocab_list
        ).astype(str)
        data_multi = list(map(lambda x: self.multival_sep.join(x), data_multi))
        random.shuffle(data_multi)
        return pa.array(data_multi)


class SeqIdMockInput(MockInput):
    """Mock sparse id sequence input data class."""

    def __init__(
        self,
        name: str,
        num_ids: Optional[int] = None,
        vocab_list: Optional[List[str]] = None,
        sequence_length: Optional[int] = None,
        sequence_delim: Optional[str] = "|",
        multival_sep: str = chr(3),
    ) -> None:
        super().__init__(name)
        self.num_ids = num_ids
        self.vocab_list = vocab_list
        self.sequence_length = sequence_length
        self.sequence_delim = sequence_delim
        self.multival_sep = multival_sep

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        num_multi_rows = random.randint(num_rows // 3, 2 * num_rows // 3)
        num_multi_id = self.sequence_length or 10
        data_multi = _create_random_id_data(
            (num_multi_rows, num_multi_id), self.num_ids, self.vocab_list
        ).astype(str)
        data_multi = list(map(lambda x: self.sequence_delim.join(x), data_multi))
        num_single_rows = (
            random.randint(num_rows // 3, num_rows - num_multi_rows)
            if has_null
            else num_rows - num_multi_rows
        )
        data_single = list(
            _create_random_id_data(
                (num_single_rows,), self.num_ids, self.vocab_list
            ).astype(str)
        )
        data = (
            data_multi
            + data_single
            + [None] * (num_rows - num_multi_rows - num_single_rows)
        )
        random.shuffle(data)
        return pa.array(data)


class RawMockInput(MockInput):
    """Mock dense raw input data class."""

    def __init__(
        self, name: str, value_dim: int = 1, multival_sep: str = chr(3)
    ) -> None:
        super().__init__(name)
        self.value_dim = value_dim
        self.multival_sep = multival_sep

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        if self.value_dim > 1:
            data = np.random.rand(num_rows, self.value_dim).astype(str)
            data = list(map(lambda x: self.multival_sep.join(x), data))
        else:
            data = np.random.rand(num_rows).astype(np.float32)
        return pa.array(data)


class SeqRawMockInput(MockInput):
    """Mock dense raw sequence input data class."""

    def __init__(
        self,
        name: str,
        value_dim: int = 1,
        sequence_length: Optional[int] = None,
        sequence_delim: Optional[str] = "|",
        multival_sep: str = chr(3),
    ) -> None:
        super().__init__(name)
        self.value_dim = value_dim
        self.sequence_length = sequence_length
        self.sequence_delim = sequence_delim
        self.multival_sep = multival_sep

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        num_multi_rows = random.randint(num_rows // 3, 2 * num_rows // 3)
        num_multi_id = self.sequence_length or 10
        data_multi = np.random.rand(
            num_multi_rows, num_multi_id, self.value_dim
        ).astype(str)
        data_multi = list(
            map(
                lambda x: self.sequence_delim.join(
                    map(lambda y: self.multival_sep.join(y), x)
                ),
                data_multi,
            )
        )
        num_single_rows = (
            random.randint(num_rows // 3, num_rows - num_multi_rows)
            if has_null
            else num_rows - num_multi_rows
        )
        data_single = np.random.rand(num_single_rows, self.value_dim).astype(str)
        data_single = list(map(lambda x: self.multival_sep.join(x), data_single))
        data = (
            data_multi
            + data_single
            + [None] * (num_rows - num_multi_rows - num_single_rows)
        )
        random.shuffle(data)
        return pa.array(data)


class SeqMockInput(MockInput):
    """Mock sequence input data class."""

    def __init__(
        self,
        name: str,
        side_infos: List[str],
        item_id: str,
        num_ids: Optional[int] = None,
        vocab_list: Optional[List[str]] = None,
        sequence_length: Optional[int] = None,
        sequence_delim: Optional[str] = "|",
        sequence_underline: Optional[str] = "__",
    ) -> None:
        super().__init__(name)
        self.num_ids = num_ids
        self.vocab_list = vocab_list
        self.sequence_length = sequence_length
        self.sequence_delim = sequence_delim
        self.side_infos = side_infos
        self.item_id = item_id
        self.sequence_underline = sequence_underline

    def create_sequence_data(
        self, num_rows: int, item_t: pa.Table
    ) -> Dict[str, pa.Array]:
        """Create mock data."""
        item_t_side_infos = []
        other_side_infos = []
        for side_info in self.side_infos:
            if side_info in item_t.column_names:
                item_t_side_infos.append(side_info)
            else:
                other_side_infos.append(side_info)

        # generate random sequence id
        row_number = []
        sequence_lengths = []
        for i in range(num_rows):
            sequence_length = random.randint(1, self.sequence_length)
            sequence_lengths.append(sequence_length)
            row_number.extend([i] * sequence_length)
        row_number = pa.array(row_number)
        id_data = pa.array(
            _create_random_id_data(len(row_number), self.num_ids, self.vocab_list)
        )
        other_id_datas = OrderedDict()
        for side_info in other_side_infos:
            other_id_datas[side_info] = pa.array(
                _create_random_id_data(len(row_number), 10)
            )
        tmp_t = pa.Table.from_arrays(
            [row_number, id_data] + list(other_id_datas.values()),
            names=["rn", self.item_id] + list(other_id_datas.keys()),
        )

        # join item side info
        selected_item_t = item_t.select(list(set([self.item_id] + item_t_side_infos)))
        join_t = tmp_t.join(selected_item_t, keys=self.item_id)

        # group by sequence item_id and concat
        new_schema = []
        for s in join_t.schema:
            new_schema.append(pa.field(s.name, pa.string()))
        # TODO(hongsheng.jhs) fillna("") when pyfg fix sequence null string
        df = join_t.cast(pa.schema(new_schema)).to_pandas().fillna("0")
        result_df = df.groupby("rn").agg(
            {key: self.sequence_delim.join for key in self.side_infos}
        )

        t = pa.Table.from_pandas(result_df)

        data = {}
        for name, arr in zip(t.column_names, t.columns):
            if name in self.side_infos:
                data[f"{self.name}{self.sequence_underline}{name}"] = arr

        return data


class MapMockInput(MockInput):
    """Mock lookup map input data class."""

    def __init__(
        self,
        name: str,
        is_sparse: bool = False,
        num_ids: Optional[int] = None,
        vocab_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name)
        self.is_sparse = is_sparse
        self.num_ids = num_ids
        self.vocab_list = vocab_list

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        num_multi_rows = random.randint(num_rows // 3, 2 * num_rows // 3)
        num_multi_id = 2
        data_key_multi = _create_random_id_data((num_multi_rows, num_multi_id)).astype(
            str
        )
        if self.is_sparse:
            data_val_multi = _create_random_id_data(
                (num_multi_rows, num_multi_id), self.num_ids, self.vocab_list
            )
        else:
            data_val_multi = np.random.rand(num_multi_rows, num_multi_id).astype(str)

        data_multi = list(
            map(
                lambda x: "\x1d".join([f"{kk}:{vv}" for kk, vv in zip(x[0], x[1])]),
                zip(data_key_multi, data_val_multi),
            )
        )

        num_single_rows = (
            random.randint(num_rows // 3, num_rows - num_multi_rows)
            if has_null
            else num_rows - num_multi_rows
        )
        data_key_single = list(
            _create_random_id_data(
                (num_single_rows,), self.num_ids, self.vocab_list
            ).astype(str)
        )
        if self.is_sparse:
            data_val_single = _create_random_id_data(
                (num_single_rows,), self.num_ids, self.vocab_list
            )
        else:
            data_val_single = np.random.rand(
                num_single_rows,
            ).astype(str)
        data_single = list(
            map(lambda x: f"{x[0]}:{x[1]}", zip(data_key_single, data_val_single))
        )

        data = (
            data_multi
            + data_single
            + [None] * (num_rows - num_multi_rows - num_single_rows)
        )
        random.shuffle(data)
        return pa.array(data)


class NestedMapInput(MockInput):
    """Mock match map input data class."""

    def __init__(
        self,
        name: str,
        is_sparse: bool = False,
        num_ids: Optional[int] = None,
        vocab_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name)
        self.is_sparse = is_sparse
        self.num_ids = num_ids
        self.vocab_list = vocab_list

    def create_data(self, num_rows: int, has_null: bool = True) -> pa.Array:
        """Create mock data."""
        num_multi_rows = random.randint(num_rows // 3, 2 * num_rows // 3)
        num_multi_id = 2
        data_pkey_multi = _create_random_id_data((num_multi_rows, num_multi_id)).astype(
            str
        )
        data_skey_multi = _create_random_id_data((num_multi_rows, num_multi_id)).astype(
            str
        )
        if self.is_sparse:
            data_val_multi = _create_random_id_data(
                (num_multi_rows, num_multi_id), self.num_ids, self.vocab_list
            )
        else:
            data_val_multi = np.random.rand(num_multi_rows, num_multi_id).astype(str)

        data_multi = list(
            map(
                lambda x: "|".join(
                    [f"{pk}^{sk}:{v}" for pk, sk, v in zip(x[0], x[1], x[2])]
                ),
                zip(data_pkey_multi, data_skey_multi, data_val_multi),
            )
        )

        num_single_rows = (
            random.randint(num_rows // 3, num_rows - num_multi_rows)
            if has_null
            else num_rows - num_multi_rows
        )
        data_pkey_single = list(
            _create_random_id_data(
                (num_single_rows,), self.num_ids, self.vocab_list
            ).astype(str)
        )
        data_skey_single = list(
            _create_random_id_data(
                (num_single_rows,), self.num_ids, self.vocab_list
            ).astype(str)
        )
        if self.is_sparse:
            data_val_single = _create_random_id_data(
                (num_single_rows,), self.num_ids, self.vocab_list
            )
        else:
            data_val_single = np.random.rand(
                num_single_rows,
            ).astype(str)
        data_single = list(
            map(
                lambda x: f"{x[0]}^{x[1]}:{x[2]}",
                zip(data_pkey_single, data_skey_single, data_val_single),
            )
        )

        data = (
            data_multi
            + data_single
            + [None] * (num_rows - num_multi_rows - num_single_rows)
        )
        random.shuffle(data)
        return pa.array(data)


def create_mock_data(
    data_dir: str,
    inputs: Dict[str, MockInput],
    label_fields: List[str],
    num_rows: int = 65536,
    num_parts: int = 1,
    unique_id: str = "",
    join_t: Optional[pa.Table] = None,
    fmt: str = "parquet",
) -> Tuple[str, pa.Table]:
    """Create mock data for testing."""
    input_data = {}
    for inp in inputs.values():
        if inp.name == unique_id:
            input_data[inp.name] = pa.array(list(range(num_rows)))
        elif isinstance(inp, SeqMockInput):
            input_data.update(inp.create_sequence_data(num_rows, join_t))
        else:
            input_data[inp.name] = inp.create_data(num_rows)

    for label_field in label_fields:
        input_data[label_field] = pa.array(np.random.randint(2, size=(num_rows,)))

    t = pa.Table.from_arrays(list(input_data.values()), names=list(input_data.keys()))
    max_rows_per_file = int(math.ceil(num_rows / num_parts))
    ds.write_dataset(
        t,
        data_dir,
        format=fmt,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=min(max_rows_per_file, 1024 * 1024),
    )

    return os.path.join(data_dir, f"*.{fmt}"), t


def create_mock_join_data(
    data_dir: str,
    inputs: Dict[str, MockInput],
    label_fields: List[str],
    join_tables: Dict[str, pa.Table],
    num_rows: int = 65536,
    num_parts: int = 2,
    id_fields: Optional[List[str]] = None,
    fmt: str = "parquet",
) -> str:
    """Create mock data for testing."""
    input_data = {}
    for inp in inputs.values():
        has_null = id_fields is None or inp.name not in id_fields
        input_data[inp.name] = inp.create_data(num_rows, has_null=has_null)

    for label_field in label_fields:
        input_data[label_field] = pa.array(np.random.randint(2, size=(num_rows,)))

    t = pa.Table.from_arrays(list(input_data.values()), names=list(input_data.keys()))
    for join_key, join_t in join_tables.items():
        t = t.join(join_t, keys=join_key)

    max_rows_per_file = int(math.ceil(num_rows / num_parts))
    ds.write_dataset(
        t,
        data_dir,
        format=fmt,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=min(max_rows_per_file, 1024 * 1024),
    )
    return os.path.join(data_dir, f"*.{fmt}")


def create_mock_item_gl_data(
    data_path: str,
    inputs: Dict[str, MockInput],
    item_id_field: str,
    neg_fields: List[str],
    attr_delimiter: str = "\x02",
    num_rows: int = 65536,
) -> str:
    """Create mock data for testing."""
    if neg_fields is None:
        neg_fields = []

    input_data = {}
    for inp in inputs.values():
        if inp.name in neg_fields:
            input_data[inp.name] = inp.create_data(num_rows, has_null=False).cast(
                pa.string()
            )

    field_data = [input_data[x].to_pylist() for x in [item_id_field] + neg_fields]
    lines = []
    lines.append("id:int64\tweight:float\tfeature:string\n")
    for row in zip(*field_data):
        lines.append("\t".join([row[0], "1", attr_delimiter.join(row[1:])]) + "\n")
    with open(data_path, "w") as f:
        f.writelines(lines)
    return data_path


def create_mock_hard_negative(
    edge_path: str,
    user_path: str,
    src_data: Dict[str, List],  # {user_id: user_t, item_id: item_t}
    num_rows: int = 10240,
) -> Tuple[str]:
    """Create hard negative mock data."""
    idx_1 = np.random.choice(
        np.arange(len(src_data["user_id"])), num_rows, replace=False
    )
    idx_2 = np.random.choice(
        np.arange(len(src_data["item_id"])), num_rows, replace=False
    )
    uid = np.array(src_data["user_id"])[idx_1].tolist()
    iid = np.array(src_data["item_id"])[idx_2].tolist()
    field_data = [uid, iid]
    lines = []
    lines.append("userid:int64\titemid:int64\tweight:float\n")
    for row in zip(*field_data):
        lines.append("\t".join([str(row[0]), str(row[1]), "1"]) + "\n")
    with open(edge_path, "w") as f:
        f.writelines(lines)

    user_lines = []
    user_lines.append("id:int64\tweight:float\n")
    for u in src_data["user_id"]:
        user_lines.append("\t".join([str(u), "1"]) + "\n")
    with open(user_path, "w") as f:
        f.writelines(user_lines)

    return edge_path, user_path


def build_mock_input_fg_encoded(
    features: List[BaseFeature], user_id: str = "", item_id: str = ""
) -> Dict[str, MockInput]:
    """Build fg encoded mock input instance list from features."""
    inputs = {}
    single_id_fields = {user_id, item_id}
    for feature in features:
        if feature.is_sequence:
            if feature.is_sparse:
                inputs[feature.inputs[0]] = SeqIdMockInput(
                    feature.inputs[0],
                    num_ids=feature.num_embeddings,
                    sequence_length=feature.sequence_length,
                    sequence_delim=feature.sequence_delim,
                )
            else:
                inputs[feature.inputs[0]] = SeqRawMockInput(
                    feature.inputs[0],
                    value_dim=feature.output_dim,
                    sequence_length=feature.sequence_length,
                    sequence_delim=feature.sequence_delim,
                )
        else:
            if feature.is_sparse:
                is_multi = (
                    random.random() < 0.5 and feature.inputs[0] not in single_id_fields
                )
                inputs[feature.inputs[0]] = IdMockInput(
                    feature.inputs[0], is_multi=is_multi, num_ids=feature.num_embeddings
                )
            else:
                inputs[feature.inputs[0]] = RawMockInput(
                    feature.inputs[0], value_dim=feature.output_dim
                )
    return inputs


def _get_vocab_list(feature: BaseFeature) -> Optional[List[str]]:
    config = feature.config
    vocab_list = None
    if len(config.vocab_list) > 0:
        vocab_list = list(config.vocab_list)
    elif len(config.vocab_dict) > 0:
        vocab_dict = OrderedDict(config.vocab_dict.items())
        length = max(vocab_dict.values()) + 1
        vocab_list = [""] * length
        for k, v in vocab_dict.items():
            vocab_list[v] = k
    elif config.HasField("vocab_file"):
        # TODO: support dict in vocab_file, now only support list
        vocab_list = []
        with open(config.vocab_file) as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    vocab_list.append(line)
    return vocab_list


def build_mock_input_with_fg(
    features: List[BaseFeature],
    user_id: str = "",
    item_id: str = "",
    is_hstu: bool = False,
) -> Dict[str, MockInput]:
    """Build mock input instance list with fg from features."""
    inputs = defaultdict(dict)
    single_id_fields = {user_id, item_id}
    for feature in features:
        if type(feature) is IdFeature:
            is_multi = (
                random.random() < 0.5 and feature.inputs[0] not in single_id_fields
            )
            side, name = feature.side_inputs[0]
            if feature.is_weighted:
                inputs[side][name] = MapMockInput(
                    name,
                    is_sparse=feature.is_sparse,
                    num_ids=feature.num_embeddings,
                    vocab_list=_get_vocab_list(feature),
                )
            else:
                inputs[side][name] = IdMockInput(
                    name,
                    is_multi=is_multi,
                    num_ids=feature.num_embeddings,
                    vocab_list=_get_vocab_list(feature),
                    multival_sep=chr(29),
                )
        elif type(feature) is RawFeature:
            side, name = feature.side_inputs[0]
            inputs[side][name] = RawMockInput(
                name,
                value_dim=feature.config.value_dim,
                multival_sep=chr(29),
            )
        elif type(feature) is ComboFeature:
            for side, input_name in feature.side_inputs:
                if input_name in inputs[side]:
                    continue
                is_multi = random.random() < 0.5 and input_name not in single_id_fields
                inputs[side][input_name] = IdMockInput(
                    input_name, is_multi=is_multi, multival_sep=chr(29)
                )
        elif type(feature) is LookupFeature:
            for i, (side, input_name) in enumerate(feature.side_inputs):
                if input_name in inputs[side]:
                    continue
                if i == 0:
                    inputs[side][input_name] = MapMockInput(
                        input_name,
                        is_sparse=feature.is_sparse,
                        num_ids=feature.num_embeddings,
                        vocab_list=_get_vocab_list(feature),
                    )
                else:
                    is_multi = (
                        random.random() < 0.5 and input_name not in single_id_fields
                    )
                    inputs[side][input_name] = IdMockInput(
                        input_name, is_multi=is_multi, multival_sep=chr(29)
                    )
        elif type(feature) is MatchFeature:
            for i, (side, input_name) in enumerate(feature.side_inputs):
                if input_name in inputs[side]:
                    continue
                if i == 0:
                    inputs[side][input_name] = NestedMapInput(input_name)
                else:
                    inputs[side][input_name] = IdMockInput(
                        input_name, multival_sep=chr(29)
                    )
        elif type(feature) is ExprFeature:
            for side, input_name in feature.side_inputs:
                if input_name in inputs[side]:
                    continue
                inputs[side][input_name] = RawMockInput(
                    input_name, multival_sep=chr(29)
                )
        elif type(feature) is TokenizeFeature:
            side, name = feature.side_inputs[0]
            inputs[side][name] = IdMockInput(
                name,
                is_multi=True,
                num_ids=feature.num_embeddings,
                multival_sep=" ",
            )
        elif type(feature) is CustomFeature:
            if feature.config.operator_name != "EditDistance":
                raise ValueError(
                    "Mock CustomFeature with operator_name"
                    f"[{feature.config.operator_name}] is not supported."
                )
            for side, input_name in feature.side_inputs:
                inputs[side][input_name] = IdMockInput(
                    input_name,
                    is_multi=True,
                    num_ids=10,
                    multival_sep="",
                )

    seq_underline = "_" if env_util.use_rtp() else "__"
    for feature in features:
        if type(feature) in [
            SequenceIdFeature,
            SequenceRawFeature,
            SequenceCustomFeature,
        ]:
            for side, input_name in feature.side_inputs:
                if feature.is_grouped_sequence:
                    sub_name = input_name.replace(
                        f"{feature.sequence_name}{seq_underline}", ""
                    )
                    if feature.sequence_name not in inputs["user"]:
                        inputs["user"][feature.sequence_name] = SeqMockInput(
                            name=feature.sequence_name,
                            side_infos=[],
                            item_id=item_id,
                            num_ids=inputs["item"][item_id].num_ids,
                            vocab_list=inputs["item"][item_id].vocab_list,
                            sequence_length=feature.sequence_length,
                            sequence_delim=feature.sequence_delim,
                            sequence_underline=seq_underline,
                        )
                    inputs["user"][feature.sequence_name].side_infos.append(sub_name)
                    if sub_name in inputs["item"]:
                        if isinstance(inputs[side][sub_name], IdMockInput):
                            inputs[side][sub_name].is_multi = False
                else:
                    if is_hstu:
                        # hstu require number of sequence item is over 2
                        inputs[side][input_name] = HSTUIdMockInput(
                            input_name,
                            is_multi=True,
                            num_ids=feature.num_embeddings,
                            multival_sep=feature.sequence_delim,
                        )
                    else:
                        inputs[side][input_name] = IdMockInput(
                            input_name,
                            is_multi=True,
                            num_ids=10
                            if isinstance(feature, SequenceCustomFeature)
                            else feature.num_embeddings,
                            multival_sep=feature.sequence_delim,
                        )
    return inputs["user"], inputs["item"]


def load_config_for_test(
    pipeline_config_path: str,
    test_dir: str,
    user_id: str = "",
    item_id: str = "",
    cate_id: str = "",
    is_hstu: bool = False,
    num_rows: Optional[int] = None,
) -> EasyRecConfig:
    """Modify pipeline config for integration tests."""
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    pipeline_config.model_dir = os.path.join(test_dir, "train")
    if len(pipeline_config.train_input_path) > 0:
        # use prepared data
        return pipeline_config

    # rewrite config with mock data
    data_config = pipeline_config.data_config
    features = create_features(
        list(pipeline_config.feature_configs),
        fg_mode=data_config.fg_mode,
    )

    data_config.num_workers = 2
    num_parts = data_config.num_workers * 2
    if data_config.fg_mode == FgMode.FG_NONE:
        inputs = build_mock_input_fg_encoded(features, user_id, item_id)
        item_inputs = inputs
        pipeline_config.train_input_path, _ = create_mock_data(
            os.path.join(test_dir, "train_data"),
            inputs,
            list(data_config.label_fields),
            num_rows=num_rows or data_config.batch_size * num_parts * 4,
            num_parts=num_parts,
        )
        pipeline_config.eval_input_path, _ = create_mock_data(
            os.path.join(test_dir, "eval_data"),
            inputs,
            list(data_config.label_fields),
            num_rows=num_rows or data_config.batch_size * num_parts * 4,
            num_parts=num_parts,
        )
    else:
        user_inputs, item_inputs = build_mock_input_with_fg(
            features, user_id, item_id, is_hstu
        )
        _, item_t = create_mock_data(
            os.path.join(test_dir, "item_data"),
            item_inputs,
            label_fields=[],
            unique_id=item_id,
            num_rows=data_config.batch_size * num_parts * 4,
            num_parts=num_parts,
        )
        _, user_t = create_mock_data(
            os.path.join(test_dir, "user_data"),
            user_inputs,
            label_fields=[],
            unique_id=user_id,
            join_t=item_t,
            num_rows=data_config.batch_size * num_parts * 4,
            num_parts=num_parts,
        )
        pipeline_config.train_input_path = create_mock_join_data(
            os.path.join(test_dir, "train_data"),
            {user_id: user_inputs[user_id], item_id: item_inputs[item_id]},
            list(data_config.label_fields),
            {user_id: user_t, item_id: item_t},
            num_rows=num_rows or data_config.batch_size * num_parts * 4,
            num_parts=num_parts,
            id_fields=[user_id, item_id],
        )
        pipeline_config.eval_input_path = create_mock_join_data(
            os.path.join(test_dir, "eval_data"),
            {user_id: user_inputs[user_id], item_id: item_inputs[item_id]},
            list(data_config.label_fields),
            {user_id: user_t, item_id: item_t},
            num_rows=num_rows or data_config.batch_size * num_parts * 4,
            num_parts=num_parts,
            id_fields=[user_id, item_id],
        )

    sampler_type = None
    sampler_config = None
    if data_config.HasField("sampler"):
        sampler_type = data_config.WhichOneof("sampler")
        sampler_config = getattr(data_config, sampler_type)
    if sampler_type is not None:
        if sampler_type == "tdm_sampler":
            all_attr_fields = item_t.column_names
            attr_fields = []
            raw_attr_fields = []
            feature_configs = pipeline_config.feature_configs
            for feature_config in feature_configs:
                feature_type = feature_config.WhichOneof("feature")
                d_feature_config = getattr(feature_config, feature_type)
                if not hasattr(d_feature_config, "feature_name"):
                    continue
                feature_name = d_feature_config.feature_name
                if feature_name not in all_attr_fields:
                    continue
                if feature_name == item_id:
                    d_feature_config.num_buckets = 2 * d_feature_config.num_buckets
                    continue
                if feature_type != "raw_feature":
                    attr_fields.append(feature_name)
                else:
                    raw_attr_fields.append(feature_name)

            item_input_path = os.path.join(test_dir, r"item_data/\*.parquet")
            cmd_str = (
                "PYTHONPATH=. python tzrec/tools/tdm/init_tree.py "
                f"--item_input_path {item_input_path} "
                f"--item_id_field {item_id} "
                f"--cate_id_field {cate_id} "
                f"--attr_fields {','.join(attr_fields)} "
                f"--raw_attr_fields {','.join(raw_attr_fields)} "
                f"--node_edge_output_file {test_dir}/init_tree "
                f"--tree_output_dir {test_dir}/init_tree "
            )
            assert misc_util.run_cmd(
                cmd_str, os.path.join(test_dir, "log_init_tree.txt"), timeout=600
            )

            sampler_config.item_input_path = os.path.join(
                test_dir, "init_tree/node_table.txt"
            )
            sampler_config.edge_input_path = os.path.join(
                test_dir, "init_tree/edge_table.txt"
            )
            sampler_config.predict_edge_input_path = os.path.join(
                test_dir, "init_tree/predict_edge_table.txt"
            )

        elif sampler_type == "negative_sampler":
            item_gl_path = create_mock_item_gl_data(
                os.path.join(test_dir, "item_gl"),
                item_inputs,
                item_id,
                # hstu only uses item_id as negative sample, \
                # as sampler_config.attr_fields is sequence
                neg_fields=[item_id] if is_hstu else list(sampler_config.attr_fields),
                attr_delimiter=sampler_config.attr_delimiter,
                num_rows=data_config.batch_size * num_parts * 4,
            )
            # assert sampler_type == "negative_sampler", (
            #     "now only negative_sampler supported."
            # )
            sampler_config.input_path = item_gl_path
        elif sampler_type == "hard_negative_sampler":
            item_gl_path = create_mock_item_gl_data(
                os.path.join(test_dir, "item_gl"),
                item_inputs,
                item_id,
                neg_fields=[item_id] if is_hstu else list(sampler_config.attr_fields),
                attr_delimiter=sampler_config.attr_delimiter,
                num_rows=data_config.batch_size * num_parts * 4,
            )

            hard_neg_edge_path, hard_neg_user_path = create_mock_hard_negative(
                os.path.join(test_dir, "hard_neg_edge"),
                os.path.join(test_dir, "hard_neg_user"),
                {
                    "user_id": user_t["user_id"].to_pylist(),
                    "item_id": item_t["item_id"].to_pylist(),
                },
                num_rows=data_config.batch_size * 4,
            )
            sampler_config.hard_neg_edge_input_path = hard_neg_edge_path
            sampler_config.user_input_path = hard_neg_user_path
            sampler_config.item_input_path = item_gl_path

    data_config.dataset_type = data_pb2.ParquetDataset

    return pipeline_config


def _standalone():
    if os.environ.get("CI", "false").lower() == "true":
        # When using GitHub Actions, a container network is created by GitHub, and
        # the host network cannot be utilized. This can lead to the error:
        # [c10d] The hostname of the client socket cannot be retrieved, error code: -3.
        return "--master_addr=localhost --master_port=#MASTER_PORT#"
    else:
        return "--standalone"


def test_train_eval(
    pipeline_config_path: str,
    test_dir: str,
    args_str: str = "",
    user_id: str = "",
    item_id: str = "",
    cate_id: str = "",
    is_hstu: bool = False,
    env_str: str = "",
    num_rows: Optional[int] = None,
) -> bool:
    """Run train_eval integration test."""
    pipeline_config = load_config_for_test(
        pipeline_config_path,
        test_dir,
        user_id,
        item_id,
        cate_id,
        is_hstu,
        num_rows=num_rows,
    )

    test_config_path = os.path.join(test_dir, "pipeline.config")
    config_util.save_message(pipeline_config, test_config_path)
    log_dir = os.path.join(test_dir, "log_train_eval")
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/train_eval.py "
        f"--pipeline_config_path {test_config_path} {args_str}"
    )
    if env_str:
        cmd_str = f"{env_str} {cmd_str}"
    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_train_eval.txt"), timeout=600
    )


def test_eval(
    pipeline_config_path: str,
    test_dir: str,
    env_str: str = "",
) -> bool:
    """Run evaluate integration test."""
    log_dir = os.path.join(test_dir, "log_eval")
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/eval.py "
        f"--pipeline_config_path {pipeline_config_path}"
    )
    if env_str:
        cmd_str = f"{env_str} {cmd_str}"
    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_eval.txt"), timeout=600
    )


def test_export(
    pipeline_config_path: str,
    test_dir: str,
    export_dir: str = "",
    asset_files: str = "",
    env_str: str = "",
) -> bool:
    """Run export integration test."""
    log_dir = os.path.join(test_dir, "log_export")
    export_dir = export_dir or f"{test_dir}/export"
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/export.py "
        f"--pipeline_config_path {pipeline_config_path} "
        f"--export_dir {export_dir} "
    )
    if env_str:
        cmd_str = f"{env_str} {cmd_str}"
    if asset_files:
        cmd_str += f"--asset_files {asset_files}"

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_export.txt"), timeout=600
    )


def test_feature_selection(pipeline_config_path: str, test_dir: str) -> bool:
    """Run export integration test."""
    log_dir = os.path.join(test_dir, "log_feature_selection")
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node=1 --log_dir {log_dir} "
        "-m tzrec.tools.feature_selection "
        f"--pipeline_config_path {pipeline_config_path} "
        "--topk 5 "
        f"--model_dir {test_dir}/train "
        f"--output_dir {test_dir}/output_dir"
    )

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_export.txt"), timeout=600
    )


def test_predict(
    scripted_model_path: str,
    predict_input_path: str,
    predict_output_path: str,
    reserved_columns: str,
    output_columns: str,
    test_dir: str,
    predict_threads: Optional[int] = None,
    predict_steps: Optional[int] = None,
    env_str: str = "",
) -> bool:
    """Run predict integration test."""
    log_dir = os.path.join(test_dir, "log_predict")
    is_trt = is_trt_predict(scripted_model_path)
    is_aot = is_aot_predict(scripted_model_path)
    if is_trt or is_aot:
        # trt script and aoti model don't support device,default is cuda:0
        nproc_per_node = 1
    else:
        nproc_per_node = 2

    nproc_per_node = min(nproc_per_node, _get_nproc_per_node())
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={nproc_per_node} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/predict.py "
        f"--scripted_model_path {scripted_model_path} "
        f"--predict_input_path {predict_input_path} "
        f"--predict_output_path {predict_output_path} "
        f"--reserved_columns {reserved_columns} "
    )
    if output_columns:
        cmd_str += f"--output_columns {output_columns} "
    if predict_threads is not None:
        cmd_str += f"--predict_threads {predict_threads} "
    if predict_steps is not None:
        cmd_str += f"--predict_steps {predict_steps} "
    if env_str:
        cmd_str = f"{env_str} {cmd_str}"

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_predict.txt"), timeout=600
    )


def test_predict_checkpoint(
    pipeline_config_path: str,
    predict_input_path: str,
    predict_output_path: str,
    reserved_columns: str,
    output_columns: str,
    test_dir: str,
    env_str: str = "",
) -> bool:
    """Run predict checkpoint integration test."""
    log_dir = os.path.join(test_dir, "log_predict_ckpt")
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/predict.py "
        f"--pipeline_config_path {pipeline_config_path} "
        f"--predict_input_path {predict_input_path} "
        f"--predict_output_path {predict_output_path} "
        f"--reserved_columns {reserved_columns} "
    )
    if output_columns:
        cmd_str += f"--output_columns {output_columns} "
    if env_str:
        cmd_str = f"{env_str} {cmd_str}"

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_predict_ckpt.txt"), timeout=600
    )


def test_create_faiss_index(
    embedding_input_path: str,
    index_output_dir: str,
    id_field: str,
    embedding_field: str,
    test_dir: str,
) -> bool:
    """Run create faiss index integration test."""
    cmd_str = (
        "PYTHONPATH=. python tzrec/tools/create_faiss_index.py "
        f"--embedding_input_path {embedding_input_path} "
        f"--index_output_dir {index_output_dir} "
        f"--id_field {id_field} "
        f"--embedding_field {embedding_field}"
    )

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_faiss.txt"), timeout=600
    )


def test_hitrate(
    user_gt_input: str,
    item_embedding_input: str,
    total_hitrate_output: str,
    hitrate_details_output: str,
    request_id_field: str,
    gt_items_field: str,
    test_dir: str,
) -> bool:
    """Run hitrate integration test."""
    log_dir = os.path.join(test_dir, "log_hitrate")
    cmd_str = (
        f"OMP_NUM_THREADS=16 PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/tools/hitrate.py "
        f"--user_gt_input {user_gt_input} "
        f"--item_embedding_input {item_embedding_input} "
        f"--total_hitrate_output {total_hitrate_output} "
        f"--hitrate_details_output {hitrate_details_output} "
        f"--request_id_field {request_id_field} "
        f"--gt_items_field {gt_items_field}"
    )

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_hitrate.txt"), timeout=600
    )


def test_create_fg_json(
    pipeline_config_path: str,
    fg_output_dir: str,
    reserves: str,
    test_dir: str,
) -> bool:
    """Run create faiss index integration test."""
    cmd_str = (
        "PYTHONPATH=. python tzrec/tools/create_fg_json.py "
        f"--pipeline_config_path {pipeline_config_path} "
        f"--fg_output_dir {fg_output_dir} "
        f"--reserves {reserves} "
    )

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_create_fg_json.txt"), timeout=600
    )


def create_predict_data(
    pipeline_config_path: str, batch_size: int, item_id: str, output_dir: str
):
    """create_predict_data.

    all user feat is same
    """
    pipeline_config = config_util.load_pipeline_config(
        os.path.join(pipeline_config_path)
    )
    data_config = pipeline_config.data_config
    assert data_config.fg_mode in [
        FgMode.FG_NORMAL,
        FgMode.FG_DAG,
    ], "You should not use fg encoded data for input_path."
    features = create_features(
        pipeline_config.feature_configs,
        fg_mode=data_config.fg_mode,
    )
    user_inputs = []
    for feature in features:
        for side, name in feature.side_inputs:
            if feature.is_grouped_sequence or side == "user":
                user_inputs.append(name)

    reader = create_reader(
        input_path=pipeline_config.train_input_path,
        batch_size=batch_size,
        quota_name=data_config.odps_data_quota_name,
    )

    infer_arrow = OrderedDict()
    infer_json = OrderedDict()
    for data in reader.to_batches():
        for name, column in data.items():
            if pa.types.is_map(column.type):
                value_list = list(map(dict, column.fill_null({}).tolist()))
            else:
                value_list = column.tolist()
            if name in user_inputs:
                infer_arrow[name] = column.take([0] * batch_size)
                infer_json[name] = {
                    "values": [value_list[0]],
                    "dtype": _type_pa_to_table(column.type),
                }
            else:
                infer_arrow[name] = column
                infer_json[name] = {
                    "values": value_list,
                    "dtype": _type_pa_to_table(column.type),
                }
            if name == item_id:
                infer_json["item_ids"] = infer_json[name]
        break

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(infer_json, f, indent=4)

    writer = create_writer(output_dir, writer_type="ParquetWriter")
    writer.write(infer_arrow)
    del writer


def save_predict_result_json(result: Dict[str, torch.Tensor], output_path: str):
    """Save the predict result in json."""
    result_dict_json = {}
    for k, v in result.items():
        result_dict_json[k] = v.detach().tolist()

    with open(output_path, "w") as json_file:
        json.dump(result_dict_json, json_file, indent=4)


def test_tdm_retrieval(
    scripted_model_path: str,
    eval_data_path: str,
    retrieval_output_path: str,
    reserved_columns: str,
    test_dir: str,
) -> bool:
    """Run tdm retrieval test."""
    log_dir = os.path.join(test_dir, "log_tdm_retrieval")
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/tools/tdm/retrieval.py "
        f"--scripted_model_path {scripted_model_path} "
        f"--predict_input_path {eval_data_path} "
        f"--predict_output_path {retrieval_output_path} "
        f"--reserved_columns {reserved_columns} "
        f"--recall_num 2 "
        f"--n_cluster 2 "
    )

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_tdm_retrieval.txt"), timeout=600
    )


def test_tdm_cluster_train_eval(
    pipeline_config_path: str,
    test_dir: str,
    item_input_path: str,
    item_id: str,
    embedding_field: str,
) -> bool:
    """Run tdm cluster and train_eval integration test."""
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    data_config = pipeline_config.data_config
    sampler_type = data_config.WhichOneof("sampler")
    sampler_config = getattr(data_config, sampler_type)

    attr_fields = []
    raw_attr_fields = []
    feature_configs = pipeline_config.feature_configs
    for feature_config in feature_configs:
        feature_type = feature_config.WhichOneof("feature")
        d_feature_config = getattr(feature_config, feature_type)
        if not hasattr(d_feature_config, "feature_name"):
            continue
        feature_name = d_feature_config.feature_name
        feature_expression = d_feature_config.expression
        if not feature_expression.startswith("item") or feature_name == item_id:
            continue
        if feature_type != "raw_feature":
            attr_fields.append(feature_name)
        else:
            raw_attr_fields.append(feature_name)

    log_dir = os.path.join(test_dir, "log_tdm_cluster")
    cluster_cmd_str = (
        "PYTHONPATH=. "
        "python tzrec/tools/tdm/cluster_tree.py "
        f"--item_input_path {item_input_path} "
        f"--item_id_field {item_id} "
        f"--embedding_field {embedding_field} "
        f"--attr_fields {','.join(attr_fields)} "
        f"--raw_attr_fields {','.join(raw_attr_fields)} "
        f"--node_edge_output_file {os.path.join(test_dir, 'learnt_tree')} "
        f"--tree_output_dir {os.path.join(test_dir, 'learnt_tree')} "
        f"--parallel 1 "
    )
    assert misc_util.run_cmd(
        cluster_cmd_str, os.path.join(test_dir, "log_tdm_cluster.txt"), timeout=600
    )

    sampler_config.item_input_path = os.path.join(
        test_dir, "learnt_tree/node_table.txt"
    )
    sampler_config.edge_input_path = os.path.join(
        test_dir, "learnt_tree/edge_table.txt"
    )
    sampler_config.predict_edge_input_path = os.path.join(
        test_dir, "learnt_tree/predict_edge_table.txt"
    )

    pipeline_config.model_dir = os.path.join(test_dir, "learnt_tree/train")
    test_config_path = os.path.join(test_dir, "learnt_tree/pipeline.config")
    config_util.save_message(pipeline_config, test_config_path)

    log_dir = os.path.join(test_dir, "log_learnt_train_eval")
    cmd_str = (
        f"PYTHONPATH=. torchrun {_standalone()} "
        f"--nnodes=1 --nproc-per-node={_get_nproc_per_node()} --log_dir {log_dir} "
        "-r 3 -t 3 tzrec/train_eval.py "
        f"--pipeline_config_path {test_config_path}"
    )

    return misc_util.run_cmd(
        cmd_str, os.path.join(test_dir, "log_train_eval.txt"), timeout=600
    )
