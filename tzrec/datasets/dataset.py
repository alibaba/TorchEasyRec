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

import os
import random
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from torch import distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from tzrec.constant import Mode
from tzrec.datasets.data_parser import DataParser
from tzrec.datasets.sampler import BaseSampler, TDMSampler
from tzrec.datasets.utils import (
    C_NEG_SAMPLE_MASK,
    C_SAMPLE_MASK,
    Batch,
    RecordBatchTensor,
    process_hstu_neg_sample,
    process_hstu_seq_data,
    remove_nullable,
)
from tzrec.features.feature import BaseFeature
from tzrec.protos import data_pb2
from tzrec.utils.load_class import get_register_class_meta
from tzrec.utils.logging_util import logger

_DATASET_CLASS_MAP = {}
_READER_CLASS_MAP = {}
_WRITER_CLASS_MAP = {}
_dataset_meta_cls = get_register_class_meta(_DATASET_CLASS_MAP)
_reader_meta_cls = get_register_class_meta(_READER_CLASS_MAP)
_writer_meta_cls = get_register_class_meta(_WRITER_CLASS_MAP)

AVAILABLE_PA_TYPES = {
    pa.int64(),
    pa.float64(),
    pa.float32(),
    pa.string(),
    pa.int32(),
    pa.list_(pa.int64()),
    pa.list_(pa.float64()),
    pa.list_(pa.float32()),
    pa.list_(pa.string()),
    pa.list_(pa.int32()),
    pa.list_(pa.list_(pa.int64())),
    pa.list_(pa.list_(pa.float64())),
    pa.list_(pa.list_(pa.float32())),
    pa.list_(pa.list_(pa.string())),
    pa.list_(pa.list_(pa.int32())),
    pa.map_(pa.string(), pa.int64()),
    pa.map_(pa.string(), pa.float64()),
    pa.map_(pa.string(), pa.float32()),
    pa.map_(pa.string(), pa.string()),
    pa.map_(pa.string(), pa.int32()),
    pa.map_(pa.int64(), pa.int64()),
    pa.map_(pa.int64(), pa.float64()),
    pa.map_(pa.int64(), pa.float32()),
    pa.map_(pa.int64(), pa.string()),
    pa.map_(pa.int64(), pa.int32()),
    pa.map_(pa.int32(), pa.int64()),
    pa.map_(pa.int32(), pa.float64()),
    pa.map_(pa.int32(), pa.float32()),
    pa.map_(pa.int32(), pa.string()),
    pa.map_(pa.int32(), pa.int32()),
}


def _expand_tdm_sample(
    input_data: Dict[str, pa.Array],
    pos_sampled: Dict[str, pa.Array],
    neg_sampled: Dict[str, pa.Array],
    data_config: data_pb2.DataConfig,
) -> Dict[str, pa.Array]:
    """Expand input data with sampled data for tdm.

    Combine the sampled positive and negative samples with the item
    features, then expand the user features based on the original user-item
    relationships, and supplement the corresponding labels according to the
    positive and negative samples. Note that in the sampling results, the
    sampled outcomes for each item are contiguous.

    for example:
        user_fea:[1, 2], item_fea:[0.1, 0.2], labels:[1,1],
        pos_sample:[0.11, 0.12, 0.21, 0.22], neg_sample:[-0.11, -0.12, -0.21, -0.22]

        concat item_fea:[0.1, 0.2, 0.11, 0.12, 0.21, 0.22, -0.11, -0.12, -0.21, -0.22]
        duplicate user_fea and keep origin user-item
        relationship: [1, 2, 1, 1, 2, 2, 1, 1, 2, 2]

        expand label: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    """
    item_fea_names = pos_sampled.keys()
    all_fea_names = input_data.keys()
    label_fields = set(data_config.label_fields)
    user_fea_names = all_fea_names - item_fea_names - label_fields

    for item_fea_name in item_fea_names:
        input_data[item_fea_name] = pa.concat_arrays(
            [
                input_data[item_fea_name],
                pos_sampled[item_fea_name],
                neg_sampled[item_fea_name],
            ]
        )

    # In the sampling results, the sampled outcomes for each item are contiguous.
    batch_size = len(input_data[list(label_fields)[0]])
    num_pos_sampled = len(pos_sampled[list(item_fea_names)[0]])
    num_neg_sampled = len(neg_sampled[list(item_fea_names)[0]])
    user_pos_index = np.repeat(np.arange(batch_size), num_pos_sampled // batch_size)
    user_neg_index = np.repeat(np.arange(batch_size), num_neg_sampled // batch_size)
    for user_fea_name in user_fea_names:
        user_fea = input_data[user_fea_name]
        pos_expand_user_fea = user_fea.take(user_pos_index)
        neg_expand_user_fea = user_fea.take(user_neg_index)
        input_data[user_fea_name] = pa.concat_arrays(
            [
                input_data[user_fea_name],
                pos_expand_user_fea,
                neg_expand_user_fea,
            ]
        )

    for label_field in label_fields:
        input_data[label_field] = pa.concat_arrays(
            [
                input_data[label_field].cast(pa.int64()),
                pa.array([1] * num_pos_sampled, type=pa.int64()),
                pa.array([0] * num_neg_sampled, type=pa.int64()),
            ]
        )

    return input_data


class BaseDataset(IterableDataset, metaclass=_dataset_meta_cls):
    """Dataset base class.

    Args:
        data_config (DataConfig): an instance of DataConfig.
        features (list): list of features.
        input_path (str): data input path.
        reserved_columns (list): reserved columns in predict mode.
        mode (Mode): train or eval or predict.
        debug_level (int): dataset debug level, when mode=predict and
            debug_level > 0, will dump fg encoded data to debug_str
    """

    def __init__(
        self,
        data_config: data_pb2.DataConfig,
        features: List[BaseFeature],
        input_path: str,
        reserved_columns: Optional[List[str]] = None,
        mode: Mode = Mode.EVAL,
        debug_level: int = 0,
    ) -> None:
        super(BaseDataset, self).__init__()
        self._data_config = data_config
        self._features = features
        self._input_path = input_path
        self._reserved_columns = reserved_columns or []
        self._mode = mode
        self._debug_level = debug_level
        self._enable_hstu = data_config.enable_hstu
        self.sampler_type = (
            self._data_config.WhichOneof("sampler")
            if self._data_config.HasField("sampler")
            else None
        )

        self._data_parser = DataParser(
            features=features,
            labels=list(data_config.label_fields)
            if self._mode != Mode.PREDICT
            else None,
            sample_weights=list(data_config.sample_weight_fields)
            if self._mode != Mode.PREDICT
            else None,
            mode=self._mode,
            fg_threads=data_config.fg_threads,
            force_base_data_group=data_config.force_base_data_group,
            sampler_type=self.sampler_type,
        )

        self._input_fields = None
        self._selected_input_names = set()
        self._selected_input_names |= self._data_parser.feature_input_names
        if self._mode == Mode.PREDICT:
            self._selected_input_names |= set(self._reserved_columns)
        else:
            self._selected_input_names |= set(data_config.label_fields)
            self._selected_input_names |= set(data_config.sample_weight_fields)
        if self._data_config.HasField("sampler") and self._mode != Mode.PREDICT:
            sampler_type = self._data_config.WhichOneof("sampler")
            sampler_config = getattr(self._data_config, sampler_type)
            if hasattr(sampler_config, "item_id_field") and sampler_config.HasField(
                "item_id_field"
            ):
                self._selected_input_names.add(sampler_config.item_id_field)
            if hasattr(sampler_config, "user_id_field") and sampler_config.HasField(
                "user_id_field"
            ):
                self._selected_input_names.add(sampler_config.user_id_field)
        # if set selected_input_names to None,
        # all columns will be reserved.
        if (
            len(self._reserved_columns) > 0
            and self._reserved_columns[0] == "ALL_COLUMNS"
        ):
            self._selected_input_names = None

        self._fg_mode = data_config.fg_mode
        self._fg_encoded_multival_sep = data_config.fg_encoded_multival_sep

        if mode != Mode.TRAIN and data_config.HasField("eval_batch_size"):
            self._batch_size = data_config.eval_batch_size
        else:
            self._batch_size = data_config.batch_size

        self._sampler = None
        self._sampler_inited = False

        self._reader = None

    def launch_sampler_cluster(
        self,
        num_client_per_rank: int = 1,
        client_id_bias: int = 0,
        cluster: Optional[Dict[str, Union[int, str]]] = None,
    ) -> None:
        """Launch sampler cluster and server."""
        if self._data_config.HasField("sampler") and self._mode != Mode.PREDICT:
            sampler_type = self._data_config.WhichOneof("sampler")
            sampler_config = getattr(self._data_config, sampler_type)
            # pyre-ignore [16]
            self._sampler = BaseSampler.create_class(sampler_config.__class__.__name__)(
                sampler_config,
                self.input_fields,
                self._batch_size,
                is_training=self._mode == Mode.TRAIN,
                multival_sep=self._fg_encoded_multival_sep
                if self._fg_mode == data_pb2.FgMode.FG_NONE
                else chr(29),
            )
            self._sampler.init_cluster(num_client_per_rank, client_id_bias, cluster)
            if cluster is None:
                self._sampler.launch_server()

    def get_sampler_cluster(self) -> Optional[Dict[str, Union[int, str]]]:
        """Get sampler cluster."""
        if self._sampler:
            return self._sampler._cluster

    def _init_input_fields(self) -> None:
        """Init input fields info."""
        self._input_fields = []
        for field in self._reader.schema:
            field_type = remove_nullable(field.type)
            if any(map(lambda x: x == field_type, AVAILABLE_PA_TYPES)):
                self._input_fields.append(field)
            else:
                raise ValueError(
                    f"column [{field.name}] with dtype {field.type} "
                    "is not supported now."
                )

    @property
    def input_fields(self) -> List[pa.Field]:
        """Input fields info, overwrote by subclass for auto infer the info."""
        if not self._input_fields:
            self._input_fields = list(self._data_config.input_fields)
        return self._input_fields

    def get_worker_info(self) -> Tuple[int, int]:
        """Get multiprocessing dataloader worker id and worker number."""
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        return rank * num_workers + worker_id, num_workers * world_size

    def __iter__(self) -> Iterator[Batch]:
        if self._sampler is not None and not self._sampler_inited:
            self._sampler.init()
            self._sampler_inited = True
        worker_id, num_workers = self.get_worker_info()
        for input_data in self._reader.to_batches(worker_id, num_workers):
            yield self._build_batch(input_data)

    def _build_batch(self, input_data: Dict[str, pa.Array]) -> Batch:
        """Process input data and build batch.

        Args:
            input_data (dict): raw input data.

        Returns:
            an instance of Batch.
        """
        use_sample_mask = self._mode == Mode.TRAIN and (
            self._data_config.negative_sample_mask_prob > 0
            or self._data_config.sample_mask_prob > 0
        )
        if use_sample_mask:
            input_data[C_SAMPLE_MASK] = pa.array(
                np.random.random(len(list(input_data.values())[0]))
                < self._data_config.sample_mask_prob
            )
        if self._sampler is not None:
            if isinstance(self._sampler, TDMSampler):
                pos_sampled, neg_sampled = self._sampler.get(input_data)
                input_data = _expand_tdm_sample(
                    input_data, pos_sampled, neg_sampled, self._data_config
                )
            elif self._enable_hstu:
                seq_attr = self._sampler._item_id_field

                (
                    input_data_k_split,
                    input_data_k_split_slice,
                    pre_seq_filter_reshaped_joined,
                ) = process_hstu_seq_data(
                    input_data=input_data,
                    seq_attr=seq_attr,
                    seq_str_delim=self._sampler.item_id_delim,
                )
                if self._mode == Mode.TRAIN:
                    # Training using all possible target items
                    input_data[seq_attr] = input_data_k_split_slice
                elif self._mode == Mode.EVAL:
                    # Evaluation using the last item for previous sequence
                    input_data[seq_attr] = input_data_k_split.values.take(
                        pa.array(input_data_k_split.offsets.to_numpy()[1:] - 1)
                    )
                sampled = self._sampler.get(input_data)
                # To keep consistent with other process, use two functions
                for k, v in sampled.items():
                    if k in input_data:
                        combined = process_hstu_neg_sample(
                            input_data,
                            v,
                            self._sampler._num_sample,
                            self._sampler.item_id_delim,
                            seq_attr,
                        )
                        # Combine here to make embddings of both user sequence
                        # and target item are the same
                        input_data[k] = pa.concat_arrays(
                            [pre_seq_filter_reshaped_joined, combined]
                        )
                    else:
                        input_data[k] = v
            else:
                sampled = self._sampler.get(input_data)
                for k, v in sampled.items():
                    if k in input_data:
                        input_data[k] = pa.concat_arrays([input_data[k], v])
                    else:
                        input_data[k] = v

            if use_sample_mask:
                input_data[C_NEG_SAMPLE_MASK] = pa.concat_arrays(
                    [
                        input_data[C_SAMPLE_MASK],
                        pa.array(
                            np.random.random(len(list(sampled.values())[0]))
                            < self._data_config.negative_sample_mask_prob
                        ),
                    ]
                )

        # TODO(hongsheng.jhs): add additional field like hard_negative
        output_data = self._data_parser.parse(input_data)
        if self._mode == Mode.PREDICT:
            batch = self._data_parser.to_batch(output_data, force_no_tile=True)
            reserved_data = {}
            if (
                len(self._reserved_columns) > 0
                and self._reserved_columns[0] == "ALL_COLUMNS"
            ):
                reserved_data = input_data
            else:
                for k in self._reserved_columns:
                    reserved_data[k] = input_data[k]
            if self._debug_level > 0:
                reserved_data["__features__"] = self._data_parser.dump_parsed_inputs(
                    output_data
                )
            if len(reserved_data) > 0:
                batch.reserves = RecordBatchTensor(pa.record_batch(reserved_data))
        else:
            batch = self._data_parser.to_batch(output_data)
        return batch

    @property
    def sampled_batch_size(self) -> int:
        """Batch size with sampler."""
        if self._sampler:
            return self._batch_size + self._sampler.estimated_sample_num
        else:
            return self._batch_size


class BaseReader(metaclass=_reader_meta_cls):
    """Reader base class.

    Args:
        input_path (str): data input path.
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        drop_remainder (bool): drop last batch.
        shuffle (bool): shuffle data or not.
        shuffle_buffer_size (int): buffer size for shuffle.
    """

    def __init__(
        self,
        input_path: str,
        batch_size: int,
        selected_cols: Optional[List[str]] = None,
        drop_remainder: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 32,
        **kwargs: Any,
    ) -> None:
        self._input_path = input_path
        self._batch_size = batch_size
        self._selected_cols = selected_cols
        self._drop_remainder = drop_remainder
        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size

    @property
    def schema(self) -> pa.Schema:
        """Table schema."""
        raise NotImplementedError

    def to_batches(
        self, worker_id: int = 0, num_workers: int = 1
    ) -> Iterator[Dict[str, pa.Array]]:
        """Get batch iterator."""
        raise NotImplementedError

    def _arrow_reader_iter(
        self, reader: Iterator[pa.RecordBatch]
    ) -> Iterator[Dict[str, pa.Array]]:
        shuffle_buffer = []
        buff_data = None
        while True:
            data = None
            if buff_data is None or len(buff_data) < self._batch_size:
                try:
                    read_data = next(reader)
                    if buff_data is None:
                        buff_data = pa.Table.from_batches([read_data])
                    else:
                        buff_data = pa.concat_tables(
                            [buff_data, pa.Table.from_batches([read_data])]
                        )
                except StopIteration:
                    data = None if self._drop_remainder else buff_data
                    buff_data = None
            elif len(buff_data) == self._batch_size:
                data = buff_data
                buff_data = None
            else:
                data = buff_data.slice(0, self._batch_size)
                buff_data = buff_data.slice(self._batch_size)

            if data is not None:
                data_dict = {}
                for name, column in zip(data.column_names, data.columns):
                    if isinstance(column, pa.ChunkedArray):
                        column = column.combine_chunks()
                    data_dict[name] = column

                if self._shuffle:
                    shuffle_buffer.append(data_dict)
                    if len(shuffle_buffer) < self._shuffle_buffer_size:
                        continue
                    else:
                        idx = random.randrange(len(shuffle_buffer))
                        data_dict = shuffle_buffer.pop(idx)

                yield data_dict

            if data is None and buff_data is None:
                break

        if len(shuffle_buffer) > 0:
            random.shuffle(shuffle_buffer)
            for data_dict in shuffle_buffer:
                yield data_dict

    def num_files(self) -> Optional[int]:
        """Get number of files in the dataset."""
        return None


class BaseWriter(metaclass=_writer_meta_cls):
    """Writer base class.

    Args:
        output_path (str): data output path.
    """

    def __init__(self, output_path: str, **kwargs: Any) -> None:
        self._lazy_inited = False
        self._output_path = output_path

    def write(self, output_dict: OrderedDict[str, pa.Array]) -> None:
        """Write a batch of data."""
        raise NotImplementedError

    def close(self) -> None:
        """Close and commit data."""
        self._lazy_inited = False

    def __del__(self) -> None:
        if self._lazy_inited:
            # pyre-ignore [16]
            logger.warning(f"You should close {self.__class__.__name__} explicitly.")


def create_reader(
    input_path: str,
    batch_size: int,
    selected_cols: Optional[List[str]] = None,
    reader_type: Optional[str] = None,
    **kwargs: Any,
) -> BaseReader:
    """Create data reader.

    Args:
        input_path (str): data input path.
        batch_size (int): batch size.
        selected_cols (list): selection column names.
        reader_type (str, optional): specify the input path reader type, if we cannot
            infer from input_path.
        **kwargs: additional params.

    Returns:
        reader: a data reader.
    """
    if input_path.startswith("odps://"):
        reader_cls_name = "OdpsReader"
    elif input_path.endswith(".csv"):
        reader_cls_name = "CsvReader"
    elif input_path.endswith(".parquet"):
        reader_cls_name = "ParquetReader"
    else:
        assert reader_type is not None, "You should set reader_type."
        reader_cls_name = reader_type

    # pyre-ignore [16]
    reader = BaseReader.create_class(reader_cls_name)(
        input_path=input_path,
        batch_size=batch_size,
        selected_cols=selected_cols,
        **kwargs,
    )
    return reader


def create_writer(
    output_path: str, writer_type: Optional[str] = None, **kwargs: Any
) -> BaseWriter:
    """Create data writer.

    Args:
        output_path (str): data output path.
        writer_type (str, optional): specify the input path writer type, if we cannot
            infer from input_path.
        **kwargs: additional params.

    Returns:
        writer: a data writer.
    """
    if output_path.startswith("odps://"):
        writer_cls_name = "OdpsWriter"
    else:
        assert writer_type is not None, "You should set writer_type."
        writer_cls_name = writer_type
    # pyre-ignore [16]
    writer = BaseWriter.create_class(writer_cls_name)(output_path=output_path, **kwargs)
    return writer


def create_dataloader(
    data_config: data_pb2.DataConfig,
    features: List[BaseFeature],
    input_path: str,
    reserved_columns: Optional[List[str]] = None,
    mode: Mode = Mode.TRAIN,
    gl_cluster: Optional[Dict[str, Union[int, str]]] = None,
    debug_level: int = 0,
) -> DataLoader:
    """Build dataloader.

    Args:
        data_config (DataConfig): dataloader config.
        features (list): list of feature.
        input_path (str): input data path.
        reserved_columns (list): reserved columns in predict mode.
        mode (Mode): train or eval or predict.
        gl_cluster (dict, bool): if set, reuse the graphlearn cluster.
        debug_level (int): dataset debug level, when mode=predict and
            debug_level > 0, will dump fg encoded data to debug_str

    Return:
        dataloader (dataloader): a DataLoader.
    """
    dataset_name = data_pb2.DatasetType.Name(data_config.dataset_type)
    # pyre-ignore [16]
    dataset_cls = BaseDataset.create_class(dataset_name)
    dataset = dataset_cls(
        data_config=data_config,
        features=features,
        input_path=input_path,
        reserved_columns=reserved_columns,
        mode=mode,
        debug_level=debug_level,
    )

    kwargs = {}
    if data_config.num_workers < 1:
        num_workers = 1
    else:
        num_workers = data_config.num_workers
        # check number of files is valid or not for file based dataset.
        num_files = dataset._reader.num_files()
        if num_files is not None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if num_files >= world_size:
                num_files_per_worker = num_files // world_size
                if num_files_per_worker < num_workers:
                    logger.info(
                        f"data_config.num_workers reset to {num_files_per_worker}"
                    )
                    num_workers = num_files_per_worker
            else:
                raise ValueError(
                    f"Number of files in the dataset[{input_path}] must greater "
                    f"than world_size: {world_size}, but got {num_files}"
                )

        kwargs["num_workers"] = num_workers
        kwargs["persistent_workers"] = True

    if mode == Mode.TRAIN:
        # When in train_and_eval mode, use 2x worker in gl cluster
        # for train_dataloader and eval_dataloader
        dataset.launch_sampler_cluster(num_client_per_rank=num_workers * 2)
    else:
        if gl_cluster:
            # Reuse the gl cluster for eval_dataloader
            dataset.launch_sampler_cluster(
                num_client_per_rank=num_workers * 2,
                client_id_bias=num_workers,
                cluster=gl_cluster,
            )
        else:
            dataset.launch_sampler_cluster(num_client_per_rank=num_workers)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        pin_memory=data_config.pin_memory if mode != Mode.PREDICT else False,
        collate_fn=lambda x: x,
        **kwargs,
    )
    # For PyTorch versions 2.6 and above, we initialize the data iterator before
    # beginning the training process to avoid potential CUDA-related issues following
    # model saving.
    iter(dataloader)
    return dataloader
