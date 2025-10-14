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

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile
import zipfile
from collections import OrderedDict

import requests
from google.protobuf import descriptor_pool, struct_pb2, symbol_database, text_format

from tzrec.constant import EASYREC_VERSION
from tzrec.protos import feature_pb2 as tzrec_feature_pb2
from tzrec.protos import (
    loss_pb2,
    metric_pb2,
    model_pb2,
    module_pb2,
    seq_encoder_pb2,
    tower_pb2,
)
from tzrec.protos import pipeline_pb2 as tzrec_pipeline_pb2
from tzrec.protos.data_pb2 import DatasetType
from tzrec.protos.models import match_model_pb2, multi_task_rank_pb2, rank_model_pb2
from tzrec.utils.logging_util import logger


def _get_easyrec(pkg_path=None):
    """Get easyrec whl and extract."""
    local_cache_dir = tempfile.mkdtemp(prefix="tzrec_tmp")
    if pkg_path is None:
        pkg_path = (
            f"https://easyrec.oss-cn-beijing.aliyuncs.com/release/whls/"
            f"easy_rec-{EASYREC_VERSION}-py2.py3-none-any.whl"
        )
    if pkg_path.startswith("http"):
        logger.info(f"downloading easyrec from {pkg_path}")
        r = requests.get(pkg_path)
        content = r.content
    else:
        with open(pkg_path, "rb") as f:
            content = f.read()
    if ".tar" in pkg_path:
        try:
            with tarfile.open(fileobj=io.BytesIO(content)) as tar:
                tar.extractall(path=local_cache_dir)
            local_package_dir = local_cache_dir
        except Exception as e:
            raise RuntimeError(f"invalid {pkg_path} tar.") from e
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as f:
                f.extractall(local_cache_dir)
            local_package_dir = local_cache_dir
        except zipfile.BadZipfile as e:
            raise RuntimeError(f"invalid {pkg_path} whl.") from e

    with open(os.path.join(local_package_dir, "easy_rec/__init__.py"), "w") as f:
        f.write("")
    sys.path.append(local_package_dir)
    _sym = symbol_database.Default()
    _sym.pool = descriptor_pool.DescriptorPool()
    from easy_rec.python.protos import feature_config_pb2 as _feature_config_pb2
    from easy_rec.python.protos import loss_pb2 as _loss_pb2
    from easy_rec.python.protos import pipeline_pb2 as _pipeline_pb2

    globals()["easyrec_pipeline_pb2"] = _pipeline_pb2
    globals()["easyrec_feature_config_pb2"] = _feature_config_pb2
    globals()["easyrec_loss_pb2"] = _loss_pb2


class ConvertConfig(object):
    """Convert EasyRec config to tzrec config.

    Args:
        easyrec_config_path (str): EasyRec config file path.
        fg_json_path (str): EasyRec use fg.json file path.
        output_tzrec_config_path (str): TzRec config file path will create.
    """

    def __init__(
        self,
        easyrec_config_path,
        output_tzrec_config_path,
        fg_json_path=None,
        use_old_fg=False,
        easyrec_package_path=None,
    ):
        if "easyrec_pipeline_pb2" not in globals():
            _get_easyrec(easyrec_package_path)
        self.output_tzrec_config_path = output_tzrec_config_path
        self.easyrec_config = self.load_easyrec_config(easyrec_config_path)

        self.feature_to_fg = {}
        self.sub_sequence_to_group = {}
        self.sequence_feature_to_fg = {}
        if fg_json_path is not None:
            self.fg_json = self.load_easyrec_fg_json(fg_json_path)
            self.analyse_fg(self.fg_json)
        self.use_old_fg = use_old_fg

    def analyse_fg(self, fg_json):
        """Analysis fg.json."""
        for feat in fg_json["features"]:
            if "sequence_name" in feat:
                sequence_name = feat["sequence_name"]
                for sub_feat in feat["features"]:
                    self.sub_sequence_to_group[
                        f"{sequence_name}__{sub_feat['feature_name']}"
                    ] = sequence_name
                self.sequence_feature_to_fg[sequence_name] = feat

            else:
                feature_name = feat["feature_name"]
                self.feature_to_fg[feature_name] = feat

    def load_easyrec_config(self, path):
        """Load easyrec config."""
        easyrec_config = easyrec_pipeline_pb2.EasyRecConfig()  # NOQA
        with open(path, "r", encoding="utf-8") as f:
            cfg_str = f.read()
            text_format.Merge(cfg_str, easyrec_config)
        return easyrec_config

    def load_easyrec_fg_json(self, path):
        """Load easyrec use fg.json."""
        with open(path, "r", encoding="utf-8") as f:
            fg_json = json.load(f)
        return fg_json

    def _create_train_config(self, pipeline_config):
        """Create easy_rec train config."""
        if not pipeline_config.HasField("train_config"):
            train_config_str = """
    train_config {
        sparse_optimizer {
            adam_optimizer {
                lr: 0.001
            }
            constant_learning_rate {
            }
        }
        dense_optimizer {
            adam_optimizer {
                lr: 0.001
            }
            constant_learning_rate {
            }
        }
        num_epochs: 1
        use_tensorboard: false
    }"""
            text_format.Merge(train_config_str, pipeline_config)
        return pipeline_config

    def _create_eval_config(self, pipeline_config):
        """Create tzrec train config."""
        if not pipeline_config.HasField("eval_config"):
            eval_config_str = "eval_config {}"
            text_format.Merge(eval_config_str, pipeline_config)
        return pipeline_config

    def _create_data_config(self, pipeline_config):
        """Create tzrec data config."""
        label_fields = list(self.easyrec_config.data_config.label_fields)
        pipeline_config.data_config.batch_size = (
            self.easyrec_config.data_config.batch_size
        )
        pipeline_config.data_config.dataset_type = DatasetType.OdpsDataset
        pipeline_config.data_config.label_fields.extend(label_fields)
        pipeline_config.data_config.num_workers = 8
        return pipeline_config

    def _easyrec_feature_2_tzrec(self, tzrec_feature, easyrec_feature=None):
        if easyrec_feature:
            if easyrec_feature.HasField("embedding_dim"):
                tzrec_feature.embedding_dim = easyrec_feature.embedding_dim
            if (
                hasattr(easyrec_feature, "hash_bucket_size")
                and easyrec_feature.hash_bucket_size > 0
                and hasattr(tzrec_feature, "hash_bucket_size")
            ):
                tzrec_feature.hash_bucket_size = easyrec_feature.hash_bucket_size
            if (
                hasattr(easyrec_feature, "boundaries")
                and len(list(easyrec_feature.boundaries)) > 0
                and hasattr(tzrec_feature, "boundaries")
            ):
                boundaries = list(easyrec_feature.boundaries)
                tzrec_feature.boundaries.extend(boundaries)
            if (
                hasattr(easyrec_feature, "num_buckets")
                and easyrec_feature.num_buckets > 0
                and hasattr(easyrec_feature, "num_buckets")
            ):
                tzrec_feature.num_buckets = easyrec_feature.num_buckets

    def _search_easyrec_config(self, feature_name):
        for cfg in self.easyrec_config.feature_configs:
            if cfg.feature_name:
                easy_feature_name = cfg.feature_name
            else:
                easy_feature_name = list(cfg.input_names)[0]
            if feature_name == easy_feature_name:
                return cfg
        return None

    def _fg_info_convert_feature(self, feature, fg_json):
        pyfg_key_2_feat_cfg_key = {
            "feature_name": "feature_name",
            "expression": "expression",
            "default_value": "default_value",
            "need_prefix": "need_prefix",
            "separator": "separator",
            "hash_bucket_size": "hash_bucket_size",
            "vocab_list": "vocab_list",
            "vocab_dict": "vocab_dict",
            "vocab_file": "vocab_file",
            "value_dim": "value_dim",
            "value_dimension": "value_dim",
            "default_bucketize_value": "default_bucketize_value",
            "stub_type": "stub_type",
            "operator_name": "operator_name",
            "operator_lib_file": "operator_lib_file",
            "is_op_thread_safe": "is_op_thread_safe",
            "normalizer": "normalizer",
            "boundaries": "boundaries",
            "variables": "variables",
            "num_buckets": "num_buckets",
            "weighted": "weighted",
            "kv_delimiter": "kv_delimiter",
            "query": "query",
            "document": "document",
            "needDiscrete": "need_discrete",
            "combiner": "combiner",
            "user": "nested_map",
            "category": "pkey",
            "item": "skey",
            "show_category": "show_pkey",
            "show_item": "show_skey",
            "title": "title",
            "method": "method",
            "tokenizer_type": "tokenizer_type",
        }
        filter_fg = {}
        for fg_k, ft_k in pyfg_key_2_feat_cfg_key.items():
            if fg_k in fg_json and hasattr(feature, ft_k):
                if isinstance(fg_json[fg_k], list):
                    attr = getattr(feature, ft_k)
                    attr.extend(fg_json[fg_k])
                else:
                    setattr(feature, ft_k, fg_json[fg_k])
            elif fg_k in fg_json:
                filter_fg[fg_k] = fg_json[fg_k]
        return feature, filter_fg

    def _create_feature_config_use_pyfg(self, pipeline_config):
        easyrec_feature_config = easyrec_feature_config_pb2.FeatureConfig()  # NOQA
        for fg_json in self.fg_json["features"]:
            feature_config = tzrec_feature_pb2.FeatureConfig()
            feature_config.ClearField("feature")
            if "feature_type" in fg_json:
                feature_type = fg_json["feature_type"]
                feature_name = fg_json["feature_name"]
                cfg = self._search_easyrec_config(feature_name)
                if feature_type == "combo_feature":
                    feature = tzrec_feature_pb2.ComboFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.combo_feature.CopyFrom(feature)
                elif feature_type == "custom_feature":
                    feature = tzrec_feature_pb2.CustomFeature()
                    feature, params = self._fg_info_convert_feature(feature, fg_json)
                    if len(params) > 0:
                        struct = struct_pb2.Struct()
                        struct.update(params)
                        feature.operator_params.CopyFrom(struct)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.custom_feature.CopyFrom(feature)
                elif feature_type == "expr_feature":
                    feature = tzrec_feature_pb2.ExprFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.expr_feature.CopyFrom(feature)
                elif feature_type == "id_feature":
                    feature = tzrec_feature_pb2.IdFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    if "combiner" in fg_json:
                        feature.pooling = fg_json["combiner"]
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.id_feature.CopyFrom(feature)
                elif feature_type == "kv_dot_product":
                    feature = tzrec_feature_pb2.KvDotProduct()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.kv_dot_product.CopyFrom(feature)
                elif feature_type == "lookup_feature":
                    feature = tzrec_feature_pb2.LookupFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.lookup_feature.CopyFrom(feature)
                elif feature_type == "match_feature":
                    feature = tzrec_feature_pb2.MatchFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.match_feature.CopyFrom(feature)
                elif feature_type == "overlap_feature":
                    feature = tzrec_feature_pb2.OverlapFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.overlap_feature.CopyFrom(feature)
                elif feature_type == "raw_feature":
                    feature = tzrec_feature_pb2.RawFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.raw_feature.CopyFrom(feature)
                elif feature_type == "tokenize_feature":
                    feature = tzrec_feature_pb2.TokenizeFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.tokenize_feature.CopyFrom(feature)
                elif feature_type == "bool_mask_feature":
                    feature = tzrec_feature_pb2.BoolMaskFeature()
                    feature, _ = self._fg_info_convert_feature(feature, fg_json)
                    self._easyrec_feature_2_tzrec(feature, cfg)
                    feature_config.bool_mask_feature.CopyFrom(feature)
                else:
                    logger.error(f"{feature_name} can't converted")
                    continue
                pipeline_config.feature_configs.append(feature_config)
            elif "sequence_name" in fg_json:
                seq_feature = tzrec_feature_pb2.SequenceFeature()
                seq_feature.sequence_name = fg_json["sequence_name"]
                seq_feature.sequence_length = fg_json["sequence_length"]
                seq_feature.sequence_delim = fg_json["sequence_delim"]
                seq_feature.sequence_pk = fg_json["sequence_pk"]
                for sub_fg_json in fg_json["features"]:
                    sub_feature_name = (
                        fg_json["sequence_name"] + "__" + sub_fg_json["feature_name"]
                    )
                    cfg = self._search_easyrec_config(sub_feature_name)
                    feature_type = sub_fg_json["feature_type"]
                    sub_feature_config = tzrec_feature_pb2.SeqFeatureConfig()
                    if feature_type == "id_feature":
                        feature = tzrec_feature_pb2.IdFeature()
                        feature, _ = self._fg_info_convert_feature(feature, sub_fg_json)
                        self._easyrec_feature_2_tzrec(feature, cfg)
                        if "combiner" in sub_fg_json:
                            feature.pooling = sub_fg_json["combiner"]
                        sub_feature_config.ClearField("feature")
                        sub_feature_config.id_feature.CopyFrom(feature)
                    else:
                        feature = tzrec_feature_pb2.RawFeature()
                        feature, _ = self._fg_info_convert_feature(feature, sub_fg_json)
                        self._easyrec_feature_2_tzrec(feature, cfg)
                        sub_feature_config.ClearField("feature")
                        sub_feature_config.raw_feature.CopyFrom(feature)
                    seq_feature.features.append(sub_feature_config)
                feature_config.sequence_feature.CopyFrom(seq_feature)
                pipeline_config.feature_configs.append(feature_config)
        return pipeline_config

    def _create_feature_config(self, pipeline_config):
        """Create tzrec feature config."""
        easyrec_feature_config = easyrec_feature_config_pb2.FeatureConfig()  # NOQA
        seq_group_cfg = OrderedDict()
        for cfg in self.easyrec_config.feature_configs:
            if cfg.feature_name:
                feature_name = cfg.feature_name
            else:
                feature_name = list(cfg.input_names)[0]
            input_names = cfg.input_names
            feature_type = cfg.feature_type

            if feature_name in self.feature_to_fg:
                fg_json = self.feature_to_fg[feature_name]
            elif feature_name in self.sub_sequence_to_group:
                pass
            elif input_names[0] in self.feature_to_fg:
                fg_json = self.feature_to_fg[input_names[0]]
            else:
                logger.error(f"in easyrec config {feature_name} not in fg.json")

            feature_config = None
            if feature_type == easyrec_feature_config.IdFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.IdFeature()
                feature.feature_name = feature_name
                feature.expression = fg_json["expression"]
                feature.embedding_dim = cfg.embedding_dim
                feature.hash_bucket_size = cfg.hash_bucket_size
                feature_config.ClearField("feature")
                feature_config.id_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.TagFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.IdFeature()
                feature.feature_name = feature_name
                feature.expression = fg_json["expression"]
                feature.embedding_dim = cfg.embedding_dim
                feature.hash_bucket_size = cfg.hash_bucket_size
                if cfg.HasField("kv_separator"):
                    feature.weighted = True
                feature_config.ClearField("feature")
                feature_config.id_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.SequenceFeature:
                if feature_name in self.sub_sequence_to_group:
                    sequence_name = self.sub_sequence_to_group[feature_name]
                    if sequence_name in seq_group_cfg:
                        seq_group_cfg[sequence_name].append(cfg)
                    else:
                        seq_group_cfg[sequence_name] = [cfg]
                elif feature_name in self.feature_to_fg:
                    feature_config = tzrec_feature_pb2.FeatureConfig()
                    if cfg.sub_feature_type == easyrec_feature_config.IdFeature:
                        feature = tzrec_feature_pb2.SequenceIdFeature()
                        feature.feature_name = feature_name
                        feature.expression = self.feature_to_fg[feature_name][
                            "expression"
                        ]
                        feature.embedding_dim = cfg.embedding_dim
                        feature.hash_bucket_size = cfg.hash_bucket_size
                        feature_config.ClearField("feature")
                        feature_config.sequence_id_feature.CopyFrom(feature)
                    else:
                        feature = tzrec_feature_pb2.SequenceRawFeature()
                        feature.feature_name = feature_name
                        feature.expression = self.feature_to_fg[feature_name][
                            "expression"
                        ]
                        boundaries = list(cfg.boundaries)
                        feature.embedding_dim = cfg.embedding_dim
                        if len(boundaries):
                            feature.boundaries.extend(boundaries)
                        feature_config.ClearField("feature")
                        feature_config.sequence_raw_feature.CopyFrom(feature)
                else:
                    logger.error(f"sequences feature: {feature_name} can't converted")
            elif feature_type == easyrec_feature_config.RawFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                if fg_json["feature_type"] == "lookup_feature":
                    feature = tzrec_feature_pb2.LookupFeature()
                    feature.feature_name = feature_name
                    map = fg_json["map"]
                    key = fg_json["key"]
                    boundaries = list(cfg.boundaries)
                    feature.feature_name = feature_name
                    feature.map = map
                    feature.key = key
                    feature.embedding_dim = cfg.embedding_dim
                    if len(boundaries):
                        feature.boundaries.extend(boundaries)
                    feature_config.ClearField("feature")
                    feature_config.lookup_feature.CopyFrom(feature)
                else:
                    feature = tzrec_feature_pb2.RawFeature()
                    feature.feature_name = feature_name
                    feature.expression = fg_json["expression"]
                    boundaries = list(cfg.boundaries)
                    feature.embedding_dim = cfg.embedding_dim
                    if len(boundaries):
                        feature.boundaries.extend(boundaries)
                    feature_config.ClearField("feature")
                    feature_config.raw_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.ComboFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.ComboFeature()
                feature.feature_name = feature_name
                for input in list(cfg.input_names):
                    if input in self.feature_to_fg:
                        tmp_fg_json = self.feature_to_fg[input]
                        feature.expression.append(tmp_fg_json["expression"])
                    else:
                        raise ValueError(f"{cfg} input_names:{input} not in fg json")
                feature.embedding_dim = cfg.embedding_dim
                feature.hash_bucket_size = cfg.hash_bucket_size
                feature_config.ClearField("feature")
                feature_config.combo_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.LookupFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.LookupFeature()
                feature.feature_name = feature_name
                map_f = cfg.input_names[0]
                key_f = cfg.input_names[1]
                if map_f in self.feature_to_fg:
                    feature.map = self.feature_to_fg[map_f]["expression"]
                else:
                    raise ValueError(f"{cfg} input names: {map_f} not in fg.json")
                if key_f in self.feature_to_fg:
                    feature.key = self.feature_to_fg[key_f]["expression"]
                else:
                    raise ValueError(f"{cfg} input names: {map_f} not in fg.json")
                feature.embedding_dim = cfg.embedding_dim
                if len(list(cfg.boundaries)):
                    feature.boundaries.extend(list(cfg.boundaries))
                feature_config.ClearField("feature")
                feature_config.lookup_feature.CopyFrom(feature)
            else:
                logger.error(f"{feature_name} can't converted")
            if feature_config is not None:
                pipeline_config.feature_configs.append(feature_config)
        for seq_name, sub_cfgs in seq_group_cfg.items():
            sequence_fg = self.sequence_feature_to_fg[seq_name]
            feature_config = tzrec_feature_pb2.FeatureConfig()
            sequence_feature_config = tzrec_feature_pb2.SequenceFeature()
            sequence_feature_config.sequence_name = sequence_fg["sequence_name"]
            sequence_feature_config.sequence_length = sequence_fg["sequence_length"]
            sequence_feature_config.sequence_delim = sequence_fg["sequence_delim"]
            features = sequence_fg["features"]
            seq_feature_to_fg = {}
            for feature in features:
                seq_feature_to_fg[f"{seq_name}__{feature['feature_name']}"] = feature
            for cfg in sub_cfgs:
                sub_feature_cfg = tzrec_feature_pb2.SeqFeatureConfig()
                feature_name = (
                    cfg.feature_name if cfg.feature_name else cfg.input_names[0]
                )
                if feature_name in seq_feature_to_fg:
                    seq_feature_fg = seq_feature_to_fg[feature_name]
                    if cfg.sub_feature_type == easyrec_feature_config.IdFeature:
                        feature = tzrec_feature_pb2.IdFeature()
                        feature.feature_name = seq_feature_fg["feature_name"]
                        feature.expression = seq_feature_fg["expression"]
                        feature.embedding_dim = cfg.embedding_dim
                        feature.hash_bucket_size = cfg.hash_bucket_size
                        sub_feature_cfg.ClearField("feature")
                        sub_feature_cfg.id_feature.CopyFrom(feature)
                    else:
                        feature = tzrec_feature_pb2.RawFeature()
                        feature.feature_name = seq_feature_fg["feature_name"]
                        feature.expression = seq_feature_fg["expression"]
                        boundaries = list(cfg.boundaries)
                        feature.embedding_dim = cfg.embedding_dim
                        if len(boundaries):
                            feature.boundaries.extend(boundaries)
                        sub_feature_cfg.ClearField("feature")
                        sub_feature_cfg.raw_feature.CopyFrom(feature)
                    sequence_feature_config.features.append(sub_feature_cfg)
                else:
                    logger.error(
                        f"sequence feature: {feature_name} not config in fg.json"
                    )

            feature_config.sequence_feature.CopyFrom(sequence_feature_config)
            pipeline_config.feature_configs.append(feature_config)

        return pipeline_config

    def _create_feature_config_no_fg(self, pipeline_config):
        """Create tzrec feature config no fg json."""
        easyrec_feature_config = easyrec_feature_config_pb2.FeatureConfig()  # NOQA
        for cfg in self.easyrec_config.feature_configs:
            if cfg.feature_name:
                feature_name = cfg.feature_name
            else:
                feature_name = list(cfg.input_names)[0]
            input_names = cfg.input_names
            feature_type = cfg.feature_type

            feature_config = None
            if feature_type == easyrec_feature_config.IdFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.IdFeature()
                feature.feature_name = feature_name
                feature.expression = f"user:{input_names[0]}"
                feature.embedding_dim = cfg.embedding_dim
                feature.hash_bucket_size = cfg.hash_bucket_size
                feature_config.ClearField("feature")
                feature_config.id_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.TagFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.IdFeature()
                feature.feature_name = feature_name
                feature.expression = f"user:{input_names[0]}"
                feature.embedding_dim = cfg.embedding_dim
                feature.hash_bucket_size = cfg.hash_bucket_size
                if cfg.HasField("kv_separator"):
                    feature.weighted = True
                feature_config.ClearField("feature")
                feature_config.id_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.SequenceFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                if cfg.sub_feature_type == easyrec_feature_config.RawFeature:
                    feature = tzrec_feature_pb2.SequenceRawFeature()
                    feature.feature_name = feature_name
                    feature.expression = f"user:{input_names[0]}"
                    feature.sequence_length = cfg.sequence_length
                    feature.sequence_delim = cfg.separator
                    feature.embedding_dim = cfg.embedding_dim
                    boundaries = list(cfg.boundaries)
                    if len(boundaries) > 0:
                        feature.boundaries.extend(boundaries)
                    feature_config.ClearField("feature")
                    feature_config.sequence_raw_feature.CopyFrom(feature)
                else:
                    feature = tzrec_feature_pb2.SequenceIdFeature()
                    feature.feature_name = feature_name
                    feature.expression = f"user:{input_names[0]}"
                    feature.sequence_length = cfg.sequence_length
                    feature.sequence_delim = cfg.separator
                    feature.embedding_dim = cfg.embedding_dim
                    if cfg.HasField("hash_bucket_size"):
                        feature.hash_bucket_size = cfg.hash_bucket_size
                    if cfg.HasField("num_buckets"):
                        feature.num_buckets = cfg.num_buckets
                    feature_config.ClearField("feature")
                    feature_config.sequence_id_feature.CopyFrom(feature)
                if cfg.sequence_length <= 1:
                    logger.error(f"{feature_name} sequence_length is invalid !!!")
            elif feature_type == easyrec_feature_config.RawFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.RawFeature()
                feature.feature_name = feature_name
                feature.expression = f"user:{input_names[0]}"
                boundaries = list(cfg.boundaries)
                if cfg.HasField("embedding_dim"):
                    feature.embedding_dim = cfg.embedding_dim
                if len(boundaries):
                    feature.boundaries.extend(boundaries)
                feature_config.ClearField("feature")
                feature_config.raw_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.ComboFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.ComboFeature()
                feature.feature_name = feature_name
                for input in list(cfg.input_names):
                    feature.expression.append(f"user:{input}")
                feature.embedding_dim = cfg.embedding_dim
                feature.hash_bucket_size = cfg.hash_bucket_size
                feature_config.ClearField("feature")
                feature_config.combo_feature.CopyFrom(feature)
            elif feature_type == easyrec_feature_config.LookupFeature:
                feature_config = tzrec_feature_pb2.FeatureConfig()
                feature = tzrec_feature_pb2.LookupFeature()
                feature.feature_name = feature_name
                feature.map = f"user:{input_names[0]}"
                feature.key = f"user:{input_names[1]}"
                if cfg.HasField("embedding_dim"):
                    feature.embedding_dim = cfg.embedding_dim
                if len(list(cfg.boundaries)):
                    feature.boundaries.extend(list(cfg.boundaries))
                feature_config.ClearField("feature")
                feature_config.lookup_feature.CopyFrom(feature)
            else:
                logger.error(f"{feature_name} can't converted")
            if feature_config is not None:
                logger.info(f"{feature_name} converted succeeded")
                pipeline_config.feature_configs.append(feature_config)

        return pipeline_config

    def _easyrec_dnn_2_tzrec_mlp(self, dnn):
        """Convert easyrec dnn to tzrec mlp."""
        mlp = module_pb2.MLP()
        mlp.hidden_units.extend(dnn.hidden_units)
        mlp.dropout_ratio.extend(dnn.dropout_ratio)
        mlp.use_bn = dnn.use_bn
        return mlp

    def _easyrec_loss_2_tzrec_loss(self, easyrec_loss):
        """Convert easyrec loss to tzrec loss."""
        tzrec_loss = loss_pb2.LossConfig()
        loss_type = easyrec_loss.loss_type
        if loss_type == easyrec_loss_pb2.LossType.JRC_LOSS:  # NOQA
            tzrec_loss.jrc_loss.CopyFrom(loss_pb2.JRCLoss())
        elif loss_type == easyrec_loss_pb2.LossType.L2_LOSS:  # NOQA
            tzrec_loss.l2_loss.CopyFrom(loss_pb2.L2Loss())
        elif loss_type == easyrec_loss_pb2.LossType.SOFTMAX_CROSS_ENTROPY:  # NOQA
            tzrec_loss.softmax_cross_entropy.CopyFrom(loss_pb2.SoftmaxCrossEntropy())
        elif loss_type == easyrec_loss_pb2.LossType.CLASSIFICATION:  # NOQA
            tzrec_loss.binary_cross_entropy.CopyFrom(loss_pb2.BinaryCrossEntropy())
        else:
            logger.error(
                f"{easyrec_loss} is not convert to tzrec loss, please adaptation"
            )
        return tzrec_loss

    def _easyrec_metrics_2_tzrec_metrics(self, easyrec_metric):
        """Convert easyrec metric to tzrec metric."""
        metric = metric_pb2.MetricConfig()
        metric_type = easyrec_metric.WhichOneof("metric")
        easyrec_metric_ob = getattr(easyrec_metric, metric_type)
        if metric_type == "auc":
            metric.auc.CopyFrom(metric_pb2.AUC())
        elif metric_type == "gauc":
            tzrec_metric_ob = metric_pb2.GroupedAUC(
                grouping_key=easyrec_metric_ob.uid_field
            )
            metric.grouped_auc.CopyFrom(tzrec_metric_ob)
        elif metric_type == "recall_at_topk":
            metric.recall_at_k.CopyFrom(metric_pb2.RecallAtK())
        elif metric_type == "mean_absolute_error":
            metric.mean_absolute_error.CopyFrom(metric_pb2.MeanAbsoluteError())
        elif metric_type == "mean_squared_error":
            metric.mean_squared_error.CopyFrom(metric_pb2.MeanSquaredError())
        elif metric_type == "accuracy":
            metric.accuracy.CopyFrom(metric_pb2.Accuracy())
        else:
            logger.error(
                f"{easyrec_metric} is not convert to tzrec metric, please adaptation"
            )
        return metric

    def _easyrec_bayes_tower_2_tzrec_bayes_tower(self, easyrec_bayes_task_tower):
        """Convert easyrec bayes tower to tzrec bayes tower."""
        tzrec_bayes_task_tower = tower_pb2.BayesTaskTower()
        tzrec_bayes_task_tower.tower_name = easyrec_bayes_task_tower.tower_name
        tzrec_bayes_task_tower.label_name = easyrec_bayes_task_tower.label_name
        tzrec_bayes_task_tower.num_class = easyrec_bayes_task_tower.num_class
        tzrec_bayes_task_tower.relation_tower_names.extend(
            easyrec_bayes_task_tower.relation_tower_names
        )
        mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_bayes_task_tower.dnn)
        tzrec_bayes_task_tower.mlp.CopyFrom(mlp)
        relation_mlp = self._easyrec_dnn_2_tzrec_mlp(
            easyrec_bayes_task_tower.relation_dnn
        )
        tzrec_bayes_task_tower.relation_mlp.CopyFrom(relation_mlp)
        for loss in easyrec_bayes_task_tower.losses:
            tzrec_bayes_task_tower.losses.append(self._easyrec_loss_2_tzrec_loss(loss))
        for metric in easyrec_bayes_task_tower.metrics_set:
            tzrec_bayes_task_tower.metrics.append(
                self._easyrec_metrics_2_tzrec_metrics(metric)
            )
        return tzrec_bayes_task_tower

    def _easyrec_task_tower_2_tzrec_task_tower(self, easyrec_task_tower):
        """Convert easyrec task tower to tzrec task tower."""
        tzrec_task_tower = tower_pb2.TaskTower()
        tzrec_task_tower.tower_name = easyrec_task_tower.tower_name
        tzrec_task_tower.label_name = easyrec_task_tower.label_name
        tzrec_task_tower.num_class = easyrec_task_tower.num_class
        mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_task_tower.dnn)
        tzrec_task_tower.mlp.CopyFrom(mlp)
        for loss in easyrec_task_tower.losses:
            tzrec_task_tower.losses.append(self._easyrec_loss_2_tzrec_loss(loss))
        for metric in easyrec_task_tower.metrics_set:
            tzrec_task_tower.metrics.append(
                self._easyrec_metrics_2_tzrec_metrics(metric)
            )
        return tzrec_task_tower

    def _easyrec_tower_2_tzrec_tower(self, easyrec_tower):
        """Convert easyrec tower to tzrec tower."""
        tzrec_tower = tower_pb2.Tower()
        tzrec_tower.input = easyrec_tower.input
        mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_tower.dnn)
        tzrec_tower.mlp.CopyFrom(mlp)
        return tzrec_tower

    def _easyrec_dssm_tower_2_tzrec_tower(self, easyrec_dssm_tower):
        """Convert easyrec dssm tower to tzrec tower."""
        tzrec_tower = tower_pb2.Tower()
        tzrec_tower.input = easyrec_dssm_tower.id
        mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_dssm_tower.dnn)
        tzrec_tower.mlp.CopyFrom(mlp)
        return tzrec_tower

    def _easyrec_extraction_network_2_tzrec_extraction_network(
        self, easyrec_extraction_network
    ):
        """Convert easyrec extraction net to tzrec extraction net."""
        tzrec_extraction_network = module_pb2.ExtractionNetwork()
        tzrec_extraction_network.network_name = easyrec_extraction_network.network_name
        tzrec_extraction_network.expert_num_per_task = (
            easyrec_extraction_network.expert_num_per_task
        )
        tzrec_extraction_network.share_num = easyrec_extraction_network.share_num
        task_expert_net = self._easyrec_dnn_2_tzrec_mlp(
            easyrec_extraction_network.task_expert_net
        )
        tzrec_extraction_network.task_expert_net.CopyFrom(task_expert_net)
        share_expert_net = self._easyrec_dnn_2_tzrec_mlp(
            easyrec_extraction_network.share_expert_net
        )
        tzrec_extraction_network.share_expert_net.CopyFrom(share_expert_net)
        return tzrec_extraction_network

    def _convert_model_feature_group(self, easyrec_feature_groups):
        """Convert easyrec feature group to tzrec feature group."""
        tz_feature_groups = []
        for easy_feature_group in easyrec_feature_groups:
            tz_feature_group = model_pb2.FeatureGroupConfig()
            tz_feature_group.group_name = easy_feature_group.group_name
            tz_feature_group.feature_names.extend(easy_feature_group.feature_names)
            if (
                easy_feature_group.wide_deep
                == easyrec_feature_config_pb2.WideOrDeep.WIDE  # NOQA
            ):
                tz_feature_group.group_type = model_pb2.FeatureGroupType.WIDE
            else:
                tz_feature_group.group_type = model_pb2.FeatureGroupType.DEEP
            for i, easyrec_sequence_group in enumerate(
                easy_feature_group.sequence_features
            ):
                tz_seq_group = model_pb2.SeqGroupConfig()
                tz_seq_encoder = seq_encoder_pb2.SeqEncoderConfig()
                seq_encoder = seq_encoder_pb2.DINEncoder()
                if easyrec_sequence_group.HasField("group_name"):
                    group_name = easyrec_sequence_group.group_name
                else:
                    group_name = f"seq_{i}"
                tz_seq_group.group_name = group_name
                seq_encoder.input = group_name
                mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_sequence_group.seq_dnn)
                seq_encoder.attn_mlp.CopyFrom(mlp)
                tz_seq_encoder.din_encoder.CopyFrom(seq_encoder)
                for seq_att_map in easyrec_sequence_group.seq_att_map:
                    tz_seq_group.feature_names.extend(seq_att_map.key)
                    tz_seq_group.feature_names.extend(seq_att_map.hist_seq)
                    tz_seq_group.feature_names.extend(seq_att_map.aux_hist_seq)
                tz_feature_group.sequence_groups.append(tz_seq_group)
                tz_feature_group.sequence_encoders.append(tz_seq_encoder)
            tz_feature_groups.append(tz_feature_group)
        return tz_feature_groups

    def _convert_model_config(self, easyrec_model_config, tz_model_config):
        """Convert easyrec model config to tzrec model config."""
        model_class = easyrec_model_config.model_class
        model_type = easyrec_model_config.WhichOneof("model")
        easyrec_model_config = getattr(easyrec_model_config, model_type)
        if model_class == "DBMTL":
            tz_model_config_ob = multi_task_rank_pb2.DBMTL()
            bottom_mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_model_config.bottom_dnn)
            expert_mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_model_config.expert_dnn)
            tz_model_config_ob.bottom_mlp.CopyFrom(bottom_mlp)
            tz_model_config_ob.expert_mlp.CopyFrom(expert_mlp)
            tz_model_config_ob.num_expert = easyrec_model_config.num_expert
            for task_tower in easyrec_model_config.task_towers:
                tz_task_tower = self._easyrec_bayes_tower_2_tzrec_bayes_tower(
                    task_tower
                )
                tz_model_config_ob.task_towers.append(tz_task_tower)
            tz_model_config.dbmtl.CopyFrom(tz_model_config_ob)
        elif model_class == "SimpleMultiTask":
            tz_model_config_ob = multi_task_rank_pb2.SimpleMultiTask()
            for task_tower in easyrec_model_config.task_towers:
                tz_task_tower = self._easyrec_task_tower_2_tzrec_task_tower(task_tower)
                tz_model_config_ob.task_towers.append(tz_task_tower)
            tz_model_config.simple_multi_task.CopyFrom(tz_model_config_ob)
        elif model_class == "MMoE":
            tz_model_config_ob = multi_task_rank_pb2.MMoE()
            expert_mlp = self._easyrec_dnn_2_tzrec_mlp(easyrec_model_config.expert_dnn)
            tz_model_config_ob.expert_mlp.CopyFrom(expert_mlp)
            tz_model_config_ob.gate_mlp.CopyFrom(expert_mlp)
            tz_model_config_ob.num_expert = easyrec_model_config.num_expert
            for task_tower in easyrec_model_config.task_towers:
                tz_task_tower = self._easyrec_task_tower_2_tzrec_task_tower(task_tower)
                tz_model_config_ob.task_towers.append(tz_task_tower)
            tz_model_config.mmoe.CopyFrom(tz_model_config_ob)
        elif model_class == "PLE":
            tz_model_config_ob = multi_task_rank_pb2.PLE()
            for extraction_network in easyrec_model_config.extraction_networks:
                tz_extraction_network = (
                    self._easyrec_extraction_network_2_tzrec_extraction_network(
                        extraction_network
                    )
                )
                tz_model_config.ple.extraction_networks.append(tz_extraction_network)
            for task_tower in easyrec_model_config.task_towers:
                tz_task_tower = self._easyrec_task_tower_2_tzrec_task_tower(task_tower)
                tz_model_config_ob.task_towers.append(tz_task_tower)
            tz_model_config.ple.CopyFrom(tz_model_config_ob)
        elif model_class == "DeepFM":
            tz_model_config_ob = rank_model_pb2.DeepFM()
            deep = self._easyrec_dnn_2_tzrec_mlp(easyrec_model_config.dnn)
            final = self._easyrec_dnn_2_tzrec_mlp(easyrec_model_config.final_dnn)
            tz_model_config_ob.deep.CopyFrom(deep)
            tz_model_config_ob.final.CopyFrom(final)
            if easyrec_model_config.HasField("wide_output_dim"):
                tz_model_config_ob.wide_embedding_dim = (
                    easyrec_model_config.wide_output_dim
                )
            tz_model_config.deepfm.CopyFrom(tz_model_config_ob)
        elif model_class == "MultiTower":
            tz_model_config_ob = rank_model_pb2.MultiTower()
            for tower in easyrec_model_config.towers:
                tz_tower = self._easyrec_tower_2_tzrec_tower(tower)
                tz_model_config_ob.towers.append(tz_tower)
            final = self._easyrec_dnn_2_tzrec_mlp(easyrec_model_config.final_dnn)
            tz_model_config_ob.final.CopyFrom(final)
            tz_model_config.multi_tower.CopyFrom(tz_model_config_ob)
        elif model_class == "DSSM":
            tz_model_config_ob = match_model_pb2.DSSM()
            user_tower = self._easyrec_dssm_tower_2_tzrec_tower(
                easyrec_model_config.user_tower
            )
            tz_model_config_ob.user_tower.CopyFrom(user_tower)
            item_tower = self._easyrec_dssm_tower_2_tzrec_tower(
                easyrec_model_config.item_tower
            )
            tz_model_config_ob.item_tower.CopyFrom(item_tower)
            tz_model_config_ob.output_dim = 32
            if hasattr(
                easyrec_model_config, "temperature"
            ) and easyrec_model_config.HasField("temperature"):
                tz_model_config_ob.temperature = easyrec_model_config.temperature
            tz_model_config.dssm.CopyFrom(tz_model_config_ob)
        else:
            logger.error(
                f"{model_class} is not convert to tzrec model, please adaptation"
            )
        return tz_model_config

    def _create_model_config(self, pipeline_config):
        """Convert easyrec model config to tzrec model config."""
        tz_model_config = model_pb2.ModelConfig()
        easyrec_model_config = self.easyrec_config.model_config
        easyrec_feature_groups = easyrec_model_config.feature_groups
        tz_feature_groups = self._convert_model_feature_group(easyrec_feature_groups)
        tz_model_config.feature_groups.extend(tz_feature_groups)
        tz_model_config = self._convert_model_config(
            easyrec_model_config, tz_model_config
        )
        pipeline_config.model_config.CopyFrom(tz_model_config)
        return pipeline_config

    def build(self):
        """Create tzrec model config order by easyrec config and fg file."""
        tzrec_config = tzrec_pipeline_pb2.EasyRecConfig()
        tzrec_config = self._create_train_config(tzrec_config)
        tzrec_config = self._create_eval_config(tzrec_config)
        tzrec_config = self._create_data_config(tzrec_config)
        if len(self.feature_to_fg):
            if self.use_old_fg:
                tzrec_config = self._create_feature_config(tzrec_config)
            else:
                tzrec_config = self._create_feature_config_use_pyfg(tzrec_config)
        else:
            tzrec_config = self._create_feature_config_no_fg(tzrec_config)
        tzrec_config = self._create_model_config(tzrec_config)
        config_text = text_format.MessageToString(tzrec_config, as_utf8=True)
        with open(self.output_tzrec_config_path, "w") as f:
            f.write(config_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easyrec_config_path",
        type=str,
        default=None,
        help="easyrec model config path",
    )
    parser.add_argument(
        "--fg_json_path", type=str, default=None, help="easyrec use fg.json path"
    )
    parser.add_argument(
        "--use_old_fg",
        action="store_true",
        default=False,
        help="if true will create tzrec based on easyrec or based on pyfg",
    )
    parser.add_argument(
        "--output_tzrec_config_path",
        type=str,
        default=None,
        help="output tzrec config path",
    )
    parser.add_argument(
        "--easyrec_package_path",
        type=str,
        default=None,
        help="easyrec whl or tar package path or url",
    )
    args, extra_args = parser.parse_known_args()
    fs = ConvertConfig(
        args.easyrec_config_path,
        args.output_tzrec_config_path,
        args.fg_json_path,
        args.use_old_fg,
        args.easyrec_package_path,
    )
    fs.build()
