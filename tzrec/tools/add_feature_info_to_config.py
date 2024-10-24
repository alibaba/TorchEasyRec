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
import json
from typing import Any, Dict, List, Tuple

from tzrec.datasets.dataset import create_reader
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


class AddFeatureInfoToConfig(object):
    """Add feature_info to config file.

    Args:
        template_model_config_path (str): template model config path.
        model_config_path (str): model_config_path.
        config_table_path (str): feature config info path.
        reader_type (str): input path reader type.
        odps_data_quota_name (str):maxcompute storage api/tunnel data quota name.
    """

    def __init__(
        self,
        template_model_config_path: str,
        model_config_path: str,
        config_table_path: str,
        reader_type: str,
        odps_data_quota_name: str,
    ):
        self.template_model_config_path = template_model_config_path
        self.model_config_path = model_config_path
        self.config_table_path = config_table_path
        self.reader_type = reader_type
        self.odps_data_quota_name = odps_data_quota_name

    def _load_feature_info(self) -> Tuple[Dict[str, Any], List[str]]:
        """Load feature info for update config."""
        feature_info_map = {}
        drop_feature_names = []
        sels = ["feature", "feature_info", "message"]
        reader = create_reader(
            self.config_table_path,
            1,
            selected_cols=sels,
            reader_type=self.reader_type,
            odps_data_quota_name=self.odps_data_quota_name,
        )
        for data in reader.to_batches():
            feature_names = data["feature"].tolist()
            feature_infos = data["feature_info"].tolist()
            messages = data["message"].tolist()
            for record in zip(feature_names, feature_infos, messages):
                feature_name = record[0]
                feature_info_map[feature_name] = json.loads(record[1])
                if record[2] is not None and "DROP IT" in record[2]:
                    drop_feature_names.append(feature_name)

        return feature_info_map, drop_feature_names

    def _drop_feature_config(self, pipeline_config, drop_feature_names) -> None:
        """Drop invalid feature config."""
        feature_configs = pipeline_config.feature_configs
        filter_feature_configs = []
        if drop_feature_names:
            for fea_cfg in feature_configs:
                oneof_feat_config = getattr(fea_cfg, fea_cfg.WhichOneof("feature"))
                feat_cls_name = oneof_feat_config.__class__.__name__
                if feat_cls_name == "SequenceFeature":
                    sequence_name = oneof_feat_config.sequence_name
                    sub_features = oneof_feat_config.features[:]
                    for sub_feat_config in sub_features:
                        feat_config = getattr(
                            sub_feat_config, sub_feat_config.WhichOneof("feature")
                        )
                        name = f"{sequence_name}__{feat_config.feature_name}"
                        if name in drop_feature_names:
                            oneof_feat_config.features.remove(sub_feat_config)
                            logger.info(f"drop sub sequence feature: {name}")
                    if len(oneof_feat_config.features) == 0:
                        feature_configs.remove(fea_cfg)
                        logger.info(f"drop sequence feature: {sequence_name}")
                    else:
                        filter_feature_configs.append(fea_cfg)
                else:
                    if oneof_feat_config.feature_name in drop_feature_names:
                        feature_configs.remove(fea_cfg)
                        logger.info(f"drop feature: {oneof_feat_config.feature_name}")
                    else:
                        filter_feature_configs.append(fea_cfg)
            pipeline_config.ClearField("feature_configs")
            pipeline_config.feature_configs.extend(feature_configs)

    def _update_feature_config(self, pipeline_config, feature_info_map) -> List[str]:
        """Add feature info to feature config."""
        feature_configs = pipeline_config.feature_configs
        general_feature = []
        for fea_cfg in feature_configs:
            feature_config = getattr(fea_cfg, fea_cfg.WhichOneof("feature"))
            feat_cls_name = feature_config.__class__.__name__
            if feat_cls_name == "SequenceFeature":
                sequence_name = feature_config.sequence_name
                sub_features = feature_config.features
                for sub_feat in sub_features:
                    sub_feat_config = getattr(sub_feat, sub_feat.WhichOneof("feature"))
                    feature_name = f"{sequence_name}__{sub_feat_config.feature_name}"
                    if feature_name in feature_info_map:
                        logger.info("edited %s" % feature_name)
                        sub_feat_config.embedding_dim = int(
                            feature_info_map[feature_name]["embedding_dim"]
                        )
                        if "boundary" in feature_info_map[feature_name]:
                            sub_feat_config.ClearField("boundaries")
                            sub_feat_config.boundaries.extend(
                                [
                                    float(i)
                                    for i in feature_info_map[feature_name]["boundary"]
                                ]
                            )
                        elif "hash_bucket_size" in feature_info_map[feature_name]:
                            sub_feat_config.hash_bucket_size = int(
                                feature_info_map[feature_name]["hash_bucket_size"]
                            )
                    else:
                        logger.error(
                            f"please check: {feature_name}, this config no info..."
                        )
            else:
                feature_name = feature_config.feature_name
                general_feature.append(feature_name)
                if feature_name in feature_info_map:
                    logger.info("edited %s" % feature_name)
                    feature_config.embedding_dim = int(
                        feature_info_map[feature_name]["embedding_dim"]
                    )
                    if "boundary" in feature_info_map[feature_name]:
                        feature_config.ClearField("boundaries")
                        feature_config.boundaries.extend(
                            [
                                float(i)
                                for i in feature_info_map[feature_name]["boundary"]
                            ]
                        )
                    elif "hash_bucket_size" in feature_info_map[feature_name]:
                        feature_config.hash_bucket_size = int(
                            feature_info_map[feature_name]["hash_bucket_size"]
                        )
                else:
                    logger.error(
                        f"please check: {feature_name}, this config no info..."
                    )
        return general_feature

    def _update_feature_group(
        self, pipeline_config, drop_feature_names, general_feature
    ) -> None:
        """Drop feature name for feature group."""
        for feature_group in pipeline_config.model_config.feature_groups:
            feature_names = feature_group.feature_names
            reserved_features = []
            for feature_name in feature_names:
                if feature_name not in drop_feature_names:
                    reserved_features.append(feature_name)
                else:
                    logger.info("feature group drop feature: %s" % feature_name)
            feature_group.ClearField("feature_names")
            feature_group.feature_names.extend(reserved_features)
            del_sequence_groups = []
            for sequence_group in feature_group.sequence_groups:
                reserved_features = []
                seq_feature_num = 0
                for feature_name in sequence_group.feature_names:
                    if feature_name not in drop_feature_names:
                        reserved_features.append(feature_name)
                        if feature_name not in general_feature:
                            seq_feature_num += 1
                    else:
                        logger.info("sequence group drop feature: %s" % feature_name)
                sequence_group.ClearField("feature_names")
                sequence_group.feature_names.extend(reserved_features)
                if seq_feature_num == 0:
                    del_sequence_groups.append(sequence_group.group_name)
                    feature_group.sequence_groups.remove(sequence_group)
                    logger.info("drop sequence group: %s" % sequence_group.group_name)
            for seq_encoded in feature_group.sequence_encoders:
                seq_module = getattr(seq_encoded, seq_encoded.WhichOneof("seq_module"))
                if seq_module.input in del_sequence_groups:
                    feature_group.sequence_encoders.remove(seq_encoded)
                    logger.info("drop sequence encoder: %s" % seq_module.input)

    def build(self) -> None:
        """Build method."""
        feature_info_map, drop_feature_names = self._load_feature_info()
        pipeline_config = config_util.load_pipeline_config(
            self.template_model_config_path
        )
        self._drop_feature_config(pipeline_config, drop_feature_names)
        general_feature = self._update_feature_config(pipeline_config, feature_info_map)
        self._update_feature_group(pipeline_config, drop_feature_names, general_feature)
        config_util.save_message(pipeline_config, self.model_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template_model_config_path",
        type=str,
        default=None,
        help="template model config path",
    )
    parser.add_argument(
        "--model_config_path", type=str, default=None, help="new model config path"
    )
    parser.add_argument(
        "--config_table_path", type=str, default=None, help="feature config info path"
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        default="OdpsReader",
        choices=["OdpsReader", "CsvReader", "ParquetReader"],
        help="input path reader type.",
    )
    parser.add_argument(
        "--odps_data_quota_name",
        type=str,
        default="pay-as-you-go",
        help="maxcompute storage api/tunnel data quota name.",
    )
    args, extra_args = parser.parse_known_args()
    fs = AddFeatureInfoToConfig(
        args.template_model_config_path,
        args.model_config_path,
        args.config_table_path,
        args.reader_type,
        args.odps_data_quota_name,
    )
    fs.build()
