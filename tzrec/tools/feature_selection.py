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
import os
from collections import OrderedDict
from typing import Dict, Optional

try:
    # pyre-ignore [21]
    import matplotlib.pyplot as plt
except Exception:
    print("please install matplotlib !!!")
import pandas as pd
from torch.distributed.checkpoint import load

from tzrec.main import _create_features, _create_model, init_process_group
from tzrec.models.model import ScriptWrapper
from tzrec.protos.model_pb2 import FeatureGroupType
from tzrec.utils import checkpoint_util, config_util
from tzrec.utils.logging_util import logger


class VariationalDropoutFS:
    """Feature selection for group variational dropout.

    Args:
        config_path (str): feature selection model config path.
        model_dir (str): feature selection model checkpoint path.
        output_dir (str): feature selection result directory.
        topk (int): select topk importance features for each feature group.
        fg_path (str): fg config path.
        visualize (boolean): visualization feature selection result or not.
    """

    def __init__(
        self,
        config_path: str,
        model_dir: str,
        output_dir: str,
        topk: int,
        fg_path: Optional[str] = None,
        visualize: Optional[bool] = False,
        clear_variational_dropout: bool = True,
    ) -> None:
        self._config_path = config_path
        self._model_dir = model_dir
        self._output_dir = output_dir
        self._topk = topk
        self._fg_path = fg_path
        self._visualize = visualize
        self._clear_variational_dropout = clear_variational_dropout

    def _feature_dim_dropout_ratio(self) -> Dict[str, Dict[str, float]]:
        """Get dropout ratio of feature_groups feature-wise."""
        pipeline_config = config_util.load_pipeline_config(self._config_path)
        data_config = pipeline_config.data_config
        # Build feature
        features = _create_features(list(pipeline_config.feature_configs), data_config)
        model = _create_model(
            pipeline_config.model_config,
            features,
            list(data_config.label_fields),
        )
        model = ScriptWrapper(model)
        checkpoint_path, _ = checkpoint_util.latest_checkpoint(self._model_dir)
        if checkpoint_path:
            model_ckpt_path = os.path.join(checkpoint_path, "model")
            logger.info(
                f"Restoring model feature dropout ratio from {model_ckpt_path}..."
            )
            state_dict = model.state_dict()
            new_state_dict = {
                k: v for k, v in state_dict.items() if "group_variational_dropouts" in k
            }
            load(
                new_state_dict,
                checkpoint_id=model_ckpt_path,
            )
        else:
            raise ValueError("checkpoint path should be specified.")

        group_feature_importance = {}
        for name, sub_model in model.named_modules():
            if "group_variational_dropouts" == name.split(".")[-1]:
                for variational_dropout in sub_model.values():
                    group_name = variational_dropout.group_name
                    values = variational_dropout.feature_p.sigmoid().tolist()
                    feature_names = variational_dropout.features_dimension.keys()
                    group_feature_p = {
                        feature_name: dropout
                        for feature_name, dropout in zip(feature_names, values)
                    }
                    group_feature_importance[group_name] = group_feature_p

        if len(group_feature_importance) == 0:
            raise ValueError(
                "you not configure variational dropout "
                "or no group can be variational dropout."
            )
        return group_feature_importance

    def _dump_to_csv(
        self, feature_importance: Dict[str, float], group_name: str
    ) -> None:
        """Dump feature importance data to a csv file."""
        csv_path = os.path.join(
            self._output_dir, "feature_dropout_ratio_%s.csv" % group_name
        )
        df = pd.DataFrame(
            # pyre-fixme [6]
            columns=["feature_name", "mean_drop_p"],
            data=[list(kv) for kv in feature_importance.items()],
        )
        df.to_csv(csv_path, index=None)

    def _visualize_feature_importance(
        self, feature_importance: Dict[str, float], group_name: str
    ) -> None:
        """Draw feature importance histogram."""
        df = pd.DataFrame(
            # pyre-fixme [6]
            columns=["feature_name", "mean_drop_p"],
            data=[list(kv) for kv in feature_importance.items()],
        )
        df["color"] = ["red" if x < 0.5 else "green" for x in df["mean_drop_p"]]
        df.sort_values("mean_drop_p", inplace=True, ascending=False)
        df.reset_index(inplace=True)
        # Draw plot
        plt.figure(figsize=(90, 200), dpi=100)
        plt.hlines(y=df.index, xmin=0, xmax=df.mean_drop_p)
        for x, y, tex in zip(df.mean_drop_p, df.index, df.mean_drop_p):
            plt.text(
                x,
                y,
                round(tex, 2),
                horizontalalignment="right" if x < 0 else "left",
                verticalalignment="center",
                fontdict={"color": "red" if x < 0 else "green", "size": 14},
            )
        # Decorations
        plt.yticks(df.index, df.feature_name, fontsize=20)
        plt.title("Dropout Ratio", fontdict={"size": 30})
        plt.grid(linestyle="--", alpha=0.5)
        plt.xlim(0, 1)
        png = os.path.join(self._output_dir, "feature_dropout_pic_%s.png" % group_name)
        plt.savefig(png, format="png")

    def _process_config(
        self, feature_importance_map: Dict[str, Dict[str, float]]
    ) -> None:
        """Process model config with feature selection."""
        config = config_util.load_pipeline_config(self._config_path)
        useful_features = set()
        for feature_importance in feature_importance_map.values():
            for i, (feature_name, _) in enumerate(feature_importance.items()):
                if i < self._topk:
                    useful_features.add(feature_name)

        for feature_group in config.model_config.feature_groups:
            group_type = feature_group.group_type
            if group_type == FeatureGroupType.SEQUENCE:
                for feature_name in feature_group.feature_names:
                    useful_features.add(feature_name)
            else:
                for sequence_groups in feature_group.sequence_groups:
                    for feature_name in sequence_groups.feature_names:
                        useful_features.add(feature_name)

        feature_configs = []
        for feature_config in config.feature_configs:
            feat_type = feature_config.WhichOneof("feature")
            if feat_type == "sequence_feature":
                feature_configs.append(feature_config)
            else:
                feature = getattr(feature_config, feat_type)
                feature_name = feature.feature_name
                if feature_name in useful_features:
                    feature_configs.append(feature_config)
        config.ClearField("feature_configs")
        config.feature_configs.extend(feature_configs)

        for feature_group in config.model_config.feature_groups:
            group_name = feature_group.group_name
            if group_name in feature_importance_map:
                feature_importance = feature_importance_map[group_name]
                group_useful_features = set()
                for i, (feature_name, _) in enumerate(feature_importance.items()):
                    if i < self._topk:
                        group_useful_features.add(feature_name)

                feature_names = []
                for feature_name in feature_group.feature_names:
                    if feature_name in group_useful_features:
                        feature_names.append(feature_name)
                feature_group.ClearField("feature_names")
                feature_group.feature_names.extend(feature_names)

        if self._clear_variational_dropout:
            config.model_config.ClearField("variational_dropout")

        config_util.save_message(
            config, os.path.join(self._output_dir, os.path.basename(self._config_path))
        )

    def process(self) -> None:
        """Feature selection process method."""
        logger.info("Loading logit_p of VariationalDropout layer ...")
        is_rank_zero = int(os.environ.get("RANK", 0)) == 0
        device, _ = init_process_group()
        if is_rank_zero:
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            group_feature_importance = self._feature_dim_dropout_ratio()
            feature_importance_map = {}
            for group_name, feature_dim_dropout_p in group_feature_importance.items():
                feature_importance = OrderedDict(
                    sorted(feature_dim_dropout_p.items(), key=lambda e: e[1])
                )
                feature_importance_map[group_name] = feature_importance
                logger.info("Dump %s feature importance to csv ..." % group_name)
                self._dump_to_csv(feature_importance, group_name)
                if self._visualize:
                    logger.info("Visualizing %s feature importance ..." % group_name)
                    self._visualize_feature_importance(feature_importance, group_name)
            logger.info("Processing model config ...")
            self._process_config(feature_importance_map)
            logger.info("feature selection complete ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="checkpoint to be evaled, if not specified, use the latest checkpoint in "
        "train_config.model_dir.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="feature selection out put directory",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="select topk importance features for each feature group",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="visualization feature selection result or not",
    )
    parser.add_argument(
        "--clear_variational_dropout",
        type=bool,
        default=True,
        help="visualization feature selection result or not",
    )
    args, extra_args = parser.parse_known_args()
    fs = VariationalDropoutFS(
        args.pipeline_config_path,
        args.model_dir,
        args.output_dir,
        args.topk,
        visualize=args.visualize,
        clear_variational_dropout=args.clear_variational_dropout,
    )
    fs.process()
