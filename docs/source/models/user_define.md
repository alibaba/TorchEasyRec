# 自定义模型

TorchEasyRec 支持从独立 Python 包加载自定义模型。用户无需修改
`tzrec/protos/model.proto` 或 `tzrec/protos/models/` 下的公共配置，模型代码和
配置 proto 可以单独维护，减少升级 TorchEasyRec 时的代码冲突。

## 目录结构

默认自定义包名为 `tzrec_custom`，推荐使用以下结构：

```text
tzrec_custom/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── custom_rank_model.py
└── protos/
    ├── __init__.py
    └── custom_rank_model.proto
```

一个自定义包可以包含多个模型。TorchEasyRec 启动时会递归导入
`tzrec_custom.models` 下的非测试模块，具体使用哪个模型由 pipeline config 中的
`class_path` 决定。

如需使用其他包名，在第一次导入 `tzrec` 前设置环境变量：

```bash
export TZREC_CUSTOM_PACKAGE=my_project.tzrec_custom
```

环境变量填写完整 Python 包名。运行命令时还需通过 `PYTHONPATH` 或安装 wheel
确保该包可以被 Python 导入。显式配置的包不存在时，TorchEasyRec 会直接报错；
未配置环境变量且默认 `tzrec_custom` 不存在时则保持原有行为。

## 编写模型配置 proto

自定义 proto 可以直接复用 TorchEasyRec 的公共 message。以下模型配置使用了
`tzrec.protos.MLP`：

```protobuf
syntax = "proto2";
package tzrec_custom.protos;

import "tzrec/protos/module.proto";

message CustomRankModelConfig {
    required tzrec.protos.MLP mlp = 1;
}
```

生成 Python binding：

```bash
PYTHONPATH=. bash scripts/gen_proto.sh
```

`scripts/gen_proto.sh` 会在生成公共 proto 后，检查
`tzrec_custom/protos/*.proto` 并生成对应的 `*_pb2.py` 和 `*_pb2.pyi`。自定义包名
不是 `tzrec_custom` 时，生成命令需要使用相同的环境变量：

```bash
TZREC_CUSTOM_PACKAGE=my_project.tzrec_custom \
PYTHONPATH=. bash scripts/gen_proto.sh
```

如果自定义包作为独立 wheel 发布，也可以在该包的构建流程中自行生成 binding。

## 编写模型

自定义模型需要继承 `tzrec.models.model.BaseModel`。排序、多目标排序和召回场景
通常可以分别继承：

- `tzrec.models.rank_model.RankModel`
- `tzrec.models.multi_task_rank.MultiTaskRank`
- `tzrec.models.match_model.MatchModel`

模型模块必须在顶层导入对应的 `*_pb2.py`。这样自动导入模型时会同时注册
protobuf descriptor，pipeline config 中的 `Any` 配置才能被解析。

以下代码展示了排序模型的主要结构：

```python
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs
from tzrec_custom.protos.custom_rank_model_pb2 import CustomRankModelConfig


class CustomRankModel(RankModel):
    """Example custom ranking model."""

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        custom_config: Optional[CustomRankModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_config,
            features,
            labels,
            sample_weights,
            custom_config=custom_config,
            **kwargs,
        )
        self.embedding_group = EmbeddingGroup(features, self.feature_groups)
        input_dim = sum(
            self.embedding_group.group_total_dim(name)
            for name in self.embedding_group.group_names()
        )
        self.mlp = MLP(
            in_features=input_dim,
            **config_to_kwargs(self._model_config.mlp),
        )
        self.output = nn.Linear(self.mlp.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Run model prediction."""
        grouped_features = self.embedding_group(batch)
        features = torch.cat(
            [
                grouped_features[name]
                for name in self.embedding_group.group_names()
            ],
            dim=-1,
        )
        return self._output_to_prediction(self.output(self.mlp(features)))
```

框架会将 `custom_model.config` 解包成强类型 message，并通过 `custom_config` 参数
传入模型。`BaseModel` 同时会将 `self._model_config` 设置为该 message，因此继承
现有基础模型时可以继续使用相同的配置访问方式。

自定义模型仍需按所继承基础模型的要求实现或复用 `predict`、`init_loss`、
`loss`、`init_metric` 和 `update_metric` 等接口。

## 配置 pipeline

`custom_model.config` 使用 `google.protobuf.Any` 保存用户定义的强类型配置：

```protobuf
model_config {
    feature_groups {
        group_name: "group1"
        feature_names: "f1"
        feature_names: "f2"
        group_type: DEEP
    }

    custom_model {
        class_path: "tzrec_custom.models.custom_rank_model.CustomRankModel"
        config {
            [type.googleapis.com/tzrec_custom.protos.CustomRankModelConfig] {
                mlp {
                    hidden_units: 128
                    hidden_units: 64
                    dropout_ratio: 0.1
                }
            }
        }
    }

    metrics {
        auc {}
    }
    losses {
        binary_cross_entropy {}
    }
}
```

方括号中的名称是 proto 文件声明的 package 与 message 名，不是 Python 文件
路径。模型模块会在 pipeline config 解析前自动导入，因此对应 descriptor 已经
注册。

## 运行

默认使用 `tzrec_custom` 包时：

```bash
PYTHONPATH=. torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    tzrec/train_eval.py \
    --pipeline_config_path custom_rank_model.config \
    --train_input_path ${TRAIN_INPUT_PATH} \
    --eval_input_path ${EVAL_INPUT_PATH} \
    --model_dir ${MODEL_DIR}
```

使用其他自定义包时，训练、评估、预测和导出命令都需要设置同一个
`TZREC_CUSTOM_PACKAGE`。`torchrun` 启动的各个 worker 会继承该环境变量。

### 打包发布

参考[开发指南](../develop.md)。自定义包需要包含模型代码和生成的 protobuf
binding，并保证运行环境可以导入该包。
