# 自定义模型

## 编写模型proto文件

TorchEasyRec使用 [Protocol Buffer](https://developers.google.com/protocol-buffers/docs/pythontutorial) 定义配置文件格式。

在 `tzrec/protos/models/rank_model.proto` 中增加一个 `CustomRankModel` Message来定义模型配置

```protobuf
message CustomRankModel {
  required MLP mlp = 1;
  ...
};
```

在 `tzrec/protos/model.proto的在` 的 `oneof model`里面增加 `CustomRankModel`

```protobuf
message ModelConfig {
   ...

   oneof model {
      ...
      CustomRankModel custom_rank_model = 1001;
      ...
   }
   ...
}
```

生成proto python `*_pb2.py` 文件

```bash
bash scripts/gen_proto.sh
```

## 编写模型文件

### 继承

继承 `tzrec.models.model.BaseModel` 来实现自定义模型，需重载以下函数

### 初始化: \_\_init\_\_

- 根据模型配置`model_config`和特征配置`features`构建子模块

### 前向: predict

- 根据输入的`batch`数据，进行前向推理，得到`predictions`
  - `batch`为`tzrec.datasets.utils.Batch`的数据结构，包含`dense_features`（稠密特征）、`sparse_features`（稀疏特征）、`sequence_dense_features` (序列稠密特征)
  - 一般可以将`dense_features`、`sparse_features`、`sequence_dense_features` 传给`EmbeddingGroup`模块`tzrec.modules.embedding.EmbeddingGroup`得到分组的Embedding结果后，再进行进一步前向推理

### 损失: init_loss & loss

- `init_loss`函数用于根据模型损失函数配置初始化loss模块，写入到`self._loss_modules`中
- `loss`函数用于根据输入的`predictions`和`batch`中的label，实际计算每个step的loss，返回一个`loss_dict`

### 评估: init_metric & update_metric

- `init_metric`函数用于根据模型初始化metric模块，写入到`self._metric_modules`中
- `update_metric`函数用于根据输入的`predictions`和`batch`中的label，更新metric模块的状态

### 常用继承

在排序、多目标排序、召回的场景下，可以直接继承以下子模型，可以只用重置前向推理函数

- 排序模型可直接继承 `tzrec.models.rank_model.RankModel`
- 多目标模型可直接继承 `tzrec.models.multi_task_rank.MultiTaskRank`
- 召回模型可直接继承 `tzrec.models.match_model.MatchModel`

以排序模型为例

```python
# tzrec/model/custom_rank_model.py
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


class CustomRankModel(RankModel):
    """CustomRankModel.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        # 构建EmbeddingGroup
        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )
        # 构建MLP
        total_in_dim = sum(self.embedding_group.group_total_dim(n) for n in self.embedding_group.group_names())
        self.mlp = MLP(
            in_features=total_in_dim,
            **config_to_kwargs(self._model_config.mlp),
        )
        final_dim = self.mlp.output_dim()
        self.output_mlp = nn.Linear(final_dim, self._num_class)
        # 初始化其他模块
        ...


    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.embedding_group(
            batch
        )
        features = torch.cat([grouped_features["group1"], grouped_features["group2"]], dim=-1)
        tower_output = self.mlp(features)
        y = self.output_mlp(tower_output)
        # 其他前向推理
        ...

        return self._output_to_prediction(y)
```

## 测试

编写 custom_rank_model.config

```

# 数据相关参数配置
data_config {
  ...
}

# 特征相关参数配置
feature_configs : {
  ...
}
feature_configs : {
  ...
}

# 训练相关的参数配置
train_config {
  ...
}

# 评估相关参数配置
eval_config {
  ...
}

# 模型相关参数配置
model_config: {
    feature_groups: {
        group_name: 'group1'
        group_name: 'all'
        feature_names: 'f1'
        feature_names: 'f2'
        ...
        wide_deep: DEEP
    }
    feature_groups: {
        group_name: 'group2'
        feature_names: 'f3'
        feature_names: 'f4'
        ...
        wide_deep: DEEP
    }
    ...
    custom_rank_model {
        mlp {
            hidden_units: [64]
        }
        ...
    }
    metrics {
        auc {}
    }
    losses {
        binary_cross_entropy {}
    }
}
```

运行

```bash
PYTHONPATH=. torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    tzrec/train_eval.py \
    --pipeline_config_path custom_rank_model.config \
    --train_input_path ${TRAIN_INPUT_PATH} \
    --eval_input_path ${EVAL_INPUT_PATH} \
    --model_dir ${MODEL_DIR}
```

### 打包发布

参考[开发指南](../develop.md)
