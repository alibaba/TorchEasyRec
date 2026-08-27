# 自定义模型

TorchEasyRec 支持在**你自己的工程**里编写自定义模型，无需修改 TorchEasyRec 仓库中的任何文件。
模型代码、模型配置 proto 和 pipeline config 都放在你的工程中，升级 TorchEasyRec 时不会产生代码冲突。

pipeline config 通过 `custom_model.class_path` 指定模型类的完整 python 路径，通过
`custom_model.config`（`google.protobuf.Any`）携带你自己定义的强类型模型配置。
TorchEasyRec 在解析 pipeline config 前会先导入 `class_path` 所在的模块，因此模型模块中
`import` 的 `*_pb2` 会自动完成 protobuf descriptor 注册。

## 工程结构

推荐的工程结构如下，包名和目录名可以自由选择：

```text
my-rec-project/
├── my_models/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── custom_rank_model.py
│   └── protos/
│       ├── __init__.py
│       └── custom_rank_model.proto
├── configs/
│   └── custom_rank_model.config
└── .vscode/
    └── settings.json
```

TorchEasyRec 可以通过 pip 安装，也可以作为 git submodule 引入，两种方式后续步骤完全一致。

### 方式一：pip 安装 TorchEasyRec

```bash
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-accelerate.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-accelerate.aliyuncs.com
```

运行命令时用 `PYTHONPATH=.` 让 python 可以导入你的包即可：

```bash
PYTHONPATH=. python -m tzrec.train_eval --pipeline_config_path configs/custom_rank_model.config ...
```

### 方式二：TorchEasyRec 作为 git submodule

适合需要固定 TorchEasyRec 版本、或需要同时修改调试 TorchEasyRec 源码的场景。

```bash
git submodule add https://github.com/alibaba/TorchEasyRec.git third_party/TorchEasyRec
pip install -r third_party/TorchEasyRec/requirements.txt
```

submodule 中不包含生成的 `*_pb2.py`，拉取和更新 submodule 后需要在 submodule 目录内生成：

```bash
cd third_party/TorchEasyRec && bash scripts/gen_proto.sh && cd -
```

运行命令时把 submodule 加入 `PYTHONPATH`：

```bash
PYTHONPATH=.:third_party/TorchEasyRec python -m tzrec.train_eval \
    --pipeline_config_path configs/custom_rank_model.config ...
```

注意：`PYTHONPATH` 中的源码版本会覆盖 pip 安装的 tzrec，请不要同时使用两种方式。

## 编写模型配置 proto

TorchEasyRec 使用 [Protocol Buffer](https://developers.google.com/protocol-buffers/docs/pythontutorial)
定义配置文件格式。在你自己的包里定义模型配置，可以直接复用 TorchEasyRec 的公共 message，
例如 `tzrec.protos.MLP`：

```protobuf
// my_models/protos/custom_rank_model.proto
syntax = "proto2";
package my_models.protos;

import "tzrec/protos/module.proto";

message CustomRankModelConfig {
    required tzrec.protos.MLP mlp = 1;
}
```

## 生成 protobuf binding

在工程根目录用 protoc 编译自己的 proto：

```bash
python -m grpc_tools.protoc -I . my_models/protos/*.proto --python_out=. --pyi_out=.
```

如果自定义 proto `import` 了 TorchEasyRec 的 proto（例如上面复用 `tzrec.protos.MLP`），
还需要把 TorchEasyRec 的目录加到 `-I` 中。protoc 读取的是 `.proto` 源文件，
所以这里要填 `tzrec/protos/` 的上一级目录：

```bash
# submodule 方式
TZREC_PATH=third_party/TorchEasyRec
# pip 安装方式
TZREC_PATH=$(python -c "import importlib.metadata as m; print(m.distribution('tzrec').locate_file(''))")

python -m grpc_tools.protoc -I . -I "${TZREC_PATH}" \
    my_models/protos/*.proto --python_out=. --pyi_out=.
```

## 编写模型

继承 `tzrec.models.model.BaseModel` 来实现自定义模型，需重载以下函数。
自定义模型的 `__init__` 签名与内置模型完全一致，框架会把 `custom_model.config` 解包成
强类型 message 后赋值给 `self._model_config`。

### 初始化: \_\_init\_\_

- 根据模型配置`model_config`和特征配置`features`构建子模块

### 前向: predict

- 根据输入的`batch`数据，进行前向推理，得到`predictions`
  - `batch`为`tzrec.datasets.utils.Batch`的数据结构，包含`dense_features`（稠密特征）、`sparse_features`（稀疏特征）、`sequence_dense_features` (序列稠密特征)
  - 一般可以将`batch` 传给`EmbeddingGroup`模块`tzrec.modules.embedding.EmbeddingGroup`得到分组的Embedding结果后，再进行进一步前向推理

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

**模型模块必须在顶层 `import` 自己的 `*_pb2`。**

以排序模型为例

```python
# my_models/models/custom_rank_model.py
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs

from my_models.protos.custom_rank_model_pb2 import CustomRankModelConfig  # NOQA


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
        # self._model_config 即解包后的 CustomRankModelConfig
        self.init_input()
        self.mlp = MLP(
            self.embedding_group.group_total_dim("deep"),
            **config_to_kwargs(self._model_config.mlp),
        )
        self.output_mlp = nn.Linear(self.mlp.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        y = self.output_mlp(self.mlp(grouped_features["deep"]))
        return self._output_to_prediction(y)
```

## 配置 pipeline

```protobuf
model_config {
    feature_groups {
        group_name: "deep"
        feature_names: "f1"
        feature_names: "f2"
        group_type: DEEP
    }

    custom_model {
        class_path: "my_models.models.custom_rank_model.CustomRankModel"
        config {
            [type.googleapis.com/my_models.protos.CustomRankModelConfig] {
                mlp {
                    hidden_units: [128, 64]
                    dropout_ratio: [0.1, 0.1]
                }
            }
        }
    }

    losses {
        binary_cross_entropy {}
    }
    metrics {
        auc {}
    }
}
```

- `class_path` 是模型类的完整 python 路径，运行时该路径必须可以被 python 导入。
- 方括号中的 `my_models.protos.CustomRankModelConfig` 是 proto 文件里声明的 `package` 加
  message 名，**不是** python 模块路径。
- `custom_model.config` 可以省略，此时 `self._model_config` 为 `None`。

## 运行

以 pip 安装方式为例，submodule 方式只需把 `PYTHONPATH` 换成 `.:third_party/TorchEasyRec`。

训练

```bash
PYTHONPATH=. torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path configs/custom_rank_model.config \
    --train_input_path ${TRAIN_INPUT_PATH} \
    --eval_input_path ${EVAL_INPUT_PATH} \
    --model_dir ${MODEL_DIR}
```

评估、导出和预测与内置模型完全一致，`class_path` 会随 pipeline config 一起保存到
`${MODEL_DIR}/pipeline.config`，因此后续命令无需任何额外配置：

```bash
PYTHONPATH=. torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.eval --pipeline_config_path ${MODEL_DIR}/pipeline.config

PYTHONPATH=. torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.export --pipeline_config_path ${MODEL_DIR}/pipeline.config \
    --export_dir ${EXPORT_DIR}
```

## 开发与调试

### 编辑器跳转

shell 中设置的 `PYTHONPATH` 不会传递给 VSCode 的 Pylance，需要在工程里增加
`.vscode/settings.json`，否则无法跳转到 `tzrec` 的定义：

```json
{
    "python.analysis.extraPaths": [".", "third_party/TorchEasyRec"],
    "python.autoComplete.extraPaths": [".", "third_party/TorchEasyRec"]
}
```

pip 安装方式下 `extraPaths` 只需要 `["."]`，`tzrec` 由解释器的 site-packages 解析；
submodule 方式下跳转会进入 submodule 源码，可以直接修改和打断点调试 TorchEasyRec。
protoc 的 `--pyi_out` 生成的 `*_pb2.pyi` 让自定义 proto 的类型也可以正常跳转和补全。

### 单进程调试

调试模型代码时使用单进程启动，可以正常使用 `pdb` / `debugpy`：

```bash
PYTHONPATH=. torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.train_eval --pipeline_config_path configs/custom_rank_model.config ...
```

### 常见问题

- `config type [xxx] is not registered`：`class_path` 指向的模型模块没有在顶层
  `import` 对应的 `*_pb2`，descriptor 未注册。
- `ModuleNotFoundError` / `class xxx is not found in module xxx`：`class_path` 写错，
  或者运行时 `PYTHONPATH` 中没有包含你的工程目录。
- `duplicate file name tzrec/protos/xxx.proto` 或 `couldn't resolve name`：自定义 proto
  编译时 `-I` 指向的 TorchEasyRec 和运行时使用的不一致，用实际运行的那一份重新编译。
- `confilict class xxx is already register`：自定义模型的类名和 TorchEasyRec 内置模型
  重名，换一个类名即可。

## 打包发布

自定义包可以打成 wheel 发布，安装后运行时不再需要设置 `PYTHONPATH`。

`setup.py`：

```python
from setuptools import find_packages, setup

setup(
    name="my_models",
    version="0.1.0",
    packages=find_packages(),
    package_data={"my_models.protos": ["*.proto", "*.pyi"]},
)
```

先生成 binding 再打包，顺序和 TorchEasyRec 的 `scripts/build_wheel.sh` 一致：

```bash
python -m grpc_tools.protoc -I . -I "${TZREC_PATH}" \
    my_models/protos/*.proto --python_out=. --pyi_out=.
python setup.py bdist_wheel
pip install dist/my_models-0.1.0-py3-none-any.whl
```

`*_pb2.py` 是普通的 python 文件，`find_packages()` 会自动打包进 wheel；`*_pb2.pyi`
和 `.proto` 需要用 `package_data` 指定。如果 `.gitignore` 中忽略了 `*_pb2.py`，
打包前必须先执行上面的 protoc 命令。
