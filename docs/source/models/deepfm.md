# DeepFM

## 简介

DeepFM是在WideAndDeep基础上加入了FM模块的改进模型。FM模块和DNN模块共享相同的特征，即相同的Embedding。

![deepfm.png](../../images/models/deepfm.png)

## 配置说明

```
model_config {
    feature_groups {
        group_name: "wide"
        feature_names: "int_0"
        feature_names: "int_1"
        ...
        feature_names: "cat_24"
        feature_names: "cat_25"
        group_type: WIDE
    }
    feature_groups {
        group_name: "fm"
        feature_names: "int_0"
        feature_names: "int_1"
        ...
        feature_names: "cat_24"
        feature_names: "cat_25"
        group_type: DEEP
    }
    feature_groups {
        group_name: "deep"
        feature_names: "int_0"
        feature_names: "int_1"
        ...
        feature_names: "cat_24"
        feature_names: "cat_25"
        group_type: DEEP
    }
    deepfm {
        deep {
            hidden_units: [512, 256, 128]
        }
        final {
            hidden_units: [64]
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

- feature_groups: 需要至少两个feature_group: wide和deep, fm可选
- deepfm:  deepfm相关的参数
  - deep: deep mlp的参数配置
    - hidden_units: mlp每一层的channel数目，即神经元的数目
  - wide_embedding_dim: wide部分输出的大小
  - final: 整合wide, fm, deep的输出, 可以选择是否使用
    - hidden_units: mlp每一层的channel数目，即神经元的数目
- losses: 损失函数配置
- metrics: 评估指标配置

## 模型输出

模型的输出名为: "logits" / "probs" / "y", 对应sigmoid之前的值/概率/回归模型的预测值

## 示例Config

[deepfm_criteo.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/deepfm_criteo.config)

## 参考论文

[DeepFM](https://arxiv.org/abs/1703.04247)
