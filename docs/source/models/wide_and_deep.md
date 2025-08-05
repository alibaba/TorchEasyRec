# WideAndDeep

### 简介

WideAndDeep包含Wide和Deep两部分，Wide部分负责记忆，Deep部分负责泛化。Wide部分可以做显式的特征交叉，Deep部分可以实现隐式自动的特征交叉。

![wide_and_deep.png](../../images/models/wide_and_deep.png)

### 配置说明

```
model_config:{
    feature_groups: {
        group_name: "deep"
        feature_names: "hour"
        feature_names: "c1"
        ...
        feature_names: "site_id_app_id"
        wide_deep:DEEP
    }
    feature_groups: {
        group_name: "wide"
        feature_names: "hour"
        feature_names: "c1"
        ...
        feature_names: "c21"
        wide_deep: WIDE
    }
    wide_and_deep {
        deep {
            hidden_units: [128, 64, 32]
        }
    }
}
```

- feature_groups: 需要两个feature_group: wide group和deep group, **group name不能变**
- wide_and_deep: wide_and_deep 相关的参数
  - deep: deep part的参数配置
    - hidden_units: dnn每一层的channel数目，即神经元的数目
  - wide_embedding_dim: wide部分输出的大小
  - wide_init_fn: wide特征嵌入初始化方式，默认不需要设置，如需自定义，可以设置任意的torch内置初始化函数，如`nn.init.zeros_`
  - final: 整合wide, fm, deep的输出, 可以选择是否使用
    - hidden_units: mlp每一层的channel数目，即神经元的数目
- losses: 损失函数配置
- metrics: 评估指标配置

## 模型输出

模型的输出名为: "logits" / "probs" / "y", 对应sigmoid之前的值/概率/回归模型的预测值

## 示例Config

[wide_and_deep_criteo.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/wide_and_deep_criteo.config)

### 参考论文

[WideAndDeep](https://arxiv.org/abs/1606.07792)
