# AutoDis Embedding

相比于类别型特征的嵌入，连续型特征由于其无限的取值个数，缺少一种有效的embedding方法。传统的分箱离散化方式鲁棒性较低，可能出现近似特征分到不同箱子，或者差异明显的特征分到同一个箱子。AutoDis Embedding提出了一种端到端的学习方式，自动学习连续型特征的离散embedding表达。

## 配置方法

AutoDis Embedding针对raw_feature特征设计，在特征配置中，添加autodis属性，例如：

```
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        autodis {
           embedding_dim: 12
           num_channels: 3
           temperature: 0.1
           keep_prob: 0.8
        }
    }
}
```

embedding_dim为embedding维度， num_channels为embedding的通道数， temperature为softmax温度系数， keep_prob为隐含层保留概率。
其中embedding_dim和num_channels为必选参数， temperature和keep_prob为可选参数。如果对多个raw_feature配置autodis embedding，建议使用相同的autodis参数, 以保证训练推理速度。多组不同的autodis参数配置，可能会导致性能下降。

## 参考文章

An Embedding Learning Framework for Numerical Features in CTR Prediction
https://arxiv.org/pdf/2012.08986
