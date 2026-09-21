# 优化器

## 简介

TorchEasyRec的优化器分为sparse_optimizer和dense_optimizer两个部分，sparse_optimizer负责embedding部分稀疏参数，dense_optimizer负责nn部分稠密参数。在dense_optimizer中，TorchEasyRec还支持part_optimizers，可以通过正则表示支持对部分参数设置单独的优化器。

## 样例配置

```
train_config {
    sparse_optimizer {
        adagrad_optimizer {
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
        ema {
            decay: 0.999
        }
        part_optimizers {
            adamw_optimizer {
                lr: 0.01
            }
            regex_pattern: "(.*)booster_mlp(.*)"
        }
        part_optimizers {
            sgd_optimizer {
                lr: 0.002
            }
            regex_pattern: "(.*)light_mlp(.*)"
            exponential_decay_learning_rate {
                decay_size: [1000, 10000]
                learning_rates:[0.001, 0.0001]
            }
        }

    }
}
```

- sparse_optimizer

  - optimizer: 优化器类型，具体见sparse optimize的[配置文档](../reference.md)
  - learning_rate: sparse_optimizer的学习率计划器,具体见sparse_optimizer中的learning_rate的[配置文档](../reference.md)

  **Note**: 被分片为`data_parallel`的Embedding表不受sparse_optimizer管理，实际由dense_optimizer更新，并跟随dense_optimizer的LR策略，详见[训练文档](../usage/train.md)的Embedding分片约束章节

  **Note**: `adagrad_optimizer`和`rowwise_adagrad_optimizer`的`initial_accumulator_value`对齐TensorFlow Adagrad的同名参数（TF默认0.1，TorchEasyRec默认0.0），对普通Embedding表和[dynamicemb](../feature/dynamicemb.md)表同时生效：普通Embedding表在建表时把整个accumulator初始化为该值，dynamicemb表则在key首次写入时把该key的accumulator初始化为该值。`ftrl_optimizer`也用该字段初始化它的accumulator

  **Note**: `ftrl_optimizer`（FTRL-Proximal，McMahan et al. 2013）**只支持[dynamicemb](../feature/dynamicemb.md)表**，FBGEMM没有FTRL的embedding kernel，模型中只要还有一张非dynamicemb的sparse表，训练会在plan阶段直接报错并给出表名。可配置`dynamicemb`的特征类型见[dynamicemb文档](../feature/dynamicemb.md)，配置了`boundaries`的`raw_feature`等不支持dynamicemb的特征，无法与`ftrl_optimizer`一起使用。被分片为`data_parallel`的表不受此限制（由dense_optimizer更新）

- dense_optimizer

  - optimizer: 优化器类型，具体见dense optimize的[配置文档](../reference.md)

  - learning_rate: dense_optimizer的学习率计划器,具体见dense_optimizer中的learning_rate的[配置文档](../reference.md)

  - ema:

    对全部稠密参数（包括`part_optimizers`管理的参数）计算指数移动平均。配置该字段后启用，`decay`取值范围为`[0, 1]`，默认为`0.999`。EMA在每次实际参数更新后执行。

  - part_optimizers:

    在train_config.dense_optimizer中可以通过part_optimizers针对部分稠密参数配置单独的优化器

    - optimizer: 和dense_optimizer可配置项一样，具体见optimize的[配置文档](../reference.md)
    - regex_pattern: 必须配置，可优化的模型参数名称正则表达式。对于某参数名称可以被多个参数优化器正则项可以匹配，则会匹配到第一个参数优化器。不能匹配上则使用dense_optimizer。
    - learning_rate: 学习率计划器，和dense_optimizer的学习率可配置项一样，如果不配则使用dense_optimizer的学习率计划器，具体见learning_rate的[配置文档](../reference.md)。
