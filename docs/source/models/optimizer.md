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
  - optimizer: 优化器类型，具体见sparse optimize的[配置文档](../reference.html)
  - learning_rate: sparse_optimizer的学习率计划器,具体见sparse_optimizer中的learning_rate的[配置文档](../reference.html)
- dense_optimizer
  - optimizer: 优化器类型，具体见dense optimize的[配置文档](../reference.html)

  - learning_rate: dense_optimizer的学习率计划器,具体见dense_optimizer中的learning_rate的[配置文档](../reference.html)

  - part_optimizers:

    在train_config.dense_optimizer中可以通过part_optimizers针对部分稠密参数配置单独的优化器

    - optimizer: 和dense_optimizer可配置项一样，具体见optimize的[配置文档](../reference.html)
    - regex_pattern: 必须配置，可优化的模型参数名称正则表达式。对于某参数名称可以被多个参数优化器正则项可以匹配，则会匹配到第一个参数优化器。不能匹配上则使用dense_optimizer。
    - learning_rate: 学习率计划器，和dense_optimizer的学习率可配置项一样，如果不配则使用dense_optimizer的学习率计划器，具体见learning_rate的[配置文档](../reference.html)。
