# 优化器

## 简介

TorchEasyRec除了像tzrec一样拥有sparse_optimizer和dense_optimizer,在dense_optimize中实现了part_optimizers 即：参数优化器。对于模型参数优先part_optimizer的正则项和参数名称匹配，如果能匹配上则该参数使用对应的part_optimizer，如果最后无法匹配上，则使用全局的dense_optimizer。

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

如上，在train_config.dense_optimizer中的part_optimizers是参数优化器

- optimizer: 和dense_optimizer可配置项一样，具体见proto的定义
- regex_pattern: 必须配置，可优化的模型参数名称正则表达式。对于某参数名称可以被多个参数优化器正则项可以匹配，则会匹配到第一个参数优化器。
- learning_rate: 学习率计划器，和dense_optimizer的学习率可配置项一样，如果不配则使用dense_optimizer的学习率计划器，具体见proto的定义。
