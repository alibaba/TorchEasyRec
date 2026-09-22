# 损失函数

## 简介

不同类型的任务有不同的loss，也可以对于同一个任务配置多个损失函数。 目前TorchEasyRec支持binary_cross_entropy，binary_focal_loss，softmax_cross_entropy，l2_loss，jrc_loss，以及listwise_rank_loss

## 损失权重 weight

`losses` 是 `LossConfig` 列表，每一项都支持 `weight` 字段，用于调节该损失项在总损失中的权重，默认值1.0。当一个任务配置了多个损失函数时，用 `weight` 平衡它们之间的相对大小

配置如下

```
model_config {
    losses {
        binary_cross_entropy {}
    }
    losses {
        listwise_rank_loss {}
        weight: 0.1
    }
}
```

参数说明：

1. weight: 该损失项的权重，默认值1.0。它与任务级的weight（`task_towers.weight` / `task_configs.weight`）相乘生效。任务级weight对该任务下的所有损失等比缩放，只有losses.weight才能调节同一个任务内各损失项的相对权重
1. 训练日志和loss metric记录的都是加权后的值，因此各损失项之和即为total_loss
1. 开启`use_pareto_loss_weight`时，pe_mtl_loss会在各损失项之上再动态求解一组权重，losses.weight会改变梯度尺度进而影响求解结果，建议不要同时调节二者

## binary_cross_entropy

二分类损失函数，其对应的任务num_class是1或者2

配置如下

```
model_config {
    losses {
        binary_cross_entropy {
            label_smoothing: 0.1
        }
    }
}
```

参数说明：

1. label_smoothing: 标签平滑系数，默认值0.0（不使用标签平滑）。设置后，标签会被平滑为 `label * (1 - label_smoothing) + 0.5 * label_smoothing`，即将硬标签0/1分别平滑为 `label_smoothing/2` 和 `1 - label_smoothing/2`。建议值：0.05 ~ 0.2。

## binary_focal_loss

二分类focal loss， 对应的任务num_class是1。

配置如下

```
model_config {
    losses {
        binary_focal_loss {
            gamma: 2.0
            alpha: 0.5
        }
    }
}
```

参数说明：

1. gamma: focal loss的指数，默认值2.0
1. alpha: 调节样本权重的类别平衡参数，建议根据正负样本比例来配置alpha，即 alpha / (1-alpha) = #Neg / #Pos, 默认值0.5

## softmax_cross_entropy

多分类损失函数，其对应的任务num_class大于1

配置如下

```
model_config {
    losses {
        softmax_cross_entropy {
        }
    }
}
```

## l2_loss

适用回归任务的损失函数，配置如下

```
model_config {
    losses {
        l2_loss {
        }
    }
}
```

## jrc_loss

适用二分类任务的损失函数，其对应的任务num_class必须是2。该损失函数除了关注样本目标自身分类的正确性，还会关注在同一个batch的同一个session中，所有正样本的概率要尽可能的大于所有负样本的概率。
https://arxiv.org/abs/2208.06164

session的来源与[listwise_rank_loss](#listwise_rank_loss)相同：设置`session_name`时，同一个batch内该字段取值相同的样本构成一个session（不要求相邻）；不设置时，DlrmHSTU系列模型以每个请求的候选序列为一个session，其他排序模型必须设置`session_name`

配置如下

```
model_config {
    losses {
        jrc_loss {
            session_name: session_id
        }
    }
}
```

对于该损失函数，要求同一个session_id的样本落在同一个batch中。

我们使用sql如下方式构造样本,该数据集的session_name是user_id

```sql
DROP TABLE IF EXISTS taobao_multitask_sample_bucketized_train_jrc;
create table  taobao_multitask_sample_bucketized_train_jrc as
select `(ds)?+.+`
from taobao_multitask_sample_bucketized
DISTRIBUTE BY user_id
SORT BY user_id asc,time_stamp asc
;
```

## listwise_rank_loss

listwise排序损失（InfoNCE），同一个list内的其他样本互为负样本。list的来源有两种：

- DlrmHSTU系列模型：模型会发布每个请求的候选数，一个请求的候选序列即为一个list，无需额外配置（设置`session_name`会报错），详见[dlrm_hstu_onerank](dlrm_hstu_onerank.md)
- 其他排序模型（多塔、DBMTL等）：必须通过`session_name`指定分组字段（如request_id或user_id），同一个batch内该字段取值相同的样本构成一个list。该字段需为模型的稀疏特征

配置如下

```
model_config {
    losses {
        binary_cross_entropy {}
    }
    losses {
        listwise_rank_loss {
            session_name: "user_id"
            temperature_init: 0.07
            learnable_temperature: true
        }
        weight: 0.1
    }
}
```

参数说明：

1. session_name: list分组的字段名，仅用于非DlrmHSTU系列模型（DlrmHSTU系列模型按请求分组，不可设置）
1. temperature_init: softmax温度初始值，logits会乘以 1 / temperature，默认值0.07
1. learnable_temperature: 温度是否可学习（训练中会被clamp防止溢出），默认值true

注意事项：

1. 一个list需要同时含有至少一个正样本和一个负样本才会计入该项损失，被屏蔽的list仍计入分母。使用`session_name`时要求同一个session的样本落在同一个batch内（不要求相邻），对于非DlrmHSTU系列模型样本构造方式与[jrc_loss](#jrc_loss)相同（`DISTRIBUTE BY session_id SORT BY session_id`），并尽量加大`batch_size`；若样本未按session分组，绝大多数list只含一条样本而被屏蔽，表现为该项损失接近0
1. 该损失只约束list内的相对打分，不保证概率校准，通常与逐点损失（如binary_cross_entropy）搭配使用，用同级的`weight`调节相对权重，经验值0.1量级

## pe_mtl_loss

Pareto-Efficient Algorithm for Multiple Objective Optimization，该方法适用于多任务场景，自动根据帕累托最优最优的原则调整不同目标的损失权重。https://dl.acm.org/doi/10.1145/3298689.3346998

配置如下

```
model_config {
    feature_groups {
        group_name: "all"
        feature_names: "user_id"
        ...
        group_type: DEEP
    }
    ${model_name} {
        task_towers {
            ...
            losses {
                ${loss_name} {}
            }
            pareto_min_loss_weight: 0.4
        }
        task_towers {
            ...
            losses {
                ${loss_name} {}
            }
            pareto_min_loss_weight: 0.4
        }
    }
    use_pareto_loss_weight: true
}
```

- use_pareto_loss_weight: 是否使用pe_mtl_loss动态loss权重
- pareto_min_loss_weight: 每个任务对应的最小损失权重，默认值为0.0，当use_pareto_loss_weight是true的时候，pareto_min_loss_weight才生效，所有tower的损失函数最小权重之和必须小于等于1.0
