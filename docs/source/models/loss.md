# 损失函数

## 简介

不同类型的任务有不同的loss，也可以对于同一个任务配置多个损失函数。 目前TorchEasyRec支持binary_cross_entropy，softmax_cross_entropy，l2_loss，以及jrc_loss

## binary_cross_entropy

二分类损失函数，其对应的任务num_class是1或者2

配置如下

```
model_config {
    losses {
        binary_cross_entropy {
        }
    }
}
```

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

对于该损失函数，要求同一个session_id的样本尽量在一个batch中进行训练，在一个session中尽量要求样本保持有序。

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

## pe_mtl_loss

该方法适用于多任务场景，主要作用于损失权重的学习。需要结合别的损失函数一起使用，根据每次batch中的数据的每个分量损失函数，给与每个分量损失不同的权重。https://dl.acm.org/doi/10.1145/3298689.3346998

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
            ${loss_name} {}
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
