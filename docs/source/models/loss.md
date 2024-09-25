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

## softmax_cross_entropy

多分类损失函数，其对应的任务num_class大于1

配置如下

```
model_config {
    losses {
        SoftmaxCrossEntropy {
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
