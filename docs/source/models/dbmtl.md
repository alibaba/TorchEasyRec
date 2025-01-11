# DBMTL

## 简介

DBMTL构建了多个目标之间的贝叶斯网络，显式建模了多个目标之间可能存在的因果关系，通过对不同任务间的贝叶斯关系来同时优化场景中的多个指标。
![dbmtl.png](../../images/models/dbmtl.png)

底层的shared layer和specific layer是通过hard parameter sharing方式来人工配置的，而google的MMoE是基于soft parameter sharing来实现不同任务底层特征和网络共享，并在Youtube场景中取得了不错的效果。因此DBMTL同样支持将shared layer和specific layer模块替换成MMoE模块，即通过task gate的方式在多组expert参数中加权组合出对应task的feature。

![dbmtl_mmoe.png](../../images/models/dbmtl_mmoe.png)

## 配置说明

### DBMTL

```
model_config {
    feature_groups {
        group_name: "all"
        feature_names: "user_id"
        feature_names: "cms_segid"
        ...
        feature_names: "price"
        group_type: DEEP
    }
    dbmtl {
        bottom_mlp {
            hidden_units: [1024, 512, 256]
        }
        task_towers {
            tower_name: "ctr"
            label_name: "clk"
            mlp {
                hidden_units: [256, 128, 64]
            }
            metrics {
                auc {}
            }
            losses {
                binary_cross_entropy {}
            }
        }
        task_towers {
            tower_name: "cvr"
            label_name: "buy"
            mlp {
                hidden_units: [256, 128, 64]
            }
            metrics {
                auc {
                    thresholds: 1000
                }
            }
            losses {
                binary_cross_entropy {}
            }
            relation_tower_names: "ctr"
            relation_mlp {
                hidden_units: [256, 128, 64]
            }
        }
    }
}
```

- feature_groups: 配置一个名为'all'的feature_group。
- dbmtl: dbmtl相关的参数
  - bottom_mlp: 底层MLP参数
    - hidden_units: mlp每一层的channel数目，即神经元的数目
  - task_towers 根据任务数配置task_towers
    - tower_name: TaskTower名
    - label_name: tower对应的label名
    - mlp: TaskTower的MLP参数配置
    - relation_tower_names: 上游关联Tower名
    - relation_mlp: 关联Tower融合的MLP参数配置
    - losses: 任务损失函数配置
    - metrics: 任务评估指标配置
    - task_space_indicator_label: 标识当前任务空间的目标名称，配合in_task_space_weight、out_task_space_weight使用。例如，对于cvr任务，可以设置task_space_indicator_label=clk，in_task_space_weight=1，out_task_space_weight=0，来使得cvr任务塔只在点击空间计算loss。
    - in_task_space_weight: 对于task_space_indicator_label>0的样本会乘以该权重
    - out_task_space_weight: 对于task_space_indicator_label\<=0的样本会乘以该权重

### DBMTL+MMOE

```
model_config {
    feature_groups {
        group_name: "all"
        feature_names: "user_id"
        feature_names: "cms_segid"
        ...
        feature_names: "price"
        group_type: DEEP
    }
    dbmtl {
        bottom_mlp {
            hidden_units: [1024, 512, 256]
        }
        expert_mlp {
            hidden_units: [512, 256, 128]
        }
        num_expert: 3
        task_towers {
            tower_name: "ctr"
            label_name: "clk"
            mlp {
                hidden_units: [256, 128, 64]
            }
            metrics {
                auc {}
            }
            losses {
                binary_cross_entropy {}
            }
        }
        task_towers {
            tower_name: "cvr"
            label_name: "buy"
            mlp {
                hidden_units: [256, 128, 64]
            }
            metrics {
                auc {
                    thresholds: 1000
                }
            }
            losses {
                binary_cross_entropy {}
            }
            relation_tower_names: "ctr"
            relation_mlp {
                hidden_units: [256, 128, 64]
            }
        }
    }
}
```

- dbmtl: dbmtl相关的参数
  - expert_mlp: MMOE的专家MLP配置
    - hidden_units: mlp每一层的channel数目，即神经元的数目
  - expert_num: 专家DNN的数目
  - 其余与dbmtl一致

## 模型输出

DBMTL模型每个塔的输出名为："logits\_" / "probs\_" / "y\_" + tower_name
其中，logits/probs/y对应: sigmoid之前的值/概率/回归模型的预测值
DBMTL模型每个塔的指标为：指标名+ "\_" + tower_name

## 示例配置

[dbmtl_taobao.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/dbmtl_taobao.config)

## 参考论文

[DBMTL](https://arxiv.org/pdf/1902.09154)
