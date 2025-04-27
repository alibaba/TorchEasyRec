# PLE

## 简介

多任务学习可能带来负迁移（negative transfer）现象，即相关性不强的任务之间的信息共享，会影响网络的表现。此前已经有部分研究来减轻负迁移现象，如谷歌提出的MMoE模型。

但通过实验发现，多任务学习中往往还存在跷跷板现象(seesaw phenomenon)，即多任务学习相对于多个单任务学习的模型，往往能够提升一部分任务的效果，同时牺牲另外部分任务的效果。即使通过MMoE这种方式减轻负迁移现象，跷跷板现象仍然是广泛存在的。

论文提出了Progressive Layered Extraction (简称PLE)，来解决多任务学习的跷跷板现象。

![ple.png](../../images/models/ple.png)

## 配置说明

```
model_config {
    feature_groups {
        group_name: "all"
        feature_names: "user_id"
        feature_names: "cms_segid"
        ...
        group_type: DEEP
    }
    ple {
        extraction_networks {
            network_name: "layer1"
            expert_num_per_task: 2
            share_num: 2
            task_expert_net {
                hidden_units: [1024, 512, 256]
            }
            share_expert_net {
                hidden_units: [1024, 512, 256]
            }
        }
        extraction_networks {
            network_name: "layer2"
            expert_num_per_task: 3
            share_num: 3
            task_expert_net {
                hidden_units: [256, 128, 64]
            }
            share_expert_net {
                hidden_units: [256, 128, 64]
            }
        }
        extraction_networks {
            network_name: "layer3"
            expert_num_per_task: 4
            share_num: 4
            task_expert_net {
                hidden_units: [128, 64, 32]
            }
            share_expert_net {
                hidden_units: [128, 64, 32]
            }
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
        }
    }
}
```

- feature_groups: 配置一个名为'all'的feature_group。
- ple: ple相关的参数
  - extraction_networks
    - network_name
    - expert_num_per_task 每个目标任务的专家数
    - share_num 共享任务的专家数
    - task_expert_net 目标任务的专家网络的dnn参数配置
    - share_expert_net 共享任务的专家网络的dnn参数配置
  - task_towers 根据任务数配置task_towers
    - tower_name：TaskTower名
    - label_name: tower对应的label名
    - mlp: TaskTower的MLP参数配置
    - weight: 任务权重名
    - sample_weight_name: 样本权重列名
    - losses: 任务损失函数配置
    - metrics: 任务评估指标配置
    - task_space_indicator_label: 标识当前任务空间的目标名称，配合in_task_space_weight、out_task_space_weight使用。例如，对于cvr任务，可以设置task_space_indicator_label=clk，in_task_space_weight=1，out_task_space_weight=0，来使得cvr任务塔只在点击空间计算loss。
      - 注: in_task_space_weight和out_task_space_weight不影响loss权重的绝对值，权重会在batch维度被归一化。例如：in_task_space_weight=10,out_task_space_weight=1跟in_task_space_weight=1,out_task_space_weight=0.1是等价的。如需要提升这个task的loss权重的绝对值，需设置weight参数
    - in_task_space_weight: 对于task_space_indicator_label>0的样本会乘以该权重
    - out_task_space_weight: 对于task_space_indicator_label\<=0的样本会乘以该权重

## 示例Config

[ple_taobao.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/ple_taobao.config)

## 参考论文

[PLE](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)
