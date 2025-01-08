# MMoE

## 简介

常用的多任务模型的预测质量通常对任务之间的关系很敏感。由于MMoE有多个expert，每个expert有不同的gate。因此当任务之间相关性低的时候，不同任务依赖不同的expert，MMoE依旧表现良好。
![mmoe.png](../../images/models/mmoe.png)

## 配置说明

```protobuf
model_config {
    feature_groups {
        group_name: "all"
        feature_names: "user_id"
        feature_names: "cms_segid"
        ...
        feature_names: "price"
        group_type: DEEP
    }
    mmoe {
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
        }
    }
}
```

- feature_groups: 配置一个名为'all'的feature_group。
- mmoe: mmoe相关的参数
  - expert_mlp: MMOE的专家MLP配置
    - hidden_units: mlp每一层的channel数目，即神经元的数目
  - expert_num: 专家MLP的数目
  - task_towers: 根据任务数配置task_towers
    - tower_name：TaskTower名
    - label_name: tower对应的label名
    - mlp: TaskTower的MLP参数配置
    - losses: 任务损失函数配置
    - metrics: 任务评估指标配置
    - task_space_indicator_label: 标识当前任务空间的目标名称，配合in_task_space_weight、out_task_space_weight使用。例如，对于cvr任务，可以设置task_space_indicator_label=ctr，in_task_space_weight=1，out_task_space_weight=0，来使得cvr任务塔只在点击空间计算loss。
    - in_task_space_weight: 对于task_space_indicator_label>0的样本会乘以该权重
    - out_task_space_weight: 对于task_space_indicator_label\<=0的样本会乘以该权重

## 模型输出

MMoE模型每个塔的输出名为："logits\_" / "probs\_" / "y\_" + tower_name
其中，logits/probs/y对应: sigmoid之前的值/概率/回归模型的预测值
MMoE模型每个塔的指标为：指标名+ "\_" + tower_name

## 示例Config

[mmoe_taobao.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/mmoe_taobao.config)

## 参考论文

[MMoE.pdf](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
