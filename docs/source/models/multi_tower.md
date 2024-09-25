# MultiTower

## 简介

- 多塔模型效果比单塔模型有明显的提升
- 不采用FM，所以embedding可以有不同的dimension。

![multi_tower.png](../../images/models/multi_tower.png)

## 模型配置

```
model_config {
    feature_groups {
        group_name: "user"
        feature_names: "user_id"
        feature_names: "cms_segid"
        ...
        group_type: DEEP
    }
    feature_groups {
        group_name: "item"
        feature_names: "adgroup_id"
        feature_names: "cate_id"
        ...
        group_type: DEEP
    }
    multi_tower {
        towers {
            input: 'user'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        towers {
            input: 'item'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        final {
            hidden_units: [64]
        }
    }
    metrics {
        auc {}
    }
    losses {
        binary_cross_entropy {}
    }
}
```

- feature_groups:  可配置多个feature_group，group name可以变
- multi_tower: multi_tower相关的参数
  - towers: 每个deep feature_group对应了一个tower。
    - input: 跟feature_group的group_name对应
    - mlp: mlp的参数配置
      - hidden_units: dnn每一层的channel数目，即神经元的数目
  - final: 整合towers和din_towers的mlp参数配置
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- losses: 损失函数配置
- metrics: 评估指标配置

## 模型输出

模型的输出名为: "logits" / "probs" / "y", 对应sigmoid之前的值/概率/回归模型的预测值

## 示例config

[multi_tower_demo.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/multi_tower_taobao.config)
