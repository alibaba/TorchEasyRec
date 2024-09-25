# DIN

## 简介

利用DIN算法建模用户点击序列。支持多组序列共同embedding，如hist_item_id, hist_category_id。

![din.png](../../images/models/din.png)

## 模型配置

```
model_config {
    feature_groups {
        group_name: "deep"
        feature_names: "user_id"
        feature_names: "cms_segid"
        feature_names: "cms_group_id"
        feature_names: "final_gender_code"
        ...
        group_type: DEEP
    }
    feature_groups {
        group_name: "seq"
        feature_names: "adgroup_id"
        feature_names: "cate_id"
        feature_names: "brand"
        feature_names: "click_50_seq__adgroup_id"
        feature_names: "click_50_seq__cate_id"
        feature_names: "click_50_seq__brand"
        group_type: SEQUENCE
    }
    multi_tower_din {
        towers {
            input: 'deep'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        din_towers {
            input: 'seq'
            attn_mlp {
                hidden_units: [256, 64]
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

- feature_groups: 可配置多个feature_group，group name可以变。
- multi_tower_din: multi_tower_din相关的参数
  - towers: 每个deep feature_group对应了一个tower。
    - input: 跟feature_group的group_name对应
    - mlp: mlp的参数配置
      - hidden_units: dnn每一层的channel数目，即神经元的数目
  - din_towers: 每个sequence feature_group对应了一个din_tower
    - input: 跟feature_group的group_name对应
    - attn_mlp: target attention mlp的参数配置
      - hidden_units: dnn每一层的channel数目，即神经元的数目
  - final: 整合towers和din_towers的mlp参数配置
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- losses: 损失函数配置
- metrics: 评估指标配置

**备注**

需保证每个SEQUENCE feature_group中的序列特征的序列长度相同

## 模型输出

模型的输出名为: "logits" / "probs" / "y", 对应sigmoid之前的值/概率/回归模型的预测值

## 示例config

[multi_tower_din_taobao.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/multi_tower_din_taobao.config)

## 参考论文

[Deep Interest Network](https://arxiv.org/abs/1706.06978)
