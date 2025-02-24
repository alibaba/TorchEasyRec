# MIND

mind召回模型, 在dssm的基础上加入了兴趣聚类功能，支持多兴趣召回，能够显著的提升召回层的效果.
![mind](../../images/models/mind.png)

## 配置说明

```
feature_configs {
    sequence_feature {
        sequence_name: "click_50_seq"
        sequence_length: 100
        sequence_delim: "|"
        features {
            id_feature {
                feature_name: "adgroup_id"
                num_buckets: 846812
                embedding_dim: 16
                expression: "item:adgroup_id"
            }
        }
        features {
            id_feature {
                feature_name: "cate_id"
                num_buckets: 12961
                embedding_dim: 16
                expression: "item:cate_id"
            }
        }
        features {
            id_feature {
                feature_name: "brand"
                num_buckets: 461498
                embedding_dim: 16
                expression: "item:brand"
            }
        }
    }

}


model_config {
    feature_groups {
        group_name: "user"
        feature_names: "user_id"
        feature_names: "cms_segid"
        feature_names: "cms_group_id"
        feature_names: "final_gender_code"
        feature_names: "age_level"
        feature_names: "pvalue_level"
        feature_names: "shopping_level"
        feature_names: "occupation"
        feature_names: "new_user_class_level"
        feature_names: "pid"
        group_type: DEEP
    }
    feature_groups {
        group_name: "item"
        feature_names: "adgroup_id"
        feature_names: "cate_id"
        feature_names: "campaign_id"
        feature_names: "customer"
        feature_names: "brand"
        feature_names: "price"
        group_type: DEEP
    }
    feature_groups {
        group_name: "hist"
        feature_names: "click_50_seq__adgroup_id"
        feature_names: "click_50_seq__cate_id"
        feature_names: "click_50_seq__brand"
        group_type: SEQUENCE
    }
    mind{
        user_tower{
            input: 'user'
            history_input: 'hist'
            user_mlp {
                hidden_units: [256, 128]
                use_bn: true
            }
            hist_seq_mlp {
                hidden_units: [256, 128]
                use_bn: true
            }
            capsule_config {
                max_k: 5
                max_seq_len: 64
                high_dim: 64
                squash_pow: 0.2
            }
            concat_mlp {
                hidden_units: [256, 128]
                use_bn: true
            }
        }
        item_tower{
            input: 'item'
            mlp {
                hidden_units: [256, 128]
                use_bn: true
            }
        }

        simi_pow: 20
        in_batch_negative: false

    }

    losses {
        softmax_cross_entropy {}
    }
}

```

- model_config: 配置3个feature_groups， group_name分别为'user', 'item', 'hist'(group_name可以按需自定义)，分别表示user, item和user历史行为序列的输入。hist group的类型为SEQUENCE，表示输入为序列特征。

- mind模型配置：

  - user_tower: 用户tower配置
    - input: 输入的feature_group 名称，即'user'
    - history_input: 输入历史行为序列的feature_group名称，即'hist'
    - user_mlp: 用户特征的mlp layer配置，包括隐藏层和BN层
    - hist_seq_mlp(可选): 历史行为序列特征的mlp layer配置，包括隐藏层和BN层
    - capsule_config: 兴趣胶囊配置
      - max_k(可选): 最大兴趣个数，默认为5
      - max_seq_len: 最大序列长度
      - high_dim: 兴趣向量维度
      - squash_pow(可选): 对squash加的power, 防止squash之后的向量值变得太小， 默认为1
      - num_iters(可选): 动态路由的迭代次数， 默认为3
      - const_caps_num(可选): 对所有用户使用相同的兴趣数量， 默认为False
      - routing_logits_scale(可选): 动态路由logits的初始化缩放系数， 默认为20
      - routing_logits_stddev(可选): 动态路由logits的初始化标准差， 默认为1
    - concat_mlp: 拼接用户特征和历史兴趣的mlp layer配置
  - item_tower: 物品tower配置，包括输入层和mlp layer
    - input: 输入的feature_group名称，即'item'
    - mlp: 物品特征的mlp layer配置，包括隐藏层和BN层。
  - simi_pow: 对相似度做的倍数, 放大interests之间的差异
  - in_batch_negative: 是否使用in-batch negative，默认为false。

## 示例Config

[mind_taobao.config](<>)

## 参考论文

[MIND.pdf](https://arxiv.org/pdf/1904.08030)
