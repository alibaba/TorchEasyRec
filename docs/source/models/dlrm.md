# DLRM

### 简介

DLRM(Deep Learning Recommendation Model for Personalization and Recommendation Systems[Facebook])是一种DNN模型, 支持使用连续值特征(price/age/...)和ID类特征(user_id/item_id/...), 并对特征之间的交互(interaction)进行了建模(基于内积的方式).

```
output:
                    probability of a click
model:                       |
       _________________>DNN(top)<___________
      /                      |               \
     /_________________>INTERACTION <_________\
    //                                        \\
  DNN(bot)                         ____________\\_________
   |                              |                       |
   |                         _____|_______           _____|______
   |                        |_Emb_|____|__|    ...  |_Emb_|__|___|
input:
[ dense features ]          [sparse indices] , ..., [sparse indices]
```

### 配置说明

#### 1. 内置模型

```protobuf
model_config: {
  feature_groups: {
    group_name: 'dense'
    feature_names: 'price'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'sparse'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'occupation'
    feature_names: 'new_user_class_level'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'
    wide_deep: DEEP
  }
  dlrm {
      bot_mlp {
        hidden_units: 64
        hidden_units: 16
      }
      top_mlp {
        hidden_units: 128
        hidden_units: 64
        hidden_units: 32
      }
      arch_with_sparse: true
    }
  num_class: 1
  metrics {
      auc {}
  }
  losses {
      binary_cross_entropy {}
  }
}
```

- feature_groups: 特征组

  - 包含两个feature_group: dense 和sparse group, **group name不能变**

  - wide_deep: dlrm模型使用的都是Deep features, 所以都设置成DEEP

- dlrm: dlrm模型相关的参数

  - bot_mlp: dense mlp的参数配置

    - hidden_units: dnn每一层的channel数目，即神经元的数目,输入dense features,最后一层channel数必须等于sparce feature得维度

  - top_mlp: 输出(logits)之前的mlp, 输入为dense features, sparse features and interact features.

    - hidden_units: dnn每一层的channel数目，即神经元的数目

  - arch_with_sparse:

    - 默认是true, sparse features也会和dense features以及interact features concat起来, 然后进入top_mlp.
    - if false, 即仅将dense features和interact features concat起来，输入bot_dnn.

### 示例Config

[DLRM_demo.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/dlrm_criteo.config)

### 参考论文

[DLRM](https://arxiv.org/abs/1906.00091)
