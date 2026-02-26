# PEPNet

## 简介

PEPNet将带有个性化先验信息的特征作为输入，通过门控机制，来动态地缩放底层网络-embedding layer和顶层网络-DNN隐藏层单元，分别称之为场景特定的EPNet和任务特定的PPNet：

- Embedding Personalized Network (EPNet) 为底层网络添加场景特定的个性化信息来生成个性化embedding门控，用于执行来自多个场景的原始embedding选择，以生成个性化的embedding
- Parameter Personalized Network (PPNet) 将用户和items的个性化信息与每一个task tower的DNN的输入进行拼接来获得个性化的门控分数，然后采用element-wise product应用到DNN的隐藏层单元上，来个性化优化DNN的参数。

PEPNet的整体结构如下图所示，可以看到，核心的组件便是这三个：Gate NU（门控网络单元）、Embedding Personalized Network (EPNet)、Parameter Personalized Network (PPNet)
![pepnet.jpg](../../images/models/pepnet.jpg)

## 模型配置

```protobuf
model_config: {
  feature_groups: {
    group_name: 'all'
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
    feature_names: 'price'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'domain'
    feature_names: 'occupation'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'uia'
    feature_names: 'user_id'
    feature_names: 'adgroup_id'
    wide_deep: DEEP
  }
  pepnet {
    main_group_name: "all"
    domain_group_name: "domain"
    epnet_hidden_unit: 128,
    uia_group_name: "uia"
    ppnet_hidden_units: [512, 256]
    ppnet_dropout_ratio: [0.1, 0.1]
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

- feature_groups: 特征组

  - 通常情况下有3个feature_group: 名称自定义，根据pepnet的配置，分为3类：all, domain, uia，其中domain和uia是可选配置，根据需求进行配置。
  - wide_deep: pepnet模型使用的都是Deep features, 所以都设置成DEEP

- pepnet: pepnet模型相关的参数

  - main_group_name: 主特征组名称,和feature_groups中的group_name对应
  - domain_group_name: domain特征组名称,和feature_groups中的group_name对应，是epnet场景个性化部分的输入
  - epnet_hidden_unit: epnet的gateGu的隐层设置，一般介于domain的dim和主特征组dim之间
  - uia_group_name: 用户和item的个性化信息，和feature_groups中的group_name对应，是ppnet用户-商品个性化部分的输入
  - ppnet_hidden_units: 个性化tower的隐藏层设置
  - task_towers: 根据任务数配置task_towers
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

## 模型输出

和其余多任务模型一样，每个塔的输出名为："logits\_" / "probs\_" / "y\_" + tower_name
其中，logits/probs/y对应: sigmoid之前的值/概率/回归模型的预测值
MMoE模型每个塔的指标为：指标名+ "\_" + tower_name

## 示例Config

[pepnet_taobao.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/pepnet_taobao.config)

## 参考论文

[PEPNet.pdf](https://arxiv.org/pdf/2302.01115)
