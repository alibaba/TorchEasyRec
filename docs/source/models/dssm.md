# DSSM

## 简介

双塔召回模型，支持训练时负采样。

![dssm](../../images/models/dssm_neg_sampler.png)

当物品池很大上百万甚至是上亿的时候，双塔召回模型常常需要在物品池中针对每个正样本采样一千甚至一万的负样本才能达到比较好的召回效果，
意味着正负样本比例达到了1: 1k，甚至是1: 1w， 要支持这个正负样本比例的训练，如果用离线构造样本的方式会导致离线存储和离线计算的压力都激增。

TorchEasyRec的DSSM支持运行时进行负采样，会以图存储的方式将物品的特征分布式地存储在节点上，并且Mini-Batch内的共享同一批负样本的计算，
使得离线存储和离线计算的压力都大大降低。

注：训练样本一般只需准备点击（正样本）的样本即可

## 配置说明

```
data_config: {
    ...
    negative_sampler {
        input_path: "data/test/tb_data/taobao_ad_feature_gl_v1"
        num_sample: 1024
        attr_fields: "adgroup_id"
        attr_fields: "cate_id"
        attr_fields: "campaign_id"
        attr_fields: "customer"
        attr_fields: "brand"
        attr_fields: "price"
        item_id_field: "adgroup_id"
        attr_delimiter: "\x02"
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
    dssm {
        user_tower {
            input: 'user'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        item_tower {
            input: 'item'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        output_dim: 64
    }
    metrics {
        recall_at_k {
            top_k: 1
        }
    }
    metrics {
        recall_at_k {
            top_k: 5
        }
    }
    losses {
        softmax_cross_entropy {}
    }
}
```

- data_config: 数据配置，其中需要配置负采样Sampler，负采样Sampler的配置详见**负采样配置**
- feature_groups: 需要两个feature_group: user和item
- 支持dssm和dssm_v2二选一
  - dssm: dssm相关的参数
    - user_tower/item_tower:
      - mlp: MLP的参数配置
        - input: 输入feature_group名
        - hidden_units: mlp每一层的channel数目，即神经元的数目
    - output_dim: user/item输出embedding维度
    - similarity: 向量相似度函数，包括[COSINE, INNER_PRODUCT]，默认INNER_PRODUCT
  - dssm_v2: 参数同dssm
    - dssm_v2可以支持user与item塔 跨塔share embedding，但训练速度相对dssm_v1稍慢
    - 注意如果使用dssm_v2，data_config.force_base_data_group需要设置为true
- losses: 损失函数配置, 目前只支持softmax_cross_entropy
- metrics: 评估指标配置，目前只支持recall_at_topk

注意，DSSM负采样版目前仅支持recall_at_topk做评估指标。

### 负采样配置

目前支持两种负采样Sampler：

- negative_sampler：加权随机负采样，会排除Mini-Batch内的Item Id
  - input_path: 负采样Item表, Schema为: id:int64 | weight:float | attrs:string，其中attr默认为":"分隔符拼接的Item特征
  - num_sample: 训练worker的负采样数
  - num_eval_sampler: 评估worker的负采样数
  - attr_fields: Item特征名，顺序与Item的attr中特征的拼接顺序保持一致
  - item_id_field: item_id列名
- negative_sampler_v2：加权随机负采样，会跟排除Mini-Batch内的User有边的Item Id
  - user_input_path: User表, Schema为: id:int64 | weight:float
  - item_input_path: 负采样Item表, Schema为: id:int64 | weight:float | attrs:string，其中attr默认为":"分隔符拼接的Item特征
  - pos_edge_input_path: Positive边表, Schema为: userid:int64 | itemid:int64 | weight:float
  - user_id_field: user_id列名
  - 其余同negative_sampler

注:

- 如果负采样表为本地文件，分隔符必须是"\\t"。以负采样Item表为例，第一行固定为 "id:int64\\tweight:float\\tattrs:string"，后续列中的id，weight，attr也以"\\t"分隔
- 负采样表中的id列也可以为string类型，当id为string类型时，模型训练和评估命令中需设置环境变量USE_HASH_NODE_ID=1来启动对id自动进行hash64操作，一般情况下hash冲突概率极低

## 示例Config

[dssm_taobao.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/dssm_taobao.config)

## 参考论文

[DSSM.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
