# HSTU Match

## 简介

HSTU Match 是基于 HSTU (Hierarchical Sequential Transduction Units) 生成式架构的双塔召回模型。User Tower 使用 HSTU 对用户行为序列 (UIH) 进行建模，输出用户表征；Item Tower 对候选 Item Embedding 做可选 MLP 投影，与 user embedding 通过相似度函数计算 logits。支持训练时负采样，与 DSSM 一致。

相较于 DSSM，HSTU Match 直接基于用户原始的超长行为序列建模，引入位置/时间编码与 action encoder，可获得更强的序列表达能力；相较于 DLRM-HSTU (排序模型)，HSTU Match 输出的是单条 user embedding，用于召回阶段的近似最近邻 (ANN) 检索。

## 配置说明

```
data_config {
    ...
    label_fields: ["cand_seq__action_weight", "cand_seq__watch_time"]
    force_base_data_group: true
    negative_sampler {
        input_path: "odps://{PROJECT}/tables/taobao_ad_feature_gl_bucketized_v1"
        num_sample: 128
        attr_fields: "video_id"
        item_id_field: "cand_seq__video_id"
        attr_delimiter: "\t"
    }
}
feature_configs {
    id_feature {
        feature_name: "user_id"
        expression: "user:user_id"
        embedding_dim: 32
        num_buckets: 10000000
    }
}
feature_configs {
    id_feature {
        feature_name: "user_active_degree"
        expression: "user:user_active_degree"
        embedding_dim: 32
        num_buckets: 8
    }
}
feature_configs {
    id_feature {
        feature_name: "follow_user_num_range"
        expression: "user:follow_user_num_range"
        embedding_dim: 32
        num_buckets: 9
    }
}
feature_configs {
    id_feature {
        feature_name: "fans_user_num_range"
        expression: "user:fans_user_num_range"
        embedding_dim: 32
        num_buckets: 9
    }
}
feature_configs {
    id_feature {
        feature_name: "friend_user_num_range"
        expression: "user:friend_user_num_range"
        embedding_dim: 32
        num_buckets: 8
    }
}
feature_configs {
    id_feature {
        feature_name: "register_days_range"
        expression: "user:register_days_range"
        embedding_dim: 32
        num_buckets: 8
    }
}
feature_configs {
    sequence_feature {
        sequence_name: "uih_seq"
        sequence_length: 4096
        sequence_delim: "|"
        features {
            id_feature {
                feature_name: "video_id"
                expression: "item:video_id"
                embedding_name: "video_id_emb"
                embedding_dim: 512
                num_buckets: 10000000
            }
        }
        features { raw_feature { feature_name: "action_timestamp" expression: "user:action_timestamp" } }
        features { raw_feature { feature_name: "action_weight"    expression: "user:action_weight"    } }
        features { raw_feature { feature_name: "watch_time"       expression: "user:watch_time"       } }
    }
}
feature_configs {
    sequence_feature {
        sequence_name: "cand_seq"
        sequence_length: 100
        sequence_delim: "|"
        features {
            id_feature {
                feature_name: "video_id"
                expression: "item:video_id"
                embedding_name: "video_id_emb"
                embedding_dim: 512
                num_buckets: 10000000
            }
        }
    }
}
model_config {
    feature_groups {
        group_name: "contextual"
        feature_names: "user_id"
        feature_names: "user_active_degree"
        feature_names: "follow_user_num_range"
        feature_names: "fans_user_num_range"
        feature_names: "friend_user_num_range"
        feature_names: "register_days_range"
        group_type: DEEP
    }
    feature_groups {
        group_name: "uih"
        feature_names: "uih_seq__video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "candidate"
        feature_names: "cand_seq__video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_action"
        feature_names: "uih_seq__action_weight"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_watchtime"
        feature_names: "uih_seq__watch_time"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_timestamp"
        feature_names: "uih_seq__action_timestamp"
        group_type: JAGGED_SEQUENCE
    }
    hstu_match {
        user_tower {
            input: "uih"
            hstu {
                stu {
                    embedding_dim: 512
                    num_heads: 4
                    hidden_dim: 128
                    attention_dim: 128
                    output_dropout_ratio: 0.1
                    use_group_norm: true
                }
                input_dropout_ratio: 0.2
                attn_num_layers: 3
                positional_encoder {
                    num_position_buckets: 8192
                    num_time_buckets: 2048
                    use_time_encoding: true
                }
                input_preprocessor {
                    uih_preprocessor {
                        action_encoder {
                            simple_action_encoder {
                                action_embedding_dim: 8
                                action_weights: [1, 2, 4, 8, 16, 32, 64, 128]
                            }
                        }
                        action_mlp {
                            simple_mlp {
                                hidden_dim: 64
                            }
                        }
                    }
                }
                output_postprocessor {
                    l2norm_postprocessor {}
                }
            }
            max_seq_len: 4096
        }
        item_tower {
            input: "candidate"
            mlp {
                hidden_units: 512
                activation: ""
            }
        }
        similarity: COSINE
        temperature: 0.05
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
    kernel: TRITON
}
```

> The full runnable counterpart of this snippet is `tzrec/tests/configs/hstu_kuairand_1k.config` — it drives the HSTUMatch integration test on the KuaiRand-1K fixture and mirrors the sample above one-to-one.

- data_config: 数据配置，其中需要配置负采样 Sampler，负采样 Sampler 的配置详见 [DSSM](dssm.md) 文档中的**负采样配置**章节

  - HSTUMatch 的候选侧是 `sequence_feature` 的子特征。在 `negative_sampler` 中，`item_id_field` 写为带 `sequence_name` 前缀的名（例如 `cand_seq__video_id`），`attr_fields` 写为不带前缀的子特征名（例如 `video_id`）。

- feature_groups: 特征组

  - uih: 用户历史行为序列，可增加 side info；类型为 JAGGED_SEQUENCE，**必填**
  - candidate: 候选 Item 序列 (训练时由正样本+负采样物品组成)；类型为 JAGGED_SEQUENCE，**必填**
  - contextual: 用户侧的 ID 特征；类型为 DEEP，可选
  - uih_action: 用户历史交互的行为事件序列，注: 该行为事件按位存储，如 expr, click, add, buy 三个行为，则一般 expr=0, click=1, add=2, buy=4；类型为 JAGGED_SEQUENCE，当 `uih_preprocessor.action_encoder` 配置时必填
  - uih_watchtime: 用户历史交互的行为时长序列；类型为 JAGGED_SEQUENCE，当 action encoder 需要 watchtime 时必填
  - uih_timestamp: 用户历史交互的行为时间戳序列；类型为 JAGGED_SEQUENCE，当 `positional_encoder.use_time_encoding=true` 时必填

  **group_name 不能变**，user_tower/item_tower 通过 group_name 索引对应的 feature_group

- hstu_match: hstu_match 模型相关的参数

  - user_tower: 用户塔，对 UIH 进行 HSTU 编码
    - input: 用户行为序列 feature_group 名 (一般为 "uih")
    - hstu: HSTU 模型参数配置，与 [DLRM-HSTU](dlrm_hstu.md) 一致
      - stu: STU 模块配置
      - input_dropout_ratio: 输入是否使用 dropout
      - attn_num_layers: STU 层数
      - positional_encoder: 位置时间编码配置
      - input_preprocessor: 输入特征预处理配置，主要用于 contextual 和 action 特征处理
      - output_postprocessor: 输出后处理配置，主要用于 normalization
    - max_seq_len: 最大序列长度
  - item_tower: 物品塔
    - input: 候选 Item 序列 feature_group 名 (一般为 "candidate")
    - mlp: MLP 投影；当未配置 `output_dim` 时 (默认)，需将 mlp 的最后一层 `hidden_units` 设置为 `user_tower.hstu.stu.embedding_dim`，使 user/item 输出维度匹配
  - output_dim: 可选，user/item 输出 embedding 维度；默认 0，表示不再追加 output Linear，由 user 塔的 STU 输出与 item 塔的 MLP 输出直接对齐
  - similarity: 向量相似度函数，包括 [COSINE, INNER_PRODUCT]，默认 INNER_PRODUCT (示例使用 COSINE)
  - temperature: 相似度缩放因子，softmax 前对 logits 除以该值，默认 1.0

- kernel: 算子实现，可选 TRITON/PYTORCH/CUTLASS，详见 [DLRM-HSTU](dlrm_hstu.md) 文档

- losses: 损失函数配置，目前只支持 softmax_cross_entropy

- metrics: 评估指标配置，目前只支持 recall_at_topk

注意：

- 暂不支持 in_batch_negative，请使用 NegativeSampler/HardNegativeSampler。
- data_config.force_base_data_group 需要设置为 true。

## 模型导出

HSTU Match 模型导出时需要设置环境变量 `ENABLE_AOT=1` 启用 AOT Inductor 导出。例如:

```
ENABLE_AOT=1 torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/hstu_match/pipeline.config \
    --export_dir experiments/hstu_match/export
```

## 参考论文

[HSTU](https://arxiv.org/abs/2402.17152)
