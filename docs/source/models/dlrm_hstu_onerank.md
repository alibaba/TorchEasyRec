# DLRM HSTU OneRank

## 简介

[OneRank](https://arxiv.org/abs/2606.16838) 是一种 Transformer 原生的多任务排序架构：取消「编码器 + 预测头」的分离，让每个任务通过任务专属通道穿过序列模型本身，用逐任务内积完成打分。DlrmHSTUOneRank 在 DlrmHSTU（HSTU 生成式排序）之上实现了这一打分头：

- **任务专属编码**：每个候选膨胀为 `2K` 个 token（候选副本与任务 token 交替排列），任务私有注意力 mask 使得躯干对每个 (候选, 任务) 对输出一个独立表示 `r^i_k`；
- **情境辨识（SD）**：每个任务用请求上下文构造 query，对本请求候选池做交叉注意力，得到请求级向量 `h_k`；
- **跨任务注意力（CrossTask）**：`K` 个 `h_k` 在级联 mask 下互相注意（时长任务可读点击任务表示，反之不行），输出 `z_k`；
- **打分**：`s^i_k = z_k · r^i_k / sqrt(D) + b_k`，逐任务内积（另有 bilinear / MLP 变体）；
- **listwise InfoNCE**：同一请求内其他候选互为负样本。

损失、标签、bitmask 解码、指标全部继承自 `fusion_mtl_tower.task_configs`，与 DlrmHSTU 可在同一份配置下直接对比；唯一差异在 contextual token 的注意力模式：DlrmHSTU 把未显式设置的 `stu.contextual_seq_len` 解析为 contextual 特征数并给这些 token 一个双向注意力块，OneRank 固定为 0（普通 causal 前缀行）。做同配置 A/B 时注意这一基线差异也会计入指标差值。

注意：

- 该模型要求 `kernel: CUTLASS`（或 `PYTORCH`）。任务私有 mask 以 NFUNC func tensor 表达，Triton 注意力内核不支持。`kernel` 是 `model_config` 层级的字段（与 `dlrm_hstu_onerank {}` 平级），不要放进 `stu {}`。`ModelConfig.kernel` 默认 `PYTORCH`，即默认走参考内核；
- `kernel: CUTLASS` 时必须开启 bf16/fp16 混合精度（`train_config { mixed_precision: "BF16" }`），CUTLASS 注意力内核只接受 fp16/bf16；`PYTORCH` 参考内核无此要求；
- rank-1 内积打分在训练早期可能长时间停在「输出全局 CTR」的常数解平台上，单 epoch 训练建议改用 `scorer_type` 的 MLP/bilinear 变体或设置 `task_bias_init`，详见下方「训练动态：常数解平台期」。

## 配置说明

`dlrm_hstu_onerank` 的字段 1-7 与 `dlrm_hstu` 完全一致，现有 `dlrm_hstu` 配置块可直接复用；OneRank 会自动把 `stu.contextual_seq_len` 的 `-1` 哨兵值解析为 0。`fusion_mtl_tower.mlp` 与 `item_embedding_hidden_dim` 被本模型忽略（打分头替换了融合塔，`mlp` 为 optional 字段，可不配置）。

`task_configs` 的字段与 DlrmHSTU 完全一致：标签须为 jagged 候选序列标签（如 `cand_seq___action_weight`，配合 `task_bitmask` 解码），损失和指标分别用 `losses` / `metrics` 重复字段表达，GAUC 为 `metrics { grouped_auc { grouping_key: "user_id" } }`。

```
model_config {
    feature_groups {
        group_name: "contextual"
        feature_names: "user_id"
        feature_names: "user_active_degree"
        group_type: DEEP
    }
    feature_groups {
        group_name: "uih"
        feature_names: "uih_seq__video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "candidate"
        feature_names: "cand_seq___video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_action"
        feature_names: "uih_seq__action_weight"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_timestamp"
        feature_names: "uih_seq__action_timestamp"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "candidate_timestamp"
        feature_names: "cand_seq___query_time"
        group_type: JAGGED_SEQUENCE
    }
    dlrm_hstu_onerank {
        hstu {
            stu {
                embedding_dim: 512
                num_heads: 4
                hidden_dim: 128
                attention_dim: 128
                output_dropout_ratio: 0.0
            }
            input_dropout_ratio: 0.0
            positional_encoder {
                num_position_buckets: 8192
                num_time_buckets: 2048
                use_time_encoding: true
            }
            input_preprocessor {
                contextual_preprocessor {
                    action_encoder {
                        simple_action_encoder {
                            action_embedding_dim: 8
                            action_weights: 1
                            action_weights: 2
                            action_weights: 4
                        }
                    }
                    action_mlp {
                        simple_mlp { hidden_dim: 128 }
                    }
                    content_encoder {
                        slice_content_encoder {}
                    }
                    content_mlp {
                        simple_mlp { hidden_dim: 256 }
                    }
                }
            }
            output_postprocessor {
                layernorm_postprocessor {}
            }
        }
        fusion_mtl_tower {
            # mlp 被本模型忽略；可不配置（FusionMTLTower.mlp 为 optional，且
            # DlrmHSTUOneRank 不会构造该塔），此处仅为兼容复用配置而保留
            mlp {
                hidden_units: 256
                activation: "nn.SiLU"
            }
            task_configs {
                task_name: "is_click"
                label_name: "cand_seq___action_weight"
                task_bitmask: 1
                losses {
                    binary_cross_entropy {}
                }
                metrics {
                    auc {}
                }
                metrics {
                    grouped_auc {
                        grouping_key: "user_id"
                    }
                }
            }
            # ... 其余任务同格式
        }
        max_seq_len: 2048
        onerank {
            # 每请求候选数上界：同时决定膨胀序列的注意力归一化除数
            # 与静态最大序列长度；超出该上界的请求会在训练时被
            # 立即拒绝（self-explanatory 报错）
            max_num_candidates: 20
            situation_discernment {
                num_heads: 4
            }
            cross_task_head {
                num_heads: 4
                ffn_hidden_dim: 128
                # 任务 k 可「读」任务 j 的表示但不可「改写」
                gradient_detachment: true
            }
            # 同请求候选互为负样本的 listwise InfoNCE；
            # 要求对应任务配置了 binary_cross_entropy 或
            # binary_focal_loss 损失
            listwise_losses {
                task_name: "is_click"
                alpha: 0.1
                temperature_init: 0.07
                learnable_temperature: true
            }
            # 打分函数：DOT_PRODUCT（默认）/ BILINEAR / MLP
            scorer_type: ONERANK_SCORER_MLP
            scorer_hidden_dim: 256
            # 可选：每任务初始 logit，数量须等于任务数 K
            # （按 task_configs 顺序），建议 logit(全局 CTR)
            task_bias_init: -1.44
            task_bias_init: -2.10
        }
    }
    # kernel 是 model_config 层级字段（与 dlrm_hstu_onerank 平级）
    kernel: CUTLASS
}
train_config {
    # CUTLASS 注意力内核要求 fp16/bf16 混合精度
    mixed_precision: "BF16"
}
```

### 不支持的配置

以下组合会在**构造、首个 batch 或 serving 调用**时报错（`ValueError` / `NotImplementedError`），为避免长训后失败，请在配置阶段避开：

| 配置                                                                | 原因                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------- |
| `kernel: TRITON`                                                    | NFUNC func tensor 路径无 Triton 实现                          |
| `stu.sla_k1` / `stu.sla_k2` > 0                                     | OneRank 自建 func mask，与 SLA 区间互斥                       |
| `stu.max_attn_len` > 0                                              | 局部窗口需折叠进 func tensor 的两个区间，区间已被分组布局占用 |
| 显式 `stu.contextual_seq_len` > 0                                   | 分组布局将两个列区间都用于 `prefix + own group`               |
| `hstu.attn_truncation_split_layer` / `attn_truncation_tail_len` > 0 | 截断会破坏 func tensor 的静态签名缓存                         |
| `input_preprocessor.contextual_interleave_preprocessor`             | 目标交织会破坏固定分组步长                                    |
| `num_class > 1`                                                     | 任务 token 通道只支持二分类任务                               |
| KV-cache 增量推理（`OneRankSTULayer.cached_forward`）               | 增量推理未实现，serving 侧调用时才抛 `NotImplementedError`；导出不受影响，见下文 |

模型导出不受上表影响：`tzrec.export` 不会拒绝 `dlrm_hstu_onerank`，也不要求 `dlrm_hstu` / `ultra_hstu` 导出所必需的 `additional_export_config.cand_seq_pk`——那是 KV-cache 增量 serving 的契约键，OneRank 无此路径。导出产物按全量 forward 加载推理即可；上表的 `NotImplementedError` 只会在 serving 栈调用 `OneRankSTULayer.cached_forward` 时抛出，而非在导出时。

### 关键参数

| 参数                            | 说明                                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `max_num_candidates`            | 每请求候选数上界；膨胀后的序列长度为 `max_seq_len + max_num_candidates * 2K`。实际候选数超出上界的请求会在训练时被立即报错拒绝           |
| `onerank.situation_discernment` | SD 模块；不配置时退化为候选表示的均值池化（论文 V5 显示这是最伤的消融）                                                                  |
| `onerank.cross_task_head`       | 跨任务注意力；`mask_type` 默认 CASCADE（时长任务读点击任务），`gradient_detachment` 默认 true                                            |
| `onerank.listwise_losses`       | 逐任务 listwise InfoNCE；对应任务须配置 `binary_cross_entropy` 或 `binary_focal_loss` 损失；每个请求需同时含至少一个正样本与一个负样本才计入该项损失，因此只对「几乎每个请求都含正样本」的任务值得开启（全正样本的请求同样被屏蔽：无负样本即无排序信号） |
| `onerank.scorer_type`           | 打分函数；单 epoch 训练建议 MLP 或 BILINEAR 规避常数解平台期                                                                             |
| `onerank.task_bias_init`        | 每任务初始 logit（`task_configs` 顺序，数量须等于 K）；设为 logit(全局 CTR) 可跳过向常数解的下降                                         |

## 训练动态：常数解平台期

论文的打分函数是 **rank-1** 的：`z_k` 是每请求一个向量，广播到该请求所有候选，请求内排序退化为候选表示在单一方向上的投影。训练早期躯干输出的候选表示还接近平坦，单一方向无法区分候选，「平坦表示 + rank-1 打分」的 BCE 最优解就是**常数预测**——每个候选都拿到全局正样本率。

这个常数是目标函数的真实局部最优，模型可能在整个单 epoch 训练的大部分时间里停在它上面：

- 下降到常数解只需约 100 步；
- 从常数解**逃逸**（躯干在微弱的 rank-1 信号下学出判别性表示）所需时间要高一个数量级，且逃逸步数对随机初始化高度敏感；
- 单 epoch 训练可能落在逃逸窗口之前、之中或之后，表现为相同配置重复跑时 AUC 的多点发散。

`DlrmHSTU` 没有这个失败模式：它的 MLP 头逐候选映射，判别性梯度从第 1 步就存在，初始化随机性被收敛冲刷掉。

这就是 `OneRankScorerType` 与 `task_bias_init` 存在的原因：

| 对策                                   | 作用                                                                                            |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `scorer_type: ONERANK_SCORER_MLP`      | `MLP_k([z_k; r^i_k]) + b_k`；随机初始化即可判别候选，训练不进入平台期                           |
| `scorer_type: ONERANK_SCORER_BILINEAR` | `z_k . (W_k r^i_k) / sqrt(D) + b_k`，`W_k` 单位阵初始化：起点与内积完全一致，训练中可把秩提离 1 |
| `task_bias_init: logit(CTR_k)`         | 把 bias 直接放在平台期的*正确*位置：跳过向常数解的下降，缩短（但不消除）平台期                  |
| 更多步数 / epoch                       | 每个运行最终都会逃逸；不改 `scorer_type` 则天花板仍是 rank-1                                    |
| 重复多次取均值                         | 不修复任何东西，但让单 epoch 指标可信；发散本身就是逃逸时间彩票                                 |

## 参考论文

[OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation](https://arxiv.org/abs/2606.16838)
