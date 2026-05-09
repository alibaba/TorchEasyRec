# ULTRA HSTU

## 简介

UltraHSTU 实现了 Meta 在 [*Bending the Scaling Law Curve in Large-Scale Recommendation Systems*](https://arxiv.org/abs/2602.16986)（Ding 等, 2026, arXiv:2602.16986）中提出的 ULTRA-HSTU 架构 —— 在 [DlrmHSTU](dlrm_hstu.md) 之上叠加四项正交的效率优化，使得长序列推荐模型在保持/提升精度的同时大幅降低算力与显存开销：

- **Semi-Local Attention (SLA)** —— 把每个 STU layer 的 causal attention 限制在「最近 `sla_k1` 个 token 的局部窗口」+「全局 prefix 中的 `sla_k2` 个 token」上。`effective_k2 = max(sla_k2, contextual_seq_len)`：contextual prefix 自动作为全局 prefix 的一部分。复杂度由 O(L²) 降到 O(L · K1) + O(L · K2)。要求 `Kernel.CUTLASS` 或 `Kernel.PYTORCH`（Triton 后端没有 NFUNC mask path）。

- **Mid-stack Attention Truncation** —— 在第 `attn_truncation_split_layer` 个 STU layer 之后丢弃 UIH 前缀，仅保留尾部 `attn_truncation_tail_len` 个 token 进入后续 layer。**Contextual prefix 与 target tokens 始终保留**；只截断 UIH。把后半段 layer 的 attention 输入压缩到 `contextual_seq_len + attn_truncation_tail_len + num_targets`，进一步压缩长序列上的 KV 占用与算力。

- **Mixture of Transducers (MoT)** —— 模型并行运行 N 个 `HSTUTransducer`，每个对应一路 UIH 通道（如点击、长播等）；每个通道有自己的 STU 栈、自己的 SLA / truncation 配置。各通道针对 candidate 的输出 embedding 在 channel 维上拼接后喂给统一的 `FusionMTLTower`。**Candidate 与 contextual 共享**，UIH 按通道拆分；channel 数由 `repeated HSTU hstu` 决定。

- **Selective Rematerialization** —— STU layer 的反向传播按需重算两类中间张量以省显存：`recompute_normed_x_in_backward` 控制 LayerNorm 输出是否存储，`recompute_uvqk_in_backward` 控制 (U, V, Q, K) 投影是否存储。两者均默认 `true`，对 attention activation memory 占主导的大模型尤其重要。

> Single-channel `UltraHSTU`（即只声明一个 `hstu` entry，且不设 `name`）在行为上等价于 DlrmHSTU，可以无痛地从 DlrmHSTU 迁移；SLA 与 Attention Truncation 即使在 single-channel 配置下也可独立启用。

注意：与 DlrmHSTU 一样，该模型的样本格式与传统推荐模型不同，一个用户一个时间窗内的行为会聚合成单条样本。

## 配置说明

UltraHSTU 的 proto 定义见 `tzrec/protos/models/multi_task_rank.proto:UltraHSTU`。它复用了 DlrmHSTU 的所有顶层字段（`fusion_mtl_tower` / `max_seq_len` / `item_embedding_hidden_dim` / `enable_global_average_loss` / `sequence_timestamp_is_ascending` / `concat_contextual_features`），只把单个 `required HSTU hstu` 替换成了 `repeated HSTU hstu`。

每个通道的 `HSTU` 子配置必须设置 `name`（参见 `tzrec/protos/module.proto:HSTU.name`）；UIH 侧的 `feature_groups` 则按 `name` 命名约定排布：

| HSTU.name        | 对应的 feature_groups                                                |
| ---------------- | -------------------------------------------------------------------- |
| `<name>`         | `<name>` / `<name>_action` / `<name>_watchtime` / `<name>_timestamp` |
| 空字符串（默认） | `uih` / `uih_action` / `uih_watchtime` / `uih_timestamp`             |

完整的双通道示例见 `tzrec/tests/configs/ultra_hstu_cutlass_kuairand_1k.config`。下面给出关键片段：

```
model_config {
    feature_groups {
        group_name: "contextual"
        feature_names: "user_id"
        feature_names: "user_active_degree"
        group_type: DEEP
    }

    # Channel "uih_click" -- the click stream.
    feature_groups {
        group_name: "uih_click"
        feature_names: "click_seq__video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_click_action"
        feature_names: "click_seq__action_weight"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_click_watchtime"
        feature_names: "click_seq__watch_time"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_click_timestamp"
        feature_names: "click_seq__action_timestamp"
        group_type: JAGGED_SEQUENCE
    }

    # Channel "uih_view" -- the long-view stream.
    feature_groups {
        group_name: "uih_view"
        feature_names: "view_seq__video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_view_action"
        feature_names: "view_seq__action_weight"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_view_watchtime"
        feature_names: "view_seq__watch_time"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "uih_view_timestamp"
        feature_names: "view_seq__action_timestamp"
        group_type: JAGGED_SEQUENCE
    }

    # Shared candidate-side groups.
    feature_groups {
        group_name: "candidate"
        feature_names: "cand_seq__video_id"
        group_type: JAGGED_SEQUENCE
    }
    feature_groups {
        group_name: "candidate_timestamp"
        feature_names: "cand_seq__query_time"
        group_type: JAGGED_SEQUENCE
    }

    ultra_hstu {
        hstu {
            name: "uih_click"
            stu {
                embedding_dim: 512
                num_heads: 4
                hidden_dim: 128
                attention_dim: 128
                # SLA: local window 256 + global prefix 32 (or contextual_seq_len, whichever is larger).
                sla_k1: 256
                sla_k2: 32
            }
            attn_num_layers: 4
            # Mid-stack truncation: drop the UIH prefix after layer 2,
            # keep only the last 512 UIH tokens for layers 2..3.
            attn_truncation_split_layer: 2
            attn_truncation_tail_len: 512
            ...
        }
        hstu {
            name: "uih_view"
            stu {
                embedding_dim: 512
                num_heads: 4
                hidden_dim: 128
                attention_dim: 128
                sla_k1: 256
                sla_k2: 32
            }
            attn_num_layers: 4
            attn_truncation_split_layer: 2
            attn_truncation_tail_len: 512
            ...
        }
        fusion_mtl_tower { ... }
        max_seq_len: 4096
    }
    kernel: CUTLASS
}
```

### 字段说明

仅列与 DlrmHSTU 不同的字段；其余字段（`fusion_mtl_tower` / `max_seq_len` / `item_embedding_hidden_dim` / `enable_global_average_loss` / `sequence_timestamp_is_ascending` / `concat_contextual_features`）与 DlrmHSTU 一致。

- **`hstu`**：`repeated HSTU`。≥ 2 个时每个 entry 必须设置唯一非空 `name`。各 channel 的 STU `embedding_dim` 不要求一致；item 侧 MLP 与 `FusionMTLTower` 的输入维度自动取所有 channel `embedding_dim` 之和。
- **`HSTU.name`**：MoT 通道名。非空时通道名 *替换* 默认 `uih` 前缀，preprocessor 据此从 `grouped_features` 读取 `<name>.sequence` / `<name>_action.sequence` / `<name>_watchtime.sequence` / `<name>_timestamp.sequence`。
- **`HSTU.stu.sla_k1` / `sla_k2`**：SLA 的局部窗口长度与全局 prefix 长度。任一 `> 0` 即启用 SLA；要求 `kernel: CUTLASS`（或 `PYTORCH`）。
- **`HSTU.attn_truncation_split_layer` / `attn_truncation_tail_len`**：mid-stack truncation 的分裂 layer 索引与 UIH 尾部保留长度。两者都必须 `> 0` 才启用，单独设置其一会被拒绝。

### Embedding 表共享

只要多个通道在同一物理特征上声明同名 `embedding_name`，`EmbeddingGroup` 就会 dedupe 成一张表（详见 `tzrec/modules/embedding.py:EmbeddingGroup._add_embedding_config`）。**默认应当共享**；只在需要每通道独立 embedding 表的特殊场景才使用每通道独立的 `embedding_name`，否则 sparse 参数量、TBE forward/backward 计算量和 all-to-all 通信量都会按通道数线性放大。

## 示例Config

[ultra_hstu_kuairand.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/models/ultra_hstu_kuairand.config)

## 参考论文

[ULTRA-HSTU: Bending the Scaling Law Curve in Large-Scale Recommendation Systems](https://arxiv.org/abs/2602.16986)
