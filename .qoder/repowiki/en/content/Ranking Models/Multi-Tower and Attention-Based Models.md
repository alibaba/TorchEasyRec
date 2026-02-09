# Multi-Tower and Attention-Based Models

<cite>
**Referenced Files in This Document**
- [multi_tower.py](file://tzrec/models/multi_tower.py)
- [multi_tower_din.py](file://tzrec/models/multi_tower_din.py)
- [rank_model.py](file://tzrec/models/rank_model.py)
- [sequence.py](file://tzrec/modules/sequence.py)
- [embedding.py](file://tzrec/modules/embedding.py)
- [multi_tower.md](file://docs/source/models/multi_tower.md)
- [din.md](file://docs/source/models/din.md)
- [multi_tower_din_taobao.config](file://examples/multi_tower_din_taobao.config)
- [seq_encoder.proto](file://tzrec/protos/seq_encoder.proto)
- [tower.proto](file://tzrec/protos/tower.proto)
</cite>

## Table of Contents

1. [Introduction](#introduction)
1. [Project Structure](#project-structure)
1. [Core Components](#core-components)
1. [Architecture Overview](#architecture-overview)
1. [Detailed Component Analysis](#detailed-component-analysis)
1. [Dependency Analysis](#dependency-analysis)
1. [Performance Considerations](#performance-considerations)
1. [Troubleshooting Guide](#troubleshooting-guide)
1. [Conclusion](#conclusion)
1. [Appendices](#appendices)

## Introduction

This document explains multi-tower and attention-based ranking models in the repository, focusing on:

- MultiTower: heterogeneous input modalities via separate embedding towers and cross-tower fusion.
- DIN (Deep Interest Network): attention over user behavior sequences to extract dynamic interests.
- Practical feature grouping strategies, attention scoring mechanics, and sequence modeling.
- Differences between static multi-tower and dynamic attention-based models.
- Implementation details and optimization tips for large-scale deployment.

## Project Structure

The relevant components span model definitions, sequence encoders, feature grouping, and configuration:

- Models: MultiTower and MultiTowerDIN
- Sequence encoders: DINEncoder, SelfAttentionEncoder, PoolingEncoder, MultiWindowDINEncoder, HSTUEncoder
- Feature grouping: EmbeddingGroup and group_total_dim
- Protos: seq_encoder.proto and tower.proto define configuration schemas
- Docs and examples: multi_tower.md, din.md, and multi_tower_din_taobao.config

```mermaid
graph TB
subgraph "Models"
MT["MultiTower<br/>multi_tower.py"]
MTD["MultiTowerDIN<br/>multi_tower_din.py"]
end
subgraph "Sequence Encoders"
DINE["DINEncoder<br/>sequence.py"]
SA["SelfAttentionEncoder<br/>sequence.py"]
POOL["PoolingEncoder<br/>sequence.py"]
MW["MultiWindowDINEncoder<br/>sequence.py"]
HSTU["HSTUEncoder<br/>sequence.py"]
end
subgraph "Feature Grouping"
EG["EmbeddingGroup<br/>embedding.py"]
end
subgraph "Configs"
SE_PROTO["seq_encoder.proto"]
TOWER_PROTO["tower.proto"]
DOC_MT["multi_tower.md"]
DOC_DIN["din.md"]
CFG_EX["multi_tower_din_taobao.config"]
end
MT --> EG
MTD --> EG
MTD --> DINE
MTD --> SA
MTD --> POOL
MTD --> MW
MTD --> HSTU
EG --> SE_PROTO
EG --> TOWER_PROTO
DOC_MT -.-> MT
DOC_DIN -.-> MTD
CFG_EX -.-> MTD
```

**Diagram sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L25-L85)
- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L26-L104)
- \[sequence.py\](file://tzrec/modules/sequence.py#L70-L134)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[seq_encoder.proto\](file://tzrec/protos/seq_encoder.proto#L6-L107)
- \[tower.proto\](file://tzrec/protos/tower.proto#L8-L27)
- \[multi_tower.md\](file://docs/source/models/multi_tower.md#L1-L72)
- \[din.md\](file://docs/source/models/din.md#L1-L89)
- \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)

**Section sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L25-L85)
- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L26-L104)
- \[sequence.py\](file://tzrec/modules/sequence.py#L70-L134)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[multi_tower.md\](file://docs/source/models/multi_tower.md#L1-L72)
- \[din.md\](file://docs/source/models/din.md#L1-L89)
- \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)

## Core Components

- MultiTower
  - Builds separate MLP towers per feature group and concatenates outputs.
  - Supports an optional final MLP before the output head.
  - Uses EmbeddingGroup to compute group_total_dim for each tower’s input size.
- MultiTowerDIN
  - Extends MultiTower by adding DIN-style sequence towers.
  - Each sequence feature group contributes a DINEncoder (or other encoders) with attention over historical items.
  - Concatenates deep towers and sequence towers, optionally followed by a final MLP.

Key implementation references:

- MultiTower initialization and predict flow: \[multi_tower.py\](file://tzrec/models/multi_tower.py#L35-L85)
- MultiTowerDIN initialization and predict flow: \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L36-L104)
- Feature grouping and group_total_dim: \[embedding.py\](file://tzrec/modules/embedding.py#L245-L270), \[rank_model.py\](file://tzrec/models/rank_model.py#L115-L132)

**Section sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L35-L85)
- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L36-L104)
- \[embedding.py\](file://tzrec/modules/embedding.py#L245-L270)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L115-L132)

## Architecture Overview

Static vs dynamic:

- Static multi-tower (MultiTower): fixed concatenation of per-group MLP features; no intra-sequence attention.
- Dynamic attention-based (MultiTowerDIN/DIN encoders): attention-weighted aggregation over historical sequences to extract dynamic interests.

```mermaid
graph TB
subgraph "Static MultiTower"
U["User Features<br/>DEEP group"]
I["Item Features<br/>DEEP group"]
UT["User Tower MLP"]
IT["Item Tower MLP"]
CAT["Concat"]
OUT["Final MLP + Linear Head"]
U --> UT
I --> IT
UT --> CAT
IT --> CAT
CAT --> OUT
end
subgraph "Dynamic Attention-Based"
SQ["Sequence Features<br/>SEQUENCE group"]
Q["Query (current)"]
ENC["DINEncoder / SelfAttentionEncoder"]
ATT["Attention Weights"]
WRAP["Weighted Sum"]
CAT2["Concat with Deep Towers"]
OUT2["Final MLP + Linear Head"]
SQ --> ENC
Q --> ENC
ENC --> ATT
ATT --> WRAP
WRAP --> CAT2
CAT2 --> OUT2
end
```

**Diagram sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L46-L85)
- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L47-L104)
- \[sequence.py\](file://tzrec/modules/sequence.py#L70-L134)
- \[sequence.py\](file://tzrec/modules/sequence.py#L226-L290)

## Detailed Component Analysis

### MultiTower

- Purpose: Separate embedding towers per feature group; simple cross-tower fusion via concatenation.
- Feature grouping:
  - EmbeddingGroup computes total embedding dimension per group via group_total_dim.
  - Each tower is an MLP whose input size equals the group’s total embedding dimension.
- Cross-tower fusion:
  - Concatenates all tower outputs along the feature dimension.
  - Optional final MLP reduces combined representation to desired scale.
- Output head:
  - Single linear layer to logits/probabilities depending on loss configuration.

```mermaid
classDiagram
class MultiTower {
+towers : ModuleDict
+final_mlp : MLP
+output_mlp : Linear
+predict(batch) Dict
}
class EmbeddingGroup {
+group_total_dim(group_name) int
+group_feature_dims(group_name) Dict
}
class MLP {
+output_dim() int
}
MultiTower --> EmbeddingGroup : "uses"
MultiTower --> MLP : "tower MLPs"
MultiTower --> MLP : "final MLP"
MultiTower --> MLP : "output head"
```

**Diagram sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L25-L85)
- \[embedding.py\](file://tzrec/modules/embedding.py#L245-L270)

**Section sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L35-L85)
- \[embedding.py\](file://tzrec/modules/embedding.py#L245-L270)

### MultiTowerDIN and DIN Encoders

- Purpose: Combine static deep features with dynamic sequence modeling via attention.
- Sequence modeling:
  - DINEncoder: computes attention over historical items using query–candidate interactions.
  - SelfAttentionEncoder: multi-head self-attention over sequences with masked attention.
  - PoolingEncoder: mean/sum pooling baseline.
  - MultiWindowDINEncoder: windowed attention across multiple time windows.
  - HSTUEncoder: advanced transduction unit with relative position/time bias.
- Attention computation:
  - DINEncoder: builds query–sequence pairs, concatenates [query, seq, query−seq, query∗seq], passes through an MLP, applies softmax over valid positions, and returns a weighted sum.
  - SelfAttentionEncoder: constructs Q/K/V from sequence embeddings, applies masked multihead attention, and averages over sequence length.

```mermaid
flowchart TD
Start(["DINEncoder.forward"]) --> GetQ["Load query and sequence"]
GetQ --> Clamp["Clamp sequence length if max_seq_length set"]
Clamp --> PadQ["Pad query to sequence_dim if needed"]
PadQ --> ExpandQ["Expand query to [B, T, C]"]
ExpandQ --> Concat["Concat [queries, sequence, queries−sequence, queries*sequence]"]
Concat --> MLP["Pass through MLP"]
MLP --> Linear["Linear to scores [B, 1, T]"]
Linear --> Transpose["Transpose to [B, T, 1]"]
Transpose --> Mask["Apply sequence_mask to padding"]
Mask --> Softmax["Softmax over time"]
Softmax --> Weighted["Multiply by sequence and sum"]
Weighted --> Return(["Return attended vector"])
```

**Diagram sources**

- \[sequence.py\](file://tzrec/modules/sequence.py#L106-L133)

**Section sources**

- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L36-L104)
- \[sequence.py\](file://tzrec/modules/sequence.py#L70-L134)
- \[sequence.py\](file://tzrec/modules/sequence.py#L226-L290)

### Sequence Encoder Protos and Configurations

- seq_encoder.proto defines message schemas for DINEncoder, SelfAttentionEncoder, PoolingEncoder, MultiWindowDINEncoder, and HSTUEncoder.
- tower.proto defines Tower and DINTower messages used in model configs.

```mermaid
classDiagram
class SeqEncoderConfig {
+oneof seq_module
}
class DINEncoder {
+string input
+MLP attn_mlp
+int32 max_seq_length
}
class SelfAttentionEncoder {
+string input
+int32 multihead_attn_dim
+int32 num_heads
+float dropout
+int32 max_seq_length
}
class PoolingEncoder {
+string input
+string pooling_type
+int32 max_seq_length
}
class MultiWindowDINEncoder {
+string input
+MLP attn_mlp
+repeated uint32 windows_len
}
class HSTUEncoder {
+string input
+int32 sequence_dim
+int32 attn_dim
+int32 linear_dim
+int32 max_seq_length
+float pos_dropout_rate
+float linear_dropout_rate
+float attn_dropout_rate
+string normalization
+string linear_activation
+string linear_config
+int32 num_heads
+int32 num_blocks
+int32 max_output_len
+int32 time_bucket_size
}
SeqEncoderConfig --> DINEncoder
SeqEncoderConfig --> SelfAttentionEncoder
SeqEncoderConfig --> PoolingEncoder
SeqEncoderConfig --> MultiWindowDINEncoder
SeqEncoderConfig --> HSTUEncoder
```

**Diagram sources**

- \[seq_encoder.proto\](file://tzrec/protos/seq_encoder.proto#L6-L107)

**Section sources**

- \[seq_encoder.proto\](file://tzrec/protos/seq_encoder.proto#L6-L107)
- \[tower.proto\](file://tzrec/protos/tower.proto#L8-L27)

### Feature Grouping Strategies and Cross-Tower Fusion

- Feature groups:
  - DEEP groups: static features (user/item/context).
  - SEQUENCE groups: historical sequences with query and sequence tensors.
- Group total dimension:
  - EmbeddingGroup.group_total_dim sums embedding dimensions per group, accounting for sequence encoders’ outputs.
- Cross-tower fusion:
  - MultiTower: concatenate per-tower MLP outputs; optional final MLP.
  - MultiTowerDIN: concatenate deep towers and sequence encoders’ outputs; optional final MLP.

Practical examples:

- Example config demonstrates DEEP and SEQUENCE feature groups and multi_tower_din configuration: \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)
- Model docs show DEEP and SEQUENCE group usage: \[din.md\](file://docs/source/models/din.md#L11-L56), \[multi_tower.md\](file://docs/source/models/multi_tower.md#L12-L51)

**Section sources**

- \[embedding.py\](file://tzrec/modules/embedding.py#L245-L270)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L115-L132)
- \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)
- \[din.md\](file://docs/source/models/din.md#L11-L56)
- \[multi_tower.md\](file://docs/source/models/multi_tower.md#L12-L51)

### Attention Scoring Functions and Multi-Head Attention

- DINEncoder scoring:
  - Concatenates query and sequence embeddings with difference and product terms, feeds through MLP, projects to scalar scores, masks padding, and normalizes with softmax.
- SelfAttentionEncoder scoring:
  - Constructs Q, K, V from sequence embeddings, applies masked multihead attention, and averages over sequence length after nan-to-num handling.
- Multi-Window DIN:
  - Computes attention per window and aggregates per-window sums normalized by effective window lengths.

```mermaid
sequenceDiagram
participant B as "Batch"
participant EG as "EmbeddingGroup"
participant D as "DINEncoder"
participant S as "SelfAttentionEncoder"
B->>EG : "sparse/dense features"
EG-->>B : "grouped embeddings {group : tensor}"
B->>D : "query, sequence, sequence_length"
D-->>B : "attended vector"
B->>S : "sequence, sequence_length"
S-->>B : "attended vector"
```

**Diagram sources**

- \[sequence.py\](file://tzrec/modules/sequence.py#L106-L133)
- \[sequence.py\](file://tzrec/modules/sequence.py#L273-L290)

**Section sources**

- \[sequence.py\](file://tzrec/modules/sequence.py#L70-L134)
- \[sequence.py\](file://tzrec/modules/sequence.py#L226-L290)

### Implementation Details and Configuration

- MultiTower config highlights:
  - feature_groups define DEEP groups.
  - multi_tower.towers map each DEEP group to an MLP.
  - optional multi_tower.final MLP before output.
  - Reference: \[multi_tower.md\](file://docs/source/models/multi_tower.md#L12-L51)
- MultiTowerDIN config highlights:
  - feature_groups include DEEP and SEQUENCE groups.
  - multi_tower_din.towers for DEEP groups.
  - multi_tower_din.din_towers specify attn_mlp for attention scoring.
  - optional multi_tower_din.final MLP.
  - Reference: \[din.md\](file://docs/source/models/din.md#L11-L56)
- Example config:
  - \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)

**Section sources**

- \[multi_tower.md\](file://docs/source/models/multi_tower.md#L12-L51)
- \[din.md\](file://docs/source/models/din.md#L11-L56)
- \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)

## Dependency Analysis

- MultiTower depends on:
  - EmbeddingGroup for per-group embedding dimensions.
  - MLP towers and optional final MLP.
- MultiTowerDIN additionally depends on:
  - Sequence encoders (DINEncoder, SelfAttentionEncoder, etc.) registered via create_seq_encoder.
  - Protobuf schemas for encoder configuration.

```mermaid
graph LR
MT["MultiTower"] --> EG["EmbeddingGroup"]
MT --> MLP1["MLP (towers)"]
MT --> MLP2["MLP (final)"]
MTD["MultiTowerDIN"] --> EG
MTD --> MLP1
MTD --> MLP2
MTD --> DINE["DINEncoder"]
MTD --> SA["SelfAttentionEncoder"]
MTD --> POOL["PoolingEncoder"]
MTD --> MW["MultiWindowDINEncoder"]
MTD --> HSTU["HSTUEncoder"]
```

**Diagram sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L46-L63)
- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L47-L80)
- \[sequence.py\](file://tzrec/modules/sequence.py#L580-L604)

**Section sources**

- \[multi_tower.py\](file://tzrec/models/multi_tower.py#L46-L63)
- \[multi_tower_din.py\](file://tzrec/models/multi_tower_din.py#L47-L80)
- \[sequence.py\](file://tzrec/modules/sequence.py#L580-L604)

## Performance Considerations

- Sequence length control:
  - Use max_seq_length in encoders to cap memory and compute.
- Multi-head attention:
  - Ensure multihead_attn_dim is divisible by num_heads; precompute Q/K/V to reduce overhead.
- Attention masking:
  - Apply attention masks to avoid attending to padded positions; handle NaNs robustly.
- Feature grouping:
  - Keep DEEP and SEQUENCE groups aligned with model capacity; avoid overly large embedding dimensions unless justified.
- Final fusion:
  - Use a smaller final MLP to reduce parameters and speed up inference.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide

- Sequence length mismatch:
  - Ensure all SEQUENCE feature groups have equal sequence lengths when using certain encoders.
  - Reference: \[din.md\](file://docs/source/models/din.md#L74-L76)
- Query and sequence dimension mismatch:
  - DINEncoder pads query to match sequence dimension; ensure query_dim ≤ sequence_dim.
  - Reference: \[sequence.py\](file://tzrec/modules/sequence.py#L93-L94)
- Attention mask construction:
  - Verify attention masks align with batch size and sequence length; confirm broadcasting semantics.
  - Reference: \[sequence.py\](file://tzrec/modules/sequence.py#L39-L51), \[sequence.py\](file://tzrec/modules/sequence.py#L283-L285)
- NaN handling:
  - SelfAttentionEncoder replaces NaNs with zeros; check inputs for numerical stability.
  - Reference: \[sequence.py\](file://tzrec/modules/sequence.py#L287-L287)

**Section sources**

- \[din.md\](file://docs/source/models/din.md#L74-L76)
- \[sequence.py\](file://tzrec/modules/sequence.py#L93-L94)
- \[sequence.py\](file://tzrec/modules/sequence.py#L39-L51)
- \[sequence.py\](file://tzrec/modules/sequence.py#L283-L285)
- \[sequence.py\](file://tzrec/modules/sequence.py#L287-L287)

## Conclusion

- MultiTower offers a scalable, static fusion of heterogeneous features via separate embedding towers.
- MultiTowerDIN augments this with attention over sequences, enabling dynamic interest modeling.
- Proper feature grouping, attention scoring, and sequence modeling choices are crucial for accuracy and performance.
- The repository provides flexible protobuf-driven configurations and modular encoders to tailor models to production needs.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Practical Examples and References

- MultiTower config example: \[multi_tower.md\](file://docs/source/models/multi_tower.md#L12-L51)
- MultiTowerDIN config example: \[din.md\](file://docs/source/models/din.md#L11-L56), \[multi_tower_din_taobao.config\](file://examples/multi_tower_din_taobao.config#L189-L243)

[No sources needed since this section lists existing references]
