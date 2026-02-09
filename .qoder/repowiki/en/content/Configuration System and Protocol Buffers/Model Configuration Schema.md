# Model Configuration Schema

<cite>
**Referenced Files in This Document**
- [model.proto](file://tzrec/protos/model.proto)
- [tower.proto](file://tzrec/protos/tower.proto)
- [module.proto](file://tzrec/protos/module.proto)
- [rank_model.proto](file://tzrec/protos/models/rank_model.proto)
- [multi_task_rank.proto](file://tzrec/protos/models/multi_task_rank.proto)
- [match_model.proto](file://tzrec/protos/models/match_model.proto)
- [general_rank_model.proto](file://tzrec/protos/models/general_rank_model.proto)
- [config_util.py](file://tzrec/utils/config_util.py)
- [dssm_taobao.config](file://examples/dssm_taobao.config)
- [deepfm_criteo.config](file://examples/deepfm_criteo.config)
- [multi_tower_taobao.config](file://examples/multi_tower_taobao.config)
- [dbmtl_taobao.config](file://examples/dbmtl_taobao.config)
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

This document describes TorchEasyRec’s model configuration schema. It explains how to define models via the ModelConfig message, including model type selection, feature groups, towers, and task-specific parameters. It also documents the base model architecture configuration, module definitions, layer specifications, and the configuration framework used to parse and edit configurations. Practical examples illustrate DSSM, DeepFM, MultiTower, and DBMTL configurations. Guidance is provided for validation, initialization, and checkpoint loading, along with best practices and troubleshooting tips.

## Project Structure

TorchEasyRec organizes model configuration definitions in Protocol Buffers (.proto) files under tzrec/protos and model-specific subdirectories. Example configurations live under examples/. The configuration loader resides in tzrec/utils/config_util.py.

```mermaid
graph TB
subgraph "Protobuf Definitions"
MP["tzrec/protos/model.proto"]
TP["tzrec/protos/tower.proto"]
UP["tzrec/protos/module.proto"]
RMP["tzrec/protos/models/rank_model.proto"]
MTRP["tzrec/protos/models/multi_task_rank.proto"]
MMP["tzrec/protos/models/match_model.proto"]
GRP["tzrec/protos/models/general_rank_model.proto"]
end
subgraph "Examples"
DSSM_CFG["examples/dssm_taobao.config"]
DF_CFG["examples/deepfm_criteo.config"]
MT_CFG["examples/multi_tower_taobao.config"]
DBMTL_CFG["examples/dbmtl_taobao.config"]
end
subgraph "Runtime Utilities"
CU["tzrec/utils/config_util.py"]
end
MP --> RMP
MP --> MTRP
MP --> MMP
MP --> GRP
TP --> RMP
TP --> MTRP
TP --> MMP
UP --> RMP
UP --> MTRP
UP --> MMP
DSSM_CFG --> CU
DF_CFG --> CU
MT_CFG --> CU
DBMTL_CFG --> CU
```

**Diagram sources**

- \[model.proto\](file://tzrec/protos/model.proto#L1-L90)
- \[tower.proto\](file://tzrec/protos/tower.proto#L1-L198)
- \[module.proto\](file://tzrec/protos/module.proto#L1-L287)
- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L1-L80)
- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L1-L75)
- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L1-L81)
- \[general_rank_model.proto\](file://tzrec/protos/models/general_rank_model.proto#L1-L16)
- \[dssm_taobao.config\](file://examples/dssm_taobao.config#L1-L267)
- \[deepfm_criteo.config\](file://examples/deepfm_criteo.config#L1-L397)
- \[multi_tower_taobao.config\](file://examples/multi_tower_taobao.config#L1-L207)
- \[dbmtl_taobao.config\](file://examples/dbmtl_taobao.config#L1-L224)
- \[config_util.py\](file://tzrec/utils/config_util.py#L1-L299)

**Section sources**

- \[model.proto\](file://tzrec/protos/model.proto#L1-L90)
- \[config_util.py\](file://tzrec/utils/config_util.py#L1-L299)

## Core Components

This section outlines the primary configuration constructs used to define models and tasks.

- ModelConfig

  - Defines feature_groups, oneof model selection, number of classes, losses, metrics, training metrics, variational dropout, kernel selection, and Pareto loss weighting flag.
  - Model selection supports many architectures (DLRM, DeepFM, MultiTower, MultiTowerDIN, MaskNet, WideAndDeep, DCN variants, xDeepFM, WuKong, SimpleMultiTask, MMoE, DBMTL, PLE, DC2VR, DlrmHSTU, DSSM/DSSMV2/HSTUMatch/MIND/TDM/RocketLaunching).

- FeatureGroupConfig and FeatureGroupType

  - Groups features by semantic roles: DEEP, WIDE, SEQUENCE, JAGGED_SEQUENCE.
  - Supports nested sequence_groups and sequence_encoders.

- Tower and TaskTower

  - Tower: input feature group and an MLP for single-tower models.
  - TaskTower: per-task configuration with label_name, metrics, losses, optional MLP, class count, weights, and task-space indicators.

- Module definitions (MLP, Cross/CrossV2, CIN, MaskNetModule, B2ICapsule, STU/HSTU, etc.)

  - MLP: hidden_units, dropout_ratio per layer, activation, batch/layer norm toggles, bias, and LN.
  - Cross/CrossV2: cross network depth.
  - CIN: Compose-Input-Noise network sizes.
  - MaskNetModule: number of blocks, block specs, top MLP, and parallel vs serial mode.
  - B2ICapsule: capsule routing parameters.
  - STU/HSTU: attention and normalization options for sequence modeling.

- Match and Multi-Task models

  - DSSM/DSSMV2/HSTUMatch/MIND/TDM: similarity, temperature, in-batch negatives, and output dimensions.
  - DBMTL/MMoE/PLE/DC2VR: shared bottlenecks, expert/gate modules, task towers, and relations.

**Section sources**

- \[model.proto\](file://tzrec/protos/model.proto#L13-L89)
- \[tower.proto\](file://tzrec/protos/tower.proto#L8-L198)
- \[module.proto\](file://tzrec/protos/module.proto#L4-L287)
- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L8-L80)
- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L7-L75)
- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L9-L81)
- \[general_rank_model.proto\](file://tzrec/protos/models/general_rank_model.proto#L8-L16)

## Architecture Overview

The configuration schema composes feature groups, model-specific blocks, and task towers into a unified ModelConfig. The runtime loads configurations from text or JSON, validates fields, and constructs model graphs accordingly.

```mermaid
sequenceDiagram
participant User as "User Config"
participant Loader as "config_util.load_pipeline_config()"
participant Parser as "protobuf parser"
participant ModelCfg as "ModelConfig"
participant Builder as "Model Builder"
User->>Loader : Provide .config path
Loader->>Parser : Parse text/json to EasyRecConfig
Parser-->>Loader : Parsed pipeline_pb2.EasyRecConfig
Loader-->>ModelCfg : Extract model_config
ModelCfg->>Builder : Build model graph (features + towers + losses)
Builder-->>User : Executable model
```

**Diagram sources**

- \[config_util.py\](file://tzrec/utils/config_util.py#L25-L48)
- \[model.proto\](file://tzrec/protos/model.proto#L40-L89)

## Detailed Component Analysis

### ModelConfig and Oneof Model Selection

ModelConfig aggregates feature groups and selects one of many supported models via a oneof. Each model variant defines its own parameters (e.g., MLP stacks, cross layers, towers, task towers).

```mermaid
classDiagram
class ModelConfig {
+FeatureGroupConfig[] feature_groups
+oneof model
+uint32 num_class
+LossConfig[] losses
+MetricConfig[] metrics
+TrainMetricConfig[] train_metrics
+VariationalDropout variational_dropout
+Kernel kernel
+bool use_pareto_loss_weight
}
class RankModels {
}
class MultiTaskModels {
}
class MatchModels {
}
class GeneralRankModels {
}
ModelConfig --> RankModels : "oneof"
ModelConfig --> MultiTaskModels : "oneof"
ModelConfig --> MatchModels : "oneof"
ModelConfig --> GeneralRankModels : "oneof"
```

**Diagram sources**

- \[model.proto\](file://tzrec/protos/model.proto#L40-L89)
- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L1-L80)
- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L1-L75)
- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L1-L81)
- \[general_rank_model.proto\](file://tzrec/protos/models/general_rank_model.proto#L1-L16)

**Section sources**

- \[model.proto\](file://tzrec/protos/model.proto#L40-L89)

### Feature Groups and Sequence Encoders

FeatureGroupConfig binds feature names to semantic groups and optionally applies sequence encoders. FeatureGroupType distinguishes DEEP/WIDE/SEQUENCE/JAGGED_SEQUENCE.

```mermaid
flowchart TD
Start(["Feature Groups"]) --> Define["Define group_name and feature_names"]
Define --> TypeSel{"Group Type?"}
TypeSel --> |DEEP/WIDE| Basic["Basic feature aggregation"]
TypeSel --> |SEQUENCE/JAGGED_SEQUENCE| SeqEnc["Apply sequence encoders"]
SeqEnc --> Pool["Pooling or attention"]
Basic --> Next["Proceed to model blocks"]
Pool --> Next
```

**Diagram sources**

- \[model.proto\](file://tzrec/protos/model.proto#L26-L32)

**Section sources**

- \[model.proto\](file://tzrec/protos/model.proto#L13-L32)

### Tower and Task Tower Specifications

Towers feed feature groups into MLPs; TaskTower adds per-task supervision and optional MLPs.

```mermaid
classDiagram
class Tower {
+string input
+MLP mlp
}
class DINTower {
+string input
+MLP attn_mlp
}
class TaskTower {
+string tower_name
+string label_name
+MetricConfig[] metrics
+TrainMetricConfig[] train_metrics
+LossConfig[] losses
+uint32 num_class
+MLP mlp
+float weight
+string sample_weight_name
+string task_space_indicator_label
+float in_task_space_weight
+float out_task_space_weight
+float pareto_min_loss_weight
}
Tower <.. TaskTower : "shared MLP pattern"
```

**Diagram sources**

- \[tower.proto\](file://tzrec/protos/tower.proto#L8-L56)

**Section sources**

- \[tower.proto\](file://tzrec/protos/tower.proto#L8-L56)

### Base Model Architectures and Modules

Common building blocks are defined in module.proto and consumed by model-specific messages.

```mermaid
classDiagram
class MLP {
+uint32[] hidden_units
+float[] dropout_ratio
+string activation
+bool use_bn
+bool bias
+bool use_ln
}
class Cross {
+uint32 cross_num
}
class CrossV2 {
+uint32 cross_num
+uint32 low_rank
}
class CIN {
+uint32[] cin_layer_size
}
class MaskNetModule {
+uint32 n_mask_blocks
+MaskBlock mask_block
+MLP top_mlp
+bool use_parallel
}
class B2ICapsule {
+uint32 max_k
+uint32 max_seq_len
+uint32 high_dim
+uint32 num_iters
+float routing_logits_scale
+float routing_logits_stddev
+float squash_pow
+bool const_caps_num
+string routing_init_method
}
class STU {
+uint32 embedding_dim
+uint32 num_heads
+uint32 hidden_dim
+uint32 attention_dim
+float output_dropout_ratio
+uint32 max_attn_len
+float attn_alpha
+bool use_group_norm
+bool recompute_normed_x
+bool recompute_uvqk
+bool recompute_y
+bool sort_by_length
+uint32 contextual_seq_len
}
MLP <.. Cross
MLP <.. CrossV2
MLP <.. CIN
MLP <.. MaskNetModule
MLP <.. B2ICapsule
MLP <.. STU
```

**Diagram sources**

- \[module.proto\](file://tzrec/protos/module.proto#L4-L287)

**Section sources**

- \[module.proto\](file://tzrec/protos/module.proto#L4-L287)

### Rank Models (Wide & Deep, DeepFM, DLRM, DCN, xDeepFM, WuKong)

These models define how features are transformed and combined.

```mermaid
classDiagram
class WideAndDeep {
+MLP deep
+MLP final
+uint32 wide_embedding_dim
+string wide_init_fn
}
class DeepFM {
+MLP deep
+MLP final
+uint32 wide_embedding_dim
+string wide_init_fn
}
class DLRM {
+MLP dense_mlp
+bool arch_with_sparse
+MLP final
}
class DCNV1 {
+Cross cross
+MLP deep
+MLP final
}
class DCNV2 {
+MLP backbone
+CrossV2 cross
+MLP deep
+MLP final
}
class xDeepFM {
+CIN cin
+MLP deep
+MLP final
+uint32 wide_embedding_dim
+string wide_init_fn
}
class WuKong {
+MLP dense_mlp
+WuKongLayer[] wukong_layers
+MLP final
}
WideAndDeep <.. DeepFM
DLRM <.. DCNV1
DLRM <.. DCNV2
xDeepFM <.. WuKong
```

**Diagram sources**

- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L8-L80)

**Section sources**

- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L8-L80)

### Multi-Task Models (SimpleMultiTask, MMoE, DBMTL, PLE, DC2VR, DlrmHSTU)

These models introduce task towers and expert/gate mechanisms.

```mermaid
classDiagram
class SimpleMultiTask {
+TaskTower[] task_towers
}
class MMoE {
+MLP expert_mlp
+MLP gate_mlp
+uint32 num_expert
+TaskTower[] task_towers
}
class DBMTL {
+MaskNetModule mask_net
+MLP bottom_mlp
+MLP expert_mlp
+MLP gate_mlp
+uint32 num_expert
+BayesTaskTower[] task_towers
}
class PLE {
+ExtractionNetwork[] extraction_networks
+TaskTower[] task_towers
}
class DC2VR {
+MLP bottom_mlp
+MLP expert_mlp
+MLP gate_mlp
+uint32 num_expert
+InterventionTaskTower[] task_towers
}
class DlrmHSTU {
+HSTU hstu
+FusionMTLTower fusion_mtl_tower
+uint32 max_seq_len
+uint32 item_embedding_hidden_dim
+bool enable_global_average_loss
+bool sequence_timestamp_is_ascending
}
```

**Diagram sources**

- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L7-L75)

**Section sources**

- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L7-L75)

### Matching and Ranking Models (DSSM, DSSMV2, HSTUMatch, MIND, TDM)

These models define user/item towers, similarity, and optional in-batch negatives.

```mermaid
classDiagram
class DSSM {
+Tower user_tower
+Tower item_tower
+int32 output_dim
+Similarity similarity
+float temperature
+bool in_batch_negative
}
class DSSMV2 {
+Tower user_tower
+Tower item_tower
+int32 output_dim
+Similarity similarity
+float temperature
+bool in_batch_negative
}
class HSTUMatch {
+HSTUMatchTower hstu_tower
+int32 output_dim
+Similarity similarity
+float temperature
+bool in_batch_negative
}
class MIND {
+MINDUserTower user_tower
+Tower item_tower
+float simi_pow
+Similarity similarity
+bool in_batch_negative
+float temperature
+int32 output_dim
}
class TDM {
+MultiWindowDINTower multiwindow_din
+MLP final
}
```

**Diagram sources**

- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L9-L81)

**Section sources**

- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L9-L81)

### General-Ranking Model (RocketLaunching)

A specialized model combining shared and boosting components.

```mermaid
classDiagram
class RocketLaunching {
+MLP share_mlp
+MLP booster_mlp
+MLP light_mlp
+bool feature_based_distillation
+Similarity feature_distillation_function
}
```

**Diagram sources**

- \[general_rank_model.proto\](file://tzrec/protos/models/general_rank_model.proto#L8-L16)

**Section sources**

- \[general_rank_model.proto\](file://tzrec/protos/models/general_rank_model.proto#L8-L16)

### Example Configurations

#### DSSM Example

- Feature groups: user and item feature sets plus a sequence group with a pooling encoder.
- Model: DSSM with user and item towers and an output dimension.
- Metrics and losses: recall@K and softmax cross entropy.

**Section sources**

- \[dssm_taobao.config\](file://examples/dssm_taobao.config#L201-L266)

#### DeepFM Example

- Feature groups: separate WIDE and DEEP groups, plus a DEEP numeric group.
- Model: DeepFM with deep and final MLPs.
- Metrics and losses: AUC and binary cross entropy.

**Section sources**

- \[deepfm_criteo.config\](file://examples/deepfm_criteo.config#L278-L396)

#### MultiTower Example

- Feature groups: user and item groups.
- Model: MultiTower with two towers and a final MLP.
- Metrics and losses: AUC and binary cross entropy.

**Section sources**

- \[multi_tower_taobao.config\](file://examples/multi_tower_taobao.config#L158-L206)

#### DBMTL Example

- Single feature group aggregating all features.
- Model: DBMTL with a shared bottom MLP, optional expert/gate modules, and two task towers (CTR and CVR) with relations.

**Section sources**

- \[dbmtl_taobao.config\](file://examples/dbmtl_taobao.config#L159-L223)

## Dependency Analysis

ModelConfig depends on model-specific protobuf definitions. Feature groups depend on module definitions for MLPs and sequence encoders. Task towers depend on loss and metric configurations.

```mermaid
graph LR
ModelProto["model.proto"] --> RankProto["models/rank_model.proto"]
ModelProto --> MTaskProto["models/multi_task_rank.proto"]
ModelProto --> MatchProto["models/match_model.proto"]
ModelProto --> GenProto["models/general_rank_model.proto"]
TowerProto["tower.proto"] --> RankProto
TowerProto --> MTaskProto
TowerProto --> MatchProto
ModuleProto["module.proto"] --> RankProto
ModuleProto --> MTaskProto
ModuleProto --> MatchProto
```

**Diagram sources**

- \[model.proto\](file://tzrec/protos/model.proto#L1-L12)
- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L1-L80)
- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L1-L75)
- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L1-L81)
- \[general_rank_model.proto\](file://tzrec/protos/models/general_rank_model.proto#L1-L16)
- \[tower.proto\](file://tzrec/protos/tower.proto#L1-L198)
- \[module.proto\](file://tzrec/protos/module.proto#L1-L287)

**Section sources**

- \[model.proto\](file://tzrec/protos/model.proto#L1-L12)

## Performance Considerations

- Kernel selection: choose PyTorch or CUDA kernels depending on hardware availability.
- Variational dropout: regularizes embeddings to reduce overfitting.
- Batch size and workers: tune data_config.batch_size and num_workers for throughput.
- Sequence encoders: pooling vs attention impacts memory and compute trade-offs.
- MLP depth and width: larger networks increase capacity but require more resources.
- Optimizer settings: configure sparse and dense optimizers and learning rate schedules in train_config.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide

- Unknown fields: load_pipeline_config supports skipping unknown fields when allow_unknown_field is enabled.
- Field type mismatches: config editing enforces type conversion and supports bracketed selectors for lists and dicts.
- Feature group mode compatibility: legacy fg_encoded is mapped to fg_mode automatically.
- Saving configs: use save_message to write proto messages to text format.

**Section sources**

- \[config_util.py\](file://tzrec/utils/config_util.py#L25-L48)
- \[config_util.py\](file://tzrec/utils/config_util.py#L144-L299)
- \[config_util.py\](file://tzrec/utils/config_util.py#L78-L92)
- \[config_util.py\](file://tzrec/utils/config_util.py#L51-L64)

## Conclusion

TorchEasyRec’s configuration schema offers a structured way to define models, feature groups, towers, and tasks. Protobuf-based definitions ensure strong typing and extensibility. Examples demonstrate practical configurations for popular architectures. Use the provided utilities to load, edit, and validate configurations, and follow best practices for performance and reliability.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Model Registration and Extension

- New models are integrated by adding entries to the ModelConfig.oneof and defining the corresponding message in a dedicated .proto under models/.
- Feature groups and sequence encoders remain consistent across models via shared definitions in model.proto and module.proto.

**Section sources**

- \[model.proto\](file://tzrec/protos/model.proto#L44-L72)

### Validation and Initialization Checklist

- Verify feature_groups bind existing feature names.
- Confirm model oneof matches the intended architecture.
- Ensure MLP hidden_units and dropout_ratio lengths align with expectations.
- Set num_class appropriately for multi-class tasks.
- Configure losses and metrics consistently with label semantics.
- Initialize embeddings and modules via model-specific parameters (e.g., wide_init_fn).

**Section sources**

- \[rank_model.proto\](file://tzrec/protos/models/rank_model.proto#L8-L80)
- \[match_model.proto\](file://tzrec/protos/models/match_model.proto#L9-L81)
- \[multi_task_rank.proto\](file://tzrec/protos/models/multi_task_rank.proto#L7-L75)

### Hyperparameters and Optimization Settings

- Learning rates and optimizers are configured outside ModelConfig (e.g., in train_config.sparse_optimizer/dense_optimizer).
- Tune batch size, epochs, and data workers for convergence and throughput.
- Use Pareto loss weighting and variational dropout judiciously for robustness.

**Section sources**

- \[dssm_taobao.config\](file://examples/dssm_taobao.config#L4-L20)
- \[deepfm_criteo.config\](file://examples/deepfm_criteo.config#L4-L23)
- \[multi_tower_taobao.config\](file://examples/multi_tower_taobao.config#L4-L22)
- \[dbmtl_taobao.config\](file://examples/dbmtl_taobao.config#L4-L22)

### Checkpoint Loading and Export

- Use export utilities to serialize trained models for serving.
- Ensure feature group and model signatures remain consistent across training and inference.

[No sources needed since this section provides general guidance]
