# Rank Model Base Class

<cite>
**Referenced Files in This Document**
- [rank_model.py](file://tzrec/models/rank_model.py)
- [model.py](file://tzrec/models/model.py)
- [embedding.py](file://tzrec/modules/embedding.py)
- [variational_dropout.py](file://tzrec/modules/variational_dropout.py)
- [focal_loss.py](file://tzrec/loss/focal_loss.py)
- [jrc_loss.py](file://tzrec/loss/jrc_loss.py)
- [train_metric_wrapper.py](file://tzrec/metrics/train_metric_wrapper.py)
- [deepfm_criteo.config](file://examples/deepfm_criteo.config)
- [dssm_taobao.config](file://examples/dssm_taobao.config)
- [model_pb2.py](file://tzrec/protos/model_pb2.py)
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

This document explains the RankModel base class that underpins all ranking models in TorchEasyRec. It covers the architectural design, inheritance from BaseModel, and how RankModel differs from candidate generation models. We describe the core methods for input initialization, building inputs, loss initialization, and metric initialization. We also detail embedding group construction, variational dropout integration, supported loss functions (binary cross entropy, softmax cross entropy, focal loss, JRC loss, L2 loss), prediction output formatting, and the metric initialization system. Finally, we provide configuration examples and explain RankModel’s place in the broader TorchEasyRec framework.

## Project Structure

RankModel resides in the models package and builds upon the generic BaseModel. It integrates with:

- EmbeddingGroup for efficient feature embedding per group
- VariationalDropout for regularized feature selection
- Loss modules (binary cross entropy, softmax cross entropy, focal loss, JRC loss, L2)
- Metric modules (AUROC, accuracy, MAE, MSE, grouped AUC/XAUC, and train-time decayed metrics)

```mermaid
graph TB
subgraph "Models"
RM["RankModel<br/>(rank_model.py)"]
BM["BaseModel<br/>(model.py)"]
end
subgraph "Modules"
EG["EmbeddingGroup<br/>(embedding.py)"]
VD["VariationalDropout<br/>(variational_dropout.py)"]
end
subgraph "Loss & Metrics"
BCE["BCEWithLogitsLoss"]
SCEL["CrossEntropyLoss"]
FL["BinaryFocalLoss<br/>(focal_loss.py)"]
JRC["JRCLoss<br/>(jrc_loss.py)"]
MSE["MSELoss"]
TMW["TrainMetricWrapper<br/>(train_metric_wrapper.py)"]
end
RM --> BM
RM --> EG
RM --> VD
RM --> BCE
RM --> SCEL
RM --> FL
RM --> JRC
RM --> MSE
RM --> TMW
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L56-L513)
- \[model.py\](file://tzrec/models/model.py#L39-L423)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[variational_dropout.py\](file://tzrec/modules/variational_dropout.py#L38-L119)
- \[focal_loss.py\](file://tzrec/loss/focal_loss.py#L18-L73)
- \[jrc_loss.py\](file://tzrec/loss/jrc_loss.py#L29-L118)
- \[train_metric_wrapper.py\](file://tzrec/metrics/train_metric_wrapper.py#L20-L63)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L56-L513)
- \[model.py\](file://tzrec/models/model.py#L39-L423)

## Core Components

- RankModel extends BaseModel and adds ranking-specific behaviors:
  - EmbeddingGroup initialization and per-feature-group variational dropout
  - Prediction output formatting tailored to loss types
  - Multi-loss support and metric initialization
- BaseModel defines the generic model interface and shared utilities (loss/metric registries, train/predict wrappers).

Key responsibilities:

- init_input(): Build EmbeddingGroup and optional per-group VariationalDropout
- build_input(): Apply embeddings and optional variational dropout to feature groups
- init_loss(): Register loss modules based on configuration
- loss(): Compute loss with optional sample weights and variational dropout penalties
- init_metric(): Initialize evaluation and train-time metrics
- update_metric()/update_train_metric(): Update metric states during training/evaluation

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L84-L132)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L182-L213)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L261-L284)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L286-L366)
- \[model.py\](file://tzrec/models/model.py#L39-L138)

## Architecture Overview

RankModel sits atop BaseModel and orchestrates feature embedding, optional regularization via variational dropout, and loss/metric computation. The pipeline is:

- Input parsing → EmbeddingGroup → optional VariationalDropout → model head → predictions → loss/metrics

```mermaid
sequenceDiagram
participant DS as "Data Loader"
participant RM as "RankModel"
participant EG as "EmbeddingGroup"
participant VD as "VariationalDropout"
participant HEAD as "Model Head"
participant MET as "Metrics"
DS->>RM : "Batch"
RM->>EG : "Forward(batch)"
EG-->>RM : "feature_dict"
RM->>VD : "Apply per-group dropout (optional)"
VD-->>RM : "noisy_features + penalty"
RM->>HEAD : "Forward(feature_dict)"
HEAD-->>RM : "logits/probs/y"
RM->>RM : "Format predictions by loss type"
RM->>MET : "Update metrics"
RM-->>DS : "predictions, losses"
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L115-L132)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L134-L180)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L261-L284)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[variational_dropout.py\](file://tzrec/modules/variational_dropout.py#L107-L119)

## Detailed Component Analysis

### RankModel Class Design

- Inherits from BaseModel and initializes internal state (number of classes, label name, optional sample weights).
- Manages:
  - EmbeddingGroup for grouped feature embeddings
  - ModuleDict of per-feature-group VariationalDropout modules
  - Loss and metric registries

```mermaid
classDiagram
class BaseModel {
+predict(batch)
+init_loss()
+loss(predictions, batch)
+init_metric()
+update_metric(...)
+compute_metric()
+compute_train_metric()
}
class RankModel {
-_num_class : int
-_label_name : str
-_sample_weight_name : str?
-_loss_collection : OrderedDict
+init_input()
+build_input(batch)
+init_loss()
+loss(predictions, batch)
+init_metric()
+update_metric(...)
+update_train_metric(...)
}
class EmbeddingGroup {
+__call__(batch)
+group_feature_dims(group_name)
}
class VariationalDropout {
+forward(feature)
}
RankModel --|> BaseModel
RankModel --> EmbeddingGroup : "uses"
RankModel --> VariationalDropout : "optional"
```

**Diagram sources**

- \[model.py\](file://tzrec/models/model.py#L39-L138)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L56-L132)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[variational_dropout.py\](file://tzrec/modules/variational_dropout.py#L38-L119)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L56-L132)
- \[model.py\](file://tzrec/models/model.py#L39-L138)

### Input Initialization and Building

- init_input():
  - Creates EmbeddingGroup from features and feature_groups
  - Optionally creates per-feature-group VariationalDropout modules for non-sequential groups
- build_input():
  - Runs batch through EmbeddingGroup
  - Applies variational dropout to each group and collects per-group penalties into loss collection

```mermaid
flowchart TD
Start(["init_input"]) --> CreateEmb["Create EmbeddingGroup"]
CreateEmb --> CheckVD{"Has Variational Dropout Config?"}
CheckVD --> |No| EndInit["Done"]
CheckVD --> |Yes| LoopFG["For each feature group"]
LoopFG --> IsSeq{"Group Type == SEQUENCE?"}
IsSeq --> |Yes| NextFG["Skip"]
IsSeq --> |No| HasDims{"Has multiple feature dims?"}
HasDims --> |No| NextFG
HasDims --> |Yes| AddVD["Create VariationalDropout for group"]
AddVD --> NextFG
NextFG --> LoopFG
LoopFG --> EndInit
BuildStart(["build_input"]) --> EmbOut["EmbeddingGroup(batch)"]
EmbOut --> HasVD{"Has VariationalDropout modules?"}
HasVD --> |No| ReturnEmb["Return feature_dict"]
HasVD --> |Yes| LoopVD["For each group"]
LoopVD --> ApplyVD["Apply VariationalDropout"]
ApplyVD --> UpdateLoss["Append penalty to _loss_collection"]
UpdateLoss --> LoopVD
LoopVD --> ReturnEmb
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L84-L132)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[variational_dropout.py\](file://tzrec/modules/variational_dropout.py#L107-L119)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L84-L132)

### Loss Initialization and Computation

Supported losses:

- Binary cross entropy (BCEWithLogitsLoss)
- Softmax cross entropy (CrossEntropyLoss) with label smoothing
- Binary focal loss (BinaryFocalLoss)
- JRC loss (JRCLoss) for session-based ranking
- L2 loss (MSELoss)

Initialization:

- init_loss() iterates configured losses and registers appropriate loss modules
- Reduction defaults to “none” when sample weights are present, otherwise “mean”

Computation:

- loss() computes per-loss values and optionally applies normalized sample weights
- For JRC loss, extracts session ids from batch sparse features and passes to loss module
- Variational dropout penalties are appended to losses from \_loss_collection

```mermaid
flowchart TD
InitStart(["init_loss"]) --> ForEach["For each LossConfig"]
ForEach --> Type{"WhichOneof('loss')"}
Type --> |binary_cross_entropy| AddBCE["Register BCEWithLogitsLoss"]
Type --> |softmax_cross_entropy| AddSCE["Register CrossEntropyLoss(label_smoothing)"]
Type --> |binary_focal_loss| AddFL["Register BinaryFocalLoss(gamma,alpha)"]
Type --> |jrc_loss| AddJRC["Register JRCLoss(alpha)"]
Type --> |l2_loss| AddMSE["Register MSELoss"]
AddBCE --> NextCfg
AddSCE --> NextCfg
AddFL --> NextCfg
AddJRC --> NextCfg
AddMSE --> NextCfg
NextCfg --> ForEach
LossStart(["loss"]) --> SW{"Sample weights?"}
SW --> |Yes| NormSW["Normalize weights"]
SW --> |No| NoSW["No weights"]
NormSW --> LoopLC["For each LossConfig"]
NoSW --> LoopLC
LoopLC --> Comp["Compute per-loss"]
Comp --> JRC{"Is JRC loss?"}
JRC --> |Yes| GetSID["Extract session ids from batch"]
GetSID --> ApplyJRC["Call JRCLoss(logits, labels, session_ids)"]
JRC --> |No| ApplyOther["Call registered loss module"]
ApplyJRC --> Acc["Accumulate losses"]
ApplyOther --> Acc
Acc --> AddVD["Add variational dropout penalties"]
AddVD --> ReturnLoss["Return losses"]
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L182-L213)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L261-L284)
- \[focal_loss.py\](file://tzrec/loss/focal_loss.py#L18-L73)
- \[jrc_loss.py\](file://tzrec/loss/jrc_loss.py#L29-L118)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L182-L213)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L261-L284)

### Prediction Output Formatting

RankModel converts raw model outputs into standardized prediction tensors depending on the configured loss:

- Binary/ focal loss: logits and probabilities
- Softmax cross entropy: logits, probabilities, and class-1 probability for two-class case
- JRC loss: logits, probabilities, and class-1 probability
- L2 loss: predicted y

```mermaid
flowchart TD
OutStart(["_output_to_prediction_impl"]) --> Which{"WhichOneof('loss')"}
Which --> |binary_cross_entropy| Bin["squeeze logits<br/>probs = sigmoid(logits)"]
Which --> |binary_focal_loss| Focal["squeeze logits<br/>probs = sigmoid(logits)"]
Which --> |softmax_cross_entropy| Soft["probs = softmax(logits)<br/>logits<br/>if num_class==2: probs1"]
Which --> |jrc_loss| JRCO["probs = softmax(logits)<br/>logits<br/>probs1"]
Which --> |l2_loss| L2O["y = squeeze(logits)"]
Bin --> ReturnOut["Return predictions"]
Focal --> ReturnOut
Soft --> ReturnOut
JRCO --> ReturnOut
L2O --> ReturnOut
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L134-L180)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L134-L180)

### Metric Initialization and Updates

RankModel supports:

- Evaluation metrics: AUROC (binary/multiclass), accuracy, MAE, MSE, grouped AUC/XAUC
- Train metrics: DecayAUC, accuracy, MAE/MSE, XAUC (with decay wrapper)
- Loss metrics: MeanMetric for each configured loss

Initialization:

- init_metric() registers:
  - Standard metrics for configured MetricConfig entries
  - Per-loss metrics via BaseModel’s helper
  - Train metrics via TrainMetricWrapper for TrainMetricConfig entries

Updates:

- update_metric() updates evaluation metrics using predictions and labels
- update_train_metric() updates train metrics with decayed behavior

```mermaid
flowchart TD
IMStart(["init_metric"]) --> Eval["For each MetricConfig"]
Eval --> MT{"metric type"}
MT --> |auc| AddAUC["torchmetrics.AUROC(binary)"]
MT --> |multiclass_auc| AddMAUC["torchmetrics.AUROC(multiclass)"]
MT --> |accuracy| AddACC["Accuracy(task=binary/multi)"]
MT --> |mean_absolute_error| AddMAE["MeanAbsoluteError()"]
MT --> |mean_squared_error| AddMSE["MeanSquaredError()"]
MT --> |grouped_auc| AddGAUC["GroupedAUC()"]
MT --> |xauc| AddXAUC["XAUC()"]
MT --> |grouped_xauc| AddGX["GroupedXAUC(max_pairs_per_group)"]
IMStart --> LossM["For each LossConfig<br/>register MeanMetric"]
IMStart --> TrainM["For each TrainMetricConfig"]
TrainM --> TMT{"metric type"}
TMT --> |auc| AddDAUC["DecayAUC(...)"]
TMT --> |accuracy| AddTACC["Accuracy(...)"]
TMT --> |mean_absolute_error| AddTMAE["MeanAbsoluteError()"]
TMT --> |mean_squared_error| AddTMSE["MeanSquaredError()"]
TMT --> |xauc| AddTX["XAUC(...)"]
AddDAUC --> Wrap["Wrap with TrainMetricWrapper"]
AddTACC --> Wrap
AddTMAE --> Wrap
AddTMSE --> Wrap
AddTX --> Wrap
Wrap --> DoneIM["Done"]
UpStart(["update_metric"]) --> EvalUp["For each MetricConfig<br/>update metric"]
UpStart --> LossUp["For each LossConfig<br/>update loss metric"]
EvalUp --> DoneUp["Done"]
LossUp --> DoneUp
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L286-L366)
- \[model.py\](file://tzrec/models/model.py#L185-L202)
- \[train_metric_wrapper.py\](file://tzrec/metrics/train_metric_wrapper.py#L20-L63)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L286-L366)
- \[model.py\](file://tzrec/models/model.py#L185-L202)
- \[train_metric_wrapper.py\](file://tzrec/metrics/train_metric_wrapper.py#L20-L63)

### Relationship Between Ranking and Candidate Generation Models

- Ranking models (RankModel subclasses) produce per-example scores or probabilities for scoring and reranking.
- Candidate generation models (e.g., DSSM) often output embeddings for retrieval and may use different loss types (e.g., softmax cross entropy for contrastive training).
- Both share the same embedding infrastructure and metric systems, but differ in output interpretation and typical loss choices.

[No sources needed since this section provides conceptual comparison]

## Dependency Analysis

RankModel depends on:

- BaseModel for shared interfaces and utilities
- EmbeddingGroup for grouped feature embeddings
- VariationalDropout for per-feature-group regularization
- Loss modules for training objectives
- Metric modules for evaluation and training monitoring

```mermaid
graph LR
RM["RankModel"] --> BM["BaseModel"]
RM --> EG["EmbeddingGroup"]
RM --> VD["VariationalDropout"]
RM --> BCE["BCEWithLogitsLoss"]
RM --> SCEL["CrossEntropyLoss"]
RM --> FL["BinaryFocalLoss"]
RM --> JRC["JRCLoss"]
RM --> MSE["MSELoss"]
RM --> TM["torchmetrics.*"]
RM --> TMW["TrainMetricWrapper"]
```

**Diagram sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L56-L513)
- \[model.py\](file://tzrec/models/model.py#L39-L138)
- \[embedding.py\](file://tzrec/modules/embedding.py#L139-L200)
- \[variational_dropout.py\](file://tzrec/modules/variational_dropout.py#L38-L119)
- \[focal_loss.py\](file://tzrec/loss/focal_loss.py#L18-L73)
- \[jrc_loss.py\](file://tzrec/loss/jrc_loss.py#L29-L118)
- \[train_metric_wrapper.py\](file://tzrec/metrics/train_metric_wrapper.py#L20-L63)

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L56-L513)
- \[model.py\](file://tzrec/models/model.py#L39-L138)

## Performance Considerations

- Variational dropout adds lightweight regularization with per-feature dropout probabilities and a penalty term scaled by batch size.
- Using grouped embeddings reduces memory footprint and improves throughput by sharing embedding tables across related features.
- Sample weights normalization ensures unbiased gradient updates when re-weighting examples.
- Mixed precision training can be enabled via the training wrapper to reduce memory and improve speed.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide

Common issues and resolutions:

- Incorrect num_class for loss/metric:
  - Binary/ focal loss requires num_class == 1
  - Softmax/JRC loss requires num_class > 1
  - Multiclass AUC requires num_class > 1
- JRC loss requires session ids in batch sparse features under the configured session_name
- Grouped AUC/XAUC require grouping keys present in batch sparse features
- Ensure feature groups are properly defined and feature names match those in the dataset

**Section sources**

- \[rank_model.py\](file://tzrec/models/rank_model.py#L143-L167)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L241-L256)
- \[rank_model.py\](file://tzrec/models/rank_model.py#L314-L326)

## Conclusion

RankModel provides a robust, extensible foundation for ranking tasks in TorchEasyRec. By centralizing embedding group management, optional variational dropout, multi-loss support, and comprehensive metrics, it enables consistent model development across diverse ranking scenarios. Its design cleanly separates concerns between input preparation, loss computation, and evaluation, while integrating seamlessly with the broader TorchEasyRec ecosystem.

## Appendices

### Configuration Examples

- Binary classification ranking (CTR) with BCE and AUC:

  - Define feature groups (wide/deep), model config, and a single binary cross entropy loss
  - Example: \[deepfm_criteo.config\](file://examples/deepfm_criteo.config#L278-L396)

- Multi-class ranking with softmax cross entropy and multiclass AUC:

  - Define feature groups and softmax cross entropy loss
  - Example: \[dssm_taobao.config\](file://examples/dssm_taobao.config#L201-L266)

- Session-aware ranking with JRC loss:

  - Configure JRC loss with session_name and ensure session ids are present in batch sparse features

- Using feature groups:

  - Feature groups define which features go into which embedding collections; see examples for wide/deep/sequence group definitions

- Model configuration schema:

  - ModelConfig includes feature_groups, losses, metrics, train_metrics, and optional variational_dropout
  - Reference: \[model_pb2.py\](file://tzrec/protos/model_pb2.py#L1-L43)

**Section sources**

- \[deepfm_criteo.config\](file://examples/deepfm_criteo.config#L278-L396)
- \[dssm_taobao.config\](file://examples/dssm_taobao.config#L201-L266)
- \[model_pb2.py\](file://tzrec/protos/model_pb2.py#L1-L43)
