# BaseSampler Abstract Class

<cite>
**Referenced Files in This Document**
- [sampler.py](file://tzrec/datasets/sampler.py)
- [sampler.proto](file://tzrec/protos/sampler.proto)
- [dataset.py](file://tzrec/datasets/dataset.py)
- [sampler_test.py](file://tzrec/datasets/sampler_test.py)
- [load_class.py](file://tzrec/utils/load_class.py)
- [env_util.py](file://tzrec/utils/env_util.py)
- [misc_util.py](file://tzrec/utils/misc_util.py)
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

## Introduction

The BaseSampler abstract class serves as the foundational interface for all sampling strategies in TorchEasyRec's graph learning pipeline. It provides a unified abstraction for negative sampling, hard negative sampling, and Tree-Structured Decomposition (TDM) sampling, enabling flexible and efficient data preparation for recommendation systems and graph neural networks.

BaseSampler integrates seamlessly with GraphLearn for distributed graph processing and PyTorch's distributed training capabilities. The class handles complex scenarios including multi-value attribute parsing, sparse feature handling, and sophisticated sampling strategies with configurable expansion factors and probability distributions.

## Project Structure

The BaseSampler implementation is organized within the datasets module, alongside protocol buffer definitions and integration utilities:

```mermaid
graph TB
subgraph "TorchEasyRec Datasets Module"
A[BaseSampler<br/>Abstract Class]
B[NegativeSampler<br/>Weighted Random]
C[HardNegativeSampler<br/>Conditional Sampling]
D[TDMSampler<br/>Tree Decomposition]
E[TDMPredictSampler<br/>Prediction Mode]
end
subgraph "Protocol Buffers"
F[sampler.proto<br/>Message Definitions]
end
subgraph "Integration Layer"
G[Dataset Integration<br/>DataPipeline]
H[GraphLearn<br/>Distributed Processing]
I[PyTorch Distributed<br/>Training Support]
end
A --> B
A --> C
A --> D
A --> E
F --> A
G --> A
H --> A
I --> A
```

**Diagram sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L219-L395)
- \[sampler.proto\](file://tzrec/protos/sampler.proto#L1-L142)

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L1-L1055)
- \[sampler.proto\](file://tzrec/protos/sampler.proto#L1-L142)

## Core Components

### BaseSampler Abstract Interface

The BaseSampler class defines the fundamental contract for all sampling implementations:

**Initialization Parameters:**

- `config`: SAMPLER_CFG_TYPES - Protocol buffer configuration containing sampling parameters
- `fields`: List[pa.Field] - Arrow field definitions for input data schema
- `batch_size`: int - Mini-batch size for training/inference
- `is_training`: bool - Training vs evaluation mode flag
- `multival_sep`: str - Multi-value separator character (default: chr(29))
- `typed_fields`: Optional\[List[pa.Field]\] - Typed field definitions for structured parsing

**Key Attributes:**

- `_batch_size`: Configured batch size for sampling operations
- `_multival_sep`: Multi-value separator for parsing composite features
- `_num_sample`: Maximum number of samples per batch
- `_cluster`: Cluster specification for distributed processing
- `_attr_names`: Names of parsed attributes
- `_attr_types`: PyArrow types for each attribute
- `_attr_gl_types`: GraphLearn compatible types
- `_attr_np_types`: NumPy types for conversion

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L219-L286)

### Attribute Handling System

BaseSampler implements sophisticated attribute parsing with support for:

- Mixed data types (int, float, string)
- Structured parsing for lists and maps
- Multi-value field separation
- Type casting and validation
- Ignored attribute handling

The attribute system automatically detects and converts between PyArrow, NumPy, and GraphLearn type systems, ensuring seamless integration across the data pipeline.

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L244-L277)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L330-L390)

## Architecture Overview

The BaseSampler architecture follows a layered approach combining GraphLearn integration, distributed computing, and PyTorch compatibility:

```mermaid
sequenceDiagram
participant Client as "Client Application"
participant Dataset as "BaseDataset"
participant Sampler as "BaseSampler"
participant GL as "GraphLearn"
participant Server as "Sampler Server"
Client->>Dataset : launch_sampler_cluster()
Dataset->>Sampler : create_instance(config, fields, batch_size)
Dataset->>Sampler : init_cluster(num_client_per_rank)
Sampler->>Server : launch_server()
Server-->>Sampler : server_ready
loop For Each Batch
Client->>Dataset : __iter__()
Dataset->>Sampler : init()
Dataset->>Sampler : get(input_data)
Sampler->>GL : graph_query()
GL-->>Sampler : sampled_nodes
Sampler->>Sampler : parse_attributes()
Sampler-->>Dataset : feature_dict
Dataset-->>Client : processed_batch
end
```

**Diagram sources**

- \[dataset.py\](file://tzrec/datasets/dataset.py#L241-L315)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L287-L324)

## Detailed Component Analysis

### Cluster Initialization System

BaseSampler implements a robust cluster initialization system for distributed graph processing:

```mermaid
flowchart TD
A["init_cluster() called"] --> B["Set client count per rank"]
B --> C["Calculate world_size<br/>and group_size"]
C --> D["Determine local_rank<br/>and group_rank"]
D --> E["_bootstrap() for server info"]
E --> F["Exchange GL server info<br/>across ranks"]
F --> G["Store cluster spec<br/>{server, client_count}"]
H["launch_server()"] --> I["Check cluster initialized"]
I --> J["Set tracker mode"]
J --> K["Launch server on LOCAL_RANK==0"]
L["init()"] --> M["Resolve client_id"]
M --> N["Calculate task_index"]
N --> O["Initialize GraphLearn client"]
```

**Diagram sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L81-L126)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L287-L324)

The cluster system supports:

- Multi-node distributed training
- Client-per-rank scaling
- Automatic port allocation
- Cross-process communication coordination

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L81-L126)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L287-L324)

### Distributed Training Support

BaseSampler integrates with PyTorch's distributed training framework:

**Environment Variables:**

- `WORLD_SIZE`: Total number of processes
- `LOCAL_WORLD_SIZE`: Processes per node
- `LOCAL_RANK`: Local rank within node
- `GROUP_RANK`: Node group identifier
- `RANK`: Global process rank

**Process Coordination:**

- Automatic client ID assignment
- Task index calculation for GraphLearn
- Worker-aware initialization
- Graceful timeout handling

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L119-L126)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L309-L324)

### GraphLearn Integration Patterns

BaseSampler leverages GraphLearn for efficient graph operations:

**Core Integration Points:**

- Graph construction from file paths
- Decoder configuration for attribute parsing
- Sampler creation for different strategies
- Distributed graph loading

**Supported Strategies:**

- Weighted random sampling
- Conditional sampling based on relationships
- Hierarchical tree traversal
- Sparse neighbor sampling

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L421-L440)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L517-L520)

### Sampling Lifecycle Management

The complete sampling lifecycle follows a structured progression:

```mermaid
stateDiagram-v2
[*] --> Initialized
Initialized --> ClusterReady : init_cluster()
ClusterReady --> ServerLaunched : launch_server()
ServerLaunched --> ClientReady : init()
ClientReady --> Sampling : get()
Sampling --> ClientReady : Next Batch
ClientReady --> [*] : Cleanup
note right of ClientReady
Batch processing with
attribute parsing
end note
```

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L309-L324)
- \[sampler.py\](file://tzrec/datasets/sampler.py#L330-L356)

### Concrete Sampler Implementations

#### NegativeSampler

Implements weighted random sampling of items not present in the current batch:

**Key Features:**

- Single graph node sampling
- Weighted selection based on node weights
- Configurable attribute fields
- Batch padding for uniform processing

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L397-L462)

#### HardNegativeSampler

Extends negative sampling with hard negative examples:

**Enhanced Capabilities:**

- Dual sampling strategy (negative + hard negatives)
- Sparse neighbor sampling for hard negatives
- Index tracking for hard negative identification
- Combined feature concatenation

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L555-L649)

#### TDMSampler

Implements Tree-Structured Decomposition sampling:

**Advanced Features:**

- Hierarchical tree traversal
- Layer-wise sampling with configurable expansion
- Probability-based layer retention
- Training and prediction modes

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L753-L966)

### Attribute Parsing System

BaseSampler implements sophisticated attribute parsing with type safety:

```mermaid
flowchart LR
A["Raw GraphLearn Nodes"] --> B["Attribute Detection"]
B --> C{"Attribute Type"}
C --> |int| D["nodes.int_attrs"]
C --> |float| E["nodes.float_attrs"]
C --> |string| F["nodes.string_attrs"]
D --> G["NumPy Conversion"]
E --> G
F --> H["UTF-8 Decoding"]
H --> G
G --> I["PyArrow Array Creation"]
I --> J["Type Validation & Casting"]
J --> K["Final Feature Arrays"]
```

**Diagram sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L330-L356)

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L330-L390)

## Dependency Analysis

BaseSampler has strategic dependencies that enable its comprehensive functionality:

```mermaid
graph TB
subgraph "Core Dependencies"
A[graphlearn] --> B[Graph Operations]
C[torch] --> D[Distributed Training]
E[pyarrow] --> F[Data Processing]
G[numpy] --> H[Array Operations]
end
subgraph "Internal Dependencies"
I[load_class.py] --> J[Class Registration]
K[env_util.py] --> L[Environment Config]
M[misc_util.py] --> N[Utility Functions]
end
subgraph "Protocol Dependencies"
O[sampler.proto] --> P[Config Schema]
Q[data.proto] --> R[Dataset Config]
end
A --> S[BaseSampler Implementation]
C --> S
E --> S
G --> S
I --> S
K --> S
M --> S
O --> S
Q --> S
```

**Diagram sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L12-L35)
- \[load_class.py\](file://tzrec/utils/load_class.py#L117-L145)

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L12-L35)
- \[load_class.py\](file://tzrec/utils/load_class.py#L117-L145)

## Performance Considerations

### Memory Management

BaseSampler implements several memory optimization strategies:

**Efficient Data Structures:**

- Resizable arrays for dynamic sampling results
- Batch-aware padding to minimize reallocation
- Lazy evaluation of expensive operations
- Proper cleanup in destructor

**Memory Optimization Techniques:**

- Reshape operations to avoid copies
- Type-specific conversions to reduce overhead
- Sparse matrix handling for neighbor sampling
- Controlled batch size management

### Performance Characteristics

**Time Complexity:**

- Sampling operations: O(k × log n) where k is sample count and n is graph size
- Attribute parsing: O(m × n) where m is attribute count and n is sample size
- Multi-value parsing: O(p × n) where p is average values per sample

**Space Complexity:**

- Sample storage: O(k × f) where f is average feature size
- Intermediate buffers: O(k × g) where g is graph connectivity
- Attribute arrays: O(n × t) where t is type size

### Scalability Features

**Horizontal Scaling:**

- Client-per-rank distribution
- Automatic load balancing
- Graceful degradation on failures
- Configurable expansion factors

**Vertical Scaling:**

- Adjustable batch sizes
- Configurable sample counts
- Optimized data types
- Efficient memory pooling

## Troubleshooting Guide

### Common Initialization Issues

**Cluster Initialization Failures:**

- Verify environment variables are set correctly
- Check network connectivity between nodes
- Ensure port availability for server binding
- Validate distributed backend initialization

**Memory Allocation Problems:**

- Monitor GPU/CPU memory usage during sampling
- Adjust batch sizes for memory-constrained environments
- Enable proper cleanup in destructors
- Use appropriate data types to reduce memory footprint

**Section sources**

- \[sampler.py\](file://tzrec/datasets/sampler.py#L326-L328)
- \[misc_util.py\](file://tzrec/utils/misc_util.py#L65-L72)

### Configuration Validation

**Protocol Buffer Validation:**

- Ensure all required fields are present
- Validate data types match field definitions
- Check attribute field names exist in schema
- Verify sampling parameters are within bounds

**Environment Configuration:**

- Confirm USE_HASH_NODE_ID setting matches graph data
- Verify distributed training environment variables
- Check GraphLearn configuration compatibility
- Validate multi-value separator consistency

**Section sources**

- \[sampler.proto\](file://tzrec/protos/sampler.proto#L1-L142)
- \[env_util.py\](file://tzrec/utils/env_util.py#L19-L22)

### Debugging Techniques

**Logging and Monitoring:**

- Enable detailed logging for sampling operations
- Monitor GraphLearn server status
- Track memory usage patterns
- Profile sampling performance bottlenecks

**Testing Strategies:**

- Unit testing with small datasets
- Distributed testing with multiple processes
- Performance benchmarking with synthetic data
- Integration testing with full pipeline

**Section sources**

- \[sampler_test.py\](file://tzrec/datasets/sampler_test.py#L1-L686)

## Conclusion

BaseSampler provides a comprehensive and extensible foundation for sampling strategies in TorchEasyRec. Its design balances flexibility with performance, supporting complex graph learning scenarios while maintaining ease of use for developers extending the framework.

The class successfully integrates multiple technologies including GraphLearn for distributed graph processing, PyTorch for machine learning workflows, and sophisticated data parsing systems. Its modular architecture enables easy extension for custom sampling strategies while maintaining compatibility with existing implementations.

Key strengths include robust distributed training support, efficient memory management, comprehensive type system integration, and extensive testing coverage. These features make BaseSampler an ideal foundation for production-scale recommendation systems and graph neural network applications.

Future enhancements could include additional sampling strategies, improved performance monitoring, and expanded integration with emerging distributed computing frameworks.
