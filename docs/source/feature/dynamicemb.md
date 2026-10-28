# Dynamic Embedding

DynamicEmbedding 是特征零Hash冲突Id化的一种方式，它相比设置`hash_bucket_size`的方式能避免hash冲突，相比设置`vocab_dict`和`vocab_list`的方式能更灵活动态地进行id的准入和驱逐。DynamicEmbedding 常用于user id，item id，combo feature等超大id枚举数的特征配置中。DynamicEmbedding 相比 ZCH 能外接PS，支撑十亿百亿甚至更多的Id枚举数，Id准入和淘汰无需攒Batch，可以更加及时。

dynamicemb安装方式:

- 官方镜像：`-cu126` 与 `-cu130` 变体已预装 dynamicemb，无需手动安装。
- pip 安装：按[本地安装教程](../quick_start/local_tutorial.md)安装 `tzrec[cu126]` 或 `tzrec[cu130]`（支持 Python 3.10/3.11/3.12），dynamicemb 已包含在对应 device 的依赖中。

支持配置 DynamicEmbedding 的特征类型为 `id_feature`、`combo_feature`、`lookup_feature`、`match_feature`、`regex_replace_feature`、`custom_feature`、`bool_mask_feature`，其余特征类型（如配置了 `boundaries` 的 `raw_feature`、`expr_feature`、`tokenize_feature`、`combine_feature`）暂不支持。

注：同一个 FeatureGroup 中若存在多个配置了 DynamicEmbedding 的特征，底层 dynamicemb 会自动将这些表融合到同一份存储里（table fusion），共享 cache/admission counter，降低显存占用并减少内存碎片，无需额外配置。

注：配置了 DynamicEmbedding 的模型导出时需设置环境变量 `USE_DISTRIBUTED_EMBEDDING=1`，使用分布式 embedding 导出模式，详见[模型导出](../usage/export.md)的环境变量章节。

以id_feature的配置为例，DynamicEmbedding 只需在id_feature新增一个dynamicemb的配置字段

```
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        dynamicemb {
            max_capacity: 100000
            initializer_args {
                mode: "NORMAL"
                std_dev: 0.05
            }
            score_strategy: "STEP"
            frequency_admission_strategy {
                threshold: 5
            }
        }
    }
}
```

- **max_capacity**: 最大的id数，Id数超过后会根据Id的驱逐策略进行淘汰

- **score_strategy**: Id驱逐策略，默认为 STEP，目前支持 TIMESTAMP | STEP | LFU | NO_EVICTION

  - TIMESTAMP: 每个Id根据最近更新的时间戳，驱逐时间戳最小的Id
  - STEP: 每个Id根据最近更新的迭代步数，驱逐步数最早的Id
  - LFU: 每个Id根据出现的频次，驱逐频次小的Id
  - NO_EVICTION: 不驱逐，表容量到达 `max_capacity` 后不再接收新Id，适合配合 `init_capacity_per_rank` 做可扩容的表

- **bucket_capacity**: （可选）dynamicemb 哈希表 bucket 大小，默认为 128，增大可提升表的装填率但会增加每次查表的探测代价

- **initializer_args**: 参数初始化设置，默认是 UNIFORM

  - mode: 初始化方式，可选 NORMAL | TRUNCATED_NORMAL | UNIFORM | CONSTANT
  - mean: NORMAL | TRUNCATED_NORMAL 初始化方式的均值，默认为 0
  - std_dev: NORMAL | TRUNCATED_NORMAL 初始化方式的标准差，默认为 sqrt(1 / embedding_dim)
  - lower: UNIFORM 初始化方式的均值的下界，默认为 - sqrt(1 / max_capacity)
  - upper: UNIFORM 初始化方式的均值的上界，默认为 sqrt(1 / max_capacity)
  - value: CONSTANT 初始化方式的值，默认为0

- **eval_initializer_args**: （可选）评估时的初始化方式，默认是 CONSTANT，value=0

- **init_capacity_per_rank**: （可选）初始的每个Rank上的id数，默认等于max_capacity。如需显式配置，建议不小于单个Batch在每个Rank上的新增Id数，过小会导致首个Batch部分Id插入失败，其embedding未初始化，训练loss为nan

- **admission_strategy**: (可选) 特征准入策略，默认不开启，目前支持 frequency_admission_strategy 和 probabilistic_admission_strategy

  - **frequency_admission_strategy**: 基于频次的特征准入策略，频次统计Counter会额外占用显存
    - threshold: 准入频次
    - initializer_args: （可选）未准入的ID的初始化方式，默认是 CONSTANT，value=0
    - counter_capacity: （可选）频次统计Counter的最大id数，默认与max_capacity相等
    - counter_bucket_capacity: （可选）频次统计Counter的每个bucket的最大id数，默认为1024
  - **probabilistic_admission_strategy**: 基于概率的特征准入策略，未准入的Id每次出现都独立抛一次硬币，平均出现 1/probability 次后准入。相比 frequency_admission_strategy 不需要频次统计Counter，不额外占用显存，也不会写入 checkpoint（保存/加载时的 `Counter table is none` warning 属正常现象）
    - probability: 单次出现的准入概率，取值范围 [0, 1]
    - initializer_args: （可选）未准入的ID的初始化方式，默认是 CONSTANT，value=0

- **init_table**: （可选）初始化表的路径，支持Odps/Parquet/Csv格式，表的第一列为id值，第二列为embedding值。

  - 注意: init_table 参数不能再训练任务中直接生效，需要前置一个使用init_table构建初始化ckpt的任务，具体步骤如下

  - 使用init_table构建初始化ckpt

    ```bash
    python -m tzrec.tools.dynamicemb.create_dynamicemb_init_ckpt \
    --pipeline_config_path {PATH_TO_CONFIG_WITH_DYNAMICEMB} \
    --world_size $WORLD_SIZE*$NPROC_PER_NODE \
    --save_dir {INIT_CKPT_PATH}
    ```

    - --pipeline_config_path: 训练配置文件
    - --world_size: 训练进程数，一般情况下为训练的torchrun命令中的 nnodes\*nproc-per-node
    - --save_dir: 初始化模型保存目录
    - --reader_worker_num:（可选）读worker数目，默认为自动根据cpu核数设置
    - --separator: (可选) embedding数据类型如果为string的情况下的分隔符

  - 使用初始化的ckpt训练模型

    ```bash
    torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.train_eval \
    --pipeline_config_path {PATH_TO_CONFIG_WITH_DYNAMICEMB} \
    --fine_tune_checkpoint {INIT_CKPT_PATH}/model.ckpt-0
    ```

## RocksDB 存储

当表的总参数（embedding + 优化器状态）超过 GPU 显存和主机内存时，可以把 DynamicEmbedding 表放到每个训练进程自带的 RocksDB 中（本地 NVMe、云盘或极速型 NAS），GPU 上只保留一个可选的热点 cache。需要额外安装 TorchEasyRec-addons 构建的 `tzrec-rocksdb` wheel（CPU-only，与 CUDA 版本无关；要求 dynamicemb 为 `13a8055` 版本）。

```
train_config {
    rocksdb_config {
        local_path: "/mnt/nvme/tzrec_rocksdb"
        gpu_cache_value_bytes_per_rank: 8589934592
        block_cache_bytes_per_rank: 34359738368
        write_buffer_bytes_per_rank: 4294967296
        staging_buffer_bytes_per_rank: 268435456
        io_threads: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "user_id"
        embedding_dim: 64
        dynamicemb {
            max_capacity: 10000000000
            score_strategy: "NO_EVICTION"
            rocksdb {}
        }
    }
}
```

- **rocksdb_config**（train_config 中，每个 rank 一份预算，只计一次）

  - local_path: 数据库与本地 checkpoint 的根目录，目录结构为 `<local_path>/<run_id>/rank_<r>/`，`run_id` 由 model_dir 计算，同一个 model_dir 的重启共享同一目录。不要使用 tmpfs（会占用主机内存）。使用 NAS 等共享存储时，节点故障后任务可在其它节点从 checkpoint 恢复；使用本地盘时，本地盘丢失后无法恢复（恢复时会明确报错，不会用空表继续训练）
  - gpu_cache_value_bytes_per_rank: GPU cache 中 embedding + 优化器状态的显存预算，按各 RocksDB 表在本 rank 上的参数量等比例分配（所有表 cache 比例相同）。0 表示不使用 GPU cache，每次查表都读 RocksDB（适合验证，吞吐较低）。必须小于全部 RocksDB 表在本 rank 上的参数总量，否则报错（dynamicemb 在显存放得下全表时会忽略外部存储）
  - block_cache_bytes_per_rank / write_buffer_bytes_per_rank / staging_buffer_bytes_per_rank: RocksDB block cache、memtable 和单次读写的 pinned 内存预算。三者之和加约 2GB 开销从规划器的主机内存容量中扣除一次
  - io_threads: 一次批量查表拆分给多少个线程并发读，网络存储（NAS、云盘）建议调大
  - max_background_jobs: RocksDB flush/compaction 线程数，默认 8

- **dynamicemb.rocksdb {}**: 该表使用 RocksDB。要求 `score_strategy: "NO_EVICTION"`，且不能配置 `cache_load_factor`（cache 大小由 `gpu_cache_value_bytes_per_rank` 统一决定）

### 容量与保留语义

- RocksDB 不会淘汰任何 Id。GPU cache 满时淘汰的行会写回 RocksDB，而不是删除，因此增量导出（delta embedding dump）不会因为 cache 淘汰产生 tombstone，`dump_evicted_tombstones` 对 RocksDB 表不生效
- `max_capacity` 只用于规划和 cache 大小计算，不是磁盘配额，也不是已准入 Id 数的精确上限。请通过 checkpoint 中的统计信息（`rocksdb/rank_<r>.json` 的 `stats`）和磁盘剩余空间监控实际行数与占用；磁盘写满或写入失败会使训练报错退出，不会生成不完整的 checkpoint
- GPU cache 至少要能放下一个 batch 在每个融合组上的去重 Id 数，否则 dynamicemb 会报 `Outstanding prefetched keys ... exceed cache capacity`，需要调大 `gpu_cache_value_bytes_per_rank` 或减小 batch_size
- 使用 frequency_admission_strategy 时，频次 Counter 仍在显存中，并随 checkpoint 一起保存
- 同一 FeatureGroup 中兼容的 RocksDB 表仍会融合，每个 rank 只有一个数据库；每张表的数据以“模块路径.表名”作为持久命名空间，调整表的顺序不影响恢复，但改变融合关系、优化器、维度或 world size 需要迁移，恢复时会报错

### Checkpoint 与恢复

- 保存 checkpoint 时，RocksDB 表不会全量写到 `model.ckpt-<step>/dynamicemb/`：先把 GPU cache 写回 RocksDB，再在 `<local_path>/<run_id>/rank_<r>/ckpt-<step>/` 生成 RocksDB 原生 checkpoint（与工作库在同一文件系统时 SST 文件为硬链接，几乎不占额外空间），并在 `model.ckpt-<step>/rocksdb/rank_<r>.json` 记录清单，最后写入 `_SUCCESS`
- 开启 RocksDB 后，`--continue_train`、评估、预测和导出只会选择带 `_SUCCESS` 的 checkpoint；比恢复点更新、但没有 `_SUCCESS` 的目录会被重命名为 `incomplete.model.ckpt-<step>.<时间戳>` 保留现场
- 恢复时清空 GPU cache，从原生 checkpoint 复制出一份新的可写数据库继续训练，原 checkpoint 保持不变；进程崩溃后遗留的工作库不会被复用
- 原生 checkpoint 跟随 `keep_checkpoint_max` 的保留策略清理：对应的 `model.ckpt-<step>` 被删除后，下次保存时各 rank 删除自己的原生 checkpoint
- 已知限制：恢复前框架会先用一个 batch 初始化优化器状态，此时 frequency admission 的 Counter 计数无法回滚

### 从内存版 DynamicEmbedding checkpoint 迁移

已有的 `dynamicemb/` checkpoint 可以离线转换为 RocksDB checkpoint（CPU 执行，按块读取，不需要把整表加载到内存或显存，保持 world size 和 key 路由不变）：

```bash
python -m tzrec.tools.dynamicemb.import_rocksdb_checkpoint \
    --pipeline_config_path {RocksDB 版本的训练配置} \
    --checkpoint_path {旧 model_dir}/model.ckpt-100000 \
    --output_model_dir {新 model_dir}
```

- 会复制 dense 参数、优化器、学习率和数据读取位置，并转换 embedding、优化器状态、score 和准入 Counter；之后用 `--fine_tune_checkpoint {新 model_dir}/model.ckpt-100000` 或在新 model_dir 上 `--continue_train` 继续训练
- `--no_optim` 只转换 embedding，优化器状态重置（清单中 `exact_resume: false`，不是精确续训）
- 训练配置中所有 DynamicEmbedding 表都需要配置 `rocksdb {}`；也可以不做离线转换，直接用旧 checkpoint 作为 `fine_tune_checkpoint`，会在恢复时逐块导入

### 排障

- `no RocksDB session is open`: 自定义入口没有调用 `rocksdb_util.start`
- `tzrec-rocksdb wheel`: 未安装 tzrec-rocksdb
- `refusing to resume RocksDB tables from an incomplete checkpoint` / `RocksDB checkpoint file ... missing or changed`: 指定的 checkpoint 不完整或原生 checkpoint 所在磁盘已丢失，请换用其它完整的 checkpoint
- 训练变慢：查看 checkpoint 清单 `stats` 中 `find_p99_s`、`rocksdb.actual-delayed-write-rate`、`rocksdb.estimate-pending-compaction-bytes` 等指标，判断是读延迟、写入限流还是 compaction 积压

## 从ZCH训练好的模型迁移

如果已有用 `zch{}` 训练好的模型，希望切换到 `dynamicemb{}` 继续训练，可以使用 `tzrec.tools.dynamicemb.zch_to_dynamicemb_convert` 将ZCH checkpoint 转换为 dynamicemb 格式的 checkpoint，下游训练任务直接通过 `--fine_tune_checkpoint` 热启动。

```bash
python -m tzrec.tools.dynamicemb.zch_to_dynamicemb_convert \
--source_checkpoint_path {ZCH_CKPT_PATH}/model.ckpt-N \
--source_pipeline_config_path {PATH_TO_CONFIG_WITH_ZCH} \
--target_pipeline_config_path {PATH_TO_CONFIG_WITH_DYNAMICEMB} \
--save_dir {CONVERTED_CKPT_PATH}
```

- --source_checkpoint_path: ZCH 训练好的 checkpoint 目录（形如 `.../model.ckpt-N/`，需包含 `model/` 子目录的 DCP 分片）
- --source_pipeline_config_path: 原始训练所用的 pipeline 配置（特征上配置了 `zch{}`）
- --target_pipeline_config_path: 切换后的 pipeline 配置（同名特征上将 `zch{}` 替换为 `dynamicemb{}`，`embedding_dim` 保持一致）
- --save_dir: 转换后 checkpoint 的输出目录，工具会写入 `{save_dir}/model.ckpt-0/`
- --world_size:（可选）转换后 dynamicemb 分片的 world_size，默认从 source checkpoint 的 `.distcp` 文件名自动识别

### 迁移内容

- **Embedding**: ZCH 表中已学到的 embedding 向量会迁移到 dynamicemb。
- **分数（Scores）**: 仅当目标 `score_strategy` 为 `LFU` 或 `STEP` 时迁移；其它策略 dynamicemb 会自行初始化分数。
- **非 ZCH 部分**: 稠密层、非 ZCH embedding 及其对应 optimizer state 直接保留。

### 未迁移内容

- ZCH embedding 的 optimizer state 不迁移，dynamicemb 加载后自行冷启动。

ZCH 本身的配置详见 [ZCH](./zch.md)。
