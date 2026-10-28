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

当表的总参数（embedding + 优化器状态）超过 GPU 显存和主机内存时，可以把 DynamicEmbedding 表放到每个训练进程自带的 RocksDB 中（本地 NVMe、ESSD 云盘或极速型 NAS），GPU 上保留一个热点 cache。需要额外安装 TorchEasyRec-addons 构建的 `tzrec-rocksdb` wheel（CPU-only，与 CUDA 版本无关；要求 dynamicemb 为 `13a8055` 版本）。

```
train_config {
    rocksdb_config {
        local_path: "/mnt/nvme/tzrec_rocksdb"
        placement: AUTO
        gpu_cache_value_bytes_per_rank: 8589934592
        block_cache_bytes_per_rank: 34359738368
        write_buffer_bytes_per_rank: 4294967296
        staging_buffer_bytes_per_rank: 268435456
        io_threads: 16
        disk_budget_bytes: 4398046511104
    }
}
feature_configs {
    id_feature {
        feature_name: "user_id"
        embedding_dim: 64
        dynamicemb {
            max_capacity: 10000000000
            score_strategy: "LFU"
        }
    }
}
```

### 表放在哪里

- **placement: EXPLICIT**（默认）: 只有配置了 `dynamicemb { rocksdb {} }` 的表使用 RocksDB，其它表与原来一样
- **placement: AUTO**: 不需要逐表配置 `rocksdb {}`。规划器先估算全部 DynamicEmbedding 表放在 GPU/主机内存所需的空间，只有放不下时才把一部分表移到 RocksDB：按“每步查表次数 / 表大小”从小到大挑选（同一融合组已在 RocksDB 的优先），跳过磁盘放不下的表，最后把不再需要的表移回内存。这是基于容量的确定性规则，不代表性能最优（没有使用访问统计）。规划后再由 TorchRec 规划器校验，如果仍放不下会再多移一张表重试；仍然无解时报错并给出显存/内存和磁盘的缺口
- 显式覆盖：`rocksdb {}` 的表总是使用 RocksDB；配置了 `cache_load_factor` 的表总是留在内存；`score_strategy: "CUSTOMIZED"` 的表不会放到 RocksDB（TorchEasyRec 没有 CUSTOMIZED 分数的生产者）
- 没有配置 rocksdb_config 时不会使用磁盘，也不需要安装 tzrec-rocksdb；AUTO 未选中任何表时不会打开数据库，也不会预留 RocksDB 的主机内存
- 训练日志中的 `dynamicemb placement` 列出每张表的去向、原因和估算的显存/内存/磁盘，`RocksDB live storage` 给出磁盘估算的分项明细
- 同一份 checkpoint 可以在预算改变后用不同的放置方式继续训练（内存版与 RocksDB 版之间可以互相迁移），见下文“Checkpoint 与恢复”

### 配置项

- **rocksdb_config**（train_config 中；内存类预算按 rank 计一次，磁盘预算按存储域计）

  - local_path: 在线数据库的根目录，目录结构为 `<local_path>/<run_id>/rank_<r>/`，`run_id` 由 model_dir 计算。使用本地盘、ESSD 或极速型 NAS，不要使用 tmpfs，也不要使用 OSS 挂载目录
  - placement: EXPLICIT 或 AUTO
  - gpu_cache_value_bytes_per_rank: RocksDB 表 GPU cache 中 embedding + 优化器状态的显存预算，按各 RocksDB 表在本 rank 上的参数量等比例分配。必须大于 0（RocksDB 表总是在 GPU cache 之后），且小于全部 RocksDB 表在本 rank 上的参数总量（dynamicemb 在显存放得下全表时会忽略外部存储）
  - block_cache_bytes_per_rank / write_buffer_bytes_per_rank / staging_buffer_bytes_per_rank: RocksDB block cache、memtable 和单次读写的 pinned 内存预算。有表放到 RocksDB 时，三者之和加约 2GB 开销从规划器的主机内存容量中扣除一次
  - io_threads: 一次批量查表拆分给多少个线程并发读，网络存储（NAS、云盘）建议调大
  - max_background_jobs: RocksDB flush/compaction 线程数，默认 8
  - disk_budget_bytes: 本任务在 local_path 所在存储上最多可以占用的字节数，有表放到 RocksDB 时必填。共享同一存储（同一 NAS 服务器、同一本地盘，或相同 storage_domain）的所有 rank 共用这一个预算
  - storage_domain: 显式指定存储域。NFS/CPFS 按服务器、本地块设备按“主机+设备”自动识别；FUSE 或无法识别的文件系统必须配置
  - disk_reserve_bytes: 预留的安全空间，默认 16GiB
  - sst_space_factor: 校准过的“SST 文件大小 / 逻辑行字节数”系数。未配置时使用 1.5，这是一个临时估计值，不是实测值。在 ESSD PL3 云盘上用 STEP/LFU/TIMESTAMP 表（dim 32/64，Adam）持续训练、表满后持续淘汰时实测为 1.02-1.05；可先用默认值训练一段，再按保存日志中的 live SST 与导出字节数之比加少量余量配置
  - capacity_check_interval_steps: 训练中检查磁盘余量的间隔步数，默认 100
  - disk_warn_headroom_bytes / disk_critical_headroom_bytes: 余量告警线和停止线，默认分别为 disk_budget_bytes 的 10% 和 5%

- **dynamicemb.rocksdb {}**: 该表总是使用 RocksDB。不能配置 `cache_load_factor`（cache 大小由 `gpu_cache_value_bytes_per_rank` 统一决定），`score_strategy` 不能为 `CUSTOMIZED`

### 淘汰策略与容量

- 支持 `STEP`、`TIMESTAMP`、`LFU` 和 `NO_EVICTION`。分数由 GPU cache 按 dynamicemb 的规则产生（STEP/TIMESTAMP 在访问时更新，LFU 累加访问次数，包括同一 batch 内的重复 Id 和 cache 命中），cache 淘汰时随行写回 RocksDB；从 RocksDB 读回的 LFU 行会带上本 batch 的频次，不会重复或漏计
- STEP/TIMESTAMP/LFU 表在每个训练步结束时按 `max_capacity`（每个分片的容量）做逻辑淘汰：删除分数最低的、不在 GPU cache 中的行（含 embedding 和优化器状态），分数相同时按 Id 排序。这是每个分片内的全局最低分，与内存版 dynamicemb 在哈希桶内淘汰最低分的结果不完全相同。每步之间可能短暂超出容量（最多一个 batch 的新 Id）
- 优先淘汰不在 GPU cache 中的行；只有这类行已全部淘汰仍超出容量时（同一融合组的表共享 GPU cache，某张表在 cache 中的行数可能超过它自己的容量），才淘汰 cache 中分数最低的行，并同时从 cache 中删除，因此 cache 写回不会让已淘汰的 Id 复活。如果一个 batch 中某张表的去重 Id 数超过它在 GPU cache 中的份额，超出部分存放在 dynamicemb cache 的 overflow 区域，无法淘汰，此时该表会高于 max_capacity 并打印告警，请调大 max_capacity 或 GPU cache 预算。被淘汰的 Id 再次出现时按新 Id 重新准入、初始化，优化器状态重置
- 开启增量导出的 tombstone 时，逻辑淘汰的 Id 会像内存版一样发布 tombstone；GPU cache 的替换只是写回，不产生 tombstone。进程重启时尚未发布的淘汰 Id 不会保存（与内存版一致）
- `NO_EVICTION` 表不淘汰，与内存版一样随新 Id 持续增长。`max_capacity` 只作为磁盘规划的声明上限，并不强制；请依赖下文的运行时磁盘监控
- GPU cache 至少要能放下一个 batch 在每个融合组上的去重 Id 数，否则 dynamicemb 会报 `Outstanding prefetched keys ... exceed cache capacity`，需要调大 `gpu_cache_value_bytes_per_rank` 或减小 batch_size
- 使用 frequency_admission_strategy 时，频次 Counter 仍在显存中，并随 checkpoint 一起保存
- 同一 FeatureGroup 中兼容的 RocksDB 表仍会融合，每个 rank 只有一个数据库；每张表的数据以“模块路径.表名”作为持久命名空间，与放置方式和表的顺序无关

### 磁盘规划与监控

- 规划时每个 rank 上报 local_path 所在的文件系统（挂载源、类型、可用空间、本任务已占用空间），按存储域汇总后估算峰值占用：逻辑行字节数（key、分数、embedding、优化器状态以及淘汰索引，按表自身的维度和 dtype 计算）× `sst_space_factor` + compaction 临时输出 + MANIFEST/日志等文件 + 一个 batch 的新 Id + 预留空间；分别估算训练、保存（cache 全量写回产生的旧版本）和恢复三个场景，取最大值，不相加
- 峰值必须同时满足：不超过 `disk_budget_bytes`；新增占用（峰值减去本任务已占用的空间）不超过文件系统当前可用空间。NAS 的配额通常无法查询，此时以 `disk_budget_bytes` 为准，日志中会注明
- 训练中每 `capacity_check_interval_steps` 步各 rank 交换一次采样（可用空间、本任务占用、SST 大小、compaction 积压），余量 = min(可用空间 - 预留, 预算 - 占用)。低于告警线时打印告警和按增长速度估算的剩余时间；低于停止线时在该步结束后停止训练：如果余量还够一次保存，先保存 checkpoint 再停止，否则直接停止（上一个完整的 checkpoint 不受影响），抛出 `DiskCapacityError`。不会为了腾空间删除 Id、快照或其它文件
- `estimate-pending-compaction-bytes` 表示待重写的数据量，不是需要额外预留的空间，日志中单独列出；如果它在连续多次检查中持续增长，日志会标记 compaction 不稳定

### Checkpoint 与恢复

- RocksDB 表与内存版 DynamicEmbedding 表使用完全相同的 checkpoint 格式：保存时 `DynamicEmbDump` 先把 GPU cache 写回，再从冻结的数据库视图（刷盘、暂停后台 compaction、持有快照）按块流式写出 `model.ckpt-<step>/dynamicemb/` 下的 keys/values/scores/opt_values 文件，最后写 `_SUCCESS`。不会在训练盘上保留历史 checkpoint，也不会先在本地落一份完整导出
- model_dir 可以是 OSS 挂载目录（如 DLC 的 ossfs 数据集），文件按顺序流式写入，由 ossfs 分片上传。单文件大小受 ossfs 分片大小 × 10000 限制，超过时报 `File too large`，请调大 `upload_buffer_size`（见 FAQ）
- 开启 RocksDB 后，`--continue_train`、评估、预测和导出只会选择带 `_SUCCESS` 的 checkpoint；比恢复点更新、但没有 `_SUCCESS` 的 checkpoint 目录会被删除（日志中打印文件数和大小），在 OSS 上重命名等于全量复制
- 恢复时先清空 GPU cache，再新建一个空的工作库，从 checkpoint 文件按块加载；上一次的工作库和崩溃遗留的数据不会被复用。因此只要 checkpoint 完整，本地盘丢失后也可以在其它节点恢复
- checkpoint 与放置方式无关：同一份 checkpoint 可以在内存版和 RocksDB 版之间恢复，也可以在调整预算后由 AUTO 选出不同的放置继续训练。若放置变化使原本不在同一融合组、优化器参数（例如 Adam 的迭代次数）不同的表融合到一起，恢复会报错。world size、维度或优化器类型的变化按 dynamicemb 自身的规则校验
- 评估、导出时 RocksDB 表同样从 checkpoint 文件加载到一个新的临时数据库（会占用与表大小相当的磁盘）
- 旧版本（原生 RocksDB checkpoint，`model.ckpt-<step>/rocksdb/rank_<r>.json`）的 checkpoint 仍可恢复（仅限 `NO_EVICTION` 表）和导出；恢复后保存一次即转换为新格式。旧的原生 checkpoint 在对应 `model.ckpt-<step>` 被清理后删除
- 保存耗时与表大小成正比（全量写出），训练在保存期间暂停（实测经 ossfs 写出约 125 GiB 需约 13 分钟，从 OSS 恢复约 6 分钟），请据此设置 `save_checkpoints_steps`；日志 `RocksDB checkpoint` 给出本次保存的 cache 写回、导出行数和字节数，以及进程启动以来训练、发布的累计写回量、淘汰行数和 RocksDB flush/compaction 写入量
- 已知限制：恢复前框架会先用一个 batch 初始化优化器状态，此时 frequency admission 的 Counter 计数无法回滚

### 排障

- `tzrec-rocksdb wheel`: 未安装 tzrec-rocksdb
- `set rocksdb_config.gpu_cache_value_bytes_per_rank > 0`: RocksDB 表需要 GPU cache
- `disk_budget_bytes is not set` / `projected peak exceeds the budget` / `growth needs ... more than are free`: 磁盘预算不足或可用空间不足，日志 `RocksDB live storage` 中有分项明细；可以调大预算、换更大的存储、降低 `max_capacity`，或在 AUTO 下增加内存预算
- `no placement fits`: 显存、内存和磁盘都放不下，错误信息列出每张表的去向和缺口
- 训练出现 NaN，且规划日志有 `GPU cache share of ... rows may be smaller than one step's lookups`：dynamicemb `13a8055` 的 GPU cache 在一个 batch 中某张融合表的去重 Id 数超过该表的 cache 份额时会返回无效的行（内存版 CACHING 布局同样存在），请调大 `gpu_cache_value_bytes_per_rank`
- `set rocksdb_config.storage_domain`: local_path 所在文件系统无法判断与谁共享容量
- `stopped at step ...: RocksDB disk headroom fell below`: 运行时磁盘余量低于停止线
- 训练变慢：日志中的 `RocksDB disk` 和保存日志给出 SST 大小、compaction 积压等指标，判断是读延迟、写入限流还是 compaction 积压

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
