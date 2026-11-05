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

- **init_capacity_per_rank**: （可选）初始的每个Rank上的id数，默认等于max_capacity。如需显式配置，建议不小于单个Batch在每个Rank上的新增Id数，过小会导致首个Batch部分Id插入失败，其embedding未初始化，训练loss为nan。单个Batch的去重Id数可以用`tzrec.tools.add_pooling_factor_to_config`测得的`unique_id_factor`乘以`batch_size`来估计（见[pooling_factor与unique_id_factor](../usage/train.md#pooling-factor)）

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

当 DynamicEmbedding 表的总参数（embedding + 优化器状态）超过 GPU 显存和主机内存时，规划器可以把一部分表放到每个训练进程自带的 RocksDB 中（本地 NVMe、ESSD 云盘或极速型 NAS），GPU 上保留一个热点 cache。需要额外安装 TorchEasyRec-addons 构建的 `tzrec-ext` wheel（CPU-only，与 CUDA 版本无关；在 dynamicemb `13a8055` 上验证）。

```
train_config {
    rocksdb_config {
        root_dir: "/mnt/nvme/tzrec_ext.rocksdb"
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

### 配置项

- **rocksdb_config**（train_config 中）
  - root_dir: 工作数据库的根目录，使用本地盘、ESSD 或极速型 NAS 等 NFS 类文件系统，不要使用 OSS 等 FUSE 挂载目录（会报错）。每个进程使用 `<root_dir>/<model_dir 标识>/rank_<rank>/<模式>-<时间>-<pid>`，不同任务、rank 和进程互不冲突
  - disk_budget_bytes: 可选，本任务在 root_dir 所在文件系统上最多可以占用的字节数，共享同一磁盘池的所有 rank 共用。不配置时只受文件系统可用空间限制
- GPU cache、block cache、memtable、pinned staging 的大小、线程数以及恢复导入的内存都由规划器决定，不需要配置

### 资源限制

- 规划器使用的主机内存为容器内存上限（cgroup）扣除 `HOST_MEMORY_RESERVE_PERCENT`（环境变量，默认 0.15，给数据加载、运行时等非 embedding 用途）后按本机 rank 数均分；显存同样扣除 `STORAGE_RESERVE_PERCENT`（默认 0.15）。这两个比例对所有任务生效，不只是使用 RocksDB 的任务
- 数据加载进程较多、内存占用较大时，请调大 `HOST_MEMORY_RESERVE_PERCENT`

### 表放在哪里

- 规划器先在不使用 RocksDB 的前提下规划；只有 GPU 和主机内存放不下全部表时，才允许把表放到 RocksDB，并且此时才扣除 RocksDB 自身的固定主机内存（memtable、staging、block cache 数据部分和约 2GB 开销，每 rank 约 4.25GB）
- 允许使用 RocksDB 时，每张可以放到 RocksDB 的表除了内存中的各种 cache 比例外，还会得到多个“RocksDB + GPU cache”选项（cache 比例 0.0001、0.001、0.01、0.1 ~ 0.9）。TorchRec 的动态规划（显存 × 内存 × 磁盘）在满足显存、内存和磁盘容量的前提下按估算性能选择每张表的位置和 cache 比例：RocksDB 表的 cache 未命中按磁盘带宽估算，所以能放进内存的表仍留在内存
- 只有 cache 比例相同的 RocksDB 表会融合到一起（dynamicemb 对一个融合组只使用一个 cache 比例），因此规划的每表 cache 行数与实际分配一致
- cache 至少要容纳一个 batch 估算的查表 Id 数（batch_size × 每个样本的 Id 数，序列特征按 `sequence_length`、多个特征共用一张表时相加），更小的比例不会被选用。这只是估算（按查表次数而非去重 Id 数），运行时每一步都会按实际的去重 Id 数检查，见下文“排障”
- 配置了 `cache_load_factor` 的表总是留在内存；`score_strategy: "CUSTOMIZED"` 的表不会放到 RocksDB（TorchEasyRec 没有 CUSTOMIZED 分数的生产者）
- 配置 rocksdb_config 时，留在内存的表只使用 CACHING 布局（GPU cache 加完整的主机表），不使用 HYBRID：HYBRID 每次加载都会重新划分 GPU 和主机两层，接近 `max_capacity` 的表会溢出哈希 bucket，恢复时丢失行
- 没有配置 rocksdb_config 时不会使用磁盘，也不需要安装 tzrec-ext；规划没有选择 RocksDB 时不会打开数据库
- 训练日志 `dynamicemb tables offloaded to RocksDB` 列出放到 RocksDB 的表、cache 比例和行数，以及规划出的 RocksDB 配置；`RocksDB disk pool` 给出磁盘估算的分项明细
- 同一份 checkpoint 可以在内存版与 RocksDB 版之间互相迁移，见下文“Checkpoint 与恢复”

### 淘汰策略与容量

- 支持 `STEP`、`TIMESTAMP`、`LFU` 和 `NO_EVICTION`。分数由 GPU cache 按 dynamicemb 的规则产生（STEP/TIMESTAMP 在访问时更新，LFU 累加访问次数，包括同一 batch 内的重复 Id 和 cache 命中），cache 淘汰时随行写回 RocksDB；从 RocksDB 读回的 LFU 行会带上本 batch 的频次，不会重复或漏计
- STEP/TIMESTAMP/LFU 表在每个训练步结束时按 `max_capacity`（每个分片的容量）做逻辑淘汰：删除分数最低的、不在 GPU cache 中的行（含 embedding 和优化器状态），分数相同时按 Id 排序。这是每个分片内的全局最低分，与内存版 dynamicemb 在哈希桶内淘汰最低分的结果不完全相同。每步之间可能短暂超出容量（最多一个 batch 的新 Id）
- 优先淘汰不在 GPU cache 中的行；只有这类行已全部淘汰仍超出容量时，才淘汰 cache 中分数最低的行，并同时从 cache 中删除，因此 cache 写回不会让已淘汰的 Id 复活。被淘汰的 Id 再次出现时按新 Id 重新准入、初始化，优化器状态重置
- 开启增量导出的 tombstone 时，逻辑淘汰的 Id 会像内存版一样发布 tombstone；GPU cache 的替换只是写回，不产生 tombstone。进程重启时尚未发布的淘汰 Id 不会保存（与内存版一致）
- `NO_EVICTION` 表不淘汰，与内存版一样随新 Id 持续增长。`max_capacity` 只作为磁盘规划的声明上限，并不强制；请依赖下文的运行时磁盘监控
- 使用 frequency_admission_strategy 时，频次 Counter 仍在显存中，并随 checkpoint 一起保存
- 每张表的数据以“模块路径.表名”作为持久命名空间，与放置方式和表的顺序无关

### 磁盘规划与监控

- 规划时每个 rank 上报 root_dir 所在磁盘池（NFS 类文件系统按服务器、本地盘按“主机+设备”，同一池的 rank 共享可用空间和 disk_budget_bytes）的可用空间和本任务已占用空间。每张 RocksDB 表占用的磁盘（逻辑行字节数 × 1.1 的 SST 系数，加一个 batch 的新 Id 和保存时 cache 写回的旧版本）作为动态规划的第三个维度；compaction 临时输出、MANIFEST/日志、恢复时最大一张表的临时 spill 文件和 16GiB 预留按池固定扣除
- 训练中每 100 步各 rank 交换一次采样（可用空间、本任务占用、SST 大小、compaction 积压），余量 = 可用空间 - 预留（配置了预算时再与“预算 - 占用”取小）。余量低于磁盘池容量的 5% 时在该步结束后停止训练：如果余量还够一次保存，先保存 checkpoint 再停止，否则直接停止（上一个完整的 checkpoint 不受影响），抛出 `DiskCapacityError`。不会为了腾空间删除 Id、快照或其它文件
- `estimate-pending-compaction-bytes` 表示待重写的数据量，不是需要额外预留的空间，日志中单独列出；如果它在连续多次检查中持续增长，日志会标记 compaction 不稳定

### Checkpoint 与恢复

- RocksDB 表与内存版 DynamicEmbedding 表使用完全相同的 checkpoint 格式：保存时 `DynamicEmbDump` 先把 GPU cache 写回，再从冻结的数据库视图（暂停后台 flush/compaction、持有快照，未落盘的 memtable 直接读取，不强制刷盘）按块流式写出 `model.ckpt-<step>/dynamicemb/` 下的 keys/values/scores/opt_values 文件，最后写 `_SUCCESS`。不会在训练盘上保留历史 checkpoint，也不会先在本地落一份完整导出
- model_dir 可以是 OSS 挂载目录（如 DLC 的 ossfs 数据集），文件按顺序流式写入，由 ossfs 分片上传。单文件大小受 ossfs 分片大小 × 10000 限制，超过时报 `File too large`，请调大 `upload_buffer_size`（见 FAQ）
- 开启 RocksDB 后，`--continue_train`、评估、预测和导出只会选择带 `_SUCCESS` 的 checkpoint，不完整的 checkpoint 会被忽略而不会被删除；只有恢复后的训练再次保存同一步时，才会替换该步不完整的目录（DynamicEmbDump 要求空目录）
- 恢复时先清空 GPU cache，再新建一个空的工作库导入 checkpoint 文件：每个线程顺序读取一个分片的一段（每个文件只打开一次，OSS 等远程挂载可以预读），只保留本 rank 的 id，排序后直接写成 SST 文件导入最底层，不经过 memtable 和 compaction。导入使用的主机内存由规划器给出（恢复时 block cache 和 memtable 还没有使用，内存版表已经分配），与表的大小无关：一张表放不下时，先扫描 key 采样划分 id 区间，第一个区间边读边转换，其余区间原样写到工作库旁的 `.spill` 目录再逐个导入，导入完即删除。上一次的工作库和崩溃遗留的数据不会被复用，因此只要 checkpoint 完整，本地盘丢失后也可以在其它节点恢复
- checkpoint 与放置方式无关：同一份 checkpoint 可以在内存版和 RocksDB 版之间恢复，也可以在资源变化后由规划器选出不同的放置继续训练。world size、维度或优化器类型的变化按 dynamicemb 自身的规则校验
- 内存表接近 `max_capacity` 时，改变 world size 恢复会让 key 落到新的哈希 bucket，装不下的 bucket 按淘汰策略丢弃其中分数最低的行（装填率约 97% 时，`bucket_capacity` 为 128 约丢失 2.3%，为 1024 约丢失 0.3%）。RocksDB 表没有 bucket，不受影响；world size 不变时恢复也不会丢行
- 评估、预测、导出时如果规划选择了 RocksDB，表同样从 checkpoint 文件加载到一个新的临时数据库（会占用与表大小相当的磁盘），进程结束（包括出错退出）时删除
- 保存耗时与表大小成正比（全量写出），训练在保存期间暂停，请据此设置 `save_checkpoints_steps`；日志 `RocksDB checkpoint` 给出本次保存的 cache 写回、导出行数和字节数、暂停后台任务的等待时间和读取的未落盘 memtable 大小，以及进程启动以来训练、发布的累计写回量、淘汰行数和 RocksDB flush/compaction 写入量
- 已知限制：恢复前框架会先用一个 batch 初始化优化器状态，此时 frequency admission 的 Counter 计数无法回滚

### 排障

- `tzrec-ext wheel`: 未安装 tzrec-ext（此时所有表留在内存，打印告警）
- `RocksDB tables need a local disk or an NFS-like filesystem`: root_dir 在 FUSE 或无法识别的文件系统上
- `Unable to find a plan for this model`: 显存、内存和磁盘都放不下；日志中有每个选项的显存/内存估算，磁盘不足时为 `RocksDB tables need ... bytes of disk`。可以换更大的磁盘、调大 disk_budget_bytes、降低 `max_capacity`，或减少 `HOST_MEMORY_RESERVE_PERCENT` / `STORAGE_RESERVE_PERCENT`
- `one step brings N unique ids to <表>, whose GPU cache holds M rows`: 一个 batch 中这张表的去重 Id 数超过了它的 GPU cache（dynamicemb 的 cache 放不下时会返回无效的行，所以在查表前报错）。请减小 batch_size，或给显存更多空间（减小 `STORAGE_RESERVE_PERCENT`）以选出更大的 cache 比例；序列或多值特征请确认 `sequence_length` 等配置反映了实际长度
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
