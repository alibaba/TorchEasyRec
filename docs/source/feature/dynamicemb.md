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

- **max_capacity**: 最大的id数，Id数超过后会根据Id的驱逐策略进行淘汰。每个分片的建表容量会按有效 `bucket_capacity` 向上对齐，因此实际容量可能略高于配置值。显式 `init_capacity_per_rank` 使用同一 bucket 对齐，并限制在分片最大容量内。

- **cache_load_factor**: （可选）显存规划请求的值预算比例，取值 [0, 1]。不配置时由规划器自动搜索合适的比例。配置后的语义：

  - `0`：零显存值预算
  - `1`：满显存值预算
  - `(0, 1)` 之间的值：非缓存候选保留请求比例。缓存候选若按 bucket 取整后已覆盖整个分片（不含 overflow），则改为满预算的非缓存候选；自动搜索也只保留一个满预算候选。若请求预算取整后为 0 字节，则只保留零预算的非缓存候选。

  自动搜索时，同类型候选如果取整后的各层容量相同，只保留第一个候选的字节预算。规划日志中的 `cache_load_factor` 表示实际显存占比：部分缓存候选按正常驻留行数计算，不含 overflow。例如 4096 行的分片请求缓存 10%，取整后占 1024 行，日志显示 0.25，仍按 10% 下发值预算。

  最终存储布局由融合后的表组共同确定；零值预算仍有索引元数据等显存开销，混入非零预算组时还可能分配最小显存分区。

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
