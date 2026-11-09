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

- **cache_load_factor**: （可选）显存规划的显式固定比例，取值 [0, 1]。不配置时由规划器自动搜索（见下文[显存规划](#%E6%98%BE%E5%AD%98%E8%A7%84%E5%88%92)）。配置后的语义：

  - `0`：HOST_ONLY，值与优化器状态全部放在主机内存
  - `1`：HBM_ONLY，值与优化器状态全部放在显存
  - `(0, 1)` 之间的值：同时生成 HYBRID 与 CACHING 两个候选，并固定在该比例；若该比例的 CACHING 缓存装不下一个批次的工作集，则只保留 HYBRID 候选

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

## 显存规划

训练启动时，规划器为每张 dynamicemb 表在四种内存布局中选择一种：

| 布局      | 值与优化器状态的位置                    | 显存占用                                             |
| --------- | --------------------------------------- | ---------------------------------------------------- |
| HOST_ONLY | 全部在主机内存                          | 仍占用哈希表元数据（key/score/digest 等，约 25B/行） |
| HYBRID    | 按比例划分在显存与主机内存              | 请求比例的值 + 双层元数据                            |
| CACHING   | 显存作缓存，主机内存保存全量 backing 表 | 缓存值 + 溢出区 + 全量元数据                         |
| HBM_ONLY  | 全部在显存                              | 全量值 + 元数据，无主机占用                          |

未显式配置 `cache_load_factor` 时，规划器自动搜索：

- HYBRID 候选：在 0.5 以下按几何间隔取 5 个点（含两端），再加上 0.5/0.6/0.7/0.8/0.9 的线性高档位。
- CACHING 候选：同样的低段几何网格，但搜索下限受批次工作集约束（见下）。
- HOST_ONLY（比例 0）与 HBM_ONLY（比例 1）各作为一个独立候选。

配置了 `embedding_constraints.unique_id_factor`（可通过
`tzrec.tools.add_pooling_factor_to_config` 从训练数据测量，多特征共享表时取最大值）时，
以 `q = ceil(5 * unique_id_factor * batch_size) / 表行数` 估计一个批次的去重工作集：

- CACHING 搜索下限为 `max(q, 0.1)`：缓存必须装得下工作集，装不下的候选会被剔除
  （比例达到或超过 1 时不提供部分缓存候选，HBM_ONLY 不受影响）
- HYBRID 搜索起点为 `min(q, 0.1)`：值分区可直接访问主机层，不受缓存驻留约束
- 未测量 unique_id_factor 时两个下限均为 0.1

规划行为说明：

- 布局会写入表的融合分组键：不同布局的表不会融合到同一份存储；同布局不同比例的表仍会融合，并保留各自的显存预算。
- 规划按 `max_capacity` 的满表口径估算（含按 bucket 对齐、缓存 1024 行 bucket 与三 bucket 溢出区、元数据与流水线缓冲），并非只算值存储。
- 与旧版本（0.1/0.2/.../1.0 线性网格、无 HOST_ONLY）相比，候选空间与最终规划都可能不同：小显存卡上原先规划失败的大表，现在可以通过 HOST_ONLY 或更低比例的 HYBRID 获得可行方案。
- 依赖：该规划需要基于 recsys-examples PR #534（`25d74cb`）及之后源码构建的 dynamicemb wheel（版本号形如 `0.1.0+<日期>.25d74cb.cu130`）；旧版 wheel（如 `13a8055`）不含按表预算与零预算修复，配对新版规划会产生错误的运行结果。安装时请确认 `dynamicemb.__version__`。

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
