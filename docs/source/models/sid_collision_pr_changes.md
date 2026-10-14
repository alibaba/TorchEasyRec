# PR #692：SID collision 工具变更说明

本文整理 [PR #692](https://github.com/alibaba/TorchEasyRec/pull/692) 的最终实现，便于逐项 review。
统计与验证记录截至 2026-10-13，代码版本为 `3dd4bf1`，分支为
`feat/sid_collision_iterative`，PR 保持 Draft。

对比基准是 PR 的 master 基线 `4458f1c`。文末另列出相对 9 月 30 日分支版本
`529f001` 的精简量；两个口径分别回答“这个 PR 带来什么”和“后续改造删掉了多少”。
本文是变更说明，运行指南见 [SID 冲突消解](sid_collision.md)。

后续已在本地增加 LargeList 类型调整：Parquet 的 SID 和 item ID 列表使用 int64 offsets，
工具尚未正式投入使用，不保留旧普通 List 产物迁移；CSV 和 ODPS 继续使用原有编码。
列名、列表内容和 manifest 格式不变，Arrow 列表 schema 会变化，不增加 offsets 容量防御。
此外已在本地增加目标 generation 目录检查，拒绝混入已有本地产物，不自动删除旧文件。
manifest 改为同目录临时文件写完并关闭后，通过 `os.replace` 原子发布。
候选扫描先匹配溢出 item ID，再解码命中行并筛选 prefix；候选需求方法改为 `requires_candidates()`。
补充 Uniform 的 torchrun 与全部统计字段断言，运行指南明确默认策略，README 增加 collision 文档入口。
这些修改尚未提交；下文的提交版本、CI 快照及行数统计仍对应 `3dd4bf1`。

## 1. 改造目标与最终结果

本次改造遵循的基本守则是：**针对 collision 工具，尽可能简化代码，删除过度防御，并使其具备人类可读性。**

最终变化包括：

1. 支持一次 `torchrun` 启动多个进程，共同处理同一份完整输入。
1. 提供 `candidate`、`iterative`、`uniform` 三种策略，替换原来的 `candidate`、`random` 组合。
1. 三种策略分别放入独立模块，复用公共的初始状态准备与结果统计。
1. 删除内部计算、分组和新增 item 写出中的人工 chunk 分块。
1. 简化候选读取、历史状态读取、产物路径管理和 manifest 写入。
1. 用明确的 SID、prefix、item 数量和数组索引命名替换含糊的 bucket、band、grouping 等名称。
1. 修复 NumPy 标量写入 manifest 时的 JSON 序列化问题。
1. 删除与旧实现绑定的测试，保留策略行为、追加一致性和分布式正确性的测试。

| 方面                     | master 基线                               | 本 PR 最终实现                          |
| ------------------------ | ----------------------------------------- | --------------------------------------- |
| 启动方式                 | 单进程，拒绝多 rank 启动                  | 支持 `torchrun --nproc_per_node`        |
| 分配策略                 | `candidate`、`random`                     | `candidate`、`iterative`、`uniform`     |
| 策略代码组织             | KNN、Random 位于公共核心文件              | 三种 resolver 各自位于独立模块          |
| 内部处理方式             | 多处 chunk 迭代、分块写出                 | 每个 rank 对保留的数据使用完整数组      |
| 占用状态传递             | 多个独立数组在调用之间传递                | 使用 `InitialCollisionState`            |
| SID 与 item 行的对应关系 | `CodebookItemGrouping`                    | `SidItemRowIndex`                       |
| 产物管理                 | `BundleLayout` 加 `Bundle` 与多种专用包装 | 单个 `Bundle`，统一按 artifact 读写     |
| manifest 写入            | Runner 包装后调用 `publish`               | Runner 直接调用 `Bundle.write_manifest` |

## 2. torchrun 多进程处理

### 2.1 一次启动，由 torchrun 创建进程

新的运行方式如下：

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    -m tzrec.tools.sid.resolve_sid_collisions \
    --input_path 'sid_predict/*.parquet' \
    --output_path sid_collision \
    --generation v1 \
    --codebook 8192,8192,8192 \
    --max_items_per_codebook 1 \
    --strategy uniform \
    --batch_size 100000
```

`--nproc_per_node=8` 由 torchrun 创建 8 个进程。collision 工具读取 `RANK` 和
`WORLD_SIZE`，决定当前进程负责哪些 SID prefixes。

用户给所有进程提供相同的输入和输出路径，外部提前拆成多个输入分区、分别启动多个
torchrun 的调度方式不再是运行前置步骤。PR 没有新增外部分区或调度脚本。

### 2.2 按完整 prefix 分配工作

prefix 是 SID 除最后一层之外的部分。例如 SID `[2, 3, 1]` 的 prefix 是 `[2, 3]`。
三种策略都只调整最后一层，因此同一个 prefix 内的全部 items 必须由同一 rank 处理。

工具将 prefix 编码空间划为连续区间。以 `--codebook 4,4,4`、4 个 rank 为例：

| rank | 负责的 prefix 编号 |
| ---- | ------------------ |
| 0    | 0～3               |
| 1    | 4～7               |
| 2    | 8～11              |
| 3    | 12～15             |

这些区间由 prefix 编码空间和进程数计算，不需要用户手动选择分区。
输入文件和读取批次的边界不会把同一个 prefix 的消解工作拆到不同 rank。

单层 SID 只有一个空 prefix，所以这类输入实际由 rank 0 处理，其他 rank 没有计算数据。

### 2.3 输入读取与空 rank

当前每个 rank 都扫描共享输入，只保留自己 prefix 区间内的数据。这能复用现有 reader，
但意味着增加进程数也会增加总扫描 I/O；前缀分布不均时，工作量也可能不均。

prefix 筛选统一放在 `_iter_batches` 中：

- `_get_prefix_mask` 始终返回 NumPy 布尔数组。
- 输入加载传入 `code_field`，后续循环不再重复计算同一筛选条件。
- 初始输入读取保留筛选后为空的批次，用于记录 item ID 的 Arrow 类型，让空 rank 能完成流程。
- 真正没有输入数据的任务仍然报错。
- 对越界 prefix 的筛选不会让错误数据悄悄消失，后续仍执行码本范围检查。

对于需要候选的策略，只要任一 rank 有溢出，相关 rank 就一致进入候选 reader 初始化。
没有溢出的 rank 可以返回空候选矩阵，但不能提前绕过 reader 初始化中的分布式同步。

### 2.4 产物与全局统计

每个 rank 写自己的 `part-*` 文件。写出完成后汇总各 rank 的统计和 artifact 行数，
最后仅 rank 0 写一个 `manifest.json`。

追加任务允许改变进程数；已有数据按本次的 prefix 区间重新读取和写出，原有 SID 分配不变。

## 3. 三种 collision 策略

| 策略参数    | 类与模块                                               | 是否读取候选 | 分配规则                                                | 容量不足时                                |
| ----------- | ------------------------------------------------------ | ------------ | ------------------------------------------------------- | ----------------------------------------- |
| `candidate` | `KnnCollisionResolver`，`knn_collision.py`             | 是           | 按稳定 item 顺序，选择第一个有容量的候选                | 保留原 SID，计为未解决                    |
| `iterative` | `IterativeCollisionResolver`，`iterative_collision.py` | 是           | 多轮同时提出申请，目标 SID 按候选顺序和 item 优先级接收 | 候选全部满后保留原 SID，计为未解决        |
| `uniform`   | `UniformCollisionResolver`，`uniform_collision.py`     | 否           | 在当前 prefix 内不断填充负载最低的末层 code             | 继续分配，最终超过容量的 items 计为未解决 |

三种策略都保留全部输入 rows，也都不改变 prefix。默认策略仍为 `candidate`。

### 3.1 Candidate：保留先到先分配的策略

KNN 使用模型提供的候选列表。一个溢出 item 按候选顺序尝试，遇到第一个仍有容量的
目标 SID 就放入，并更新该 SID 的 item 数量。

本 PR 将 KNN 实现和专属测试从公共核心中移入独立文件，直接使用 `occupancy.get`，
删除没有必要的局部方法别名。算法仍然是按稳定 item 顺序执行的 first-fit 分配。

### 3.2 Iterative：多轮候选申请与竞争

每一轮分为以下步骤：

1. 判断每个待处理 item 的候选 SID 是否还有容量。
1. 没有任何可用候选的 item 退出申请循环，最终保留原 SID。
1. 同一原始 SID 内，对每个候选列限制参与申请的 item 数量，避免大量相似候选产生过多申请。
1. 目标 SID 优先接受将它排在更靠前候选位置的 item，再按稳定 item 优先级打破并列。
1. 一个 item 可以有多条申请被接受，最终只采用候选顺序最靠前的一条。
1. 实际采用的申请更新占用数量；被接受但没有采用的申请不占用容量，其他 items 可在下一轮继续申请。

这与 Candidate 的 first-fit 规则不同，不能假设 Iterative 对所有数据都解决更多碰撞。
它优先考虑候选排序，在某些数据下也可能留下更多未解决 items。

Iterative 的主要精简包括：

- 不再重复筛选 `_initial_occupancy` 已经限定好的 prefix 数据。
- 使用相邻 offsets 的差计算每个 prefix 的初始 SID 数量，减少一次范围搜索。
- 用 `np.nonzero` 直接获取申请矩阵的行列索引，删除“展平、再用 divmod 还原”的步骤。
- 明确区分原输入 row、溢出 item 索引、局部 SID 索引和 proposal 索引。
- 保留 `proposal` 命名，明确区分“目标接受申请”和“item 最终选择申请”。

### 3.3 Uniform：按负载均匀填充

Uniform 不依赖模型候选，使用当前 prefix 内所有末层 codes。相同负载时，按 prefix 的
稳定哈希得到的轮转顺序选择 code，避免所有 prefix 都优先使用同一个末层 code。

实现分成两条路径：

- 空 SID 数量足够：直接为每个溢出 item 取一个完全空的末层 code，新增 item 的 slot 为 1。
- 空 SID 数量不够：针对该 prefix 调用 `_water_fill`，计算提升到某个负载层级需要放入多少 items，再分配剩余 items。

`_get_empty_last_codes` 查找的是完全没有 item 的 SID，不是仍有剩余容量的部分占用 SID。
`_water_fill` 独立处理每个 prefix，通过 NumPy 数组和负载层级计算实现逐次选择最低负载的效果。

本 PR 没有改成逐 item 的 Python 堆轮询。每个 prefix 独立计算的方式保留下来，减少绕弯的
索引准备，同时继续使用数组运算完成分配。

Uniform 不再拒绝传入的候选矩阵，而是直接忽略。候选是否需要加载，由各 resolver 的
`is_requires_candidates()` 决定：KNN、Iterative 返回 `True`，Uniform 返回 `False`。
Runner 不再通过两处 `strategy != "uniform"` 判断来识别这项能力。

## 4. 输入与追加模式的行为

### 4.1 输入格式

输入至少包含 `item_id` 和 `codes`。Candidate、Iterative 有溢出时还需要
`candidate_codes`，Uniform 不需要候选列。

`candidate_codes` 表示每个 item 的多个完整候选 SID，按候选优先级展开成
`topk × SID 层数` 个整数。消解时取每个候选的最后一层，prefix 仍由原 SID 决定。

例如三层 SID 的两个候选：

```text
候选 SID：[[2, 3, 1], [2, 3, 5]]
输入列表：[2, 3, 1, 2, 3, 5]
用于分配的末层候选：[1, 5]
```

Parquet 保留整数列表；CSV 使用逗号分隔的字符串。ODPS 继续复用仓库已有 reader 和 writer。
本 PR 没有为算法层引入文件读取依赖。

### 4.2 已有 items 与本次输入 items

追加模式下，已有 items 来自 `--from_generation` 指定的上一代产物；本次输入只包含要追加的 items。

已有 items 的 SID 和 slot 不重新分配。新的 item 从该 SID 已有数量之后继续编号：

```text
某个 SID 已有 3 个 items，slot 为 1、2、3。
新增到该 SID 的 items 依次使用 slot 4、5、6。
```

代码里的 `input_item_counts` 统计本次输入，`existing_item_counts` 统计已有数量。
这两个概念与“是否创建了一个新 SID”无关：新增 item 可以进入已有 SID。

已经超过容量的历史 SID 也保留完整数量，不会将旧 slot 重新分配给新增 item。
本次输入中出现已经发布的 item ID，会报错。

### 4.3 追加产物

| 产物                 | 内容                                                            |
| -------------------- | --------------------------------------------------------------- |
| `item_to_sid`        | 上一代已有 items 加本次输入 items 的完整映射                    |
| `sid_to_items`       | 追加后的完整 SID 与 item 列表                                   |
| `delta_item_to_sid`  | 仅本次输入 items 的最终映射                                     |
| `delta_sid_to_items` | 本次获得输入 items 的完整 SID 列表，包含这些 SID 下的已有 items |
| `manifest.json`      | 本代产物位置、格式、行数、码本及代际关联                        |

`delta_sid_to_items` 的一行可以包含旧 item 和新 item，适合下游按 SID 做幂等覆盖。
它并不是只保存新增 item IDs 的列表。

追加使用的 `--codebook` 必须与上一代一致。若容量配置改变，只记录警告并保留旧 items 的占用。
输入与历史 item ID 类型保持一致，已有 SID 列表的排序和跨产物数量一致性仍会检查。

## 5. 删除内部 chunk 分块

本次清理覆盖计算、SID 行索引构建和新增数据写出，而不只是删除某个函数名字。

具体包括：

- 删除旧 first-fit 核心中的分块控制流程。
- 删除构建 SID 行索引时的 `row_chunks` 生成器和 chunk 对齐、计数校验。
- 删除 `_group_chunk_bounds` 以及 SID 输出的人工分块路径。
- 删除 `_ITEM_TO_SID_WRITE_ROWS`、`_SID_TO_ITEMS_WRITE_SIZE` 等内部写出大小常量。
- 新增 item 的 `item_to_sid` 写出一次构造对应数组，SID 分组写出直接使用完整行索引。
- 最终实现中不再保留开发阶段的 `_bucket_chunks`。

需要区分三种概念：

| 概念                    | 当前处理方式                          |
| ----------------------- | ------------------------------------- |
| 工具内部的人工 chunk    | 已删除                                |
| reader 的 batch         | 保留，`--batch_size` 控制读文件的批次 |
| torchrun 的 prefix 区间 | 保留，决定每个 rank 负责的数据范围    |

追加时，历史产物仍按 reader 批次读取、复制和合并；整个历史语料并没有改成一次性全部载入。

`pa.chunked_array` 和 `combine_chunks` 仍用于 Arrow 数组拼接与重排。这里的 chunk 是
Arrow 的存储表示，不是 collision 的人工任务分区或调度逻辑。

## 6. 公共状态与结果构造的精简

三种 resolver 都需要读取初始占用、处理无溢出的情况并生成统计。公共部分现在集中在基类中：

| 定义                     | 作用                                                                |
| ------------------------ | ------------------------------------------------------------------- |
| `_initial_occupancy`     | 合并已有数量和本次保留的 items，得到溢出 prefixes 内的初始 SID 占用 |
| `InitialCollisionState`  | 一起传递 prefix 掩码、初始 SID keys 和 item 数量                    |
| `_prepare_collision`     | 准备初始状态，无溢出时直接生成结果                                  |
| `_build_result`          | 合并受影响和未受影响的统计，统一构造结果                            |
| `is_requires_candidates` | 说明当前策略是否需要读取候选                                        |

每种策略的开头统一为：

```python
initial_state, result = self._prepare_collision(plan, collect_sid_metadata)
if result is not None:
    return result
```

原来多处重复的初始化和无溢出返回逻辑因此不再分别维护。

SID 与本次输入 rows 的对应关系使用 `SidItemRowIndex` 表达，其内容是行索引，而不是
item ID 本身。原始与消解后 SID 的两个构建函数仍放在 `collision.py`，由 Runner 写出时调用，
本轮没有继续移动它们。

对于已明确要求 NumPy 数组的 plan 输入，删除重复的 `np.asarray` 转换。
核心文件中没有实际用途的 `from __future__ import annotations` 也已删除。
这些调整没有删除必要的类、函数 docstring。

## 7. 命名与接口对齐

### 7.1 相对 master 的主要命名变化

| 原名称                         | 当前名称                                              | 更明确的含义                          |
| ------------------------------ | ----------------------------------------------------- | ------------------------------------- |
| `layer_sizes`                  | `codebook`                                            | 各层码本大小，与模型和 CLI 的概念对齐 |
| `resolution_config`            | `collision_config`                                    | collision 算法配置                    |
| `band` / `sid_band_ids`        | `prefix` / `CollisionResolutionConfig.prefix_ids`     | SID 除最后一层之外的前缀              |
| `sid_bucket_keys`              | `CollisionResolutionConfig.convert_codes_to_sid_keys` | 将 SID codes 转换为混合进制整数 key   |
| `sid_offset_codes`             | `CollisionResolutionConfig.offset_codes`              | 为各层 code 加词表偏移                |
| `original_last_codes`          | `origin_last_codes`                                   | 本次输入的原始末层 code               |
| `origin_bucket_indices`        | `origin_sid_key_indices`                              | 每个输入 row 对应哪个唯一 SID         |
| `bucket_keys`                  | `sid_keys`                                            | SID 的整数编码                        |
| plan 的 `bucket_counts`        | `input_item_counts`                                   | 每个 SID 的本次输入 item 数量         |
| prior 的 `bucket_counts`       | `item_counts`                                         | 每个已有 SID 的 item 数量             |
| `prior_bucket_counts`          | `existing_item_counts`                                | 当前 SID 对应的已有数量               |
| `CodebookItemGrouping`         | `SidItemRowIndex`                                     | 按 SID 和 slot 组织输入 row 索引      |
| `row_order`                    | `item_row_indices`                                    | 排列后的输入 row 索引                 |
| `collect_grouping`             | `collect_sid_metadata`                                | 是否收集最终 SID keys 和数量          |
| `grouping_collected`           | `sid_metadata_collected`                              | 是否已经收集这些元数据                |
| `lookup_sorted`                | `find_in_sorted_array`                                | 在已有升序数组中查找，不修改数组      |
| `_codes_matrix`                | `_convert_codes_to_numpy`                             | 将 Arrow 输入转换为 NumPy codes 数组  |
| `_load_prior_state` 等读取包装 | `_load_existing_sid_state`                            | 加载追加所需的已有 SID 占用和数量信息 |
| `Bundle.publish`               | `Bundle.write_manifest`                               | 构造并写入 manifest，并记录日志       |

SID keys 的编码规则没有改变。例如 `codebook=(2, 3, 4)`、SID `[1, 2, 3]`：

```text
prefix_id = 1 × 3 + 2 = 5
sid_key = 5 × 4 + 3 = 23
```

### 7.2 进一步评审后的命名

评审过程中还将同值区间辅助函数命名为 `get_same_value_start_indices` 和
`get_same_value_positions`，并在 docstring 中给出输入输出示例。

Iterative 明确区分 `proposal_item_indices`、`proposal_sid_indices`、
`proposal_candidate_positions` 和 `selected_proposal_indices`，减少同一个 `rows`、
`keys`、`positions` 在不同数组空间中反复使用造成的误解。

Uniform 的目标末层 code 和 slot 使用 `target_*`，空 code 的索引使用 `empty_*`。
两个相邻范围位置继续用 offsets 表达，没有引入另一套 boundaries 命名。

`quality.py` 中同步替换 bucket 相关变量和说明文字为 SID 表述，Gini、entropy 等指标算法没有改变。

## 8. Bundle 与 manifest 写入

原来的 `BundleLayout` 和 `Bundle` 被合并，路径、已有 manifest 和产物行数由同一对象管理。
按产物种类提供的多层专用路径、计数包装，简化为 `read_path(artifact)`、
`write_path(artifact)` 和 `rows[artifact]`。

Runner 的 `_write_manifest` 只负责转发参数和记录日志，已经删除。
现在 `run()` 直接调用 `self._bundle.write_manifest(...)`。

`Bundle.write_manifest` 负责：

1. 合并本次观察到的最大 SID 数量与历史最大值。
1. 为每个实际写出的 artifact 记录格式、位置和行数。
1. 写入 `manifest.json`。
1. 记录 bundle UUID 和 manifest 路径的日志。

该方法返回 `None`，调用方不再接收只用于日志的路径。
manifest 仍在所有产物写出和 rank 汇总之后写入，文件出现代表本代已经完成。

存储格式仍保持如下规则：

- `item_to_sid` 和 `delta_item_to_sid` 跟随所选 writer，可以是 CSV、Parquet 或 ODPS。
- `sid_to_items` 和 `delta_sid_to_items` 固定为 Parquet，保留原生 item ID 列表。
- manifest 固定为 JSON，与 SID 列表产物一起位于文件系统的输出根目录。

由于 SID 列表产物固定为 Arrow 原生列表，删除了 `_decode_grouped_item_ids` 中间拆解包装，
不再为列表的展开值与长度额外维护另一套数据结构。

## 9. NumPy 2.4 的 JSON 序列化修复

追加合并时，需要统计本次创建了多少个以前不存在的 SID。
验证环境中的 NumPy 2.4.6 返回 NumPy `int64`，直接将结果记入 manifest 的行数会导致
JSON 编码报错。

修复方式是在统计写入 `Bundle.rows` 前，将数量转换为 Python `int`：

```python
existing_sid_count + int(np.count_nonzero(~shared))
```

实际数量不变，只统一序列化边界的数据类型。该问题影响追加产物的 manifest，
不涉及 collision 的目标 SID 分配算法。

## 10. 全局统计与 rate_only

`_gather_stats` 收集各 rank 的统计字典和 artifact 行数，并在每个 rank 上生成相同的全局结果。
产物行数按 artifact 求和，rank 0 使用汇总后的数量写 manifest。

| 当前字段                    | 含义                                          | 跨 rank 汇总 |
| --------------------------- | --------------------------------------------- | ------------ |
| `total_items`               | 本次输入的 item 数量                          | 求和         |
| `raw_colliding_sid_count`   | 本次输入涉及的 SID 在分配前超过容量的数量     | 求和         |
| `final_colliding_sid_count` | 本次受影响范围内，分配后仍超过容量的 SID 数量 | 求和         |
| `relocated_count`           | 最终分配到容量以内的溢出 item 数量            | 求和         |
| `unresolved_count`          | 最终 slot 仍超过容量的溢出 item 数量          | 求和         |
| `max_final_items_per_sid`   | 本次统计范围内，最终单个 SID 的最大 item 数量 | 取最大值     |

追加模式的 `total_items` 不包含上一代已有 items；SID 的容量判断会计入已有数量。
已有语料中与本次输入无关的 prefix 不会全部载入 plan，manifest 的历史最大值另由 Bundle 保留。

统计聚合从嵌套字典推导式改成普通循环和 `if/else`，最后仍用
`CollisionResolutionStats(**merged_stats)` 构造对象。

`--rate_only` 仍执行计划、候选读取和消解，只跳过产物与 manifest 写入。
它适合比较策略结果，但不会跳过分配阶段的计算和内存占用。

## 11. 兼容性与需要调整的调用

| 使用方式                             | 需要关注的变化                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 使用 CLI 的 `candidate` 策略         | 仍是默认策略，候选列格式不变                                                                         |
| 使用 CLI 的 `random` 策略            | 该策略已删除，可按需求选择 `uniform`；两者分配结果不等价                                             |
| 传入 `--random_num_candidates`       | 参数已删除                                                                                           |
| 使用外部分区和多任务调度             | 改为一次 torchrun，共享完整输入，使用原生进程数参数                                                  |
| 直接构造算法配置                     | `layer_sizes` 改为 `codebook`                                                                        |
| 直接引用旧核心中的 KNN 类            | 从 `tzrec.utils.sid.knn_collision` 导入                                                              |
| 直接引用分组类或构建函数             | 改用 `SidItemRowIndex`、`build_origin_sid_item_row_index`、`build_resolved_sid_item_row_index`       |
| 使用 `collect_grouping` 或结果旧字段 | 更新为 `collect_sid_metadata` 和新的 SID 字段名                                                      |
| 调用 `Bundle.publish` 并依赖返回路径 | 改为 `write_manifest`；该方法不返回路径                                                              |
| 解析统计对象字段                     | 旧 `raw_collision_buckets`、`final_collision_buckets`、`max_final_bucket_size` 对应新的 SID 统计名称 |

已有产物中的 `item_id`、`origin_codebook`、`codebook`、`offset_codebook`、`index`、`itemids`
等列名保持不变，SID 编码和 offset_codebook 计算规则保持不变。
改变的是 Python 接口与统计名称，不能将其误认为输出表列的整体重命名。

## 12. 内存和性能影响

删除 chunk 简化了控制流程，同时提高了某些阶段一次保留的临时数组大小。
每个 rank 的候选矩阵约占：

```text
溢出 item 数 × topk × 8 字节
```

Iterative 的稠密 SID item 计数表约占：

```text
溢出 prefix 数 × 最后一层码本大小 × 8 字节
```

还需要计入输入 codes、item IDs、排序索引、申请数组、消解结果和 Arrow 写出缓冲。
`--batch_size` 只控制读文件的批次，不约束完整消解数组。

增加 rank 数可分散每个进程持有的数据，但整机内存应合计所有进程，并考虑重复扫描和批次缓冲。
不能因为机器有 256 GiB，就仅根据样本数量判断所有策略一定能容纳。

已有合成数据实验测到，2.55 亿 items 在一次性写出阶段约使用 45.10 GiB RSS，
原分块写出约 26.71 GiB。实验仅包含特定输入和写出阶段，不包含完整 plan、候选矩阵与消解临时数组，
不能作为亿级真实输入的全流程峰值承诺。
实验条件和其他规模结果见 [运行指南的内存部分](sid_collision.md#%E5%86%85%E5%AD%98)。

本次没有将分配阶段的局部性能比较当成整个工具的速度结论，也没有宣称所有真实数据都能按进程数线性加速。

## 13. 测试清理与验证结果

### 13.1 删除或合并的测试

与已删除实现绑定的测试不再保留，主要包括：

- Random 候选生成、随机分配及专用配置检查。
- `test_resolution_chunking_preserves_greedy_results` 等核心分块行为测试。
- `test_sid_to_items_chunks_by_item_count_without_splitting_buckets`。
- `test_map_writer_chunks_by_row_count`。
- 异常路径中强制关闭 writer 的实现细节测试；当前流程明确关闭成功写出的 writer，失败任务不会发布 manifest。
- 与追加链测试重复的独立“旧 item 不移动”用例，以及重复执行相同输入的确定性检查。

KNN 专属测试移到 `knn_collision_test.py`，并不代表对应覆盖被删除。
Iterative 与 Uniform 各有独立测试文件。旧测试名称里的 `publish`、`groupings` 已更新为
`write_manifest`、`sid_item_row_indices`。

最近检查发现的 `test_item_to_sid_artifact_may_be_odps` 与参数化测试有重复覆盖，
目前仍保留在代码中；本轮只确认了可精简点，没有执行删除。

### 13.2 保留的关键覆盖

- 三种策略的分配结果、容量判断、slot 连续性与未解决 items。
- 输入顺序变化后的确定性。
- Iterative 的候选优先级、同源 SID 请求限制、下一轮重新竞争和完整 topk 消费。
- Uniform 与逐 item 最低负载参考实现的对比。
- 已有 SID 超容量、历史专有 SID、只给本次输入建立行索引等追加边界。
- CSV、Parquet 格式组合、原生列表与 item ID 类型保持。
- SID key 范围、已有产物严格递增及跨产物数量一致性。
- torchrun 与单进程结果一致、空 rank、rate_only 和追加时改变 rank 数。
- manifest 与实际 artifact 行数一致、代际关联和历史最大 SID 数量保持。

### 13.3 指定远端的验证

使用现有 PyTorch 2.14.0 环境、NumPy 2.4.6，从仓库根目录设置 `PYTHONPATH=.` 运行：

```bash
python -m unittest \
    tzrec.utils.sid.collision_test \
    tzrec.utils.sid.knn_collision_test \
    tzrec.utils.sid.iterative_collision_test \
    tzrec.utils.sid.uniform_collision_test \
    tzrec.utils.sid.bundle_test \
    tzrec.tools.sid.resolve_sid_collisions_test
```

结果为 **109 项测试全部通过，测试执行耗时 42.469 秒**，包含 torchrun 用例。
这是 collision 相关测试集合，不是整个 TorchEasyRec 的完整测试套件。

`pyrefly check` 为 **0 errors**，仍有工具配置抑制项和 warnings；结果不表示完全没有告警。
安装仓库指定的 Ruff 0.16.10 后，相关 Python 文件的 lint 和 format 检查通过。

测试通过后只做了格式修正，对修改前后进行 AST 等价检查，确认没有逻辑变化，未重复运行整组测试。

### 13.4 GitHub CI 快照

上一提交 `d1b8447` 的 Code Style CI 因 Ruff 格式化修改了 5 个文件而失败。
修正换行和空行后，最新提交 `3dd4bf1` 的查询快照为：

| 检查                                                                                               | 查询时状态 |
| -------------------------------------------------------------------------------------------------- | ---------- |
| [Code Style CI](https://github.com/alibaba/TorchEasyRec/actions/runs/38375738180/job/115223510932) | 通过       |
| [PyTyping CI](https://github.com/alibaba/TorchEasyRec/actions/runs/38375738066/job/115223524323)   | 通过       |
| [Build Test CI](https://github.com/alibaba/TorchEasyRec/actions/runs/38375738055/job/115223510356) | 通过       |
| Unit Test CI / CPU / H20 / PPU                                                                     | 仍在运行   |

远端完整 `pre-commit run -a` 曾因 GitHub TLS 握手错误无法初始化 hook；本地不将其记为全部通过。
最新 GitHub Code Style CI 已完成完整流程并通过。其他远端 CI lane 的最终结论应查看 PR 的后续状态。

## 14. 文件范围与代码行数

### 14.1 PR 相对 master

以下数字来自 `git diff 4458f1c...3dd4bf1`，共 15 个文件；不包含本说明文档。

| 类别                   | 增加 | 删除 | 净变化 |
| ---------------------- | ---- | ---- | ------ |
| 生产 Python 代码       | 1761 | 1595 | +166   |
| 测试 Python 代码       | 1362 | 1356 | +6     |
| 原有运行文档与目录入口 | 101  | 0    | +101   |
| 合计                   | 3224 | 2951 | +273   |

虽然原有核心、Runner 和测试大量缩减，新增两种策略及对应测试，使整个 PR 相对 master 的总行数仍增加。
独立模块之间的代码移动也会同时记为删除和新增，不能只看增加行数判断新增了多少逻辑。

### 14.2 相对 9 月 30 日分支版本

在本 PR 相同的 15 个文件范围内，对比 `529f001` 与 `3dd4bf1`：

| 类别               | 增加 | 删除 | 净变化 |
| ------------------ | ---- | ---- | ------ |
| 生产 Python 代码   | 1464 | 1954 | -490   |
| 测试 Python 代码   | 985  | 1680 | -695   |
| 运行文档与目录入口 | 101  | 0    | +101   |
| 合计               | 2550 | 3634 | -1084  |

这一口径更接近本轮逐项精简的结果：在三种策略已经存在的基础上，生产代码减少 490 行，
测试代码减少 695 行，同时补充运行文档。
该统计限定在 SID PR 文件范围内，不包含同步 upstream master 带来的其他模块变化。

### 14.3 主要文件职责

| 文件                                           | 本 PR 中的职责或变化                                        |
| ---------------------------------------------- | ----------------------------------------------------------- |
| `tzrec/tools/sid/resolve_sid_collisions.py`    | 输入读取、rank 筛选、候选加载、追加合并、产物写出与全局统计 |
| `tzrec/utils/sid/collision.py`                 | 配置、plan、公共初始状态、结果统计及 SID 输入行索引         |
| `tzrec/utils/sid/knn_collision.py`             | 独立的候选 first-fit 策略                                   |
| `tzrec/utils/sid/iterative_collision.py`       | 多轮同步候选竞争                                            |
| `tzrec/utils/sid/uniform_collision.py`         | 空 code 分配和按 prefix 的最低负载填充                      |
| `tzrec/utils/sid/bundle.py`                    | 统一产物路径、历史 manifest 和本代 manifest 写入            |
| `tzrec/utils/sid/quality.py`                   | SID 相关命名与说明对齐，指标算法不变                        |
| 上述 collision、Bundle、Runner 的 6 个测试文件 | 策略、追加、格式、分布式与 manifest 回归覆盖                |
| `docs/source/models/sid_collision.md`          | 中文运行指南、torchrun、追加模式与内存说明                  |
| `docs/source/models/sid_model.rst`             | 将 collision 运行指南加入文档目录                           |
