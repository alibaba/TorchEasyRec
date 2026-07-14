# SID 碰撞处理：Fresh 与 Append 重构方案

> 状态：设计方案；不包含代码实现变更。

## 1. 范围与术语

重构后应显式支持两种模式：

- **Fresh**（需求中的“纯新增”）：只读取一个预测文件，完整执行当前碰撞处理算法。没有历史状态时，分配结果、索引、fallback 行为、统计和输出顺序都必须与当前实现保持一致。
- **Append**：读取一个新预测文件，以及已有的 `original_groups` 和 `resolved_groups` 快照。历史分配不可变，只处理并追加新 item。

`original_groups` 保存“原始 SID -> 历史 item IDs”；`resolved_groups` 保存“最终 SID -> item IDs”。item 在 resolved bucket 中的位置具有业务含义：第 `k` 个位置对应当前 map 中从 1 开始的索引 `k`。

本次实际检查的是工作区中的 `experiments/sid_collision/sample.parquet` 和 `sid_collision/{original_groups,resolved_groups}`；当前环境中不存在字面量 `/mnt/data/...` 路径。sample 包含 500 个唯一 item，而且正是现有快照的数据来源。因此，若把该 sample 作为 append 新批次，正确行为应是重复 ID 校验失败，而不是再次追加 500 份数据。

## 2. 行为契约

| 属性                | Fresh                 | Append                        |
| ------------------- | --------------------- | ----------------------------- |
| 历史 group 输入     | 禁止                  | 必须同时提供两个快照          |
| 历史最终 SID 与索引 | 不适用                | 永不改变                      |
| 容量初始值          | 空                    | 来自 `resolved_groups` 的计数 |
| 原始归属            | 当前输入              | 历史 groups 加新输入          |
| Candidate 处理范围  | 当前批次全部 overflow | 只处理新 item 的 overflow     |
| Map 输出            | 当前输入              | 只输出新批次增量              |
| Group 输出          | 完整快照              | 新的完整快照                  |

Append 有意地**不等价于**对“历史 + 新数据”重新执行 Fresh。原因是历史 candidate 列表不可用、历史分配必须稳定，而且旧 item 的优先级高于新 item。因此，append 结果必然与批次历史有关。

重复 ID 的默认策略应为 `error`：新批次内部必须唯一，并且不得与历史 `original_groups` 相交。不能默认静默跳过或 upsert，因为 group 状态无法证明重复 item 的 SID 和 candidate 数据完全一致。

Append 输出路径必须与两个输入快照不同。`rate_only` 仍应完成相同的校验和规划，并报告新增与合并后的统计，但不发布 group 或 map 数据。

## 3. 稳定的持久化标识

当前 dense band ID 由“本批次实际出现的 prefix”生成，适合单次内存分组，却不能标识持久化 bucket。例如，prefix `B` 在一个批次中可能是 band 1，在另一个只有 `B` 的批次中却变成 band 0。

应引入 `SidKeyCodec`：先按 `layer_sizes` 校验每一层 code，再把完整 SID 编码为稳定且有溢出检查的 mixed-radix `int64` key。最后一层的步长为 1，因此可以得到不依赖当前批次的 band key。如果整个 key 空间无法放入 `int64`，应拒绝配置；若未来确有需要，再使用结构化多列 key 作为替代方案。

还应持久化带版本的 state manifest。仅从 group 文件无法安全恢复 capacity 或 layer sizes：本次检查到 sample 生成器使用 `(64, 64, 64)`，而本地调用脚本配置为 `(256, 256, 256)`。manifest 至少应包含：

- state 格式、SID key codec、排序/hash 和分配算法版本；
- `layer_sizes`、capacity、strategy、fallback policy 和 random-count 配置；
- Arrow item-ID 类型或 schema fingerprint；
- original/resolved item 数、父 state 版本，以及 map 范围（`delta`）；
- 各 artifact 的位置、格式和完成标记。

旧快照需要进行一次 bootstrap：显式提供配置，完整校验后生成 manifest，但不改变任何历史分配。

## 4. Append 规划算法

1. **校验 state。** 读取 manifest，检查配置与 item-ID 类型兼容性；拒绝 CSV/Parquet 混合目录；确认输入与输出路径不重叠。
1. **读取紧凑 occupancy。** 从 `resolved_groups` 读取并校验唯一、已排序的 SID key 和 bucket 内容；规划阶段只保留 canonical SID key 与计数。占用容量必须来自 `resolved_groups`，而不是 `original_groups`。
1. **校验新 ID。** 检查新批次内部唯一性；对新 ID 排序一次，再流式扫描历史 `original_groups.itemids` 并查找交集，避免为全部历史 item 创建 Python set。
1. **对新 origin 排序。** 在每个新 original-SID bucket 内继续使用当前确定性的 hash rank。
1. **在 origin 接纳新 item。** 对 origin bucket `s`，计算 `free(s) = max(capacity - existing_resolved_count(s), 0)`。前 `free(s)` 个新 item 留在 origin，其绝对索引从 `existing_resolved_count(s) + 1` 开始；其余标记为 overflow。
1. **先预占全部 origin。** 处理任何 candidate 之前，先把所有被 origin 接纳的新 item 加入 occupancy。这样才能保留当前 Fresh 中“直接归属优先于 candidate 迁入”的规则。
1. **处理新 overflow。** 按确定性顺序，只对新 overflow 执行当前 first-fit 或 random 策略。candidate occupancy 的初值为历史 resolved 计数加上全部新 origin 接纳项；candidate 仍须限制在同一个 canonical prefix band 中。
1. **应用 fallback。** `error` 在发布前终止；`drop` 不把 unresolved 新 item 写入 resolved groups 和 map，但仍保留其 original membership；`keep_original` 把它追加回 origin，即使 bucket 因此超过 capacity。
1. **构建增量结果。** 保存每个新 item 的最终 SID 和绝对的、从 1 开始的索引；不重建或重写旧 map 行。

当 base occupancy 为空时，通用 planner 的逻辑结果必须与当前 Fresh 实现完全一致。

## 5. 快照合并与输出规则

第一阶段应输出完整的新快照：

- **Original 快照：** 每个 original SID 一行；旧 item 列表必须保持为前缀，新 ID 按当前 bucket 内确定性顺序追加。
- **Resolved 快照：** 每个 final SID 一行；旧列表必须原样保持为前缀，新 ID 按已分配的绝对索引顺序追加。这样每个新 map index 都可以直接在合并后的列表中验证。
- **Map：** 只输出当前新输入的记录，保持 `output_path` 表示“本次输入的映射”这一现有语义。

继续复用 TorchEasyRec 已有 reader/writer，包括 CSV writer。可以增加一个小型、面向 state 的适配模块，负责 manifest 校验、CSV/Parquet group 统一解码、流式合并和发布；没有必要再建立一套通用 I/O 框架。

CSV 会把整个 `itemids` group 放进一个类似 JSON 的字段，因此热点 bucket 可能产生超过 reader block size 的单行。大规模 state 应优先选择 Parquet 或 ODPS。写入 chunk 可以控制 table/CSV 批次内存，但在不修改 schema 的前提下，无法拆分单个超大 group。

当前 writer 会彼此独立地覆盖输出，因此不能原地更新 state。应先把所有 artifact 写入一个新版本，关闭并校验后，最后发布 manifest 或 current-version 指针。manifest 中记录父版本，并使用锁或 compare-and-swap 校验，避免并发 append 互相覆盖。

如果 1 亿 item 场景下频繁重写完整快照成本过高，可以在后续阶段引入显式的“base + 有序 delta + 定期 compaction”格式。当前“一 bucket 一行”的快照契约下，不能直接输出重复 codebook 行。

## 6. ODPS Append 能力检查

**结论（验证日期：2026-07-14）：MaxCompute 支持物理行追加，但 TorchEasyRec 当前 writer 没有暴露该能力；而且对于现有 grouped-state schema，直接物理追加是错误的。**

| 层次                      | Append 能力                                                           | 对本设计的影响                                       |
| ------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------- |
| MaxCompute SQL            | `INSERT INTO` 可追加到普通表或静态分区；`INSERT OVERWRITE` 会替换目标 | 平台具备追加能力，但受表类型等限制                   |
| PyODPS / Storage API      | non-overwrite write 可以追加；`overwrite=True` 表示替换               | 底层 client 能表达两种模式                           |
| TorchEasyRec `OdpsWriter` | 固定使用 `TableBatchWriteRequest(..., overwrite=True)`                | 每次 writer 运行都会替换目标表或分区                 |
| SID group 快照            | 每个 `codebook` 必须只有一行，且 `itemids` 是完整列表                 | 对已存在 bucket 追加行只会产生重复行，不会更新原列表 |

平台能力可参见 [MaxCompute `INSERT INTO` / `INSERT OVERWRITE`](https://www.alibabacloud.com/help/en/maxcompute/user-guide/insert-or-update-data-into-a-table-or-a-static-partition)、[TableTunnel overwrite 语义](https://www.alibabacloud.com/help/en/maxcompute/user-guide/tabletunnel)、[PyODPS table write](https://pyodps.readthedocs.io/en/latest/base-tables.html#write-data-to-tables) 和 [PyODPS Storage API write](https://pyodps.readthedocs.io/en/stable/base-storage-api-v2.html#writing-data)。MaxCompute 还明确说明：`INSERT INTO` 不支持 clustered table，并且对同一目标的并发 INSERT 没有 table lock 保护；其 [ACID 文档](https://www.alibabacloud.com/help/en/maxcompute/product-overview/acid-semantics) 也没有提供跨多个 SID 输出表的统一事务。

仓库检查显示，`tzrec/datasets/odps_dataset.py:762-764` 始终传入 `overwrite=True`；`OdpsWriter.__init__` 和 collision 流程都没有 append 参数。同一个 writer 内多次调用 `write()` 只是在同一个 overwrite session 内累积 batch。现有 ODPS 测试只验证一次 writer session，没有验证两次独立运行后是否保留旧数据。仓库固定使用 PyODPS 0.12.5.1；该版本的 [`TableBatchWriteRequest`](https://raw.githubusercontent.com/aliyun/aliyun-odps-python-sdk/v0.12.5.1/odps/apis/storage_api/storage_api.py) 接受 overwrite flag，但 TorchEasyRec 显式传入的 `True` 阻止了 append。

例如，ODPS 物理 append 会把 `(A, [1, 2])` 加 `(A, [3])` 变成两行 codebook `A`，而不是 `(A, [1, 2, 3])`。因此，即使数据库追加成功，结果仍违反当前 snapshot invariant。

因此，第一阶段 ODPS 实现必须区分**逻辑 Append**和**物理行 append**：

1. 读取父版本的 `original_groups` 与 `resolved_groups` 分区。
1. 在应用层合并旧列表和新 delta。
1. 使用现有 overwrite writer，把完整 group 快照写到新的不可变 `state_version=<run_id>` 分区。
1. 校验全部 artifact 后，最后发布 completed-version manifest。
1. 使用 single-writer 或父版本校验，因为四类输出不共享一个事务，而且 MaxCompute 没有 insert lock。

不要仅为本流程给通用 `OdpsWriter` 增加简单的 `append=True` 开关。只有未来显式引入不同的 delta schema，例如 `(state_version, sequence, codebook, itemids_delta)`，并要求 reader 在使用前折叠这些 delta 时，物理 append 才适合。

当前工作区没有配置 ODPS credential 或 CI project，因此本次验证基于官方文档、固定版本和已安装版本的 PyODPS request model，以及仓库源码，而不是远程 live write。发布前应增加 ODPS integration gate：使用带 `ARRAY<BIGINT>` item IDs 的临时非 clustered 普通表和静态分区，先写 `(A, [1, 2])`，再用 `overwrite=False` 追加 `(B, [3])` 与 `(A, [4])`；确认旧行仍存在且 `A` 产生重复行而不是合并，验证 `overwrite=True` 会替换目标，并注入版本发布失败以确认父快照仍为 active。

## 7. 建议的模块划分

- `collision_resolution.py`：纯数组的 Fresh/Append 规划逻辑，不涉及存储。
- `SidKeyCodec`：稳定 SID 编解码、范围检查和 band 标识。
- `BucketOccupancy`：紧凑的历史计数，以及仅包含本次触达 bucket 的可变 overlay。
- `CollisionPlan` / `CollisionResult`：明确区分 base counts、新接纳项、新 item 的最终 code/index、unresolved 行和合并后计数。
- 新的 state 模块（如 `collision_state.py`）：manifest、group reader、完整性检查、快照合并，以及基于现有仓库 I/O 的原子发布。
- `collision_prevention.py`：CLI 参数、mode 校验、流程编排、统计和 writer 选择。

I/O 边界继续使用 Arrow，CPU planner 使用 NumPy。把全部数据统一成 tensor 会增加复制和 device 管理，却不会改善这种排序、分组和合并工作负载。1 亿数据规模下，应避免全量 Python dict 或 item-ID set，改用排序数组、向量化查找、流式历史 ID，以及只保存新批次触达 bucket 的小型 overlay。

现有 grouping helper 不能直接把新行 scatter 到以“合并后 counts”为大小的数组中。Append index 是绝对位置，而 delta buffer 使用相对位置；构造新 group fragment 时应使用 `delta_position = absolute_index - base_bucket_count`。

## 8. 会影响结果的关键决策

以下规则需要写入兼容性契约：

- 历史 item 永远优先占用容量，且绝不迁移。
- 历史 resolved item 顺序以及对应 index 永远不变。
- 必须先预占所有新 origin 接纳项，再执行任何新 candidate 分配。
- 新 item 排序保持确定性，并且不受输入行顺序影响。
- capacity、`layer_sizes`、item-ID 类型、hash 版本和分配算法版本必须与父 state 一致。
- `drop` 允许 resolved state 的 item 数少于 original state；`keep_original` 允许 bucket 超过 capacity。
- Append map 是增量。若需要完整历史 map，就必须把旧 map 作为额外 state 输入，或承担昂贵的重建成本。

当前工作树中存在临时的全数组/debug 输出；在大规模测试前必须删除，否则输出本身会主导耗时和内存。但该清理不属于本次仅设计文档的变更范围。

## 9. 验证计划

1. Golden/differential 测试：空 state 下的 Fresh 在 first-fit、random、全部 fallback 和 `rate_only` 模式中都与当前实现一致。
1. 稳定 key 测试：历史 prefix 为 `A,B`，新批次只有 `B`，用于捕获 batch-local band ID 问题。
1. Append 场景：空、部分占用、满和已超容量 origin；已占用 candidate target；仅新数据 bucket；candidate 竞争。
1. 不变性检查：所有旧 final SID、item-list 前缀和 index 保持不变；每个新 map index 都指向合并后的 resolved list。
1. 重复检查：新批次内部、新批次与历史之间；把提供的 sample append 到提供的快照时，应检测并拒绝 500 个重复 ID。
1. 兼容性检查：capacity、layer sizes、item-ID 类型、manifest 版本、schema、格式和路径不匹配。
1. Fallback 与格式矩阵：`error`、`drop`、`keep_original`；CSV/Parquet round trip；拒绝混合格式目录。
1. 运维测试：连续两次 append、输入顺序置换、发布失败、并发父版本冲突，以及旧 active version 恢复。
1. 规模测试：1 亿历史 item、大量 bucket、单热点 bucket、planner 内存上界、快照合并吞吐和 CSV 单行大小限制。
1. ODPS integration：在临时普通表和静态分区验证平台 append、验证当前 `OdpsWriter` 两次独立运行的替换行为、版本分区隔离，以及拒绝 snapshot 中的重复 codebook 行。

## 10. 实施顺序

1. 用 golden 测试冻结 Fresh 行为，并定义 manifest/key 兼容性契约。
1. 增加稳定 SID key、state 数据类，以及旧 state 校验和 bootstrap。
1. 让纯 planner 接受 base occupancy，并证明空 base 与当前实现等价。
1. 基于现有 I/O 增加 delta grouping 和完整快照的流式合并。
1. 增加显式 `fresh|append` CLI 校验、版本化发布、统计和失败处理。
1. 完成正确性、格式与规模测试矩阵；只有完整快照重写成本确实不可接受时，再引入 delta 文件和 compaction。
