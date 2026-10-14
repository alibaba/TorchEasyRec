# PR #692：评审意见处理建议

本清单针对 [PR #692](https://github.com/alibaba/TorchEasyRec/pull/692) 在提交
`3dd4bf1` 上收到的评审意见，整理日期为 2026-10-13。
共核对 8 条行内意见，以及评审汇总中补充的 1 条 README 建议。

第 1、2、3、5、6、7、8、9 项已在本地处理，前面标记为“[已处理]”；第 4 项驳回，保留现有实现。
“已处理”不表示已提交或已回复：本轮没有 commit、push 或处理 GitHub 评审线程。
完整 PR 变更说明见 [SID collision PR 变更说明](sid_collision_pr_changes.md)。

## 处理结论

建议接受 7 条、部分接受 1 条、驳回 1 条。
判断遵循本次改造目标：简化 collision 工具、删除过度防御、提高人类可读性。
接受意见时优先采取小范围修改，不恢复已经明确删除的内部 chunk 分块，也不增加重复验证。

| 序号       | 评审意见                                 | 建议     | 优先级   | 建议处理范围                                                                   |
| ---------- | ---------------------------------------- | -------- | -------- | ------------------------------------------------------------------------------ |
| [已处理] 1 | 失败重跑可能残留旧 part 文件             | 接受     | 高       | 拒绝复用已有内容的本地目标 generation，不自动删除文件                          |
| [已处理] 2 | manifest 应原子写入                      | 接受     | 高       | 临时文件写完后通过 `os.replace` 发布                                           |
| [已处理] 3 | 恢复 chunk、限制写出内存与 Arrow offsets | 部分接受 | 中       | Parquet 改用 LargeList，不增加容量检查；驳回恢复 chunk 与巨大 row group 的判断 |
| 4          | 删除溢出状态的跨 rank 汇总               | 驳回     | 无需修改 | 保留同步，避免 ODPS reader 初始化时 collective 不一致                          |
| [已处理] 5 | 候选读取先匹配 ID，再解码 codes          | 接受     | 中       | 只解码命中的 rows，仍对这些 rows 做 prefix 筛选                                |
| [已处理] 6 | 补 Uniform 的 torchrun 覆盖与统计断言    | 接受     | 中       | 补必要分支覆盖，复用既有产物对照断言                                           |
| [已处理] 7 | `is_requires_candidates` 命名不自然      | 接受     | 低       | 改为 `requires_candidates()`，保留方法形式                                     |
| [已处理] 8 | 文档中的“16 个分区”和默认策略说明        | 接受     | 低       | 删除无上下文的数字，说明默认策略为 `candidate`                                 |
| [已处理] 9 | README 增加 collision 文档入口           | 接受     | 低       | 在 SID 相关入口处增加链接                                                      |

## [已处理] 1. 失败重跑残留旧 part 文件

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256974957)

**结论：接受，优先处理。**

同一个 generation 的失败任务改用不同进程数重跑时，原实现只覆盖本次写出的 rank 文件，
不会处理超出本次 rank 范围的旧文件。例如旧任务启动 8 个进程，新任务启动 4 个进程，
旧的高 rank part 文件可能残留在输出目录中。

后续 reader 按 `part-*` 读取时，可能将这些旧文件与本次产物一起读入。
即使本次 manifest 已经写入，这种混合数据也可能无法被立即识别。

已在 `run()` 读取输入之前检查本次本地目标 generation，已有内容时直接拒绝复用。
已有上一代 generation 不应因此被拒绝；追加本来就需要读取上一代。

rank 0 检查 `output_path` 和本地 `item_to_sid_root` 的目标 generation，并向所有 rank 广播结果，
确保检查完成前没有进程创建输出或覆盖 part 文件。目标目录不存在或为空时允许写入，
`rate_only` 不执行这项检查，ODPS 继续沿用现有分区写入语义。

不采用自动删除旧文件的方案，避免重跑时误删已有产物，也不引入复杂的断点恢复或自动清理流程。
需要区分“追加新一代时改变进程数”和“复用失败任务的同一 generation 重跑”，两者不是同一场景。

远端 8 项相关测试通过，覆盖两个本地输出根目录的残留高 rank 文件、已有 manifest、空目录、
正常追加、torchrun 改变 rank 数和 rate_only；Ruff 与类型检查通过。

## [已处理] 2. manifest 原子写入

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256975432)

**结论：接受。**

原实现的 `Bundle.write_manifest` 直接打开目标文件并写入。
如果写入中途进程退出，读者可能看到已经存在但内容不完整的 `manifest.json`。
这与“manifest 出现代表本代完成”的约定冲突。

已改为在同一目录先写 `manifest.json.tmp`，关闭后通过 `os.replace` 替换为正式 manifest。
这样正式路径不会在 JSON 尚未写完整时就被读者看到。
日志仍在替换成功后记录，manifest 的 JSON 结构保持不变。

远端 4 项相关测试通过，覆盖 manifest 正常发布、替换失败不产生完成标记、
全量产物和追加产物的 manifest；Ruff 与类型检查通过。

本次接受的是原子发布，不扩展为完整的掉电恢复、文件系统持久化或自动恢复机制。
临时文件加 rename 的方案也不能被描述成对所有掉电场景都提供持久化保证。

## [已处理] 3. chunk、写出内存与 Arrow offsets 上限

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256976761)

**结论：部分接受，需要拆开判断。**

### 3.1 驳回恢复内部 chunk

删除内部计算、分组和写出的 chunk 控制流，是已经确认的改造目标。
整批构造输出数组确实可能增加峰值内存，运行文档已经记录了这一取舍和写出阶段的实测数据。

不能仅凭已知的内存取舍，就将之前删除的分块、调度及配套分支重新加回。
本次不接受恢复 `_ROW_CHUNK_SIZE` 等人工分块控制。

### 3.2 接受上限问题，改用 LargeList 解决

评审对应的 `3dd4bf1` 中，`_item_lists` 使用 int32 offsets；非 CSV codes 输出也转换为普通 Arrow ListArray。
这些数据结构的偏移容量存在真实上限，超过时会报错。

后续已确认直接将 Parquet 列表输出改为 LargeList，使用 int64 offsets。
工具尚未正式投入使用，本地修改不保留旧普通 List 产物迁移；CSV 字符串和 ODPS 原有 List 编码保持不变。
不新增 int32 或 int64 容量检查，不恢复分块逻辑，也不引入按数据规模自动切换格式的机制。

原先普通 List 的上限口径按列表展开后的子元素数量，而不是统一按样本行数：

- codes 列的子元素数量约为 `item 数 × SID 层数`。
- itemids 列的子元素数量是相关 SID 列表中的 item 总数。
- int32 offsets 的最大值为 `2^31 - 1`。

因此，原意见中“一个 rank 超过 `2^31` 个样本才有问题”的表述不准确。
三层 codes 在约 7.16 亿行时，展开元素数量就可能触及这个上限。

### 3.3 驳回“一次 write 产生巨大 row group”的判断

仓库 Parquet writer 调用 PyArrow writer，没有指定 `row_group_size`。
核对验证环境的 PyArrow 17.0.0，默认 row group 行数不超过 `1024 × 1024`。
一次传入很大的 RecordBatch，并不意味着整批数据一定写成一个巨大 row group。

这不否认构造 RecordBatch 的内存开销；只是将写出前的数组内存和 Parquet 文件的 row group 划分分开讨论。

## 4. 删除溢出状态的跨 rank 汇总

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256977146)

**结论：驳回，保留现有同步。**

评审只考虑了每个 rank 加载自己的候选数组，忽略了 reader 初始化也可能需要同步。
当前执行顺序是：

1. 调用 `_load_candidate_last_codes`。
1. 通过 `_make_reader` 创建 reader。
1. 判断本 rank 的溢出 item 数是否为零，若为零才返回空数组。

ODPS reader 在构造期间初始化读取 session，并执行 `dist.broadcast_object_list`。
如果只让有溢出的 rank 创建候选 reader，没有溢出的 rank 就不会参加这次广播，
会导致不同 rank 的 collective 调用不一致。

`all_gather_object` 的作用是确认任一 rank 是否需要候选，让相关 rank 一致进入 reader 初始化。
空 rank 可以不扫描候选批次，但不能绕过这个初始化步骤。

因此它不是没有用途的额外 barrier；除非同时重新设计 reader 初始化协议，否则不应删除。

## [已处理] 5. 候选读取先匹配 ID，再解码 codes

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256978572)

**结论：接受优化建议，保留必要的 prefix 筛选。**

原实现的候选扫描先解码整个 batch 的 codes，再筛选本 rank 的 prefix，之后才匹配溢出 item IDs。
在溢出 items 只占少数时，会为大量不需要候选的数据重复解码 codes。

已调整为先匹配本 rank 的溢出 item IDs，只取命中的 rows，再解码这些 rows 的 codes
并判断 prefix，最后读取候选。

不能直接删除 prefix 筛选：工具允许本次输入包含重复 item ID，不同 prefix 的重复 ID
可能落在不同 rank。只匹配 ID 可能将另一个 prefix 的候选关联到当前 rank 的 item。

未缓存整个输入的 prefix 掩码，避免增加另一套全量中间状态。
新增回归用例核对未命中的 ID 不参与 codes 解码、其他 rank 的同 ID 不覆盖候选，
并保留本 rank 重复溢出 ID 的候选共享行为。

## [已处理] 6. Uniform 的 torchrun 覆盖与统计断言

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256978971)

**结论：接受。**

原有 torchrun 测试覆盖 Candidate 和 Iterative，没有覆盖 Uniform。
因此 Uniform 跳过候选溢出同步的分支，以及不同 rank 分别处理多个 prefix 的结果，
此前没有通过分布式用例验证。

已补一个 Uniform 的双进程 torchrun 用例，使用两个输入文件中的多 prefix 数据，
输入不包含候选列，复用既有单进程产物和 manifest 对照断言。
没有扩展成策略、文件格式、rank 数的全排列测试。

同一用例核对全部 6 个统计字段。10 个 items 的预期结果为原始冲突 SID 数 2、
最终冲突 SID 数 1、成功迁移 5、未解决 1、最大 SID item 数 2。
两 rank 分别处理容量充足和不足的 prefix，分布式全局日志必须与单进程统计完整一致，
同时验证普通计数求和、最大 SID item 数取 max 的行为。

## [已处理] 7. 候选能力方法命名

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256980567)

**结论：接受，已改为 `requires_candidates()`。**

`is_requires_candidates()` 的组合不自然，`requires_candidates()` 已经清楚表达布尔判断。
继续使用方法形式，符合此前选择；无需改成 property 或引入额外能力配置对象。

范围仅包括基类定义、Uniform override 和 Runner 的两处调用。
Candidate 与 Iterative 继续继承返回 True 的默认实现。

## [已处理] 8. 文档中的分区数量与默认策略

[原始评审意见](https://github.com/alibaba/TorchEasyRec/pull/692#discussion_r4256980970)

**结论：接受。**

“不需要提前生成 16 个输入分区”来自之前的讨论，新读者没有这段背景。
数字 16 并不是运行约束，与上面的 8 进程示例也容易产生不必要的联想。

已改为直接说明无需提前拆分输入、无需额外并发调度脚本，不保留具体数字。
选择策略部分已补充：省略 `--strategy` 时默认使用 `candidate`。

这些属于文档可读性修正，不影响工具行为。

## [已处理] 9. README 增加 collision 文档入口

[评审汇总中的补充建议](https://github.com/alibaba/TorchEasyRec/pull/692#issuecomment-6142613544)

**结论：接受，低优先级。**

README 的 Semantic ID 表原先只有 RQVAE 和 RQKMeans 链接，已增加 collision 工具文档链接，
方便使用者找到预测后的冲突消解步骤。

“新文档不可达”的表述需要修正：`docs/source/models/sid_model.rst` 的 Sphinx 目录
已经包含 `sid_collision`，页面可以从文档站目录进入。
这里改善的是 README 导航，不是修复一个完全没有入口的文档页面。

第 5-9 项的远端验证已完成：5 项相关用例通过，覆盖候选跨批次对齐、重复 ID、
匹配后 prefix 筛选、Uniform 双进程统计和 Iterative 改变 rank 数的追加流程。
Ruff 与类型检查通过，未重复执行完整测试集；运行指南和 README 的本地文档链接已检查。

## 后续处理边界

- 第 1、2、3、5、6、7、8、9 项已在本地处理，第 4 项保留现有实现，评论均未回复。
- 不恢复内部 chunk、外部分区或多任务调度。
- 不自动删除旧输出，不增加复杂恢复协议。
- 测试只补实际缺失的覆盖，不重复执行已经通过且不受修改影响的检查。
- 所有修改留在本地供审查，本次没有 commit、push 或回复 GitHub 评审。
