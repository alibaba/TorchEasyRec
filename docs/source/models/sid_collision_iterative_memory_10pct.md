# Iterative collision 10% 样本内存实验

六个版本全部完整运行成功；item_to_sid 和 sid_to_items 的所有 Parquet 文件与 baseline
逐文件 SHA256 完全一致。该样本的整次峰值主要来自候选读取，未显示显著的整体内存或
耗时改善；计算窗口的峰值也没有随着优化逐项下降。

实验日期：2026-10-14。使用同一份固定样本，先运行优化前的 baseline，再依次叠加优化
1、2、3、4、5。所有测试均通过 collision 工具的 CLI 完成，未新增外部测试脚本。

## 数据和环境

- 原始输入：
  `/mnt/data/aop_lab/collision_exp/all_item_candidate_raw/TorchEasyRec/exps/part-*.parquet`。
- 原始数据：6 个 Parquet 文件，共 255,172,938 行。
- 固定样本：25,517,293 行，即原始行数的 10% 向下取整。
- 样本大小：55,025,012,200 字节，约 51.25 GiB；准备耗时 722.93 秒。
- 抽样方式：随机种子 `20261014`；按源文件行数分配样本额度，随机选择完整 Parquet
  row group，最后一个抽中的 row group 裁到目标行数。输出保留所有原始列。
- 这是 row group 块采样，并非逐行独立随机抽样。六个版本使用完全相同的数据；抽样的
  row group 索引、各文件行数和大小记录在样本目录的 `sampling.json` 中。
- 样本目录：
  `/home/admin/workspace/TorchEasyRec/experiments/uniform_simplify_20261012/experiments/iterative_memory_10pct_20261014/sample`。
- 服务器：`8.130.128.114:1033`；64 个逻辑 CPU，内存限制 256 GiB，无 swap。
- 环境：PyTorch `2.14.0+cu130`、NumPy `2.4.6`、PyArrow `17.0.0`。

## 运行配置

六个版本都运行完整流程，包括 item_to_sid、sid_to_items 和 manifest 写出。

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 /opt/conda/bin/torchrun \
  --standalone --nproc_per_node=8 \
  -m tzrec.tools.sid.resolve_sid_collisions \
  --input_path 'experiments/iterative_memory_10pct_20261014/sample/part-*.parquet' \
  --output_path 'experiments/iterative_memory_10pct_20261014/runs/<版本>/artifacts' \
  --generation v1 \
  --codebook 8192,8192,8192 \
  --max_items_per_codebook 5 \
  --item_id_field id \
  --strategy iterative \
  --batch_size 100000
```

版本顺序如下，后一个版本包含前一个版本的优化：

| 版本                                 | 累计改动                                                    |
| ------------------------------------ | ----------------------------------------------------------- |
| [已完成] `stage0_baseline`           | 优化 1 之前的 iterative 实现                                |
| [已完成] `stage1_inplace`            | 1：候选末层 code 原地转换为 SID 索引                        |
| [已完成] `stage2_release`            | 1+2：阶段中间数组使用完后及时释放                           |
| [已完成] `stage3_candidate_lifetime` | 1+2+3：resolver 内加载候选，避免调用者持有完整矩阵          |
| [已完成] `stage4_column_repeat`      | 1+2+3+4：按候选列展开调整量，避免完整的二维 repeat 临时矩阵 |
| [已完成] `stage5_column_proposals`   | 1+2+3+4+5：按候选列查询容量、累计计数并生成 proposal        |

各版本增加相同的规模日志，记录溢出 item 数、相关 prefix 数和候选列数。除此以外，
baseline 保留优化前的实现，其他既有修复和输入输出逻辑保持一致。

## 统计口径

- PSS 和 RSS：每隔约 2 秒读取 torchrun 及其进程树的 `/proc/<pid>/smaps_rollup`，
  合计所有进程。PSS 按比例计算共享页面，适合比较整次任务的物理内存占用。
- CPU：按进程树 CPU 时间的采样增量计算，单个逻辑 CPU 为 100%，整机为 6400%。
- 完整耗时及累计 user/system CPU 时间：使用 Bash 内置 `time`。
- 内存和 CPU 峰值是采样最大值，不是连续采集的绝对高水位；PSS 不包含未映射的文件缓存。
- 各 rank 会在不同时间进入计算和输出，按日志划分的阶段时间窗口会重叠，阶段峰值不能相加。
- 顺序运行且不清空文件缓存。每版仅跑一次，耗时差异包含缓存和运行波动，不能据此做严格显著性判断。
- 10% 样本的碰撞数量和溢出数量不会随行数线性增长，不能把内存或耗时直接乘十预测全量。

## 实测结果

六个版本退出码均为 0。六次运行累计耗时 1,629.49 秒，约 27 分 9.5 秒，不包含抽样准备。

| 版本     | 完整耗时     | 峰值 PSS  | 峰值 RSS  | 峰值 CPU／整机占比 | 与 baseline 的结果比较 |
| -------- | ------------ | --------- | --------- | ------------------ | ---------------------- |
| baseline | 4 分 36.7 秒 | 30.61 GiB | 39.40 GiB | 820.1%／12.81%     | 基准                   |
| +优化 1  | 4 分 30.3 秒 | 30.14 GiB | 38.94 GiB | 813.1%／12.70%     | 文件内容完全一致       |
| +优化 2  | 4 分 30.1 秒 | 29.53 GiB | 38.33 GiB | 811.1%／12.67%     | 文件内容完全一致       |
| +优化 3  | 4 分 31.1 秒 | 30.56 GiB | 39.36 GiB | 813.3%／12.71%     | 文件内容完全一致       |
| +优化 4  | 4 分 27.2 秒 | 29.77 GiB | 38.56 GiB | 833.2%／13.02%     | 文件内容完全一致       |
| +优化 5  | 4 分 34.1 秒 | 29.46 GiB | 38.25 GiB | 810.6%／12.67%     | 文件内容完全一致       |

各阶段窗口的采样峰值 PSS：

| 版本     | 读取 SID、准备 plan | 候选读取  | 从首个 rank 进入计算到发布 manifest | 所有 rank 都进入计算之后 |
| -------- | ------------------- | --------- | ----------------------------------- | ------------------------ |
| baseline | 12.90 GiB           | 30.61 GiB | 19.01 GiB                           | 14.86 GiB                |
| +优化 1  | 12.85 GiB           | 30.14 GiB | 23.71 GiB                           | 13.53 GiB                |
| +优化 2  | 12.39 GiB           | 29.53 GiB | 23.82 GiB                           | 14.81 GiB                |
| +优化 3  | 12.61 GiB           | 30.56 GiB | 26.91 GiB                           | 13.16 GiB                |
| +优化 4  | 12.50 GiB           | 29.77 GiB | 25.51 GiB                           | 14.92 GiB                |
| +优化 5  | 12.00 GiB           | 29.46 GiB | 27.10 GiB                           | 14.30 GiB                |

候选读取开始时间由首条进度日志中的行数和速度估算。计算开始时间取各 rank 的
`Iterative collision:` 规模日志。倒数第二列包含部分 rank 尚在扫描候选的时段；
最后一列排除这段扫描重叠，但仍包含计算、结果组装和输出，也可能漏掉较早 rank 的
计算峰值。因此，这些窗口均不是单个函数或所有 rank 同一计算步骤的精确峰值。

累计 CPU 时间和平均占用：

| 版本     | User CPU 时间 | System CPU 时间 | 平均 CPU／整机占比 |
| -------- | ------------- | --------------- | ------------------ |
| baseline | 1,510.243 秒  | 597.249 秒      | 761.65%／11.90%    |
| +优化 1  | 1,491.889 秒  | 576.591 秒      | 765.53%／11.96%    |
| +优化 2  | 1,499.763 秒  | 577.750 秒      | 769.16%／12.02%    |
| +优化 3  | 1,496.927 秒  | 575.133 秒      | 764.59%／11.95%    |
| +优化 4  | 1,500.106 秒  | 574.620 秒      | 776.46%／12.13%    |
| +优化 5  | 1,503.213 秒  | 569.981 秒      | 756.64%／11.82%    |

CPU 时间是所有进程的累计值，会大于墙钟耗时。

## 本次碰撞规模和数组开销

六个版本的规模和最终统计完全一致：

| 项目                        | 数值       |
| --------------------------- | ---------- |
| 输入 item 数                | 25,517,293 |
| 初始超容量 SID 数           | 85,271     |
| 溢出 item 数                | 564,680    |
| 包含溢出 items 的 prefix 数 | 47,900     |
| 每个 item 的候选数          | 200        |
| 成功迁移数                  | 564,680    |
| 未解决数                    | 0          |
| 最终超容量 SID 数           | 0          |
| 最终每个 SID 最大 item 数   | 5          |
| item_to_sid 行数            | 25,517,293 |
| sid_to_items 行数           | 22,637,430 |

对应的数组大小推算如下。这是完整候选规模下的数组字节数，不是实测 PSS，也不能把
各项相加作为某一时刻的峰值。

| 内容                  | baseline 的数组大小 | 最终版本的处理                        |
| --------------------- | ------------------- | ------------------------------------- |
| 已加载的候选矩阵      | 0.841 GiB           | 原地转为 SID 索引，继续使用这一份数组 |
| 额外候选 SID 索引矩阵 | 0.841 GiB           | 不再分配                              |
| 容量查询临时结果      | 0.841 GiB           | 每次一列，合计规模约 4.31 MiB         |
| 候选有容量的布尔矩阵  | 0.105 GiB           | 每次一列，合计规模约 0.54 MiB         |
| int32 累计计数矩阵    | 0.421 GiB           | 每次一列，合计规模约 2.15 MiB         |
| repeat 展开的调整量   | 0.421 GiB           | 每次一列，合计规模约 2.15 MiB         |
| 密集 SID 容量表       | 2.924 GiB           | 本次未修改                            |

逐列数据的数字按各 rank 本轮仍包含全部溢出 items 推算，后续轮次会随着待处理数量减少。
proposal 和排序索引仍随实际请求数分配，本次没有改为稀疏容量表或逐项贪心策略。

## 各优化点的实际改动

1. [已完成] 原地转换候选矩阵：消除完整矩阵的第二份分配；会修改传入的内存数组，
   不会改写源 Parquet。
1. [已完成] 提前释放中间数组：生成 proposal 后释放容量布尔矩阵和累计计数；完成
   选择后释放不再使用的 proposal、排序索引和标记，减少与下一轮数组的重叠。
1. [已完成] 调整候选所有权：CLI 传入加载函数，由 resolver 内部调用。直接写成
   `resolve(plan, loader())` 时，调用参数仍会持有完整数组，不能解决生命周期问题。
   resolver 在完成原地转换后删除原变量，筛选待处理候选时旧矩阵可以释放。
1. [已完成] 逐列展开调整量：该中间版本仍保留完整累计计数矩阵，但避免一次展开
   所有候选列的 repeat 结果。
1. [已完成] 逐列生成 proposal：容量查询、累计计数和调整量都缩小到一列；保持目标
   SID、候选位置和 item 优先级的竞争规则。由于请求改为按列生成，接受后按 item 和
   候选位置选择最早候选，保证与 baseline 的结果一致。

## 如何解读结果

- 最终版本相对 baseline，整次峰值 PSS 低 1.15 GiB，约 3.77%；完整耗时少 2.61 秒，
  约 0.94%。但读取阶段代码未因这些优化改变，所以不能把整次峰值差异直接归因于优化。
- 所有 rank 进入计算后的窗口峰值从 14.86 GiB 到 14.30 GiB，减少约 0.57 GiB。
  中间版本存在回升，未显示逐项稳定下降；采样粒度、rank 时间差、输出以及内存分配器
  的页面保留都会影响这个窗口。
- 本次样本的溢出 items 只有全量原始运行中约 2,939 万溢出 items 的 1.92%。因此候选
  矩阵只有 0.841 GiB，而全量对应约 43.8 GiB；候选复制和临时矩阵的压力明显不同。
- 这次实验确认六版都能完整运行、产物没有变化，也确认相关完整数组已按代码改动消除；
  没有充分证据证明整体内存或性能显著改善，不能据此保证全量 iterative 已能在
  256 GiB 内完成。

## 正确性检查

- 最终版本的 iterative 和公共 collision 测试共 35 项通过。
- 六个版本在 100 个固定随机用例上与 baseline 比较，最终 code、slot、未解决行、
  SID 数量及统计完全一致，覆盖追加数据、多个 prefix 和空候选。
- 六版大样本统计完全一致；共 96 个 Parquet 文件逐文件 SHA256 与对应 baseline 文件
  完全一致。每个版本的两个产物均有 8 个文件，各文件 footer 行数合计与 manifest 一致。
- 固定样本的 6 个文件 footer 行数合计为 25,517,293，与抽样记录一致。
- Ruff 检查、格式检查和修改文件的 Pyrefly 类型检查通过；未重复运行大样本测试。

## 原始记录和产物

实验根目录：

```text
/home/admin/workspace/TorchEasyRec/experiments/uniform_simplify_20261012/experiments/iterative_memory_10pct_20261014
```

每个版本的 `runs/<版本>/` 包含运行日志、时间记录、资源采样、代码 SHA256 和退出状态。
各版本代码快照保存在实验目录的 `variants/` 下。

各版本 manifest 都已经生成。将下面的相对路径接在实验根目录之后即可访问：

```text
runs/stage0_baseline/artifacts/v1/manifest.json
runs/stage1_inplace/artifacts/v1/manifest.json
runs/stage2_release/artifacts/v1/manifest.json
runs/stage3_candidate_lifetime/artifacts/v1/manifest.json
runs/stage4_column_repeat/artifacts/v1/manifest.json
runs/stage5_column_proposals/artifacts/v1/manifest.json
```

最终版本 manifest 的完整路径：

```text
/home/admin/workspace/TorchEasyRec/experiments/uniform_simplify_20261012/experiments/iterative_memory_10pct_20261014/runs/stage5_column_proposals/artifacts/v1/manifest.json
```

汇总记录为实验根目录下的 `results.json`，包含每版资源指标、统计、manifest 路径和
各产物文件的 SHA256。本地副本位于
`experiments/iterative_memory_10pct_20261014/results.json`，属于 gitignored 实验数据。
本报告的远端副本位于实验根目录下的 `report.md`。

代码和文档修改保留在本地，未 commit、未 push。

随后使用累计优化 1–5 完成了原始 2.55 亿行全量运行，实测结果见
[全量实验报告](sid_collision_iterative_memory_full.md)。本页保留 10% 实验本身的结论，
全量结果由实际运行确认。
