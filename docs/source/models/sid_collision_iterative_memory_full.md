# Iterative collision 累计优化 1–5 全量实验

全量运行成功，退出码为 0，manifest 已生成。完整耗时 75 分 40.0 秒，采样峰值 PSS
235.19 GiB、RSS 236.33 GiB。这次在 256 GiB 环境中完成了计算和输出，内存余量仍较小。

实验日期：2026-10-14。使用 10% 实验中最终版本的代码，在原始 255,172,938 行数据上
运行完整流程，包括两个 Parquet 产物和 manifest 写出。

## 输入、版本和环境

- 输入：
  `/mnt/data/aop_lab/collision_exp/all_item_candidate_raw/TorchEasyRec/exps/part-*.parquet`。
- 数据：6 个文件，255,172,938 行，约 512.44 GiB；本次没有抽样或拆分输入文件。
- 策略：`iterative`；累计优化 1–5，与 [10% 实验](sid_collision_iterative_memory_10pct.md)
  的 `stage5_column_proposals` 相同。
- 服务器：`8.130.128.114:1033`；64 个逻辑 CPU，内存限制 256 GiB，无 swap。
- 环境：PyTorch `2.14.0+cu130`、NumPy `2.4.6`、PyArrow `17.0.0`。
- codebook：`8192,8192,8192`；每个 SID 容量：5；候选数：200。
- 一次 torchrun，8 个进程；batch size：100000；`OMP_NUM_THREADS=1`。
- 本地和远端的五个相关实现文件 SHA256 一致，实际运行版本记录在实验目录的
  `source.sha256` 中。

## 命令和监控

工作目录：

```text
/home/admin/workspace/TorchEasyRec/experiments/uniform_simplify_20261012
```

实际碰撞计算只运行仓库自带 CLI：

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 /opt/conda/bin/torchrun \
  --standalone --nproc_per_node=8 \
  -m tzrec.tools.sid.resolve_sid_collisions \
  --input_path '/mnt/data/aop_lab/collision_exp/all_item_candidate_raw/TorchEasyRec/exps/part-*.parquet' \
  --output_path 'experiments/collision_255m_iterative_optimized_20261014/artifacts' \
  --generation v1 \
  --codebook 8192,8192,8192 \
  --max_items_per_codebook 5 \
  --item_id_field id \
  --strategy iterative \
  --batch_size 100000
```

没有新增或运行外部碰撞包装、分区或调度脚本。启动使用临时 Shell 命令，资源监控使用
系统的 `top`、`ps`、`awk` 和 Bash 内置 `time`。没有通过外部程序替代 collision
算法，也没有把数据拆给多个独立 torchrun。

PSS/RSS 每隔约 2 秒从进程的 smaps 采集，合计 torchrun 和 8 个 worker；CPU 由
`top` 的同一组进程采样。峰值都是采样最大值。单个逻辑 CPU 为 100%，整机为 6400%。
PSS 按比例计算共享页面，不包含未映射的文件缓存。

## 全量结果

启动时间：`2026-10-14T04:44:05Z`。

结束时间：`2026-10-14T05:59:45Z`。

| 指标                                  | 实测值                         |
| ------------------------------------- | ------------------------------ |
| 完整耗时                              | 4,540.020 秒，即 75 分 40.0 秒 |
| 峰值 PSS，torchrun 和全部 worker 合计 | 235.19 GiB                     |
| 峰值 RSS，全部进程合计                | 236.33 GiB                     |
| 单进程峰值 RSS                        | 36.33 GiB                      |
| 峰值 CPU，单个逻辑 CPU 为 100%        | 839.0%，约占整机 13.11%        |
| 平均 CPU                              | 780.08%，约占整机 12.19%       |
| 累计 User CPU 时间                    | 25,768.214 秒                  |
| 累计 System CPU 时间                  | 9,647.450 秒                   |
| 退出码                                | 0                              |
| 内核 OOM kill 计数                    | 运行前后均为 0                 |

各阶段窗口的采样峰值 PSS：

| 时间窗口                         | 峰值 PSS   |
| -------------------------------- | ---------- |
| 读取原始 SID、准备 plan          | 85.47 GiB  |
| 候选读取，直到首个 rank 进入计算 | 102.99 GiB |
| 首个 rank 进入计算至运行结束     | 235.19 GiB |
| 所有 rank 都进入计算之后         | 235.19 GiB |

第一个 rank 完成候选读取的时间为 `05:48:45.455 UTC`，最后一个为
`05:50:37.570 UTC`。各 rank 的计算开始时间介于 `05:48:47.936 UTC` 和
`05:50:40.453 UTC`。因此计算窗口包含部分 rank 仍在读取的重叠时段，也包含结果组装和
输出；不能将窗口峰值当作单个函数的精确内存开销。

计算窗口 PSS 从约 100 GiB 升至 235.19 GiB。首轮陆续完成后下降到约 148 GiB，
输出阶段继续下降。本次通过了此前失败的首轮计算，并完整写出产物。

### 碰撞规模和剩余内存开销

| 项目                          | 数值       |
| ----------------------------- | ---------- |
| 溢出 item 数                  | 29,392,918 |
| 包含溢出 items 的 prefix 数   | 988,452    |
| 候选数                        | 200        |
| 原地复用的 int64 候选矩阵大小 | 43.80 GiB  |
| 密集 SID 容量表的数组大小     | 60.33 GiB  |

这两项是数组字节数推算，不是各自测得的 PSS。密集容量表尚未优化；proposal 和排序
索引仍按请求数量分配，也是剩余的重要内存开销。候选转换、逐列计算和中间数组释放
已经改动，但整次运行仍接近 256 GiB 限制，不能把本次成功泛化到其他配置或数据分布。

### 最终统计和产物

| 项目                      | 数值           |
| ------------------------- | -------------- |
| 输入 item 数              | 255,172,938    |
| 初始超容量 SID 数         | 3,719,947      |
| 成功迁移的溢出 item 数    | 29,238,615     |
| 未解决 item 数            | 154,303        |
| 最终超容量 SID 数         | 3,077          |
| 最终单个 SID 最大 item 数 | 3,086          |
| item_to_sid 行数／文件数  | 255,172,938／8 |
| sid_to_items 行数／文件数 | 178,016,249／8 |

两个产物共 16 个文件，Parquet footer 行数合计与 manifest 一致。item_to_sid 行数与
原始输入行数一致；检查的 Parquet schema 中，SID 列和 itemids 列均为 LargeList。
工具保留未解决 items，因此最终仍有超容量 SID；成功退出不代表所有冲突都已消除。

## 与优化前的全量运行比较

| 版本                   | 状态                 | 完整耗时或终止前耗时      | 峰值 PSS   | 峰值 RSS   |
| ---------------------- | -------------------- | ------------------------- | ---------- | ---------- |
| candidate              | 完整成功             | 67 分 24.6 秒             | 103.50 GiB | 112.40 GiB |
| 优化前 iterative       | 计算中被 nohang 终止 | 66 分 23.2 秒，非完整耗时 | 234.22 GiB | 240.20 GiB |
| 累计优化 1–5 iterative | 完整成功             | 75 分 40.0 秒             | 235.19 GiB | 236.33 GiB |

历史 iterative 在可用内存低于 3% 时由服务器 nohang 服务发送 SIGTERM，未生成
manifest；它的终止前耗时不能作为完整执行耗时比较。三次运行均为同一输入、8 个进程、
相同 codebook、容量和 batch size，但顺序运行、文件缓存未清空，每版只运行一次。

本次 iterative 比历史 candidate 多用 8 分 15.5 秒，约 12.25%；峰值 PSS 约为其
2.27 倍。两种策略的分配规则不同，最终统计如下：

| 指标                      | 历史 candidate | 本次 iterative |
| ------------------------- | -------------- | -------------- |
| 成功迁移数                | 29,264,462     | 29,238,615     |
| 未解决数                  | 128,456        | 154,303        |
| 最终超容量 SID 数         | 1,946          | 3,077          |
| 最终单个 SID 最大 item 数 | 3,054          | 3,086          |

优化前 iterative 的 234.22 GiB 是被终止前采样到的峰值，既不是完整运行峰值，也可能
漏掉快速分配时的更高占用。它没有产物可供全量结果比较。能够确认的是累计优化版本
在本次环境中完整跑通；不能根据两个相近的采样数值宣称整体峰值已下降。

## 记录和产物路径

实验根目录：

```text
/home/admin/workspace/TorchEasyRec/experiments/uniform_simplify_20261012/experiments/collision_255m_iterative_optimized_20261014
```

目录包含运行日志、时间记录、PSS/RSS/CPU 采样、代码 SHA256、开始/结束时间和退出状态。
manifest 已生成，完整路径如下：

```text
/home/admin/workspace/TorchEasyRec/experiments/uniform_simplify_20261012/experiments/collision_255m_iterative_optimized_20261014/artifacts/v1/manifest.json
```

产物目录：

```text
<实验根目录>/artifacts/v1/item_to_sid
<实验根目录>/artifacts/v1/sid_to_items
```

实验根目录的 `results.json` 保存汇总指标、统计、产物 schema、footer 行数核对和
manifest 内容。本地副本位于
`experiments/collision_255m_iterative_optimized_20261014/results.json`，属于 gitignored
实验记录。报告的远端副本为实验根目录下的 `report.md`。

没有 commit 或 push。
