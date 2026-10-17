# SID 冲突消解

`resolve_sid_collisions` 处理 SID 预测结果：保留每个 SID 容量内的 item，将超出容量的 item
重新分配到同一 SID 前缀下的末层码。前缀是除最后一层外的 SID，不会在消解时改变。

输入包含 `item_id` 和 `codes`；`candidate`、`iterative` 策略还需要 `candidate_codes`。
Parquet 的 `codes` 是整数列表，`candidate_codes` 是按候选顺序展开的 `topk × SID 层数`
个整数。CSV 使用逗号分隔的字符串表示这些列表。

## 全量处理模式

输入包含全部待处理物品。在仓库根目录执行，给所有进程传入同一份完整输入和同一输出目录：

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

`--nproc_per_node=8` 由 torchrun 创建 8 个进程。工具依据 `RANK/WORLD_SIZE`，将 SID
前缀空间划为连续区间，每个进程负责其中一个区间。同一前缀的全部 item 在同一个进程中
消解，不会因输入文件或批次的边界而被拆开。无需提前拆分输入，也不需要
额外的并发启动脚本。单进程使用 `--nproc_per_node=1`。

当前各进程扫描完整输入批次，只保留自己负责的前缀；候选读取也只保留本进程的溢出
item。增加进程数可减少每个进程保留的数组，但不能保证读取速度按进程数同比提高。
前缀分布不均时，各进程的工作量也会不同。单层 SID 只有一个空前缀，由 rank 0 处理。

各进程写出各自的 `part-*` 文件，rank 0 在全部写出完成后发布一个 `manifest.json`，
其中的统计和产物行数是全局汇总值。所有进程必须能访问相同的输入和输出路径。
manifest 先写入同目录临时文件，完整写入并关闭后再原子替换为正式文件；
正式 `manifest.json` 出现才表示本代完成。

写出前检查本次 generation 的本地目录，目录已有内容时直接报错，不覆盖或自动清理。
检查同时覆盖 `output_path` 和本地 `item_to_sid_root`；不存在或为空的目标目录可以使用。
追加的新 generation 可以与已有上一代共用输出根目录，`--rate_only` 不执行写出目录检查。

## 新增物品处理模式

追加输入只包含新 item，已发布 item 的 SID 不移动。追加可以使用不同的进程数：

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
    -m tzrec.tools.sid.resolve_sid_collisions \
    --input_path 'sid_predict_new/*.parquet' \
    --output_path sid_collision \
    --generation v2 \
    --from_generation v1 \
    --codebook 8192,8192,8192 \
    --max_items_per_codebook 1 \
    --strategy uniform
```

每一代的 `item_to_sid` 和 `sid_to_items` 都是完整语料。追加还写出
`delta_item_to_sid` 和 `delta_sid_to_items`，后者包含本次获得新 item 的完整 SID 分组，方便下游
进行幂等更新。新 item ID 如果已存在于上一代，会报错。

Parquet 产物的 SID 和 item ID 列表使用普通 Arrow `List`，其 offsets 为 int32。
构造列表时，展开后的元素总数超过 `2^31 - 1` 会报错终止，不自动分块或切换为 LargeList。
对于三层 codes，单次构造最多容纳 715,827,882 行；itemids 列表按展开后的 item 总数检查。
该限制作用于每个 rank 构造的数组，输入拆成多个 Parquet 文件不会改变上限。
CSV 仍输出列表字符串，ODPS 的 item 映射仍使用普通 `List`。
列名、列表内容、`manifest.json` 的结构和 `save_type` 不变。

`--rate_only` 计算全局统计并跳过产物写出，可用于比较不同策略。

## 选择策略

省略 `--strategy` 时默认使用 `candidate`。

- `candidate`：按稳定 item 顺序，选择第一个仍有容量的候选末层码。
- `iterative`：多轮提议和竞争，优先接收将目标 SID排在更靠前位置的 item。
- `uniform`：无需候选，将溢出 item 分配到前缀内负载最小的末层码。

`candidate`、`iterative` 在候选耗尽后保留原 SID；`uniform` 在前缀容量不足时继续均匀
分配，最终仍超过SID 容量的 item 计为未解决。所有输入 item 都会保留。

## 排查运行问题

- 未出现 `manifest.json`：该代尚未完成，查看 torchrun 中报错进程的日志；重跑使用新的
  `generation`，或先清理该次未完成的输出目录。
- 部分进程没有 item：其前缀区间没有数据，仍会参加全局统计汇总。
- 内存不足：查看溢出前缀数、末层码本大小和候选矩阵规模；调整 `--batch_size` 只影响
  读取批次，不能降低完整计算数组的占用。
