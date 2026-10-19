# SID 冲突消解

`resolve_sid_collisions` 处理 SID 预测结果：保留每个 SID 容量内的物品，将超出容量的物品重新分配到同一 SID 前缀下的末层码。前缀是除最后一层外的 SID，不会在消解时改变。所有输入物品都会保留，无法分配的物品保留原 SID，并计为未解决。

输入包含 `item_id` 和 `codes`；默认的 `candidate` 策略还需要 `candidate_codes`。该字段由 SID 模型在预测阶段输出，需要在模型配置中启用 `candidate_output_config.enabled`，相关配置参见 [SID 模型文档](../models/sid_model.rst)。Parquet 的 `codes` 是整数列表，`candidate_codes` 是按候选顺序展开的 `topk × SID 层数` 个整数。CSV 使用逗号分隔的字符串表示这些列表。

## 全量处理模式

输入包含全部待处理物品。在仓库根目录执行：

```bash
PYTHONPATH=. python -m tzrec.tools.sid.resolve_sid_collisions \
    --input_path 'sid_predict/*.parquet' \
    --output_path sid_collision \
    --generation v1 \
    --codebook 8192,8192,8192 \
    --max_items_per_codebook 1 \
    --strategy candidate \
    --batch_size 100000
```

`--codebook` 与生成预测结果的模型配置一致，`--max_items_per_codebook` 指定每个 SID 最多容纳的物品数。工具当前为单进程，通过 `python -m` 启动。

## 新增物品处理模式

输入只包含新增物品，通过 `--from_generation` 指定已发布的一代。已有物品的 SID 不移动，新增物品根据已有的 SID 物品数继续分配位置：

```bash
PYTHONPATH=. python -m tzrec.tools.sid.resolve_sid_collisions \
    --input_path 'sid_predict_new/*.parquet' \
    --output_path sid_collision \
    --generation v2 \
    --from_generation v1 \
    --codebook 8192,8192,8192 \
    --max_items_per_codebook 1 \
    --strategy candidate
```

每一代的 `item_to_sid` 和 `sid_to_items` 都包含完整语料。新增物品处理还写出 `delta_item_to_sid` 和 `delta_sid_to_items`，后者包含本次获得新物品的完整 SID 分组，方便下游进行幂等更新。新增物品 ID 如果已存在于上一代，会报错。

## 选择策略

省略 `--strategy` 时默认使用 `candidate`。

- `candidate`：先按原 SID 键、再按物品哈希的确定性顺序处理溢出物品，为每个物品选择第一个仍有容量的候选末层码。
- `random`：无需模型候选，为溢出物品生成确定性的随机末层码候选，再按相同的规则分配。 `--random_num_candidates` 控制每个溢出物品的抽样次数，默认 64，最多为末层码本大小减一。抽样允许重复，命中原码或重复候选时不补抽，因此前缀有剩余容量也可能无法分配。

`--rate_only` 计算统计并跳过产物写出，可用于比较不同策略。

## 产物与使用

默认情况下，本代产物写入 `<output_path>/<generation>/`：

| 产物                  | 内容                                                            |
| --------------------- | --------------------------------------------------------------- |
| `item_to_sid/`        | 每个物品的最终 SID 和 SID 内位置                                |
| `sid_to_items/`       | 每个 SID 对应的完整物品列表                                     |
| `delta_item_to_sid/`  | 本次新增物品的结果，仅新增物品处理模式输出                      |
| `delta_sid_to_items/` | 本次获得新增物品的 SID 及其完整物品列表，仅新增物品处理模式输出 |
| `manifest.json`       | 本代配置、产物位置、格式和行数                                  |

`item_to_sid` 默认使用输入格式，可通过 `--item_to_sid_writer_type` 指定格式，或通过 `--item_to_sid_root` 指定独立存储位置；`delta_item_to_sid` 使用相同格式和存储根目录。 `sid_to_items` 和 `delta_sid_to_items` 始终使用 Parquet，保留原生列表。

只有正式的 `manifest.json` 出现后，本代结果才可供下游使用。下游应根据 manifest 记录的位置和格式读取产物。目标 generation 的目录或 ODPS 分区已存在时，工具会报错退出，不覆盖已有数据。运行中捕获到异常时，工具会清理本次新建的 generation，包括独立存储位置中的本代产物，并抛出原始异常。清理失败会记录日志；OOM 强制终止、`kill -9` 或机器断电可能使清理无法执行，此时需手动清理残留后重跑，或使用新的 generation。

## 内存与数据规模要求

运行前需要确保单进程可用内存足以容纳本次输入的物品 ID、SID、排序索引和分配结果。候选策略还会加载溢出物品的候选末层码，溢出物品越多、候选数越大，内存需求越高。新增物品处理模式还需要保留相关前缀下已有 SID 的物品数。内存不足时应增加可用内存或减少本次输入规模。

`--batch_size` 只控制读取批次，不限制消解计算和产物构造时的完整数组大小，调小它不能让任意规模的数据都能在固定内存中处理。

三层 SID 单次处理最多支持 715,827,882 行；其他层数的行数上限为 `floor((2^31 - 1) / SID 层数)`。单个物品列表列展开后的物品总数也不能超过 `2^31 - 1`，超出时工具会报错终止。输入拆成多个 Parquet 文件或调小读取批次不会改变这些上限。
