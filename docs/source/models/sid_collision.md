# SID 冲突消解

`resolve_sid_collisions` 处理 SID 预测结果：保留每个 SID 容量内的物品，将超出容量的物品
重新分配到同一 SID 前缀下的末层码。前缀是除最后一层外的 SID，不会在消解时改变。
所有输入物品都会保留，无法分配的物品保留原 SID，并计为未解决。

输入包含 `item_id` 和 `codes`；默认的 `candidate` 策略还需要 `candidate_codes`。
Parquet 的 `codes` 是整数列表，`candidate_codes` 是按候选顺序展开的 `topk × SID 层数`
个整数。CSV 使用逗号分隔的字符串表示这些列表。

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

`--codebook` 与生成预测结果的模型配置一致，`--max_items_per_codebook` 指定每个 SID
最多容纳的物品数。工具当前为单进程，通过 `python -m` 启动。

## 新增物品处理模式

输入只包含新增物品，通过 `--from_generation` 指定已发布的一代。已有物品的 SID 不移动，
新增物品根据已有的 SID 物品数继续分配位置：

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

每一代的 `item_to_sid` 和 `sid_to_items` 都包含完整语料。新增物品处理还写出
`delta_item_to_sid` 和 `delta_sid_to_items`，后者包含本次获得新物品的完整 SID 分组，
方便下游进行幂等更新。新增物品 ID 如果已存在于上一代，会报错。

## 选择策略

省略 `--strategy` 时默认使用 `candidate`。

- `candidate`：按稳定的物品顺序，选择第一个仍有容量的候选末层码。
- `random`：无需模型候选，为溢出物品生成确定性的随机末层码候选，再按相同的规则分配。
  `--random_num_candidates` 控制每个溢出物品的抽样次数，默认 64，最多为末层码本大小减一。
  抽样允许重复，命中原码或重复候选时不补抽，因此前缀有剩余容量也可能无法分配。

`--rate_only` 计算统计并跳过产物写出，可用于比较不同策略。

## 产物与限制

产物位于 `<output_path>/<generation>/<artifact>`，`manifest.json` 在全部产物写出后生成。
`item_to_sid` 默认使用输入格式，可通过 `--item_to_sid_writer_type` 指定格式，或通过
`--item_to_sid_root` 指定独立存储位置。`sid_to_items` 始终使用 Parquet，保留原生列表。

Parquet 产物的 SID 和物品 ID 列表使用普通 Arrow `List`，其 offsets 为 int32。
构造列表时，展开后的元素总数超过 `2^31 - 1` 会报错终止，不自动分块或切换为 LargeList。
对于三层 codes，单次构造最多容纳 715,827,882 行；itemids 列表按展开后的物品总数检查。
输入拆成多个 Parquet 文件不会改变单次构造数组的上限。

`--batch_size` 控制读取批次，不限制消解计算和产物构造时的完整数组大小。
