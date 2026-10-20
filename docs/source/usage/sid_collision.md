# SID 冲突后处理工具

`resolve_sid_collisions` 处理 SID 预测结果：保留每个 SID 容量内的物品，将超出容量的物品重新分配到同一 SID 前缀下的末层码。前缀是除最后一层外的 SID，不会在后处理时改变。所有输入物品都会保留；`candidate`、`iterative` 和 `random` 无法分配的物品保留原 SID，`uniform` 会继续均匀分配，最终仍超过容量的物品计为未解决。

输入包含 `item_id` 和 `codes`；默认的 `candidate` 策略及 `iterative` 策略还需要 `candidate_codes`。该字段由 SID 模型在预测阶段输出，需要在模型配置中启用 `candidate_output_config.enabled`，相关配置参见 [SID 模型文档](../models/sid_model.rst)。Parquet 的 `codes` 是整数列表，`candidate_codes` 是按候选顺序展开的 `topk × SID 层数` 个整数。CSV 使用逗号分隔的字符串表示这些列表。

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
    --strategy candidate \
    --batch_size 100000
```

`--codebook` 与生成预测结果的模型配置一致，`--max_items_per_codebook` 指定每个 SID 最多容纳的物品数。`--nproc_per_node=8` 由 torchrun 创建 8 个进程，工具将 SID 前缀空间划为连续区间，每个进程负责一个区间。同一前缀的全部物品在同一个进程中处理，无需提前拆分输入或使用额外的调度脚本。单进程使用 `--nproc_per_node=1`，也可直接通过 `python -m` 启动。

各进程扫描完整输入批次，只保留自己负责的前缀；增加进程数可减少每个进程保留的数组，但读取速度不一定同比提高，前缀分布不均也会使工作量不同。单层 SID 只有一个空前缀，由 rank 0 处理。各进程写出自己的 `part-*` 文件，rank 0 在全部写出完成后发布一个 `manifest.json`，其中的统计和产物行数为全局汇总值。所有进程必须能访问相同的输入和输出路径。

## 新增物品处理模式

输入只包含新增物品，通过 `--from_generation` 指定已发布的一代。已有物品的 SID 不移动，新增物品根据已有的 SID 物品数继续分配位置。追加可以使用不同的进程数：

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
    -m tzrec.tools.sid.resolve_sid_collisions \
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
- `iterative`：通过多轮请求和竞争分配候选，优先接收将目标 SID 排在更靠前候选位置的物品；候选耗尽时保留原 SID，未解决数量可能多于 `candidate`。
- `uniform`：无需模型候选，将溢出物品分配到同一前缀内负载最小的末层码；前缀总容量不足时继续均匀分配，超过 SID 容量的物品计为未解决。
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

当物品结果使用 Parquet 输出时，`item_to_sid` 和 `delta_item_to_sid` 每行对应一个物品，字段如下。输出字段名固定，不随输入的 `--item_id_field` 和 `--code_field` 改变。

| 字段              | Parquet 类型           | 含义                                                                                       |
| ----------------- | ---------------------- | ------------------------------------------------------------------------------------------ |
| `item_id`         | 与输入物品 ID 类型一致 | 物品 ID                                                                                    |
| `origin_codebook` | `list<int64>`          | 后处理前的原始 SID，未做层间偏移                                                           |
| `codebook`        | `list<int64>`          | 后处理后的最终 SID，未做层间偏移                                                           |
| `offset_codebook` | `list<int64>`          | 最终 SID 加上各层码本大小的累计偏移                                                        |
| `index`           | `int64`                | 物品在最终 SID 内的位置，从 1 开始；大于 `--max_items_per_codebook` 表示该物品的冲突未解决 |

`sid_to_items` 和 `delta_sid_to_items` 每行对应一个 SID，字段如下。新增物品处理模式下，`delta_sid_to_items` 的 `itemids` 也包含该 SID 中已有的物品。

| 字段              | Parquet 类型             | 含义                                                               |
| ----------------- | ------------------------ | ------------------------------------------------------------------ |
| `codebook`        | `list<int64>`            | 后处理后的最终 SID，未做层间偏移                                   |
| `offset_codebook` | `list<int64>`            | 最终 SID 加上各层码本大小的累计偏移                                |
| `itemids`         | `list<输入物品 ID 类型>` | 该 SID 的完整物品列表，按 SID 内位置排列，第一个物品对应 `index=1` |

`offset_codebook` 可直接用于下游 GenRec 的 SID 历史和目标字段，无需再次进行层间偏移。GenRec 的 `prompt_config.sid_space.codebook` 应与本工具的 `--codebook` 一致。例如码本为 `[64, 64, 64]` 时，`codebook=[1, 2, 3]` 对应 `offset_codebook=[1, 66, 131]`。GenRec 会自动追加语言模型基础词表的偏移。

只有正式的 `manifest.json` 出现后，本代结果才可供下游使用。下游应根据 manifest 记录的位置和格式读取产物。目标 generation 的目录或 ODPS 分区已存在时，工具会报错退出，不覆盖已有数据。rank 0 负责准备 generation，并在捕获到异常时清理本次新建的 generation，包括独立存储位置中的本代产物，再抛出原始异常。清理失败会记录日志；其他 rank 失败后 torchrun 终止 rank 0、OOM 强制终止、`kill -9` 或机器断电可能使清理无法执行，此时需手动清理残留后重跑，或使用新的 generation。

## 内存与数据规模要求

运行前需要确保每个进程的可用内存足以容纳其负责前缀内的物品 ID、SID、排序索引和分配结果。`candidate` 和 `iterative` 还会加载溢出物品的候选末层码，溢出物品越多、候选数越大，内存需求越高；`random` 会生成候选矩阵，`uniform` 无需候选。新增物品处理模式还需要保留相关前缀下已有 SID 的物品数。内存不足时应增加可用内存、调整进程数或减少本次输入规模，并关注各进程的负载是否均衡。

`--batch_size` 只控制读取批次，不限制消解计算和产物构造时的完整数组大小，调小它不能让任意规模的数据都能在固定内存中处理。

三层 SID 每个 rank 单次构造最多支持 715,827,882 行；其他层数的行数上限为 `floor((2^31 - 1) / SID 层数)`。单个物品列表列展开后的物品总数也不能超过 `2^31 - 1`，超出时工具会报错终止。输入拆成多个 Parquet 文件或调小读取批次不会改变这些上限。
