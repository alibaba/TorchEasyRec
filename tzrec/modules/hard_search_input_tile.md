# Hard-search 紧凑输入契约

`HARD_SEARCH_INPUT_TILE=1` 在 `INPUT_TILE=2/3` 的 FX/TorchScript 导出中启用。
`model_acc.json` 写入 `hard_search_input_tile` 对象，`version` 为整数 `1`，
`features` 将所有生成的 hard-search FG 节点完整名称映射到其唯一的
`sequence_name`，包括派生节点与 FG 中间节点。Processor 仅在读取此标记后发送紧凑输入。

每个序列组有独立的 `b` 和 `<sequence_name>__inverse_indices`：

- `inverse_indices` 是长度为 `B` 的 int64 Tensor，取值范围为 `[0, b)`。
- 每个紧凑特征的原始 `.lengths` 有 `b` 行，`.values` 只含这 `b` 行的数据。
- 多值序列的 `.lengths` 表示事件数量，`.key_lengths` 表示每个事件的值数量；
  分布式 EC 输入 `__lengths` 表示每个紧凑行的值总数。
- 缺失查询 key 指向一份共享空输入。其解析结果遵循 FG 默认值，不能假定解析后长度为零。
- 普通候选保持 `B` 行，普通用户特征沿用原有 INPUT_TILE 语义。

DataParser 为固定 stride 的 KJT 补零长度行到 `B`，不重复 ID。原始紧凑长度
通过 `Batch.additional_infos` 保留，`Batch.to_dict()` 用它覆盖补零后的长度。
EC 查表后裁去零长度行，再执行 `fx_mark_seq_ec_jt`；多值序列先对每事件 pooling，
随后以紧凑事件长度重建 JT。Padding 处理 `b` 行，进入序列编码器前通过
`index_select` 恢复 `B` 行及长度。候选 query 保留各自的 attention 计算。

EBC 的 FX 标记位于候选恢复之前。普通导出中 EBC 输出包含 KJT 补零行；Processor
分布式 EBC 查表使用 `b` 行，拼接为 KeyedTensor 前同样补零到 `B`，由模型按
inverse indices 恢复结果。分布式 EC 不补行，直接发送紧凑 values 和 lengths。
派生的非序列 dense 输入在 DataParser 拼接之前恢复；其原始值也通过
additional_infos 保留，避免导出样本往返时重复展开。

共享只允许依赖同一份历史及共享 user/constant 输入的特征。跨历史来源或依赖
候选 item/context 的 FG 派生特征在启用优化时拒绝。特征表和 checkpoint 参数名
不变；离线预测继续按样本提供历史，由预测入口添加恒等映射。
