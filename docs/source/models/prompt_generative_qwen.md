# Prompt 原生生成式推荐（PromptGenerativeQwen）

以 Qwen 为骨干，把用户历史的语义 ID（SID）拼进 prompt，让模型直接生成下一个物品的 SID。

prompt 的模板、槽位、SID 空间与词表都由新的 `prompt_config` 描述，`model_config` 只保留属于 LM 的部分。

## 1. 数据准备

SID 必须以 **offset 形式**进入 tzrec，即 SID 生成工具 `resolve_sid_collisions` 输出的 `offset_codebook` 列：

```
第 l 层的取值 = level_offsets[l] + code        code 属于 [0, codebook[l])
```

以 `codebook: 4 4 4` 为例，`level_offsets` 为 `[0, 4, 8]`，因此一个 item 的三层取值分别落在 `[0,4)`、`[4,8)`、`[8,12)`。

```{warning}
只读 `offset_codebook`。`codebook` 与 `origin_codebook` 两列同样格式合法，但前者未加 offset、后者是冲突解析**之前**的 SID；误用不会报格式错，而是训练出静默错误的模型。assembler 的 band 校验能挡住未加 offset 的列，但挡不住手工对 `origin_codebook` 施加 offset 得到的流。
```

一行历史是若干个 item 的三层 code 依次拼平，长度必须是层数的整数倍。

## 2. 配置

一个最小可运行的配置：

```
data_config {
    batch_size: 4
    dataset_type: ParquetDataset
    fg_mode: FG_NONE
    label_fields: "answer"
}

feature_configs {
    sequence_raw_feature { feature_name: "hist" expression: "user:hist" }
}
feature_configs {
    sequence_raw_feature { feature_name: "answer" expression: "item:answer" }
}

prompt_config {
    tokenizer: "path/to/tokenizer.json"
    prompt:   "用户历史行为为：{{hist}}。请预测下一个商品："
    response: "{{answer}}"
    sid_space { codebook: 256 codebook: 256 codebook: 256 }
    max_length: 4096
}

model_config {
    prompt_generative_qwen {
        hf_model_id: "Qwen/Qwen2.5-0.5B"
        common {
            beam_widths: 100
            beam_widths: 200
            beam_widths: 400
            num_return_sequences: 50
        }
    }
}
```

### prompt_config

| 字段                      | 说明                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `tokenizer`               | **基础** tokenizer 的路径或 hub id。注意它与 `hf_model_id` 不同：后者只表示权重，且只在冷启动时读取一次 |
| `prompt`                  | 模板。`{{name}}` 之间的静态文本自动成为相邻槽位的前后缀，无需逐槽位配置                                 |
| `response`                | 监督目标。定义 loss 覆盖的范围；推理时不生成该段                                                        |
| `sid_space.codebook`      | 每层的 SID 词表大小                                                                                     |
| `sid_space.manifest_path` | 可选。指向 SID manifest，编译期与 `codebook` 逐元素比对，不一致直接报错                                 |
| `max_length`              | 校验上限，**不是**截断开关：超长的行会报错，不会被截断                                                  |

### 槽位如何被推导

`{{name}}` 默认解析为同名特征。槽位的填充方式不需要配置，由成员特征推导：

| 槽位成员                                                 | 填充方式  | 说明                                                |
| -------------------------------------------------------- | --------- | --------------------------------------------------- |
| 单个序列特征且不声明 embedding（`sequence_raw_feature`） | INLINE    | SID 直接进入 token 流，与答案共享 embedding         |
| 其他情形（如 `sequence_id_feature`、标量特征、多成员）   | PROJECTED | 走自己的 embedding 表，再经一次投影抵达 LM 隐层维度 |

PROJECTED 槽位在 token 流中占位为 sentinel，真实取值在前向时写入对应位置。

### model_config

| 字段                   | 说明                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `hf_model_id`          | 预训练权重的 hub id 或本地目录                              |
| `beam_widths`          | 每层一个宽度，长度必须等于 `codebook` 的层数                |
| `num_return_sequences` | 不得超过最后一层的宽度                                      |
| `param_dtype`          | 主权重精度，默认 FP32。bf16 会让 Adam 的小更新在 ULP 下丢失 |

## 3. 训练

```bash
torchrun --master_addr=localhost --master_port=32555 --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.train_eval --pipeline_config_path prompt_qwen.config
```

续跑加 `--continue_train`。

每个 `model.ckpt-N/` 除权重外还会写出 `prompt/` 目录：

```
model.ckpt-N/prompt/
  sid_space.json       解析后的 SID 空间：codebook、level_offsets、band、target_vocab
  prompt_plan.json     assembler 的遍历顺序与各项上界
  prompt_hashes.json   vocab_hash 与 plan_hash
  tokenizer/           扩展后的 tokenizer（含 SID atom）
```

即 checkpoint 自带词表契约，服务端无需另行配置。

## 4. 预测

```bash
torchrun --master_addr=localhost --master_port=32555 --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.predict --pipeline_config_path experiments/run/pipeline.config \
    --predict_input_path 'data/*.parquet' --predict_output_path out
```

输出列 `generated_sids`，形状为 `(num_return_sequences, 层数)`，取值是**局部 0-based** code，可直接与 SID 映射表的 `codebook` 列对齐。

## 5. 导出

只支持导出为 HuggingFace 目录：

```
export_config { export_format: HF }
```

```bash
torchrun ... -m tzrec.export --pipeline_config_path experiments/run/pipeline.config \
    --export_dir exported
```

产出目录同时包含权重与 prompt 契约，可直接被 `AutoModelForCausalLM.from_pretrained` 加载：

```
exported/
  config.json  generation_config.json  model.safetensors
  prompt/      sid_space.json  prompt_plan.json  prompt_hashes.json  tokenizer/
```

```{note}
本模型不支持 TorchScript 导出。它的输入是 dataloader 组装出的 token 流，而导出期的伪造 batch 无法提供。配置为默认的 TORCHSCRIPT 时会直接报错并提示改用 HF。
```

## 6. 常见问题

**`SID values must already carry their level offset ... Read the offset_codebook column`**

读错了列。改用 `offset_codebook`，见第 1 节。

**`prompt vocabulary does not match checkpoint`**

`codebook`、`atom_token_format` 或 tokenizer 变了，与该 checkpoint 训练时的词表不一致。这是硬失败：解码 band 会指向这批权重从未学过的行，继续跑只会产出看似合理的错误结果。要么改回原配置，要么从头训练。

若只是模板或槽位变了（`plan_hash` 不同、`vocab_hash` 相同），只会告警，权重仍可用。

**`beam_widths has N entries but the codebook has M levels`**

每层一个宽度，两者长度必须相等。

**`assembled row X is N tokens, over max_length`**

超长的行不会被截断。请在特征上用 `sequence_length` 限制历史长度，而不是调大 `max_length`。

**`static_prefix_len is 0`（告警）**

模板开头就是一个槽位，导致服务端前缀缓存无内容可共享。把静态指令文本放在最前、变长槽位放在最后即可。

**`a prompt-native model exports to a HuggingFace directory, not TorchScript`**

见第 5 节，设置 `export_config.export_format: HF`。
