# GenRec

## 简介

GenRec 是基于 HuggingFace Causal LM 的生成式召回模型。它将用户画像、行为历史和文本等特征按 Prompt 模板组合，使用自回归模型直接生成下一个物品的语义 ID（Semantic ID，简称 SID），再通过 SID 到物品的映射表得到召回结果。

当前模型配置为 `genrec_causal_lm_model`，骨干模型使用 `hf_model_name_or_path` 指定，支持 Qwen2.5/3 等 Decoder-Only 模型。骨干需要支持 `logits_to_keep` 和 HuggingFace 的 `Cache` 接口。物品 SID 可以由 [SID RQVAE](sid_rqvae.md) 或 [SID RQKMeans](sid_rqkmeans.md) 生成；GenRec 训练和解码使用的码本必须与数据中的 SID 一致。

GenRec 支持两种特征填充方式，均由特征配置自动推导：

- INLINE：SID 编码或文本分词 ID 直接进入 LM 的 token 序列，使用 LM 的 embedding 表。
- PROJECTED：结构化特征经自己的 embedding 或数值特征处理后，投影到 LM 的隐藏维度，在 Prompt 中占据一个或多个位置。

GenRec 使用 `prompt_config` 进行模板、槽位和 SID 空间配置，`model_config` 负责 LM 参数和解码参数。

## 数据格式

### SID 编码

SID 的每一层有独立的码本。**历史物品和训练标签都必须使用按层偏移后的编码（offset code），不能直接使用每层从 0 开始的编码，也不能预先加上 LM 的词表大小。**

```
level_offsets[l] = sum(codebook[:l])
offset_code[l] = level_offsets[l] + local_code[l]
LM token_id[l] = base_vocab_size + offset_code[l]
```

例如，`codebook: [256, 256, 256]` 的层偏移为 `[0, 256, 512]`：

| 表示方式            | 示例                        | 用途                                          |
| ------------------- | --------------------------- | --------------------------------------------- |
| 每层从 0 开始的编码 | `[3, 5, 7]`                 | SID 模型的 `codes` 输出、GenRec 的预测输出    |
| 按层偏移后的编码    | `[3, 261, 519]`             | GenRec 的历史输入和训练标签                   |
| LM token ID         | `[V + 3, V + 261, V + 519]` | 模型内部使用，`V` 为基础 tokenizer 的词表大小 |

如果使用 `tzrec.tools.sid.resolve_sid_collisions` 生成 SID 映射表，读取最终的 `offset_codebook` 列构建历史和标签；`codebook` 列未加层偏移，`origin_codebook` 列是碰撞处理前的编码。物品映射、历史、标签和可选的 SID manifest 应来自同一套 SID 数据。

### 样本字段

以下配置使用 Parquet 原始特征和 `fg_mode: FG_DAG`。一行对应一个用户的历史和一个目标物品：

| 列名              | 类型                  | 示例及说明                                                                           |
| ----------------- | --------------------- | ------------------------------------------------------------------------------------ |
| `user_id`         | string                | `"u1"`，画像特征，也可在预测结果中透传                                               |
| `user_age`        | double                | `34.0`，数值画像特征                                                                 |
| `hist__sid_codes` | array\<array\<int64>> | `[[3, 261, 519], [9, 270, 522]]`，按顺序排列的历史物品，每个物品包含三层 offset code |
| `title`           | string                | `"运动鞋"`，作为 INLINE 文本输入                                                     |
| `answer`          | array\<int64>         | `[8, 260, 524]`，目标物品的三层 offset code，仅训练和评估需要                        |

`hist__sid_codes` 是 `sequence_name: "hist"` 与子特征输入字段 `sid_codes` 组合后的列名，模板引用的特征名则为 `hist__sid`。历史每个物品的编码数和 `answer` 的长度都应等于码本层数；不要把历史编码存成嵌套数组的字符串。

预测数据可以不包含 `answer` 列，但配置仍需保留 `data_config.label_fields: "answer"` 和 `prompt_config.response: "{{answer}}"`，供编译器解析监督段。预测时不会读取或拼接该段。

## 配置说明

下面是使用画像投影、SID 历史和文本槽位的完整配置示例。将其保存为 `experiments/genrec/genrec.config`，并准备训练数据、评估数据及本地骨干模型目录。示例中的 `data/genrec/backbone` 应包含与基础 `tokenizer.json` 对应的 HuggingFace 模型配置和预训练权重。

```
train_input_path: "data/genrec/train/*.parquet"
eval_input_path: "data/genrec/eval/*.parquet"
model_dir: "experiments/genrec/train"

train_config {
    sparse_optimizer {
        adagrad_optimizer { lr: 0.01 }
        constant_learning_rate {}
    }
    dense_optimizer {
        adam_optimizer { lr: 0.0001 }
        constant_learning_rate {}
    }
    num_epochs: 1
    save_checkpoints_epochs: 1
    mixed_precision: "BF16"
}
eval_config {}
export_config {
    mixed_precision: "BF16"
    use_dense_ema: false
}
data_config {
    batch_size: 32
    dataset_type: ParquetDataset
    fg_mode: FG_DAG
    label_fields: "answer"
    num_workers: 2
}
feature_configs {
    id_feature {
        feature_name: "user_id"
        expression: "user:user_id"
        embedding_dim: 32
        hash_bucket_size: 100000
    }
}
feature_configs {
    raw_feature {
        feature_name: "user_age"
        expression: "user:user_age"
        value_dim: 1
    }
}
feature_configs {
    sequence_feature {
        sequence_name: "hist"
        sequence_length: 50
        sequence_delim: ";"
        features {
            id_feature {
                feature_name: "sid"
                expression: "item:sid_codes"
                value_dim: 3
                num_buckets: 768
            }
        }
    }
}
feature_configs {
    tokenize_feature {
        feature_name: "title"
        expression: "user:title"
        tokens_as_sequence: true
        sequence_length: 32
    }
}
prompt_config {
    tokenizer_path: "data/genrec/backbone/tokenizer.json"
    prompt: "用户画像：{{profile}}。历史物品：{{hist__sid}}。查询文本：{{title}}。推荐下一个物品："
    response: "{{answer}}"
    slots {
        name: "profile"
        feature_names: "user_id"
        feature_names: "user_age"
        projection {
            mlp { hidden_units: 128 }
            bias: true
        }
    }
    sid_space {
        codebook: 256
        codebook: 256
        codebook: 256
    }
    max_length: 256
}
model_config {
    genrec_causal_lm_model {
        hf_model_name_or_path: "data/genrec/backbone"
        common {
            lm_parameter_dtype: FP32
            attn_kernel: "sdpa"
            beam_widths: 100
            beam_widths: 200
            beam_widths: 400
            num_return_sequences: 50
        }
    }
}
```

### prompt_config

| 参数                                 | 说明                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `tokenizer_path`                     | 必填，基础 tokenizer JSON 文件的本地路径。使用未扩展 SID token 的词表，并与骨干模型对应                                   |
| `prompt`                             | 输入模板，静态文本直接分词，`{{name}}` 指定特征或槽位的位置；不会自动套用聊天模板                                         |
| `response`                           | 监督目标模板，示例为 `{{answer}}`，其中的占位符引用 `data_config.label_fields`；预测时也必须保留配置                      |
| `slots`                              | 可选，声明槽位的 `name`、`feature_names`、`projection` 和 `projection_name`                                               |
| `sid_space.codebook`                 | 必填，每层的码本大小，各项必须为正数；列表长度决定生成 SID 的层数                                                         |
| `sid_space.token_format`             | SID token 的文本格式，默认 `<\|sid_{i}\|>`，必须包含 `{i}`；`i` 是按层偏移后的编码                                        |
| `sid_space.vocab_pad_to_multiple_of` | LM embedding 表大小对齐的倍数，默认 `128`；额外的补齐位置不参与 SID 解码                                                  |
| `sid_space.manifest_path`            | 可选，SID bundle 的本地 manifest JSON 路径。其 `codebook` 必须与配置一致，并包含非空 `bundle_uuid`；导出配置会记录该 UUID |
| `max_length`                         | 编译时的长度校验上限，默认 `0`。计算范围包含静态文本、槽位和 response；已知最大长度超过此值时编译报错                     |
| `sentinel_token`                     | PROJECTED 槽位的占位 token，默认 `<\|pg_hole\|>`；模型前向时用投影向量替换该位置                                          |

模板中的 `{{name}}` 优先匹配 `slots.name`；没有显式声明时，按同名特征创建单成员槽位。例如 `{{hist__sid}}` 和 `{{title}}` 不需要额外的 `slots` 配置。显式声明的槽位必须在模板中被引用。

INLINE 和 PROJECTED 是编译器根据槽位引用的特征及其组合自动推导的填充方式，无需显式配置。单成员序列槽位中，`id_feature` 或 `tokenize_feature` 省略 `embedding_dim` 时使用 INLINE，数值序列特征不使用独立 embedding 时也使用 INLINE；标量槽位、带 embedding 的序列槽位或多成员槽位使用 PROJECTED。`projection` 仅配置 PROJECTED 槽位的投影方式，不用于选择填充方式。

- PROJECTED 槽位的成员必须全部为标量或全部为序列，不能混合两者。序列成员应按相同顺序对应同一批物品，并保持长度一致。
- PROJECTED 槽位的特征组由编译器自动生成，无需手工添加 `model_config.feature_groups`。输入维度由特征决定，输出维度取骨干 LM 的 `hidden_size`。
- `projection` 省略或为空时使用线性投影；设置 `projection.mlp.hidden_units` 时，先经过 MLP，再经过输出到 LM 隐藏维度的线性层。不要配置空的 `mlp {}`。
- 相同的 `projection_name` 可让不同槽位共享投影参数，但它们的投影配置和输入总维度必须相同。
- INLINE 槽位不能配置 `projection`。

`max_length` 不会在运行时逐行测量或截断输入。应通过特征的 `sequence_length` 控制历史和文本的长度；示例中的 50 个历史物品最多占据 `50 * 3 = 150` 个 SID token。PROJECTED 序列槽位必须声明长度上限；INLINE 多值序列还需要固定每个元素的值数，编译器才能推导总长度上界。

基础 tokenizer 不应已经包含将要添加的 SID token 或 sentinel token。当前实现从 `<|im_end|>` 和 `<|endoftext|>` 中解析 EOS、PAD，基础词表至少需要包含其中一个。

### genrec_causal_lm_model

| 参数                          | 说明                                                                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `hf_model_name_or_path`       | HuggingFace 模型 ID 或本地目录，默认 `Qwen/Qwen2.5-0.5B`。用于构建骨干结构，首次训练时加载预训练权重；恢复训练时从 checkpoint 读取权重 |
| `common.lm_parameter_dtype`   | LM 参数精度，`FP32`、`BF16` 或 `FP16`，默认 `FP32`；计算的混合精度由 `train_config.mixed_precision` 控制                               |
| `common.attn_kernel`          | HuggingFace 注意力实现，默认 `"sdpa"`。`"flash_attention_2"` 使用无 padding 的变长训练路径；其他实现使用按行左侧 padding 的路径        |
| `common.beam_widths`          | 必填，每层一个 beam 宽度，长度必须等于码本层数，且每项至少为 `1`；可使用固定宽度或逐层递增宽度                                         |
| `common.num_return_sequences` | 必填，返回的 SID 数量，至少为 `1`，不得超过最后一层实际可保留的 beam 数                                                                |
| `common.generated_sids_key`   | 预测输出列名，默认 `"generated_sids"`                                                                                                  |
| `common.ignore_index`         | 非监督位置的标签值，默认 `-100`，这些位置不参与交叉熵损失                                                                              |

每层实际 beam 数为 `min(配置宽度, 上一层实际 beam 数 * 本层码本大小)`，第一层的上一层 beam 数为 `1`。例如码本为 `[4, 4, 4]`、宽度为 `[100, 200, 400]` 时，实际宽度是 `[4, 16, 64]`，最多返回 64 个 SID。

默认的 FP32 参数可以避免低精度参数下过小的优化器更新被舍入；示例通过 `mixed_precision: "BF16"` 使用 BF16 计算。切换为 `flash_attention_2` 时，需要可用的 NVIDIA GPU、与项目环境匹配的 FlashAttention 依赖及 BF16/FP16 计算精度。该优化用于训练和评估前向，beam 解码仍使用左侧 padding 和 KV cache。

模型自动计算监督段的 `ce_loss`，并在评估时输出同名指标，无需额外声明 `model_config.losses` 或 `metrics`。评估采用真实答案作为输入的 teacher forcing；召回的 HitRate、Recall 等指标需根据预测 SID 与真实物品另行计算。

## 示例

环境准备方式同 [local_tutorial](../quick_start/local_tutorial.md)。以下命令均在配置好项目依赖的 Docker 环境中、仓库根目录下执行；从源码运行时设置 `PYTHONPATH=.`。示例使用单机单进程，多卡训练可调整 `--nproc-per-node`。

### 训练和评估

```bash
PYTHONPATH=. torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m tzrec.train_eval \
    --pipeline_config_path experiments/genrec/genrec.config
```

断点恢复使用相同配置并追加 `--continue_train`。恢复、评估和预测需要保留对应的基础 tokenizer 与骨干配置文件；码本大小、SID token 格式和基础词表应与训练时一致。

独立评估：

```bash
PYTHONPATH=. torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m tzrec.eval \
    --pipeline_config_path experiments/genrec/train/pipeline.config
```

评估默认读取 `model_dir` 下的最新 checkpoint，也可以通过 `--checkpoint_path` 指定。

### 离线预测

生成 SID 使用 [Checkpoint 预测](../usage/predict.md#%E9%A2%84%E6%B5%8Bcheckpoint)：

```bash
PYTHONPATH=. torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m tzrec.predict \
    --pipeline_config_path experiments/genrec/train/pipeline.config \
    --predict_input_path 'data/genrec/predict/*.parquet' \
    --predict_output_path experiments/genrec/predict_result \
    --reserved_columns user_id \
    --output_columns generated_sids
```

省略 `--checkpoint_path` 时使用 `model_dir` 下的最新 checkpoint。预测 batch 可以通过 `--batch_size` 调整；较大的 beam 宽度和历史长度会增加 KV cache 的显存占用。

输出列 `generated_sids` 的每行形状为 `(num_return_sequences, 码本层数)`，各 SID 按 beam 累计得分从高到低排列。例如 `[[3, 5, 7], [8, 4, 12], ...]`。**输出是每层从 0 开始的编码**，应与最终 SID 映射表的 `codebook` 列对齐；若使用 `offset_codebook` 做关联，需先加回层偏移。

解码每层只允许生成该层码本对应的 token，生成长度固定为码本层数，不依赖 EOS 停止。该约束保证每层编码合法，但不会检查完整 SID 是否存在于物品映射表；召回时应过滤无映射的 SID，并展开同一 SID 对应的物品列表。

### 导出和服务

```bash
PYTHONPATH=. QUANT_EMB=0 \
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m tzrec.export \
    --pipeline_config_path experiments/genrec/train/pipeline.config \
    --export_dir experiments/genrec/export
```

默认导出包含两部分：

- TorchScript 特征前端及 `fg.json`、`pipeline.config` 等配置，用于组装 Prompt 和计算 PROJECTED 槽位向量。
- LM 的 `model.safetensors`、组合式 `config.json`、扩展后的 `tokenizer.json` 和 `tokenizer_config.json`，供生成式推理引擎使用；骨干提供生成配置时还会导出 `generation_config.json`。

导出的 `scripted_model.pt` 仅计算前端结果，不包含 LM 解码逻辑。离线生成 `generated_sids` 时，使用上一节的 checkpoint 预测命令。

部署时需要支持 `GenRecForCausalLM`（`model_type: "genrec"`）的推理引擎，将前端产出的 Prompt 和投影向量送入 LM。`config.json.text_config` 保存骨干配置，`sid_space` 保存 `codebook`、`base_vocab_size` 和 `token_format`；配置了 manifest 时还会保存 `bundle_uuid`，供服务核对所使用的 SID 映射。

`export_config.mixed_precision` 指定服务端加载 LM 的精度。当前前端导出仅支持 TorchScript，需关闭 `ENABLE_AOT`、`ENABLE_TRT` 和 `USE_RTP`；GenRec 的 HF 权重导出不支持 Dense EMA，启用 EMA 训练时须显式设置 `export_config.use_dense_ema: false`。量化及分布式 embedding 的通用参数参见 [导出](../usage/export.md)。

仓库中的 [GenRec mock 配置](https://github.com/alibaba/TorchEasyRec/blob/master/tzrec/tests/configs/genrec_causal_lm_model_mock.config) 还提供了多个画像槽位共享投影、多成员序列投影和 INLINE 文本的组合示例。它的模型、词表和数据由集成测试准备，不能直接作为业务训练数据运行。

## 常见问题

| 现象或报错                                                        | 排查方式                                                                                                       |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `prompt slot [...] names [...] which are not in ...`              | 检查输入占位符是否引用正确的特征名；response 占位符引用的列须在 `data_config.label_fields` 中声明              |
| `prompt_config.response is required`                              | 训练、评估、预测配置都保留 `response`；预测数据可以不提供标签列                                                |
| `an inline SID history needs value_dim` 或 `declares an id space` | SID 历史设置 `value_dim = 码本层数`、`num_buckets = 各层码本大小之和`，省略 embedding 和其他 ID 映射参数       |
| INLINE 文本 tokenizer 不一致                                      | `vocab_file` 使用与 `tokenizer_path` 相同的基础词表，或省略后让配置加载器自动填充                              |
| `the prompt can reach ... but max_length is ...`                  | 根据静态文本、槽位和答案的总长度调整上限，或减小特征的 `sequence_length`；`max_length` 不负责截断              |
| `PROJECTED and unbounded`                                         | 为序列槽位的特征设置 `sequence_length`                                                                         |
| `beam_widths has ... entries` 或返回数量超出 beam                 | 每层配置一个正的宽度，返回数量不能超过最后一层经过候选数限制后的宽度                                           |
| `nothing in this batch can be supervised`                         | 检查标签是否为空，以及答案前是否至少有一个 Prompt token；可在模板中保留静态指令文本                            |
| `SID tokens are already in the base tokenizer`                    | 输入未扩展的基础 tokenizer；导出的 tokenizer 已包含 SID token，应供服务使用                                    |
| manifest 不匹配或缺少 `bundle_uuid`                               | 使用与历史、标签对应的完整 SID bundle manifest，并核对码本大小                                                 |
| SID 每层取值合法，但召回结果为空或错位                            | 核对输入是否使用最终 `offset_codebook`，输出是否按每层从 0 开始的编码关联最终 `codebook`，以及映射版本是否一致 |
