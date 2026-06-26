# SID RQKMeans

## 简介

RQKMeans (Residual K-Means) 是一种语义ID (Semantic ID, 简称 SID) 生成模型, 和 [SID RQVAE](sid_rqvae.md) 一样把物品 embedding 量化成一串离散编码 `(code_0, code_1, ..., code_{n-1})`, 用作生成式推荐的物品token。区别在于: RQKMeans **用 FAISS K-Means 直接对残差做聚类来得到每层码本**——码本是离线一次性拟合出来的。

拟合过程逐层进行: 第0层对原始 embedding 做 K-Means, 得到 `codebook_0` 个聚类中心作为该层码本; 每个样本减去其最近中心得到残差, 第 1 层再对残差做 K-Means, 依此类推。物品的 SID 即各层最近中心的下标元组。相比 RQVAE, RQKMeans 训练更快、无需调梯度超参, 适合在数据充足时快速产出码本。

注意: RQKMeans 当前仅支持 CPU 且单进程——若检测到可用 CUDA 设备或 `world_size > 1` 会直接报错, 因此**务必使用 `--nproc-per-node=1`** 并在 CPU 环境运行。需要多卡/梯度训练时请改用 [SID RQVAE](sid_rqvae.md)。

## 数据格式

RQKMeans 只需要物品 embedding 一列, 通过`fg_mode: FG_DAG` 读取数组列:

| 列名        | 类型           | 说明                                                               |
| ----------- | -------------- | ------------------------------------------------------------------ |
| `item_id`   | string         | 物品 ID (透传列, 模型不消费, 预测时可用 `--reserved_columns` 带出) |
| `embedding` | array\<double> | 物品 embedding, 维度需与 `value_dim` 一致 (示例为 512)             |

> 拟合要求样本数 `N >= max(codebook)`。示例使用较小的 `codebook: 256` 以适配小样本; 生产规模的 `codebook: 8192` 需要远多于 8192 行的数据。样本偏少时 FAISS 会打印 `please provide at least ... training points` 告警, 不影响跑通。

## 配置说明

```
train_config {
    sparse_optimizer { adagrad_optimizer { lr: 0.001 } constant_learning_rate {} }
    dense_optimizer { adam_optimizer { lr: 0.00002 } constant_learning_rate {} }
    num_epochs: 1
    save_checkpoints_steps: 0
    save_checkpoints_epochs: 0
}

feature_configs {
    raw_feature { feature_name: "emb" expression: "item:embedding" value_dim: 512 }
}

model_config {
    feature_groups { group_name: "deep" feature_names: "emb" group_type: DEEP }
    sid_rqkmeans {
        codebook: 256
        codebook: 256
        codebook: 256
        normalize_residuals: true
        faiss_kmeans_kwargs {
            niter: 20
            seed: 42
            verbose: true
            spherical: false
        }
    }
}
```

- train_config: 训练流程配置。RQKMeans 不做梯度训练, 因此优化器与 epoch 数仅为框架训练循环所需的形式配置
  - sparse_optimizer / dense_optimizer: 必填 (训练循环要构建优化器), 但模型只有一个 dummy 参数、损失恒为 0, **因此其学习率不影响最终码本**
  - num_epochs: **设 `1` 即可**; 每个 epoch 只把 embedding 流式写入蓄水池采样 (reservoir), 真正的 FAISS 拟合在训练结束 (`on_train_end`) 时一次性完成, 多跑 epoch 不会迭代/精炼码本
  - save_checkpoints_steps / save_checkpoints_epochs: **必须项，设 `0` 关闭周期性保存**; 拟合好的码本只随训练结束时的最终 checkpoint 持久化 (周期性保存可能会忽略checkpoint落盘, 故关闭)
- feature_configs / feature_groups: 同 RQVAE, 但只需主物品 embedding 一组 (`deep`); 其拼接后的总维度即 K-Means 的向量维度
- sid_rqkmeans: RQKMeans 模型参数
  - codebook: 每层聚类中心数; **列表长度即残差层数 (= SID 的位数)**; 示例 `[256, 256, 256]` (生产常用 `[8192, 8192, 8192]`); 支持非均匀如 `[256, 512, 1024]` (每层独立拟合一个 faiss.Kmeans)
  - normalize_residuals: 每层聚类前是否对残差做 L2 归一化, 默认 `false` (开启后效果更佳)
  - faiss_kmeans_kwargs: 透传给 `faiss.Kmeans(D, K, **kwargs)` 的强类型参数, 未设置的字段回落到 FAISS 自身默认值
    - niter: 每层 K-Means 迭代次数 (FAISS 默认 25)
    - nredo: 重复聚类并取最优的次数 (默认 1)
    - seed: 随机种子, 影响中心初始化/采样 (默认 1234)
    - max_points_per_centroid: 每个中心的最大样本数, FAISS 会下采样到 `K * 该值` (默认 256); 也用于自动推导蓄水池容量
    - min_points_per_centroid: 每个中心的最小样本数 (默认 39), 低于 `K * 该值` 仅告警不报错
    - spherical: 是否做球面 (余弦) K-Means, 即每轮对中心归一化 (默认 false)
    - verbose: 是否打印 FAISS 自身的逐轮日志 (默认 false)
  - train_sample_size: 为拟合做蓄水池采样的目标样本数, 用于限制 host 内存; `0` (默认) 自动取 `max(codebook) * max_points_per_centroid` (即 FAISS 内部会下采样到的规模); 若显式设为小于 `max(codebook)` 的值, 模型在初始化时即报错 (需 `>= max(codebook)`)

## 示例

模型的训练和评估方式同[local_tutorial](../quick_start/local_tutorial.md)，示例数据和配置参数如下：

### 数据

[sid_generation_item_only_sample_4w.parquet](https://tzrec.oss-accelerate.aliyuncs.com/data/models/sid_generation_item_only_sample_4w.parquet)

### 配置文件

[sid_rq_kmeans.config](https://tzrec.oss-accelerate.aliyuncs.com/config/models/sid_rq_kmeans.config)

### 训练参数

```bash
OMP_NUM_THREADS=$(nproc) \
    torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path sid_rq_kmeans.config \
    --train_input_path "data/sid_example/item_only/*.parquet" \
    --eval_input_path "data/sid_example/item_only/*.parquet" \
    --model_dir experiments/sid_rqkmeans
```

`--nproc-per-node` 必须为 `1`。训练时每个 batch 只把 embedding 写入蓄水池并返回占位 (全 0) 编码; 训练结束时日志会打印 `[SidRqkmeans.on_train_end] fitting FAISS on N samples` 以及逐层聚类信息, 随后做一次 eval 输出 `mse` / `rel_loss` / `unique_sid_ratio`。

**`OMP_NUM_THREADS=$(nproc)` 为必须配置项，否则默认设置为1，影响模型训练推理速度。**

### 模型输出

预测输出与输入 `dataset_type` 一致, 每行包含:

- codes: `array<int64>`, 即该物品的 SID, 长度等于 `codebook` 层数, 每个元素为对应残差层的中心下标 (取值范围 `[0, codebook_i)`)。例如 `[8, 31, 26]`。
- item_id: 由 `--reserved_columns` 透传的原始物品 ID。
