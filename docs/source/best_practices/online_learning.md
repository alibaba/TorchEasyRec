# 天级增量 + 流式训练的在线学习

本文介绍基于 DataWorks、PAI-DLC、PAI-EAS 搭建在线学习链路的推荐方式：天级增量训练作为基线，每天基于天级模型启动一个流式训练任务；新的流式任务追上旧任务后，旧任务自动退出，再由 DataWorks 把 EAS 服务切换到新任务的产出。

## 整体流程

以业务日期 T 为例：

1. 天级增量训练：从 T-1 的天级模型继续训练 MaxCompute 分区 `ds=T`，得到天级模型 T。
1. 全量导出：把天级模型 T 的 embedding 全量上传到 FeatureDB 版本 `ol_T`。
1. 流式训练 stream_T：从天级模型 T 出发，从 T+1 日 00:00 的 Kafka 位点开始消费，追赶到最新数据后持续训练，同时增量上传 embedding、发布 dense 模型。
1. 交接：stream_T 追上 stream\_{T-1} 后，stream\_{T-1} 在下一次检查时自动结束训练并正常退出。
1. 切换：DataWorks 在 stream\_{T-1} 成功退出后，校验 stream_T 的发布产物，再更新 EAS 服务到 stream_T。

同一时刻最多有两条"链路"（lineage）在运行：正在服务的 stream\_{T-1}，以及正在追赶的 stream_T。每天的链路切换同时起到了[全量再基线](../usage/export.md)的作用：只要天级任务按时成功，崩溃重启带来的 FeatureDB 漂移最多存在于一条链路（约一天）。

## 链路隔离

每条链路的可变资源都按业务日期区分，两条链路互不干扰：

| 资源                                     | 链路 T 的取值                          | 说明                                                                                                  |
| ---------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 流式任务 `model_dir`                     | `/mnt/data/online/stream/${T}/model`   | checkpoint 与 Kafka 位点                                                                              |
| 发布根目录 `ONLINE_DENSE_EXPORT_DIR`     | `/mnt/data/online/stream/${T}/serving` | `<发布根目录>/dense_hot_export/current.json` 既是 EAS 读取的入口，也是交接判断使用的进度信号          |
| Kafka `group.id`                         | `tzrec_ol_${T}`                        | KafkaDataset 以消费组订阅 topic，两条链路共用一个 `group.id` 会互相 rebalance、瓜分分区，**必须不同** |
| FeatureDB `feature_store_config.version` | `ol_${T}`                              | 全量导入与增量写入不能叠写在同一个版本上                                                              |

示例中 `/mnt/data` 是 DLC 与 EAS 都挂载的 OSS 路径。在线 dense 发布依赖 `os.rename`，并要求 EAS 读到新的 `current.json` 时，它指向的版本目录已经可读。请确认所用的 OSS 挂载方式支持 rename，且写入对其他客户端及时可见；不满足时发布根目录请改用 NAS/CPFS，`model_dir` 仍可以使用 OSS。

## DataWorks 调度

在每个节点上配置调度参数（日调度时 `$[yyyymmdd]` 是业务日期的后一天）：

```
bizdate=$[yyyymmdd-1] prev=$[yyyymmdd-2] next=$[yyyymmdd]
```

| 节点         | 节点类型 | 依赖                                                              | 作用                                           |
| ------------ | -------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| sample       | ODPS SQL | 上游样本表                                                        | 产出 `ds=${bizdate}` 样本分区                  |
| train_daily  | PAI-DLC  | sample；**依赖上一周期（本节点）**                                | 天级增量训练                                   |
| full_export  | PAI-DLC  | train_daily                                                       | FeatureDB 全量上传                             |
| train_stream | PAI-DLC  | full_export；**不依赖上一周期**，与前一天的 train_stream 并行运行 | 流式训练，追上下一条链路后自动退出             |
| deploy       | Shell    | full_export；**依赖上一周期（本节点 + train_stream 节点）**       | 前一天的 train_stream 退出后校验产物并切换 EAS |

- train_stream 节点在业务日期 T 的实例即流式任务 stream_T，会运行约 1.5 天（当天全天 + 次日追赶时间），节点超时时间需大于该时长；DLC 任务开启失败自动重启。
- deploy 依赖上一周期的本节点，保证切换严格按日期顺序执行，较旧的链路不会覆盖较新的部署。
- 首次上线时没有前一天的 train_stream，EAS 服务需手动创建。

### sample

天级样本建议与 Kafka 样本流出自同一个 Flink 作业，并按 Kafka 消息时间戳切分分区，这样 `ds=T` 的结束时刻恰好等于 stream_T 的起始位点，不重不漏。切分与 `start.timestamp.ms` 的计算都需显式指定时区。

### train_daily

```bash
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.train_eval \
    --pipeline_config_path /mnt/data/online/daily.config \
    --model_dir /mnt/data/online/daily/${bizdate} \
    --fine_tune_checkpoint /mnt/data/online/daily/${prev} \
    --train_input_path odps://${project}/tables/${sample_table}/ds=${bizdate}
```

### full_export

将天级模型的 embedding 全量上传到本链路的 FeatureDB 版本，详见[dynemb 流式导出与 FeatureDB 全量上传](../usage/export.md)。天级配置中 `export_config.dynamic_embedding.feature_store_config` 不写 `version`，导出时通过 `--edit_config_json` 按业务日期指定：

```bash
USE_DISTRIBUTED_EMBEDDING=1 \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.export \
    --pipeline_config_path /mnt/data/online/daily/${bizdate}/pipeline.config \
    --export_dir /mnt/data/online/daily/${bizdate}/export \
    --edit_config_json '{"export_config.dynamic_embedding.feature_store_config.version":"ol_'${bizdate}'"}'
```

### train_stream

流式训练配置（只列出与在线学习相关的部分）：

```
train_config {
    num_epochs: 1
    save_checkpoints_timestamp_interval: 3600
    keep_checkpoint_max: 3
    delta_embedding_dump_config {
        paired_dump_interval_minutes: 10
        feature_store_config {
            region: "cn-beijing"
            project_name: "recommendation"
            feature_view_name: "model_embeddings"
            # version 不写，由启动命令的 --edit_config_json 按链路指定
        }
    }
    ...
}
data_config {
    dataset_type: KafkaDataset
    ...
}
```

- 使用 `num_epochs: 1` 而不是 `num_steps`：Kafka 数据流不会结束，任务只会因交接而正常退出。
- 学习率建议使用常数学习率；从天级模型 fine-tune 时不会恢复学习率调度进度。

启动命令：

```bash
START_MS=$(( $(date -d "${next:0:4}-${next:4:2}-${next:6:2} 00:00:00 +0800" +%s) * 1000 ))

USE_DISTRIBUTED_EMBEDDING=1 \
ONLINE_DENSE_EXPORT=1 \
ONLINE_DENSE_EXPORT_DIR=/mnt/data/online/stream/${bizdate}/serving \
ONLINE_LEARNING_SUCCESSOR_DIR=/mnt/data/online/stream/${next}/serving \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.train_eval \
    --pipeline_config_path /mnt/data/online/stream.config \
    --continue_train \
    --fine_tune_checkpoint /mnt/data/online/daily/${bizdate} \
    --model_dir /mnt/data/online/stream/${bizdate}/model \
    --train_input_path "kafka://${broker}/${topic}?group.id=tzrec_ol_${bizdate}&start.timestamp.ms=${START_MS}" \
    --edit_config_json '{"train_config.delta_embedding_dump_config.feature_store_config.version":"ol_'${bizdate}'"}'
```

首次启动与失败重启使用同一条命令：

- 首次启动时 `model_dir` 中没有 checkpoint，从天级模型加载参数；天级 checkpoint 中没有 Kafka 位点，因此按 `start.timestamp.ms` 开始消费。
- 重启时 `model_dir` 中已有 checkpoint，优先从中恢复参数和 Kafka 位点，`--fine_tune_checkpoint` 不再生效。

### deploy

stream\_{T-1} 退出不一定意味着 stream_T 已经追上（例如被人为配置了 `num_steps` 而提前结束），因此 deploy 节点在切换前按同样的规则重新校验发布产物，校验失败时节点失败，由 DataWorks 重跑：

```bash
set -e
OSS_ROOT=oss://${bucket}/online/stream
NEW=$(ossutil cat ${OSS_ROOT}/${bizdate}/serving/dense_hot_export/current.json)
OLD=$(ossutil cat ${OSS_ROOT}/${prev}/serving/dense_hot_export/current.json)
VERSION=$(python3 - "$NEW" "$OLD" <<'EOF'
import json, math, sys
new, old = json.loads(sys.argv[1]), json.loads(sys.argv[2])
def ts(m):
    t = m.get("data_timestamp")
    ok = isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0
    return t if ok else None
assert ts(new) is not None and ts(old) is not None and ts(new) >= ts(old), (new, old)
print(new["version"])
EOF
)
ossutil stat ${OSS_ROOT}/${bizdate}/serving/dense_hot_export/versions/${VERSION}/READY
```

校验通过后更新 EAS 服务，让服务改为读取 `/mnt/data/online/stream/${bizdate}/serving/dense_hot_export` 与 FeatureDB 版本 `ol_${bizdate}`，并等待滚动更新完成。**这两个配置项对应的 Processor 参数名请与 Processor 文档/负责人确认**，再写入 `eascmd modify ${service} -s service.json` 使用的配置文件。切换成功后可清理 FeatureDB 版本 `ol_${prev}` 之前的版本与对应的 OSS 目录。

## 交接规则

stream\_{T-1} 每隔 `log_step_count_steps` 步检查一次：rank 0 读取自己和下一条链路的 `dense_hot_export/current.json`，当下一条链路的 `data_timestamp` 不小于自己的 `data_timestamp` 时，所有 rank 在同一步结束训练。退出时：

- 不再保存最终 checkpoint，也不触发 epoch 结束时的保存：退役的链路不会再被恢复，跳过保存可以缩短退出时间。
- 尾部增量 embedding 仍会 dump 并上传，最后一个 dense 版本仍会正常发布。

任一侧的 `current.json` 不存在、无法解析，或 `data_timestamp` 不是有效的正数时，视为未追上，继续训练。设置 `ONLINE_LEARNING_SUCCESSOR_DIR` 时必须同时开启 `ONLINE_DENSE_EXPORT=1`，且两个发布根目录不能相同，否则训练启动即报错。

这是一个近似的"发布新鲜度"规则，存在以下已知限制：

- `data_timestamp` 是 rank 0 已消费数据的事件时间，不代表所有 Kafka 分区都已追上。
- stream\_{T-1} 在退出过程中可能再发布一个版本，切换后服务的新鲜度可能短暂回退。
- `current.json` 只说明某个版本已经发布，不说明下一条链路的训练进程当前健康；EAS 是否成功加载该版本仍取决于产物加载与 Processor 侧的校验。
- 由于 KafkaDataset 默认 `enable.auto.commit=false`，消费组已提交位点计算出的 lag 不能用于判断追赶进度。

## 时间线示例

假设 Kafka 积压 4.5 小时、流式训练速度是实时数据速度的 3 倍：

| 时间（T+1 日） | 事件                                                                             |
| -------------- | -------------------------------------------------------------------------------- |
| 01:30          | `ds=T` 样本就绪                                                                  |
| 03:30          | 天级训练完成                                                                     |
| 04:30          | 全量导出完成，stream_T 从 00:00 的位点开始追赶                                   |
| ~06:45         | stream_T 的发布时间戳追上 stream\_{T-1}                                          |
| 之后           | stream\_{T-1} 在下一次检查时发现并开始退出，退出完成后 deploy 校验并滚动更新 EAS |

追赶耗时约为 `积压时长 × 生产速度 / (训练速度 − 生产速度)`。从 stream\_{T-1} 停止发布到 EAS 滚动更新完成的这段时间内，服务停留在 stream\_{T-1} 的最后一个版本上，时长取决于退出过程（上传排空、最终发布）与滚动更新耗时，通常为分钟级，但没有严格上界。

## 运维建议

- Kafka 保留时长 ≥ 天级样本延迟 + 天级训练与导出耗时 + 追赶耗时 + 一天的重跑余量。
- Kafka 分区数需是 `nproc-per-node * nnodes * num_workers` 的倍数。
- 下一条链路迟迟追不上时（天级任务失败、追赶太慢），stream\_{T-1} 会继续训练并服务，但受 train_stream 节点超时、DLC 最长运行时长与 FeatureDB 视图 TTL 的限制，需配置告警及时处理。

| 故障                      | 影响                                            | 处理                                       |
| ------------------------- | ----------------------------------------------- | ------------------------------------------ |
| 天级训练或导出失败        | stream_T 不启动，stream\_{T-1} 继续服务         | 修复后重跑节点，下游自动继续               |
| train_stream 任务异常退出 | DLC 自动重启，从 `model_dir` 的 checkpoint 恢复 | 重启失败时重跑 train_stream 节点，命令不变 |
| deploy 校验失败           | EAS 保持原链路                                  | 检查下一条链路的发布情况后重跑 deploy 节点 |
| EAS 滚动更新失败          | EAS 保持原链路                                  | 排查 Processor 日志后重跑 deploy 节点      |
