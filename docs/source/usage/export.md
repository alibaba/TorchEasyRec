# 导出

## export_config

模型导出任务依赖pipeline.config中的export_config 配置

```
export_config {
}
```

- exporter_type: 导出类型, latest | best ，默认latest
  - latest 导出最新的模型
  - best 导出最好的模型
- best_exporter_metric: 当exporter_type为best的时候，确定最优导出模型的metric，注意该metric要在对应任务的metrics设置了才行。对于多任务模型则需要设置 {metric_name}\_{tower_name}。
- metric_larger_is_better: 确定最优导出模型的metric是越大越好，还是越小越好，默认是越大越好

## 导出命令

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/multi_tower_din_taobao_local/pipeline.config \
    --export_dir experiments/multi_tower_din_taobao_local/export
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认导出model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录，对于向量召回模型，导出命令会自动进行切图，分别放在user和item的子目录下

(export-env-vars)=

### 环境变量

- ODPS_ENDPOINT: 在PAI-DLC/PAI-DSW环境，数据为MaxCompute表的情况下需设置，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_CONFIG_FILE_PATH: 在本地环境，数据为MaxCompute表的情况下需设置为odps_conf的路径，详见[文档](../feature/data.md)的OdpsDataset章节
- QUANT_EMB: 对EmebddingBagCollection（非序列特征）参数进行量化，默认开启，INT8量化在大部分场景中对AUC等指标基本无损，且能大幅提升推理性能
  - **QUANT_EMB=INT8**：启用量化，默认已启用，并且默认为INT8量化，可以支持FP32，FP16，INT8，INT4，INT2
  - **QUANT_EMB=0**：关闭量化
- QUANT_EC_EMB: 对EmebddingCollection（序列特征）参数进行量化，默认关闭，INT8量化在大部分场景中对AUC等指标基本无损，且能大幅提升推理性能
  - **QUANT_EC_EMB=INT8**：启用量化，默认关闭，可以支持FP32，FP16，INT8，INT4，INT2
- INPUT_TILE: 对User侧特征自动扩展，开启可减少请求大小、网络传输时间和计算时间，默认关闭。必须在TorchEasyRec Processor的fg_mode=normal下使用
  - **INPUT_TILE=2**：user侧特征fg仅计算一次
  - **INPUT_TILE=3**：user侧fg和embedding计算仅一次，适用于user侧特征比较多的情况
- INPUT_TILE_3_ONLINE: 配合INPUT_TILE=3使用，对User侧序列特征使用在线推理模式，序列特征在线模型服务中推理性能更好，但导出的模型无法用于离线预测
  - **INPUT_TILE_3_ONLINE=1**：启用序列特征的在线推理模式
- USE_DISTRIBUTED_EMBEDDING: 开启分布式 embedding 导出模式，导出 dense graph 与分片 sparse embedding 参数；该模式会自动使用 `INPUT_TILE=3`
  - **USE_DISTRIBUTED_EMBEDDING=1**：启用分布式 embedding 导出
- DIST_QUANT: 分布式 embedding 导出模式下的 sparse embedding 参数量化开关，默认关闭；当前仅支持 INT8，与普通导出的 `QUANT_EMB` / `QUANT_EC_EMB` 不同
  - **USE_DISTRIBUTED_EMBEDDING=1 DIST_QUANT=INT8**：启用分布式 sparse embedding INT8 量化
  - 未设置 **DIST_QUANT** 时，不启用分布式 sparse embedding 量化
- ENABLE_AOT: 使用AOT(Ahead Of Time)编译优化导出的模型，可显著提升推理性能。**AOT 编译产物与导出机器的 GPU 架构强绑定，在线服务的 GPU 类型必须与导出时使用的 GPU 类型完全一致**，详见下文 [在 PAI 上导出 AOT 模型](#export-aot-on-pai) 章节
  - **ENABLE_AOT=1**: 使用AOT编译优化导出模型（sparse 部分用 JIT，dense 部分用 AOTI）
  - **ENABLE_AOT=2**: 使用统一 AOTI 模型编译优化 (sparse + dense 融合为单一 .pt2) [experimental]

(online-dense-export)=

## 在线学习 dense 模型热导出

在 `USE_DISTRIBUTED_EMBEDDING=1` 的在线学习场景下，sparse embedding 由分布式 embedding 服务独立更新，训练进程可以在训练过程中按分钟级（或自定义节奏）持续导出 dense 图，供推理 Processor 热切换。该能力**不依赖 checkpoint**：rank 0 在训练启动时一次性构建 serving 侧 dense 图，之后每次触发时全体 rank 从内存中的 DMP 模型收集 dense 权重（仅收集 dense 图实际携带的参数，不涉及 sparse/dynamicemb/MCH 状态），rank 0 在后台线程热替换权重、script 并原子发布新版本，训练主流程不阻塞在导出上。

### 启用方式

训练命令前加上以下环境变量即可（`ONLINE_DENSE_EXPORT_DIR` 与 `ONLINE_DENSE_EXPORT_STEPS` / `ONLINE_DENSE_EXPORT_INTERVAL` 至少一项为必填）：

```bash
ONLINE_DENSE_EXPORT=1 \
ONLINE_DENSE_EXPORT_DIR=/mnt/data/serving \
ONLINE_DENSE_EXPORT_INTERVAL=60 \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.train_eval \
    --pipeline_config_path experiments/online_learning/pipeline.config
```

### 环境变量

- ONLINE_DENSE_EXPORT: 开启训练内在线 dense 导出，默认关闭
  - **ONLINE_DENSE_EXPORT=1**：启用（要求同时设置 `USE_DISTRIBUTED_EMBEDDING=1`）
- ONLINE_DENSE_EXPORT_DIR: 在线服务读取的根目录（必填）。导出产物发布在 `<ONLINE_DENSE_EXPORT_DIR>/dense_hot_export` 下，必须为本地路径（发布依赖 `os.rename` 原子换版）；`model_dir` 可以是远端（如 OSS）
- ONLINE_DENSE_EXPORT_STEPS: 按训练步数间隔触发导出（如 `300` 表示每 300 步导出一次），0 表示关闭
- ONLINE_DENSE_EXPORT_INTERVAL: 按数据事件时间（`Batch.data_timestamp`，Unix 秒）间隔触发导出（如 `60` 表示每消费过 60 秒事件时间、且跨过整分边界时导出一次），0 表示关闭。事件时间触发会在各 rank 间做 quorum 对齐，与 checkpoint 的时间触发机制一致
- ONLINE_DENSE_EXPORT_QUORUM: 事件时间触发的 worker 越界比例阈值，取值 (0, 1\]，默认 0.5
- ONLINE_DENSE_EXPORT_KEEP_VERSIONS: 保留的历史版本数，默认 3（serving 需要当前版本 + 上一版本用于原子切换）。`current.json` 指向的版本永远不被清理
- ONLINE_DENSE_EXPORT_TIMEOUT: 单次导出的预算秒数，默认 3600。超时不会中断导出线程，仅打印告警，并用于训练结束时 drain 的等待上限

`ONLINE_DENSE_EXPORT_STEPS` 与 `ONLINE_DENSE_EXPORT_INTERVAL` 至少要设置一个；训练结束时还会强制导出一次最终状态。导出频率与 checkpoint 频率完全独立，checkpoint 仍按原有配置保存、用于训练恢复。

### 导出产物与切换契约

```
<ONLINE_DENSE_EXPORT_DIR>/dense_hot_export/
├── current.json                  # 服务指向的最新版本指针（原子写）
└── versions/
    └── <yyyyMMddHHmmss>/         # 一个不可变的版本目录
        ├── scripted_model.pt     # TorchScript dense 模型
        ├── dense_meta.json       # placeholder 名 -> serving embedding 名映射
        ├── graph/                # 图 dump（排查用）
        └── READY                 # 目录完整的标记文件，先写 READY 再原子换版
```

`current.json` 内容：

```json
{
  "checkpoint_step": 1200,
  "created_at": "2026-07-24T05:20:00.000000+00:00",
  "data_timestamp": 1782365432.0,
  "version": "20260724052000"
}
```

- `version`: 版本目录名，单调递增的时间戳。
- `checkpoint_step` / `data_timestamp`: 导出时训练步数与该 rank 消费到的事件时间，供 serving 侧与 sparse 状态对齐版本一致性。
- 推理 Processor 应只读取 `current.json` 指向的版本，且只消费最新版本。

### 运行与排障说明

- 仅 rank 0 执行建图与发布；启动时会对整条建图 + script 链路做一次试运行，任何 trace/script 失败会在训练开始前就暴露（fail-fast）。
- 启动时还会校验 dense 图的每个参数都能在训练模型 state dict 中找到来源（INPUT_TILE=3 的 user 侧孪生模块自动回落到非 user 侧权重），校验失败训练不会启动。
- 导出在后台线程执行，失败（日志中 `online dense export task failed`）只跳过该版本、不影响训练，也不改变 `current.json` 指向。
- 两次导出间隔小于单次导出耗时时，排队中的旧任务被最新版本顶替（latest-wins），不产生积压。
- MatchModel（向量召回）与 TDM 模型的完整导出按 user/item（或按模块）分目录，与单体 dense 热导出的布局不兼容，启用会直接报错，请使用完整的 `tzrec.export`。
- 手动/离线从某个 checkpoint 导出一个版本（与训练内导出发布到同一目录结构）：

```bash
USE_DISTRIBUTED_EMBEDDING=1 ONLINE_DENSE_EXPORT_DIR=/mnt/data/serving \
python -m tzrec.tools.online_dense_export \
    --pipeline_config_path experiments/online_learning/pipeline.config \
    --checkpoint_path /mnt/data/model/model.ckpt-1200 \
    --model_dir /mnt/data/model \
    --checkpoint_step 1200
```

(export-aot-on-pai)=

## 在 PAI 上导出 AOT 模型

### GPU 架构匹配要求

开启 `ENABLE_AOT` 后，导出产物（`.pt2` 中包含针对导出机器 GPU 算力/SM 版本编译的 cubin 与 Triton kernel）与 GPU 架构强绑定。因此：

- **PAI-EAS 在线服务使用的 GPU 类型，必须与导出时使用的 GPU 类型完全一致**，否则在线加载或推理会失败、或产生错误结果。
- 更换 EAS 服务的 GPU 类型后，必须在对应 GPU 上重新导出模型。

根据 PAI-DLC 是否能提供与目标 EAS 服务相同的 GPU 类型，有以下两种导出方式。

### 场景一：PAI-DLC 有与 EAS 服务 GPU 类型相同的资源

直接在 PAI-DLC 上导出（推荐）。在 [DLC Tutorial 的导出章节](../quick_start/dlc_tutorial.md) 的命令前加上 `ENABLE_AOT=1` 即可，例如：

```bash
ENABLE_AOT=1 CUDA_HOME=/usr/local/cuda-12 \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.export \
    --pipeline_config_path /mnt/data/${MODEL_DIR}/pipeline.config \
    --export_dir /mnt/data/${MODEL_DIR}/export
```

- `CUDA_HOME`: AOT 编译需要 `nvcc`，指向 CUDA 安装目录。
- 其余量化、INPUT_TILE 等环境变量见上文 [环境变量](#export-env-vars) 章节，按需添加。

### 场景二：PAI-DLC 没有与 EAS 服务 GPU 类型匹配的资源

此时在 PAI-EAS 上以 `workload_type=elasticjob` 任务方式导出。由于该任务与在线服务运行在**相同的 EAS 资源组（同一 GPU 硬件）**，可保证导出产物与在线服务的 GPU 架构一致；导出产物写入挂载的数据集（NAS/OSS），随后即可用于部署在线服务（参见 [模型服务](serving.md)）。

任务服务的 JSON 配置示例如下（各字段含义参见 [PAI-EAS 服务参数文档](https://help.aliyun.com/zh/pai/parameters-of-model-services)，请将 `${...}` 占位符替换为自己的资源标识）：

```json
{
    "cloud": {
        "networking": {
            "security_group_id": "${SG_ID}",
            "vpc_id": "${VPC_ID}",
            "vswitch_id": "${VSWITCH_ID}"
        }
    },
    "containers": [
        {
            "image": "dsw-registry-vpc.{region}.cr.aliyuncs.com/pai/torcheasyrec:1.3.0-pytorch2.12.1-gpu-py311-cu129-ubuntu22.04",
            "port": 8000,
            "script": "set -e\npip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-accelerate.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-accelerate.aliyuncs.com\n\nCUDA_HOME=/usr/local/cuda-12 ODPS_ENDPOINT=http://service.{region}-vpc.maxcompute.aliyun-inc.com/api \\\nENABLE_AOT=1 QUANT_EC_EMB=INT8 \\\ntorchrun --master_addr=127.0.0.1 --master_port=12345 \\\n    --nnodes=1 --nproc-per-node=1 --node_rank=0 \\\n    -m tzrec.export \\\n    --pipeline_config_path /mnt/data/${MODEL_DIR}/pipeline.config \\\n    --export_dir /mnt/data/${MODEL_DIR}/export/ \\\n    --additional_export_config '{\"cand_seq_pk\": \"cand_seq\"}'"
        }
    ],
    "metadata": {
        "cpu": 60,
        "disk": "60Gi",
        "gpu": 1,
        "instance": 1,
        "memory": 383000,
        "name": "tzrec_aot_export",
        "resource": "${EAS_RESOURCE_ID}",
        "resource_burstable": false,
        "rpc": {
            "keepalive": 5000
        },
        "workload_type": "elasticjob",
        "workspace_id": "${WORKSPACE_ID}"
    },
    "options": {
        "enable_ram_role": true
    },
    "storage": [
        {
            "dataset": {
                "id": "${DATASET_ID}",
                "read_only": false,
                "version": "v1"
            },
            "mount_path": "/mnt/data/"
        }
    ]
}
```

关键字段说明：

- `cloud.networking`: 任务运行所在的 VPC/交换机/安全组。由于脚本需从公网 `pip install` 安装 tzrec，而 EAS 默认不开放公网访问，需为该 VPC 配置 NAT 网关并绑定 EIP 以开启公网访问。详见 [配置网络连通](https://help.aliyun.com/zh/pai/configure-network-connectivity)。
- `metadata.workload_type`: 设为 `elasticjob`，即以一次性任务（而非常驻服务）方式运行，导出完成后任务自动退出。
- `metadata.resource`: 填写**目标在线服务所用的 EAS 专属资源组 ID**，使导出运行在与服务相同的 GPU 上，从而保证 GPU 架构匹配。
- `metadata.gpu` / `instance`: 单卡单实例即可完成导出。
- `containers.image`: 运行任务的容器镜像，使用 PAI 官方 TorchEasyRec 镜像（已内置 CUDA、PyTorch 等依赖），`{region}` 替换为所在地域，镜像 tag 按需选择。
- `containers.script`: 安装 tzrec 后执行 `tzrec.export`。其中 `ENABLE_AOT=1` 开启 AOT 编译，`QUANT_EC_EMB`、`CUDA_HOME`、`ODPS_ENDPOINT` 等见上文 [环境变量](#export-env-vars) 章节按需设置；HSTU 类模型通过 `--additional_export_config '{"cand_seq_pk": "cand_seq"}'` 指定候选序列特征名。
- `storage`: 挂载数据集到 `/mnt/data/`，`--export_dir` 写入该挂载路径下，供后续在线服务加载。
- `options.enable_ram_role`: **必须设为 `true`**，任务通过 RAM 角色访问 MaxCompute、OSS/NAS、镜像仓库等云资源，无需配置长期 AccessKey。需提前在 RAM 中创建受信主体为 PAI（`eas.pai.aliyuncs.com`）的角色并授予对应资源访问权限。详见 [配置 EAS 的 RAM 角色](https://help.aliyun.com/zh/pai/configure-the-eas-ram-role)。

将上述 JSON 保存为 `aot_export.json`，可在 PAI-EAS 控制台创建任务服务，或通过 `eascmd` 提交：

```bash
eascmd -i ${ACCESS_KEY_ID} -k ${ACCESS_KEY_SECRET} -e ${ENDPOINT} create aot_export.json
```

任务运行结束后，`--export_dir` 指向的目录即为导出好的模型，将其作为在线服务的模型路径部署即可（部署方式参见 [模型服务](serving.md)）。
