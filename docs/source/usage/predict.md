# 离线预测

## 预测命令

### 预测导出模型

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/multi_tower_din_taobao_local/export \
    --predict_input_path data/taobao_data_eval/\*.parquet \
    --predict_output_path experiments/multi_tower_din_taobao_local/predict_result \
    --reserved_columns user_id,adgroup_id,clk
```

- --scripted_model_path: 要预测的导出模型
- --predict_input_path: 预测数据的输入路径
- --predict_output_path: 预测结果的输出路径，如果输出为MaxCompute表，且表不存在，会自动新建MaxCompute表
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列
- --batch_size: 预测的batch_size，默认为训练时pipeline_config中指定的batch_size
- --predict_threads: 离线预测的线程数
- --dataset_type: 读数据的Dataset类型，默认用data_config.dataset_type
- --writer_type: 写数据的Writer类型，默认用data_config.dataset_type对应的Writer
- --predict_steps: 离线预测的步数，默认为完整的输入表
- --is_profiling: 是否进行离线预测性能的Profiling

注：

- 该预测方式预测的模型前向逻辑已经固化在导出模型中，预测结果不受模型代码的变更影响
- DynamicEmb 模型导出时会自动启用分布式 embedding 模式（`USE_DISTRIBUTED_EMBEDDING=1`），其导出产物的预测方式见下文 [预测分布式 embedding 导出模型](#predict-distributed-embedding)
- 离线预测现已将序列稠密特征的 `.lengths` 统一转为 `int64`（与稀疏特征及在线推理保持一致）；若历史导出的模型固化了 `int32` 长度类型（如含序列稠密特征的旧版 `ENABLE_TRT=1` 导出产物），请使用对应版本预测或重新导出

(predict-distributed-embedding)=

### 预测分布式 embedding 导出模型

`USE_DISTRIBUTED_EMBEDDING=1` 导出的模型同样用上文的 `--scripted_model_path` 命令预测，会根据导出目录下的 `model_acc.json` 自动识别分布式 embedding 模式（`"DISTRIBUTED_EMBEDDING": "1"`）与 AOTI 模式（`"ENABLE_AOT": "2"` 或 `"1"`；若 `model_acc.json` 缺少 `ENABLE_AOT` 字段，也会兜底检测是否存在 `aoti/aoti_model.pt2` 且无 `scripted_model.pt`）：

- 对于 TorchScript 导出产物（`scripted_model.pt`）与 AOTI 导出产物（`aoti/aoti_model.pt2`），使用**完全相同的 `tzrec.predict` 命令**即可，无需额外命令行参数
- 检测到 AOTI 产物时，进程会在每张可用 GPU 上分别加载独立 `device_index` 的 `aoti/aoti_model.pt2` 副本，由 NVE 在 GPU 上完成 sparse 查表并将 GPU 驻留张量直接送入 AOTI dense 模型执行前向推理（注意：预测机器的 GPU SM 架构必须与导出机器一致）
- 建议每台机器只启动一个预测进程（`--nproc-per-node=1`），该进程使用本机全部 GPU，共享同一份 host 内存中的 embedding；每张 GPU 一个进程也能运行，但 host 内存占用随进程数成倍增加
- `--predict_threads` 默认每张 GPU 一个前向线程；每多一个线程，GPU 上就多一份不计入 GPU 缓存预算的 embedding 查询缓冲区（大模型可达数 GB）。对于 AOTI 模型，同一张 GPU 上的多线程会通过各自独立的 CUDA Stream 并发查表并按每卡设备锁串行进入该卡的 AOTI dense Runner；设置更多线程时若显存不足，请调低 `DIST_EMB_GPU_CACHE_RATIO` 为 AOTI 工作区和线程查询缓冲区预留显存，或适当减小 `--batch_size` / `--predict_threads`

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/dynamicemb/export \
    --predict_input_path data/taobao_data_eval/\*.parquet \
    --predict_output_path experiments/dynamicemb/predict_result \
    --reserved_columns user_id,adgroup_id,clk
```

### 预测Checkpoint

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.predict \
    --pipeline_config_path experiments/multi_tower_din_taobao_local/pipeline.config \
    --checkpoint_path experiments/multi_tower_din_taobao_local/model.ckpt-10
    --predict_input_path data/taobao_data_eval/\*.parquet \
    --predict_output_path experiments/multi_tower_din_taobao_local/predict_result \
    --reserved_columns user_id,adgroup_id,clk
```

- --pipeline_config_path: 模型配置文件
- --checkpoint_path: 模型预测的checkpoint路径，如不填，默认为model_dir下的最新的模型
- --predict_input_path: 预测数据的输入路径
- --predict_output_path: 预测结果的输出路径，如果输出为MaxCompute表，且表不存在，会自动新建MaxCompute表
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列
- --batch_size: 预测的batch_size，默认为训练时pipeline_config中指定的batch_size
- --dataset_type: 读取数据的Dataset类型，默认用data_config.dataset_type
- --writer_type: 写数据的Writer类型，默认用data_config.dataset_type对应的Writer
- --predict_steps: 离线预测的步数，默认为完整的输入表
- --is_profiling: 是否进行离线预测性能的Profiling

注：该预测方式预测的模型前向逻辑受模型代码的变更影响，最好保证训练和预测用同一个版本的代码

### 环境变量

- ODPS_ENDPOINT: 在PAI-DLC/PAI-DSW环境，数据为MaxCompute表的情况下需设置，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_CONFIG_FILE_PATH: 在本地环境，数据为MaxCompute表的情况下需设置为odps_conf的路径，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_TABLE_LIFECYCLE: 自动创建的ODPS输出表的lifecycle
- DIST_EMB_GPU_CACHE_RATIO: 预测分布式 embedding 导出模型时，每张 GPU 缓存的 embedding 比例（按存储字节数计算），默认 `auto`：按各 GPU 的空闲显存（预留 20% 显存给 dense 前向等）和 embedding 存储大小自动计算，日志会打印实际使用的比例（如 `DIST_EMB_GPU_CACHE_RATIO=auto resolved to 0.35`）；也可设置为 (0, 1\] 的数值手动指定。若显存不足（OOM），可设置比日志中更小的数值；未缓存的 id 从 host 内存读取
