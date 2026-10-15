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
- DynamicEmb 等模型需用分布式 embedding 模式导出（`USE_DISTRIBUTED_EMBEDDING=1`）后预测，见下文 [预测分布式 embedding 导出模型](#predict-distributed-embedding)

(predict-distributed-embedding)=

### 预测分布式 embedding 导出模型

`USE_DISTRIBUTED_EMBEDDING=1` 导出的模型（`model_acc.json` 中 `DISTRIBUTED_EMBEDDING` 为 `"1"`）同样用上文的 `--scripted_model_path` 命令预测，会自动识别。此时 `scripted_model.pt` 只含 dense 部分，sparse embedding 由预测进程从导出目录的 `sparse/` 加载，并使用与在线 Processor 相同的 NVIDIA NV-Embeddings（NVE）分层结构查表：每张表在 host 内存中保存一份全量（哈希表），每张 GPU 上各有一个 GPU cache。存储类型、维度和默认行都相同的表会合并为一张 host 表和一个 GPU cache，同组所有特征一次查表完成；NVE 每次查表都有固定开销，表或特征很多时合并可显著降低查表耗时。`DIST_QUANT=INT8` 导出的 INT8 表以量化格式加载，查表时反量化为 fp16，再转为 fp32 输入 dense 模型，与在线服务一致。

- 依赖 `pynve`（GPU 镜像已预装），仅支持 GPU 预测
- 建议每台机器只启动一个预测进程（`--nproc-per-node=1`），该进程使用本机全部 GPU，所有 GPU 共享同一份 host 内存中的 embedding 表；每张 GPU 启动一个进程也能运行，但每个进程各加载一份 embedding 表，host 内存占用随进程数成倍增加，日志中会有对应告警
- `--predict_threads` 默认不少于 GPU 数，预测线程按序号轮流分配到各 GPU
- host 内存占用约为 `行数 x 向上对齐到 32 字节的行字节数` 加上 key 的开销；INT8 表每行为 `embedding_dim + 4` 字节（如 64 维为 68 字节，对齐后 96 字节；fp32 为 256 字节）
- 查不到的 id（含 -1）返回全 0 向量；由零冲突 Hash（zch）转换而来的表返回导出的默认行，与在线服务一致
- 合并表时，除第一张表外，各表的 key 与该表固定的 63 位盐值异或后存入同一张 host 表：同一张表内的 key 不会冲突，不同表之间的 key 冲突概率与 FG 的 63 位 id 哈希冲突概率相当
- 带权重的稀疏特征（`use_weight: true`）与训练一致按权重加权求和，MEAN 池化为加权和除以 id 个数

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
- DIST_EMB_GPU_CACHE_RATIO: 预测分布式 embedding 导出模型时，每张 GPU 为每组合并后的 embedding 表缓存的比例（按表的存储字节数计算），取值 (0, 1\]，默认 1.0 即整表缓存到 GPU；GPU 显存不足时调小，未命中 GPU cache 的 id 会从 host 内存的全量表读取
