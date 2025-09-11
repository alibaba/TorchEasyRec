# 导出

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

### 环境变量

- ODPS_ENDPOINT: 在PAI-DLC/PAI-DSW环境，数据为MaxCompute表的情况下需设置，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_CONFIG_FILE_PATH: 在本地环境，数据为MaxCompute表的情况下需设置为odps_conf的路径，详见[文档](../feature/data.md)的OdpsDataset章节
- QUANT_EMB: 对EmebddingBagCollection（非序列特征）参数进行量化，默认开启，INT8量化在大部分场景中对AUC等指标基本无损，且能大幅提升推理性能
  - **QUANT_EMB=INT8**：启用量化，默认已启用，并且默认为INT8量化，可以支持FP32，FP16，INT8，INT4，INT2
  - **QUANT_EMB=0**：关闭量化
- QUANT_EC_EMB: 对EmebddingCollection（序列特征）参数进行量化，默认开启，INT8量化在大部分场景中对AUC等指标基本无损，且能大幅提升推理性能
  - **QUANT_EC_EMB=INT8**：启用量化，默认关闭，可以支持FP32，FP16，INT8，INT4，INT2
- INPUT_TILE: 对User侧特征自动扩展，开启可减少请求大小、网络传输时间和计算时间。必须在fg_mode=normal下使用，并且TorchEasyRec导出时需加上此环境变量
  - **INPUT_TILE=2**：user侧特征fg仅计算一次
  - **INPUT_TILE=3**：user侧fg和embedding计算仅一次，适用于user侧特征比较多的情况
- ENABLE_AOT:
  - **ENABLE_AOT=1**: 使用AOT(Ahead Of Time)编译优化导出优化的模型(experimental)
