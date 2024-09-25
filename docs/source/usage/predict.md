# 离线预测

## 预测命令

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
- --is_profiling: 是否进行离线预测性能的Profiling

### 环境变量

- ODPS_ENDPOINT: 在PAI-DLC/PAI-DSW环境，数据为MaxCompute表的情况下需设置，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_CONFIG_FILE_PATH: 在本地环境，数据为MaxCompute表的情况下需设置为odps_conf的路径，详见[文档](../feature/data.md)的OdpsDataset章节
