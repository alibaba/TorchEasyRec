# 评估

## 评估命令

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.eval \
    --pipeline_config_path experiments/multi_tower_din_taobao_local/pipeline.config \
    --eval_input_path data/taobao_data_eval/\*.parquet
```

- --pipeline_config_path: 评估用的配置文件
- --checkpoint_path: 指定要评估的checkpoint, 默认评估model_dir下面最新的checkpoint
- --eval_type: 指定要评估的checkpoint的类型(best、latest), 默认评估model_dir下面最新的checkpoint
- --eval_input_path: 评估数据的输入路径
- --eval_result_filename: 评估指标的结果文件名

### 环境变量

- ODPS_ENDPOINT: 在PAI-DLC/PAI-DSW环境，数据为MaxCompute表的情况下需设置，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_CONFIG_FILE_PATH: 在本地环境，数据为MaxCompute表的情况下需设置为odps_conf的路径，详见[文档](../feature/data.md)的OdpsDataset章节

## 评估配置

评估配置是指配置文件中的eval_config，详细参考[配置参考手册](../reference.md)

```
eval_config {
}
```

- num_steps: 评估的步数，默认为评估eval_input_path中指定的所有数据
- log_step_count_steps: 评估打印log和summary的步数间隔（如果打印时间间隔小于1s，会跳过打印）
