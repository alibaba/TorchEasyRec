# 训练

## 训练命令

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path multi_tower_din_taobao_local.config \
    --train_input_path data/taobao_data_train/\*.parquet \
    --eval_input_path data/taobao_data_eval/\*.parquet \
    --model_dir experiments/multi_tower_din_taobao_local
```

- --pipeline_config_path: 训练用的配置文件
- --model_dir: 模型训练目录
- --train_input_path: 训练数据的输入路径，可以支持MaxCompute表/Parquet/CSV等路径，详见[文档](../feature/data.md)
- --eval_input_path: 评估数据的输入路径
- --continue_train: 是否增量训练
- --fine_tune_checkpoint: 增量训练的checkpoint路径，如experiments/multi_tower_din_taobao_local/model.ckpt-0，如果不设置，增量训练使用model_dir下最近的检查点
- --edit_config_json: 命令行以json的方式动态修改配置文件，如{"model_dir":"experiments/","feature_configs\[0\].raw_feature.boundaries":\[4,5,6,7\]}

### 环境变量

- ODPS_ENDPOINT: 在PAI-DLC/PAI-DSW环境，数据为MaxCompute表的情况下需设置，详见[文档](../feature/data.md)的OdpsDataset章节
- ODPS_CONFIG_FILE_PATH: 在本地环境，数据为MaxCompute表的情况下需设置为odps_conf的路径，详见[文档](../feature/data.md)的OdpsDataset章节

## 训练配置

训练配置是指配置文件中的train_config，详细参考[配置参考手册](../reference.md)

- sparse_optimizer: 稀疏参数（Embedding）的优化器和LR策略设置

```
sparse_optimizer {
    adagrad_optimizer {
        lr: 0.001
    }
    constant_learning_rate {
    }
}
```

- dense_optimizer: 稠密参数的优化器LR策略设置

```
dense_optimizer {
    adam_optimizer {
        lr: 0.001
    }
    exponential_decay_learning_rate {
        decay_size: 1
        decay_factor: 0.7
        by_epoch: true
    }
}
```

LR策略可以支持按epoch更新或者按step更新

- num_steps: 训练的步数，不能跟num_epochs同时设置
- num_epochs: 训练的epoch数
- save_checkpoints_steps: 保存模型的步数间隔，保存模型后会做一次评估
- fine_tune_checkpoint: 增量训练的checkpoint路径，也可以设置checkpoint目录，将会使用目录下最新的checkpoint
- fine_tune_ckpt_var_map: 需要restore的参数列表文件路径，文件的每一行是{variable_name in current model}\\t{variable name in old model ckpt}
  - 需要设置fine_tune_ckpt_var_map的情形:
    - 现在的模型和原有模型参数名不完全匹配，但想迁移，如embedding的名字不一样
      - 原有模型checkpoint的参数列表名可以通过如下命令获取
      ```bash
      python -m tzrec.tools.list_distcp_param --checkpoint_path experiments/multi_tower_din_taobao_local/model.ckpt-0
      ```
    - 现在的模型需要让模型加载一些预训练的Embedding参数
      - 预训练的参数可以通过如下方式转成tzrec的检查点
      ```python
      # convert.py
      import torch
      from torch import distributed as dist
      from torch.distributed.checkpoint import save
      dist.init_process_group(backend='gloo')
      state_local = torch.load('pretrain.pth')
      state_dist = {'embedding.weight': state_local['embedding.weight']}
      save(state_dist, checkpoint_id="pretrain/model.ckpt-0"))
      ```
      ```bash
      torchrun --master_addr=localhost --master_port=32555 --nnodes=1 --nproc-per-node=1 --node_rank=0 convert.py
      ```
- log_step_count_steps: 打印log和summary的步数间隔（如果打印时间间隔小于1s，会跳过打印）
- is_profiling: 是否做训练性能分析，设置为true，会在模型目录下记录trace文件
- use_tensorboard: 是否使用tensorboard，默认为true

## 训练性能优化

TorchEasyRec是以模型混合并行的方式进行训练的，会根据机间和卡间的通信拓扑环境的设置寻优最好的Embedding分片和并行计算的方式，在显存约束下最小化计算和通信开销

默认的机间的通信环境为[RDMA](https://help.aliyun.com/zh/ecs/user-guide/erdma-overview)，卡间的通信环境为NVLINK，如任务所运行的环境没有RDMA和NVLINK，需对如下环境变量做一些调整

对于使用的是不带NVLINK的机型，如`ecs.gn7i-c32g1.32xlarge(4 * NVIDIA A10)`，`ecs.gn6i-c24g1.24xlarge(4 * NVIDIA T4)`等，需调小卡间带宽的环境变量`INTRA_NODE_BANDWIDTH`

```bash
export INTRA_NODE_BANDWIDTH=$(awk 'BEGIN {printf("%f", 30 * 1024 * 1024 * 1024 / 1000)}')
```

对于使用的是不带RDMA的机型，如`ecs.gn7i-c32g1.32xlarge(4 * NVIDIA A10)`，`ecs.gn6i-c24g1.24xlarge(4 * NVIDIA T4)`，`ecs.gn6e-c12g1.24xlarge(8 * V100)`等，需调小机间带宽的环境变量`CROSS_NODE_BANDWIDTH`，具体可以参考各机型的[基础网络带宽文档](https://help.aliyun.com/zh/ecs/user-guide/gpu-accelerated-compute-optimized-and-vgpu-accelerated-instance-families-1)并减去读数据所需网络带宽设置

```bash
export CROSS_NODE_BANDWIDTH=$(awk 'BEGIN {printf("%f", 3 * 1024 * 1024 * 1024 / 1000)}')
```

如果遇到CUDA OOM（out of memory）的情况，可以调高`STORAGE_RESERVE_PERCENT`适当增加显存的预留比例，默认值为0.15。调高后TorchEasyRec会在新的显存限制下重新寻优最好的Embedding分片和并行计算的方式

```bash
export STORAGE_RESERVE_PERCENT=0.5
```
