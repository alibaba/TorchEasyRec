# 使用TorchEasyRec读取MaxCompute表进行模型训练

此文档提供在阿里云的PAI-DLC（分布式训练容器）中使用TorchEasyRec读取MaxCompute（MC）表进行模型训练的步骤和配置示例。

## 环境准备

### 创建数据集
1. 登录[PAI控制台](https://pai.console.aliyun.com/)。
2. 点击 **AI资源管理 - 数据集 -> 创建数据集**。
3. 选择数据存储为 **阿里云文件存储(NAS)**，并挂载到 `/mnt/data` 下。

### TorchEasyRec安装
确保已安装TorchEasyRec的**Nightly版本**。可以通过以下命令确认可用版本：

```plain
pip index versions tzrec -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
```

### 配置文件
一个最简单的数据配置如下：

```plain
data_config {
    batch_size: 8192
    dataset_type: OdpsDataset
    fg_encoded: true
    label_fields: "clk"
    num_workers: 8
}
```

- `dataset_type`: 设置为 `OdpsDataset` 以读取MaxCompute表。
- `fg_encoded`: 如果输入数据已通过特征生成器编码，则设为 `true`。
- `label_fields`: 指定标签字段，通常是点击率（clk）。

### 前置条件
- 在[MaxCompute控制台](https://maxcompute.console.aliyun.com/)的「租户管理」->「租户属性」页面打开**开放存储(Storage API)开关**。
- 根据需要赋予用户权限，具体参考[租户权限](https://help.aliyun.com/zh/maxcompute/user-guide/perform-access-control-based-on-tenant-level-roles#section-mt7-tmu-f49)。

```plain
{
    "Version": "1",
    "Statement": [
        {
            "Action": "odps:Usage",
            "Effect": "Allow",
            "Resource": [
                "acs:odps:*:regions/*/quotas/pay-as-you-go"
            ]
        }
    ]
}
```

## 创建DLC任务

### 配置任务
1. 进入[PAI控制台](https://pai.console.aliyun.com/)，选择工作空间，点击 **模型开发与训练-分布式训练(DLC)**，创建新的训练任务。
2. **节点镜像**选择 `torcheasyrec:latest`（或具体版本）。
3. **资源配置**选择框架为PyTorch，任务资源可以选择单机8卡V100或更高性能的机器。
4. 在 **执行命令** 中填入以下命令：

### 训练命令
设置`ODPS_ENDPOINT`为对应的MaxCompute服务地址，执行训练命令：

```plain
cd /mnt/data
export ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.train_eval \
--pipeline_config_path /mnt/data/multi_tower_din_odps.config \
--train_input_path odps://{project_name}/tables/{your_table_name} \
--eval_input_path odps://{project_name}/tables/{your_eval_table_name} \
--model_dir /mnt/data/multi_tower_din_odps_dlc
```

- `--pipeline_config_path`: 训练用的配置文件路径。
- `--train_input_path`: 指定MaxCompute的表。
- `--eval_input_path`: 指定评估用的MaxCompute表。

## 评估与导出模型

### 评估命令
完成模型训练后，进行模型评估：

```plain
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.eval \
--pipeline_config_path /mnt/data/multi_tower_din_odps.config \
--eval_input_path odps://{project_name}/tables/{your_eval_table_name}
```

### 导出模型
导出训练好的模型：

```plain
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.export \
--pipeline_config_path /mnt/data/multi_tower_din_odps.config \
--export_dir /mnt/data/exported_model
```

## 预测
使用导出的模型进行预测：

```plain
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.predict \
--scripted_model_path /mnt/data/exported_model \
--predict_input_path odps://{project_name}/tables/{your_predict_table_name} \
--predict_output_path /mnt/data/predict_result \
--reserved_columns user_id,adgroup_id,clk
```

## 注意事项
1. 确保在MaxCompute控制台中已为相关用户授予必要的权限以访问数据表。
2. 在训练命令中，`ODPS_ENDPOINT`环境变量需设置为对应的MaxCompute服务地址。
3. 配置文件的 `data_config.dataset_type`需设置为 `OdpsDataset`。
4. 根据实际需求调整配置文件中的字段名和模型参数。