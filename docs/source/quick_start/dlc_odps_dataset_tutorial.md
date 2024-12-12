# DLC Tutorial (MaxCompute Table）

此文档提供在阿里云的PAI-DLC（分布式训练容器）中使用TorchEasyRec读取MaxCompute（MC）表进行模型训练的步骤和配置示例。

## 环境准备

### 创建oss数据集

1. 登录[PAI控制台](https://pai.console.aliyun.com/)。
1. 点击 **AI资源管理 - 数据集 -> 创建数据集**。
1. 选择数据存储为 **阿里云文件存储(NAS)**，并挂载到 `/mnt/data` 下。

### 加载训练数据到maxcompute

1. 下载数据到本地

- 训练数据: [taobao_data_train](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz)
- 评估数据: [taobao_data_eval](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz)

2. 在maxcompute创建数据表

- 创建训练表：

```bash
CREATE TABLE taobao_data_train (
    user_id BIGINT,
    cms_segid DOUBLE,
    cms_group_id DOUBLE,
    final_gender_code DOUBLE,
    age_level DOUBLE,
    pvalue_level DOUBLE,
    shopping_level DOUBLE,
    occupation DOUBLE,
    new_user_class_level DOUBLE,
    time_stamp BIGINT,
    click_50_seq__adgroup_id STRING,
    click_50_seq__cate_id STRING,
    click_50_seq__brand STRING,
    adgroup_id BIGINT,
    cate_id BIGINT,
    campaign_id BIGINT,
    customer BIGINT,
    brand DOUBLE,
    price DOUBLE,
    pid STRING,
    clk INT,
    buy INT
)
;
```

- 创建评估表：

```bash
CREATE TABLE taobao_data_test (
    user_id BIGINT,
    cms_segid DOUBLE,
    cms_group_id DOUBLE,
    final_gender_code DOUBLE,
    age_level DOUBLE,
    pvalue_level DOUBLE,
    shopping_level DOUBLE,
    occupation DOUBLE,
    new_user_class_level DOUBLE,
    time_stamp BIGINT,
    click_50_seq__adgroup_id STRING,
    click_50_seq__cate_id STRING,
    click_50_seq__brand STRING,
    adgroup_id BIGINT,
    cate_id BIGINT,
    campaign_id BIGINT,
    customer BIGINT,
    brand DOUBLE,
    price DOUBLE,
    pid STRING,
    clk INT,
    buy INT,
    ds STRING
)
;
```

4. 将数据导入maxcompute数据表

- 将数据上传oss
- 在dataworks中使用数据集成将数据导入MC。参考:[dataworks数据集成](https://help.aliyun.com/zh/dataworks/user-guide/overview-6?spm=a2c4g.11186623.0.i1)

### 前置条件

- 在[MaxCompute控制台](https://maxcompute.console.aliyun.com/)的「租户管理」->「租户属性」页面打开**开放存储(Storage API)开关**。
- 根据需要赋予用户权限，具体参考[租户权限](https://help.aliyun.com/zh/maxcompute/user-guide/perform-access-control-based-on-tenant-level-roles#section-mt7-tmu-f49)。

```bash
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

#### 配置文件

配置文件以DIN为例 [multi_tower_din_taobao_dlc_mc.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_dlc_mc.config)

### 配置任务

进入[PAI控制台](https://pai.console.aliyun.com)，并选择需要使用的工作空间，点击 **模型开发与训练-分布式训练(DLC)**，点击创建任务。

**节点镜像** 选择官方镜像`torcheasyrec:0.6.0-pytorch2.5.0-gpu-py311-cu121-ubuntu22.04`

**数据集配置** 选择刚新建的NAS数据集

**资源配置** 选择框架为PyTorch，任务资源我们以选择单机8卡V100为例（建议优先选择单机多卡机型，需要多机多卡训练时建议选择带RDMA的机型）

**执行命令** 如下

### 训练命令

设置`ODPS_ENDPOINT`为对应的MaxCompute服务地址，执行训练命令：

```bash
cd /mnt/data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_dlc_mc.config
# 安装tzrec并启动训练
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.train_eval \
--pipeline_config_path /mnt/data/multi_tower_din_taobao_dlc_mc.config \
--train_input_path odps://rec_template/tables/taobao_data_train \
--eval_input_path odps://rec_template/tables/taobao_data_test \
--model_dir /mnt/data/multi_tower_din_odps_dlc
```

- `--pipeline_config_path`: 训练用的配置文件路径。
- `--train_input_path`: 指定训练用的MaxCompute的表。
- `--eval_input_path`: 指定评估用的MaxCompute表。

MaxCompute表按如下格式设置：
odps://{project}/tables/{table_name}/{partition}，多表按逗号分隔
如果单表需要设置多个分区，可以用&简写，来分隔多个分区，odps://{project}/tables/{table_name}/{partition1}&{partition2}

### 评估命令

完成模型训练后，进行模型评估：

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.eval \
--pipeline_config_path /mnt/data/multi_tower_din_odps_dlc/pipeline.config \
--eval_input_path odps://rec_template/tables/taobao_data_test
```

### 导出模型

导出训练好的模型：

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.export \
--pipeline_config_path /mnt/data/multi_tower_din_odps_dlc/pipeline.config \
--export_dir /mnt/data/multi_tower_din_odps_dlc/exported_model
```

### 预测

使用导出的模型进行预测：

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.predict \
--scripted_model_path /mnt/data/multi_tower_din_odps_dlc/exported_model \
--predict_input_path odps://rec_template/tables/taobao_data_test \
--predict_output_path odps://rec_template/tables/taobao_data_test_output \
--reserved_columns user_id,adgroup_id,clk
```

## 注意事项

1. 确保在MaxCompute控制台中已为相关用户授予必要的权限以访问数据表。
1. 在训练命令中，`ODPS_ENDPOINT`环境变量需设置为对应的MaxCompute服务地址。
