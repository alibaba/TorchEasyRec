# DLC Tutorial

[PAI-DLC](https://help.aliyun.com/zh/pai/user-guide/container-training) 是阿里巴巴分布式训练平台，为您提供灵活、稳定、易用和极致性能的深度学习训练环境。

## 创建数据集

进入[PAI控制台](https://pai.console.aliyun.com/)，点击 **AI资源管理-数据集** -> **创建数据集**。选择数据存储为**阿里云文件存储(NAS)**，挂载到`/mnt/data`下。任务运行时，会从挂载路径下读取训练数据和配置文件以及写入模型检查点等。

## 创建DLC任务

### 前置准备

TorchEasyRec查询Nightly版本

```bash
pip index versions tzrec -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
```

#### 数据

输入数据以parquet格式为例

- 训练数据: [taobao_data_train](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz)
- 评估数据: [taobao_data_eval](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz)

#### 配置文件

配置文件以DIN为例 [multi_tower_din_taobao_local.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_local.config)

### 配置任务

进入[PAI控制台](https://pai.console.aliyun.com)，并选择需要使用的工作空间，点击 **模型开发与训练-分布式训练(DLC)**，点击创建任务。

**节点镜像** 填入镜像地址`mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:${TZREC_DOCKER_VERSION}`

**数据集配置** 选择刚新建的NAS数据集

**资源配置** 选择框架为PyTorch，任务资源我们以选择单机8卡V100为例（建议优先选择单机多卡机型，需要多机多卡训练时建议选择带RDMA的机型）

**执行命令** 如下

#### 训练

```bash
# 下载数据和配置文件并解压到nas上，如果数据已存在，则不需要上述下载和解压命令
cd /mnt/data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_dlc.config
tar xf taobao_data_train.tar.gz
tar xf taobao_data_eval.tar.gz
cd -
# 安装tzrec并启动训练
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.train_eval \
    --pipeline_config_path /mnt/data/multi_tower_din_taobao_dlc.config \
    --train_input_path /mnt/data/taobao_data_train/\*.parquet \
    --eval_input_path /mnt/data/taobao_data_eval/\*.parquet \
    --model_dir /mnt/data/multi_tower_din_taobao_dlc
```

- --pipeline_config_path: 训练用的配置文件
- --train_input_path: 训练数据的输入路径
- --eval_input_path: 评估数据的输入路径
- --model_dir: 模型训练目录

#### 评估

```bash
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.eval \
    --pipeline_config_path /mnt/data/multi_tower_din_taobao_dlc/pipeline.config \
    --eval_input_path /mnt/data/taobao_data_eval/\*.parquet
```

- --pipeline_config_path: 评估用的配置文件
- --checkpoint_path: 指定要评估的checkpoint, 默认评估model_dir下面最新的checkpoint
- --eval_input_path: 评估数据的输入路径

#### 导出

```bash
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.export \
    --pipeline_config_path /mnt/data/multi_tower_din_taobao_dlc/pipeline.config \
    --export_dir /mnt/data/multi_tower_din_taobao_dlc/export
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认导出model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录

#### 预测

```bash
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.predict \
    --scripted_model_path /mnt/data/multi_tower_din_taobao_dlc/export \
    --predict_input_path /mnt/data/taobao_data_eval/\*.parquet \
    --predict_output_path /mnt/data/multi_tower_din_taobao_dlc/predict_result \
    --reserved_columns user_id,adgroup_id,clk
```

- --scripted_model_path: 要预测的导出模型
- --predict_input_path: 预测数据的输入路径
- --predict_output_path: 预测结果的输出路径
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列

### 直读MaxCompute数据

设置`ODPS_ENDPOINT`的环境变量，并新建任务时，「角色信息」选择**PAI默认角色**，可以直读MaxCompute表。配置文件的data_config.dataset_type需设置为OdpsDataset。

```bash
ODPS_ENDPOINT=http://service.{region}.maxcompute.aliyun-inc.com/api \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
--nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
-m tzrec.train_eval \
--pipeline_config_path ${PIPELINE_CONFIG} \
--train_input_path odps://{project}/tables/{table_name}/{partition}
```

### 例行训练

点击任务右上角**生成脚本**生成如下DLC命令行脚本，可以通过[DLC命令行](https://help.aliyun.com/document_detail/214317.html)提交任务, 方便在DataWorks等的调度平台中做例行训练

```bash
dlc submit pytorchjob \
    --name=${JOB_NAME} \
    --command='pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK -m tzrec.train_eval --pipeline_config_path /mnt/data/multi_tower_din_taobao_dlc.config --train_input_path /mnt/data/data/taobao_data_train/\*.parquet --eval_input_path /mnt/data/taobao_data_eval/\*.parquet --model_dir /mnt/data/multi_tower_din_taobao_dlc' \
    --data_sources=${DATA_SOURCE} \
    --workspace_id=${WORKSPACE_ID} \
    --vpc_id=${VPC_ID} \
    --switch_id=${VSWITCH_ID} \
    --security_group_id=${SG_ID} \
    --priority=1 \
    --workers=1 \
    --worker_image=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:${TZREC_DOCKER_VERSION} \
    --worker_spec=ecs.gn6v-c10g1.20xlarge
```

通过dlc命令提交的任务也可以在 **任务列表** 中查看.
