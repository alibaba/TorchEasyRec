# Local Tutorial: TDM召回

### 前置准备

TorchEasyRec环境准备参考[Local Tutorial](./local_tutorial.md)

#### 数据

输入数据以parquet格式为例

- 训练样本数据: [taobao_data_recall_train](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_train.tar.gz)
- 评估样本数据: [taobao_data_recall_eval](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_eval.tar.gz)
- 物品池特征数据: [taobao_ad_feature](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature.tar.gz)

#### 配置文件

[tdm_taobao_local.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/tdm_taobao_local.config)

```bash
# 下载并解压
mkdir -p data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_train.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_eval.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/tdm_taobao_local.config
tar xf taobao_data_recall_train.tar.gz -C data
tar xf taobao_data_recall_eval.tar.gz -C data
tar xf taobao_ad_feature.tar.gz -C data
```

### 启动命令

#### 建初始树

```bash
python -m tzrec.tools.tdm.init_tree \
    --item_input_path data/taobao_ad_feature/\*.parquet \
    --item_id_field adgroup_id \
    --cate_id_field cate_id \
    --attr_fields cate_id,campaign_id,customer,brand,price \
    --node_edge_output_file data/init_tree \
    --tree_output_dir data/init_tree
```

- --item_input_path: 建树用的item特征文件
- --item_id_field: 代表item的id的列名
- --cate_id_field: 代表item的类别的列名
- --attr_fields: (可选) 除了item_id外的item非数值型特征列名, 用逗号分开. 注意和配置文件中tdm_sampler顺序一致
- --raw_attr_fields: (可选) item的数值型特征列名, 用逗号分开. 注意和配置文件中tdm_sampler顺序一致
- --attr_delimiter: (可选) 产出的负采样表中attr字段的特征分隔符，默认为","
- --node_edge_output_file: 根据树生成的node和edge表的保存路径, 支持`ODPS表`和`本地txt`两种
  - ODPS表：设置形如`odps://{project}/tables/{tb_prefix}`，将会产出用于TDM训练负采样的GL Node表`odps://{project}/tables/{tb_prefix}_node_table`、GL Edge表`odps://{project}/tables/{tb_prefix}_edge_table`、用于离线检索的GL Edge表`odps://{project}/tables/{tb_prefix}_predict_edge_table`、用于Serving的Node特征表`odps://{project}/tables/{tb_prefix}_node_feature`
  - 本地txt：设置的为目录， 将在目录下产出用于TDM训练负采样的GL Node表`node_table.txt`,GL Edge表`edge_table.txt`、用于离线检索的GL Edge表`predict_edge_table.txt`、用于Serving的Node特征表`node_feature`
- --tree_output_dir: (可选) 树的保存目录, 将会在目录下存储`serving_tree`文件用于线上服务
- --n_cluster: (可选,默认为2)树的分叉数

#### 训练

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=8 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path tdm_taobao_local.config \
    --train_input_path data/taobao_data_recall_train/\*.parquet \
    --eval_input_path data/taobao_data_recall_eval/\*.parquet \
    --model_dir experiments/tdm_taobao_local
    --edit_config_json '{"data_config.tdm_sampler.item_input_path":"data/init_tree/node_table.txt", "data_config.tdm_sampler.edge_input_path":"data/init_tree/edge_table.txt", "data_config.tdm_sampler.predict_edge_input_path":"data/init_tree/predict_edge_table.txt"}'
```

- --pipeline_config_path: 训练用的配置文件
- --train_input_path: 训练数据的输入路径
- --eval_input_path: 评估数据的输入路径
- --model_dir: 模型训练目录
- --edit_config_json: 使用json修改config

#### 导出模型和embedding模块

导出命令会自动导出模型和embedding模块, 模型模块会存在model子目录下，embedding模块会存在embedding子目录下

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=8 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/tdm_taobao_local/pipeline.config \
    --export_dir experiments/tdm_taobao_local/export \
    --asset_files data/init_tree/serving_tree
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认评估model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录
- --asset_files: 需额拷贝到模型目录的文件。tdm需拷贝serving_tree树文件用于线上服务

#### 导出item embedding

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=8 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/tdm_taobao_local/export/embedding \
    --predict_input_path data/taobao_ad_feature/\*.parquet \
    --predict_output_path experiments/tdm_taobao_local/item_emb \
    --reserved_columns adgroup_id,cate_id,campaign_id,customer,brand,price \
    --output_columns item_emb
```

- --scripted_model_path: 要预测的导出模型
- --predict_input_path: 预测数据的输入路径
- --predict_output_path: 预测结果的输出路径
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列

#### 根据item embedding重新聚类建树

```bash
python -m tzrec.tools.tdm.cluster_tree \
    --item_input_path experiments/tdm_taobao_local/item_emb/\*.parquet \
    --item_id_field adgroup_id \
    --embedding_field item_emb \
    --attr_fields cate_id,campaign_id,customer,brand,price \
    --node_edge_output_file data/learnt_tree \
    --tree_output_dir data/learnt_tree \
    --parallel 16
```

- --item_input_path: 建树用的item embedding及特征文件
- --item_id_field: 代表item的id的列名
- --embedding_field: 代表item embedding的列名
- --attr_fields: (可选) 除了item_id外的item非数值型特征列名, 用逗号分开. 注意和配置文件中tdm_sampler顺序一致
- --raw_attr_fields: (可选) item的数值型特征列名, 用逗号分开. 注意和配置文件中tdm_sampler顺序一致
- --attr_delimiter: (可选) 产出的负采样表中attr字段的特征分隔符，默认为","
- --node_edge_output_file: 根据树生成的node和edge表的保存路径, 支持`ODPS表`和`本地txt`两种，同初始树
- --tree_output_dir: (可选) 树的保存目录, 将会在目录下存储`serving_tree`文件用于线上服务
- --n_cluster: (可选,默认为2)树的分叉数
- --parllel: (可选，默认为16)聚类时CPU并行数

#### 重新训练

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=8 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path tdm_taobao_local.config \
    --train_input_path data/taobao_data_recall_train/\*.parquet \
    --eval_input_path data/taobao_data_recall_eval/\*.parquet \
    --model_dir experiments/tdm_taobao_local_learnt \
    --edit_config_json '{"data_config.tdm_sampler.item_input_path":"data/learnt_tree/node_table.txt", "data_config.tdm_sampler.edge_input_path":"data/learnt_tree/edge_table.txt", "data_config.tdm_sampler.predict_edge_input_path":"data/learnt_tree/predict_edge_table.txt"}'
```

- --pipeline_config_path: 训练用的配置文件
- --train_input_path: 训练数据的输入路径
- --eval_input_path: 评估数据的输入路径
- --model_dir: 模型训练目录
- --edit_config_json: 使用json修改config

#### 导出

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=8 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/tdm_taobao_local_learnt/pipeline.config \
    --export_dir experiments/tdm_taobao_local_learnt/export \
    --asset_files data/learnt_tree/serving_tree
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认评估model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录
- --asset_files: 需额拷贝到模型目录的文件。tdm需拷贝serving_tree树文件用于线上服务

#### Recall评估

任意一次训练导出后均可按需评测, 以训练完初始树评测为例:

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=8 --node_rank=0 \
    -m tzrec.tools.tdm.retrieval \
    --scripted_model_path experiments/tdm_taobao_local/export/model/ \
    --predict_input_path data/taobao_data_recall_eval/\*.parquet \
    --predict_output_path data/init_tree/taobao_data_eval_recall \
    --recall_num 200 \
    --n_cluster 2 \
    --reserved_columns user_id,adgroup_id \
    --batch_size 32
```

- --scripted_model_path: 要预测的模型
- --predict_input_path: 预测输入数据的路径
- --predict_output_path: 预测输出数据的路径
- --recall_num:(可选, 默认为200) 召回的数量
- --n_cluster:(可选, 默认为2) 数的分叉数量, 应与建树时输入保持一致
- --reserved_columns: 预测结果中要保留的输入列

#### 注

- 如果item_id为string类型，建树、模型训练、评估命令中需设置环境变量USE_HASH_NODE_ID=1来启动对id自动进行hash64操作，一般情况下hash冲突概率极低

#### 参考手册

[TorchEasyRec配置参考手册](../reference.md)
