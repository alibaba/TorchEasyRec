# Local Tutorial: U2I向量召回

## 前置准备

TorchEasyRec环境准备参考[Local Tutorial](./local_tutorial.md)

### 数据

输入数据以parquet格式为例

- 训练样本数据: [taobao_data_recall_train](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_train.tar.gz)
- 评估样本数据: [taobao_data_recall_eval](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_eval.tar.gz)
- 物品池特征数据: [taobao_ad_feature](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature.tar.gz)
- 负采样物品数据: [taobao_ad_feature_gl](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature_gl)
  - 当物品池很大上百万甚至是上亿的时候，U2I双塔召回模型常常需要在物品池中针对每个正样本采样一千甚至一万的负样本才能跟在线向量检索推理的分布一致，达到比较好的召回效果。因此需要将物品池组织成GraphLearn格式的负采样表，在训练时动态采样负样本。
  - Local模式下，GraphLearn格式的加权随机负采样包含3列，以"\\t"分割
    - 第一行固定为 "id:int64\\tweight:float\\tattrs:string"
    - 后续列中，id为物品id的值，weight为采样权重值，attrs为"\\x02"分隔符拼接的物品特征

### 配置文件

配置文件以DSSM为例 [dssm_taobao_local.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/dssm_taobao_local.config)

```bash
# 下载并解压
mkdir -p data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_train.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_recall_eval.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_ad_feature_gl -O data/taobao_ad_feature_gl
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/dssm_taobao_local.config
tar xf taobao_data_recall_train.tar.gz -C data
tar xf taobao_data_recall_eval.tar.gz -C data
tar xf taobao_ad_feature.tar.gz -C data
```

## 启动命令

### 训练

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path dssm_taobao_local.config \
    --train_input_path data/taobao_data_recall_train/\*.parquet \
    --eval_input_path data/taobao_data_recall_eval/\*.parquet \
    --model_dir experiments/dssm_taobao_local \
    --edit_config_json '{"data_config.negative_sampler.input_path":"data/taobao_ad_feature_gl"}'
```

- --pipeline_config_path: 训练用的配置文件
- --train_input_path: 训练数据的输入路径
- --eval_input_path: 评估数据的输入路径
- --model_dir: 模型训练目录
- --edit_config_json: 使用json修改config

### 导出 & 切图

导出命令会自动对向量召回模型进行切图，分别放在user和item的子目录下

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/dssm_taobao_local/pipeline.config \
    --export_dir experiments/dssm_taobao_local/export
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认导出model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录

### 向量推理

**Item**: 对物品池做推理得到物品池的向量

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/dssm_taobao_local/export/item \
    --predict_input_path data/taobao_ad_feature/\*.parquet \
    --predict_output_path experiments/dssm_taobao_local/item_emb \
    --reserved_columns adgroup_id \
    --output_columns item_tower_emb
```

- --scripted_model_path: 要预测的导出模型
- --predict_input_path: 预测数据的输入路径
- --predict_output_path: 预测结果的输出路径
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列

**User**: 对评估样本中用户向量进行推理，预留物品ID列作为后续hitrate评估计算的Label

NOTE: 一般情况下可以对评估样本中按RequestId或UserId去重，减少用户向量推理的计算量

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/dssm_taobao_local/export/user \
    --predict_input_path data/taobao_data_recall_eval/\*.parquet \
    --predict_output_path experiments/dssm_taobao_local/user_emb \
    --reserved_columns user_id,adgroup_id \
    --output_columns user_tower_emb
```

### HitRate评估

```bash
OMP_NUM_THREADS=16 torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.tools.hitrate \
    --user_gt_input experiments/dssm_taobao_local/user_emb/\*.parquet \
    --item_embedding_input experiments/dssm_taobao_local/item_emb/\*.parquet \
    --item_id_field adgroup_id \
    --request_id_field user_id \
    --gt_items_field adgroup_id
```

- --user_gt_input: 用户向量和Label表，需包含request_id | gt_items | user_tower_emb三列
- --item_embedding_input: 物品池向量表，需包含item_id | item_tower_emb两列
- --total_hitrate_output: （可选）hitrate输出表
- --hitrate_details_output: （可选）hitrate详情输出表，会包含id | topk_ids | topk_dists
  | hitrate | hit_ids 五列
- --batch_size: 评估batch_size，默认为1024
- --index_type: 评估检索方式，默认为IVFFlatIP，可以选 [IVFFlatIP, IVFFlatL2]
- --top_k: 评估TopK召回的Hitrate，默认200
- --ivf_nlist: IVFFlat索引的聚簇中心个数，默认为1000
- --ivf_nprobe: IVFFlat索引的检索中心个数，默认为800
- --item_id_field: 物品池向量表中item_id的列名
- --item_embedding_field: 物品池向量表中embedding的列名，该列支持数组或","分隔的string
- --request_id_field: 用户向量表中的id列名
- --gt_items_field: 用户向量表中的标签gt_items列名，该列支持数组或","分隔的string
- --user_embedding_field: 用户向量表中embedding的列名，该列支持数组或","分隔的string

### 索引构建

在线推理时， TorchEasyRec模型推理服务中会同时做向量检索，以保证模型版本和索引版本的一致性。可以用`create_faiss_index`命令构建物品池的索引供模型推理服务使用

```bash
python -m tzrec.tools.create_faiss_index \
    --embedding_input_path experiments/dssm_taobao_local/item_emb/\*.parquet \
    --index_output_dir experiments/dssm_taobao_local/export/user \
    --id_field adgroup_id \
    --embedding_field item_tower_emb
```

- --embedding_input_path: 物品池向量表，需包含item_id | item_tower_emb两列
- --index_output_dir: 物品池索引输出目录，一般指定用户塔目录，以保证模型版本和索引版本同时切换
- --batch_size: 索引构建batch_size，默认为1024
- --index_type: 评估检索方式，默认为IVFFlatIP，可以选 [IVFFlatIP, HNSWFlatIP, IVFFlatL2, HNSWFlatL2]
- --ivf_nlist: IVFFlat索引的聚簇中心个数，默认为1000
- --hnsw_M: HNSWFlat索引的M参数
- --hnsw_efConstruction: HNSWFlat索引的efConstruction参数
- --id_field: 物品池向量表中item_id的列名
- --embedding_field: 物品池向量表中embedding的列名，该列支持数组或","分隔的string

#### 注

- 负采样表中的id列也可以为string类型，当id为string类型时，模型训练和评估命令中需设置环境变量USE_HASH_NODE_ID=1来启动对id自动进行hash64操作，一般情况下hash冲突概率极低

### 参考手册

[TorchEasyRec配置参考手册](../reference.md)
