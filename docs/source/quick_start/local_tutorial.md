# Local Tutorial: 排序

## 安装TorchEasyRec

我们提供了**本地Conda安装**和**Docker镜像启动**两种方式。

TorchEasyRec查询Nightly版本

```bash
pip index versions tzrec -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
```

### 本地Conda安装

```bash
conda create -n tzrec python=3.11
conda activate tzrec
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install fbgemm-gpu==1.2.0 --index-url https://download.pytorch.org/whl/cu126
pip install torchmetrics==1.0.3 tensordict
pip install torchrec==1.2.0 --index-url https://download.pytorch.org/whl/cu126
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
```

### Docker镜像启动 (推荐)

```bash
docker run -td --gpus all --shm-size 10gb  --network host mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:${TZREC_DOCKER_VERSION}
docker exec -it <CONTAINER_ID> bash
pip install tzrec==${TZREC_NIGHTLY_VERSION} -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com
```

注：

```
GPU版本（CUDA 12.6) 镜像地址：
  mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:${TZREC_DOCKER_VERSION}-cu126
CPU版本 镜像地址:
  mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:${TZREC_DOCKER_VERSION}-cpu
```

## 前置准备

### 数据

输入数据以parquet格式为例

- 训练数据: [taobao_data_train](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz)
- 评估数据: [taobao_data_eval](https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz)

### 配置文件

配置文件以DIN为例 [multi_tower_din_taobao_local.config](https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_local.config)

```bash
# 下载并解压
mkdir -p data
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_train.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/quick_start/taobao_data_eval.tar.gz
wget https://tzrec.oss-cn-beijing.aliyuncs.com/config/quick_start/multi_tower_din_taobao_local.config
tar xf taobao_data_train.tar.gz -C data
tar xf taobao_data_eval.tar.gz -C data
```

## 启动命令

### 训练

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
- --train_input_path: 训练数据的输入路径
- --eval_input_path: 评估数据的输入路径
- --model_dir: 模型训练目录

### 评估

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.eval \
    --pipeline_config_path experiments/multi_tower_din_taobao_local/pipeline.config \
    --eval_input_path data/taobao_data_eval/\*.parquet
```

- --pipeline_config_path: 评估用的配置文件
- --checkpoint_path: 指定要评估的checkpoint, 默认评估model_dir下面最新的checkpoint
- --eval_input_path: 评估数据的输入路径

### 导出

```bash
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/multi_tower_din_taobao_local/pipeline.config \
    --export_dir experiments/multi_tower_din_taobao_local/export
```

- --pipeline_config_path: 导出用的配置文件
- --checkpoint_path: 指定要导出的checkpoint, 默认导出model_dir下面最新的checkpoint
- --export_dir: 导出到的模型目录

### 预测

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
- --predict_output_path: 预测结果的输出路径
- --reserved_columns: 预测结果中要保留的输入列
- --output_columns: 预测结果中的模型输出列

## 配置文件

### 输入输出

```
# 训练文件和评估文件
train_input_path: "data/taobao_data_train/*.parquet"
eval_input_path: "data/taobao_data_eval/*.parquet"
# 模型保存路径
model_dir: "experiments/multi_tower_din_taobao_local"
```

### 数据相关

```
# 数据相关的描述
data_config {
    batch_size: 8192
    # 输入数据类型，还可以支持 CsvDataset | OdpsDataset 等
    dataset_type: ParquetDataset
    # 数据是否已经进行FG编码，如果为false，将会进行训练时FG
    fg_encoded: false
    # Label的名称
    label_fields: "clk"
    # 每个proc上的读数据并行度
    num_workers: 8
}
```

### 特征相关

特征配置具体见：[特征](../feature/feature.md)

```
feature_configs {
    # 特征类型
    id_feature {
        # 特征名称
        feature_name: "user_id"
        # 特征来源字段
        expression: "user:user_id"
        # 分箱大小
        num_buckets: 1141730
        # embedding向量的dimension
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "cms_segid"
        expression: "user:cms_segid"
        num_buckets: 98
        embedding_dim: 16
    }
}
...
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        boundaries: [1.1, 2.2, 3.6, 5.2, 7.39, 9.5, 10.5, 12.9, 15, 17.37, 19, 20, 23.8, 25.8, 28, 29.8, 31.5, 34, 36, 38, 39, 40, 45, 48, 49, 51.6, 55.2, 58, 59, 63.8, 68, 69, 72, 78, 79, 85, 88, 90, 97.5, 98, 99, 100, 108, 115, 118, 124, 128, 129, 138, 139, 148, 155, 158, 164, 168, 171.8, 179, 188, 195, 198, 199, 216, 228, 238, 248, 258, 268, 278, 288, 298, 299, 316, 330, 352, 368, 388, 398, 399, 439, 478, 499, 536, 580, 599, 660, 699, 780, 859, 970, 1080, 1280, 1480, 1776, 2188, 2798, 3680, 5160, 8720]
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "pid"
        expression: "context:pid"
        hash_bucket_size: 20
        embedding_dim: 16
    }
}
feature_configs {
    sequence_feature {
        # 序列特征名称
        sequence_name: "click_50_seq"
        sequence_length: 50
        sequence_delim: "|"
        features {
            id_feature {
                feature_name: "adgroup_id"
                expression: "item:adgroup_id"
                num_buckets: 846812
                embedding_dim: 16
            }
        }
        features {
            id_feature {
                feature_name: "cate_id"
                expression: "item:cate_id"
                num_buckets: 12961
                embedding_dim: 16
            }
        }
        ...
    }
}
```

### 训练相关

```
train_config {
    # Embedding优化器相关的参数
    sparse_optimizer {
        adagrad_optimizer {
            lr: 0.001
        }
        constant_learning_rate {
        }
    }
    # NN优化器相关的参数
    dense_optimizer {
        adam_optimizer {
            lr: 0.001
        }
        constant_learning_rate {
        }
    }
    # 训练的epoch数
    num_epochs: 1
}
```

### 评估相关

```
eval_config {
  # 仅仅评估10步的样本
  num_steps: 10
}
```

### 模型相关

```
model_config {
    # DNN特征组配置
    feature_groups {
        group_name: "deep"
        feature_names: "user_id"
        feature_names: "cms_segid"
        feature_names: "cms_group_id"
        feature_names: "final_gender_code"
        feature_names: "age_level"
        feature_names: "pvalue_level"
        feature_names: "shopping_level"
        feature_names: "occupation"
        feature_names: "new_user_class_level"
        feature_names: "adgroup_id"
        feature_names: "cate_id"
        feature_names: "campaign_id"
        feature_names: "customer"
        feature_names: "brand"
        feature_names: "price"
        feature_names: "pid"
        group_type: DEEP
    }
    # 序列特征组配置
    feature_groups {
        group_name: "seq"
        feature_names: "adgroup_id"
        feature_names: "cate_id"
        feature_names: "brand"
        feature_names: "click_50_seq__adgroup_id"
        feature_names: "click_50_seq__cate_id"
        feature_names: "click_50_seq__brand"
        group_type: SEQUENCE
    }
    multi_tower_din {
        towers {
            input: 'deep'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        din_towers {
            input: 'seq'
            attn_mlp {
                hidden_units: [256, 64]
            }
        }
        final {
            hidden_units: [64]
        }
    }
    # 指标配置
    metrics {
        auc {}
    }
    # 损失函数配置
    losses {
        binary_cross_entropy {}
    }
}
```

### 参考手册

[TorchEasyRec配置参考手册](../reference.md)
