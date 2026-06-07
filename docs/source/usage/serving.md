# 模型服务

[TorchEasyRec Processor](https://help.aliyun.com/zh/pai/user-guide/torcheasyrec-processor)是一套高性能的Torch推荐模型的推理服务，可以在阿里云[模型在线服务(PAI-EAS)](https://help.aliyun.com/document_detail/113696.html)来部署。TorchEasyRec Processor联合优化了特征生成(FG)、[FeatureStore](https://help.aliyun.com/zh/pai/user-guide/featurestore-overview)特征缓存和TorchEasyRec的模型推理，在保证离在线一致性的同时，提供了高性能的推理。

## 部署服务

```
cat << EOF > tzrec_rank.json
{
  "metadata": {
    "cpu": 16,
    "instance": 1,
    "memory": 20000,
    "name": "tzrec_rank",
    "resource": "eas-r-xxx",
    "resource_burstable": false,
    "rpc": {
      "enable_jemalloc": 1,
      "max_queue_size": 256,
      "worker_threads": 16
    }
  },
  "model_config": {
    "fg_mode": "normal",
    "fg_threads": 8,
    "region": "YOUR_REGION",
    "fs_project": "YOUR_FS_PROJECT",
    "fs_model": "YOUR_FS_MODEL",
    "fs_entity": "item",
    "load_feature_from_offlinestore": true,
    "access_key_id":"YOUR_ACCESS_KEY_ID",
    "access_key_secret":"YOUR_ACCESS_KEY_SECRET"
  },
  "storage": [
    {
      "mount_path": "/home/admin/docker_ml/workspace/model/",
      "oss": {
        "path": "oss://xxx/xxx/export",
        "readOnly": false
      },
      "properties": {
        "resource_type": "code"
      }
    }
  ],
  "processor":"easyrec-torch-2.2"
}
EOF

### 创建服务
eascmd -i <AccessKeyID> -k <AccessKeySecret> -e <EndPoint> create alirec_rank.json

### 更新服务
eascmd -i <AccessKeyID> -k <AccessKeySecret> -e <EndPoint> modify alirec_rank -s alirec_rank.json
```

## 模型服务参数

其中关键参数说明如下，其他参数说明，请参见[服务模型所有相关参数说明](https://help.aliyun.com/zh/pai/user-guide/parameters-of-model-services)。

| **参数**                       | **是否必选** | **描述**                                                                                                                                                                                                                                                                | **示例**                                                             |
| ------------------------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **fg_mode**                    | 否           | 特征生成(FG)模式，取值如下：<br>• bypass（默认值）：不使用FG <br>• normal：使用FG                                                                                                                                                                                       | "fg_mode": "normal"                                                  |
| **fg_threads**                 | 否           | 单请求执行FG的并发线程数                                                                                                                                                                                                                                                | "fg_threads": 15                                                     |
| outputs                        | 否           | 模型预测的输出变量名，如probs_ctr，多个输出用逗号分隔。默认输出所有的输出变量                                                                                                                                                                                           | "outputs":"probs_ctr,probs_cvr"                                      |
| item_empty_score               | 否           | item id不存在时，默认的打分情况。默认值为0                                                                                                                                                                                                                              | "item_empty_score": -1                                               |
| path                           | 是           | 服务模型文件的OSS存储路径，用于存放导出的模型（通常通过 storage 挂载配置）                                                                                                                                                                                              | "path": "oss://examplebucket/xxx/export"                             |
| **FeatureStore相关参数**       |              |                                                                                                                                                                                                                                                                         |                                                                      |
| fs_project                     | 否           | FeatureStore 项目名称，使用 FeatureStore 时需指定该字段                                                                                                                                                                                                                 | "fs_project": "fs_demo"                                              |
| fs_model                       | 否           | FeatureStore模型特征名称                                                                                                                                                                                                                                                | "fs_model": "fs_rank_v1"                                             |
| fs_entity                      | 否           | FeatureStore实体名称，支持多个实体，用半角逗号分隔，如 "item,raw"                                                                                                                                                                                                       | "fs_entity": "item"                                                  |
| region                         | 否           | FeatureStore所在的地区                                                                                                                                                                                                                                                  | "region": "cn-beijing"                                               |
| access_key_id                  | 否           | FeatureStore的access_key_id。                                                                                                                                                                                                                                           | "access_key_id": "xxxxx"                                             |
| access_key_secret              | 否           | FeatureStore的access_key_secret。                                                                                                                                                                                                                                       | "access_key_secret": "xxxxx"                                         |
| load_feature_from_offlinestore | 否           | 离线特征是否直接从FeatureStore OfflineStore中获取数据，取值如下：<br>• True：会从FeatureStore OfflineStore中获取数据<br>• False（默认值）：会从FeatureStore OnlineStore中获取数据                                                                                       | "load_feature_from_offlinestore": True                               |
| featuredb_username             | 否           | featuredb用户名                                                                                                                                                                                                                                                         | "featuredb_username":"xxx"                                           |
| featuredb_passwd               | 否           | featuredb密码                                                                                                                                                                                                                                                           | "featuredb_passwd":"xxx"                                             |
| **特征自动扩展相关参数**       |              |                                                                                                                                                                                                                                                                         |                                                                      |
| INPUT_TILE                     | 否           | 对User侧特征自动扩展，开启可减少请求大小、网络传输时间和计算时间。必须在fg_mode=normal下使用，并且TorchEasyRec导出时需加上此环境变量 <br>• **INPUT_TILE=2**：user侧特征fg仅计算一次 <br>• **INPUT_TILE=3**：user侧fg和embedding计算仅一次，适用于user侧特征比较多的情况 | "processor_envs": <br>\[{"name": "INPUT_TILE", <br>"value": "2"}\]   |
| NO_GRAD_GUARD                  | 否           | 推理时禁止梯度计算。<br>当设置为1时，可能会出现部分模型不兼容的情况。如果在第二次运行推理过程中遇到卡顿问题，可以通过添加环境变量PYTORCH_TENSOREXPR_FALLBACK=2来解决，这样可以跳过编译步骤，同时保留一定的图优化功能。                                                  | "processor_envs": <br>\[{"name": "NO_GRAD_GUARD", <br>"value":"1"}\] |
| **向量召回相关参数**           |              |                                                                                                                                                                                                                                                                         |                                                                      |
| faiss_nprobe                   | 否           | 向量检索时的检索簇数                                                                                                                                                                                                                                                    | "faiss_nprobe": 100                                                  |
| faiss_neigh_num                | 否           | 向量检索时的召回数                                                                                                                                                                                                                                                      | "faiss_neigh_num": 200                                               |
| **模型预热相关参数**           |              |                                                                                                                                                                                                                                                                         |                                                                      |
| warmup_data_path               | 否           | 开启warmup功能并指定warmup文件的保存路径，需在storage中增加一个oss挂载到该路径                                                                                                                                                                                          | "warmup_data_path": "/warmup"                                        |
| warmup_cnt_per_file            | 否           | 每个warmup pb文件的预热次数，适当增大可保证预热充分，但预热时间会延长。默认值20                                                                                                                                                                                         | "warmup_cnt_per_file": 20                                            |
| warmup_pb_files_count          | 否           | 保存线上请求的个数，保存为pb文件供下次启动预热使用，保存路径为warmup_data_path。默认值64                                                                                                                                                                                | "warmup_pb_files_count": 64                                          |
| **请求落盘与慢请求相关参数**   |              |                                                                                                                                                                                                                                                                         |                                                                      |
| request_log_path               | 否           | 请求pb文件的落盘路径，需在storage中用oss挂载该路径                                                                                                                                                                                                                      | "request_log_path": "/online_log_pb"                                 |
| background_feature_thread_num  | 否           | 负责后台落盘任务的线程数，落盘任务较重时可适当调大。默认值4                                                                                                                                                                                                             | "background_feature_thread_num": 8                                   |
| long_request_threshold         | 否           | 慢请求的时间阈值，单位ms，超过阈值的请求会在日志中记录各阶段执行时间。默认值200                                                                                                                                                                                         | "long_request_threshold": 200                                        |
| save_long_request              | 否           | 出现慢请求(超过long_request_threshold)时是否将请求保存为pb文件，保存路径为模型目录下torch_req文件夹。默认false                                                                                                                                                          | "save_long_request": true                                            |
| **TDM召回相关参数**            |              |                                                                                                                                                                                                                                                                         |                                                                      |
| tdm_retrieval_num              | 否           | TDM召回检索时的召回数                                                                                                                                                                                                                                                   | "tdm_retrieval_num": 200                                             |

## 服务调用

### 通过PAI-REC引擎调用模型服务（建议）

[PAI-REC引擎](https://help.aliyun.com/zh/airec/pairec/user-guide/basic-introduction-1)是一款基于 go 的在线推荐服务引擎的框架，用户可以基于此框架快速搭建推荐在线服务。

基于PAI-REC引擎调用模型服务，可以简化我们的请求构造方式，参考[精排配置](https://help.aliyun.com/zh/airec/pairec/user-guide/fine-discharge-configuration)进行简单的配置就可以实现调用。并且PAI-REC还可以支持进一步对模型服务做[离在线一致性诊断](https://help.aliyun.com/zh/airec/pairec/user-guide/consistency-check)，[AB实验](https://help.aliyun.com/zh/airec/pairec/user-guide/lbvk1rmr56ksdihg)等。

### 直接调用TorchEasyRec Processor

TorchEasyRec Processor的请求是proto格式定义，定义如下。可参考[TorchEasyRec Processor文档](https://help.aliyun.com/zh/pai/user-guide/torcheasyrec-processor)的调用服务章节使用sdk进行调用
