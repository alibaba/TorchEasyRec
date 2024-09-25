# 模型服务

TorchEasyRec Processor是一套高性能的Torch推荐模型的推理服务，可以在阿里云[模型在线服务(PAI-EAS)](https://help.aliyun.com/document_detail/113696.html)来部署。TorchEasyRec Processor联合优化了特征生成(FG)、[FeatureStore](https://help.aliyun.com/zh/pai/user-guide/featurestore-overview)特征缓存和TorchEasyRec的模型推理，在保证离在线一致性的同时，提供了高性能的推理。

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
  "processor":"easyrec-torch-0.1"
}
EOF

### 创建服务
eascmd -i <AccessKeyID> -k <AccessKeySecret> -e <EndPoint> create alirec_rank.json

### 更新服务
eascmd -i <AccessKeyID> -k <AccessKeySecret> -e <EndPoint> modify alirec_rank -s alirec_rank.json
```

## 模型服务参数

其中关键参数说明如下，其他参数说明，请参见[服务模型所有相关参数说明](https://help.aliyun.com/zh/pai/user-guide/parameters-of-model-services)。

| **参数**                       | **是否必选** | **描述**                                                                                                                                                                                                                                                          | **示例**                                                             |
| ------------------------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **fg_mode**                    | 否           | 特征生成(FG)模式，取值如下：<br>• bypass（默认值）：不使用FG <br>• normal：使用FG                                                                                                                                                                                 | "fg_mode": "normal"                                                  |
| **fg_threads**                 | 否           | 单请求执行FG的并发线程数                                                                                                                                                                                                                                          | "fg_threads": 15                                                     |
| outputs                        | 否           | 模型预测的输出变量名，如probs_ct，多个输出用逗号分隔。默认输出所有的输出变量                                                                                                                                                                                      | "outputs":"probs_ctr,probs_cvr"                                      |
| item_empty_score               | 否           | item id不存在时，默认的打分情况。默认值为0                                                                                                                                                                                                                        | "item_empty_score": -1                                               |
| **FeatureStore相关参数**       |              |                                                                                                                                                                                                                                                                   |                                                                      |
| fs_project                     | 否           | FeatureStore 项目名称，使用 FeatureStore 时需指定该字段                                                                                                                                                                                                           | "fs_project": "fs_demo"                                              |
| fs_model                       | 否           | FeatureStore模型特征名称                                                                                                                                                                                                                                          | "fs_model": "fs_rank_v1"                                             |
| fs_entity                      | 否           | FeatureStore实体名称                                                                                                                                                                                                                                              | "fs_entity": "item"                                                  |
| region                         | 否           | FeatureStore所在的地区                                                                                                                                                                                                                                            | "region": "cn-beijing"                                               |
| access_key_id                  | 否           | FeatureStore的access_key_id。                                                                                                                                                                                                                                     | "access_key_id": "xxxxx"                                             |
| access_key_secret              | 否           | FeatureStore的access_key_secret。                                                                                                                                                                                                                                 | "access_key_secret": "xxxxx"                                         |
| load_feature_from_offlinestore | 否           | 离线特征是否直接从FeatureStore OfflineStore中获取数据，取值如下：<br>• True：会从FeatureStore OfflineStore中获取数据<br>• False（默认值）：会从FeatureStore OnlineStore中获取数据                                                                                 | "load_feature_from_offlinestore": True                               |
| featuredb_username             | 否           | featuredb用户名                                                                                                                                                                                                                                                   | "featuredb_username":"xxx"                                           |
| featuredb_password             | 否           | featuredb密码                                                                                                                                                                                                                                                     | "featuredb_password":"xxx"                                           |
| **特征自动扩展相关参数**       |              |                                                                                                                                                                                                                                                                   |                                                                      |
| INPUT_TILE                     | 否           | 对User侧特征自动扩展，开启可减少请求大小、网络传输时间和计算时间。必须在fg_mode=normal下使用，并且TorchEasyRec导出时需加上此环境变量 <br>• **INPUT_TILE=2**：user侧特征fg仅计算一次 <br>• **INPUT_TILE=3**：user侧embedding计算仅一，适用于user侧特征比较多的情况 | "processor_envs": <br>\[{"name": "INPUT_TILE", <br>"value": "2"}\]   |
| NO_GRAD_GUARD                  | 否           | 推理时禁止梯度计算                                                                                                                                                                                                                                                | "processor_envs": <br>\[{"name": "NO_GRAD_GUARD", <br>"value":"1"}\] |

## 服务调用

### 通过PAI-REC引擎调用模型服务（建议）

[PAI-REC引擎](https://help.aliyun.com/zh/airec/pairec/user-guide/basic-introduction-1)是一款基于 go 的在线推荐服务引擎的框架，用户可以基于此框架快速搭建推荐在线服务。

基于PAI-REC引擎调用模型服务，可以简化我们的请求构造方式，参考[精排配置](https://help.aliyun.com/zh/airec/pairec/user-guide/fine-discharge-configuration)进行简单的配置就可以实现调用。并且PAI-REC还可以支持进一步对模型服务做[离在线一致性诊断](https://help.aliyun.com/zh/airec/pairec/user-guide/consistency-check)，[AB实验](https://help.aliyun.com/zh/airec/pairec/user-guide/lbvk1rmr56ksdihg)等。

### 直接调用TorchEasyRec Processor

TorchEasyRec Processor的请求是proto格式定义，定义如下。可参考[EAS构建通用Processor请求文档](https://help.aliyun.com/zh/pai/user-guide/construct-a-request-for-a-tensorflow-service)自行实现调用逻辑，支持Python、Java或其他语言

#### torch_predict.proto：Torch模型的请求定义

```protobuf
syntax = "proto3";

package torch.eas;
option cc_enable_arenas = true;

enum ArrayDataType {
  // Not a legal value for DataType. Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types
}

// Dimensions of an array
message ArrayShape {
repeated int64 dim = 1 [packed = true];
}

// Protocol buffer representing an array
message ArrayProto {
  // Data Type.
  ArrayDataType dtype = 1;

  // Shape of the array.
  ArrayShape array_shape = 2;

  // DT_FLOAT.
repeated float float_val = 3 [packed = true];

  // DT_DOUBLE.
repeated double double_val = 4 [packed = true];

  // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
repeated int32 int_val = 5 [packed = true];

  // DT_STRING.
  repeated bytes string_val = 6;

  // DT_INT64.
repeated int64 int64_val = 7 [packed = true];

}


message PredictRequest {

  // Input tensors. choose one of 2
  repeated ArrayProto inputs = 1;
  map<string, ArrayProto> map_inputs = 2;

  // Output filter.
  repeated int32 output_filter = 3;
}

// Response for PredictRequest on successful run.
message PredictResponse {
  // Output tensors.choose one of 2
  repeated ArrayProto outputs = 1;
  map<string, ArrayProto> map_outputs = 2;
}

```

#### predict.proto: Torch模型+FG的请求定义

```protobuf
syntax = "proto3";

package com.alibaba.pairec.processor;

import "torch_predict.proto";

//long->others
message LongStringMap {
  map<int64, string> map_field = 1;
}
message LongIntMap {
  map<int64, int32> map_field = 1;
}
message LongLongMap {
  map<int64, int64> map_field = 1;
}
message LongFloatMap {
  map<int64, float> map_field = 1;
}
message LongDoubleMap {
  map<int64, double> map_field = 1;
}

//string->others
message StringStringMap {
  map<string, string> map_field = 1;
}
message StringIntMap {
  map<string, int32> map_field = 1;
}
message StringLongMap {
  map<string, int64> map_field = 1;
}
message StringFloatMap {
  map<string, float> map_field = 1;
}
message StringDoubleMap {
  map<string, double> map_field = 1;
}

//int32->others
message IntStringMap {
  map<int32, string> map_field = 1;
}
message IntIntMap {
  map<int32, int32> map_field = 1;
}
message IntLongMap {
  map<int32, int64> map_field = 1;
}
message IntFloatMap {
  map<int32, float> map_field = 1;
}
message IntDoubleMap {
  map<int32, double> map_field = 1;
}

// list
message IntList {
  repeated int32 features = 1;
}
message LongList {
  repeated int64 features  = 1;
}

message FloatList {
  repeated float features = 1;
}
message DoubleList {
  repeated double features = 1;
}
message StringList {
  repeated string features = 1;
}

// lists
message IntLists {
  repeated IntList lists = 1;
}
message LongLists {
  repeated LongList lists = 1;
}

message FloatLists {
  repeated FloatList lists = 1;
}
message DoubleLists {
  repeated DoubleList lists = 1;
}
message StringLists {
  repeated StringList lists = 1;
}

message PBFeature {
  oneof value {
    int32 int_feature = 1;
    int64 long_feature = 2;
    string string_feature = 3;
    float float_feature = 4;
    double double_feature=5;

    LongStringMap long_string_map = 6;
    LongIntMap long_int_map = 7;
    LongLongMap long_long_map = 8;
    LongFloatMap long_float_map = 9;
    LongDoubleMap long_double_map = 10;

    StringStringMap string_string_map = 11;
    StringIntMap string_int_map = 12;
    StringLongMap string_long_map = 13;
    StringFloatMap string_float_map = 14;
    StringDoubleMap string_double_map = 15;

    IntStringMap int_string_map = 16;
    IntIntMap int_int_map = 17;
    IntLongMap int_long_map = 18;
    IntFloatMap int_float_map = 19;
    IntDoubleMap int_double_map = 20;

    IntList int_list = 21;
    LongList long_list =22;
    StringList string_list = 23;
    FloatList float_list = 24;
    DoubleList double_list = 25;

    IntLists int_lists = 26;
    LongLists long_lists =27;
    StringLists string_lists = 28;
    FloatLists float_lists = 29;
    DoubleLists double_lists = 30;

}
}

// context features
message ContextFeatures {
  repeated PBFeature features = 1;
}

// PBRequest specifies the request for aggregator
message PBRequest {
  // debug mode
  int32 debug_level = 1;

  // user features
  map<string, PBFeature> user_features = 2;

  // item ids
  repeated string item_ids = 3;

  // context features for each item
  map<string, ContextFeatures> context_features = 4;

// number of nearest neighbors(items) to retrieve
  // from faiss
  int32 faiss_neigh_num = 5;
}

// PBResponse specifies the response for aggregator
message PBResponse {
  // torch output tensors
  map<string, torch.eas.ArrayProto> map_outputs = 1;

  // fg output features
  map<string, string> generate_features = 2;

  // all fg input features
  map<string, string> raw_features = 3;

  // item ids
  repeated string item_ids = 4;

}
```
