# 数据格式

TorchEasyRec作为阿里云PAI的推荐算法包，可以无缝对接MaxCompute的数据表，也可以读取OSS、NAS或Local环境中的CSV, Parquet文件。

## DataConfig

**一个最简单的data config的配置**

这个配置里面，读取MaxCompute的表作为输入数据（OdpsDataset），并且输入数据已经编码好，每个worker上以8192的batch_size，并行度为8来读取数据

```
data_config {
    batch_size: 8192
    dataset_type: OdpsDataset
    fg_mode: FG_NONE
    label_fields: "clk"
    num_workers: 8
}
```

如果希望在训练过程带上样本权重，支持在data_config中增加配置项
sample_weight_fields: 'col_name'

### dataset_type

目前支持一下几种[input_type](../proto.html#tzrec.protos.DatasetType):

- OdpsDataset: 输入数据为MaxCompute表

  - **前置条件**:

    - 在[MaxCompute控制台](https://maxcompute.console.aliyun.com/)的「租户管理」->「租户属性」页面打开**开放存储(Storage API)开关**
    - 「租户管理」->「新增成员」给相应用户授予「admin」权限；或参考[租户权限](https://help.aliyun.com/zh/maxcompute/user-guide/overview-1#cabfa502c288o)文档，精细授予用户Quota的使用权限

  - input_path: 按如下格式设置

    - `odps://{project}/tables/{table_name}/{partition}`，多表按逗号分隔
    - 如果单表需要设置多个分区，可以用`&`简写，来分隔多个分区，`odps://{project}/tables/{table_name}/{partition1}&{partition2}`

  - 运行训练/评估/导出/预测等命令时

    - **本地环境**：
      - 需要准备一个odps_conf文件，并在启动命令中设置在`ODPS_CONFIG_FILE_PATH`环境变量中
      ```bash
      cat << EOF >> odps_conf
      project_name=${PROJECT_NAME}
      access_id=${ACCESS_ID}
      access_key=${ACCESS_KEY}
      end_point=http://service.{region}-vpc.maxcompute.aliyun-inc.com/api
      EOF

      ODPS_CONFIG_FILE_PATH=odps_conf \
      torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
      -m tzrec.train_eval \
      --pipeline_config_path ${PIPELINE_CONFIG}
      ```
    - **PAI-DLC/PAI-DSW环境**：
      - 需要设置`ODPS_ENDPOINT`的环境变量，并新建任务时，「角色信息」选择**PAI默认角色**
      ```bash
      ODPS_ENDPOINT=http://service.{region}-vpc.maxcompute.aliyun-inc.com/api \
      torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
      -m tzrec.train_eval \
      --pipeline_config_path ${PIPELINE_CONFIG}
      ```

  - 如果是预付费Quota，参考[独享数据传输服务](https://help.aliyun.com/zh/maxcompute/user-guide/purchase-and-use-exclusive-resource-groups-for-dts)文档购买和授权，可通过`odps_data_quota_name`传入购买的Quota名

  - 如果CPU/GPU利用率都不高，可能是网络传输带宽瓶颈，可以尝试设置`odps_data_compression`为`ZSTD`来增大数据的压缩率，减少数据网络传输带宽

- ParquetDataset: 输入数据为parquet格式

  - input_path: 按如下格式设置
    - `${PATH_TO_DATA_DIR}/*.parquet`
  - 注意: 如果每个parquet文件中的数据量不相等或文件数据小于worker数，ParquetDataset会自动重分配数据，来保证每个worker读取的数据量相等。但仍建议parquet文件数是 `nproc-per-node * nnodes * num_workers`的倍数，并且每个parquet文件的数据量基本相等，减少数据自动重分配的IO开销。

- CsvDataset: 输入数据为csv格式

  - input_path: 按如下格式设置
    - `${PATH_TO_DATA_DIR}/*.csv`
  - 需设置`delimiter`来指名列分隔符，默认为`,`
  - 注意:
    - 训练和评估时需要csv文件数是 `nproc-per-node * nnodes * num_workers`的倍数，并且每个csv文件的数据量相等
    - 暂不支持没有header的csv文件
    - csv格式数据读性能有瓶颈

- KafkaDataset: 输入数据为Kafka消息流（每条消息为序列化的Arrow Batch）

  - input_path: 按如下格式设置

    - `kafka://broker:9092/topic?group.id=consumer_group`

  - 需设置`input_fields`来指定数据的schema，因为Kafka消息中的Arrow Batch不包含schema信息

    ```
    input_fields: {
        input_name: "user_id"
        input_type: INT64
    }
    input_fields: {
        input_name: "item_id"
        input_type: INT64
    }
    input_fields: {
        input_name: "label"
        input_type: FLOAT
    }
    ```

  - 注意:

    - 每条Kafka消息应为一个zstd/lz4压缩的schema-less Arrow RecordBatch

### fg_mode

- FG(Feature Generator) 的运行模式，支持FG_DAG, FG_NONE, FG_BUCKETIZE, FG_NORMAL

  - FG是进入模型推理前的一层特征变换，可以保证离在线特征变换的一致性，特征变换包含Combo/Lookup/Match/Expr等类型，详见[特征](./feature.md)章节。以LookupFeature的一种配置为例，特征变换为从`cate_map`中用`cate_key`查出值后，用`boundaries`进行分箱再进入模型推理

  ```
  feature_configs {
      lookup_feature {
          feature_name: "lookup_feat"
          map: "user:cate_map"
          key: "item:cate_key"
          embedding_dim: 16
          boundaries: [0, 1, 2, 3, 4]
      }
  }
  ```

  - 特征输入的side一共支持五种 \[`user`, `item`, `context`, `feature`, `const`\]，上述`lookup_feat`中的`cate_map`则是属于`user`side
    - `user`: 用户侧特征输入，线上推理时从请求中传入
    - `item`: 物品侧特征输入，线上推理时会从实时缓存在内存中的特征表里获取
    - `context`: 由上下文产生物品侧特征输入，线上推理时从请求中传入，如`recall_name`等
    - `feature`: 来自其他特征FG的输出，如下述`lookup_age_feat`的输入`age_binning`来自于RawFeature `age`的分箱结果
    - `const`: 输入为常量
    ```
    feature_configs {
        raw_feature {
            feature_name: "age_binning"
            expression: "user:age"
            embedding_dim: 16
            boundaries: [18, 25, 30]
        }
    }
    feature_configs {
        lookup_feature {
            feature_name: "lookup_age_feat"
            map: "item:age_map"
            key: "feature:age_binning"
            embedding_dim: 16
            boundaries: [0, 1, 2, 3, 4]
        }
    }
    feature_configs {
        lookup_feature {
            feature_name: "lookup_age_feat"
            map: "item:age_map"
            key: "const:age1"
            embedding_dim: 16
            boundaries: [0, 1, 2, 3, 4]
        }
    }
    ```

#### fg_mode=FG_DAG

- 训练时会在Dataset中执行FG，数据列名与各**特征的FG所依赖字段来源**同名，详见[特征](./feature.md)，Dataset会自动分析所有特征依赖的字段来源来读取数据。
  - 以上文LookupFeature为例，**特征FG所依赖字段来源**为`[cate_map, cate_key]`，Dataset会从输入表中读取名为`cate_map`和`cate_key`的列来做FG得到`lookup_feat`
- 该模式可以帮忙我们快速验证FG的训练效果，调优FG的配置，但由于训练时多了FG的过程，训练速度会受到一定程度的影响

#### fg_mode=FG_NORMAL

- 训练时会在Dataset中执行FG，但不是以DAG方式运行。因此特征的输入中如果有`feature`,`const`side的输入，也需要在输入表中。目前更建议使用`FG_DAG`模式

#### fg_mode=FG_NONE

- 训练时不会在Dataset中执行FG，输入数据为Fg编码后的数据，数据列名与**特征名**(`feature_name`)同名，Dataset会自动分析所有特征的特征名来读取数据
  - 以上文LookupFeature为例，**特征名**为`lookup_feat`，Dataset会从输入表中直接读取编码后的`lookup_feat`列直接进行模型训练和推理
- 该模式训练速度最佳，但需提前对数据提前进行FG编码，目前仅提供MaxCompute方式，步骤如下：
  - 在DLC/DSW/Local环境中生成fg json配置，上传至DataWorks的资源中，如果fg_output_dir中有vocab_file等其他文件，也需要上传至资源中
    ```shell
    cat <<EOF>> odps_conf
    access_id=${ACCESS_ID}
    access_key=${ACCESS_KEY}
    end_point=http://service.${region}-vpc.maxcompute.aliyun-inc.com/api
    EOF

    ODPS_CONFIG_FILE_PATH=odps_conf \
    python -m tzrec.tools.create_fg_json \
        --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
        --fg_output_dir fg_output \
        --reserves ${COLS_YOU_WANT_RESERVE} \
        --fg_resource_name ${FG_RESOURCE_NAME} \
        --odps_project_name ${PROJECT_NAME}
    ```
    - --pipeline_config_path: 模型配置文件。
    - --fg_output_dir: fg json的输出文件夹。
    - --reserves: 需要透传到输出表的列，列名用逗号分隔。一般需要保留Label列，也可以保留request_id，user_id，item_id列，注意：如果模型的feature_config中有user_id，item_id作为特征，feature_name需避免与样本中的user_id，item_id列名冲突。
    - --fg_resource_name: 可选，fg json在MaxCompute中的资源名，默认为fg.json
    - --odps_project_name: 可选，将fg json文件上传到MaxCompute项目名，该参数必须配合参数fg_resource_name和环境变量ODPS_CONFIG_FILE_PATH一起使用
    - --ODPS_CONFIG_FILE_PATH: 该环境变量指向的是odpscmd的配置文件
  - 在[DataWorks](https://workbench.data.aliyun.com/)的独享资源组中安装pyfg，「资源组列表」- 在一个调度资源组的「操作」栏 点「运维助手」-「创建命令」（选手动输入）-「运行命令」
    ```shell
    /home/tops/bin/pip3 install http://tzrec.oss-cn-beijing.aliyuncs.com/third_party/pyfg101-1.0.1-cp37-cp37m-linux_x86_64.whl --index-url=https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.cloud.aliyuncs.com
    ```
  - 在DataWorks中建立`PyODPS 3`节点运行FG，节点调度参数中配置好bizdate参数
    ```
    from pyfg101 import offline_pyfg
    offline_pyfg.run(
      o,
      input_table="YOU_PROJECT.TABLE_NAME",
      output_table="YOU_PROJECT.TABLE_NAME",
      fg_json_file="YOU_FG_FILE_NAME",
      partition_value=args['bizdate']
    )
    ```

| 参数                      | 默认值 | 说明                                                                                                                                                                                                 |
| ------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input_table               | 无     | 输入表                                                                                                                                                                                               |
| output_table              | 无     | 输出表，会自动创建                                                                                                                                                                                   |
| fg_json_file              | 无     | FG 配置文件，json 格式                                                                                                                                                                               |
| partition_value           | 无     | 指定输入表的分区作为 FG 的输入，可支持多分区表，以逗号分隔                                                                                                                                           |
| force_delete_output_table | False  | 是否删除输出表，设置为 True 时会先自动删除输出表, 再运行任务                                                                                                                                         |
| force_update_resource     | False  | 是否更新资源，设置为 True 时会先自动更新资源, 再运行任务                                                                                                                                             |
| set_sql                   | 无     | 任务执行的[flag](https://help.aliyun.com/zh/maxcompute/user-guide/flag-parameters?spm=a2c4g.11186623.0.0.383a20d9CQnpaR#concept-2278178)，如set odps.stage.mapper.split.size=32;，注意需要以分号结尾 |

#### fg_mode=FG_BUCKETIZE

- 训练时在Dataset中执行FG的Bucketize部分，输入数据为Fg编码但未进行Bucketize的数据，Bucketize配置包含`hash_bucket_size`,`boundaries`,`vocab_dict`,`vocab_list`,`num_buckets`

  - 数据列名与**特征名**(`feature_name`)同名，Dataset会自动分析所有特征的特征名来读取数据
  - 以上文LookupFeature为例，**特征名**为`lookup_feat`，Dataset会从输入表中直接读取编码后的`lookup_feat`列直接进行模型训练和推理

- 该模式训练速度介于`FG_DAG`和`FG_NONE`之间，适用于在需要统计Bucketize前的数据分布来设置合适的Bucketize参数的情况，可以避免离线提前跑两次Fg编码

- 注意：在这种模式下，对数据提前进行FG编码时，使用的fg json配置不应该包含Bucketize配置，可以在`create_fg_json`时增加`--remove_bucketizer`参数来去除fg json配置中的Bucketize配置

  ```shell
  python -m tzrec.tools.create_fg_json \
      --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
      --fg_output_dir fg_output \
      --reserves ${COLS_YOU_WANT_RESERVE} \
      --remove_bucketizer

  ```

### fg_threads

- 每个dataloader worker上fg的运行线程数，默认为1，`nproc-per-node * num_workers * fg_threads`建议小于单机CPU核数

### label_fields

- label相关的列名，至少设置一个，可以根据算法需要设置多个，如多目标算法

  ```
    label_fields: "click"
    label_fields: "buy"
  ```

### drop_remainder

- 是否丢弃掉最后一个不足batch_size的batch数据，默认为false

### batch_cost_size

- 用于限制一个batch数据的最大cost，主要适用于变长序列的模型（如DlrmHSTU）同步训练时，不同worker间由于序列长度差异过大，导致workload不一样导致的
- 注：需结合data_config.sample_cost_field使用，sample_cost_field用于指定样本表中的表示样本cost的列名，dataset会根据样本中的cost列，裁切出batch_cost_size限制下的动态Batch。
  - 对于DlrmHSTU模型，cost列一般可以考虑设置为进入模型的token数 (contextual feature num + uih seq length + cand seq length)
- 注：需将data_config.batch_size设置得较大一些，来保证有足够的数据裁切够batch_cost_size

### num_workers

- 每个`proc`上的读数据并发度，`nproc-per-node * num_workers`建议小于单机CPU核数
- 如果`num_workers==0`，数据进程和训练进程将会在一个进程中，便于调试

### shuffle

- 是否训练时打散数据，默认为false

### shuffle_buffer_size

- 最多缓存多少个batch用于打散数据，默认为32

### fg_encoded_multival_sep

- fg_mode=FG_NONE 数据已经被FG编码好 时，数据的多值分割符，默认为chr(3)

### input_fields

```
input_fields: {
    input_name: "input1"
}
input_fields: {
    input_name: "input2"
    input_type: DOUBLE
}
```

- 当使用CsvDataset，如果出现以下情况，需要按如下方式指定`input_fields`，其余Dataset可以自动推理字段类型
  - 情况一：csv文件没有header行时 => 只需设置`input_name`
  - 情况二：csv文件中存在某列的整列为空值时，或遇到`column [xxx] with dtype null is not supported now`报错时 => 需进一步设置`input_type`，目前`input_type`支持设置 INT32|INT64|STRING|FLOAT|DOUBLE

### 更多配置

- [参考文档](../proto.html#tzrec.protos.DataConfig)
