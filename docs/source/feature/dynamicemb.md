# Dynamic Embedding

DynamicEmbedding 是特征零Hash冲突Id化的一种方式，它相比设置`hash_bucket_size`的方式能避免hash冲突，相比设置`vocab_dict`和`vocab_list`的方式能更灵活动态地进行id的准入和驱逐。DynamicEmbedding 常用于user id，item id，combo feature等超大id枚举数的特征配置中。DynamicEmbedding使用[HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) 作为Hash表的后端，相比 ZCH 能外接PS，支撑十亿百亿甚至更多的Id枚举数，Id准入和淘汰无需攒Batch，可以更加及时。

注：目前使用DynamicEmbedding还处于实验阶段，配置和接口都可能调整，暂只支持训练和评估，暂不包含在官方提供的镜像环境中，使用前需要额外安装如下whl包

```bash
pip install https://tzrec.oss-accelerate.aliyuncs.com/third_party/dynamicemb/dynamicemb-0.0.1%2B20260126.60bd31e-cp311-cp311-linux_x86_64.whl
```

以id_feature的配置为例，DynamicEmbedding 只需在id_feature新增一个dynamicemb的配置字段

```
feature_configs {
    id_feature {
        feature_name: "cate"
        expression: "item:cate"
        embedding_dim: 32
        dynamicemb {
            max_capacity: 100000
            initializer_args {
                mode: "NORMAL"
                std_dev: 0.05
            }
            score_strategy: "STEP"
            frequency_admission_strategy {
                threshold: 5
            }
        }
    }
}
```

- **max_capacity**: 最大的id数，Id数超过后会根据Id的驱逐策略进行淘汰

- **score_strategy**: Id驱逐策略，默认为 STEP，目前支持 TIMESTAMP | STEP | LFU

  - TIMESTAMP: 每个Id根据最近更新的时间戳，驱逐时间戳最小的Id
  - STEP: 每个Id根据最近更新的迭代步数，驱逐步数最早的Id
  - LFU: 每个Id根据出现的频次，驱逐频次小的Id

- **initializer_args**: 参数初始化设置，默认是 UNIFORM

  - mode: 初始化方式，可选 NORMAL | TRUNCATED_NORMAL | UNIFORM | CONSTANT
  - mean: NORMAL | TRUNCATED_NORMAL 初始化方式的均值，默认为 0
  - std_dev: NORMAL | TRUNCATED_NORMAL 初始化方式的标准差，默认为 sqrt(1 / embedding_dim)
  - lower: UNIFORM 初始化方式的均值的下界，默认为 - sqrt(1 / max_capacity)
  - upper: UNIFORM 初始化方式的均值的上界，默认为 sqrt(1 / max_capacity)
  - value: CONSTANT 初始化方式的值，默认为0

- **eval_initializer_args**: （可选）评估时的初始化方式，默认是 CONSTANT，value=0

- **init_capacity_per_rank**: （可选）初始的每个Rank上的id数，默认等于max_capacity

- **admission_strategy**: (可选) 特征准入策略，默认不开启，目前只支持frequency_admission_strategy

  - **frequency_admission_strategy**: 基于频次的特征准入策略
    - threshold: 准入频次
    - initializer_args: （可选）未准入的ID的初始化方式，默认是 CONSTANT，value=0
    - counter_capacity: （可选）频次统计Counter的最大id数，默认与max_capacity相等
    - counter_bucket_capacity: （可选）频次统计Counter的每个bucket的最大id数，默认为1024

- **init_table**: （可选）初始化表的路径，支持Odps/Parquet/Csv格式，表的第一列为id值，第二列为embedding值。

  - 注意: init_table 参数不能再训练任务中直接生效，需要前置一个使用init_table构建初始化ckpt的任务，具体步骤如下

  - 使用init_table构建初始化ckpt

    ```bash
    python -m tzrec.tools.dynamicemb.create_dynamicemb_init_ckpt \
    --pipeline_config_path {PATH_TO_CONFIG_WITH_DYNAMICEMB} \
    --world_size $WORLD_SIZE*$NPROC_PER_NODE \
    --save_dir {INIT_CKPT_PATH}
    ```

    - --pipeline_config_path: 训练配置文件
    - --world_size: 训练进程数，一般情况下为训练的torchrun命令中的 nnodes\*nproc-per-node
    - --save_dir: 初始化模型保存目录
    - --reader_worker_num:（可选）读worker数目，默认为自动根据cpu核数设置
    - --separator: (可选) embedding数据类型如果为string的情况下的分隔符

  - 使用初始化的ckpt训练模型

    ```bash
    torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK \
    -m tzrec.train_eval \
    --pipeline_config_path {PATH_TO_CONFIG_WITH_DYNAMICEMB} \
    --fine_tune_checkpoint {INIT_CKPT_PATH}/model.ckpt-0
    ```
