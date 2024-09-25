# 特征选择

对输入层使用变分dropout计算特征重要性，根据重要性排名进行特征选择。

## 训练

模型中配置如下字段进行训练：

```
model_config {
  feature_groups {
  ...
  }
  variational_dropout{
      regularization_lambda:0.01
      embedding_wise_variational_dropout:false
  }
  ...
}
```

- regularization_lambda: 变分dropout层的正则化系数设置
- embedding_wise_variational_dropout: 变分dropout层维度是否为embedding维度（true：embedding维度；false：feature维度；默认false）

## 查看特征重要性

训练完成后，运行如下脚本查看特征重要性:

```
torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.tools.feature_selection \
    --pipeline_config_path examples/dbmtl_taobao_chen_seq.config \
    --model_dir experiments/taobao/dbmtl_ce \
    --topk 100 \
    --output_dir experiments/taobao/dbmtl_ce/output_dir \
    --clear_variational_dropout true\
    --visualize false
```

- --pipeline_config_path: 训练用的配置文件路径
- --model_dir: 模型训练的目录
- --topk 100: 在训练配置文件钟保存top_k重要的特征
- --output_dir: 新的模型配置文件以及重要性分析保存的目录
- --clear_variational_dropout: 新的模型配置文件中是否删除变分dropout的配置，默认true
- --visualize: 是否画图展示特征重要性，默认false。如果需要，则需要安装matplotlib
