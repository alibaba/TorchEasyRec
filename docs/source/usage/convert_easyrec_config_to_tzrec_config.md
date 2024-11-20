# EasyRec迁移TorchEasyRec

推荐模型一般特征和模型配置较为复杂，TorchEasyRec提供了配置转换工具，可以便捷地将EasyRec的配置文件转换为TorchEasyRec文件。

## 转换命令

torcheasyrec的pipeline.config包含了feature generate的配置，因此需要有easyrec训练使用的pipeline.config和fg.json两部分才可以转换为torcheasyrec的pipeline.config

```bash
PYTHONPATH=. python tzrec/tools/convert_easyrec_config_to_tzrec_config.py \
  --easyrec_config_path ./easyrec.config \
  --fg_json_path ./fg.json \
  --output_tzrec_config_path ./tzrec.config
```

- --easyrec_config_path: easyrec训练使用的pipeline.config路径
- --fg_json_path: easyrec训练和推理使用的fg.json路径
- --output_tzrec_config_path: 生成tzrec的config路径

如果使用自定义的EasyRec安装包，应使用如下转换命令

```bash
EASYREC_URL=http://xxx.whl
PYTHONPATH=. python tzrec/tools/convert_easyrec_config_to_tzrec_config.py \
  --easyrec_config_path ./easyrec.config \
  --fg_json_path ./fg.json \
  --output_tzrec_config_path ./tzrec.config
```

- EASYREC_URL: 是http开头的EasyRec tar包或者whl的url地址
