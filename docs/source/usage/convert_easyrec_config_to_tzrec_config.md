# Easyrec Config转换TZRec Config

## 转换命令

必须有easyrec训练使用的config和fg.json才可以转化tzrec的config

```bash
PYTHONPATH=. python tzrec/tools/convert_easyrec_config_to_tzrec_config.py \
  --easyrec_config_path ./easyrec.config \
  --fg_json_path ./fg.json \
  --output_tzrec_config_path ./tzrec.config
```

- --easyrec_config_path: easyrec训练使用的pipeline.config路径
- --fg_json_path: easyrec训练和推理使用的fg.json路径
- --output_tzrec_config_path: 生成tzrec的config路径
