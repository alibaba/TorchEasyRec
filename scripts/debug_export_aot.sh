rm -rf experiments/multi_tower_din_taobao_local/export

ENABLE_AOT=1 torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/multi_tower_din_taobao_local/pipeline.config \
    --export_dir experiments/multi_tower_din_taobao_local/export
