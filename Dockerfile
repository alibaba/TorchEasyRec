# 强制构建/运行平台：linux/amd64（PAI GPU 节点一般是 x86_64）
FROM --platform=linux/amd64 mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:1.0-cu126

WORKDIR /workspace/TorchEasyRec
COPY . /workspace/TorchEasyRec

# 生成 pb2（自定义模型改 proto 后必做）
RUN bash scripts/gen_proto.sh

# 安装你修改后的 tzrec（保证训练/导出时能识别你的 pepnet proto + model class）
RUN python -m pip install -U pip && \
    python -m pip install -e .

# 构建时做一次 smoke test，早发现“少依赖/导入失败”
RUN python -c "import tzrec; print('tzrec import ok')"
