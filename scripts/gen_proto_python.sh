#!/usr/bin/env bash
# Usage: bash scripts/gen_proto_python.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 清除 tzrec/protos 下由 protobuf 生成的文件与 Python 缓存
# 若还使用 scripts/gen_proto.sh 且需强制重下 protoc-gen-doc，可手动: rm -rf protoc
find tzrec/protos -type f \( -name '*_pb2.py' -o -name '*_pb2.pyi' \) -print -delete
find tzrec/protos -depth -type d -name '__pycache__' -print -exec rm -rf {} + 2>/dev/null || true

# 仅生成 Python protobuf（*_pb2.py / *_pb2.pyi），不打文档。供本地打 wheel / CI 使用。
# 完整文档生成见 scripts/gen_proto.sh（需 wget OSS 上的 protoc-gen-doc）。
exec python -m grpc_tools.protoc \
  -I . \
  tzrec/protos/*.proto \
  tzrec/protos/models/*.proto \
  --python_out=. \
  --pyi_out=.
