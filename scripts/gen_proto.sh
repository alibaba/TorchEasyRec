#!/usr/bin/env bash

VERSION="1.5.1"
ROOT_URL="http://tzrec.oss-cn-beijing.aliyuncs.com/third_party/"
if [ "$(uname)" == "Darwin" ]; then
    OS='darwin'
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    OS='linux'
fi
if [ "$(uname -m)" == "arm64" ] || [ "$(uname -m)" == "aarch64" ]; then
    ARCH='aarch_64'
elif [ "$(uname -m)" == "x86_64" ]; then
    ARCH='amd64'
fi

PROTO_DIR=protoc/${OS}_${ARCH}
if [ ! -d protoc ]; then
    mkdir -p ${PROTO_DIR}
    wget ${ROOT_URL}protoc-gen-doc_${VERSION}_${OS}_${ARCH}.tar.gz -O ${PROTO_DIR}/protoc-gen-doc_${VERSION}.tar.gz
    tar xf ${PROTO_DIR}/protoc-gen-doc_${VERSION}.tar.gz -C ${PROTO_DIR}
fi

python -m grpc_tools.protoc -I . tzrec/protos/*.proto tzrec/protos/models/*.proto  --python_out=. --pyi_out=. --doc_out=html,proto.html:docs/source --plugin=protoc-gen-doc=./${PROTO_DIR}/protoc-gen-doc
