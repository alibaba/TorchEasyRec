#!/usr/bin/env bash

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
DOCKER_TAG=0.5

cp requirements.txt docker/
rm -rf docker/requirements
cp -r requirements/ docker/requirements
cd docker

docker build -t ${REGISTRY}/tzrec-devel:latest .
docker images -q ${REGISTRY}/tzrec-devel:latest | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker push ${REGISTRY}/tzrec-devel:latest
docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
