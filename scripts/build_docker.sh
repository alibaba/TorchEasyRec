#!/usr/bin/env bash

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
DOCKER_TAG=0.6

cp requirements.txt docker/
rm -rf docker/requirements
cp -r requirements/ docker/requirements
cd docker

for DEVICE in cu121 cpu
do
    docker build --network host -t ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-${DEVICE} --build-arg DEVICE=${DEVICE} .
    docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-${DEVICE}
done

docker images -q ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-cu121 | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker images -q ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-cu121 | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:latest
docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker push ${REGISTRY}/tzrec-devel:latest
