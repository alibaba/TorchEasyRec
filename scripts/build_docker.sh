#!/usr/bin/env bash

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
REPO_NAME=tzrec-devel
DOCKER_TAG=1.0
DOCKER_TAG_SUFFIX=

rm -rf docker/requirements*
cp -r requirements*.txt docker/
cp -r requirements/ docker/requirements
cd docker

for DEVICE in cpu cu126
do
    docker build --network host -t ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX} --build-arg DEVICE=${DEVICE} .
    docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
done

docker images -q ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-cu126${DOCKER_TAG_SUFFIX} | xargs -I {} docker tag {} ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker images -q ${REGISTRY}${REPO_NAME}:${DOCKER_TAG}-cu126${DOCKER_TAG_SUFFIX} | xargs -I {} docker tag {} ${REGISTRY}/${REPO_NAME}:latest
docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker push ${REGISTRY}/${REPO_NAME}:latest
