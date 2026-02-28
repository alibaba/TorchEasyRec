#!/usr/bin/env bash
set -e

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
REPO_NAME=tzrec-test
DOCKER_TAG=1.1
DOCKER_TAG_SUFFIX=

rm -rf docker/requirements*
cp -r requirements*.txt docker/
cp -r requirements/ docker/requirements
cd docker

for DEVICE in cpu cu126 cu129
do
    for PYTHON_VERSION in 3.10 3.11 3.12
    do
        PY_VERSION=py${PYTHON_VERSION//./}
        docker build --network host -t ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}-${PY_VERSION}${DOCKER_TAG_SUFFIX} --build-arg DEVICE=${DEVICE} --build-arg PYTHON_VERSION=${PYTHON_VERSION} .
        docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}-${PY_VERSION}${DOCKER_TAG_SUFFIX}
    done
    docker images -q ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}-py311${DOCKER_TAG_SUFFIX} | xargs -I {} docker tag {} ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
    docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
done

docker images -q ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-cu129-${DOCKER_TAG_SUFFIX} | xargs -I {} docker tag {} ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker images -q ${REGISTRY}${REPO_NAME}:${DOCKER_TAG}-cu129-${DOCKER_TAG_SUFFIX} | xargs -I {} docker tag {} ${REGISTRY}/${REPO_NAME}:latest
docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker push ${REGISTRY}/${REPO_NAME}:latest
