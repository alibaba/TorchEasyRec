#!/usr/bin/env bash
set -eo pipefail

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
REPO_NAME=tzrec-test
DOCKER_TAG=1.5
DOCKER_TAG_SUFFIX=-u3

rm -rf docker/requirements*
cp -r requirements*.txt docker/
cp -r requirements/ docker/requirements
cd docker

for VARIANT in cpu cu126 cu130 cu130-erdma
do
    DEVICE=${VARIANT%-erdma}
    case ${VARIANT} in
        cpu) RDMA= ;;
        *-erdma) RDMA=erdma ;;
        *) RDMA=hpn ;;
    esac
    docker build --network host -t ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${VARIANT}${DOCKER_TAG_SUFFIX} --build-arg DEVICE=${DEVICE} --build-arg RDMA=${RDMA} ${PIP_MIRROR:+--build-arg PIP_MIRROR=${PIP_MIRROR}} .
    docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-${VARIANT}${DOCKER_TAG_SUFFIX}
done

docker images -q ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-cu130${DOCKER_TAG_SUFFIX} | xargs -I {} docker tag {} ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
