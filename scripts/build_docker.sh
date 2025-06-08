#!/usr/bin/env bash

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
DOCKER_TAG=0.8

rm -rf docker/requirements
cp -r requirements/ docker/requirements
cp -r requirements/ docker/requirements
cd docker

for DEVICE in cu126 cpu
do
    case ${DEVICE} in
        "cu126") LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat ;;
        * )      LD_LIBRARY_PATH= ;;
    esac
    docker build --network host -t ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-${DEVICE} --build-arg DEVICE=${DEVICE} --build-arg LD_LIBRARY_PATH=${LD_LIBRARY_PATH} .
    docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-${DEVICE}
done

docker images -q ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-cu126 | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker images -q ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-cu126 | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:latest
docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker push ${REGISTRY}/tzrec-devel:latest
