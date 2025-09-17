#!/usr/bin/env bash

REGISTRY=pai-automation-test-registry.cn-shanghai.cr.aliyuncs.com/pai-automation-test
DOCKER_TAG=0.8

rm -rf ppu/requirements*
cp -r requirements*.txt ppu/
cp -r requirements/ ppu/requirements
cd ppu

for DEVICE in ppu
do
    docker build --network host -t ${REGISTRY}/tzrec-test:${DOCKER_TAG}-${DEVICE} --build-arg BASE_IMAGE=reg.docker.alibaba-inc.com/aisw/ppu:v1.6.1-cuda12.8-ubuntu24-py312 .
    docker push ${REGISTRY}/tzrec-test:${DOCKER_TAG}-${DEVICE}
done
