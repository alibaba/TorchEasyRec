#!/usr/bin/env bash
set -eo pipefail

# Build the PPU image on top of the vendor PPU base image. torch/triton/faiss are the
# PPU builds from the FlyTiger Eco pypi index; everything else comes from requirements/.

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
REPO_NAME=tzrec-test
DOCKER_TAG=1.2
DOCKER_TAG_SUFFIX=-u1
BASE_IMAGE=pkg.flytiger-eco.com/docker_release/ppu:v2.1.1-cuda13.0-ubuntu24-py312
PPU_PIP_INDEX=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple

rm -rf docker/requirements*
cp -r requirements*.txt docker/
cp -r requirements/ docker/requirements
cd docker

docker build --network host -f Dockerfile.ppu -t ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-ppu${DOCKER_TAG_SUFFIX} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg PPU_PIP_INDEX=${PPU_PIP_INDEX} \
    ${PIP_MIRROR:+--build-arg PIP_MIRROR=${PIP_MIRROR}} .
docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-ppu${DOCKER_TAG_SUFFIX}
