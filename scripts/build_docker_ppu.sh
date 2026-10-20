#!/usr/bin/env bash
set -eo pipefail

# Build the PPU image on top of the vendor PPU base image. Only torch is
# installed from the FlyTiger Eco pypi index; everything else is pinned in
# requirements/ppu.txt.

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
REPO_NAME=tzrec-test
DOCKER_TAG=1.5
DOCKER_TAG_SUFFIX=-u5
BASE_IMAGE=ais-hpn-acr-wulan-registry.cn-wulanchabu.cr.aliyuncs.com/eic/pfed-runtime:2.13.0-ubuntu24.04-cuda13.0-py312-eic-ppu202610070948
PPU_PIP_INDEX=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple

rm -rf docker/requirements*
cp -r requirements*.txt docker/
cp -r requirements/ docker/requirements
cd docker

docker build --network host -f Dockerfile.ppu -t ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-ppu${DOCKER_TAG_SUFFIX} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg PPU_PIP_INDEX=${PPU_PIP_INDEX} \
    ${PIP_MIRROR:+--build-arg PIP_MIRROR=${PIP_MIRROR}} .
docker push ${REGISTRY}/${REPO_NAME}:${DOCKER_TAG}-ppu${DOCKER_TAG_SUFFIX}
