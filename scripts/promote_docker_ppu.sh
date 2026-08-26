#!/usr/bin/env bash
set -e

# Promote tzrec-test:<tag>-ppu<suffix> to tzrec-devel:<tag>-ppu after PPU CI passes.
# Separate from promote_docker.sh: the PPU image is built and validated on its own
# cadence and may lag the cpu/cu* images, and it never takes the <tag>/latest aliases.

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
SRC_REPO=tzrec-test
DST_REPO=tzrec-devel
DOCKER_TAG=1.2
DOCKER_TAG_SUFFIX=-u1

docker pull ${REGISTRY}/${SRC_REPO}:${DOCKER_TAG}-ppu${DOCKER_TAG_SUFFIX}
docker tag ${REGISTRY}/${SRC_REPO}:${DOCKER_TAG}-ppu${DOCKER_TAG_SUFFIX} ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-ppu
docker push ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-ppu
