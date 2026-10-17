#!/usr/bin/env bash
set -e

# Promote tzrec-test:<tag>-{cpu,cu126,cu130,cu130-erdma} images to tzrec-devel after CI passes.
# Retags the already-pushed test images and pushes them to the devel repo,
# plus refreshes the <tag> (= cu130) and latest aliases.

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
SRC_REPO=tzrec-test
DST_REPO=tzrec-devel
DOCKER_TAG=1.5
DOCKER_TAG_SUFFIX=-u3

for DEVICE in cpu cu126 cu130 cu130-erdma
do
    docker pull ${REGISTRY}/${SRC_REPO}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
    docker tag ${REGISTRY}/${SRC_REPO}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX} ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-${DEVICE}
    docker push ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-${DEVICE}
done

docker tag ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-cu130 ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}
docker tag ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-cu130 ${REGISTRY}/${DST_REPO}:latest
docker push ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}
docker push ${REGISTRY}/${DST_REPO}:latest
