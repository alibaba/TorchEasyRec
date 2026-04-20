#!/usr/bin/env bash
set -e

# Promote tzrec-test:<tag>-{cpu,cu126,cu129} images to tzrec-devel after CI passes.
# Retags the already-pushed test images and pushes them to the devel repo,
# plus refreshes the <tag> (= cu129) and latest aliases.

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
SRC_REPO=tzrec-test
DST_REPO=tzrec-devel
DOCKER_TAG=1.2
DOCKER_TAG_SUFFIX=

for DEVICE in cpu cu126 cu129
do
    docker pull ${REGISTRY}/${SRC_REPO}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
    docker tag ${REGISTRY}/${SRC_REPO}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX} ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
    docker push ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-${DEVICE}${DOCKER_TAG_SUFFIX}
done

docker tag ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-cu129${DOCKER_TAG_SUFFIX} ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker tag ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}-cu129${DOCKER_TAG_SUFFIX} ${REGISTRY}/${DST_REPO}:latest
docker push ${REGISTRY}/${DST_REPO}:${DOCKER_TAG}${DOCKER_TAG_SUFFIX}
docker push ${REGISTRY}/${DST_REPO}:latest
