#!/usr/bin/env bash

# make proto
bash scripts/gen_proto.sh
sed -i 's#<p>#<pre>#g;s#</p>#</pre>#g' docs/source/proto.html

# copy intro
sed 's#(docs/source/#(#g;s#(docs/images/#(../images/#g;s#"docs/images/#"../images/#g' README.md > docs/source/intro.md
sed -i '1i\# 简介' docs/source/intro.md

# replace wheel and docker version
ALL_WHEEL_VERSIONS=$(pip index versions tzrec -f http://tzrec.oss-accelerate.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-accelerate.aliyuncs.com)
LATEST_WHEEL_VERSION=$(echo "${ALL_WHEEL_VERSIONS}" | awk 'NR==1{print $2}' | sed 's/[()]//g')
# PPU runs on the 1.1.x line; pick the latest 1.1.x from the available versions list
LATEST_PPU_WHEEL_VERSION=$(echo "${ALL_WHEEL_VERSIONS}" | grep -i 'available versions' | tr ',' '\n' | tr -d ' ' | grep '^1\.1\.' | head -1)
LATEST_DOCKER_VERSION=$(grep DOCKER_TAG= scripts/build_docker.sh | awk -F= '{print $2}')
for f in docs/source/quick_start/*.md; do
    cp $f $f.bak
    sed -i 's/${TZREC_NIGHTLY_VERSION}/'"${LATEST_WHEEL_VERSION}"'/g;s/${TZREC_PPU_NIGHTLY_VERSION}/'"${LATEST_PPU_WHEEL_VERSION}"'/g;s/${TZREC_DOCKER_VERSION}/'"${LATEST_DOCKER_VERSION}"'/g' $f
done
