#!/usr/bin/env bash

# kuairand-1k
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/test/kuairand-1k-train-c4096-s100-3c725f3b7de8d38ed281d229e56fab37.parquet -O data/test/kuairand-1k-train-c4096-s100.parquet
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/test/kuairand-1k-eval-c4096-s100-7e841625beda7501876ea8e2ea76523f.parquet -O data/test/kuairand-1k-eval-c4096-s100.parquet
# kuairand-1k-rtp
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/test/kuairand-1k-rtp-train-c4096-s100-ec988712427189b5c478caa14f2b619f.parquet -O data/test/kuairand-1k-rtp-train-c4096-s100.parquet
wget https://tzrec.oss-cn-beijing.aliyuncs.com/data/test/kuairand-1k-rtp-eval-c4096-s100-6ed81eac5a694945e5b4d44afe1ed514.parquet -O data/test/kuairand-1k-rtp-eval-c4096-s100.parquet
