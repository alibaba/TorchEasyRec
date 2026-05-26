#!/usr/bin/env bash

# kuairand-1k
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-train-c4096-s100-3c725f3b7de8d38ed281d229e56fab37.parquet -O data/test/kuairand-1k-train-c4096-s100.parquet
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-eval-c4096-s100-7e841625beda7501876ea8e2ea76523f.parquet -O data/test/kuairand-1k-eval-c4096-s100.parquet
# kuairand-1k-rtp
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-rtp-train-c4096-s100-ec988712427189b5c478caa14f2b619f.parquet -O data/test/kuairand-1k-rtp-train-c4096-s100.parquet
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-rtp-eval-c4096-s100-6ed81eac5a694945e5b4d44afe1ed514.parquet -O data/test/kuairand-1k-rtp-eval-c4096-s100.parquet
# kuairand-mot-1k (per-channel-split UIH for ULTRA-HSTU MoT integration test)
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-mot-1k-train-c4096-s100-e28061f3c88f543b9e18f40be6ddb94d.parquet -O data/test/kuairand-mot-1k-train-c4096-s100.parquet
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-mot-1k-eval-c4096-s100-f185f38e3b4a49cb791d2e4302087a1f.parquet -O data/test/kuairand-mot-1k-eval-c4096-s100.parquet
# kuairand-1k-match (HSTUMatch integration test fixtures)
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-match-train-c4096-s100-aa77964ed7f50ca30645f8dd08dbf10d.parquet -O data/test/kuairand-1k-match-train-c4096-s100.parquet
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-match-eval-c4096-s100-8678a3ac699fb08f0602f4c06cef2edf.parquet -O data/test/kuairand-1k-match-eval-c4096-s100.parquet
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-match-item-gl-3d459148303acd9f838da108efcc40e5.txt -O data/test/kuairand-1k-match-item-gl.txt
wget https://tzrec.oss-accelerate.aliyuncs.com/data/test/kuairand-1k-match-item-c1-8dcadabdc3e9049ed9c2250565b4b134.parquet -O data/test/kuairand-1k-match-item-c1.parquet
