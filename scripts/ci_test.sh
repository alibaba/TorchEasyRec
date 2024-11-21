#!/usr/bin/env bash

pip install -r requirements.txt
bash scripts/gen_proto.sh

# just workaround for torch-tensorrt (dynamic shape) https://github.com/pytorch/TensorRT/pull/3289/files
cp tzrec/acc/_aten_lowering_pass.py /opt/conda/lib/python3.11/site-packages/torch_tensorrt/dynamo/lowering/passes/_aten_lowering_pass.py
cp tzrec/acc/_decompositions.py /opt/conda/lib/python3.11/site-packages/torch_tensorrt/dynamo/lowering/_decompositions.py

MKL_THREADING_LAYER=GNU TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. python tzrec/tests/run.py
