#!/usr/bin/env bash

pip install -r requirements/runtime.txt
pip install -r requirements/test.txt
pip install -r requirements/gpu.txt
bash scripts/gen_proto.sh

MKL_THREADING_LAYER=GNU TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. python tzrec/tests/run.py
