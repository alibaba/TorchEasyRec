#!/usr/bin/env bash

pip install -r requirements/runtime.txt
pip install -r requirements/test.txt
pip install -r requirements/gpu.txt
bash scripts/gen_proto.sh

MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py
