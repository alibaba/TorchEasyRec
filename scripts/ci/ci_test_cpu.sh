#!/usr/bin/env bash

pip install -r requirements-cpu.txt
bash scripts/gen_proto.sh

MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py
