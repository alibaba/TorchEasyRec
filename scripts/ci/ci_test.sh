#!/usr/bin/env bash
export SHELLOPTS

pip install -r requirements-gpu.txt
pip install -r requirements/extra.txt
bash scripts/gen_proto.sh
bash scripts/ci/ci_data.sh
export -n SHELLOPTS

MKL_THREADING_LAYER=GNU PYTHONPATH=. exec python tzrec/tests/run.py "$@"
