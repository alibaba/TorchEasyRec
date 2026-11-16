#!/usr/bin/env bash
export SHELLOPTS

pip install -r requirements-ppu.txt --extra-index-url https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple

bash scripts/gen_proto.sh
bash scripts/ci/ci_data.sh
export -n SHELLOPTS

MKL_THREADING_LAYER=GNU PYTHONPATH=. exec python tzrec/tests/run.py "$@"
