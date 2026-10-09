#!/usr/bin/env bash

pip install -r requirements-ppu.txt --extra-index-url https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple

bash scripts/gen_proto.sh
bash scripts/ci/ci_data.sh

MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py "$@"
