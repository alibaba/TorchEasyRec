#!/usr/bin/env bash

if [ "${1:-}" = --container ]; then
  shift
  exec "${TZCI_PYTHON:-python3}" tzrec/tests/ci_job.py --lane cpu "$@"
fi

native=false
if [ "${1:-}" = --native ]; then
  shift
  native=true
  set -euo pipefail
  python -m venv --system-site-packages /job/env
  export PATH="/job/env/bin:$PATH"
  printf '%s\n' '#!/bin/sh' 'exec /job/env/bin/python -m torch.distributed.run "$@"' > /job/env/bin/torchrun
  chmod +x /job/env/bin/torchrun
fi

pip install -r requirements-cpu.txt
bash scripts/gen_proto.sh
bash scripts/ci/ci_data.sh

if [ "$native" = true ]; then
  exec /job/env/bin/python tzrec/tests/run.py "$@"
fi

MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py
