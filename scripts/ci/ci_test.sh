#!/usr/bin/env bash

if [ -n "${TZREC_CI_ENV:-}" ]; then
  set -euo pipefail
  python -m venv --system-site-packages "$TZREC_CI_ENV"
  export PATH="$TZREC_CI_ENV/bin:$PATH"
  printf '#!/bin/sh\nexec "%s/bin/python" -m torch.distributed.run "$@"\n' "$TZREC_CI_ENV" > "$TZREC_CI_ENV/bin/torchrun"
  chmod +x "$TZREC_CI_ENV/bin/torchrun"
fi

pip install -r requirements-gpu.txt
pip install -r requirements/extra.txt
bash scripts/gen_proto.sh
bash scripts/ci/ci_data.sh

if [ "${1:-}" = --parallel ]; then
  shift
  exec env MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/dispatch.py local "$@"
fi

MKL_THREADING_LAYER=GNU PYTHONPATH=. python tzrec/tests/run.py "$@"
