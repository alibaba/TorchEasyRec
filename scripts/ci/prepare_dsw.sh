#!/usr/bin/env bash
set -euo pipefail

lane="${1:?Usage: prepare_dsw.sh cpu|gpu|ppu}"
index_args=()
case "$lane" in
  cpu|gpu) ;;
  ppu)
    index_args=(--extra-index-url https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple)
    ;;
  *)
    echo "Unsupported CI lane: $lane" >&2
    exit 2
    ;;
esac

python -m venv --system-site-packages /job/env
/job/env/bin/python -m pip install -r "requirements-${lane}.txt" "${index_args[@]}"
if [ "$lane" = gpu ]; then
  /job/env/bin/python -m pip install -r requirements/extra.txt
fi

printf '%s\n' '#!/bin/sh' 'exec /job/env/bin/python -m torch.distributed.run "$@"' > /job/env/bin/torchrun
chmod +x /job/env/bin/torchrun
bash -e -o pipefail scripts/gen_proto.sh
bash -e -o pipefail scripts/ci/ci_data.sh
