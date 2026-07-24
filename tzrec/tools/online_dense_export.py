# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import datetime
import json
import os
import shutil
from typing import Any, Dict, Optional

from tzrec.acc import utils as acc_utils
from tzrec.main import _create_features, _create_model
from tzrec.models.match_model import MatchModel
from tzrec.models.model import ScriptWrapper
from tzrec.models.tdm import TDM
from tzrec.utils import config_util
from tzrec.utils.export_util import (
    ensure_input_tile_for_distributed_embedding,
    export_dense_model_cpu,
)
from tzrec.utils.logging_util import logger
from tzrec.utils.online_dense_export_util import make_version, resolve_dense_export_root

VERSIONS_DIR = "versions"
CURRENT_JSON = "current.json"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _publish_current(current_path: str, payload: Dict[str, Any]) -> None:
    _atomic_write_json(current_path, payload)


def _read_current_version(current_path: str) -> Optional[str]:
    """Return the version current.json points at, or None if absent/unreadable.

    Best-effort: a missing or corrupt current.json means there is no live
    pointer to spare, so pruning falls back to pure newest-K retention.
    """
    try:
        with open(current_path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("could not read current version from %s: %s", current_path, e)
        return None
    return data.get("version") if isinstance(data, dict) else None


def _max_kept_versions() -> int:
    """Max published dense versions to retain (0 = keep all)."""
    return int(os.environ.get("ONLINE_DENSE_EXPORT_KEEP_VERSIONS", "3"))


def _prune_old_dense_versions(export_root: str, versions_root: str) -> None:
    """Best-effort retention: keep the newest K versions, sweep stale tmp artifacts.

    Serving reads current.json (the newest pointer) and needs the previous
    version for an atomic swap, so K defaults to 3. The version current.json
    points at is always spared even when it sorts outside the newest K: an
    explicit --version or clock rollback after a restart can publish an older
    timestamp, and deleting it would leave the serving pointer referencing a
    missing directory. Stale ``*.tmp.<pid>`` dirs and current.json.tmp.<pid>
    files left by crashed exports are swept so they don't accumulate under the
    serving-facing tree.
    """
    max_versions = _max_kept_versions()
    for base in (versions_root, export_root):
        try:
            entries = os.listdir(base)
        except FileNotFoundError:
            continue
        for name in entries:
            if ".tmp." not in name and not name.endswith(".tmp"):
                continue
            path = os.path.join(base, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.info("removed stale dense export tmp: %s", path)
            except OSError as e:
                logger.warning("failed to remove stale tmp %s: %s", path, e)
    if max_versions <= 0:
        return
    try:
        entries = os.listdir(versions_root)
    except FileNotFoundError:
        return
    current_version = _read_current_version(os.path.join(export_root, CURRENT_JSON))
    version_dirs = sorted(
        os.path.join(versions_root, name)
        for name in entries
        if os.path.isdir(os.path.join(versions_root, name))
    )
    for path in version_dirs[:-max_versions]:
        if os.path.basename(path) == current_version:
            continue
        try:
            shutil.rmtree(path)
            logger.info("removed old dense export version: %s", path)
        except OSError as e:
            logger.warning("failed to remove old version %s: %s", path, e)


def export_online_dense_model(
    pipeline_config_path: str,
    checkpoint_path: str,
    model_dir: str,
    version: Optional[str] = None,
    checkpoint_step: Optional[int] = None,
    data_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Export and publish one online-learning dense model version."""
    if not acc_utils.use_distributed_embedding():
        raise RuntimeError("ONLINE_DENSE_EXPORT requires USE_DISTRIBUTED_EMBEDDING=1.")

    ensure_input_tile_for_distributed_embedding()

    version = version or make_version()
    export_root = resolve_dense_export_root(model_dir)
    versions_root = os.path.join(export_root, VERSIONS_DIR)
    version_dir = os.path.join(versions_root, version)
    tmp_dir = f"{version_dir}.tmp.{os.getpid()}"

    if os.path.exists(version_dir):
        raise RuntimeError(f"dense version already exists: {version_dir}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(versions_root, exist_ok=True)

    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    features = _create_features(
        list(pipeline_config.feature_configs), pipeline_config.data_config
    )
    model = _create_model(
        pipeline_config.model_config,
        features,
        list(pipeline_config.data_config.label_fields),
        sampler_type=None,
    )
    model.set_is_inference(True)
    if isinstance(model, (MatchModel, TDM)):
        # The full export emits per-tower (MatchModel) or per-module (TDM)
        # artifacts; a single monolithic dense export cannot mirror that
        # layout, so a hot swap would load an incompatible artifact.
        raise RuntimeError(
            f"ONLINE_DENSE_EXPORT does not support {type(model).__name__} "
            "models; use the full export (export_model) instead."
        )
    scripted_model = ScriptWrapper(model)

    try:
        export_dense_model_cpu(
            pipeline_config=pipeline_config,
            model=scripted_model,
            checkpoint_path=checkpoint_path,
            save_dir=tmp_dir,
        )

        required_files = ["scripted_model.pt", "dense_meta.json"]
        for file_name in required_files:
            file_path = os.path.join(tmp_dir, file_name)
            if not os.path.exists(file_path):
                raise RuntimeError(f"missing dense export artifact: {file_path}")

        ready_path = os.path.join(tmp_dir, "READY")
        with open(ready_path, "w") as f:
            f.write(_utc_now())
            f.write("\n")

        os.rename(tmp_dir, version_dir)
    except BaseException:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise

    current_payload: Dict[str, Any] = {
        "version": version,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "created_at": _utc_now(),
    }

    # Keep the service-facing pointer beside the immutable dense export versions.
    _publish_current(os.path.join(export_root, CURRENT_JSON), current_payload)
    _prune_old_dense_versions(export_root, versions_root)
    logger.info("published online dense export version %s to %s", version, version_dir)
    return current_payload


def main() -> None:
    """Run the online dense export command-line entrypoint."""
    parser = argparse.ArgumentParser(
        description="Export one online-learning dense model version."
    )
    parser.add_argument("--pipeline_config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--data_timestamp", type=float, default=None)
    args = parser.parse_args()

    export_online_dense_model(
        pipeline_config_path=args.pipeline_config_path,
        checkpoint_path=args.checkpoint_path,
        model_dir=args.model_dir,
        version=args.version,
        checkpoint_step=args.checkpoint_step,
        data_timestamp=args.data_timestamp,
    )


if __name__ == "__main__":
    main()
