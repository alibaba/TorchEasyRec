# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Read back sampled delta embeddings from FeatureStore.

The tool scans the local delta parquet outbox for the latest (or specified)
step, samples keys from its shard set, and queries the configured explicit
FeatureDB version through ``DynamicEmbeddingFeatureView.get_online_features``.

Example::

    export ALIBABA_CLOUD_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
    export ALIBABA_CLOUD_ACCESS_KEY_SECRET=YOUR_ACCESS_KEY_SECRET
    export FEATUREDB_USERNAME=YOUR_FEATUREDB_USERNAME
    export FEATUREDB_PASSWORD=YOUR_FEATUREDB_PASSWORD
    python -m tzrec.tools.feature_store.check_feature_store_delta \
        --pipeline_config path/to/pipeline.config \
        --output_dir path/to/delta_embedding_dump \
        --sample_count 10
"""

import argparse
import glob
import inspect
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from tzrec.utils import config_util
from tzrec.utils.feature_store_delta_uploader import (
    FEATURE_STORE_PK_FIELD,
    FEATURE_STORE_SK_FIELD,
    FEATURE_STORE_VALUE_FIELD,
    FeatureStoreUploadSettings,
    feature_store_delta_file_prefix,
)


@dataclass(frozen=True)
class LocalSample:
    """One local parquet record selected for remote readback."""

    embedding_name: str
    key_id: int
    embedding: npt.NDArray[np.float32] = field(repr=False)
    source_path: str


def resolve_output_dir(
    pipeline_config_path: str,
    model_dir: str,
    configured_output_dir: str,
    output_dir_override: Optional[str],
) -> str:
    """Resolve the local delta outbox, including relocated pipeline configs.

    Args:
        pipeline_config_path: Source pipeline config path.
        model_dir: Model directory from the pipeline config.
        configured_output_dir: Explicit delta dump output directory, if any.
        output_dir_override: Command-line output directory override, if any.

    Returns:
        Absolute delta outbox directory path.
    """
    if output_dir_override:
        return os.path.abspath(output_dir_override)
    if configured_output_dir:
        return os.path.abspath(configured_output_dir)

    configured_path = os.path.abspath(os.path.join(model_dir, "delta_embedding_dump"))
    if os.path.isdir(configured_path):
        return configured_path

    colocated_path = os.path.join(
        os.path.dirname(os.path.abspath(pipeline_config_path)),
        "delta_embedding_dump",
    )
    if os.path.isdir(colocated_path):
        return colocated_path
    return configured_path


def resolve_upload_step(
    output_dir: str,
    file_prefix: str,
    world_size: int,
    global_step: Optional[int] = None,
) -> Tuple[int, List[str]]:
    """Find the latest (or specified) step with complete parquet shards.

    Args:
        output_dir: Delta parquet outbox directory.
        file_prefix: Scoped file prefix for parquet filenames.
        world_size: Expected number of rank shards per step.
        global_step: Specific step to inspect, or None for the latest.

    Returns:
        Tuple of (global_step, shard_paths).

    Raises:
        FileNotFoundError: If no complete shard set is found.
    """
    if global_step is not None:
        if global_step <= 0:
            raise ValueError("global_step must be > 0")
        paths = _shard_paths_for_step(output_dir, file_prefix, global_step, world_size)
        if not paths:
            raise FileNotFoundError(
                f"no delta parquet shards found for step {global_step} "
                f"under {output_dir}"
            )
        return global_step, paths

    best_step = -1
    best_paths: List[str] = []
    if world_size == 1:
        pattern = os.path.join(output_dir, f"{file_prefix}_step_*.parquet")
        for path in glob.glob(pattern):
            basename = os.path.basename(path)
            step_str = basename.replace(f"{file_prefix}_step_", "").replace(
                ".parquet", ""
            )
            try:
                step = int(step_str)
            except ValueError:
                continue
            if step > best_step:
                best_step = step
                best_paths = [path]
    else:
        pattern = os.path.join(output_dir, "step_*")
        for step_dir in sorted(glob.glob(pattern)):
            if not os.path.isdir(step_dir):
                continue
            dir_name = os.path.basename(step_dir)
            step_str = dir_name.replace("step_", "")
            try:
                step = int(step_str)
            except ValueError:
                continue
            paths = _shard_paths_for_step(output_dir, file_prefix, step, world_size)
            if paths and step > best_step:
                best_step = step
                best_paths = paths

    if best_step <= 0:
        raise FileNotFoundError(
            f"no complete delta parquet shard set found under {output_dir}"
        )
    return best_step, best_paths


def _shard_paths_for_step(
    output_dir: str, file_prefix: str, global_step: int, world_size: int
) -> List[str]:
    """Resolve expected shard paths for one step, returning them if complete."""
    if world_size == 1:
        path = os.path.join(output_dir, f"{file_prefix}_step_{global_step}.parquet")
        return [path] if os.path.isfile(path) else []
    step_dir = os.path.join(output_dir, f"step_{global_step}")
    paths = [
        os.path.join(
            step_dir,
            f"{file_prefix}_step_{global_step}_rank_{rank}_of_{world_size}.parquet",
        )
        for rank in range(world_size)
    ]
    if all(os.path.isfile(path) for path in paths):
        return paths
    return []


def sample_local_records(
    parquet_paths: Sequence[str],
    sample_count: int,
    embedding_name: Optional[str] = None,
) -> List[LocalSample]:
    """Read a bounded set of unique records from canonical parquet shards."""
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")

    columns = [
        FEATURE_STORE_PK_FIELD,
        FEATURE_STORE_SK_FIELD,
        FEATURE_STORE_VALUE_FIELD,
    ]
    samples: List[LocalSample] = []
    seen: set[Tuple[str, int]] = set()
    for path in parquet_paths:
        parquet_file = pq.ParquetFile(path)
        missing_columns = [
            name for name in columns if name not in parquet_file.schema_arrow.names
        ]
        if missing_columns:
            raise ValueError(
                f"delta parquet {path} is missing columns {missing_columns}"
            )
        for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
            values = batch.to_pydict()
            for name, key_id, vector in zip(
                values[FEATURE_STORE_PK_FIELD],
                values[FEATURE_STORE_SK_FIELD],
                values[FEATURE_STORE_VALUE_FIELD],
            ):
                name = str(name)
                if embedding_name is not None and name != embedding_name:
                    continue
                identity = (name, int(key_id))
                if identity in seen:
                    continue
                if vector is None or len(vector) == 0:
                    raise ValueError(
                        f"delta parquet {path} contains an empty embedding"
                    )
                seen.add(identity)
                samples.append(
                    LocalSample(
                        embedding_name=name,
                        key_id=int(key_id),
                        embedding=np.asarray(vector, dtype=np.float32),
                        source_path=path,
                    )
                )
                if len(samples) >= sample_count:
                    return samples
    if not samples:
        suffix = f" for embedding_name={embedding_name!r}" if embedding_name else ""
        raise ValueError(f"no sampleable delta records were found{suffix}")
    return samples


def _normalize_remote_key(value: Any) -> int:
    """Normalize the SDK's string/bytes/integer SK representation."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def verify_samples(
    view: Any,
    version: str,
    samples: Sequence[LocalSample],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Query sampled keys and classify presence and value equality."""
    grouped: DefaultDict[str, List[LocalSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.embedding_name].append(sample)

    results: List[Dict[str, Any]] = []
    for embedding_name, group in grouped.items():
        remote_rows = view.get_online_features(
            feature_name=embedding_name,
            keys=[sample.key_id for sample in group],
            version=version,
        )
        remote_by_key: Dict[int, npt.NDArray[np.float32]] = {}
        for row in remote_rows:
            raw_key = row.get("sk", row.get(FEATURE_STORE_SK_FIELD))
            if raw_key is None:
                raise ValueError("FeatureStore readback row is missing its SK")
            key_id = _normalize_remote_key(raw_key)
            if key_id in remote_by_key:
                raise ValueError(
                    "FeatureStore returned duplicate rows for "
                    f"{embedding_name}/{key_id}"
                )
            vector = row.get(FEATURE_STORE_VALUE_FIELD)
            if vector is None:
                raise ValueError(
                    f"FeatureStore readback row is missing embedding for "
                    f"{embedding_name}/{key_id}"
                )
            remote_by_key[key_id] = np.asarray(vector, dtype=np.float32)

        for sample in group:
            remote = remote_by_key.get(sample.key_id)
            result: Dict[str, Any] = {
                "embedding_name": sample.embedding_name,
                "key_id": sample.key_id,
                "local_dimension": int(sample.embedding.size),
                "source_path": sample.source_path,
            }
            if remote is None:
                result.update(
                    {
                        "status": "MISSING",
                        "remote_dimension": None,
                        "remote_embedding": None,
                    }
                )
            else:
                same_shape = remote.shape == sample.embedding.shape
                matches = same_shape and bool(
                    np.allclose(remote, sample.embedding, rtol=1e-5, atol=1e-6)
                )
                max_abs_diff = (
                    float(np.max(np.abs(remote - sample.embedding)))
                    if same_shape and remote.size > 0
                    else None
                )
                result.update(
                    {
                        "status": "MATCH" if matches else "PRESENT_DIFFERENT",
                        "remote_dimension": int(remote.size),
                        "remote_embedding": remote.tolist(),
                        "max_abs_diff": max_abs_diff,
                    }
                )
            results.append(result)

    summary = {
        "requested": len(results),
        "found": sum(result["status"] != "MISSING" for result in results),
        "matching": sum(result["status"] == "MATCH" for result in results),
        "present_different": sum(
            result["status"] == "PRESENT_DIFFERENT" for result in results
        ),
        "missing": sum(result["status"] == "MISSING" for result in results),
    }
    return results, summary


def create_feature_store_view(settings: FeatureStoreUploadSettings) -> Any:
    """Create the SDK client and return the existing DynamicEmbedding view."""
    try:
        from feature_store_py import FeatureStoreClient
    except ImportError as exc:
        raise RuntimeError(
            "feature_store_py is required; install requirements/feature_store.txt"
        ) from exc

    try:
        from alibabacloud_credentials.client import Client as CredClient
    except ImportError as exc:
        raise RuntimeError(
            "alibabacloud_credentials is required; "
            "install it via: pip install alibabacloud_credentials"
        ) from exc

    credential = CredClient().get_credential()
    kwargs = {
        "access_key_id": credential.access_key_id,
        "access_key_secret": credential.access_key_secret,
        "region": settings.region or None,
        "endpoint": settings.endpoint or None,
        "security_token": credential.security_token or None,
        "featuredb_username": os.environ.get("FEATUREDB_USERNAME") or None,
        "featuredb_password": os.environ.get("FEATUREDB_PASSWORD") or None,
    }
    try:
        parameters = inspect.signature(FeatureStoreClient).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "test_mode" in parameters:
        kwargs["test_mode"] = True

    client = FeatureStoreClient(**kwargs)
    project = client.get_project(settings.project_name)
    if project is None:
        raise RuntimeError(
            f"FeatureStore project {settings.project_name!r} was not found"
        )
    view = project.get_dynamic_embedding_feature_view(settings.feature_view_name)
    if view is None:
        raise RuntimeError(
            f"DynamicEmbedding FeatureView {settings.feature_view_name!r} was not found"
        )
    actual_fields = (view.pk_field, view.sk_field, view.embedding_field)
    expected_fields = (
        FEATURE_STORE_PK_FIELD,
        FEATURE_STORE_SK_FIELD,
        FEATURE_STORE_VALUE_FIELD,
    )
    if actual_fields != expected_fields:
        raise RuntimeError(
            "DynamicEmbedding FeatureView schema mismatch: "
            f"expected={expected_fields}, actual={actual_fields}"
        )
    return view


def run_check(args: argparse.Namespace) -> int:
    """Run one local-parquet plus remote-readback verification."""
    pipeline_config = config_util.load_pipeline_config(args.pipeline_config)
    train_config = pipeline_config.train_config
    if not train_config.HasField("delta_embedding_dump_config"):
        raise ValueError("pipeline config has no delta_embedding_dump_config")
    dump_config = train_config.delta_embedding_dump_config
    if not dump_config.HasField("feature_store_config"):
        raise ValueError(
            "pipeline config delta_embedding_dump_config has no feature_store_config"
        )
    feature_store_config = dump_config.feature_store_config
    settings = FeatureStoreUploadSettings.from_proto(feature_store_config)
    output_dir = resolve_output_dir(
        args.pipeline_config,
        pipeline_config.model_dir,
        dump_config.output_dir,
        args.output_dir,
    )
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"delta embedding output directory not found: {output_dir}"
        )

    base_prefix = dump_config.file_prefix or "delta_embedding"
    scoped_prefix = feature_store_delta_file_prefix(feature_store_config, base_prefix)
    world_size = args.world_size
    global_step, parquet_paths = resolve_upload_step(
        output_dir, scoped_prefix, world_size, args.global_step
    )
    samples = sample_local_records(
        parquet_paths,
        args.sample_count,
        embedding_name=args.embedding_name,
    )

    view = create_feature_store_view(settings)
    try:
        results, summary = verify_samples(view, settings.version, samples)
    finally:
        close = getattr(view, "close", None)
        if callable(close):
            close(wait=True)

    report: Dict[str, Any] = {
        "target": {
            "project_name": settings.project_name,
            "feature_view_name": settings.feature_view_name,
            "version": settings.version,
        },
        "parquet_source": {
            "global_step": global_step,
            "parquet_paths": [
                os.path.relpath(path, output_dir) for path in parquet_paths
            ],
        },
        "summary": summary,
        "presence_verified": summary["missing"] == 0,
        "value_match_verified": summary["matching"] == summary["requested"],
        "samples": results,
    }
    if summary["present_different"]:
        report["value_match_note"] = (
            "PRESENT_DIFFERENT confirms the key exists but its value differs from "
            "the sampled parquet; a later upload may have updated the same key."
        )
    print(json.dumps(report, indent=2, sort_keys=True))

    if summary["missing"]:
        return 1
    if args.require_value_match and summary["present_different"]:
        return 1
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample a committed delta parquet and read the same keys from "
            "FeatureStore using the configured explicit version."
        )
    )
    parser.add_argument("--pipeline_config", required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override delta_embedding_dump output_dir (useful across mounts).",
    )
    parser.add_argument(
        "--global_step",
        type=int,
        default=None,
        help="Step to inspect; defaults to the latest available shard set.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of rank shards per step (default: 1 for single-rank).",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=10,
        help="Maximum number of local keys to read back (default: 10).",
    )
    parser.add_argument(
        "--embedding_name",
        default=None,
        help="Only sample rows for this canonical embedding name.",
    )
    parser.add_argument(
        "--require_value_match",
        action="store_true",
        help="Exit nonzero when a key exists but differs from the sampled local value.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    try:
        exit_code = run_check(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
