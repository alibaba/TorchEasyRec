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

The tool uses the latest locally committed FeatureStore upload by default, samples
keys from its canonical parquet shard set, and queries the configured explicit
FeatureDB version through ``DynamicEmbeddingFeatureView.get_online_features``.

Example::

    python -m tzrec.tools.feature_store.check_feature_store_delta \
        --pipeline_config path/to/pipeline.config \
        --ak YOUR_ACCESS_KEY_ID \
        --sk YOUR_ACCESS_KEY_SECRET \
        --featuredb_username YOUR_FEATUREDB_USERNAME \
        --featuredb_password YOUR_FEATUREDB_PASSWORD \
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
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from tzrec.utils import config_util
from tzrec.utils.feature_store_delta_uploader import (
    DELTA_OPERATION_UPSERT,
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


def _read_json_object(path: str) -> Dict[str, Any]:
    """Read one JSON object from disk."""
    with open(path) as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _target_matches(
    value: Mapping[str, Any], settings: FeatureStoreUploadSettings
) -> bool:
    """Return whether a credential-free marker identifies this remote target."""
    return (
        value.get("project_name") == settings.project_name
        and value.get("feature_view_name") == settings.feature_view_name
        and value.get("version") == settings.version
    )


def load_committed_upload(
    output_dir: str,
    settings: FeatureStoreUploadSettings,
    global_step: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Load the local committed watermark and its success marker.

    Args:
        output_dir: Delta parquet outbox directory.
        settings: Resolved FeatureStore target settings.
        global_step: Specific committed step to inspect, or None for the latest.

    Returns:
        Tuple of state directory, committed watermark, and success marker.

    Raises:
        FileNotFoundError: If no committed marker exists for the configured target.
        ValueError: If markers are ambiguous, inconsistent, or the step is invalid.
    """
    state_root = os.path.join(output_dir, ".feature_store_upload")
    committed_candidates: List[Tuple[str, Dict[str, Any]]] = []
    for path in sorted(glob.glob(os.path.join(state_root, "*", "committed.json"))):
        committed = _read_json_object(path)
        if _target_matches(committed, settings):
            committed_candidates.append((os.path.dirname(path), committed))

    if not committed_candidates:
        raise FileNotFoundError(
            "no committed FeatureStore upload marker was found for "
            f"project={settings.project_name!r}, "
            f"view={settings.feature_view_name!r}, version={settings.version!r} "
            f"under {state_root}"
        )
    if len(committed_candidates) > 1:
        raise ValueError(
            "multiple local FeatureStore state directories match the configured "
            f"target under {state_root}; use a target-specific output_dir"
        )

    state_dir, committed = committed_candidates[0]
    committed_step = int(committed.get("committed_global_step", -1))
    selected_step = committed_step if global_step is None else int(global_step)
    if selected_step <= 0:
        raise ValueError("global_step must be > 0")
    if selected_step > committed_step:
        raise ValueError(
            f"step {selected_step} is newer than the locally committed "
            f"FeatureStore watermark {committed_step}"
        )

    success_path = os.path.join(state_dir, f"step_{selected_step}._FS_SUCCESS.json")
    if not os.path.isfile(success_path):
        raise FileNotFoundError(
            f"committed FeatureStore success marker was not found: {success_path}"
        )
    success = _read_json_object(success_path)
    if not _target_matches(success, settings):
        raise ValueError(f"FeatureStore success marker target mismatch: {success_path}")
    if int(success.get("global_step", -1)) != selected_step:
        raise ValueError(f"FeatureStore success marker step mismatch: {success_path}")
    if int(success.get("success_records", -1)) != int(success.get("total_records", -2)):
        raise ValueError(
            f"FeatureStore success marker is not fully successful: {success_path}"
        )
    return state_dir, committed, success


def committed_parquet_paths(
    output_dir: str,
    file_prefix: str,
    global_step: int,
    expected_shards: int,
) -> List[str]:
    """Resolve the canonical parquet shards for one committed upload step."""
    single_rank_path = os.path.join(
        output_dir, f"{file_prefix}_step_{global_step}.parquet"
    )
    if os.path.isfile(single_rank_path):
        paths = [single_rank_path]
    else:
        paths = sorted(
            glob.glob(
                os.path.join(
                    output_dir,
                    f"step_{global_step}",
                    f"{file_prefix}_step_{global_step}_rank_*_of_*.parquet",
                )
            )
        )
    if not paths:
        raise FileNotFoundError(
            f"no canonical delta parquet shards found for committed step {global_step}"
        )
    if expected_shards > 0 and len(paths) != expected_shards:
        raise ValueError(
            f"committed step {global_step} has {len(paths)} local parquet shards, "
            f"but its success marker records {expected_shards}"
        )
    return paths


def sample_local_records(
    parquet_paths: Sequence[str],
    sample_count: int,
    embedding_name: Optional[str] = None,
) -> List[LocalSample]:
    """Read a bounded set of unique UPSERT records from canonical parquet shards."""
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")

    columns = [
        FEATURE_STORE_PK_FIELD,
        FEATURE_STORE_SK_FIELD,
        FEATURE_STORE_VALUE_FIELD,
        "operation",
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
            for name, key_id, vector, operation in zip(
                values[FEATURE_STORE_PK_FIELD],
                values[FEATURE_STORE_SK_FIELD],
                values[FEATURE_STORE_VALUE_FIELD],
                values["operation"],
            ):
                if operation != DELTA_OPERATION_UPSERT:
                    continue
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
        raise ValueError(f"no sampleable UPSERT delta records were found{suffix}")
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

    kwargs = {
        "access_key_id": settings.access_key_id,
        "access_key_secret": settings.access_key_secret,
        "region": settings.region or None,
        "endpoint": settings.endpoint or None,
        "security_token": settings.security_token or None,
        "featuredb_username": settings.featuredb_username or None,
        "featuredb_password": settings.featuredb_password or None,
    }
    try:
        parameters = inspect.signature(FeatureStoreClient).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "test_mode" in parameters:
        # Match the uploader's FeatureDB routing when supported by the SDK wheel.
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
    """Run one local-marker plus remote-readback verification."""
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
    feature_store_config.access_key_id = args.access_key_id
    feature_store_config.access_key_secret = args.access_key_secret
    feature_store_config.featuredb_username = args.featuredb_username
    feature_store_config.featuredb_password = args.featuredb_password
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

    _, committed, success = load_committed_upload(
        output_dir, settings, args.global_step
    )
    global_step = int(success["global_step"])
    base_prefix = dump_config.file_prefix or "delta_embedding"
    scoped_prefix = feature_store_delta_file_prefix(feature_store_config, base_prefix)
    shard_count = len(success.get("shards", []))
    parquet_paths = committed_parquet_paths(
        output_dir,
        scoped_prefix,
        global_step,
        expected_shards=shard_count,
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
        "local_commit": {
            "global_step": global_step,
            "committed_watermark": int(committed["committed_global_step"]),
            "success_records": int(success["success_records"]),
            "total_records": int(success["total_records"]),
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
            "the sampled committed parquet; a later in-flight step may have updated "
            "the same key in this version."
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
        "--access_key_id",
        "--ak",
        dest="access_key_id",
        required=True,
        help="Alibaba Cloud AccessKey ID.",
    )
    parser.add_argument(
        "--access_key_secret",
        "--sk",
        dest="access_key_secret",
        required=True,
        help="Alibaba Cloud AccessKey Secret.",
    )
    parser.add_argument(
        "--featuredb_username",
        required=True,
        help="FeatureDB username.",
    )
    parser.add_argument(
        "--featuredb_password",
        required=True,
        help="FeatureDB password.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override delta_embedding_dump output_dir (useful across mounts).",
    )
    parser.add_argument(
        "--global_step",
        type=int,
        default=None,
        help="Committed step to inspect; defaults to the latest committed watermark.",
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
