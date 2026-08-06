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

"""Locations and manifest for one generation of the SID bundle.

A bundle is the set of artifacts one resolve run publishes, plus a manifest that
identifies them, locates them and describes the SID space they encode. The
manifest is written last, so its presence is the completion marker.

Artifacts fall into two families with different storage. The serving set --
``sid_to_item_groups``, ``delta_sid_to_item_groups`` and the manifest -- is
always parquet and JSON beneath ``bundle_root``. The map set -- ``item_to_sid_map``
and ``delta_item_to_sid_map`` -- follows the caller's writer type beneath
``map_root``, which may be an ODPS table. On ODPS the artifact name is a
partition key rather than a path segment, because MaxCompute requires every
partition to be ``key=value``.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

MAP_ARTIFACT = "item_to_sid_map"
DELTA_MAP_ARTIFACT = "delta_item_to_sid_map"
GROUPS_ARTIFACT = "sid_to_item_groups"
DELTA_GROUPS_ARTIFACT = "delta_sid_to_item_groups"

MAP_ARTIFACTS = (MAP_ARTIFACT, DELTA_MAP_ARTIFACT)
GROUPS_ARTIFACTS = (GROUPS_ARTIFACT, DELTA_GROUPS_ARTIFACT)


def is_odps_path(path: str) -> bool:
    """Whether a root addresses ODPS rather than a filesystem."""
    return path.startswith("odps://")


def artifact_location(root: str, artifact: str, partition: str) -> str:
    """Return where one artifact's partition lives under ``root``."""
    if is_odps_path(root):
        return f"{root.rstrip('/')}/artifact={artifact}/generation={partition}"
    return os.path.join(root, artifact, partition)


def artifact_read_pattern(root: str, artifact: str, partition: str, suffix: str) -> str:
    """Return the reader path for one artifact's partition.

    A bare directory only globs to itself, so a derived read path needs the
    ``part-*`` pattern the writers emit. An ODPS partition selects itself.
    """
    location = artifact_location(root, artifact, partition)
    if is_odps_path(root):
        return location
    return os.path.join(location, f"part-*.{suffix}")


def manifest_path(bundle_root: str, partition: str) -> str:
    """Return the manifest location for one generation."""
    return os.path.join(bundle_root, "manifest", f"{partition}.json")


@dataclass
class ArtifactEntry:
    """Where one artifact landed and what it holds."""

    type: str
    rows: int
    schema: Dict[str, str]
    path: Optional[str] = None
    table: Optional[str] = None
    partition: Optional[str] = None


@dataclass
class BundleManifest:
    """Identity and shape of one published generation."""

    bundle_uuid: str
    codebook: List[int]
    capacity: int
    max_observed_items_per_sid: int
    artifacts: Dict[str, ArtifactEntry] = field(default_factory=dict)
    since_bundle_uuid: Optional[str] = None
    source_model: Optional[str] = None
    item_id_type: Optional[str] = None

    def to_json(self) -> str:
        """Serialize, dropping keys that carry no value."""
        payload: Dict[str, Any] = asdict(self)
        payload["artifacts"] = {
            name: {k: v for k, v in entry.items() if v is not None}
            for name, entry in payload["artifacts"].items()
        }
        return json.dumps({k: v for k, v in payload.items() if v is not None}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BundleManifest":
        """Parse a manifest, rejecting one this tool could not have written."""
        payload = json.loads(text)
        missing = [
            key
            for key in (
                "bundle_uuid",
                "codebook",
                "capacity",
                "max_observed_items_per_sid",
                "artifacts",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(f"manifest is missing required keys {missing}.")
        entries = {
            name: ArtifactEntry(**entry)
            for name, entry in payload.pop("artifacts").items()
        }
        for name in GROUPS_ARTIFACTS:
            entry = entries.get(name)
            if entry is not None and entry.type != "parquet":
                raise ValueError(
                    f"manifest declares {name} as {entry.type!r}; the serving set "
                    "is always parquet, so this bundle cannot be opened."
                )
        return cls(artifacts=entries, **payload)


def artifact_entry(
    root: str,
    artifact: str,
    partition: str,
    kind: str,
    rows: int,
    schema: Dict[str, str],
) -> ArtifactEntry:
    """Describe where one artifact landed, in the shape the manifest records."""
    location = artifact_location(root, artifact, partition)
    if is_odps_path(root):
        table = root.rstrip("/").rpartition("/tables/")[2]
        return ArtifactEntry(
            type="odps",
            rows=rows,
            schema=schema,
            table=table,
            partition=f"artifact={artifact}/generation={partition}",
        )
    return ArtifactEntry(type=kind, rows=rows, schema=schema, path=location)


def write_manifest(path: str, manifest: BundleManifest) -> None:
    """Write the manifest, creating its directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(manifest.to_json())


def read_manifest(path: str) -> BundleManifest:
    """Read a published manifest.

    Raises:
        FileNotFoundError: If the manifest is absent, which means the run that
            should have written it did not finish.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no manifest at {path!r}; the manifest is written last, so its "
            "absence means that generation did not finish."
        )
    with open(path) as handle:
        return BundleManifest.from_json(handle.read())
