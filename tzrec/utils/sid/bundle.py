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
    """Where one artifact landed and what it holds.

    ``location`` is the address in the form this repository's readers take, so
    an ODPS artifact carries its full ``odps://`` URL rather than a split table
    and partition; ``save_type`` says how to open it.
    """

    save_type: str
    rows: int
    location: str


@dataclass
class BundleManifest:
    """Identity and shape of one published generation."""

    bundle_uuid: str
    codebook: List[int]
    capacity: int
    max_observed_items_per_sid: int
    artifacts: Dict[str, ArtifactEntry] = field(default_factory=dict)
    since_bundle_uuid: Optional[str] = None
    item_id_type: Optional[str] = None

    def to_json(self) -> str:
        """Serialize, dropping keys that carry no value."""
        payload: Dict[str, Any] = asdict(self)
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
            if entry is not None and entry.save_type != "parquet":
                raise ValueError(
                    f"manifest declares {name} as {entry.save_type!r}; the serving set "
                    "is always parquet, so this bundle cannot be opened."
                )
        return cls(artifacts=entries, **payload)


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


@dataclass(frozen=True)
class BundleLayout:
    """Every location one run reads and writes, resolved once.

    Binding the two roots and the partitions here is what keeps the family rule
    -- the serving set under ``bundle_root``, the map family under ``map_root``
    -- in this module instead of at each call site.
    """

    map_root: Optional[str]
    bundle_root: Optional[str]
    partition: Optional[str]
    from_partition: Optional[str]
    reader_type: Optional[str] = None

    def root_for(self, artifact: str) -> str:
        """Return the root that owns one artifact's family."""
        root = self.bundle_root if artifact in GROUPS_ARTIFACTS else self.map_root
        if root is None:
            raise RuntimeError(f"no root is configured for {artifact}.")
        return root

    def write_path(self, artifact: str) -> str:
        """Return where this generation writes one artifact."""
        if self.partition is None:
            raise RuntimeError("no partition is configured to write.")
        return artifact_location(self.root_for(artifact), artifact, self.partition)

    def read_path(self, artifact: str) -> str:
        """Return where the appended-onto generation holds one artifact."""
        if self.from_partition is None:
            raise RuntimeError("no from_partition is configured to read.")
        is_csv_map = artifact in MAP_ARTIFACTS and self.reader_type == "CsvReader"
        suffix = "csv" if is_csv_map else "parquet"
        return artifact_read_pattern(
            self.root_for(artifact), artifact, self.from_partition, suffix
        )

    def entry(self, artifact: str, file_save_type: str, rows: int) -> ArtifactEntry:
        """Describe where one artifact landed, for the manifest."""
        odps = is_odps_path(self.root_for(artifact))
        return ArtifactEntry(
            save_type="odps" if odps else file_save_type,
            rows=rows,
            location=self.write_path(artifact),
        )

    def write_manifest(self, manifest: "BundleManifest") -> str:
        """Write this generation's manifest and return where it landed."""
        if self.bundle_root is None or self.partition is None:
            raise RuntimeError("no manifest location is configured.")
        path = manifest_path(self.bundle_root, self.partition)
        write_manifest(path, manifest)
        return path

    def prior_manifest_path(self) -> str:
        """Return the manifest of the generation being appended onto."""
        if self.bundle_root is None or self.from_partition is None:
            raise RuntimeError("no prior manifest location is configured.")
        return manifest_path(self.bundle_root, self.from_partition)
