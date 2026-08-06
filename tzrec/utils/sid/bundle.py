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
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tzrec.utils.path_util import is_odps_path

MAP_ARTIFACT = "item_to_sid_map"
DELTA_MAP_ARTIFACT = "delta_item_to_sid_map"
GROUPS_ARTIFACT = "sid_to_item_groups"
DELTA_GROUPS_ARTIFACT = "delta_sid_to_item_groups"

MAP_ARTIFACTS = (MAP_ARTIFACT, DELTA_MAP_ARTIFACT)
GROUPS_ARTIFACTS = (GROUPS_ARTIFACT, DELTA_GROUPS_ARTIFACT)


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

    def __post_init__(self) -> None:
        if self.bundle_root is not None and is_odps_path(self.bundle_root):
            raise ValueError(
                "bundle_root holds the SID groups and the manifest, which are "
                f"always files; got {self.bundle_root!r}."
            )

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

    def entry(
        self, artifact: str, rows: int, writer_type: Optional[str]
    ) -> ArtifactEntry:
        """Describe where one artifact landed, for the manifest.

        ``save_type`` is read off the address, so it cannot contradict
        ``location``; the serving set can only be parquet because a root that
        would make it otherwise is refused at construction.
        """
        if is_odps_path(self.root_for(artifact)):
            save_type = "odps"
        elif artifact in GROUPS_ARTIFACTS:
            save_type = "parquet"
        else:
            save_type = "csv" if writer_type == "CsvWriter" else "parquet"
        return ArtifactEntry(
            save_type=save_type, rows=rows, location=self.write_path(artifact)
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


class Bundle:
    """The generation a run publishes: where each artifact goes, and its manifest.

    Artifact names live here rather than at the call site, so a caller says
    which artifact it wrote rather than naming it. Only artifacts it is told
    about reach the manifest, which is how a full resolve records that it
    published no delta.
    """

    def __init__(self, layout: BundleLayout) -> None:
        self._layout = layout
        self._rows: Dict[str, int] = {}
        self.uuid = uuid.uuid4().hex

    @property
    def map_path(self) -> str:
        """Where this generation's item map is written."""
        return self._layout.write_path(MAP_ARTIFACT)

    @property
    def delta_map_path(self) -> str:
        """Where this generation's added item rows are written."""
        return self._layout.write_path(DELTA_MAP_ARTIFACT)

    @property
    def groups_path(self) -> str:
        """Where this generation's SID groups are written."""
        return self._layout.write_path(GROUPS_ARTIFACT)

    @property
    def delta_groups_path(self) -> str:
        """Where this generation's touched buckets are written."""
        return self._layout.write_path(DELTA_GROUPS_ARTIFACT)

    @property
    def prior_map_path(self) -> str:
        """Where the appended-onto generation holds its item map."""
        return self._layout.read_path(MAP_ARTIFACT)

    @property
    def prior_groups_path(self) -> str:
        """Where the appended-onto generation holds its SID groups."""
        return self._layout.read_path(GROUPS_ARTIFACT)

    def prior_locations(self) -> List[Tuple[str, str]]:
        """Every published location an append reads, labelled for errors."""
        wanted = [
            (GROUPS_ARTIFACT, self.prior_groups_path),
            ("manifest", self._layout.prior_manifest_path()),
        ]
        if self._layout.map_root is not None:
            wanted.append((MAP_ARTIFACT, self.prior_map_path))
        return wanted

    def record_map(self, rows: int) -> None:
        """Record how many rows the merged item map holds."""
        self._rows[MAP_ARTIFACT] = rows

    def record_delta_map(self, rows: int) -> None:
        """Record how many rows this run added to the item map."""
        self._rows[DELTA_MAP_ARTIFACT] = rows

    def record_groups(self, rows: int) -> None:
        """Record how many buckets the merged SID groups hold."""
        self._rows[GROUPS_ARTIFACT] = rows

    def record_delta_groups(self, rows: int) -> None:
        """Record how many buckets this run touched."""
        self._rows[DELTA_GROUPS_ARTIFACT] = rows

    def publish(
        self,
        codebook: List[int],
        capacity: int,
        observed: int,
        item_id_type: Optional[str],
        writer_type: Optional[str],
    ) -> str:
        """Write the manifest last and return where it landed.

        ``observed`` is carried forward as a running maximum: an append only
        sees the buckets it touched, so the published ceiling is the larger of
        this run's and the one it appended onto.
        """
        prior = (
            read_manifest(self._layout.prior_manifest_path())
            if self._layout.from_partition is not None
            else None
        )
        if prior is not None:
            observed = max(observed, prior.max_observed_items_per_sid)
        manifest = BundleManifest(
            bundle_uuid=self.uuid,
            since_bundle_uuid=prior.bundle_uuid if prior is not None else None,
            codebook=codebook,
            capacity=capacity,
            max_observed_items_per_sid=observed,
            item_id_type=item_id_type,
            artifacts={
                name: self._layout.entry(name, rows, writer_type)
                for name, rows in self._rows.items()
            },
        )
        return self._layout.write_manifest(manifest)
