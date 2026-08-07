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

Everything lands at ``{root}/{generation}/{artifact}``, so one generation is one
self-contained directory. On ODPS the two levels are partition keys rather than
path segments -- ``generation=.../artifact=...`` -- because MaxCompute requires
every partition to be ``key=value``; the nesting order is the same either way.

Artifacts fall into two families. The serving set -- ``sid_to_items``,
``delta_sid_to_items`` and the manifest -- is always parquet and JSON beneath
``output_path``, which therefore can never be an ODPS table. The item-to-SID set
-- ``item_to_sid`` and ``delta_item_to_sid`` -- follows the caller's writer type
beneath ``item_to_sid_root``, which may be an ODPS table and otherwise defaults
to ``output_path``.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from tzrec.utils.logging_util import logger
from tzrec.utils.path_util import is_odps_path

ITEM_TO_SID_ARTIFACT = "item_to_sid"
DELTA_ITEM_TO_SID_ARTIFACT = "delta_item_to_sid"
SID_TO_ITEMS_ARTIFACT = "sid_to_items"
DELTA_SID_TO_ITEMS_ARTIFACT = "delta_sid_to_items"

ITEM_TO_SID_ARTIFACTS = (ITEM_TO_SID_ARTIFACT, DELTA_ITEM_TO_SID_ARTIFACT)
SID_TO_ITEMS_ARTIFACTS = (SID_TO_ITEMS_ARTIFACT, DELTA_SID_TO_ITEMS_ARTIFACT)


def artifact_location(root: str, generation: str, artifact: str) -> str:
    """Return where one artifact of one generation lives under ``root``."""
    if is_odps_path(root):
        return f"{root.rstrip('/')}/generation={generation}/artifact={artifact}"
    return os.path.join(root, generation, artifact)


def artifact_read_pattern(
    root: str, generation: str, artifact: str, suffix: str
) -> str:
    """Return the reader path for one artifact of one generation.

    A bare directory only globs to itself, so a derived read path needs the
    ``part-*`` pattern the writers emit. An ODPS partition selects itself.
    """
    location = artifact_location(root, generation, artifact)
    if is_odps_path(root):
        return location
    return os.path.join(location, f"part-*.{suffix}")


def manifest_path(output_path: str, generation: str) -> str:
    """Return the manifest location inside one generation."""
    return os.path.join(output_path, generation, "manifest.json")


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
        for name in SID_TO_ITEMS_ARTIFACTS:
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
        ValueError: If the manifest is present but cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no manifest at {path!r}; the manifest is written last, so its "
            "absence means that generation did not finish."
        )
    with open(path) as handle:
        text = handle.read()
    try:
        return BundleManifest.from_json(text)
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError(f"the manifest at {path!r} is not readable: {err}") from err


@dataclass(frozen=True)
class BundleLayout:
    """Every location one run reads and writes, resolved once.

    Binding the two roots and the generations here is what keeps the family rule
    -- the serving set under ``output_path``, the item-to-SID set under
    ``item_to_sid_root`` -- in this module instead of at each call site.
    ``item_to_sid_root`` arrives already resolved, so its default is not this
    class's concern.
    """

    output_path: Optional[str]
    item_to_sid_root: Optional[str]
    generation: Optional[str]
    from_generation: Optional[str]

    def __post_init__(self) -> None:
        if self.output_path is not None and is_odps_path(self.output_path):
            raise ValueError(
                "output_path holds sid_to_items and the manifest, which are "
                f"always files; got {self.output_path!r}."
            )

    def root_for(self, artifact: str) -> str:
        """Return the root that owns one artifact's family."""
        root = (
            self.item_to_sid_root
            if artifact in ITEM_TO_SID_ARTIFACTS
            else self.output_path
        )
        if root is None:
            raise RuntimeError(f"no root is configured for {artifact}.")
        return root

    def write_path(self, artifact: str) -> str:
        """Return where this generation writes one artifact."""
        if self.generation is None:
            raise RuntimeError("no generation is configured to write.")
        return artifact_location(self.root_for(artifact), self.generation, artifact)

    def read_path(self, artifact: str, suffix: str) -> str:
        """Return where the appended-onto generation holds one artifact.

        ``suffix`` comes from what that generation recorded it wrote, not from
        this run's flags, so a chain does not depend on the reader type being
        repeated.
        """
        if self.from_generation is None:
            raise RuntimeError("no from_generation is configured to read.")
        return artifact_read_pattern(
            self.root_for(artifact), self.from_generation, artifact, suffix
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
        elif artifact in SID_TO_ITEMS_ARTIFACTS:
            save_type = "parquet"
        else:
            save_type = "csv" if writer_type == "CsvWriter" else "parquet"
        return ArtifactEntry(
            save_type=save_type, rows=rows, location=self.write_path(artifact)
        )

    def write_manifest(self, manifest: "BundleManifest") -> str:
        """Write this generation's manifest and return where it landed."""
        if self.output_path is None or self.generation is None:
            raise RuntimeError("no manifest location is configured.")
        path = manifest_path(self.output_path, self.generation)
        write_manifest(path, manifest)
        return path

    def prior_manifest_path(self) -> str:
        """Return the manifest of the generation being appended onto."""
        if self.output_path is None or self.from_generation is None:
            raise RuntimeError("no prior manifest location is configured.")
        return manifest_path(self.output_path, self.from_generation)


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

    @cached_property
    def _prior(self) -> Optional[BundleManifest]:
        """The manifest of the generation being appended onto, read once."""
        if self._layout.from_generation is None:
            return None
        return read_manifest(self._layout.prior_manifest_path())

    def _prior_entry(self, artifact: str) -> ArtifactEntry:
        """Return what the appended-onto generation recorded for one artifact."""
        prior = self._prior
        if prior is None:
            raise RuntimeError("no prior generation is configured to read.")
        entry = prior.artifacts.get(artifact)
        if entry is None:
            raise ValueError(
                f"generation {self._layout.from_generation!r} records no "
                f"{artifact} in its manifest, so it cannot be appended onto."
            )
        return entry

    @property
    def item_to_sid_path(self) -> str:
        """Where this generation's item_to_sid rows are written."""
        return self._layout.write_path(ITEM_TO_SID_ARTIFACT)

    @property
    def delta_item_to_sid_path(self) -> str:
        """Where this generation's added item rows are written."""
        return self._layout.write_path(DELTA_ITEM_TO_SID_ARTIFACT)

    @property
    def sid_to_items_path(self) -> str:
        """Where this generation's sid_to_items rows are written."""
        return self._layout.write_path(SID_TO_ITEMS_ARTIFACT)

    @property
    def delta_sid_to_items_path(self) -> str:
        """Where this generation's touched buckets are written."""
        return self._layout.write_path(DELTA_SID_TO_ITEMS_ARTIFACT)

    @property
    def prior_item_to_sid_path(self) -> str:
        """Where the appended-onto generation holds its item_to_sid rows."""
        save_type = self._prior_entry(ITEM_TO_SID_ARTIFACT).save_type
        return self._layout.read_path(ITEM_TO_SID_ARTIFACT, save_type)

    @property
    def prior_sid_to_items_path(self) -> str:
        """Where the appended-onto generation holds its sid_to_items rows."""
        return self._layout.read_path(SID_TO_ITEMS_ARTIFACT, "parquet")

    def prior_locations(self) -> List[Tuple[str, str]]:
        """Every published artifact an append reads, labelled for errors.

        The manifest is not among them: resolving these paths is what reads it,
        so its absence has already raised by the time this returns.
        """
        return [
            (SID_TO_ITEMS_ARTIFACT, self.prior_sid_to_items_path),
            (ITEM_TO_SID_ARTIFACT, self.prior_item_to_sid_path),
        ]

    def check_prior_compatible(self, codebook: List[int], capacity: int) -> None:
        """Reject an append whose SID space differs from the published corpus.

        The codebook fixes how a SID packs into a bucket key and how its layers
        shift into one vocabulary, so appending under a different one re-keys
        every published SID and rewrites its ``offset_codebook``.

        Raises:
            ValueError: If the codebook differs from the published generation's.
        """
        prior = self._prior
        if prior is None:
            return
        if codebook != prior.codebook:
            raise ValueError(
                f"--codebook {codebook} differs from the {prior.codebook} "
                f"published at generation {self._layout.from_generation!r}; "
                "appending would re-key every published SID and rewrite its "
                "offset_codebook. Append with the codebook the corpus was "
                "built with."
            )
        if capacity != prior.capacity:
            logger.warning(
                "capacity is %d but generation %s was published with %d; already "
                "published buckets keep the occupancy they have",
                capacity,
                self._layout.from_generation,
                prior.capacity,
            )

    def record_item_to_sid(self, rows: int) -> None:
        """Record how many rows the merged item_to_sid holds."""
        self._rows[ITEM_TO_SID_ARTIFACT] = rows

    def record_delta_item_to_sid(self, rows: int) -> None:
        """Record how many rows this run added to item_to_sid."""
        self._rows[DELTA_ITEM_TO_SID_ARTIFACT] = rows

    def record_sid_to_items(self, rows: int) -> None:
        """Record how many buckets the merged sid_to_items holds."""
        self._rows[SID_TO_ITEMS_ARTIFACT] = rows

    def record_delta_sid_to_items(self, rows: int) -> None:
        """Record how many buckets this run touched."""
        self._rows[DELTA_SID_TO_ITEMS_ARTIFACT] = rows

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
        prior = self._prior
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
