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

import glob
import os
from itertools import combinations, product
from typing import List, Optional, Sequence, Set, Tuple

from odps.types import PartitionSpec

from tzrec.datasets.odps_dataset import _parse_table_path

_OdpsPathIdentity = Tuple[str, Optional[str], str, Optional[str]]


def _local_path_identities(path: str) -> Set[str]:
    """Return resolved local write locations represented by one path group."""
    identities: Set[str] = set()
    for pattern in path.split(","):
        if not pattern:
            continue
        matched_paths = glob.glob(pattern)
        if not matched_paths:
            identities.add(os.path.realpath(pattern))
            continue
        for matched_path in matched_paths:
            if os.path.isdir(matched_path):
                identities.add(os.path.realpath(matched_path))
                continue
            identities.add(os.path.realpath(os.path.dirname(matched_path) or "."))
            if os.path.islink(matched_path):
                identities.add(os.path.dirname(os.path.realpath(matched_path)))
    return identities


def _odps_path_identities(path: str) -> Set[_OdpsPathIdentity]:
    """Return ODPS table locations represented by one path group."""
    identities: Set[_OdpsPathIdentity] = set()
    for table_path in path.split(","):
        project, table_name, partitions, schema = _parse_table_path(table_path)
        if partitions:
            identities.update(
                (
                    project,
                    schema,
                    table_name,
                    str(PartitionSpec(partition.replace("/", ",")))
                    if partition
                    else None,
                )
                for partition in partitions
            )
        else:
            identities.add((project, schema, table_name, None))
    return identities


def _odps_identities_overlap(left: _OdpsPathIdentity, right: _OdpsPathIdentity) -> bool:
    """Return whether two ODPS identities overlap."""
    left_project, left_schema, left_table, left_partition = left
    right_project, right_schema, right_table, right_partition = right
    return (left_project, left_schema, left_table) == (
        right_project,
        right_schema,
        right_table,
    ) and (
        left_partition is None
        or right_partition is None
        or left_partition == right_partition
    )


def _format_odps_identity(identity: _OdpsPathIdentity) -> str:
    """Format an ODPS identity as a normalized table path."""
    project, schema, table_name, partition = identity
    qualified_table = f"{schema}.{table_name}" if schema else table_name
    table_path = f"odps://{project}/tables/{qualified_table}"
    return f"{table_path}/{partition}" if partition else table_path


def _find_local_path_conflict(paths: Sequence[str]) -> Optional[str]:
    """Return the first local path-group conflict message, if any."""
    if len(paths) < 2:
        return None
    groups = [(path, _local_path_identities(path)) for path in paths]
    for (left_path, left_identities), (right_path, right_identities) in combinations(
        groups, 2
    ):
        overlapping_identities = left_identities & right_identities
        if overlapping_identities:
            location = min(overlapping_identities)
            return (
                f"path conflict: {left_path!r} and {right_path!r} resolve to the "
                f"same local directory {location!r}."
            )
    return None


def _find_odps_path_conflict(paths: Sequence[str]) -> Optional[str]:
    """Return the first ODPS path-group conflict message, if any."""
    if len(paths) < 2:
        return None
    groups = [(path, _odps_path_identities(path)) for path in paths]
    for (left_path, left_identities), (right_path, right_identities) in combinations(
        groups, 2
    ):
        identity_pairs = product(
            sorted(left_identities, key=_format_odps_identity),
            sorted(right_identities, key=_format_odps_identity),
        )
        for left_identity, right_identity in identity_pairs:
            if _odps_identities_overlap(left_identity, right_identity):
                left_location = _format_odps_identity(left_identity)
                right_location = _format_odps_identity(right_identity)
                return (
                    f"path conflict: {left_path!r} and {right_path!r} contain "
                    f"overlapping ODPS locations {left_location!r} and "
                    f"{right_location!r}."
                )
    return None


def check_path_conflict(paths: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Check whether any two path groups can write to the same location.

    Each element of ``paths`` is one logical group. Comma-separated local
    patterns or ODPS tables and ampersand-separated ODPS partitions are
    expanded within that group. Aliases inside one group are ignored. Local
    paths conflict at the resolved output-directory level. ODPS paths conflict
    when their table identity matches and either side is the whole table or
    both sides select the same partition. Writer path groups must identify one
    atomic destination rather than use comma, glob, or multi-partition reader
    syntax.

    Args:
        paths: Reader or writer path groups to compare.

    Returns:
        A conflict flag and an actionable message. The message is ``None``
        when no conflict exists.
    """
    local_paths: List[str] = []
    odps_paths: List[str] = []
    for path in paths:
        if path.startswith("odps://"):
            odps_paths.append(path)
        else:
            local_paths.append(path)

    conflict_message = _find_local_path_conflict(local_paths)
    if conflict_message:
        return True, conflict_message

    conflict_message = _find_odps_path_conflict(odps_paths)
    if conflict_message:
        return True, conflict_message
    return False, None
