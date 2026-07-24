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

"""Shared sparse-embedding identity used by export, delta dump and serving."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

SPARSE_EC_ROLE = "ec"
SPARSE_EBC_ROLE = "ebc"
SPARSE_EMBEDDING_ROLES = frozenset((SPARSE_EC_ROLE, SPARSE_EBC_ROLE))
# NvEmbeddings and the future Processor consumer reserve this key as invalid.
# Other negative int64 values remain valid bit patterns for dynamic uint64 IDs.
SPARSE_EMBEDDING_INVALID_KEY = -1


@dataclass(frozen=True)
class SparseEmbeddingIdentity:
    """Physical sparse table identity and its cross-system canonical name."""

    role: str
    table_name: str
    embedding_name: str
    dimension: int
    feature_names: Tuple[str, ...] = ()


def build_sparse_embedding_name_map(
    table_identities: Iterable[Tuple[str, str]],
) -> Dict[Tuple[str, str], str]:
    """Allocate canonical names for ``(collection role, table name)`` pairs.

    EC and EBC are separate physical collections. A table name used in only one
    collection keeps its historical name. If it appears in both collections,
    each physical table receives a role suffix. Candidate collisions with other
    raw table names are resolved deterministically with a numeric suffix.
    """
    roles_by_name = defaultdict(set)
    for role, table_name in table_identities:
        if role not in SPARSE_EMBEDDING_ROLES:
            raise ValueError(f"unsupported sparse embedding role: {role!r}")
        if not table_name:
            raise ValueError("sparse embedding table_name must not be empty")
        roles_by_name[table_name].add(role)

    used_names = set(roles_by_name.keys())
    name_by_identity: Dict[Tuple[str, str], str] = {}
    for table_name, roles in roles_by_name.items():
        if len(roles) == 1:
            role = next(iter(roles))
            name_by_identity[(role, table_name)] = table_name
            continue

        for role in sorted(roles):
            base_candidate = f"{table_name}__{role}"
            candidate = base_candidate
            suffix = 1
            while candidate in used_names:
                candidate = f"{base_candidate}_{suffix}"
                suffix += 1
            used_names.add(candidate)
            name_by_identity[(role, table_name)] = candidate
    return name_by_identity


def resolve_sparse_embedding_name(
    name_by_identity: Dict[Tuple[str, str], str],
    table_name: str,
    role: Optional[str],
) -> str:
    """Resolve a table to its canonical name, rejecting ambiguous identities."""
    if role is not None and (role, table_name) in name_by_identity:
        return name_by_identity[(role, table_name)]

    candidates = [
        embedding_name
        for (_role, name), embedding_name in name_by_identity.items()
        if name == table_name
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"sparse embedding {table_name!r} is not in model metadata")
    raise ValueError(
        f"sparse embedding {table_name!r} appears in multiple collection kinds; "
        f"cannot resolve canonical name without role, got role={role!r}"
    )
