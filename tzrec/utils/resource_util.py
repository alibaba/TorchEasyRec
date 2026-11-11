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

"""Homogeneous per-rank host-memory capacity budgets.

Ranks are assumed to share one uniform deployment: the planning rank's
local observation is representative for the whole job, and sharing counts
come from configuration, not from filesystem or pool discovery.
"""

import psutil


def _container_memory_limit_bytes() -> int:
    """Physical RAM capped by a finite container (cgroup) limit.

    Inside a container psutil reports the node's RAM, not the container's
    limit, so a budget sized by it can exceed what the container may use.
    """
    total = psutil.virtual_memory().total
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            with open(path) as f:
                limit = f.read().strip()
        except OSError:
            continue
        if limit.isdigit():
            total = min(total, int(limit))
            break
    return total


def host_memory_capacity_per_rank(
    *, local_world_size: int, reserve_percent: float
) -> int:
    """Host memory one rank may use for embeddings, in bytes.

    The measured budget is the container's RAM less ``reserve_percent``
    (zero disables only the margin), split evenly among the ranks sharing
    one host; the caller must not divide the result again.

    Args:
        local_world_size: ranks sharing one host; at least one.
        reserve_percent: fraction held back for everything but embeddings,
            between 0 and 1.

    Returns:
        Integer bytes per rank, ready for ``Topology(ddr_cap=...)``.
    """
    assert 0 <= reserve_percent <= 1
    assert local_world_size >= 1
    total = _container_memory_limit_bytes()
    return int(total * (1 - reserve_percent) / local_world_size)
