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

"""The prompt front-end a serving runtime loads: assemble, look up, project.

It wraps the collator's own ``PromptAssembler`` -- the one walk, scripted --
and adds the two stages serving needs after it: the slot lookup, when the host
has no embedding stage of its own, and the trained projections into the LM
input space.
"""

from typing import Dict, List, Optional

import torch
from torch import nn

from tzrec.prompt.assembler import PromptAssembler, batch_device

SLOT_EMBEDS = "slot_embeds"


class SlotTable(nn.Module):
    """One projected slot's members, as plain embedding tables.

    Used when the host does not run an embedding stage of its own. The lookup
    has to live somewhere: either the host does it and hands over vectors, or
    the artifact carries the tables. Keeping both shapes behind one module means
    the walk, the fold and the projections do not change between them.

    Args:
        value_keys: each member's value key.
        length_keys: each member's per-row counts.
        key_length_keys: each member's per-item value counts, empty when the
            member holds one value per item.
        num_embeddings: each member's table size.
        dims: each member's embedding dimension.
        is_sequence: whether holes are items rather than rows.
        modes: each member's pooling, ``sum`` or ``mean``.
    """

    value_keys: List[str]
    length_keys: List[str]
    key_length_keys: List[str]

    def __init__(
        self,
        value_keys: List[str],
        length_keys: List[str],
        key_length_keys: List[str],
        num_embeddings: List[int],
        dims: List[int],
        is_sequence: bool,
        modes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.value_keys = value_keys
        self.length_keys = length_keys
        self.key_length_keys = key_length_keys
        self.is_sequence = is_sequence
        modes = modes if modes is not None else ["sum"] * len(dims)
        self.tables = nn.ModuleList(
            [
                nn.EmbeddingBag(rows, dim, mode=mode, include_last_offset=True)
                for rows, dim, mode in zip(num_embeddings, dims, modes)
            ]
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Look every member up and concatenate along the feature axis.

        Returns:
            ``(num_holes, group_total_dim)``, ordered by hole.
        """
        parts: List[torch.Tensor] = []
        index = 0
        for table in self.tables:
            values = batch[self.value_keys[index]].to(torch.int64).reshape(-1)
            key = self.key_length_keys[index]
            if key != "":
                counts = batch[key].to(torch.int64)
            elif self.is_sequence:
                counts = torch.ones_like(values)
            else:
                counts = batch[self.length_keys[index]].to(torch.int64)
            offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=values.device),
                    torch.cumsum(counts, dim=0),
                ]
            )
            parts.append(table(values, offsets))
            index += 1
        return torch.cat(parts, dim=1)


class PromptFrontEnd(nn.Module):
    """The serving artifact: assemble, look up, project.

    Exported whether or not any slot is projected -- with none it degenerates
    to the assembler and three empty streams, which is cheap and keeps what a
    prompt happens to contain from deciding whether a serving-critical artifact
    exists.

    ``slot_embeds`` is ordered by ascending hole, matching ``hole_positions``
    entry for entry. That ordering is an obligation rather than a check: the
    engine scatters positionally, so a permuted source gives every hole a
    neighbour's embedding, the counts still agree, and nothing raises.

    Lookup is a stage of its own. Without a host embedding stage the artifact
    carries the tables and the batch holds raw ids. Under a host stage the
    embeddings are already in the batch when this runs, one entry per key in
    ``embed_keys`` for each slot, concatenated on the feature axis. Everything
    downstream of the lookup is identical either way.

    Args:
        assembler: the walk.
        projections: one per projected slot in ``projected_slots`` order; slots
            sharing a module appear more than once, by reference.
        embed_keys: per projected slot, the batch keys holding its members'
            looked-up rows, in member order. Ignored when ``tables`` is given.
        tables: one per projected slot, when the artifact carries the slot
            tables rather than receiving vectors from a host stage.
        vocab_hash: the compiled prompt's vocabulary digest, so a loader can
            refuse a front-end that does not pair with its ``prompt.json``.
        plan_hash: the compiled prompt's plan digest.
        bundle_uuid: identity of the SID bundle the prompt was compiled against.
    """

    embed_keys: List[List[str]]
    vocab_hash: str
    plan_hash: str
    bundle_uuid: str

    def __init__(
        self,
        assembler: PromptAssembler,
        projections: List[nn.Module],
        embed_keys: List[List[str]],
        tables: Optional[List[SlotTable]] = None,
        vocab_hash: str = "",
        plan_hash: str = "",
        bundle_uuid: str = "",
    ) -> None:
        super().__init__()
        self.assembler = assembler
        self.projections = nn.ModuleList(projections)
        self.embed_keys = embed_keys
        self.has_tables = tables is not None
        self.tables = nn.ModuleList(tables if tables is not None else [])
        self.vocab_hash = vocab_hash
        self.plan_hash = plan_hash
        self.bundle_uuid = bundle_uuid

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Assemble one batch and project its holes.

        The second argument is not decoration: a C++ host that runs this as its
        JIT stage calls ``forward(data, device)``, the same pair tzrec's own
        ScriptWrapper takes. A Python caller may omit it, in which case the
        batch stays where it is.

        Args:
            data: the parsed feature dict, plus looked-up rows under a host
                lookup stage.
            device: where to run; the batch is moved there first.

        Returns:
            The assembler's outputs plus ``slot_embeds``.
        """
        target = batch_device(data) if device is None else device
        batch: Dict[str, torch.Tensor] = {}
        for key, value in data.items():
            batch[key] = value.to(target)
        out = self.assembler(batch)

        # gathered first, into a plain list: TorchScript indexes a ModuleList
        # only with a literal, so the two lists cannot be walked together
        features: List[torch.Tensor] = []
        if self.has_tables:
            for table in self.tables:
                features.append(table(batch))
        else:
            for keys in self.embed_keys:
                members: List[torch.Tensor] = []
                for key in keys:
                    members.append(batch[key])
                features.append(torch.cat(members, dim=1))

        parts: List[torch.Tensor] = []
        index = 0
        for projection in self.projections:
            parts.append(projection(features[index]))
            index += 1

        # literals rather than the module constants: TorchScript cannot see a
        # module-level global. ``frontend_test`` pins them together.
        if len(parts) > 0:
            out["slot_embeds"] = torch.cat(parts, dim=0)
        else:
            out["slot_embeds"] = torch.zeros(
                0, 0, dtype=torch.float32, device=out["input_ids"].device
            )
        return out
