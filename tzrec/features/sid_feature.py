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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from tzrec.datasets.utils import ParsedData
from tzrec.features.feature import BaseFeature, FgMode
from tzrec.protos.feature_pb2 import FeatureConfig


class SidFeature(BaseFeature):
    """Semantic-ID sequence feature.

    A flat stream of 0-based per-level SID codes -- whole items in level order --
    plus the prompt text wrapping them. Generated offline, so only
    ``fg_mode = FG_NONE`` works; there is no pyfg counterpart.

    Args:
        feature_config (FeatureConfig): a instance of feature config.
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        **kwargs: Any,
    ) -> None:
        # BaseFeature.__del__ dereferences _fg_op, so seed it before any raise.
        self._fg_op = None
        # checked before super(), which calls init_fg() for FG_NORMAL and would
        # surface the missing pyfg handler instead of this explanation.
        fg_mode = kwargs.get("fg_mode", FgMode.FG_NONE)
        if fg_mode != FgMode.FG_NONE:
            raise ValueError(
                f"{self.__class__.__name__}"
                f"[{feature_config.sequence_sid_feature.feature_name}] supports "
                f"data_config.fg_mode = FG_NONE only (SID codes are generated "
                f"offline by a SID model), got {fg_mode}."
            )
        super().__init__(feature_config, **kwargs)
        # only fg (which this feature forbids) and an fx export marker read it,
        # so it would cap nothing; the model owns the real, item-aligned budget.
        if self.config.HasField("sequence_length"):
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: "
                f"sequence_length does not truncate a SID feature; set "
                f"model_config.common.max_sequence_length instead."
            )
        self._codebook = self._read_codebook()
        self._level_sizes = np.asarray(self._codebook)
        self._level_offsets = np.cumsum(self._level_sizes) - self._level_sizes

    def _read_codebook(self) -> List[int]:
        """Validate the declared codebook once and normalize it to a list.

        Every derived quantity reads it on the parse hot path, so the repeated
        scalar container is checked and converted here, not per access.
        """
        codebook = [int(c) for c in self.config.codebook]
        if not codebook:
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: codebook "
                f"must be non-empty."
            )
        if any(c <= 0 for c in codebook):
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: every "
                f"codebook size must be positive, got {codebook}."
            )
        return codebook

    @property
    def value_dim(self) -> int:
        """Fg value dimension of the feature."""
        return self.config.value_dim

    @property
    def output_dim(self) -> int:
        """Output dimension: SID codes pass through to the LM's own table."""
        return self.value_dim

    @property
    def num_embeddings(self) -> int:
        """Get embedding row count."""
        raise RuntimeError(
            f"{self.__class__.__name__}[{self.config.feature_name}] has no "
            f"embedding table; SID codes index the LM vocabulary."
        )

    @property
    def prefix_text(self) -> str:
        """Text emitted immediately before this feature's SID tokens."""
        return self.config.prefix_text

    @property
    def suffix_text(self) -> str:
        """Text emitted immediately after this feature's SID tokens."""
        return self.config.suffix_text

    @property
    def codebook(self) -> List[int]:
        """Per-level SID vocabulary sizes; validated once at construction."""
        return self._codebook

    @property
    def num_levels(self) -> int:
        """Codes per item -- also the answer width."""
        return len(self._codebook)

    @property
    def sid_vocab_size(self) -> int:
        """Atoms the model must append to the backbone vocabulary."""
        return sum(self._codebook)

    @property
    def level_offsets(self) -> List[int]:
        """Flat offset of each level, i.e. ``cumsum(sizes) - sizes``."""
        return self._level_offsets.tolist()

    def _build_side_inputs(self) -> Optional[List[Tuple[str, str]]]:
        """Input field names with side."""
        if self.config.HasField("expression"):
            return [tuple(self.config.expression.split(":"))]
        else:
            return None

    def _parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """Parse the SID stream into flat indices in the shared space.

        Codes are 0-based, so the flat index IS the atom index and the model only
        adds ``base_vocab``. Offsets are folded in here, in the dataloader
        workers, not on the forward path -- and validating here keeps a malformed
        row off the collective path, where one rank raising hangs its peers.
        """
        parsed = super()._parse(input_data)
        num_levels = len(self._codebook)
        bad = np.nonzero(parsed.seq_lengths % num_levels)[0]
        if bad.size:
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: every "
                f"row must hold whole {num_levels}-level items; rows "
                f"{bad.tolist()[:10]} have lengths "
                f"{parsed.seq_lengths[bad].tolist()[:10]}."
            )
        # rows are whole items, so (-1, num_levels) lines every column up with
        # its level and the per-level bounds/offsets broadcast down it.
        codes = parsed.values.reshape(-1, num_levels)
        if ((codes < 0) | (codes >= self._level_sizes)).any():
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: SID "
                f"codes must be local 0-based values in [0, codebook[level])."
            )
        # keep the value dtype: float32 + int64 offsets would promote to float64
        # and double the bytes crossing worker IPC and the H2D copy.
        offsets = self._level_offsets.astype(codes.dtype, copy=False)
        parsed.values = (codes + offsets).reshape(parsed.values.shape)
        return parsed

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl."""
        raise RuntimeError(
            f"{self.__class__.__name__}[{self.config.feature_name}] has no fg "
            f"representation; SID codes are generated offline (fg_mode=FG_NONE)."
        )
