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
from tzrec.features.feature import BaseFeature
from tzrec.protos.feature_pb2 import FeatureConfig


class SidFeature(BaseFeature):
    """Semantic-ID sequence feature.

    A flat stream of 0-based per-level SID codes -- whole items in level order --
    plus the prompt text wrapping them. Under fg it is a passthrough.

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
        super().__init__(feature_config, **kwargs)
        self._codebook = self._read_codebook()
        # fg truncates by VALUE count and keeps the head, so a cap that is not a
        # whole number of items would hand the model partial items. The model's
        # own max_sequence_length is item-aligned and keeps the recent tail.
        if self.config.HasField("sequence_length"):
            if self.config.sequence_length % len(self._codebook):
                raise ValueError(
                    f"{self.__class__.__name__}[{self.config.feature_name}]: "
                    f"sequence_length ({self.config.sequence_length}) must be a "
                    f"multiple of the {len(self._codebook)}-level codebook, or "
                    f"fg would cut an item in half. Prefer "
                    f"model_config.common.max_sequence_length, which is "
                    f"item-aligned and keeps the most RECENT items."
                )
        self._level_sizes = np.asarray(self._codebook)
        self._level_offsets = np.cumsum(self._level_sizes) - self._level_sizes

    def _read_codebook(self) -> List[int]:
        """Validate the declared codebook once and normalize it to a list."""
        codebook = [int(c) for c in self.config.codebook]
        if not codebook:
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: codebook "
                f"is required; give one vocabulary size per SID level."
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

        Offsets are folded in here, in the dataloader workers: validating on the
        forward path would let one rank raise and hang its peers on the
        collective.
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
        # rows are whole items, so each column is one level.
        codes = parsed.values.reshape(-1, num_levels)
        if ((codes < 0) | (codes >= self._level_sizes)).any():
            raise ValueError(
                f"{self.__class__.__name__}[{self.config.feature_name}]: SID "
                f"codes must be local 0-based values in [0, codebook[level])."
            )
        # keep the dtype: int64 offsets would promote float32 to float64.
        offsets = self._level_offsets.astype(codes.dtype, copy=False)
        parsed.values = (codes + offsets).reshape(parsed.values.shape)
        return parsed

    def _fg_json(self) -> List[Dict[str, Any]]:
        """Get fg json config impl.

        A PASSTHROUGH: no fg feature_type can add ``level_offsets[i % levels]``,
        so fg only reaches the codes and ``_parse`` does the arithmetic. It
        exists because ``fg_mode`` is a data_config-level switch -- refusing it
        here would block every other feature in the config.
        """
        # SCALAR form: fg_json prepends "sequence_" and injects the seq keys.
        fg_cfg: Dict[str, Any] = {
            "feature_type": "raw_feature",
            "feature_name": self.config.feature_name,
            "expression": self.config.expression,
            "default_value": self.config.default_value,
            "value_type": "float",
        }
        if self.config.HasField("stub_type"):
            fg_cfg["stub_type"] = self.config.stub_type
        return [fg_cfg]
