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

"""ResidualQuantizer: abstract base for multi-layer residual quantizers."""

from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid.types import (
    QuantizeOutput,
    ResidualPassOutput,
    ResidualQuantizerOutput,
)
from tzrec.modules.utils import BaseModule


def normalize_n_embed(n_embed: Union[int, List[int]], n_layers: int) -> List[int]:
    """Broadcast a scalar codebook size to a per-layer list (or validate one).

    Args:
        n_embed (int|List[int]): codebook size, shared or per-layer.
        n_layers (int): number of residual quantization layers.

    Returns:
        List[int]: per-layer codebook sizes, length ``n_layers``.
    """
    if isinstance(n_embed, int):
        return [n_embed] * n_layers
    assert len(n_embed) == n_layers, (
        "length of n_embed and n_layers must be same, "
        f"but got {len(n_embed)} vs {n_layers}"
    )
    return list(n_embed)


class ResidualQuantizer(BaseModule):
    """Abstract base for multi-layer residual quantization.

    Shared contract for the two SID quantizer backends — the VQ-based,
    gradient-trained :class:`ResidualVectorQuantizer` and the K-Means-based,
    offline-FAISS-trained :class:`ResidualKMeansQuantizer`. Both quantize the
    residual of the previous layer:

        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = layer_i(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1}).

    This base owns the structural invariants (``embed_dim``, ``n_layers``,
    per-layer codebook sizes, residual normalization toggle) and the shared
    residual walk (:meth:`_residual_pass`, :meth:`get_codes`,
    :meth:`decode_codes`, :meth:`output_dim`), plus the concrete encode
    primitive :meth:`_quantize_layer` (which delegates to each layer's
    ``quantize``). Subclasses build ``self.layers`` and implement the decode
    primitive :meth:`_lookup_code`, plus :meth:`forward` and
    :meth:`get_codebook_embeddings`.

    Args:
        embed_dim (int): feature / codebook dimension.
        n_layers (int): number of residual quantization layers.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        normalize_residuals (bool): L2-normalize residuals before each
            layer. Default: False.
        candidate_output_config (Mapping|None): optional inference-time candidate
            SID settings (``enabled`` / ``topk`` / ``strategy``); see
            :meth:`_init_candidate_output_config`. Default: None (disabled).
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        candidate_output_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        assert n_layers >= 1, f"n_layers must be >= 1, got {n_layers}"
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals
        self.n_embed_list = normalize_n_embed(n_embed, n_layers)
        self._init_candidate_output_config(candidate_output_config)
        self.layers: nn.ModuleList = nn.ModuleList()

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim

    def forward(self, input: torch.Tensor):  # noqa: ANN201
        """Assign codes per layer and accumulate the quantized output."""
        raise NotImplementedError

    def _init_candidate_output_config(
        self,
        candidate_output_config: Optional[Mapping[str, Any]],
    ) -> None:
        """Read optional inference candidate-output settings."""
        # Always define the fields (last layer, greedy-only) so downstream reads
        # stay safe even when candidate output is disabled.
        self._candidate_output_enabled = False
        self._candidate_layer_idx = self.n_layers - 1
        self._candidate_output_topk = 1

        if candidate_output_config is None or not candidate_output_config["enabled"]:
            return

        self._candidate_output_topk = int(candidate_output_config["topk"])

        n_embed = self.n_embed_list[self._candidate_layer_idx]
        if self._candidate_output_topk < 1 or self._candidate_output_topk > n_embed:
            raise ValueError(
                f"candidate_output_config.topk must be in [1, {n_embed}], "
                f"got {self._candidate_output_topk}"
            )
        if candidate_output_config["strategy"] != "last_layer_knn":
            raise ValueError(
                "candidate_output_config.strategy supports only 'last_layer_knn' in v1."
            )

        self._candidate_output_enabled = True

    def _should_output_candidates(self) -> bool:
        """Whether forward should emit candidate SID tensors.

        Candidate output is a configured, inference-only behavior: inference mode
        + the enabling config + a codebook that can actually produce candidates
        (:meth:`_candidates_available`), with no per-call flag.
        """
        return (
            self.is_inference
            and self._candidate_output_enabled
            and self._candidates_available()
        )

    def _candidates_available(self) -> bool:
        """Whether the codebook can currently produce candidate SIDs.

        Base default: always. A backend whose codebook may be unfit (e.g.
        :class:`~tzrec.modules.sid.residual_kmeans_quantizer.ResidualKMeansQuantizer`
        before ``train_offline``) overrides this so candidate output is skipped —
        not crashed — until the fit lands.
        """
        return True

    def _quantize_layer(
        self,
        layer_idx: int,
        residual: torch.Tensor,
        topk: int = 1,
    ) -> QuantizeOutput:
        """Assign one layer's codes and look up its quantized vector.

        Backend primitive behind the residual walk (encode-direction mirror of
        :meth:`_lookup_code`).

        Args:
            layer_idx (int): quantization layer index.
            residual (Tensor): current residual, shape (B, D).
            topk (int): number of nearest neighbors to return.

        Returns:
            QuantizeOutput: selected ids/embeddings and optional top-k neighbors.
        """
        if layer_idx >= len(self.layers):
            raise NotImplementedError(
                "ResidualQuantizer subclasses must populate self.layers."
            )
        return self.layers[layer_idx].quantize(residual, topk=topk)

    def _residual_pass(self, input: torch.Tensor) -> ResidualPassOutput:
        """Shared residual walk: per-layer assign, subtract, accumulate.

        The quantized vector is subtracted detached (keeps the residual chain
        gradient-free) and accumulated (keeps gradient when the backend
        supplies it, e.g. VQ). Whether candidate SIDs are produced is decided
        internally by :meth:`_should_output_candidates` (inference mode + the
        enabling config), so callers never pass a flag; the candidate walk
        reuses the last-layer residual from this same pass.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            ResidualPassOutput: the walk result (``cluster_ids``, raw
                ``aggregated`` sum, per-layer ``cumulative`` sums, and optional
                ``candidate_codes`` / ``candidate_scores``).
        """
        include_candidates = self._should_output_candidates()
        if include_candidates and self.training:
            raise RuntimeError("candidate SID output requires eval/inference mode.")
        residual = input
        all_codes: List[torch.Tensor] = []
        cumulative: List[torch.Tensor] = []
        candidate_layer_output: Optional[QuantizeOutput] = None
        aggregated = torch.zeros_like(input)
        for i in range(self.n_layers):
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)
            is_candidate_layer = include_candidates and i == self._candidate_layer_idx
            layer_topk = self._candidate_output_topk if is_candidate_layer else 1
            out = self._quantize_layer(i, residual, topk=layer_topk)
            all_codes.append(out.ids)
            aggregated = aggregated + out.embeddings
            cumulative.append(aggregated)
            residual = residual - out.embeddings.detach()
            if is_candidate_layer:
                candidate_layer_output = out
        cluster_ids = torch.stack(all_codes, dim=-1)

        candidate_codes: Optional[torch.Tensor] = None
        candidate_scores: Optional[torch.Tensor] = None
        if candidate_layer_output is not None:
            candidate_codes, candidate_scores = self._build_code_candidates(
                cluster_ids,
                candidate_layer_output,
            )
        return ResidualPassOutput(
            cluster_ids=cluster_ids,
            aggregated=aggregated,
            cumulative=cumulative,
            candidate_codes=candidate_codes,
            candidate_scores=candidate_scores,
        )

    def _residual_output(
        self,
        walk: ResidualPassOutput,
        quantized_embeddings: torch.Tensor,
        with_latents: bool,
    ) -> ResidualQuantizerOutput:
        """Pack a residual walk into the public output type.

        ``quantized_embeddings`` is supplied separately because the VQ backend
        replaces the raw ``walk.aggregated`` sum with its STE-adjusted version.
        ``latents`` (the (B, n_layers, D) cumulative stack) is built only when a
        consumer needs it — the RQ-VAE commitment loss — and skipped otherwise
        (K-Means and the inference path never read it).
        """
        return ResidualQuantizerOutput(
            cluster_ids=walk.cluster_ids,
            quantized_embeddings=quantized_embeddings,
            latents=torch.stack(walk.cumulative, dim=1) if with_latents else None,
            candidate_codes=walk.candidate_codes,
            candidate_scores=walk.candidate_scores,
        )

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without updating the codebook.

        Shared encode-direction mirror of :meth:`decode_codes`.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
        return self._residual_pass(input).cluster_ids

    def _build_code_candidates(
        self,
        cluster_ids: torch.Tensor,
        layer_output: QuantizeOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build candidate SID tuples from one residual walk.

        Keeps the greedy prefix and varies only the last layer across its
        top-k nearest codes, best-first (slot 0 is the greedy code).
        """
        topk = self._candidate_output_topk
        assert layer_output.topk_ids is not None
        assert layer_output.topk_scores is not None

        candidate_layer_codes = layer_output.topk_ids
        candidate_scores = layer_output.topk_scores

        prefix_codes = (
            cluster_ids[:, : self._candidate_layer_idx]
            .unsqueeze(1)
            .expand(-1, topk, -1)
        )
        candidate_codes = torch.cat(
            [prefix_codes, candidate_layer_codes.unsqueeze(-1)],
            dim=-1,
        )

        return candidate_codes, candidate_scores

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get the codebook (centroid) weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        raise NotImplementedError

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up the codebook vectors for ``code_idx`` at ``layer_idx``.

        The single backend-specific primitive :meth:`decode_codes` builds on
        (VQ reads ``embedding(idx)``, K-Means reads ``centroids[idx]``).

        Args:
            layer_idx (int): index of the quantization layer.
            code_idx (Tensor): codebook indices, shape (B,).

        Returns:
            Tensor: looked-up codebook vectors, shape (B, embed_dim).
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from semantic ID codes (centroid sum).

        Args:
            codes (Tensor): cluster ids, shape (B, n_layers).

        Returns:
            Tensor: reconstructed embeddings, shape (B, embed_dim).
        """
        # Seed from the first lookup so dtype/device follow the codebook (not fp32/AMP).
        quantized_sum = self._lookup_code(0, codes[:, 0])
        for i in range(1, self.n_layers):
            quantized_sum = quantized_sum + self._lookup_code(i, codes[:, i])
        return quantized_sum
