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

from typing import Any, List

import numpy as np
import torch

_DISTRIBUTED_SPARSE_QUANT_STORAGE_DTYPE = "uint8"
_DISTRIBUTED_SPARSE_QUANT_SCALE_OFFSET_BYTES = 4
DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS: List[str] = ["QUint8RowwiseF16"]


def _quantize_quint8_rowwise_f16(
    values: Any, emb_dim: int, emb_name: str
) -> np.ndarray:
    """Encode rows as nvembedding QUint8RowwiseF16."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    src = np.ascontiguousarray(values, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] != emb_dim:
        raise ValueError(
            f"Expected a 2D sparse embedding tensor with dim={emb_dim}, "
            f"got shape={list(src.shape)}"
        )

    rows = src.shape[0]
    row_bytes = emb_dim + _DISTRIBUTED_SPARSE_QUANT_SCALE_OFFSET_BYTES
    if row_bytes % 2 != 0:
        raise ValueError(
            "Distributed sparse quant export failed for embedding "
            f"'{emb_name}': DIST_QUANT=INT8 is exported as nvembedding "
            "QUint8RowwiseF16, whose stored row is "
            "[uint8 values][float16 scale][float16 offset]. This makes "
            f"row_bytes = embedding_dim + 4 = {emb_dim} + 4 = {row_bytes}. "
            "nvembedding GPU lookup requires quantized stored row_bytes to be "
            "even, but this table has an odd embedding_dim, so row_bytes is "
            "odd. QUint8RowwiseF16 is affine uint8 rowwise quantization, "
            "matching FBGEMM's fused rowwise INT8 value encoding with "
            "per-row scale and offset. To fix this, change the table's "
            "embedding_dim to an even value and retrain/re-export the model, "
            "or disable distributed sparse quantization by unsetting DIST_QUANT or "
            "setting DIST_QUANT=0/NONE. TorchEasyRec does not auto-pad this export "
            "because padding would change the nvembedding logical value count "
            "and serving contract."
        )
    out = np.empty((rows, row_bytes), dtype=np.uint8)
    if rows == 0:
        return out

    row_min = np.min(src, axis=1, keepdims=True).astype(np.float32)
    row_max = np.max(src, axis=1, keepdims=True).astype(np.float32)
    offset_fp16 = row_min.astype(np.float16)
    offset = offset_fp16.astype(np.float32)
    value_range = (row_max - offset).astype(np.float32)
    scale = np.where(value_range != 0, value_range / 255.0, np.float32(1.0))
    scale = scale.astype(np.float32)
    scale_fp16 = scale.astype(np.float16)
    scale = scale_fp16.astype(np.float32)
    scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    scale_fp16 = scale.astype(np.float16)
    quantized = np.rint((src - offset) / scale)
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)

    out[:, :emb_dim] = quantized
    out[:, emb_dim : emb_dim + 2] = scale_fp16.view(np.uint8).reshape(rows, 2)
    out[:, emb_dim + 2 : emb_dim + 4] = offset_fp16.view(np.uint8).reshape(rows, 2)
    return out


def distributed_quantize_embeddings(
    values: Any, emb_dim: int, emb_name: str, quant_format: str
) -> np.ndarray:
    """Quantize embedding values for distributed export.

    Args:
        values: Embedding values to quantize.
        emb_dim: Logical embedding dimension.
        emb_name: Embedding name used in validation errors.
        quant_format: Distributed sparse quantization format.

    Returns:
        Quantized embedding values.

    Raises:
        ValueError: If the quantization format is unsupported.
    """
    if quant_format == DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS[0]:
        return _quantize_quint8_rowwise_f16(values, emb_dim, emb_name)
    raise ValueError(
        f"Unsupported distributed sparse quant format: {quant_format}; "
        f"supported formats: {DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS}"
    )
