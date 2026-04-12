# Unified AOTI Export — Status

## What Works

- Non-quantized models: unified AOTI export works ✓
- Quantized models with two-stage fallback: works ✓

## What Doesn't Work Yet

- Quantized models (QUANT_EC_EMB) with unified AOTI export

## Root Cause (QUANT_EC_EMB limitation)

Quantized EBC uses torchrec's KJT processing which calls `.item()`
on tensor VALUES to extract lengths/strides as Python scalars. During
`torch.export`, `.item()` creates unbacked SymInts because the values
are data-dependent (sequence lengths vary per batch). These unbacked
SymInts flow through the model and hit guards in nn.BatchNorm1d
(`Eq(512*u444, 0)`) that torch.export can't resolve.

This is a fundamental limitation of `torch.export` with jagged/variable-
length sequence models that use `.item()` for shape computations. The
two-stage export avoids it because the sparse model (JIT) handles all
KJT processing with concrete values.

## Patches Applied

1. Triton op zero-check guards (`not torch.compiler.is_compiling()`)
1. Triton dropout deterministic seeds during export
1. fbgemm schema mutation suppression (MutationChecker)
1. autotune_max_seq_len torch.\_check moved inside branch
1. CudaAutocastWrapper for mixed precision
1. export_patches context manager for future fbgemm fixes

## Upstream Issues Blocking QUANT_EC_EMB + Unified

1. torchrec KJT `.item()` creates unbacked SymInts
1. fbgemm permute_2D_sparse_data wrong schema annotation
1. fbgemm bounds_check_indices wrong schema annotation
