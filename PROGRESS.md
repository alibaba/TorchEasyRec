# Unified AOTI Export - DLC Job Fix Progress

## Error Chain

1. **Original (dlc13efuoayb0nz2)**: `GuardOnDataDependentSymNode: Eq(u50, 0)` in
   `triton_swish_layer_norm_fwd` → `if N == 0:` inside triton_op
1. **After fix 1 (triton_op guards)**: Same error at `_triton_weighted_layer_norm_fwd` line 340
   — helper function called FROM triton_op
1. **After fix 2 (ALL guards)**: `GuardOnDataDependentSymNode: Eq(512*u444, 0)` in MLP perceptron
   — unbacked symbol from `aten.item()` (jagged sequence nnz)
1. **After fix 3 (deferred asserts)**: Same — `prefer_deferred_runtime_asserts` doesn't help for
   unbacked symbols
1. **After fix 4 (propagate_real_tensors)**: `fbgemm::permute_2D_sparse_data` schema mutation error
   — `propagate_real_tensors` mode detects fbgemm schema bug

## Fixes Applied

- `84edc62`: Guard ALL `if N == 0` checks with `not torch.compiler.is_compiling()`
- `59fadce`: `strict=False, prefer_deferred_runtime_asserts_over_guards=True` (didn't help)
- `f6317b2`: `TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1` (didn't help)
- `ea7c03e`: `fake_tensor_propagate_real_tensors=True` (got past GuardOnDataDependentSymNode
  but hit fbgemm schema issue)

## Current Approach

Reproducing locally with QUANT_EC_EMB=INT8 to debug interactively.

## DLC Jobs

| Job ID           | Wheel   | Result                                                          |
| ---------------- | ------- | --------------------------------------------------------------- |
| dlc13efuoayb0nz2 | 00ec8d1 | GuardOnDataDependentSymNode in triton_swish_layer_norm_fwd      |
| dlc1r9s64q7i515c | 2161b27 | Missing ODPS config (no CredentialConfig)                       |
| dlc1s3scui3g8ndd | 2161b27 | Missing ODPS config                                             |
| dlc1n412iyq3eho9 | 2161b27 | GuardOnDataDependentSymNode in \_triton_weighted_layer_norm_fwd |
| dlcxvag3uksal7ft | 84edc62 | GuardOnDataDependentSymNode: Eq(512\*u444, 0)                   |
| dlc12v8r9ma2dzhp | 59fadce | Same (deferred asserts didn't help)                             |
| dlcvnoxl6al1vzjl | f6317b2 | Same (env var didn't help)                                      |
| dlcxvrax6jlaqbwt | ea7c03e | fbgemm permute_2D_sparse_data schema mutation                   |
