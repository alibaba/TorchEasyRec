# Unified AOTI Export Progress

## Completed

- [x] Add `UnifiedAOTIModelWrapper` to `tzrec/models/model.py`
- [x] Add `export_unified_model_aot()` to `tzrec/acc/aot_utils.py`
- [x] Update `load_model_aot()` for backward compatibility
- [x] Wire up unified export in `tzrec/utils/export_util.py`
- [x] Update `create_test_model()` in `tzrec/utils/test_util.py`
- [x] Fix device handling (bake device as string constant in FX graph)
- [x] Fix dynamic shapes (group same-size tensors, handle scalars)
- [x] Fix data dict ordering (sort keys to match pytree spec)

## Tests Passing

- [x] `multi_tower_din_test` - all 4 variants (NORMAL, FX_TRACE, JIT_SCRIPT, AOT_INDUCTOR)
- [x] `dlrm_hstu_test` - AOT_INDUCTOR variant (hypothesis, CI mode)
- [x] Integration: `test_multi_tower_din_with_fg_train_eval_aot_export_input_tile`
  - ENABLE_AOT=1 (no tile) - export + predict OK
  - QUANT_EC_EMB=1 INPUT_TILE=2 ENABLE_AOT=1 - export + predict OK
  - QUANT_EC_EMB=1 INPUT_TILE=3 ENABLE_AOT=1 - export + predict OK
- [ ] Integration: `test_rank_dlrm_hstu_train_eval_export` - SKIPPED (missing test data: kuairand-1k-\*.parquet)

## Key Issues Resolved

1. `torch.device` not supported as torch.export input -> replaced device placeholder with string constant in FX graph
1. Dynamic shape constraint violations -> grouped tensors by dim-0 size to share Dims
1. Predict data dict ordering mismatch -> sort keys in UnifiedAOTIModelWrapper
1. Scalar tensors missing from dynamic_shapes -> include with empty dict
