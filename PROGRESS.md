# ULTRA-HSTU Implementation Progress

## Completed

- [x] CUTLASS backend: plumb `func` tensor + SLA mask + FP8 dtype relaxation
  - `cutlass_hstu_attention.py`: added `func` to schemas, impls, autograd, and public wrapper
  - `build_sla_func_tensor()`: constructs NFUNC=3 int32 tensor for SLA mask intervals
- [x] PyTorch SLA reference: `pytorch_sla_hstu_mha` in `pt_hstu_attention.py`
- [x] Proto changes: `sla_k1`, `sla_k2`, `selective_rematerialization` in STU; `attn_truncation_*` in HSTU
- [x] Attention truncation: `truncate_jagged_tail()` helper + `STUStack` split-layer logic
- [x] Selective activation rematerialization: `_forward_with_selective_remat` in STULayer
- [x] Thread `sla_k1`/`sla_k2` from STULayer → hstu_compute → hstu_mha → CUTLASS

## In Progress

- [ ] Tests + integration config
- [ ] Action-Item Merging preprocessor
- [ ] Mixture of Transducers

## Deferred

- INT4 embedding serving
- LBSL distributed sampler
- fbgemm_gpu_hstu migration
- delta_hstu_mha CUTLASS port
- FP8 `quant_mode` exposure (pending wheel interface verification)
