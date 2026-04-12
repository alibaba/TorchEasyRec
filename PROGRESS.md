# ULTRA-HSTU Implementation Progress

## Completed

- [x] CUTLASS backend: plumb `func` tensor + SLA mask + FP8 dtype relaxation
- [x] PyTorch SLA reference: `pytorch_sla_hstu_mha` in `pt_hstu_attention.py`
- [x] Proto changes: `sla_k1`, `sla_k2`, `selective_rematerialization` in STU;
  `attn_truncation_*` in HSTU; `action_item_merge_preprocessor` oneof
- [x] Attention truncation: `truncate_jagged_tail()` + `STUStack` split-layer
- [x] Selective activation rematerialization in STULayer
- [x] Thread `sla_k1`/`sla_k2` through STULayer -> hstu_compute -> hstu_mha
- [x] Action-Item Merging: reuses existing `contextual_preprocessor` path
  (content + action sum when `enable_interleaving=False`); added named
  alias `action_item_merge_preprocessor` in factory + proto
- [x] Mixture of Transducers: `MoTHSTUTransducer` in `mot_hstu.py` with
  sum / concat_mlp fusion over parallel HSTUTransducer channels
- [x] Tests: SLA parity (CUTLASS vs PyTorch ref), integration config

## Deferred

- INT4 embedding serving
- LBSL distributed sampler
- fbgemm_gpu_hstu migration
- delta_hstu_mha CUTLASS port
- FP8 `quant_mode` exposure (pending wheel interface verification)
