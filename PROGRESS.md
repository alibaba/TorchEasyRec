# TorchEasyRec 1.3.0 upgrade — progress

Branch: `bump_torch_2.12.1` (worktree). Base: master @ 52002e2.

## Dependency matrix

torch 2.11.0→2.12.1 · triton 3.6.0→3.7.1 · torch_tensorrt 2.11.0→2.12.0 · tensorrt_cu12 10.15.1.29→10.16.1.11
fbgemm 1.6.0→1.7.0 · torchrec 1.6.0→1.7.0 · dynamicemb e0c1fbb→20260630.5dc46a2 · hstu cu12.9→20260626.9fd44403
project 1.2.21→1.3.0 · docker tag 1.2→1.3

## Compatibility verdict

- REQUIRED: `tzrec/optim/optimizer.py` apply_split_helper re-sync (fbgemm 1.7.0 added make_persistent +
  preallocated_host_buffer; internal caller passes them → TypeError otherwise).
- KEEP: aot_utils.py #178147 backport (still absent in 2.12.1; lands 2.13.0+). Comment refresh only.
- torchrec/pytorch: no source change needed.
- triton: KEEP DISABLE_MMA_V3 workaround — #9514 RS-WGMMA fix is NOT in the 3.7.x *release* line
  (verified: v3.7.1 tag + installed wheel lack it; main-only). FAQ Q15 updated to DISABLE_MMA_V3 remedy.
- TRT DROPPED for 1.3.0 cu129/cu126 (user decision; cu130+TRT later): torch_tensorrt 2.12 has no cu12
  build (generic wheel links libcudart.so.13 / requires tensorrt-cu13; cu129 index stops at 2.11.0 which
  pins torch\<2.12). Removed torch-tensorrt + tensorrt_cu12 from requirements/cu126,cu129 + Dockerfile.
  has_tensorrt=False at runtime (graceful, like cpu image); TRT integration tests skip; AOTI unaffected.

## Steps

- [ ] version + requirements
- [ ] Dockerfile (versions, dist-info, PIP_MIRROR)
- [ ] optimizer.py apply_split_helper + aot_utils comment
- [ ] scripts + docs + pre-commit
- [ ] build/push tzrec-test:1.3 images
- [ ] workflows→tzrec-test, push origin, PR→master
- [ ] CI green
- [ ] promote→tzrec-devel, revert workflows, finalize

## Log

- created worktree; starting edits.
