# SID Collision Resolution: Fresh and Append Refactoring Plan

> Status: design proposal; no implementation changes are included.

## 1. Scope and terminology

The refactoring should expose two explicit modes:

- **Fresh** (the requested “Pure Addition” scenario): read one prediction file and run exactly the current collision-resolution algorithm. With no prior state, assignments, indexes, fallback behavior, statistics, and output ordering must remain unchanged.
- **Append**: read a new prediction file plus an existing `original_groups` snapshot and `resolved_groups` snapshot. Existing assignments are immutable; only new items are resolved and appended.

`original_groups` records the historical mapping from each original SID to its item IDs. `resolved_groups` records the final SID buckets. The position of an item in a resolved bucket is semantic: position `k` corresponds to the current 1-based map index `k`.

The inspected workspace artifacts are `experiments/sid_collision/sample.parquet` and `sid_collision/{original_groups,resolved_groups}`. The literal `/mnt/data/...` path was not present in this environment. The sample contains 500 unique items and is exactly the source of the existing snapshots, so using it as an append batch must fail duplicate validation rather than append another 500 copies.

## 2. Behavioral contract

| Property                        | Fresh                 | Append                           |
| ------------------------------- | --------------------- | -------------------------------- |
| Prior group inputs              | Forbidden             | Both snapshots required          |
| Existing final SIDs and indexes | Not applicable        | Never changed                    |
| Capacity seed                   | Empty                 | Counts from `resolved_groups`    |
| Original membership             | Current input         | Historical groups plus new input |
| Candidate resolution            | All current overflows | New overflows only               |
| Map output                      | Current input         | New-input delta only             |
| Group outputs                   | Complete snapshots    | New complete snapshots           |

Append is intentionally **not** equivalent to rerunning Fresh over the union. Old candidate lists are unavailable, historical assignments must remain stable, and existing items receive priority over new items. Consequently, append results are history-dependent.

Default duplicate policy is `error`: item IDs must be unique in the new batch and disjoint from historical `original_groups`. Neither silent skipping nor upsert is safe because the grouped state cannot prove that a repeated item has identical SID and candidate data.

All append outputs must use paths distinct from their input snapshots. `rate_only` should perform the same validation and planning and report new and combined statistics, but should not publish group or map data.

## 3. Stable persisted identity

The current dense band IDs are produced from prefixes present in one run. They are suitable for in-memory grouping but cannot identify a persisted bucket: for example, prefix `B` may be band 1 in one batch and band 0 in another.

Introduce a `SidKeyCodec` that validates every code against `layer_sizes` and converts a full SID to a stable, checked mixed-radix `int64` key. Because the last code has unit stride, the corresponding band key can be derived independently of the current batch. Reject configurations whose key space cannot fit in `int64`; a structured multi-column key is the future fallback if such configurations must be supported.

Persist a versioned state manifest. The group files alone cannot safely recover capacity or layer sizes—the inspected sample generator uses `(64, 64, 64)`, while the local invocation script specifies `(256, 256, 256)`. The manifest should contain:

- state format, SID-key codec, ordering/hash, and assignment-algorithm versions;
- `layer_sizes`, capacity, strategy, fallback policy, and random-count settings;
- Arrow item-ID type/schema fingerprint;
- original and resolved item counts, parent state version, and map scope (`delta`);
- artifact locations/formats and a completion marker.

Legacy snapshots should be bootstrapped once using explicit configuration, fully validated, and then accompanied by a manifest without changing any assignment.

## 4. Append planning algorithm

1. **Validate state.** Load the manifest, verify configuration and item-ID type compatibility, reject mixed CSV/Parquet directories, and ensure input and output paths do not overlap.
1. **Load compact occupancy.** Read `resolved_groups`, validate unique sorted SID keys and bucket contents, and retain only canonical SID keys and counts for planning. These counts—not `original_groups`—represent occupied capacity.
1. **Validate new IDs.** Check uniqueness in the new input. Sort its IDs once, then stream historical `original_groups` item lists through lookup against that sorted array. This avoids a Python set containing every historical item.
1. **Rank new origins.** Preserve the current deterministic hash ranking within each new original-SID bucket.
1. **Admit at origins.** For origin bucket `s`, compute `free(s) = max(capacity - existing_resolved_count(s), 0)`. Keep the first `free(s)` new items there; their absolute indexes start at `existing_resolved_count(s) + 1`. Mark the remainder as overflow.
1. **Reserve all origins first.** Add every newly retained origin item to occupancy before processing any candidate. This preserves the current Fresh rule that direct origin ownership has priority over candidate relocation.
1. **Resolve new overflow.** In deterministic order, apply the current first-fit or random strategy only to new overflow rows. Candidate occupancy starts from historical resolved counts plus all newly retained origin items. Candidates remain confined to the same canonical prefix band.
1. **Apply fallback.** `error` aborts before publication; `drop` omits unresolved new items from resolved groups and the map but retains their original membership; `keep_original` appends them to their origin even when this exceeds capacity.
1. **Build the delta.** Record each new item’s final SID and absolute 1-based index. Existing map rows are neither reconstructed nor rewritten.

With empty base occupancy, the generalized planner must produce byte-for-byte-equivalent logical results to the current Fresh implementation.

## 5. Snapshot merge and output rules

The first implementation should publish complete replacement snapshots:

- **Original snapshot:** one row per original SID; preserve the old item list as a prefix and append new IDs in the current deterministic within-bucket order.
- **Resolved snapshot:** one row per final SID; preserve the old list exactly as a prefix and append new IDs in assigned absolute-index order. This invariant makes every emitted map index verifiable by direct lookup.
- **Map:** emit rows for the new input only, matching the existing meaning of `output_path` as the mapping for the input being processed.

Use TorchEasyRec’s existing readers and writers, including the CSV writer. A small state-specific adapter is justified for manifest validation, normalized CSV/Parquet group decoding, streaming merge, and publication; a second generic I/O framework is not.

CSV represents an entire `itemids` group as one JSON-like field, so a hot bucket can create a line larger than the reader block size. Parquet or ODPS should be preferred for large state. Write chunks bound table/CSV batch memory, but they cannot split one oversized group without changing the schema.

Writers currently replace outputs independently, so do not update state in place. Write all artifacts under a new version, close and validate them, and publish the manifest/current-version pointer last. Include the parent version and use a lock or compare-and-swap check to prevent concurrent append jobs from losing updates.

If repeated full snapshots become too expensive at 100 million items, add an explicit base-plus-ordered-deltas format with periodic compaction as a later phase. Do not emit duplicate codebook rows under the current one-row-per-bucket snapshot contract.

## 6. ODPS append support check

**Conclusion (verified 2026-07-14): MaxCompute supports physical row append, but the current TorchEasyRec writer does not expose it, and direct physical append is incorrect for the grouped-state schema.**

| Layer                     | Append capability                                                                             | Consequence for this design                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| MaxCompute SQL            | `INSERT INTO` appends to a standard table or static partition; `INSERT OVERWRITE` replaces it | The platform has append capability, subject to table restrictions            |
| PyODPS / Storage API      | Non-overwrite writes can append; `overwrite=True` requests replacement                        | The underlying client can represent both modes                               |
| TorchEasyRec `OdpsWriter` | Hardcodes `TableBatchWriteRequest(..., overwrite=True)`                                       | Every writer run replaces the target table or partition                      |
| SID group snapshot        | Requires one row per `codebook` with the complete `itemids` list                              | Appending a touched bucket would create a duplicate row, not update its list |

The platform behavior is documented by [MaxCompute `INSERT INTO` / `INSERT OVERWRITE`](https://www.alibabacloud.com/help/en/maxcompute/user-guide/insert-or-update-data-into-a-table-or-a-static-partition), [TableTunnel overwrite semantics](https://www.alibabacloud.com/help/en/maxcompute/user-guide/tabletunnel), [PyODPS table writes](https://pyodps.readthedocs.io/en/latest/base-tables.html#write-data-to-tables), and [PyODPS Storage API writes](https://pyodps.readthedocs.io/en/stable/base-storage-api-v2.html#writing-data). MaxCompute also states that `INSERT INTO` is unsupported for clustered tables and that concurrent inserts to the same target are not protected by table locking. Its [ACID documentation](https://www.alibabacloud.com/help/en/maxcompute/product-overview/acid-semantics) does not provide a transaction spanning the separate SID output tables.

Repository inspection confirms that `tzrec/datasets/odps_dataset.py:762-764` always supplies `overwrite=True`; neither `OdpsWriter.__init__` nor the collision workflow passes an append option. Repeated `write()` calls accumulate batches only inside that one overwrite session. Existing ODPS tests validate a single writer session, not retention across two writer runs. The repository pins PyODPS 0.12.5.1, whose [`TableBatchWriteRequest`](https://raw.githubusercontent.com/aliyun/aliyun-odps-python-sdk/v0.12.5.1/odps/apis/storage_api/storage_api.py) accepts an overwrite flag, but the explicit `True` in TorchEasyRec prevents append.

For example, physical ODPS append changes `(A, [1, 2])` plus `(A, [3])` into two rows for codebook `A`; it does not produce `(A, [1, 2, 3])`. This violates the current snapshot invariant even though the database append itself succeeds.

Therefore, the initial ODPS implementation should keep **logical Append** separate from **physical row append**:

1. Read the parent `original_groups` and `resolved_groups` partitions.
1. Merge the old lists with the new delta in the application.
1. Write complete group snapshots to new immutable `state_version=<run_id>` partitions using the existing overwrite writer.
1. Validate all artifacts and publish the completed-version manifest last.
1. Enforce a single writer or parent-version check because the four outputs do not share one transaction and MaxCompute provides no insert lock.

Do not add a generic `append=True` switch to `OdpsWriter` merely for this workflow. Physical append becomes appropriate only if a future, explicitly different schema stores ordered bucket fragments such as `(state_version, sequence, codebook, itemids_delta)` and readers fold those deltas before use.

No ODPS credentials or CI project are configured in this workspace, so this verification combines official documentation, the pinned and installed PyODPS request models, and repository source inspection rather than a live remote write. Before release, add an ODPS integration gate using a temporary non-clustered table and static partition with `ARRAY<BIGINT>` item IDs: write `(A, [1, 2])`, append `(B, [3])` and `(A, [4])` with `overwrite=False`, confirm that old rows survive and `A` is duplicated rather than merged, verify `overwrite=True` replacement, and inject a failed version publication to prove the parent snapshot remains active.

## 7. Proposed modularization

- `collision_resolution.py`: pure array-based Fresh/Append planning; no storage concerns.
- `SidKeyCodec`: stable SID encoding, decoding, range checks, and band identity.
- `BucketOccupancy`: compact historical counts plus a mutable overlay for touched buckets.
- `CollisionPlan` / `CollisionResult`: distinguish base counts, new admissions, new final codes/indexes, unresolved rows, and combined counts.
- A state module such as `collision_state.py`: manifest, group readers, integrity checks, snapshot merge, and atomic publication using existing repository I/O.
- `collision_prevention.py`: CLI parsing, mode validation, orchestration, statistics, and writer selection.

Keep Arrow arrays at the I/O boundary and NumPy arrays in the CPU planner. Converting everything to tensors would add copies and device concerns without benefiting this sort/group/merge workload. Avoid a global Python dictionary or item-ID set at 100-million scale; use sorted arrays, vectorized search, streaming historical IDs, and a small overlay for buckets touched by the new batch.

The existing grouping helper must not scatter new rows directly into arrays sized by combined counts. Append indexes are absolute, while a delta buffer is relative; use `delta_position = absolute_index - base_bucket_count` when constructing new group fragments.

## 8. Result-affecting decisions

These rules must be documented and treated as compatibility contracts:

- Existing items always win capacity and never move.
- Existing resolved item order—and therefore indexes—is immutable.
- All new origin admissions are reserved before any new candidate assignment.
- New-item ordering remains deterministic and input-order invariant.
- The capacity, `layer_sizes`, item-ID type, hash version, and assignment version must match the parent state.
- `drop` permits resolved state to contain fewer items than original state; `keep_original` permits over-capacity buckets.
- The append map is a delta. A full historical map would require the old map as an additional state input or a costly reconstruction.

Temporary full-array/debug printing currently present in the working tree must be removed before scale tests because it would dominate runtime and memory, but that cleanup is outside this design-only change.

## 9. Verification plan

1. Golden/differential test: Fresh with empty state exactly matches the current implementation for first-fit, random, all fallbacks, and `rate_only`.
1. Stable-key test: historical prefixes `A,B`, followed by a new batch containing only `B`, catches batch-local band IDs.
1. Append cases: empty, partially filled, full, and already-overfull origins; occupied candidate targets; new-only buckets; and competing candidates.
1. Immutability checks: every old final SID, item-list prefix, and index remains unchanged; every new map index points to the merged resolved list.
1. Duplicate checks: within-new and new-versus-history; appending the supplied sample to the supplied snapshots must reject all 500 overlaps.
1. Compatibility checks: capacity, layer sizes, item-ID type, manifest version, schema, format, and path mismatch.
1. Fallback and format matrix: `error`, `drop`, `keep_original`; CSV and Parquet round trips; mixed-format directories rejected.
1. Operational tests: two sequential appends, input-order permutation, failed publication, concurrent-parent mismatch, and recovery of the prior active version.
1. Scale benchmarks: 100-million historical items, many buckets, one hot bucket, bounded planner memory, snapshot merge throughput, and CSV line-size limits.
1. ODPS integration: verify platform append in a temporary standard table/static partition, current `OdpsWriter` replacement across separate runs, version-partition isolation, and rejection of duplicate-codebook snapshot rows.

## 10. Implementation sequence

1. Freeze Fresh behavior with golden tests and define the manifest/key compatibility contract.
1. Add stable SID keys, state data classes, and legacy-state validation/bootstrap.
1. Generalize the pure planner to accept base occupancy; prove empty-base equivalence.
1. Add delta grouping and streaming full-snapshot merge through existing I/O.
1. Add explicit `fresh|append` CLI validation, versioned publication, statistics, and failure handling.
1. Complete the correctness/format/scale matrix, then consider delta files plus compaction only if snapshot rewrite cost warrants it.
