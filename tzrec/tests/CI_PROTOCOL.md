# Test dispatch protocol v1

Each lane workflow launches one job container. Inside it, `dispatch local` owns the baseline, coordinator and pair-worker processes. All processes use the same prepared Python environment; workers receive disjoint device views, private checkouts and private compilation caches.

| Process     | Entry point                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| Baseline    | `python tzrec/tests/dispatch.py baseline --session /job/session --session-id ID --protocol 1 --scope gpu` |
| Coordinator | `python tzrec/tests/dispatch.py serve --session /job/session --session-id ID --protocol 1 --workers N`    |
| Pair worker | `python tzrec/tests/worker.py serve --session /job/session --protocol 1 --slot I`                         |

Baseline success means the handoff is complete, not that all tests passed. It writes `phase.json` with `version`, `session_id`, and `next` (`pairs` or `finish`). A failed test still has a receipt; a broken preparation does not publish a handoff. For `finish`, local dispatch runs the coordinator with zero workers to perform the same final accounting.

The immutable manifest contains a protocol version, session identity, deadline, original test IDs, fixture boundaries, routes, static skip decisions, and groups. Its digest covers the entire payload excluding the digest field. Group names are generated hashes. All processes must use the same prepared checkout and dependency environment.

The job-private `dispatch.sock` uses bounded newline-delimited JSON. Workers register a unique slot, version, session ID, manifest digest, and visible-device count. The coordinator assigns one group at a time. Workers execute each group in a new process, atomically publish its result, and acknowledge the exact assigned group. Disconnect, identity mismatch, missing/duplicate results or abnormal process exit fail the job. Local dispatch terminates its process groups if the coordinator or any worker fails; workflow cleanup removes only its named job container.

Results preserve one parent-case outcome, subtest details, and fixture events. Skipped subtests remain in the parent record as `subtest_skips`; they do not create extra case IDs. Cases blocked by failed fixtures are explicit failures. Detailed case/fixture journals remain inside the test report. Before recording a known failure, execution persists the generic `status/failed` marker. The optional `TZREC_CI_RECEIPT` directory contains a versioned `state.json` with run, attempt, lane, boot identity and terminal code. Addons reads only this generic receipt and marker, never the test journal. Preparation failures before the test phase have no receipt and do not authorize an automatic retry.

`summary.json` contains `version`, `session_id`, `digest`, `code`, `planned`, `reported`, outcome counts, missing/unexpected IDs, and measured durations. Success requires complete result accounting and normal coordinator/worker exits. Group receipts are internal to the tested checkout, not a cross-repository Python API.
