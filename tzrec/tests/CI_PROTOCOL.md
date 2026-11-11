# Test dispatch protocol v1

The host owns containers and device views; TorchEasyRec owns discovery and test IDs. The host invokes fixed commands and reads only flat phase/result envelopes. It never executes a command or mount specification supplied by a manifest.

| Process     | Entry point                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| Baseline    | `python tzrec/tests/dispatch.py baseline --session /job/session --session-id ID --protocol 1 --scope gpu` |
| Coordinator | `python tzrec/tests/dispatch.py serve --session /job/session --session-id ID --protocol 1 --workers N`    |
| Pair worker | `python tzrec/tests/worker.py serve --session /job/session --protocol 1 --slot I`                         |

Baseline success means the handoff is complete, not that all tests passed. It writes `phase.json` with `version`, `session_id`, and `next` (`pairs` or `finish`). A failed test still has a receipt; a broken preparation does not publish a handoff. For `finish`, the host runs the coordinator with zero workers to perform the same final accounting.

The immutable manifest contains a protocol version, session identity, deadline, original test IDs, fixture boundaries, routes, static skip decisions, and groups. Its digest covers the entire payload excluding the digest field. Group names are generated hashes. All processes must use the same prepared checkout and dependency environment.

The job-private `dispatch.sock` uses bounded newline-delimited JSON. Workers register a unique slot, version, session ID, manifest digest, and visible-device count. The coordinator assigns one group at a time. Workers execute each group in a new process, atomically publish its result, and acknowledge the exact assigned group. Disconnect, identity mismatch, missing/duplicate results or abnormal process exit fail the job. The host terminates the entire job if the coordinator or any worker fails.

Results preserve one parent-case outcome, subtest details, and fixture events. Skipped subtests remain in the parent record as `subtest_skips`; they do not create extra case IDs. Cases blocked by failed fixtures are explicit failures. Case/fixture outcomes are fsynced to an append-only journal before completion so a recovered node can retain known test failures when considering an infrastructure retry. Malformed or unsupported recovery evidence is not permission to retry.

`summary.json` contains `version`, `session_id`, `digest`, `code`, `planned`, `reported`, outcome counts, missing/unexpected IDs, and measured durations. Host success requires both a valid summary and matching normal process exits. Group receipts are internal to the tested checkout, not a cross-repository Python API.
