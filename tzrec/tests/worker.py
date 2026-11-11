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

"""Run a certified test group in an already isolated device view."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from tzrec.tests import ci_execution as execution
from tzrec.tests import ci_protocol as wire


def run_group(session: Path, manifest: dict, group: dict, test_dir: str) -> dict:
    """Execute a group in a fresh process and require a complete receipt."""
    key = group["key"]
    report_path = session / "results" / f"{key}.json"
    if report_path.exists():
        raise ValueError(f"group already executed: {key}")
    log_path = session / "logs" / f"{key}.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("w") as log:
        code = wire.run_process(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "group",
                "--session",
                str(session),
                "--protocol",
                str(wire.VERSION),
                "--key",
                key,
                "--test-dir",
                test_dir,
            ],
            timeout=manifest["deadline"] - time.time(),
            env=dict(
                os.environ,
                TZREC_CI_EVENT_LOG=str(session / "results" / f"{key}.events.jsonl"),
            ),
            stdout=log,
            stderr=log,
        )
    if not report_path.exists():
        raise RuntimeError(
            f"{key}: worker exited {code} without a receipt; see {log_path}"
        )
    report = wire.read_json(report_path)
    if code != report["code"] or code not in (0, 1):
        raise RuntimeError(f"{key}: abnormal group exit {code}; see {log_path}")
    return report


def execute_group(session: Path, manifest: dict, group: dict, test_dir: str) -> int:
    """Run main suites or the repository's per-case subprocess route."""
    report: dict[str, Any]
    if group["route"] == "main":
        report = execution.execute(group, test_dir)
    else:
        reports = []
        for index, case in enumerate(group["cases"]):
            output = session / "results" / f"{group['key']}-{index}.json"
            code = wire.run_process(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "case",
                    "--session",
                    str(session),
                    "--protocol",
                    str(wire.VERSION),
                    "--key",
                    group["key"],
                    "--index",
                    str(index),
                    "--test-dir",
                    test_dir,
                ],
                timeout=min(600, manifest["deadline"] - time.time()),
                env=execution.child_environment(case, test_dir),
            )
            one = wire.read_json(output)
            if code != one["code"] or code not in (0, 1):
                raise RuntimeError(
                    f"subprocess failed without valid results: {case['id']}"
                )
            reports.append(one)
        report = dict(
            code=max(r["code"] for r in reports),
            cases=[c for r in reports for c in r["cases"]],
            fixtures=[c for r in reports for c in r["fixtures"]],
        )
    report.update(version=wire.VERSION, digest=manifest["digest"], group=group["key"])
    wire.write_json(session / "results" / f"{group['key']}.json", report)
    return report["code"]


def serve(session: Path, slot: int, test_dir: str) -> int:
    """Request work from the coordinator, never launching two groups on a slot."""
    import torch

    manifest = wire.read_manifest(session)
    if manifest["version"] != wire.VERSION:
        raise ValueError("unsupported manifest version")
    visible = torch.cuda.device_count()
    if visible != 2:
        raise ValueError("a pair worker needs exactly two visible accelerators")
    groups = {g["key"]: g for g in manifest["groups"]}
    ready_deadline = min(time.time() + 120, manifest["deadline"])
    while True:
        try:
            sock = wire.session_socket(session, listen=False)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.time() >= ready_deadline:
                raise TimeoutError("coordinator did not become ready") from None
            time.sleep(0.2)
    with sock:
        sock.settimeout(max(0.01, manifest["deadline"] - time.time()))
        with sock.makefile("rwb") as stream:
            wire.send(
                stream,
                dict(
                    kind="ready",
                    version=wire.VERSION,
                    slot=slot,
                    session_id=manifest["session_id"],
                    digest=manifest["digest"],
                    devices=visible,
                ),
            )
            while True:
                request = wire.receive(stream)
                if request["kind"] == "stop":
                    return 0
                if (
                    request["kind"] != "run"
                    or request.get("digest") != manifest["digest"]
                ):
                    raise ValueError("invalid work assignment")
                group = groups[request["group"]]
                if group["devices"] == "baseline" or group["devices"] > visible:
                    raise ValueError("work assignment does not fit the device view")
                report = run_group(session, manifest, group, test_dir)
                wire.send(
                    stream,
                    dict(kind="done", group=group["key"], digest=report["digest"]),
                )


def main() -> int:
    """Handle a worker process or a single fresh group execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "group", "case", "single"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--protocol", required=True, type=int, choices=(wire.VERSION,))
    parser.add_argument("--test-dir", default="tzrec")
    parser.add_argument("--slot", type=int)
    parser.add_argument("--key")
    parser.add_argument("--index", type=int)
    parser.add_argument("--test-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "single":
        if not args.test_id or not args.output:
            parser.error("single requires --test-id and --output")
        path = str(Path(args.test_dir).resolve())
        sys.path.insert(0, path)
        import unittest

        suite = unittest.defaultTestLoader.loadTestsFromName(args.test_id)
        found = list(execution.leaves(suite))
        if len(found) != 1 or found[0].id() != args.test_id:
            raise ValueError("single case could not be loaded")
        report = execution.execute(
            dict(cases=[dict(id=args.test_id, skip=execution.static_skip(found[0]))]),
            args.test_dir,
        )
        wire.write_json(args.output, report)
        return report["code"]
    if not args.session:
        parser.error("worker requires --session")
    session = args.session.resolve()
    if args.command == "serve":
        if args.slot is None or args.slot < 0:
            parser.error("serve requires a nonnegative --slot")
        return serve(session, args.slot, args.test_dir)
    manifest = wire.read_manifest(session)
    if manifest["version"] != wire.VERSION:
        raise ValueError("unsupported manifest version")
    group = next(g for g in manifest["groups"] if g["key"] == args.key)
    if args.command == "case":
        if args.index is None:
            parser.error("case requires --index")
        report = execution.execute(
            dict(cases=[group["cases"][args.index]]), args.test_dir
        )
        wire.write_json(
            session / "results" / f"{group['key']}-{args.index}.json", report
        )
        return report["code"]
    return execute_group(session, manifest, group, args.test_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
