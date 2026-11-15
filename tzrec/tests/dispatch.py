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
"""Allocate slots and collect native unittest results in discovery order."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import TextIO


def group_cases(cases: list[dict]) -> list[dict]:
    """Keep each declared fixture boundary together and reserve its peak slots."""
    groups: dict[str, dict] = {}
    for case in cases:
        split = case["split"]
        key = (
            case["module"]
            if split == "module"
            else case["id"].rsplit(".", 1)[0]
            if split == "class"
            else case["id"]
        )
        group = groups.setdefault(
            key,
            dict(name=key, ids=[], slots=1, env=case["env"], timeout=case["timeout"]),
        )
        if (group["env"], group["timeout"]) != (case["env"], case["timeout"]):
            raise ValueError(f"incompatible environment or timeout within {key}")
        group["slots"] = max(group["slots"], case["slots"])
        group["ids"].append(case["id"])
    return list(groups.values())


def terminate(process: subprocess.Popen) -> None:
    """Reap an owned process group, including subprocesses left by a test."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def schedule(
    cases: list[dict],
    capacity: int,
    device_ids: list[str],
    test_dir: str,
    started: float,
    timeout: float,
) -> int:
    """Run groups FIFO, then print their failures and one suite-wide summary."""
    groups = group_cases(cases)
    if any(g["slots"] > capacity for g in groups):
        raise ValueError("test group exceeds lane capacity")
    counts: Counter[str] = Counter()
    failed_groups: set[int] = set()
    problems: dict[int, str] = {}
    active: dict[subprocess.Popen, tuple[int, list[int], TextIO, float]] = {}
    free = set(range(capacity))
    index = 0
    with (
        tempfile.TemporaryDirectory(dir=Path("tmp").resolve(), prefix="ci-") as tmp,
        tempfile.TemporaryDirectory(dir="/tmp", prefix="tzrec-") as alias,
    ):
        scratch = Path(tmp)
        # multiprocessing creates Unix sockets under TMPDIR with a short path limit.
        short_tmp = Path(alias) / "groups"
        short_tmp.symlink_to(scratch, target_is_directory=True)
        failure_logs = scratch.parent / "ci-failures" / scratch.name
        try:
            while index < len(groups) or active:
                if time.monotonic() >= started + timeout:
                    if index < len(groups):
                        problems[index] = f"job timeout; {len(groups) - index} pending"
                    for i, _, _, _ in active.values():
                        problems[i] = (
                            f"{groups[i]['name']}: interrupted at job deadline"
                        )
                    break
                for proc, (i, lease, log, end) in list(active.items()):
                    rc = proc.poll()
                    if rc is None and time.monotonic() < end:
                        continue
                    terminate(proc)
                    log.close()
                    del active[proc]
                    free.update(lease)
                    try:
                        if rc is None:
                            raise ValueError(f"timed out after {groups[i]['timeout']}s")
                        result = json.loads((scratch / f"{i}.counts").read_text())
                        failed = any(
                            result[k]
                            for k in ("failures", "errors", "unexpected successes")
                        )
                        if rc != int(failed):
                            raise ValueError("exit status disagrees with unittest")
                        counts.update(result)
                        if failed:
                            failed_groups.add(i)
                    except (OSError, ValueError, KeyError) as exc:
                        problems[i] = f"{groups[i]['name']}: {exc} (exit {rc})"
                while index < len(groups) and len(free) >= groups[index]["slots"]:
                    i = index
                    group = groups[i]
                    index += 1
                    lease = sorted(free)[: group["slots"]]
                    group_tmp = scratch / str(i)
                    group_tmp.mkdir()
                    assigned_device_ids = (
                        [device_ids[j] for j in lease] if device_ids else []
                    )
                    env = dict(os.environ, **group["env"])
                    env.update(
                        PYTHONPATH=".",
                        TZREC_TEST_TMPDIR=str(group_tmp),
                        TMPDIR=str(short_tmp / str(i)),
                        CUDA_VISIBLE_DEVICES=",".join(assigned_device_ids),
                    )
                    request = scratch / f"{i}.group"
                    request.write_text(
                        json.dumps(
                            dict(
                                ids=group["ids"],
                                test_dir=test_dir,
                                device_count=len(assigned_device_ids),
                            )
                        )
                    )
                    log = (scratch / f"{i}.log").open("w")
                    try:
                        proc = subprocess.Popen(
                            [
                                sys.executable,
                                str(Path(__file__).with_name("run_group.py").resolve()),
                                str(request),
                            ],
                            env=env,
                            start_new_session=True,
                            stdout=log,
                            stderr=log,
                        )
                    except OSError as exc:
                        log.close()
                        problems[i] = f"{group['name']}: could not start: {exc}"
                        continue
                    free.difference_update(lease)
                    active[proc] = (
                        i,
                        lease,
                        log,
                        time.monotonic() + (group["timeout"] or timeout),
                    )
                    print(
                        f"Group {i}: {group['name']} slots={lease} "
                        f"device_ids={assigned_device_ids}",
                        flush=True,
                    )
                if active:
                    time.sleep(0.1)
        finally:
            for proc in active:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for proc, (_, _, log, _) in active.items():
                terminate(proc)
                log.close()

            for i in sorted(failed_groups | problems.keys()):
                log_path = scratch / f"{i}.log"
                if not log_path.exists():
                    continue
                failure_logs.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as stream:
                    if i in problems:
                        print(f"\nGROUP ERROR: {problems[i]}", file=stream)
                    else:
                        with (scratch / f"{i}.errors").open() as errors:
                            shutil.copyfileobj(errors, stream)
                log_path.replace(failure_logs / f"{i}.log")
            if failure_logs.exists():
                print(f"Failed group logs: {failure_logs}", file=sys.stderr)

        return _print_summary(
            failure_logs, scratch, len(groups), counts, problems, started
        )


def _print_summary(
    failure_logs: Path,
    scratch: Path,
    group_count: int,
    counts: Counter[str],
    problems: dict[int, str],
    started: float,
) -> int:
    """Replay native failure blocks and report totals for completed groups."""
    for i in range(group_count + 1):
        if i in problems:
            path = failure_logs / f"{i}.log"
            if not path.exists():
                print(f"GROUP ERROR: {problems[i]}", file=sys.stderr)
        else:
            path = scratch / f"{i}.errors"
        if path.exists():
            with path.open(errors="replace") as stream:
                shutil.copyfileobj(stream, sys.stderr)
    failed = bool(
        problems
        or counts["errors"]
        or counts["failures"]
        or counts["unexpected successes"]
    )
    if problems:
        counts["group errors"] = len(problems)
    stats = ", ".join(
        f"{k}={counts[k]}"
        for k in (
            "failures",
            "errors",
            "skipped",
            "expected failures",
            "unexpected successes",
            "group errors",
        )
        if counts[k]
    )
    n = counts["tests"]
    elapsed = time.monotonic() - started
    print("-" * 70, file=sys.stderr)
    print(
        f"Ran {n} test{'s' if n != 1 else ''} in {elapsed:.3f}s\n",
        file=sys.stderr,
    )
    print(
        ("FAILED" if failed else "OK") + (f" ({stats})" if stats else ""),
        file=sys.stderr,
    )
    return int(failed)
