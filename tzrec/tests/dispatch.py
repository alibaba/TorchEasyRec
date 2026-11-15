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
    session: Path,
    test_dir: str,
    started: float,
    timeout: float,
) -> int:
    """Run groups FIFO, then print their failures and one suite-wide summary."""
    groups = group_cases(cases)
    if any(g["slots"] > capacity for g in groups):
        raise ValueError("test group exceeds lane capacity")
    counts: Counter[str] = Counter()
    problems: dict[int, str] = {}
    active: dict[subprocess.Popen, tuple[int, list[int], TextIO, float]] = {}
    free = set(range(capacity))
    index = 0
    with tempfile.TemporaryDirectory(
        dir=Path("tmp").resolve(), prefix="ci-slots-"
    ) as tmp:
        scratch = Path(tmp)
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
                    except (OSError, ValueError, KeyError) as exc:
                        problems[i] = f"{groups[i]['name']}: {exc} (exit {rc})"
                while index < len(groups) and len(free) >= groups[index]["slots"]:
                    i = index
                    group = groups[i]
                    index += 1
                    lease = sorted(free)[: group["slots"]]
                    checkout = scratch / str(lease[0])
                    if not checkout.exists():
                        shutil.copytree(
                            Path.cwd(),
                            checkout,
                            symlinks=True,
                            ignore=shutil.ignore_patterns(
                                ".git",
                                "tmp",
                                "experiments",
                                "__pycache__",
                                ".hypothesis",
                                ".ruff_cache",
                            ),
                        )
                    assigned_device_ids = (
                        [device_ids[j] for j in lease] if device_ids else []
                    )
                    env = dict(os.environ, **group["env"])
                    env.update(
                        PYTHONPATH=".",
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
                    log = (session / f"{i}.log").open("w")
                    try:
                        proc = subprocess.Popen(
                            [
                                sys.executable,
                                str(Path(__file__).with_name("run_group.py").resolve()),
                                str(request),
                            ],
                            cwd=checkout,
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

        return _print_summary(session, scratch, len(groups), counts, problems, started)


def _print_summary(
    session: Path,
    scratch: Path,
    group_count: int,
    counts: Counter[str],
    problems: dict[int, str],
    started: float,
) -> int:
    """Replay native failure blocks and report totals for completed groups."""
    with (session / "summary.txt").open("w") as output:
        for i in range(group_count + 1):
            if i in problems:
                print(f"GROUP ERROR: {problems[i]}", file=output)
                path = session / f"{i}.log"
            else:
                path = scratch / f"{i}.errors"
            if path.exists():
                with path.open(errors="replace") as stream:
                    shutil.copyfileobj(stream, output)
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
        print("-" * 70, file=output)
        print(
            f"Ran {n} test{'s' if n != 1 else ''} in {elapsed:.3f}s\n",
            file=output,
        )
        print(
            ("FAILED" if failed else "OK") + (f" ({stats})" if stats else ""),
            file=output,
        )
    with (session / "summary.txt").open() as summary:
        shutil.copyfileobj(summary, sys.stderr)
    return int(failed)
