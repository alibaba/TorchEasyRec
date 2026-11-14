# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dispatch unittest groups to CPU batches or a pool of accelerator cards."""

import argparse
import importlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from tzrec.tests import run
else:
    run = importlib.import_module("run")


def group_cases(cases: list[dict]) -> list[dict]:
    """Group at certified fixture boundaries, retaining discovery order."""
    groups: dict[tuple, dict] = {}
    for case in cases:
        boundary = case["module"]
        if case["split"] == "class":
            boundary = case["id"].rsplit(".", 1)[0]
        elif case["split"] == "case":
            boundary = case["id"]
        key = (case["split"], boundary)
        group = groups.setdefault(key, dict(cases=[], slots=case["slots"]))
        group["cases"].append(case)
        if group["slots"] is not None:
            group["slots"] = (
                None if case["slots"] is None else max(group["slots"], case["slots"])
            )
    return list(groups.values())


def cpu_shards(groups: list[dict], count: int) -> list[dict]:
    """Distribute certified fixture boundaries across CPU slots in rotation."""
    shards: list[list[dict]] = [[] for _ in range(count)]
    for index, group in enumerate(groups):
        shards[index % count].extend(group["cases"])
    return [dict(cases=cases, slots=2) for cases in shards if cases]


def visible_pool(mask: str | None, count: int, capacity: int) -> list[str]:
    """Preserve the parent CUDA IDs; never widen an empty or restricted mask."""
    cards = mask.split(",") if mask is not None else list(map(str, range(count)))
    if (
        count != capacity
        or len(cards) != capacity
        or len(set(cards)) != capacity
        or not all(cards)
    ):
        raise ValueError(
            f"expected {capacity} distinct visible cards, got {count}: {cards}"
        )
    return cards


def check_result(group: dict, report: dict, code: int) -> None:
    """Require exactly one valid outcome per planned case and a matching exit."""
    expected = Counter(c["id"] for c in group["cases"])
    observed = Counter(c["id"] for c in report["cases"])
    outcomes = {c["outcome"] for c in report["cases"]}
    failures = {"failure", "error", "unexpected_success", "blocked_by_fixture"}
    failed = bool(outcomes & failures) or any(
        e["outcome"] != "skip" for e in report["fixtures"]
    )
    if (
        type(report["code"]) is not int
        or code != report["code"]
        or code != int(failed)
        or report["group"] != group["key"]
        or report["session_id"] != group["session_id"]
        or not outcomes <= failures | {"success", "skip", "expected_failure"}
        or expected != observed
        or any(n != 1 for n in expected.values())
    ):
        raise RuntimeError(f"incomplete or inconsistent group: {group['key']}")


def schedule(
    groups: list[dict],
    cards: list[str],
    capacity: int,
    env: dict,
    directory: Path,
    session: Path,
    deadline: float,
    test_dir: str,
) -> list[dict]:
    """Lease slots in FIFO order and recycle them only after child cleanup."""
    queue = deque(groups)
    free = set(range(capacity))
    active: dict[subprocess.Popen, tuple] = {}
    reports = []
    try:
        while queue or active:
            if time.monotonic() >= deadline:
                raise TimeoutError("test job deadline exceeded")
            for process, (group, lease, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                run.terminate(process)
                log.close()
                path = session / "results" / (group["key"] + ".json")
                if not path.is_file():
                    raise RuntimeError(
                        f"group {group['key']} exited {code} without a result; "
                        f"see {log.name}"
                    )
                report = run.read_json(path)
                check_result(group, report, code)
                reports.append(report)
                del active[process]
                free.update(lease)
            while queue:
                group = queue[0]
                units = group["slots"] or capacity
                if len(free) < units:
                    break
                queue.popleft()
                lease = sorted(free)[:units]
                free.difference_update(lease)
                selected = [cards[i] for i in lease[: group["visible_devices"]]]
                child_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(selected))
                checkout = Path.cwd()
                if group["slots"] is not None:
                    worker = lease[0]
                    checkout = directory / str(worker)
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
                    for variable, kind in (
                        ("TRITON_CACHE_DIR", "triton"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                    ):
                        cache = (
                            Path(env.get(variable, str(directory / kind)))
                            / f"worker-{worker}"
                        )
                        cache.mkdir(parents=True, exist_ok=True)
                        child_env[variable] = str(cache.resolve())
                group["assigned_devices"] = selected
                group_file = session / "groups" / (group["key"] + ".json")
                run.write_json(group_file, group)
                log = (session / "logs" / (group["key"] + ".log")).open("w")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "tzrec/tests/run.py",
                            "--group-file",
                            str(group_file),
                            "--output",
                            str(session / "results" / group_file.name),
                            "--test_dir",
                            test_dir,
                        ],
                        cwd=checkout,
                        env=child_env,
                        start_new_session=True,
                        stdout=log,
                        stderr=log,
                    )
                except BaseException:
                    log.close()
                    raise
                active[process] = (group, lease, log)
                print(
                    f"Starting group {group['key']}: slots={lease} cards={selected}",
                    flush=True,
                )
            if active:
                time.sleep(0.1)
    finally:
        for process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, (_, _, log) in active.items():
            run.terminate(process)
            log.close()
    return reports


def dispatch(args: argparse.Namespace) -> int:
    """Discover once in the lane's original view, then schedule the full plan."""
    cpu = args.hardware == "cpu"
    capacity = args.capacity
    if capacity < 1 or (cpu and (args.scope or capacity % 2)):
        raise ValueError(
            "capacity must be positive; CPU uses two slots per full-suite batch"
        )
    env = dict(os.environ, PYTHONPATH=".")
    cards = []
    if cpu:
        env.update(
            CUDA_VISIBLE_DEVICES="",
            NVIDIA_VISIBLE_DEVICES="void",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
    else:
        import torch

        cards = visible_pool(
            env.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count(), capacity
        )
    baseline_view = cards[:2] if args.hardware == "ppu" else cards
    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    if any(session.iterdir()):
        raise ValueError("use an empty report directory")
    (session / "logs").mkdir()
    (session / "results").mkdir()
    started = time.monotonic()
    deadline = started + args.timeout
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "discovery.json"
        command = [
            sys.executable,
            "tzrec/tests/run.py",
            "--discover-to",
            str(output),
            "--pattern",
            args.pattern,
            "--test_dir",
            args.test_dir,
        ]
        if not cpu:
            command.extend(["--scope", args.scope or "gpu"])
        code = run.run_process(
            command,
            timeout=max(0.01, deadline - time.monotonic()),
            env=dict(env, CUDA_VISIBLE_DEVICES=",".join(baseline_view)),
        )
        if code:
            raise RuntimeError(f"discovery exited with code {code}")
        cases = run.read_json(output)
    groups = group_cases(cases)
    plan = (
        cpu_shards(groups, capacity // 2)
        if cpu
        else sorted(groups, key=lambda g: g["slots"] is not None)
    )
    identity = uuid.uuid4().hex
    for i, group in enumerate(plan):
        if group["slots"] and group["slots"] > capacity:
            raise ValueError("test group exceeds lane capacity")
        group.update(key=str(i), session_id=identity)
        group["visible_devices"] = (
            (group["slots"] or len(baseline_view)) if cards else 0
        )
    counts = Counter(str(g["slots"] or "baseline") for g in plan)
    run.write_json(
        session / "plan.json",
        dict(session_id=identity, capacity=capacity, slot_counts=counts, groups=plan),
    )
    print(f"Capacity {capacity}, groups by slot demand: {dict(counts)}", flush=True)
    Path("tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=Path("tmp").resolve(), prefix="ci-slots-"
    ) as directory:
        reports = schedule(
            plan,
            cards,
            capacity,
            env,
            Path(directory),
            session,
            deadline,
            args.test_dir,
        )
    expected = Counter(c["id"] for c in cases)
    actual = Counter(c["id"] for report in reports for c in report["cases"])
    if actual != expected or any(n != 1 for n in expected.values()):
        raise RuntimeError("missing, unexpected or duplicate case results")
    code = max(report["code"] for report in reports)
    summary = dict(
        session_id=identity,
        planned=sum(expected.values()),
        reported=sum(actual.values()),
        code=code,
        outcomes=dict(Counter(c["outcome"] for r in reports for c in r["cases"])),
        seconds=time.monotonic() - started,
    )
    run.write_json(session / "summary.json", summary)
    return code


def main() -> int:
    """Run all selected tests within one job-wide deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--hardware", choices=("cpu", "h20", "ppu"), required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--test-dir", default="tzrec")
    parser.add_argument("--pattern", default="*_test.py")
    parser.add_argument("--scope", choices=("gpu", "h20"))
    parser.add_argument("--timeout", type=float, default=21600)

    def cancel(signum: int, frame: Any) -> None:
        """Unwind through owned process cleanup when GitHub cancels the job."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    return dispatch(parser.parse_args())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
