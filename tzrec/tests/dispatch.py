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
"""Dispatch exact unittest groups into isolated CPU slots or accelerator pairs."""

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
        key = (case["route"], case["split"], boundary)
        group = groups.setdefault(key, dict(cases=[], devices=case["devices"]))
        group["cases"].append(case)
        if case["devices"] == "baseline":
            group["devices"] = "baseline"
    return list(groups.values())


def cpu_shards(groups: list[dict], count: int) -> list[dict]:
    """Stripe the groups of each module across CPU slots without splitting fixtures."""
    shards: list[list[dict]] = [[] for _ in range(count)]
    for index, group in enumerate(groups):
        shards[index % count].extend(group["cases"])
    return [dict(cases=cases, devices=0) for cases in shards if cases]


def device_groups(value: str, visible: list[str]) -> list[list[str]]:
    """Resolve disjoint pairs against the parent's allowed CUDA device view."""
    groups = [[int(x) for x in group.split(",")] for group in value.split(";")]
    flat = [x for group in groups for x in group]
    if any(len(g) != 2 for g in groups) or len(flat) != len(set(flat)):
        raise ValueError("device groups must be disjoint pairs")
    if any(x < 0 or x >= len(visible) for x in flat):
        raise ValueError("device index outside the parent's visible set")
    return [[visible[i] for i in group] for group in groups]


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
    slots: list[tuple[Path, dict]],
    session: Path,
    deadline: float,
    test_dir: str,
) -> list[dict]:
    """Start run.py directly, assigning another group when a slot becomes free."""
    queue = deque(groups)
    active: dict[int, tuple] = {}
    reports = []
    try:
        while queue or active:
            if time.monotonic() >= deadline:
                raise TimeoutError("test job deadline exceeded")
            for slot, (checkout, env) in enumerate(slots):
                if slot in active:
                    process, group, log = active[slot]
                    code = process.poll()
                    if code is None:
                        continue
                    run.terminate(process)
                    log.close()
                    del active[slot]
                    path = session / "results" / (group["key"] + ".json")
                    if not path.is_file():
                        raise RuntimeError(
                            f"group {group['key']} exited {code} without a result; "
                            f"see {log.name}"
                        )
                    report = run.read_json(path)
                    check_result(group, report, code)
                    reports.append(report)
                if queue:
                    group = queue.popleft()
                    group_file = session / "groups" / (group["key"] + ".json")
                    output = session / "results" / group_file.name
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
                                str(output),
                                "--test_dir",
                                test_dir,
                            ],
                            cwd=checkout,
                            env=env,
                            start_new_session=True,
                            stdout=log,
                            stderr=log,
                        )
                    except BaseException:
                        log.close()
                        raise
                    active[slot] = (process, group, log)
            if active:
                time.sleep(0.1)
        return reports
    finally:
        for process, _, _ in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, _, log in active.values():
            run.terminate(process)
            log.close()


def dispatch(args: argparse.Namespace) -> int:
    """Execute the baseline first, then overlap only certified independent groups."""
    cpu = args.hardware == "cpu"
    if cpu:
        if args.workers < 1 or args.device_groups or args.scope:
            raise ValueError(
                "CPU requires positive slots and full, device-free discovery"
            )
        pairs = [[] for _ in range(args.workers)]
    else:
        import torch

        count = torch.cuda.device_count()
        visible = (
            os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(map(str, range(count)))
        ).split(",")[:count]
        if not args.device_groups:
            raise ValueError("accelerator lanes require disjoint device pairs")
        pairs = device_groups(args.device_groups, visible)
    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    if any(session.iterdir()):
        raise ValueError("use an empty report directory")
    (session / "logs").mkdir()
    (session / "results").mkdir()
    deadline = time.monotonic() + args.timeout
    started = time.monotonic()
    env = dict(os.environ, PYTHONPATH=".")
    if cpu:
        env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    elif args.hardware == "ppu":
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pairs[0])
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
            command, timeout=max(0.01, deadline - time.monotonic()), env=env
        )
        if code:
            raise RuntimeError(f"discovery exited with code {code}")
        cases = run.read_json(output)
    groups = group_cases(cases)
    baseline = [g for g in groups if g["devices"] == "baseline"]
    queued = [g for g in groups if g["devices"] != "baseline"]
    if cpu:
        baseline = (
            [dict(cases=[c for g in baseline for c in g["cases"]], devices="baseline")]
            if baseline
            else []
        )
        queued = cpu_shards(queued, len(pairs))
    plan = baseline + queued
    identity = uuid.uuid4().hex
    for i, group in enumerate(plan):
        group.update(key=str(i), session_id=identity)
        if cpu or group["devices"] != "baseline":
            group["visible_devices"] = 0 if cpu else 2
    run.write_json(session / "plan.json", dict(session_id=identity, groups=plan))
    reports = schedule(baseline, [(Path.cwd(), env)], session, deadline, args.test_dir)
    if queued:
        Path("tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=Path("tmp").resolve(), prefix="ci-slots-"
        ) as directory:
            slots = []
            for i, pair in enumerate(pairs[: len(queued)]):
                checkout = Path(directory) / str(i)
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
                slot_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(pair))
                if cpu:
                    slot_env.update(
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                    )
                for variable, kind in (
                    ("TRITON_CACHE_DIR", "triton"),
                    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                ):
                    cache = (
                        Path(env.get(variable, str(Path(directory) / kind)))
                        / f"worker-{i}"
                    )
                    cache.mkdir(parents=True, exist_ok=True)
                    slot_env[variable] = str(cache.resolve())
                slots.append((checkout, slot_env))
            reports += schedule(queued, slots, session, deadline, args.test_dir)
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device-groups")
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
