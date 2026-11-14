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

"""Dispatch unittest groups locally or to externally isolated CI workers."""

import argparse
import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from tzrec.tests import ci_execution as execution
    from tzrec.tests import ci_protocol as wire
else:
    execution = importlib.import_module("ci_execution")
    wire = importlib.import_module("ci_protocol")


def baseline(args: argparse.Namespace) -> int:
    """Discover once and complete groups requiring the original device view."""
    from tzrec.tests.worker import run_group

    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    if (session / "manifest.json").exists():
        raise ValueError("session already contains a manifest; use a fresh session")
    for name in ("results", "logs"):
        (session / name).mkdir(exist_ok=True)
    durations = wire.read_json(args.durations) if args.durations else {}
    cases = execution.discover(args.test_dir, args.pattern, args.scope)
    manifest = dict(
        version=wire.VERSION,
        session_id=args.session_id,
        scope=args.scope,
        test_dir=args.test_dir,
        pattern=args.pattern,
        deadline=time.time() + args.timeout,
        groups=execution.group_cases(cases, durations),
    )
    manifest["digest"] = wire.digest(manifest)
    wire.write_json(session / "manifest.json", manifest)
    for group in manifest["groups"]:
        if group["devices"] == "baseline":
            run_group(session, manifest, group, args.test_dir)
    pairs = any(g["devices"] != "baseline" for g in manifest["groups"])
    wire.write_json(
        session / "phase.json",
        dict(
            version=wire.VERSION,
            session_id=args.session_id,
            next="pairs" if pairs else "finish",
        ),
    )
    return 0


def serve(session: Path, workers: int, session_id: str) -> int:
    """Assign each group once and reject incomplete worker results."""
    manifest = wire.read_manifest(session)
    if manifest["version"] != wire.VERSION or manifest["session_id"] != session_id:
        raise ValueError("manifest identity mismatch")
    pairs = [g for g in manifest["groups"] if g["devices"] != "baseline"]
    if workers < 0 or (pairs and workers == 0):
        raise ValueError("no workers for pending device groups")
    queue = deque(sorted(pairs, key=lambda g: (-g["estimate"], g["key"])))
    seen: set[int] = set()
    completed: set[str] = set()
    errors: list[str] = []
    lock = threading.Lock()
    threads = []
    sockets = []
    server = wire.session_socket(session, listen=True)

    def handle(sock: socket.socket) -> None:
        """Keep a single worker's assignments and receipts in order."""
        try:
            sock.settimeout(max(0.01, manifest["deadline"] - time.time()))
            with sock.makefile("rwb") as stream:
                message = wire.receive(stream)
                slot = message.get("slot")
                with lock:
                    if (
                        message.get("kind") != "ready"
                        or message.get("version") != wire.VERSION
                        or message.get("session_id") != session_id
                        or message.get("digest") != manifest["digest"]
                        or type(slot) is not int
                        or slot not in range(workers)
                        or slot in seen
                        or message.get("devices", 0) < 2
                    ):
                        raise ValueError("invalid or duplicate worker registration")
                    seen.add(slot)
                while True:
                    with lock:
                        if errors:
                            raise RuntimeError("another worker failed")
                        group = queue.popleft() if queue else None
                    if group is None:
                        wire.send(stream, dict(kind="stop"))
                        return
                    wire.send(
                        stream,
                        dict(kind="run", group=group["key"], digest=manifest["digest"]),
                    )
                    message = wire.receive(stream)
                    if message != dict(
                        kind="done", group=group["key"], digest=manifest["digest"]
                    ):
                        raise ValueError("receipt does not match assignment")
                    report = wire.read_json(
                        session / "results" / f"{group['key']}.json"
                    )
                    check = execution.verify(dict(manifest, groups=[group]), [report])
                    if check["code"] == 2:
                        raise ValueError("invalid group receipt")
                    with lock:
                        completed.add(group["key"])
                    print(
                        f"Completed {group['key']} ({len(group['cases'])} cases)",
                        flush=True,
                    )
        except Exception as error:
            with lock:
                errors.append(str(error))
        finally:
            sock.close()

    try:
        server.settimeout(min(120, max(0.01, manifest["deadline"] - time.time())))
        for _ in range(workers):
            sock, _ = server.accept()
            sockets.append(sock)
            thread = threading.Thread(target=handle, args=(sock,), daemon=True)
            threads.append(thread)
            thread.start()
        while any(t.is_alive() for t in threads):
            if errors or time.time() >= manifest["deadline"]:
                raise RuntimeError("; ".join(errors) or "job deadline exceeded")
            for thread in threads:
                thread.join(timeout=0.1)
        if errors or seen != set(range(workers)) or len(completed) != len(pairs):
            raise RuntimeError("; ".join(errors) or "workers did not finish the plan")
        reports = [
            wire.read_json(session / "results" / f"{g['key']}.json")
            for g in manifest["groups"]
        ]
        summary = execution.verify(manifest, reports)
        summary["durations"] = {
            c["id"]: c["seconds"] for r in reports for c in r["cases"]
        }
        wire.write_json(session / "summary.json", summary)
        return summary["code"]
    finally:
        server.close()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        (session / "dispatch.sock").unlink(missing_ok=True)


def device_groups(value: str, visible: list[str]) -> list[list[str]]:
    """Resolve disjoint pairs against the parent's allowed CUDA device view."""
    groups = [[int(x) for x in group.split(",")] for group in value.split(";")]
    flat = [x for group in groups for x in group]
    if any(len(g) != 2 for g in groups) or len(flat) != len(set(flat)):
        raise ValueError("device groups must be disjoint pairs")
    if any(x < 0 or x >= len(visible) for x in flat):
        raise ValueError("device index outside the parent's visible set")
    return [[visible[i] for i in group] for group in groups]


def local(args: argparse.Namespace) -> int:
    """Run the same protocol with fresh processes on a local CUDA machine."""
    import torch

    count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        visible = [str(i) for i in range(count)]
    visible = visible[:count]
    pairs = device_groups(args.device_groups, visible)
    from tzrec.tests import ci_devices

    buses = ci_devices.visible_buses()
    expected = {device: buses[index] for index, device in enumerate(visible)}
    baseline_env = dict(os.environ)
    if args.hardware == "ppu":
        baseline_env["CUDA_VISIBLE_DEVICES"] = ",".join(pairs[0])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "baseline",
        "--session",
        str(args.session),
        "--session-id",
        args.session_id,
        "--test-dir",
        args.test_dir,
        "--pattern",
        args.pattern,
        "--timeout",
        str(args.timeout),
    ]
    if args.scope:
        command += ["--scope", args.scope]
    if args.durations:
        command += ["--durations", str(args.durations)]
    code = wire.run_process(command, timeout=args.timeout, env=baseline_env)
    if code:
        return code
    session = args.session.resolve()
    manifest = wire.read_manifest(session)
    if not any(g["devices"] != "baseline" for g in manifest["groups"]):
        return serve(session, 0, args.session_id)
    Path("tmp").mkdir(exist_ok=True)
    workdirs = tempfile.TemporaryDirectory(
        dir=Path("tmp").resolve(), prefix="ci-workers-"
    )
    processes: list[subprocess.Popen] = []
    streams = []
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(signum: int, frame: Any) -> None:
        """Clean all workers when the local job is cancelled."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        coordinator = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "serve",
                "--session",
                str(session),
                "--session-id",
                args.session_id,
                "--protocol",
                str(wire.VERSION),
                "--workers",
                str(len(pairs)),
            ],
            start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        processes.append(coordinator)
        source = Path.cwd()
        for slot, pair in enumerate(pairs):
            checkout = Path(workdirs.name) / f"checkout-{slot}"
            shutil.copytree(
                source,
                checkout,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "tmp",
                    "experiments",
                    ".hypothesis",
                    "__pycache__",
                    ".ruff_cache",
                ),
            )
            log = (session / f"worker-{slot}.log").open("w")
            streams.append(log)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(pair),
                TZREC_CI_EXPECTED_BUSES=json.dumps(
                    [expected[device] for device in pair]
                ),
                TZREC_CI_HARDWARE=args.hardware,
                PYTHONPATH=".",
            )
            for variable, kind in (
                ("TRITON_CACHE_DIR", "triton"),
                ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ):
                cache = (
                    Path(env.get(variable, str(Path(workdirs.name) / kind)))
                    / f"worker-{slot}"
                )
                cache.mkdir(parents=True, exist_ok=True)
                env[variable] = str(cache.resolve())
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "tzrec/tests/worker.py",
                        "serve",
                        "--session",
                        str(session),
                        "--protocol",
                        str(wire.VERSION),
                        "--slot",
                        str(slot),
                        "--test-dir",
                        args.test_dir,
                    ],
                    cwd=checkout,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            )
        while coordinator.poll() is None:
            if any(p.poll() not in (None, 0) for p in processes[1:]):
                raise RuntimeError("worker process failed")
            if time.time() >= manifest["deadline"]:
                raise TimeoutError("job deadline exceeded")
            time.sleep(0.2)
        for process in processes[1:]:
            if process.wait(timeout=15):
                raise RuntimeError("worker exit disagrees with coordinator")
        return coordinator.returncode
    finally:
        for process in reversed(processes):
            wire.terminate(process)
        for stream in streams:
            stream.close()
        signal.signal(signal.SIGTERM, previous)
        (session / "dispatch.sock").unlink(missing_ok=True)
        workdirs.cleanup()


def main() -> int:
    """Expose the versioned CLI without depending on the hosting platform."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("baseline", "serve", "local"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--protocol", type=int, choices=(wire.VERSION,), default=wire.VERSION
    )
    parser.add_argument("--scope", choices=("gpu", "h20"))
    parser.add_argument("--test-dir", default="tzrec")
    parser.add_argument("--pattern", default="*_test.py")
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--durations", type=Path)
    parser.add_argument("--device-groups")
    parser.add_argument("--hardware", choices=("h20", "ppu"), default="h20")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.command != "local" and (not args.session or not args.session_id):
        parser.error("baseline/serve require --session and --session-id")
    args.session_id = args.session_id or uuid.uuid4().hex
    args.session = (
        args.session or Path("tmp") / ("dispatch-" + args.session_id)
    ).resolve()
    if args.command == "local":
        if not args.device_groups:
            parser.error("local requires --device-groups")
        return local(args)
    if args.command == "baseline":
        return baseline(args)
    return serve(args.session, args.workers, args.session_id)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        traceback.print_exc()
        raise SystemExit(2) from None
