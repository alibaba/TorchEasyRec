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
"""Discover the requested tests and run them with a bounded slot budget."""

import argparse
import importlib
import os
import signal
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterator

if __package__:
    from . import dispatch
else:
    dispatch = importlib.import_module("dispatch")


def leaves(suite: Any) -> Iterator[unittest.TestCase]:
    """Visit each test in discovery order."""
    if isinstance(suite, unittest.TestCase):
        yield suite
    else:
        for child in suite:
            yield from leaves(child)


def discover(test_dir: str, pattern: str, scope: str | None) -> list[dict]:
    """Discover IDs, checking import errors before applying the scope filter."""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    if loader.errors:
        raise ImportError("\n".join(map(str, loader.errors)))
    cases = []
    for case in leaves(suite):
        cls = type(case)
        method = getattr(case, case._testMethodName)
        if scope and not any(
            scope in getattr(x, "_ci_scopes", ()) for x in (cls, method)
        ):
            continue
        slots, split, timeout = getattr(
            method, "_ci_resources", getattr(cls, "_ci_resources", (1, "module", None))
        )
        if split != "module" and any(
            hasattr(sys.modules[case.__module__], name)
            for name in ("setUpModule", "tearDownModule")
        ):
            raise ValueError(f"{case.id()}: splitting crosses a module fixture")
        if split == "case" and any(
            getattr(getattr(cls, name), "__func__", None)
            != getattr(unittest.TestCase, name).__func__
            for name in ("setUpClass", "tearDownClass")
        ):
            raise ValueError(f"{case.id()}: splitting crosses a class fixture")
        cases.append(
            dict(
                id=case.id(),
                module=case.__module__,
                slots=slots,
                split=split,
                timeout=timeout,
                env=case._run_env() if hasattr(case, "_run_env") else {},
            )
        )
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("empty or duplicate test discovery")
    return cases


def main() -> int:
    """List tests or schedule the full selection, using two slots by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_tests", action="store_true")
    parser.add_argument("--test-dir", "--test_dir", default="tzrec")
    parser.add_argument("--pattern", default="*_test.py")
    parser.add_argument("--scope", choices=("gpu", "h20"))
    parser.add_argument("--hardware", choices=("cpu", "h20", "ppu", "gpu"))
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--timeout", type=float, default=21600)
    args = parser.parse_args()
    if args.capacity < 1 or args.timeout <= 0:
        parser.error("capacity and timeout must be positive")
    if args.hardware == "cpu":
        os.environ.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    try:
        with redirect_stdout(sys.stderr):
            cases = discover(args.test_dir, args.pattern, args.scope)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if args.list_tests:
        for case in cases:
            print(case["id"])
        return 0

    device_ids = []
    if args.hardware != "cpu":
        import torch

        count = torch.cuda.device_count()
        if args.hardware or count:
            if args.hardware is None:
                args.capacity = min(args.capacity, count)
            mask = os.environ.get("CUDA_VISIBLE_DEVICES")
            visible = (
                mask.split(",") if mask is not None else list(map(str, range(count)))
            )
            if (
                count < args.capacity
                or len(visible) != count
                or not all(visible)
                or len(set(visible)) != count
                or (args.hardware in ("h20", "ppu") and count != args.capacity)
            ):
                parser.error(
                    f"capacity {args.capacity} incompatible with {count} devices: "
                    f"{visible}"
                )
            device_ids = visible[: args.capacity]

    Path("tmp").mkdir(exist_ok=True)
    session = args.session or Path(tempfile.mkdtemp(dir="tmp", prefix="ci-report-"))
    session.mkdir(parents=True, exist_ok=True)
    if any(session.iterdir()):
        parser.error("use an empty report directory")
    print(f"Test logs: {session.resolve()}", flush=True)
    return dispatch.schedule(
        cases,
        args.capacity,
        device_ids,
        session.resolve(),
        args.test_dir,
        started,
        args.timeout,
    )


if __name__ == "__main__":

    def cancel(signum: int, frame: Any) -> None:
        """Interrupt the scheduler so it cleans up its active children."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
