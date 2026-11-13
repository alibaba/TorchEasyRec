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

"""Discovery, fixture boundaries, and complete unittest result accounting."""

import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING or __package__:
    from tzrec.tests import ci_protocol as wire
else:
    wire = importlib.import_module("ci_protocol")


def leaves(suite: Any) -> Iterator[unittest.TestCase]:
    """Visit leaf cases in discovery order."""
    if isinstance(suite, unittest.TestCase):
        yield suite
    else:
        for child in suite:
            yield from leaves(child)


def static_skip(case: unittest.TestCase) -> str | None:
    """Capture the effective import-time skip decision."""
    for obj in (type(case), getattr(case, case._testMethodName)):
        if getattr(obj, "__unittest_skip__", False):
            return str(getattr(obj, "__unittest_skip_why__", ""))
    return None


def discover(test_dir: str, pattern: str, scope: str | None) -> list[dict]:
    """Reuse the repository's scope and subprocess routing without parsing text."""
    from tzrec.tests import run

    unittest.defaultTestLoader.errors.clear()
    main, child = run._gather_test_cases(
        SimpleNamespace(
            test_dir=test_dir, pattern=pattern, scope=scope, list_tests=False
        )
    )
    if unittest.defaultTestLoader.errors:
        raise ValueError(
            "test discovery failed:\n"
            + "\n".join(map(str, unittest.defaultTestLoader.errors))
        )
    cases = []
    for route, suite in (("main", main), ("subproc", child)):
        for case in leaves(suite):
            cls = type(case)
            method = getattr(case, case._testMethodName)
            devices, split = getattr(
                method,
                "_ci_resources",
                getattr(cls, "_ci_resources", ("baseline", "module")),
            )
            module = sys.modules[case.__module__]
            if split != "module" and any(
                hasattr(module, name) for name in ("setUpModule", "tearDownModule")
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
                    route=route,
                    module=case.__module__,
                    devices=devices,
                    split=split,
                    skip=static_skip(case),
                )
            )
    ids = [case["id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate test discovery")
    return cases


def group_cases(cases: list[dict], durations: dict[str, float]) -> list[dict]:
    """Group cases at certified fixture boundaries, keeping their original order."""
    groups: dict[tuple, dict] = {}
    for case in cases:
        boundary = case["module"]
        if case["split"] == "class":
            boundary = case["id"].rsplit(".", 1)[0]
        elif case["split"] == "case":
            boundary = case["id"]
        key = (case["route"], case["split"], boundary)
        group = groups.setdefault(key, dict(cases=[], devices=0, route=case["route"]))
        group["cases"].append(case)
        if "baseline" in (group["devices"], case["devices"]):
            group["devices"] = "baseline"
        else:
            group["devices"] = max(group["devices"], case["devices"])
    result = []
    for group in groups.values():
        group["key"] = wire.digest(group)[:24]
        group["estimate"] = sum(
            max(0, durations.get(c["id"], 0)) for c in group["cases"]
        )
        result.append(group)
    return result


def load_cases(group: dict, test_dir: str) -> unittest.TestSuite:
    """Load exactly the discovered cases and preserve their skip decisions."""
    test_path = str(Path(test_dir).resolve())
    if test_path not in sys.path:
        sys.path.insert(0, test_path)
    suite = unittest.TestSuite()
    for spec in group["cases"]:
        one = unittest.defaultTestLoader.loadTestsFromName(spec["id"])
        found = list(leaves(one))
        if len(found) != 1 or found[0].id() != spec["id"]:
            raise ValueError(f"test disappeared: {spec['id']}")
        if static_skip(found[0]) != spec["skip"]:
            raise ValueError(f"device view changed skip decision: {spec['id']}")
        suite.addTest(one)
    return suite


class RecordingResult(unittest.TextTestResult):
    """Record one outcome per case, including subtests and fixture failures."""

    def __init__(self, *args: Any, test_dir: str = "tzrec", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.test_dir = test_dir
        self.records: dict[str, dict] = {}
        self.fixture_events: list[dict] = []
        self.started: dict[str, float] = {}

    def startTest(self, test: unittest.TestCase) -> None:
        """Start the timer before setup or the test body."""
        super().startTest(test)
        self.started[test.id()] = time.monotonic()

    def stopTest(self, test: unittest.TestCase) -> None:
        """Finalize elapsed time without inventing a success outcome."""
        if test.id() in self.records:
            self.records[test.id()]["seconds"] = (
                time.monotonic() - self.started[test.id()]
            )
        super().stopTest(test)

    def _record(self, test: Any, outcome: str, detail: str = "") -> None:
        """Merge repeated teardown/subtest events with failure precedence."""
        failure_file = os.environ.get("TZREC_CI_FAILURE_FILE")
        if failure_file and outcome in (
            "failure",
            "error",
            "unexpected_success",
            "blocked_by_fixture",
        ):
            fd = os.open(failure_file, os.O_WRONLY | os.O_CREAT, 0o600)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            directory = os.open(Path(failure_file).parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        journal = os.environ.get("TZREC_CI_EVENT_LOG")
        if journal:
            first = not Path(journal).exists()
            with open(journal, "a") as stream:
                stream.write(json.dumps({"id": test.id(), "outcome": outcome}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if first:
                directory = os.open(Path(journal).parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        if not isinstance(test, unittest.TestCase):
            self.fixture_events.append(
                dict(id=test.id(), outcome=outcome, detail=detail)
            )
            return
        priority = (
            "success",
            "skip",
            "expected_failure",
            "unexpected_success",
            "failure",
            "error",
        )
        record = self.records.setdefault(
            test.id(), dict(id=test.id(), outcome=outcome, details=[], seconds=0)
        )
        if priority.index(outcome) > priority.index(record["outcome"]):
            record["outcome"] = outcome
        if detail:
            record["details"].append(detail)

    def addSuccess(self, test: Any) -> None:
        """Record a passing case."""
        super().addSuccess(test)
        self._record(test, "success")

    def addSkip(self, test: Any, reason: str) -> None:
        """Preserve case and fixture skip reasons."""
        super().addSkip(test, reason)
        parent = getattr(test, "test_case", None)
        if parent is not None:
            self._record(parent, "success")
            self.records[parent.id()].setdefault("subtest_skips", []).append(
                dict(id=test.id(), reason=reason)
            )
        else:
            self._record(test, "skip", reason)

    def addFailure(self, test: Any, err: Any) -> None:
        """Record assertion failure details."""
        super().addFailure(test, err)
        self._record(test, "failure", self._exc_info_to_string(err, test))

    def addError(self, test: Any, err: Any) -> None:
        """Record setup, body, and teardown errors."""
        super().addError(test, err)
        self._record(test, "error", self._exc_info_to_string(err, test))

    def addExpectedFailure(self, test: Any, err: Any) -> None:
        """Preserve an expected failure as its own terminal outcome."""
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure", self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test: Any) -> None:
        """Treat an unexpected success as unsuccessful, like unittest."""
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test: Any, subtest: Any, err: Any) -> None:
        """Fold failing subtests into the parent's unique outcome."""
        super().addSubTest(test, subtest, err)
        if err:
            outcome = (
                "failure" if issubclass(err[0], test.failureException) else "error"
            )
            self._record(
                test,
                outcome,
                f"{subtest.id()}: {self._exc_info_to_string(err, subtest)}",
            )

    def run_subprocess(self, test: unittest.TestCase) -> None:
        """Preserve isolated-case environment and propagate real child outcomes."""
        self.startTest(test)
        Path("tmp").mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                dir="tmp", prefix="ci-child-"
            ) as directory:
                output = Path(directory).resolve() / "result.json"
                spec = dict(id=test.id(), skip=static_skip(test))
                env = dict(os.environ)
                if hasattr(test, "_run_env"):
                    env.update(test._run_env())
                preserve_execution_environment(env)
                code = wire.run_process(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("worker.py")),
                        "single",
                        "--protocol",
                        str(wire.VERSION),
                        "--test-id",
                        test.id(),
                        "--output",
                        str(output),
                        "--test-dir",
                        self.test_dir,
                    ],
                    timeout=600,
                    env=env,
                )
                report = wire.read_json(output)
                record = report["cases"][0]
                if (
                    code != report["code"]
                    or record["id"] != spec["id"]
                    or len(report["cases"]) != 1
                ):
                    raise ValueError("invalid subprocess result")
                self.records[test.id()] = record
                self.fixture_events.extend(report["fixtures"])
                self.skipped.extend(
                    (test, f"{item['id']}: {item['reason']}")
                    for item in record.get("subtest_skips", [])
                )
                outcome = record["outcome"]
                details = "\n".join(record["details"])
                if outcome == "skip":
                    self.skipped.append((test, details))
                elif outcome == "expected_failure":
                    self.expectedFailures.append((test, details))
                elif outcome == "unexpected_success":
                    self.unexpectedSuccesses.append(test)
                elif outcome in ("failure", "error", "blocked_by_fixture"):
                    self.failures.append((test, details))
                if code and not self.failures:
                    self.errors.append((test, "subprocess fixture error"))
        except Exception:
            self.addError(test, sys.exc_info())
        finally:
            self.stopTest(test)

    def report(self, cases: list[dict]) -> dict:
        """Account for cases blocked by module or class fixtures."""
        records = dict(self.records)
        for spec in cases:
            if spec["id"] in records:
                continue
            relevant = []
            for event in self.fixture_events:
                boundary = event["id"].partition("(")[2].removesuffix(")")
                if boundary and spec["id"].startswith(boundary + "."):
                    relevant.append(event)
            if relevant:
                skipped = all(e["outcome"] == "skip" for e in relevant)
                records[spec["id"]] = dict(
                    id=spec["id"],
                    outcome="skip" if skipped else "blocked_by_fixture",
                    details=[e["detail"] for e in relevant],
                    seconds=0,
                )
        complete = Counter(records.keys()) == Counter(c["id"] for c in cases)
        return dict(
            cases=list(records.values()),
            fixtures=self.fixture_events,
            code=2 if not complete else (0 if self.wasSuccessful() else 1),
        )


def execute(group: dict, test_dir: str) -> dict:
    """Execute a suite through the same runner as the serial entry point."""
    from tzrec.tests.run import TZRecTestRunner

    result = TZRecTestRunner(
        verbosity=2, resultclass=partial(RecordingResult, test_dir=test_dir)
    ).run(load_cases(group, test_dir), unittest.TestSuite())
    return result.report(group["cases"])


def child_environment(case: dict, test_dir: str) -> dict[str, str]:
    """Get the existing per-case environment before starting a fresh interpreter."""
    loaded = next(leaves(load_cases(dict(cases=[case]), test_dir)))
    env = dict(os.environ)
    if hasattr(loaded, "_run_env"):
        env.update(loaded._run_env())
    preserve_execution_environment(env)
    return env


def preserve_execution_environment(env: dict[str, str]) -> None:
    """Keep device isolation and durable result journals across child processes."""
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "TZREC_CI_EVENT_LOG",
        "TZREC_CI_FAILURE_FILE",
        "TZREC_CI_SESSION_ID",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
        else:
            env.pop(key, None)


def verify(manifest: dict, reports: list[dict]) -> dict:
    """Reject missing, duplicate, stale, or abnormal results before reporting green."""
    groups = manifest["groups"]
    expected = Counter(c["id"] for g in groups for c in g["cases"])
    observed = Counter(c["id"] for r in reports for c in r.get("cases", []))
    complete = (
        Counter(g["key"] for g in groups) == Counter(r.get("group") for r in reports)
        and all(
            r.get("digest") == manifest["digest"] and r.get("version") == wire.VERSION
            for r in reports
        )
        and expected == observed
        and all(n == 1 for n in expected.values())
        and all(type(r.get("code")) is int and r["code"] in (0, 1) for r in reports)
        and all(
            c.get("outcome")
            in (
                "success",
                "skip",
                "expected_failure",
                "unexpected_success",
                "failure",
                "error",
                "blocked_by_fixture",
            )
            for r in reports
            for c in r.get("cases", [])
        )
    )
    outcomes = Counter(c["outcome"] for r in reports for c in r.get("cases", []))
    failed = any(
        outcomes[k]
        for k in ("failure", "error", "unexpected_success", "blocked_by_fixture")
    )
    failed |= any(
        e["outcome"] != "skip" for r in reports for e in r.get("fixtures", [])
    )
    return dict(
        version=wire.VERSION,
        session_id=manifest["session_id"],
        digest=manifest["digest"],
        code=2
        if not complete
        else (1 if failed or any(r["code"] for r in reports) else 0),
        planned=sum(expected.values()),
        reported=sum(observed.values()),
        outcomes=dict(outcomes),
        subtest_skips=sum(
            len(c.get("subtest_skips", [])) for r in reports for c in r.get("cases", [])
        ),
        missing=list((expected - observed).elements()),
        unexpected=list((observed - expected).elements()),
    )
