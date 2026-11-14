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
"""Run all unit tests."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.runner import TextTestRunner
from unittest.signals import registerResult

SUBPROC_TEST_PATTERN = [
    ".dataset_test.",
    ".odps_dataset_test.",
    ".parquet_dataset_test.",
    ".sampler_test.",
    ".tdm.gen_tree.",
    ".convert_easyrec_config_to_tzrec_config_test.",
]


def _is_in_scope(test_case: unittest.TestCase, scope: str) -> bool:
    method = getattr(test_case, test_case._testMethodName, None)
    method_scopes = set(getattr(method, "_ci_scopes", ()) or ())
    if scope in method_scopes:
        return True
    cls = type(test_case)
    cls_scopes = set(getattr(cls, "_ci_scopes", ()) or ())
    return scope in cls_scopes


def _filter_in_scope(suite, scope: str) -> unittest.TestSuite:
    """Prune tests not tagged with ``scope``; preserves nested suite layout.

    Robust against ``_FailedTest`` placeholders (modules whose import
    failed at discovery): they aren't iterable, so handle the leaf-case
    first and just drop them from a scoped run.
    """
    new_suite = unittest.TestSuite()
    if isinstance(suite, unittest.TestCase):
        if _is_in_scope(suite, scope):
            new_suite.addTest(suite)
        return new_suite
    for child in suite:
        if isinstance(child, unittest.TestCase):
            if _is_in_scope(child, scope):
                new_suite.addTest(child)
        else:
            filtered = _filter_in_scope(child, scope)
            if filtered.countTestCases() > 0:
                new_suite.addTest(filtered)
    return new_suite


def _gather_test_cases(args):
    test_dir = os.path.abspath(args.test_dir)
    test_suite_main = unittest.TestSuite()
    test_suite_subproc = unittest.TestSuite()
    discover = unittest.defaultTestLoader.discover(
        test_dir, pattern=args.pattern, top_level_dir=None
    )
    for suite_discovered in discover:
        for test_case in suite_discovered:
            if args.scope is not None:
                test_case = _filter_in_scope(test_case, args.scope)
                if test_case.countTestCases() == 0:
                    continue
            test_case_str = str(test_case)
            is_subproc = False
            for subp_pattern in SUBPROC_TEST_PATTERN:
                if subp_pattern in test_case_str:
                    is_subproc = True
                    break
            if not is_subproc or isinstance(test_case, unittest.TestCase):
                test_suite_main.addTest(test_case)
            else:
                test_suite_subproc.addTest(test_case)

            if hasattr(test_case, "__iter__"):
                for subcase in test_case:
                    if args.list_tests:
                        print(subcase)
            else:
                if args.list_tests:
                    print(test_case)
    return test_suite_main, test_suite_subproc


def _error_info_of_subproc_result(result):
    start = False
    endline = 0
    error_info = ""
    for line in result.stderr.splitlines():
        if "====" in line:
            start = True
            endline = 0
        if start:
            error_info += line + "\n"
            if "----" in line:
                endline += 1
            if endline == 2:
                break
    return error_info


class TZRecTestRunner(TextTestRunner):
    """Test Runner for TorchEasyRec."""

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)

    # pyrefly: ignore[bad-override]
    def run(self, main_test, subproc_test):
        """Run the given test case or test suite."""
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals
        with warnings.catch_warnings():
            if self.warnings:
                # if self.warnings is set, use it to filter all the warnings
                warnings.simplefilter(self.warnings)
                # if the filter is 'default' or 'always', special-case the
                # warnings from the deprecated unittest methods to show them
                # no more than once per module, because they can be fairly
                # noisy.  The -Wd and -Wa flags can be used to bypass this
                # only when self.warnings is None.
                if self.warnings in ["default", "always"]:
                    warnings.filterwarnings(
                        "module",
                        category=DeprecationWarning,
                        message=r"Please use assert\w+ instead.",
                    )
            startTime = time.perf_counter()
            startTestRun = getattr(result, "startTestRun", None)
            if startTestRun is not None:
                startTestRun()
            try:
                main_test(result)
                # run subproc test in subprocess to prevent some global var fork problem
                for subp_test_group in subproc_test:
                    for one_subp_test in subp_test_group:
                        if hasattr(result, "run_subprocess"):
                            result.run_subprocess(one_subp_test)
                            continue
                        subp_test_module = f"tzrec.{one_subp_test.__module__}"
                        subp_test_name = f"{one_subp_test.__class__.__name__}.{one_subp_test._testMethodName}"  # NOQA
                        try:
                            run_env = {}
                            if hasattr(one_subp_test, "_run_env"):
                                run_env = one_subp_test._run_env()
                            subp_result = subprocess.run(
                                ["python", "-m", subp_test_module, subp_test_name],
                                capture_output=True,
                                text=True,
                                timeout=600,
                                env=dict(os.environ, **run_env),
                            )
                            if subp_result.returncode != 0:
                                result.failures.append(
                                    (
                                        one_subp_test,
                                        _error_info_of_subproc_result(subp_result),
                                    )
                                )
                                self.stream.write("F")
                            else:
                                result.addSuccess(one_subp_test)
                                self.stream.write(".")
                        except subprocess.TimeoutExpired:
                            result.failures.append((one_subp_test, "timeout"))
                            self.stream.write("F")
                        self.stream.flush()
            finally:
                stopTestRun = getattr(result, "stopTestRun", None)
                if stopTestRun is not None:
                    stopTestRun()

            stopTime = time.perf_counter()
        timeTaken = stopTime - startTime
        result.printErrors()
        if hasattr(result, "separator2"):
            self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln(
            "Ran %d test%s in %.3fs" % (run, run != 1 and "s" or "", timeTaken)
        )
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(
                len,
                (result.expectedFailures, result.unexpectedSuccesses, result.skipped),
            )
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        infos = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = len(result.failures), len(result.errors)
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        self.stream.flush()
        return result


def write_json(path: Path, value: Any) -> None:
    """Publish a complete artifact by atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path) -> Any:
    """Read an artifact produced by this protocol."""
    with path.open() as stream:
        return json.load(stream)


def terminate(process: subprocess.Popen) -> None:
    """Reap an owned process group, including its test subprocesses."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_process(command: list[str], *, timeout: float, **kwargs: Any) -> int:
    """Execute a fresh process group and clean it up on timeout or cancellation."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(signum: int, frame: Any) -> None:
        """Unwind through the process-group cleanup."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        return process.wait(timeout=max(0.01, timeout))
    finally:
        signal.signal(signal.SIGTERM, previous)
        terminate(process)


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
    unittest.defaultTestLoader.errors.clear()
    main, child = _gather_test_cases(
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
            if scope is None:
                parallel = getattr(
                    method, "_ci_cpu_parallel", getattr(cls, "_ci_cpu_parallel", False)
                )
                devices, split = (0, split) if parallel else ("baseline", "module")
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
                plan = output.with_name("case.json")
                write_json(plan, dict(cases=[dict(spec, route="main")]))
                code = run_process(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--group-file",
                        str(plan),
                        "--output",
                        str(output),
                        "--test_dir",
                        self.test_dir,
                    ],
                    timeout=600,
                    env=env,
                )
                report = read_json(output)
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
    main, child = unittest.TestSuite(), unittest.TestSuite()
    for route, suite in (("main", main), ("subproc", child)):
        cases = [c for c in group["cases"] if c.get("route", "main") == route]
        if cases:
            suite.addTest(
                unittest.TestSuite(leaves(load_cases(dict(cases=cases), test_dir)))
            )
    result = TZRecTestRunner(
        verbosity=2, resultclass=partial(RecordingResult, test_dir=test_dir)
    ).run(main, child)
    return result.report(group["cases"])


def preserve_execution_environment(env: dict[str, str]) -> None:
    """Keep device isolation and durable result journals across child processes."""
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "TZREC_CI_EVENT_LOG",
        "TZREC_CI_SESSION_ID",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
        else:
            env.pop(key, None)


def main() -> int:
    """Run the requested suites, optionally collecting structured results."""
    parser = argparse.ArgumentParser(description="Test TorchEasyRec")
    parser.add_argument("--list_tests", action="store_true", help="list all tests")
    parser.add_argument(
        "--pattern", type=str, default="*_test.py", help="test file pattern"
    )
    parser.add_argument(
        "--test_dir", type=str, default="tzrec", help="directory to be tested"
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="filter tests by CI scope marker (e.g. 'h20'); "
        "default None runs the full suite",
    )
    parser.add_argument(
        "--report-dir", help="write CI results and a durable outcome journal"
    )
    parser.add_argument(
        "--discover-to", type=Path, help="write the exact cases for dispatch"
    )
    parser.add_argument(
        "--group-file", type=Path, help="execute only these discovered case IDs"
    )
    parser.add_argument(
        "--output", type=Path, help="write the result of the assigned group"
    )
    args = parser.parse_args()

    if args.discover_to:
        write_json(args.discover_to, discover(args.test_dir, args.pattern, args.scope))
        return 0
    if args.group_file or args.output:
        if not args.group_file or not args.output:
            parser.error("--group-file and --output must be used together")
        group = read_json(args.group_file)
        expected_devices = group.get("visible_devices")
        if expected_devices is not None:
            import torch

            if torch.cuda.device_count() != expected_devices:
                raise RuntimeError("wrong device view for test group")
        os.environ["TZREC_CI_EVENT_LOG"] = str(args.output.with_suffix(".events.jsonl"))
        report = execute(group, args.test_dir)
        report.update(group=group.get("key"), session_id=group.get("session_id"))
        write_json(args.output, report)
        return report["code"]

    runner = TZRecTestRunner()
    test_suite_main, test_suite_subproc = _gather_test_cases(args)

    if not args.list_tests and args.report_dir:
        report_dir = Path(args.report_dir).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TZREC_CI_EVENT_LOG"] = str(report_dir / "events.jsonl")
        report_cases = [
            {"id": case.id()}
            for suite in (test_suite_main, test_suite_subproc)
            for case in leaves(suite)
        ]
        expected_ids = [case["id"] for case in report_cases]
        if not expected_ids or len(expected_ids) != len(set(expected_ids)):
            raise ValueError("empty or duplicate test discovery")
        runner = TZRecTestRunner(
            resultclass=partial(RecordingResult, test_dir=args.test_dir)
        )
        runner_result = runner.run(test_suite_main, test_suite_subproc)
        report = runner_result.report(report_cases)
        report.update(
            version=1,
            session_id=os.environ.get("TZREC_CI_SESSION_ID", ""),
            planned=len(report_cases),
            reported=len(report["cases"]),
        )
        if {case["id"] for case in report["cases"]} != set(expected_ids):
            report["code"] = 2
        write_json(report_dir / "serial-result.json", report)
        return report["code"]
    elif not args.list_tests:
        runner_result = runner.run(test_suite_main, test_suite_subproc)
        failed, errored = len(runner_result.failures), len(runner_result.errors)
        if failed > 0 or errored > 0:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
