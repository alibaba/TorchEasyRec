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

"""Own test containers, device pairs, and worker lifecycles for CI."""

import argparse
import grp
import importlib
import json
import os
import pwd
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from tzrec.tests import ci_devices as devices
else:
    devices = importlib.import_module("ci_devices")


def docker(*arguments, check=True):
    """Call Docker without shell interpolation."""
    return subprocess.run(
        ["docker", *map(str, arguments)],
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


def atomic(path: Path, data: dict[str, Any]):
    """Publish small host-owned state for node recovery."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    for parent in (path.parent, path.parent.parent):
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def artifact_text(directory: Path, name: str, *, tail: bool = False) -> str:
    """Read a bounded flat artifact without following checkout-created symlinks."""
    if "/" in name or name in (".", ".."):
        raise ValueError("artifact name must be flat")
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            st = os.fstat(stream.fileno())
            limit = (1 if tail else 8) * 1024 * 1024
            if not stat.S_ISREG(st.st_mode) or (st.st_size > limit and not tail):
                raise ValueError("artifact is not a bounded regular file")
            truncated = tail and st.st_size > limit
            if truncated:
                stream.seek(st.st_size - limit)
            data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError("artifact too large")
            text = data.decode("utf-8", errors="replace" if tail else "strict")
            return (
                "[Earlier log output omitted; showing the last 1 MiB.]\n"
                if truncated
                else ""
            ) + text
    finally:
        os.close(parent)


def artifact_json(directory: Path, name: str) -> dict:
    """Decode one host-facing artifact without following symlinks."""
    result = json.loads(artifact_text(directory, name))
    if not isinstance(result, dict):
        raise ValueError("artifact must be an object")
    return result


def limits() -> tuple[int, int]:
    """Use the effective cgroup CPU and memory allowance, not bare-host capacity."""
    cores = len(os.sched_getaffinity(0))
    for quota, period in (
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        if Path(quota).exists() and int(Path(quota).read_text()) > 0:
            cores = min(
                cores, int(Path(quota).read_text()) // int(Path(period).read_text())
            )
    if Path("/sys/fs/cgroup/cpu.max").exists():
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = min(cores, int(quota) // int(period))
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            )
        )
        * 1024
    )
    for file in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        if Path(file).exists():
            value = Path(file).read_text().strip()
            if value != "max":
                memory = min(memory, int(value))
    return cores, memory // 1024**3


def capacity(
    config: dict[str, Any], cores: int, ram_gib: int, existing_containers: int
) -> int:
    """Bound pair workers while reserving a concurrent CPU job and coordinator."""
    b = config["budget"]
    return max(
        0,
        min(
            len(config["devices"]["verified_pairs"]),
            (cores - b["host_cpu"] - b["cpu_cores"] - b["coordinator_cpu"])
            // b["worker_cpu"],
            (ram_gib - b["host_ram_gib"] - b["cpu_ram_gib"] - b["coordinator_ram_gib"])
            // b["worker_ram_gib"],
            b["containers_limit"] - existing_containers - 2,
        ),
    )


class JobRunner:
    """Own one job's container lifecycle; test selection remains in TorchEasyRec."""

    def __init__(
        self,
        config: dict[str, Any],
        lane: str,
        checkout: Path,
        root: Path,
        test_args: list[str] | None = None,
    ):
        self.config, self.lane, self.root = config, lane, root
        self.test_args = test_args or []
        self.checkout = checkout.resolve()
        self.job_id = uuid.uuid4().hex
        self.session_id = self.job_id
        self.role = "cpu" if lane == "cpu" else "accel"
        self.containers: list[str] = []
        self.image = ""
        self.timeout = config["job_timeout_seconds"][lane]
        self.deadline = time.monotonic() + self.timeout
        self.state = dict(
            job=self.job_id,
            lane=lane,
            run_id=int(os.getenv("GITHUB_RUN_ID", "0")),
            attempt=int(os.getenv("GITHUB_RUN_ATTEMPT", "0")),
            phase="preparing",
            version=1,
            boot_id=os.environ.get("TZCI_BOOT_ID", ""),
            code=None,
            source=str(self.checkout),
        )

    def environment(self) -> list[str]:
        """Pass only the lane's allowlisted values, never the runner environment."""
        env = dict(
            HOME="/job/home",
            PYTHONPATH=".",
            CI="true",
            GITHUB_ACTIONS="true",
            TZREC_CI_SESSION_ID=self.session_id,
            TZREC_CI_FAILURE_FILE="/job/status/failed",
            MKL_THREADING_LAYER="GNU",
            PIP_CACHE_DIR="/job/cache/pip",
            PATH="/job/env/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
            TRITON_CACHE_DIR="/job/cache/triton",
            TORCHINDUCTOR_CACHE_DIR="/job/cache/inductor",
        )
        if self.lane == "gpu":
            env.update(
                CUDA_HOME="/usr/local/cuda-13",
                LD_LIBRARY_PATH="/usr/local/cuda-13/compat",
            )
        if self.lane == "ppu":
            env["TZREC_TEST_TIMEOUT_SCALE"] = "4"
        if self.lane == "cpu":
            for key in (
                "ALIBABA_CLOUD_ECS_METADATA",
                "CI_ODPS_PROJECT_NAME",
                "ODPS_ENDPOINT",
                "CI_ALIKAFKA_INSTANCE_ID",
                "CI_ALIKAFKA_BROKERS",
                "CI_REGION",
            ):
                if key in os.environ:
                    env[key] = os.environ[key]
            env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
        return [part for key, value in env.items() for part in ("-e", f"{key}={value}")]

    def start(
        self,
        name: str,
        command: list[str],
        *,
        source: Path,
        devices: list[str],
        cores: int,
        memory: int,
        shm: int,
        prepared: bool = True,
    ) -> str:
        """Launch a fixed entry point with only its own checkout, cache and session."""
        container = f"tzci-{self.job_id}-{name}"
        cache = self.root / f"cache-{name}"
        home = self.root / f"home-{name}"
        for path in (
            cache,
            home,
            self.root / "env",
            self.root / "session",
            self.root / "status",
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not (self.root / "passwd").exists():
            uid, gid = os.getuid(), os.getgid()
            users = {
                0: "root:x:0:0:root:/root:/bin/bash\n",
                65534: "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n",
            }
            users[uid] = (
                f"{pwd.getpwuid(uid).pw_name}:x:{uid}:{gid}:CI:/job/home:/bin/bash\n"
            )
            groups = {0: "root", 65534: "nogroup", gid: grp.getgrgid(gid).gr_name}
            (self.root / "passwd").write_text("".join(users.values()))
            (self.root / "group").write_text(
                "".join(f"{name}:x:{number}:\n" for number, name in groups.items())
            )
        args = [
            "run",
            "--detach",
            "--pull=never",
            "--name",
            container,
            "--label",
            f"tzci.node={self.config['name']}",
            "--label",
            f"tzci.role={self.role}",
            "--label",
            f"tzci.job={self.job_id}",
            "--log-opt",
            "max-size=20m",
            "--log-opt",
            "max-file=3",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpus",
            str(cores),
            "--memory",
            f"{memory}g",
            "--shm-size",
            f"{shm}g",
            "--ulimit",
            "memlock=-1",
            "--workdir",
            "/job/repo",
            "-v",
            f"{source}:/job/repo",
            "-v",
            f"{self.root / 'env'}:/job/env" + (":ro" if prepared else ""),
            "-v",
            f"{self.root / 'session'}:/job/session",
            "-v",
            f"{self.root / 'status'}:/job/status",
            "-v",
            f"{cache}:/job/cache",
            "-v",
            f"{home}:/job/home",
            "-v",
            f"{self.root / 'passwd'}:/etc/passwd:ro",
            "-v",
            f"{self.root / 'group'}:/etc/group:ro",
        ]
        if devices:
            if self.config["hardware"] == "ppu":
                args += [f"--device=/dev/alixpu_ppu{d}" for d in devices]
                args += ["--device=/dev/alixpu", "--device=/dev/alixpu_ctl"]
            else:
                args += ["--gpus", '"device=' + ",".join(devices) + '"']
        else:
            args += ["-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES="]
        args += self.environment() + [self.image, *command]
        self.containers.append(container)
        docker(*args)
        return container

    def wait(
        self,
        container: str,
        *,
        companions: list[str] | None = None,
        timeout: int | None = None,
    ) -> int:
        """Wait for completion and fail the whole job if a companion disappears."""
        deadline = (
            min(self.deadline, time.monotonic() + timeout)
            if timeout is not None
            else self.deadline
        )
        while time.monotonic() < deadline:
            state = json.loads(
                docker("inspect", "--format", "{{json .State}}", container).stdout
            )
            for name in companions or []:
                other = json.loads(
                    docker("inspect", "--format", "{{json .State}}", name).stdout
                )
                if not other["Running"] and (
                    other["ExitCode"] or other.get("OOMKilled")
                ):
                    raise RuntimeError(f"worker {name} exited abnormally")
            if not state["Running"]:
                if state.get("OOMKilled"):
                    raise RuntimeError(f"container OOM: {container}")
                return state["ExitCode"]
            time.sleep(1)
        raise TimeoutError("CI job exceeded its deadline")

    def logs(self, container: str):
        """Retain bounded Docker logs without copying untrusted artifact paths."""
        report = self.root / "report"
        report.mkdir(exist_ok=True)
        with (report / (container + ".log")).open("w") as stream:
            subprocess.run(
                ["docker", "logs", container],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=60,
            )

    def remove(self, container: str):
        """Remove an owned container after recording its exit and logs."""
        self.logs(container)
        docker("rm", "-f", container)
        self.containers.remove(container)

    def probe(self, source: Path) -> dict:
        """Validate this job's image and devices using owned, cancellable containers."""
        if self.lane == "cpu":
            name = self.start(
                "probe-cpu",
                ["python", "-c", devices.CPU_PROBE],
                source=source,
                devices=[],
                cores=1,
                memory=2,
                shm=1,
                prepared=False,
            )
            code = self.wait(name, timeout=90)
            self.remove(name)
            if code:
                raise RuntimeError("CPU image/isolation probe failed; see probe log")
            return dict(baseline=[], verified_pairs=[])
        inventory = sorted(
            devices.inventory(self.config["hardware"]), key=lambda d: d["index"]
        )
        if len(inventory) != self.config["expected_devices"]:
            raise ValueError("physical device inventory does not match CI policy")
        pairs = [inventory[i : i + 2] for i in range(0, len(inventory), 2)]
        probes = []
        for slot, pair in enumerate(pairs):
            command = devices.probe_command(pair)
            name = self.start(
                f"probe-{slot}",
                command,
                source=source,
                devices=[x["id"] for x in pair],
                cores=2,
                memory=16 if self.lane == "ppu" else 4,
                shm=1,
                prepared=False,
            )
            probes.append((name, pair))
        for name, pair in probes:
            code = self.wait(name, timeout=120)
            output = docker("logs", name).stdout
            self.remove(name)
            if code:
                raise RuntimeError(f"device probe failed: {name}; see probe log")
            devices.check_probe(output, pair, self.config["hardware"])
        return dict(
            verified_pairs=[[x["id"] for x in pair] for pair in pairs],
            baseline=[
                x["id"] for x in (inventory if self.lane == "gpu" else inventory[:2])
            ],
        )

    def run(self) -> int:
        """Prepare once, run the baseline, then provide isolated pair workers."""
        if self.root.exists():
            raise ValueError("job directory already exists")
        self.root.mkdir(parents=True, mode=0o700)
        (self.root / "status").mkdir()
        atomic(self.root / "state.json", self.state)
        source = self.root / "repo"
        shutil.copytree(
            self.checkout,
            source,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", "tmp", "experiments", "__pycache__", ".hypothesis"
            ),
        )
        image = self.config["images"][self.lane]
        subprocess.run(["docker", "pull", image], check=True, timeout=1200)
        self.image = docker(
            "image", "inspect", "--format", "{{.Id}}", image
        ).stdout.strip()
        self.config["devices"] = self.probe(source)
        self.state["image"] = self.image
        b = self.config["budget"]
        cores, ram = limits()
        if (
            cores <= b["host_cpu"] + b["cpu_cores"]
            or ram <= b["host_ram_gib"] + b["cpu_ram_gib"]
        ):
            raise ValueError("insufficient effective node resources")
        visible = [] if self.lane == "cpu" else self.config["devices"]["baseline"]
        alloc_cpu = (
            b["cpu_cores"]
            if self.lane == "cpu"
            else cores - b["host_cpu"] - b["cpu_cores"]
        )
        alloc_ram = (
            b["cpu_ram_gib"]
            if self.lane == "cpu"
            else ram - b["host_ram_gib"] - b["cpu_ram_gib"]
        )
        script = {
            "cpu": "ci_test_cpu.sh",
            "gpu": "ci_test.sh",
            "ppu": "ci_test_ppu.sh",
        }[self.lane]
        arguments = (
            ["--report-dir", "/job/session"]
            if self.lane == "cpu"
            else [
                "--session",
                "/job/session",
                "--session-id",
                self.session_id,
                "--protocol",
                "1",
                "--scope",
                "gpu",
                "--timeout",
                str(self.timeout),
            ]
        )
        prepare = shlex.join(
            ["bash", "scripts/ci/" + script, "--native", *arguments, *self.test_args]
        )
        base = self.start(
            "baseline",
            ["bash", "-c", prepare],
            source=source,
            devices=visible,
            cores=alloc_cpu,
            memory=alloc_ram,
            shm=min(
                b["cpu_shm_gib" if self.lane == "cpu" else "baseline_shm_gib"],
                alloc_ram,
            ),
            prepared=False,
        )
        rc = self.wait(base)
        self.remove(base)
        if self.lane == "cpu":
            report = artifact_json(self.root / "session", "serial-result.json")
            if (
                type(report.get("code")) is not int
                or report["code"] != rc
                or report.get("version") != 1
                or report.get("session_id") != self.session_id
                or not report.get("planned")
                or report.get("planned") != report.get("reported")
            ):
                raise RuntimeError("invalid serial test receipt")
            return rc
        if rc:
            return rc
        phase = artifact_json(self.root / "session", "phase.json")
        if (
            phase.get("version") != 1
            or phase.get("session_id") != self.session_id
            or phase.get("next") not in ("pairs", "finish")
        ):
            raise RuntimeError("invalid baseline handoff")
        count = 0
        if phase["next"] == "pairs":
            count = capacity(
                self.config, cores, ram, len(docker("ps", "-q").stdout.split())
            )
            if count < 1:
                raise RuntimeError("no validated pair capacity")
        coordinator = self.start(
            "coordinator",
            [
                "/job/env/bin/python",
                "tzrec/tests/dispatch.py",
                "serve",
                "--session",
                "/job/session",
                "--session-id",
                self.session_id,
                "--protocol",
                "1",
                "--workers",
                str(count),
            ],
            source=source,
            devices=[],
            cores=b["coordinator_cpu"],
            memory=b["coordinator_ram_gib"],
            shm=1,
        )
        peers = []
        for slot, pair in enumerate(self.config["devices"]["verified_pairs"][:count]):
            checkout = self.root / f"worker-{slot}"
            shutil.copytree(
                source,
                checkout,
                symlinks=True,
                ignore=shutil.ignore_patterns("tmp", "__pycache__", ".hypothesis"),
            )
            peers.append(
                self.start(
                    f"worker-{slot}",
                    [
                        "/job/env/bin/python",
                        "tzrec/tests/worker.py",
                        "serve",
                        "--session",
                        "/job/session",
                        "--protocol",
                        "1",
                        "--slot",
                        str(slot),
                    ],
                    source=checkout,
                    devices=pair,
                    cores=b["worker_cpu"],
                    memory=b["worker_ram_gib"],
                    shm=b["worker_shm_gib"],
                )
            )
        rc = self.wait(coordinator, companions=peers)
        summary = artifact_json(self.root / "session", "summary.json")
        if (
            summary.get("session_id") != self.session_id
            or type(summary.get("code")) is not int
            or summary["code"] != rc
            or summary.get("version") != 1
            or not summary.get("planned")
            or summary.get("planned") != summary.get("reported")
        ):
            raise RuntimeError("coordinator result does not match process exit")
        exits = [self.wait(p) for p in peers]
        if any(exits):
            raise RuntimeError("a worker exited abnormally")
        return rc

    def cleanup(self, code: int):
        """Close only this job's resources, preserving durable failure evidence."""
        errors = []
        for container in list(self.containers):
            try:
                docker("stop", "--time", "20", container, check=False)
                self.remove(container)
            except Exception as error:
                errors.append(f"{container}: {error}")
        self.state.update(phase="finished", code=2 if errors else code)
        if self.root.exists():
            report = self.root / "report"
            report.mkdir(exist_ok=True)
            for name in ("manifest.json", "summary.json", "serial-result.json"):
                try:
                    value = artifact_json(self.root / "session", name)
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, RecursionError) as error:
                    errors.append(f"{name}: {error}")
                    self.state["code"] = 2
                    continue
                atomic(report / name, value)
            for folder, pattern in (
                ("results", r"[0-9a-f]{24}(?:-[0-9]+)?\.json"),
                ("logs", r"[0-9a-f]{24}\.log"),
            ):
                source = self.root / "session" / folder
                try:
                    directory = os.open(
                        source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    try:
                        names = os.listdir(directory)
                    finally:
                        os.close(directory)
                    if len(names) > 4096:
                        raise ValueError("too many test artifacts")
                    destination = report / folder
                    destination.mkdir(exist_ok=True)
                    for name in names:
                        if re.fullmatch(pattern, name):
                            (destination / name).write_text(
                                artifact_text(source, name, tail=folder == "logs")
                            )
                except FileNotFoundError:
                    pass
                except (OSError, ValueError) as error:
                    errors.append(f"{folder}: {error}")
                    self.state["code"] = 2
            atomic(self.root / "state.json", self.state)
            output = os.environ.get("GITHUB_OUTPUT")
            if output:
                with open(output, "a") as stream:
                    stream.write(f"report_dir={report}\n")
        if errors:
            raise RuntimeError("job cleanup failed: " + "; ".join(errors))


def main(argv: list[str]) -> int:
    """Run one lane using runner identity and repository-owned execution policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("cpu", "gpu", "ppu"), required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    args, test_args = parser.parse_known_args(argv)
    role = "cpu" if args.lane == "cpu" else "accel"
    if os.environ["TZCI_ROLE"] != role:
        raise ValueError("lane does not match the runner slot")
    hardware = os.environ["TZCI_HARDWARE"]
    if args.lane != "cpu" and hardware != {"gpu": "h20", "ppu": "ppu"}[args.lane]:
        raise ValueError("lane does not match the node hardware")
    config = json.loads(Path(__file__).with_name("ci_config.json").read_text())
    config = dict(config, **config["hardware"][hardware])
    config.update(name=os.environ["TZCI_NODE"], hardware=hardware)
    base = Path(os.environ["TZCI_JOB_ROOT"])
    base.mkdir(parents=True, exist_ok=True)
    runner = JobRunner(
        config, args.lane, args.checkout, base / uuid.uuid4().hex, test_args
    )
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(signum, frame):
        """Unwind through owned-container cleanup on runner cancellation."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    code = 2
    try:
        code = runner.run()
        return code
    except KeyboardInterrupt:
        code = 130
        return code
    finally:
        runner.cleanup(code)
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
