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

"""Versioned local transport and artifacts for CI test dispatch."""

import hashlib
import json
import os
import signal
import socket
import subprocess
from pathlib import Path
from typing import Any, BinaryIO

VERSION = 1
MAX_MESSAGE = 1024 * 1024


def digest(value: Any) -> str:
    """Return a stable identity for a JSON payload."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def read_manifest(session: Path) -> dict[str, Any]:
    """Validate the immutable manifest before accepting its execution plan."""
    value = read_json(session / "manifest.json")
    claimed = value.pop("digest")
    if value.get("version") != VERSION or digest(value) != claimed:
        raise ValueError("unsupported or modified manifest")
    value["digest"] = claimed
    return value


def send(stream: BinaryIO, value: dict[str, Any]) -> None:
    """Write one bounded JSON message."""
    data = json.dumps(value).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("dispatch message is too large")
    stream.write(data)
    stream.flush()


def receive(stream: BinaryIO) -> dict[str, Any]:
    """Read one complete message, treating disconnect as an error."""
    data = stream.readline(MAX_MESSAGE + 1)
    if not data.endswith(b"\n") or len(data) > MAX_MESSAGE:
        raise ValueError("truncated or oversized dispatch message")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("dispatch message must be an object")
    return value


def session_socket(session: Path, *, listen: bool) -> socket.socket:
    """Open the job socket without exceeding Unix socket path limits."""
    sock = socket.socket(socket.AF_UNIX)
    original = os.getcwd()
    try:
        os.chdir(session)
        if listen:
            sock.bind("dispatch.sock")
            os.chmod("dispatch.sock", 0o600)
            sock.listen()
        else:
            sock.connect("dispatch.sock")
    except BaseException:
        sock.close()
        raise
    finally:
        os.chdir(original)
    return sock


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
