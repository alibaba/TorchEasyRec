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

import os
import random
import socket
import string
import subprocess
from datetime import datetime

RUN_CMD_RETRY_NUM = 3


def random_name(length: int = 8) -> str:
    """Generate a random name with ascii_letters and digits."""
    random.seed(int(datetime.now().timestamp()))
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for i in range(length)]
    )


# pyre-ignore [2, 3]
def run_cmd(cmd_str, log_file, env=None, timeout=None):
    """Run a shell cmd."""
    cmd_str = cmd_str.replace("\r", " ").replace("\n", " ")
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for _ in range(RUN_CMD_RETRY_NUM):
        if "#MASTER_PORT#" in cmd_str:
            run_cmd_str = cmd_str.replace("#MASTER_PORT#", str(get_free_port()))
        else:
            run_cmd_str = cmd_str
        print("RUNCMD: %s > %s 2>&1 " % (run_cmd_str, log_file))
        with open(log_file, "w") as lfile:
            proc = subprocess.Popen(
                run_cmd_str, stdout=lfile, stderr=subprocess.STDOUT, shell=True, env=env
            )
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            proc.wait()
            raise e
        if proc.returncode == 0:
            return True
        else:
            with open(log_file) as lfile:
                if "The server socket has failed to listen" in lfile.read():
                    continue
                else:
                    return False
    return False


def get_free_port(host: str = "127.0.0.1") -> int:
    """Get free port in localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
