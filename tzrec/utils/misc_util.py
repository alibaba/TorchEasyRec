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


def random_name(length: int = 8) -> str:
    """Generate a random name with ascii_letters and digits."""
    random.seed(int(datetime.now().timestamp()))
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for i in range(length)]
    )


# pyre-ignore [2, 3]
def run_cmd(cmd_str, log_file, env=None):
    """Run a shell cmd."""
    cmd_str = cmd_str.replace("\r", " ").replace("\n", " ")
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("RUNCMD: %s > %s 2>&1 " % (cmd_str, log_file))
    with open(log_file, "w") as lfile:
        proc = subprocess.Popen(
            cmd_str, stdout=lfile, stderr=subprocess.STDOUT, shell=True, env=env
        )
        return proc


def get_free_port(host: str = "127.0.0.1") -> int:
    """Get free port in localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
