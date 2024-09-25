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


import logging
import time

logger = logging.getLogger("tzrec")
logger.setLevel(logging.INFO)


class ProgressLogger:
    """Logger with iterate speed."""

    def __init__(
        self, desc: str, start_n: int = -1, mininterval: float = 1, miniters: int = 0
    ) -> None:
        self._desc = desc
        self._last_time = time.time()
        self._last_n = start_n
        self._mininterval = mininterval
        self._miniters = miniters

    def set_description(self, desc: str) -> None:
        """Set logger description."""
        self._desc = desc

    def log(self, n: int, suffix: str = "") -> None:
        """Log iteration."""
        dn = n - self._last_n
        if dn > self._miniters:
            cur_time = time.time()
            dt = cur_time - self._last_time
            if dt > self._mininterval:
                logger.info(f"{self._desc}: {n}it [{dn/dt:.2f}it/s] {suffix}")
                self._last_time = cur_time
                self._last_n = n
