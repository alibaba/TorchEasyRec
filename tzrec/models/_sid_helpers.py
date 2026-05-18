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

"""Shared helpers for SID-generation model wrappers."""

from typing import List


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated int string, e.g. '256,128' -> [256, 128]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated float string, e.g. '1.0,0.5' -> [1.0, 0.5]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]
