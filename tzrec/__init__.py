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

import os as _os

import torch as _torch  # NOQA

if "OMP_NUM_THREADS" not in _os.environ:
    _os.environ["OMP_NUM_THREADS"] = "1"

try:
    # import graphlearn before set GLOG_logtostderr, prevent graphlearn's glog to stderr
    import graphlearn as _gl  # NOQA
except Exception:
    pass

# make pyfg's glog to stderr
if "GLOG_logtostderr" not in _os.environ:
    _os.environ["GLOG_logtostderr"] = "1"

try:
    import pyfg as _pyfg  # NOQA
except Exception:
    pass

import logging as _logging  # NOQA

from tzrec.utils import load_class as _load_class  # NOQA

_logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s")
_load_class.auto_import()
