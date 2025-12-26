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
import warnings as _warnings

# TODO(hongsheng.jhs): remove the warning filter when fbgemm team fix it.
_warnings.filterwarnings(
    "ignore",
    message=".*fbgemm::jagged_to_padded_dense: an autograd kernel was not registered.*",
)
_warnings.filterwarnings(
    "ignore",
    message=".*fbgemm::dense_embedding_codegen_lookup_function: an autograd kernel was not registered.*",  # NOQA
)

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
import torch as _torch  # NOQA
import numpy as _np  # NOQA
from tzrec.utils import load_class as _load_class  # NOQA

_log_level = _os.getenv("LOG_LEVEL")
if _log_level:
    _log_level = getattr(_logging, _log_level)
_logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(message)s", level=_log_level
)

# reproducibility
_torch_manual_seed = _os.getenv("TORCH_MANUAL_SEED")
if _torch_manual_seed:
    _torch.manual_seed(int(_torch_manual_seed))
_numpy_manual_seed = _os.getenv("NUMPY_MANUAL_SEED")
if _numpy_manual_seed:
    _np.random.seed(int(_numpy_manual_seed))
_use_deterministic_algorithms = _os.getenv("USE_DETERMINISTIC_ALGORITHMS", "0") == "1"
if _use_deterministic_algorithms:
    _torch.use_deterministic_algorithms(True)

_load_class.auto_import()

from tzrec.utils.filesystem_util import (  # NOQA
    register_external_filesystem as _register_external_filesystem,  # NOQA
)  # NOQA

_register_external_filesystem()
