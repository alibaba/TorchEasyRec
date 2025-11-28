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

import torch

from tzrec.utils.logging_util import logger


def use_hash_node_id() -> bool:
    """Use hash node id or not."""
    return os.environ.get("USE_HASH_NODE_ID", "0") == "1"


def enable_tma() -> bool:
    """Enable TMA (Tensor Memory Accelerator) for triton ops."""
    flag = os.environ.get("ENABLE_TMA", "0") == "1"
    if flag:
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 9:
                import triton
                from packaging import version

                if version.parse(triton.__version__) >= version.parse("3.5.0"):
                    return True
                else:
                    logger.warning("triton version lower than 3.5.0, we disable TMA.")
            else:
                logger.warning("device capability lower than 9.0, we disable TMA.")
        else:
            logger.warning("CUDA not available, we disable TMA.")
    return False
