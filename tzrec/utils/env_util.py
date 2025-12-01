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

from tzrec.utils.logging_util import logger


def use_hash_node_id() -> bool:
    """Use hash node id or not."""
    return os.environ.get("USE_HASH_NODE_ID", "0") == "1"


def use_rtp() -> bool:
    """Use RTP for online inference or not."""
    flag = os.environ.get("USE_RTP", "0") == "1"
    if flag and os.environ.get("USE_FARM_HASH_TO_BUCKETIZE", "false") != "true":
        logger.warning(
            "you should set USE_FARM_HASH_TO_BUCKETIZE=true for "
            "train/eval/export when use rtp for online inference."
        )
    return flag
