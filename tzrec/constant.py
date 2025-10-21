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
from enum import Enum


class Mode(Enum):
    """Train/Eval/Predict Mode."""

    TRAIN = 1
    EVAL = 2
    PREDICT = 3


EASYREC_VERSION = "0.7.5"

EVAL_RESULT_FILENAME = "train_eval_result.txt"

PREDICT_QUEUE_TIMEOUT = int(os.environ.get("PREDICT_QUEUE_TIMEOUT") or 600)

TENSORBOARD_SUMMARIES = [
    "loss",
    "learning_rate",
    "parameter",
    "global_gradient_norm",
    "gradient_norm",
    "gradient",
]
