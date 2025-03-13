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

import argparse

from tzrec.main import train_and_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="will update the model_dir in pipeline_config",
    )
    parser.add_argument(
        "--train_input_path", type=str, default=None, help="train data input path"
    )
    parser.add_argument(
        "--eval_input_path", type=str, default=None, help="eval data input path"
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        default=False,
        help="continue train using existing model_dir",
    )
    parser.add_argument(
        "--fine_tune_checkpoint",
        type=str,
        default=None,
        help="will update the train_config.fine_tune_checkpoint in pipeline_config",
    )
    parser.add_argument(
        "--edit_config_json",
        type=str,
        default=None,
        help='edit pipeline config str, example: {"model_dir":"experiments/",'
        '"feature_configs[0].raw_feature.boundaries":[4,5,6,7]}',
    )
    args, extra_args = parser.parse_known_args()

    # import random
    # random.seed(42)
    train_and_evaluate(
        args.pipeline_config_path,
        args.train_input_path,
        args.eval_input_path,
        args.model_dir,
        args.continue_train,
        args.fine_tune_checkpoint,
        args.edit_config_json,
    )
