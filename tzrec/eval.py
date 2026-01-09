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

from tzrec.main import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="checkpoint to be evaled, if not specified, use the latest checkpoint in "
        "train_config.model_dir",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="latest",
        help="eval type, decide which type of checkpoint to use (type: best, latest)",
    )
    parser.add_argument(
        "--eval_input_path",
        type=str,
        default=None,
        help="eval input path, will override pipeline_config.eval_input_path",
    )
    parser.add_argument(
        "--eval_result_filename",
        type=str,
        default="eval_result.txt",
        help="eval result metric filename",
    )
    args, extra_args = parser.parse_known_args()

    evaluate(
        args.pipeline_config_path,
        checkpoint_path=args.checkpoint_path,
        eval_type=args.eval_type,
        eval_input_path=args.eval_input_path,
        eval_result_filename=args.eval_result_filename,
    )
