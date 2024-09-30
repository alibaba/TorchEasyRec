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

from tzrec.main import export

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
        "train_config.model_dir.",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="directory where model should be exported to.",
    )
    parser.add_argument(
        "--asset_files",
        type=str,
        default=None,
        help="more files will be copy to export_dir.",
    )
    args, extra_args = parser.parse_known_args()

    export(
        args.pipeline_config_path,
        export_dir=args.export_dir,
        checkpoint_path=args.checkpoint_path,
    )
