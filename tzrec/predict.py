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

from tzrec.main import predict, predict_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripted_model_path",
        type=str,
        default=None,
        help="exported scripted model to be predict, if not specified, use the "
        "latest checkpoint in pipeline_config.model_dir",
    )
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
        help="checkpoint model to be predict.",
    )
    parser.add_argument(
        "--predict_input_path",
        type=str,
        default=None,
        help="inference data input path",
    )
    parser.add_argument(
        "--predict_output_path",
        type=str,
        default=None,
        help="inference data output path",
    )
    parser.add_argument(
        "--reserved_columns",
        type=str,
        default=None,
        help="column names to reserved in output",
    )
    parser.add_argument(
        "--output_columns",
        type=str,
        default=None,
        help="column names of model output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="predict batch size, default will use batch_size in config.",
    )
    parser.add_argument(
        "--predict_threads",
        type=int,
        default=None,
        help="predict threads num, default will use num_workers in data_config.",
    )
    parser.add_argument(
        "--is_profiling",
        action="store_true",
        default=False,
        help="profiling predict progress.",
    )
    parser.add_argument(
        "--debug_level",
        type=int,
        default=0,
        help="debug level for debug parsed inputs etc.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        help="dataset type, default will use dataset type in data_config.",
    )
    parser.add_argument(
        "--writer_type",
        type=str,
        default=None,
        help="data writer type, default will be same as dataset_type in data_config.",
    )
    parser.add_argument(
        "--edit_config_json",
        type=str,
        default=None,
        help='edit pipeline config str, example: {"data_config.fg_mode":"FG_DAG"}',
    )
    parser.add_argument(
        "--predict_steps",
        type=int,
        default=None,
        help="total predict steps limit.",
    )
    args, extra_args = parser.parse_known_args()

    if args.pipeline_config_path is not None:
        predict_checkpoint(
            pipeline_config_path=args.pipeline_config_path,
            predict_input_path=args.predict_input_path,
            predict_output_path=args.predict_output_path,
            checkpoint_path=args.checkpoint_path,
            reserved_columns=args.reserved_columns,
            output_columns=args.output_columns,
            batch_size=args.batch_size,
            debug_level=args.debug_level,
            dataset_type=args.dataset_type,
            writer_type=args.writer_type,
            edit_config_json=args.edit_config_json,
            predict_steps=args.predict_steps,
        )
    elif args.scripted_model_path is not None:
        predict(
            predict_input_path=args.predict_input_path,
            predict_output_path=args.predict_output_path,
            scripted_model_path=args.scripted_model_path,
            reserved_columns=args.reserved_columns,
            output_columns=args.output_columns,
            batch_size=args.batch_size,
            is_profiling=args.is_profiling,
            debug_level=args.debug_level,
            dataset_type=args.dataset_type,
            predict_threads=args.predict_threads,
            writer_type=args.writer_type,
            edit_config_json=args.edit_config_json,
            predict_steps=args.predict_steps,
        )
    else:
        raise ValueError("pipeline_config_path or scripted_model_path must be set.")
