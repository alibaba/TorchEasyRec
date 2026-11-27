# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entry point for dumping a FeatureDB DynamicEmbedding FeatureView."""

import argparse

from tzrec.utils.fsview_dump_util import dump_fsview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Dump FeatureDB DynamicEmbedding FeatureView and merge with "
            "baseline sparse NPZs."
        )
    )
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        required=True,
        help="Path to streaming training pipeline config file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Full-export model directory containing baseline sparse/ and "
            "target sparse_fs_dump/."
        ),
    )
    parser.add_argument(
        "--edit_config_json",
        type=str,
        default=None,
        help=(
            "Optional JSON string overrides for pipeline config, e.g. "
            "'{\"train_config.delta_embedding_dump_config."
            'feature_store_config.version":"ol_v1"}\'.'
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help=(
            "Concurrent shard/part worker threads per rank "
            "(defaults to feature_store_config.max_workers)."
        ),
    )
    parser.add_argument(
        "--read_chunk_size_mb",
        type=int,
        default=64,
        help="Streaming chunk budget in MiB for compaction and NPZ merging.",
    )
    parser.add_argument(
        "--keep_versions",
        type=int,
        default=3,
        help=(
            "Maximum historical versions to keep under "
            "sparse_fs_dump/versions (0 keeps all)."
        ),
    )
    args, _ = parser.parse_known_args()

    dump_fsview(
        pipeline_config_path=args.pipeline_config_path,
        model_dir=args.model_dir,
        edit_config_json=args.edit_config_json,
        max_workers=args.max_workers,
        read_chunk_size_mb=args.read_chunk_size_mb,
        keep_versions=args.keep_versions,
    )
