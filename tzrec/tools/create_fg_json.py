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
import json
import os
import shutil
import tempfile

from odps import ODPS

from tzrec.datasets.odps_dataset import _create_odps_account
from tzrec.features.feature import create_fg_json
from tzrec.main import _create_features, _get_dataloader
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--fg_output_dir",
        type=str,
        default=None,
        help="Directory to output feature generator json file.",
    )
    parser.add_argument(
        "--reserves",
        type=str,
        default=None,
        help="Reserved column names, e.g. label,request_id.",
    )
    parser.add_argument(
        "--odps_project_name",
        type=str,
        default=None,
        help="odps project name.",
    )
    parser.add_argument(
        "--odps_schema_name",
        type=str,
        default=None,
        help="odps project name.",
    )
    parser.add_argument(
        "--fg_resource_name",
        type=str,
        default=None,
        help="fg json resource name. if specified, will upload fg.json to odps.",
    )
    parser.add_argument(
        "--force_update_resource",
        action="store_true",
        default=False,
        help="if true will update fg.json.",
    )
    parser.add_argument(
        "--remove_bucketizer",
        action="store_true",
        default=False,
        help="remove bucktizer params in fg json.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="debug feature config and fg json or not.",
    )
    args, extra_args = parser.parse_known_args()

    pipeline_config = config_util.load_pipeline_config(args.pipeline_config_path)
    features = _create_features(
        list(pipeline_config.feature_configs), pipeline_config.data_config
    )

    if args.debug:
        pipeline_config.data_config.num_workers = 1
        dataloader = _get_dataloader(
            pipeline_config.data_config, features, pipeline_config.train_input_path
        )
        iterator = iter(dataloader)
        _ = next(iterator)

    tmp_dir = tempfile.mkdtemp(prefix="tzrec_")
    fg_json = create_fg_json(
        features, asset_dir=tmp_dir, remove_bucketizer=args.remove_bucketizer
    )

    if args.reserves is not None:
        reserves = []
        for column in args.reserves.strip().split(","):
            reserves.append(column.strip())
        fg_json["reserves"] = reserves

    fg_name = args.fg_resource_name if args.fg_resource_name else "fg.json"
    fg_path = os.path.join(tmp_dir, fg_name)
    with open(fg_path, "w") as f:
        json.dump(fg_json, f, indent=4)

    if args.fg_output_dir:
        shutil.copytree(tmp_dir, args.fg_output_dir, dirs_exist_ok=True)

    project = args.odps_project_name
    fg_resource_name = args.fg_resource_name
    if project is not None and fg_resource_name is not None:
        account, odps_endpoint = _create_odps_account()
        o = ODPS(
            account=account,
            project=project,
            endpoint=odps_endpoint,
            schema=args.odps_schema_name if args.odps_schema_name else None,
        )
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            if o.exist_resource(fname):
                if args.force_update_resource:
                    o.delete_resource(fname)
                    logger.info(
                        f"{fname} has already existed, will update this resource !"
                    )
                    resource = o.create_resource(
                        fname, "file", file_obj=open(fpath, "rb")
                    )
                else:
                    raise ValueError(
                        f"{fname} already existed in the {project}. "
                        f"You can add '--force_update_resource' in the command "
                        f"to force update."
                    )
            else:
                logger.info(f"uploading resource [{fname}].")
                resource = o.create_resource(fname, "file", file_obj=open(fpath, "rb"))

    if tmp_dir is None:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
