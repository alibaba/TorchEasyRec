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

from tzrec.datasets.dataset import create_reader
from tzrec.main import predict
from tzrec.tests.utils import create_predict_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripted_model_path",
        type=str,
        default=None,
        help="scripted model to infer",
    )
    parser.add_argument(
        "--item_id",
        type=str,
        default=None,
        help="item id name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="online predict batch size.",
    )

    args, extra_args = parser.parse_known_args()
    if args.item_id is None:
        assert args.batch_size == 1, (
            "embedding online predict only support batch_size=1"
        )

    data_dir = os.path.join(args.scripted_model_path, "debug_data")
    result_dir = os.path.join(args.scripted_model_path, "debug_result")

    create_predict_data(
        os.path.join(args.scripted_model_path, "pipeline.config"),
        batch_size=args.batch_size,
        item_id=args.item_id,
        output_dir=data_dir,
    )

    predict(
        scripted_model_path=args.scripted_model_path,
        predict_input_path=os.path.join(data_dir, "*.parquet"),
        predict_output_path=result_dir,
        batch_size=args.batch_size,
        debug_level=1,
        dataset_type="ParquetDataset",
        reserved_columns=args.item_id,
    )

    reader = create_reader(
        os.path.join(result_dir, "*.parquet"), batch_size=args.batch_size
    )
    result_json = {}
    output_json = {}
    for data in reader.to_batches():
        if args.item_id is None:
            item_id_data = ["mock_id"]
        else:
            item_id_data = data[args.item_id]
        for iid, features in zip(item_id_data, data["__features__"]):
            x = str(features)
            value_parts = [part.split(":") for part in x.split(" | ")]
            value_dict = {k: v for k, v in value_parts}
            output_json[str(iid)] = value_dict
        for k, v in data.items():
            if k not in [args.item_id, "__features__"]:
                result_json[k] = v.tolist()

    with open(os.path.join(result_dir, "fgout.json"), "w") as f:
        json.dump(output_json, f, indent=4)
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_json, f, indent=4)
