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

from tzrec.tools.tdm.gen_tree.tree_generator import TreeGenerator
from tzrec.tools.tdm.gen_tree.tree_search_util import TreeSearch
from tzrec.utils.logging_util import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item_input_path",
        type=str,
        default=None,
        help="The file path where the item information is stored.",
    )
    parser.add_argument(
        "--item_id_field",
        type=str,
        default=None,
        help="The column name representing item_id in the file.",
    )
    parser.add_argument(
        "--cate_id_field",
        type=str,
        default=None,
        help="The column name representing the category of item in the file.",
    )
    parser.add_argument(
        "--attr_fields",
        type=str,
        default=None,
        help="The column names representing the non-raw features of item in the file.",
    )
    parser.add_argument(
        "--raw_attr_fields",
        type=str,
        default=None,
        help="The column names representing the raw features of item in the file.",
    )
    parser.add_argument(
        "--tree_output_file",
        type=str,
        default=None,
        help="The tree output file.",
    )
    parser.add_argument(
        "--node_edge_output_file",
        type=str,
        default=None,
        help="The nodes and edges table output file.",
    )
    parser.add_argument(
        "--n_cluster",
        type=int,
        default=2,
        help="The branching factor of the nodes in the tree.",
    )
    args, extra_args = parser.parse_known_args()

    generator = TreeGenerator(
        item_input_path=args.item_input_path,
        item_id_field=args.item_id_field,
        cate_id_field=args.cate_id_field,
        attr_fields=args.attr_fields,
        raw_attr_fields=args.raw_attr_fields,
        tree_output_file=args.tree_output_file,
        n_cluster=args.n_cluster,
    )
    if args.tree_output_file:
        save_tree = True
    else:
        save_tree = False
    root = generator.generate(save_tree)
    logger.info("Tree init done. Start save nodes and edges table.")
    tree_search = TreeSearch(
        output_file=args.node_edge_output_file,
        root=root,
        child_num=args.n_cluster,
    )
    tree_search.save()
    tree_search.save_predict_edge()
    tree_search.save_serving_tree()
    logger.info("Save nodes and edges table done.")
