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

from tzrec.utils import faiss_util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_input_path",
        type=str,
        default=None,
        help="Path to embedding table.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size.",
    )
    parser.add_argument(
        "--index_output_dir",
        type=str,
        default=None,
        help="index output directory.",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="IVFFlatIP",
        choices=["IVFFlatIP", "HNSWFlatIP", "IVFFlatL2", "HNSWFlatL2"],
        help="index type.",
    )
    parser.add_argument(
        "--ivf_nlist", type=int, default=1000, help="nlist of IVFFlat index."
    )
    parser.add_argument("--hnsw_M", type=int, default=32, help="M of HNSWFlat index.")
    parser.add_argument(
        "--hnsw_efConstruction",
        type=int,
        default=200,
        help="efConstruction of HNSWFlat index.",
    )
    parser.add_argument("--id_field", type=str, default=None, help="id field name.")
    parser.add_argument(
        "--embedding_field", type=str, default=None, help="embedding field name."
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        default=None,
        choices=["OdpsReader", "CsvReader", "ParquetReader"],
        help="input path reader type.",
    )
    parser.add_argument(
        "--odps_data_quota_name",
        type=str,
        default="pay-as-you-go",
        help="maxcompute storage api/tunnel data quota name.",
    )
    args, extra_args = parser.parse_known_args()

    index, index_id_map = faiss_util.build_faiss_index(
        args.embedding_input_path,
        id_field=args.id_field,
        embedding_field=args.embedding_field,
        index_type=args.index_type,
        batch_size=args.batch_size,
        ivf_nlist=args.ivf_nlist,
        hnsw_M=args.hnsw_M,
        hnsw_efConstruction=args.hnsw_efConstruction,
        reader_type=args.reader_type,
        odps_data_quota_name=args.odps_data_quota_name,
    )

    faiss_util.write_faiss_index(index, index_id_map, args.index_output_dir)
