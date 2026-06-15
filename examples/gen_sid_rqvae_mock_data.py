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

"""Write a mock embedding parquet for the SidRqvae Gumbel+CLIP smoke config.

Columns match ``examples/sid_rqvae_gumbel_clip_local.config``:
    item1_embedding  (list<float32>[dim])  -- the SID input embedding
    item2_embedding  (list<float32>[dim])  -- the CLIP-paired embedding
    is_contrastive   (float32 scalar)      -- 1.0 = CLIP pair, 0.0 = recon-only

Usage:
    python examples/gen_sid_rqvae_mock_data.py --out_dir ./tmp/sid_rqvae_mock \
        --num_rows 4096 --dim 512 --clip_ratio 0.5
"""

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    """Generate the mock parquet shard."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./tmp/sid_rqvae_mock")
    parser.add_argument("--num_rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.5,
        help="fraction of rows flagged as CLIP pairs (is_contrastive=1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    item1 = rng.standard_normal((args.num_rows, args.dim)).astype(np.float32)
    # item2 is a noisy view of item1 so the contrastive pairs are learnable.
    item2 = (item1 + 0.1 * rng.standard_normal(item1.shape)).astype(np.float32)
    is_clip = (rng.random(args.num_rows) < args.clip_ratio).astype(np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "part-0.parquet")
    pq.write_table(
        pa.table(
            {
                "item1_embedding": pa.array(list(item1)),
                "item2_embedding": pa.array(list(item2)),
                "is_contrastive": pa.array(is_clip),
            }
        ),
        out_path,
    )
    print(
        f"wrote {args.num_rows} rows (dim={args.dim}, "
        f"clip_pairs={int(is_clip.sum())}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
