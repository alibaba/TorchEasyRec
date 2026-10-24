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

"""Import a built-in dynamicemb checkpoint into RocksDB native checkpoints.

Reads ``<checkpoint>/dynamicemb/<collection>/<table>_emb_*.rank_R.world_size_W``
in bounded chunks on CPU (no GPU, the table is never materialized), and
writes, for the same world size and key routing:

    <output_model_dir>/model.ckpt-<step>/   dense/optimizer/scheduler/dataloader
                                            state copied from the source, plus
                                            rocksdb/rank_<r>.json manifests,
                                            counters and _SUCCESS
    <local_path>/<run_id>/rank_<r>/ckpt-<step>/   one native checkpoint per rank

Optimizer state is converted with DynamicEmb's own checkpoint conversion.
Without optimizer files the manifest says ``exact_resume: false`` and the
tables start with reset optimizer state. Every dynamicemb table found is
imported; the target pipeline config must declare them all with ``rocksdb {}``.
Resume the result with ``fine_tune_checkpoint`` (or as model_dir with
``--continue_train``).
"""

import argparse
import functools
import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List

import numpy as np
import torch
from dynamicemb.batched_dynamicemb_tables import (
    encode_checkpoint_file_path,
    encode_counter_checkpoint_file_path,
)
from dynamicemb.key_value_table import (
    _iter_batches_from_files,
    resolve_checkpoint_value_layout,
)
from dynamicemb.optimizer import (
    AdaGradDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizer,
    DynamicEmbOptimType,
    EmbOptimType,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
    RowWiseAdaGradDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizer,
    string_to_opt_type,
)

from tzrec.utils import checkpoint_util, config_util, rocksdb_util
from tzrec.utils.logging_util import logger

_COPIED = (
    "model",
    "optimizer",
    "dense_ema",
    "plan",
    "meta",
    checkpoint_util.DATALOADER_CKPT_FILENAME,
    "lr_scheduler.json",
)
_KEYS_RE = re.compile(
    r"^(?P<table>.+)_emb_keys\.rank_(?P<rank>\d+)\.world_size_(?P<w>\d+)$"
)


def _optimizer(meta: Dict[str, Any], dtype: torch.dtype):
    opt_type = string_to_opt_type(meta["opt_type"])
    classes = {
        EmbOptimType.SGD: SGDDynamicEmbeddingOptimizer,
        EmbOptimType.EXACT_SGD: SGDDynamicEmbeddingOptimizer,
        EmbOptimType.ADAM: AdamDynamicEmbeddingOptimizer,
        EmbOptimType.EXACT_ADAGRAD: AdaGradDynamicEmbeddingOptimizer,
        DynamicEmbOptimType.FTRL: FTRLDynamicEmbeddingOptimizer,
    }
    if opt_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        optimizer = RowWiseAdaGradDynamicEmbeddingOptimizer(OptimizerArgs(), dtype)
    else:
        optimizer = classes[opt_type](OptimizerArgs())
    optimizer.set_opt_args(meta)
    return optimizer


def _find_tables(dynamicemb_dir: str) -> Dict[str, Dict[str, Any]]:
    """Map namespace -> {dir, table, world_size} of every dumped table."""
    tables: Dict[str, Dict[str, Any]] = {}
    for key_file in sorted(
        glob.glob(os.path.join(dynamicemb_dir, "*", "*_emb_keys.*"))
    ):
        m = _KEYS_RE.match(os.path.basename(key_file))
        if m is None:
            continue
        collection = os.path.basename(os.path.dirname(key_file))
        # Dump directories are named "model.<module path>"; namespaces use the
        # module path of the planned model, as in tzrec's export reader.
        module = (
            collection[len("model.") :]
            if collection.startswith("model.")
            else collection
        )
        namespace = f"{module}.{m.group('table')}"
        world = int(m.group("w"))
        entry = tables.setdefault(
            namespace,
            {
                "dir": os.path.dirname(key_file),
                "table": m.group("table"),
                "world_size": world,
            },
        )
        if entry["world_size"] != world:
            raise ValueError(f"{namespace} has inconsistent world sizes")
    return tables


def import_checkpoint(
    pipeline_config_path: str,
    checkpoint_path: str,
    output_model_dir: str,
    include_optim: bool = True,
    chunk_rows: int = 65536,
) -> str:
    """Import one checkpoint; returns the output checkpoint directory."""
    from tzrec_rocksdb import open_session
    from tzrec_rocksdb.checkpoint import create_checkpoint

    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    train_config = pipeline_config.train_config
    if not train_config.HasField("rocksdb_config"):
        raise ValueError(
            "the target pipeline config has no train_config.rocksdb_config"
        )
    rocksdb_util.set_rocksdb_config(train_config.rocksdb_config)
    step = checkpoint_util._get_checkpoint_step(checkpoint_path)
    out_ckpt = os.path.join(output_model_dir, f"model.ckpt-{step}")
    if os.path.exists(out_ckpt):
        raise FileExistsError(f"{out_ckpt} already exists")
    tables = _find_tables(os.path.join(checkpoint_path, "dynamicemb"))
    if not tables:
        raise ValueError(f"{checkpoint_path} has no dynamicemb tables")
    world_size = next(iter(tables.values()))["world_size"]
    if any(t["world_size"] != world_size for t in tables.values()):
        raise ValueError("dynamicemb tables were dumped with different world sizes")
    os.makedirs(os.path.join(out_ckpt, rocksdb_util.MANIFEST_DIR), exist_ok=True)
    exact = include_optim
    for rank in range(world_size):
        root = rocksdb_util.rank_dir(output_model_dir, rank)
        work = os.path.join(root, f"import-{step}")
        shutil.rmtree(work, ignore_errors=True)
        session = open_session(rocksdb_util.backend_config(), work)
        ids = session.namespace_ids(tables.keys())
        groups: List[Dict[str, Any]] = []
        for namespace, info in tables.items():
            path = functools.partial(
                encode_checkpoint_file_path,
                info["dir"],
                info["table"],
                rank,
                world_size,
            )
            with open(os.path.join(info["dir"], f"{info['table']}_opt_args.json")) as f:
                meta = json.load(f)
            num_keys = os.path.getsize(path("keys")) // 8
            emb_dtype, opt_dtype, dim = resolve_checkpoint_value_layout(
                meta,
                path("values"),
                num_keys,
                int(meta.get("embedding_dim", 0)) or None,
            )
            optimizer = _optimizer(meta, emb_dtype)
            state_dim = optimizer.get_state_dim(dim)
            has_opt = (
                state_dim > 0 and include_optim and os.path.exists(path("opt_values"))
            )
            exact = exact and (state_dim == 0 or has_opt)
            elem = torch.empty(0, dtype=emb_dtype).element_size()
            layout = {
                "dtype": str(emb_dtype).replace("torch.", ""),
                "dim": dim,
                "state_dim": state_dim,
                "opt_type": optimizer.get_opt_args().get("opt_type"),
                "record_bytes": 8 + (dim + state_dim) * elem,
            }
            session.set_table_meta(namespace, layout)
            native_layout = np.array(
                [[ids[namespace], dim * elem, state_dim * elem, dim * elem]],
                dtype=np.int64,
            )
            dist_type = meta.get("dist_type", "roundrobin")
            rows = 0
            cpu = torch.device("cpu")
            for keys, emb, scores, opt in _iter_batches_from_files(
                path("keys"),
                path("values"),
                path("scores") if os.path.exists(path("scores")) else None,
                path("opt_values") if has_opt else None,
                dim,
                optimizer.get_ckpt_state_dim(dim) if has_opt else 0,
                cpu,
                dist_type,
                emb_dtype=emb_dtype,
                opt_state_dtype=opt_dtype,
                shard_index=rank,
                num_shards=world_size,
            ):
                m = keys.numel()
                if m == 0:
                    continue
                row = torch.empty(m, dim + state_dim, dtype=emb_dtype)
                row[:, :dim] = emb.view(m, dim)
                if state_dim > 0:
                    if opt is not None:
                        row[:, dim:] = optimizer.states_from_checkpoint(
                            opt, dim, emb_dtype, cpu
                        ).view(m, state_dim)
                    else:
                        optimizer.reset_optimizer_states(row[:, dim:], emb_dims=dim)
                session.db.write(
                    native_layout,
                    np.zeros(m, dtype=np.int64),
                    keys.numpy().view(np.uint64),
                    row.view(torch.uint8).numpy(),
                    scores.view(-1).to(torch.int64).numpy()
                    if scores is not None
                    else None,
                    False,
                )
                rows += m
            counter = False
            counter_keys = encode_counter_checkpoint_file_path(
                info["dir"], info["table"], rank, world_size, "keys"
            )
            if os.path.exists(counter_keys):
                dst = rocksdb_util.counter_paths(out_ckpt, namespace, rank)
                os.makedirs(os.path.dirname(dst[0]), exist_ok=True)
                shutil.copyfile(counter_keys, dst[0])
                shutil.copyfile(
                    encode_counter_checkpoint_file_path(
                        info["dir"], info["table"], rank, world_size, "frequencies"
                    ),
                    dst[1],
                )
                counter = True
            groups.append(
                {
                    "tables": [info["table"]],
                    "namespaces": [namespace],
                    "layouts": {namespace: layout},
                    "dist_types": [dist_type],
                    "num_shards": world_size,
                    "shard_index": rank,
                    "opt_args": optimizer.get_opt_args(),
                    "scores": {info["table"]: meta["step_score"]}
                    if "step_score" in meta
                    else {},
                    "counter": counter,
                }
            )
            logger.info(f"rank {rank}: imported {rows} rows of {namespace}")
        native = os.path.join(root, f"ckpt-{step}")
        shutil.rmtree(native, ignore_errors=True)
        native_manifest = create_checkpoint(session, native, {"step": step})
        session.close()
        shutil.rmtree(work, ignore_errors=True)
        manifest = {
            "format_version": rocksdb_util.FORMAT_VERSION,
            "step": step,
            "rank": rank,
            "world_size": world_size,
            "native_checkpoint": native,
            "native_manifest": {
                k: native_manifest[k]
                for k in ("num_sst", "num_sst_hard_linked", "bytes", "files")
            },
            "imported": True,
            "imported_from": os.path.realpath(checkpoint_path),
            "groups": groups,
        }
        with open(
            os.path.join(out_ckpt, rocksdb_util.MANIFEST_DIR, f"rank_{rank}.json"), "w"
        ) as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    for name in _COPIED:
        src = os.path.join(checkpoint_path, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out_ckpt, name))
        elif os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_ckpt, name))
    for rank in range(world_size):
        path = os.path.join(out_ckpt, rocksdb_util.MANIFEST_DIR, f"rank_{rank}.json")
        with open(path) as f:
            manifest = json.load(f)
        manifest["exact_resume"] = exact
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    with open(os.path.join(out_ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME), "w"):
        pass
    logger.info(f"imported {checkpoint_path} into {out_ckpt} (exact_resume={exact})")
    return out_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        required=True,
        help="target pipeline config (its train_config.rocksdb_config is used).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="source model.ckpt-<step> directory with a dynamicemb/ dump.",
    )
    parser.add_argument(
        "--output_model_dir", type=str, required=True, help="output model directory."
    )
    parser.add_argument(
        "--no_optim",
        action="store_true",
        help="import embeddings only (optimizer state reset; not an exact resume).",
    )
    args, _ = parser.parse_known_args()
    import_checkpoint(
        args.pipeline_config_path,
        args.checkpoint_path,
        args.output_model_dir,
        include_optim=not args.no_optim,
    )
