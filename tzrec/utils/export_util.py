# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import glob
import json
import operator
import os
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torchrec import JaggedTensor, KeyedTensor
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.utils import Tracer
from torchrec.inference.modules import quantize_embeddings
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    quant_prep_enable_cache_features_order,
)
from torchrec.sparse import jagged_tensor

from tzrec.acc import utils as acc_utils
from tzrec.acc.aot_utils import export_model_aot, export_unified_model_aot
from tzrec.acc.trt_utils import export_model_trt
from tzrec.constant import TARGET_REPEAT_INTERLEAVE_KEY, Mode
from tzrec.datasets.data_parser import _tile_size
from tzrec.datasets.dataset import (
    create_dataloader,
)
from tzrec.features.feature import (
    BaseFeature,
    create_feature_configs,
    create_fg_json,
)
from tzrec.modules.utils import BaseModule
from tzrec.protos import model_pb2
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import checkpoint_util, config_util, env_util
from tzrec.utils.dist_util import DistributedModelParallel, init_process_group
from tzrec.utils.filesystem_util import url_to_fs
from tzrec.utils.fx_util import (
    fx_mark_keyed_tensor,
    fx_mark_seq_ec_jt,
    fx_mark_seq_len,
    fx_mark_seq_tensor,
    fx_mark_tensor,
    symbolic_trace,
)
from tzrec.utils.logging_util import logger
from tzrec.utils.plan_util import create_planner, get_default_sharders
from tzrec.utils.state_dict_util import fix_mch_state, init_parameters


def ensure_input_tile_for_distributed_embedding() -> None:
    """Ensure distributed embedding export uses INPUT_TILE=3."""
    if not env_util.use_distributed_embedding():
        return

    # Distributed embedding export only supports INPUT_TILE=3.
    # Default to 3 when unset; warn and override if set to anything else.
    current_input_tile = os.environ.get("INPUT_TILE")
    if current_input_tile is None:
        os.environ["INPUT_TILE"] = "3"
    elif current_input_tile != "3":
        logger.warning(
            "USE_DISTRIBUTED_EMBEDDING=1 requires INPUT_TILE=3, "
            f"got INPUT_TILE={current_input_tile}. Overriding to 3."
        )
        os.environ["INPUT_TILE"] = "3"


def _is_input_tile_user_keyed_tensor(name: str) -> bool:
    """Whether a split KeyedTensor is the INPUT_TILE=3 user sparse side."""
    return name.endswith("__ebc_user") or name.endswith("__mc_ebc_user")


def _dedup_key_files_by_realpath(key_files: List[str]) -> List[str]:
    """Keep one dynamicemb key shard path for each physical file.

    Normal training checkpoints do not create INPUT_TILE user-side dynamicemb
    aliases, but externally staged checkpoints may contain symlinked or aliased
    table directories. Since export discovers dynamicemb shards by globbing the
    checkpoint directory, loading the same physical shard twice would duplicate
    its keys and values in the exported dynamic embedding data.
    """
    seen_real: Set[str] = set()
    deduped_key_files = []
    for key_file in key_files:
        real = os.path.realpath(key_file)
        if real in seen_real:
            continue
        seen_real.add(real)
        deduped_key_files.append(key_file)
    return deduped_key_files


def export_model(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
    additional_export_config: Optional[Dict[str, Union[bool, str]]] = None,
    data_input_path: Optional[str] = None,
) -> None:
    """Export a EasyRec model, may be a part of model in PipelineConfig.

    `data_input_path` (optional): override for the predict-mode dataloader
    input path; falls back to `pipeline_config.train_input_path` when None.
    """
    use_rtp = env_util.use_rtp()
    use_dist_embedding = env_util.use_distributed_embedding()
    if use_rtp:
        impl = export_rtp_model
    elif use_dist_embedding:
        impl = export_distributed_embedding
    else:
        impl = export_model_normal
    fs, local_path = url_to_fs(save_dir)
    use_local_cache_dir = False
    if fs is not None:
        # scripted model and safetensors use io in cpp,
        # so that we can not use fsspec to patch cpp io operations.
        local_path = os.environ.get("LOCAL_CACHE_DIR", local_path)
        use_local_cache_dir = True
    impl(
        pipeline_config=pipeline_config,
        model=model,
        checkpoint_path=checkpoint_path,
        save_dir=local_path,
        assets=assets,
        use_local_cache_dir=use_local_cache_dir,
        additional_export_config=additional_export_config,
        data_input_path=data_input_path,
    )
    if use_local_cache_dir and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"uploading {local_path} to {save_dir}.")
        fs.upload(local_path, save_dir, recursive=True, file_thread_num=os.cpu_count())
        logger.info(f"finish upload {local_path} to {save_dir}.")
        shutil.rmtree(local_path)


def _move_quantized_modules_to_device(model: nn.Module, device: torch.device) -> None:
    """Move quantized fbgemm modules to device after CPU quantization.

    torchrec stores fbgemm IntNBitTableBatchedEmbeddingBagsCodegen in a
    plain list (_emb_modules), not nn.ModuleList, so model.cuda() skips
    them. Use fbgemm's move_to_device_with_cache() which properly handles
    weight migration, placement metadata, row alignment, and buffer views.

    Must use cache_load_factor=1.0 to place weights in device HBM
    (EmbeddingLocation.DEVICE). The default 0.0 places weights in UVM
    (MANAGED) which is incompatible with the inference forward path.
    """
    from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
        IntNBitTableBatchedEmbeddingBagsCodegen,
    )

    for m in model.modules():
        if not hasattr(m, "_emb_modules"):
            continue
        for emb in m._emb_modules:
            if (
                isinstance(emb, IntNBitTableBatchedEmbeddingBagsCodegen)
                and emb.current_device != device
            ):
                emb.move_to_device_with_cache(device, 1.0)


def export_model_normal(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
    additional_export_config: Optional[Dict[str, Union[bool, str]]] = None,
    data_input_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Export a EasyRec model on aliyun."""
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    if not is_rank_zero:
        logger.warning("Only first rank will be used for export now.")
        return
    else:
        if os.environ.get("WORLD_SIZE") != "1":
            logger.warning(
                "export only support WORLD_SIZE=1 now, we set WORLD_SIZE to 1."
            )
            os.environ["WORLD_SIZE"] = "1"

    if not dist.is_initialized():
        dist.init_process_group("gloo")

    # make dataparser to get user feats before create model
    data_config = copy.deepcopy(pipeline_config.data_config)
    features = cast(List[BaseFeature], model.features)
    if acc_utils.is_cuda_export():
        # export batch_size too large may OOM in compile phase
        max_batch_size = acc_utils.get_max_export_batch_size()
        data_config.batch_size = min(data_config.batch_size, max_batch_size)
        logger.info("using new batch_size: %s in export", data_config.batch_size)
    data_config.num_workers = 1
    input_path = data_input_path or pipeline_config.train_input_path
    dataloader = create_dataloader(data_config, features, input_path, mode=Mode.PREDICT)

    ckpt_param_map_path = None
    if checkpoint_path:
        if acc_utils.is_input_tile_emb():
            # generate embedding name mapping file
            ckpt_param_map_path = os.path.join(save_dir, "emb_ckpt_mapping.txt")
            if is_rank_zero:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                acc_utils.write_mapping_file_for_input_tile(
                    model.state_dict(), ckpt_param_map_path
                )
                dist.barrier()
    else:
        raise ValueError("checkpoint path should be specified.")

    if is_rank_zero:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        init_parameters(model, torch.device("cpu"))
        checkpoint_util.restore_model(
            checkpoint_path, model, ckpt_param_map_path=ckpt_param_map_path
        )
        # for mc modules, fix output_segments_tensor is a meta tensor.
        fix_mch_state(model)

        batch = next(dataloader.get_iterator())  # pyre-ignore[16]

        # Quantize on CPU before moving to CUDA. The fbgemm CUDA kernel
        # for nbit quantization uses int32 pointer arithmetic which
        # overflows for large embedding tables. The CPU kernel uses
        # int64 and is safe for any table size.
        if acc_utils.is_quant() or acc_utils.is_ec_quant():
            logger.info("quantize embeddings...")
            additional_qconfig_spec_keys = []
            additional_mapping = {}
            cache_order_types = [EmbeddingBagCollection]
            if acc_utils.is_ec_quant():
                additional_qconfig_spec_keys.append(EmbeddingCollection)
                additional_mapping[EmbeddingCollection] = QuantEmbeddingCollection
                cache_order_types.append(EmbeddingCollection)
            # Cache the feature-permute order as an on-device buffer instead of
            # rebuilding `torch.tensor(order, device=cuda)` (a blocking H2D copy)
            # on every forward. Must run before quantize_embeddings so the quant
            # modules pick it up via `from_float`.
            quant_prep_enable_cache_features_order(model, cache_order_types)
            quantize_embeddings(
                model,
                dtype=acc_utils.quant_dtype(),
                inplace=True,
                additional_qconfig_spec_keys=additional_qconfig_spec_keys,
                additional_mapping=additional_mapping,
            )
            logger.info("finish quantize embeddings...")

        if acc_utils.is_cuda_export():
            _move_quantized_modules_to_device(model, torch.device("cuda:0"))
            model = model.cuda()

        model.eval()

        data = batch.to_dict(sparse_dtype=torch.int64)
        mixed_precision = acc_utils.mixed_precision_for_export(pipeline_config)
        autocast_dtype = acc_utils.mixed_precision_to_dtype(mixed_precision)
        if acc_utils.is_trt() or acc_utils.is_aot():
            data = OrderedDict(sorted(data.items()))
            # no_grad: avoid retaining activations that OOM AOTI autotune.
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ),
            ):
                result = model(data, "cuda:0")
            result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
            logger.info(f"Model Outputs: {result_info}")
            del result
            torch.cuda.empty_cache()
            if acc_utils.is_trt():
                sparse, dense, meta_info = split_model(data, model, save_dir)
                export_model_trt(
                    sparse, dense, data, save_dir, mixed_precision=mixed_precision
                )
            elif acc_utils.is_unified_aot():
                export_unified_model_aot(
                    model, data, save_dir, mixed_precision=mixed_precision
                )
            else:
                sparse, dense, meta_info = split_model(data, model, save_dir)
                export_model_aot(
                    sparse,
                    dense,
                    data,
                    meta_info,
                    save_dir,
                    mixed_precision=mixed_precision,
                )
        else:
            result = model(data)
            result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
            logger.info(f"Model Outputs: {result_info}")

            gm = symbolic_trace(model)
            with open(os.path.join(save_dir, "gm.code"), "w") as f:
                f.write(gm.code)

            scripted_model = torch.jit.script(gm)
            scripted_model.save(os.path.join(save_dir, "scripted_model.pt"))

        feature_configs = create_feature_configs(features, asset_dir=save_dir)
        pipeline_config = copy.copy(pipeline_config)
        pipeline_config.ClearField("feature_configs")
        pipeline_config.feature_configs.extend(feature_configs)
        config_util.save_message(
            pipeline_config, os.path.join(save_dir, "pipeline.config")
        )
        logger.info("saving fg json...")
        fg_json = create_fg_json(features, asset_dir=save_dir)
        with open(os.path.join(save_dir, "fg.json"), "w") as f:
            json.dump(fg_json, f, indent=4)
        with open(os.path.join(save_dir, "model_acc.json"), "w") as f:
            json.dump(
                acc_utils.export_acc_config(
                    additional_export_config=additional_export_config
                ),
                f,
                indent=4,
            )

        if assets is not None:
            for asset in assets:
                shutil.copy(asset, save_dir)


def _prepare_single_rank_distributed_embedding_export() -> bool:
    """Force distributed-embedding export to run as a single rank."""
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        logger.warning(
            "Only first rank will be used for distributed embedding export now."
        )
        return False

    forced_env = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
    }
    changed = [
        f"{key}={value}"
        for key, value in forced_env.items()
        if os.environ.get(key) != value
    ]
    if changed:
        logger.warning(
            "distributed embedding export only supports single-rank export now, "
            "we set %s.",
            ", ".join(changed),
        )
    os.environ.update(forced_env)
    return True


def _get_sharded_leaf_module_names(model: torch.nn.Module) -> List[str]:
    """Get ShardedModule as leaf modules."""

    def _get_leaf_module_names_helper(
        model: torch.nn.Module,
        path: str,
        leaf_module_names: Set[str],
    ) -> bool:
        if isinstance(model, ShardedModule):
            leaf_module_names.add(path)
        else:
            for name, child in model.named_children():
                _get_leaf_module_names_helper(
                    child,
                    name if path == "" else path + "." + name,
                    leaf_module_names,
                )

    leaf_module_names: Set[str] = set()
    _get_leaf_module_names_helper(
        model,
        "",
        leaf_module_names,
    )
    return list(leaf_module_names)


def _get_dense_embedding_leaf_module_names(model: torch.nn.Module) -> List[str]:
    """Get dense-embedding modules to keep as FX leaf modules during export.

    ``AutoDisEmbedding`` / ``MLPEmbedding`` override ``state_dict`` /
    ``_load_from_state_dict`` with split per-feature names. Tracing them as
    leaves keeps the class so ``reset_parameters`` and the custom restore run
    after the graph is flattened; otherwise their params stay uninitialized.
    """
    from tzrec.modules.dense_embedding_collection import (
        AutoDisEmbedding,
        MLPEmbedding,
    )

    names = []
    for path, module in model.named_modules():
        if isinstance(module, (AutoDisEmbedding, MLPEmbedding)):
            names.append(path)
    return names


def _get_rtp_feature_to_embedding_info(
    model: nn.Module,
) -> Dict[str, BaseEmbeddingConfig]:
    feature_to_embedding_info = dict()
    feature_to_module_path = dict()
    emb_name_to_module_path = dict()
    q = Queue()
    q.put(("", model))
    while not q.empty():
        child_path, m = q.get()
        if isinstance(m, EmbeddingBagCollectionInterface) or isinstance(
            m, EmbeddingCollectionInterface
        ):
            embedding_configs = (
                m.embedding_bag_configs()
                if isinstance(m, EmbeddingBagCollectionInterface)
                else m.embedding_configs()
            )
            for t in embedding_configs:
                if t.name in emb_name_to_module_path:
                    raise RuntimeError(
                        f"RTP do not support embedding name [{t.name}] used in "
                        f"two modules now. [{emb_name_to_module_path[t.name]}] vs "
                        f"{child_path}, you should specify new embedding names for "
                        "feature in conflict feature group."
                    )
                emb_name_to_module_path[t.name] = child_path
                for fname in t.feature_names:
                    if fname in feature_to_module_path:
                        raise RuntimeError(
                            f"RTP do not support feature [{fname}] used in two modules "
                            f"now. [{feature_to_module_path[fname]}] vs {child_path}, "
                            f"you should create a new feature with same feature_config "
                            f"as [{fname}] and use it in conflict feature group."
                        )
                    feature_to_embedding_info[fname] = t
                    feature_to_module_path[fname] = child_path
        else:
            for name, child in m.named_children():
                if child_path == "":
                    q.put((name, child))
                else:
                    q.put((f"{child_path}.{name}", child))
    return feature_to_embedding_info


def _add_module_by_dotted_path(
    root: nn.Module, dotted_path: str, module: nn.Module
) -> None:
    """Add a module to dotted_path in root model."""
    parts = dotted_path.split(".")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            parent.add_module(p, nn.Module())  # create empty container
        parent = getattr(parent, p)
    parent.add_module(parts[-1], module)


def _prune_unused_param_and_buffer(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Prune unused parameters and buffers in GraphModule."""
    new_root = nn.Module()
    name_to_obj = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            module_path = node.target
            if module_path not in name_to_obj:
                submodule = gm.get_submodule(module_path)
                _add_module_by_dotted_path(new_root, module_path, submodule)
                name_to_obj[module_path] = submodule
        elif node.op == "get_attr":
            param_path = node.target
            module_path, _, param_name = param_path.rpartition(".")

            if param_path not in name_to_obj:
                if module_path == "":
                    submodule = gm
                elif module_path not in name_to_obj:
                    submodule = gm.get_submodule(module_path)
                    _add_module_by_dotted_path(new_root, module_path, submodule)
                    name_to_obj[module_path] = submodule
                else:
                    submodule = name_to_obj[module_path]
                current_obj = getattr(submodule, param_name)
                parent = new_root.get_submodule(module_path)

                if isinstance(current_obj, nn.Parameter):
                    parent.register_parameter(param_name, current_obj)
                elif isinstance(current_obj, torch.Tensor):  # It's a buffer
                    parent.register_buffer(param_name, current_obj)
                name_to_obj[param_path] = current_obj

    new_gm = torch.fx.GraphModule(new_root, gm.graph)
    return new_gm


def _get_rtp_embedding_tensor(
    model: nn.Module, checkpoint_path: str, embedding_infos: List[BaseEmbeddingConfig]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Get Embedding Tensors for RTP."""
    emb_name_to_emb_dim = dict()
    for emb_info in embedding_infos:
        emb_name_to_emb_dim[emb_info.name] = emb_info.embedding_dim

    def _remove_prefix(src: str, prefix: str = "torch.") -> str:
        if src.startswith(prefix):
            return src[len(prefix) :]
        return src

    out = {}
    value_name_to_key = {}
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    for name, values in model.state_dict().items():
        emb_name = name.split(".")[-2]
        emb_dim = emb_name_to_emb_dim[emb_name]
        if isinstance(values, DTensor):
            raise ValueError("DTensors are not considered yet.")
        elif isinstance(values, ShardedTensor):
            _len_local_shards = len(values.local_shards())
            assert _len_local_shards in [0, 1], "other cases are not considered."
            num_shards = len(values.metadata().shards_metadata)
            if _len_local_shards == 1:
                for idx, shards_meta in enumerate(values.metadata().shards_metadata):
                    placement = shards_meta.placement
                    assert placement is not None
                    if placement.rank() == rank:
                        name = name + f"/part_{idx}_{num_shards}"
                        local_tensor = values.local_tensor()
                        if list(local_tensor.shape)[-1] == emb_dim:
                            # dynamicemb may have a dummy tensor in state_dict, skip it.
                            out[name] = local_tensor
                            value_name_to_key[name] = None
        elif list(values.shape)[-1] == emb_dim:
            # dynamicemb may have a dummy tensor in state_dict, skip it.
            out[name] = values
            value_name_to_key[name] = None

    dynamicemb_path = os.path.join(checkpoint_path, "dynamicemb")
    if os.path.exists(dynamicemb_path):
        key_files = sorted(
            glob.glob(os.path.join(dynamicemb_path, "*/*_emb_keys.rank_*.world_size_*"))
        )
        key_files = _dedup_key_files_by_realpath(key_files)
        key_pattern = re.compile(
            r"^(?P<emb_name>.+)_emb_keys\.rank_(?P<idx>\d+)\.world_size_(?P<num_shards>\d+)$"
        )
        for i in range(rank, len(key_files), world_size):
            key_file = key_files[i]
            path_parts = key_file.split(os.path.sep)
            match = key_pattern.match(path_parts[-1])
            if match:
                emb_name = match.group("emb_name")
                emb_dim = emb_name_to_emb_dim[emb_name]
                idx = match.group("idx")
                num_shards = match.group("num_shards")
                with open(key_file, "rb") as f:
                    keys = torch.tensor(
                        np.fromfile(f, dtype=np.int64), dtype=torch.int64
                    )
                with open(
                    os.path.join(
                        os.path.dirname(key_file),
                        f"{emb_name}_emb_values.rank_{idx}.world_size_{num_shards}",
                    ),
                    "rb",
                ) as f:
                    values = torch.tensor(
                        np.fromfile(f, dtype=np.float32), dtype=torch.float32
                    )
                key_name = f"{path_parts[-2]}.{emb_name}.keys/part_{idx}_{num_shards}"
                value_name = (
                    f"{path_parts[-2]}.{emb_name}.values/part_{idx}_{num_shards}"
                )
                out[key_name] = keys
                out[value_name] = values.view([-1, emb_dim])
                value_name_to_key[value_name] = key_name

    # TODO(hongsheng.jhs): support mczch

    meta = {}
    for name, key_name in value_name_to_key.items():
        values = out[name]
        dimension = list(values.shape)[-1]
        dtype = _remove_prefix(str(values.dtype))
        assert dtype == "float32", "RTP only support float32 sparse weights now."
        memory: int = int(values.nbytes)
        shape = list(values.shape)
        t_meta = {
            "name": name,
            "dense": False,
            "dimension": dimension,
            "dtype": dtype,
            "memory": memory,
            "shape": shape,
        }
        if key_name is not None:
            t_meta["hashmap_value"] = name
            t_meta["hashmap_key"] = key_name
            t_meta["hashmap_key_dtype"] = "int64"
            t_meta["is_hashmap"] = True
        else:
            t_meta["is_hashmap"] = False
        meta[name] = t_meta
    return out, meta


RTP_INVALID_BUCKET_KEYS = ["vocab_dict", "vocab_list", "vocab_file"]


def _adjust_one_feature_for_rtp(
    feature: Dict[str, Any], embedding_info: Optional[BaseEmbeddingConfig]
) -> None:
    assert feature["feature_type"] in [
        "id_feature",
        "raw_feature",
        "combo_feature",
        "match_feature",
        "lookup_feature",
        "overlap_feature",
    ]
    if embedding_info is not None:
        feature["shared_name"] = embedding_info.name
        feature["embedding_dimension"] = embedding_info.embedding_dim
        feature["gen_val_type"] = "lookup"
    else:
        feature["gen_val_type"] = "idle"

    if "value_dim" in feature:
        feature["value_dimension"] = feature["value_dim"]
        feature.pop("value_dim")
    if "need_discrete" in feature:
        feature["needDiscrete"] = feature["need_discrete"]
        feature.pop("need_discrete")
    if "boundaries" in feature:
        feature["boundaries"] = ",".join(map(str, feature["boundaries"]))
        feature["gen_key_type"] = "boundary"
    elif "hash_bucket_size" in feature:
        feature["gen_key_type"] = "hash"
    elif "num_buckets" in feature:
        # in RTP, hash_bucket_size and gen_key_type=mod equal to num_buckets
        feature["gen_key_type"] = "mod"
        feature["hash_bucket_size"] = feature["num_buckets"]
        feature.pop("num_buckets")
    else:
        for k in RTP_INVALID_BUCKET_KEYS:
            if k in feature:
                raise ValueError(f"{k} is not supported when use rtp.")
        feature["gen_key_type"] = "idle"


def _adjust_fg_json_for_rtp(
    fg_json: Dict[str, Any], feature_to_embedding_info: Dict[str, BaseEmbeddingConfig]
) -> None:
    """Adjust fg json to rtp style."""
    for feature in fg_json["features"]:
        if "features" not in feature:
            feature_name = feature["feature_name"]
            embedding_info = feature_to_embedding_info.get(feature_name, None)
            _adjust_one_feature_for_rtp(feature, embedding_info)
        else:
            sequence_name = feature["sequence_name"]
            if "sequence_table" not in feature:
                feature["sequence_table"] = "item"
            for sub_feature in feature["features"]:
                feature_name = sub_feature["feature_name"]
                embedding_info = feature_to_embedding_info.get(
                    f"{sequence_name}_{feature_name}", None
                )
                _adjust_one_feature_for_rtp(sub_feature, embedding_info)


@torch.fx.wrap
def _rtp_pad_to_max_seq_len(x: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    pad_len = max_seq_len - x.size(1)
    x_padded = F.pad(x, (0, 0, 0, pad_len))
    return x_padded


@torch.fx.wrap
def _rtp_slice_with_seq_len(
    x: torch.Tensor, seq_len: torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    seq_len_int = seq_len.max().item()
    torch._check(seq_len_int >= 0)
    torch._check(seq_len_int <= max_seq_len)
    return x[:, :seq_len_int, :]


@torch.fx.wrap
def _rtp_torch_asynchronous_complete_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(x), x])


@torch.fx.wrap
def _rtp_torch_jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_lengths: List[int],
    padding_value: float = 0,
) -> torch.Tensor:
    values = values.unsqueeze(0)
    # dummy pad for fix:
    #   Could not guard on data-dependent expression
    #   Eq((u12//(u6 + u7 + 6)), 1)
    pad_len = max_lengths[0] - values.size(1)
    return F.pad(values, (0, 0, 0, pad_len))


@torch.fx.wrap
def _rtp_torch_dense_to_jagged(
    dense: torch.Tensor, offsets: List[torch.Tensor], total_L: Optional[int] = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    return dense.squeeze(0), offsets


@torch.fx.wrap
def _rtp_torch_jagged_dense_elementwise_add_jagged_output(
    x_values: torch.Tensor, x_offsets: List[torch.Tensor], y: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    return x_values + y.squeeze(0), x_offsets


FBGEMM_RTP_TORCH_OP_MAPPING = {
    torch.ops.fbgemm.asynchronous_complete_cumsum: _rtp_torch_asynchronous_complete_cumsum,  # NOQA
    torch.ops.fbgemm.jagged_to_padded_dense: _rtp_torch_jagged_to_padded_dense,
    torch.ops.fbgemm.dense_to_jagged: _rtp_torch_dense_to_jagged,
    torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output: _rtp_torch_jagged_dense_elementwise_add_jagged_output,  # NOQA
}


def export_rtp_model(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
    use_local_cache_dir: bool = False,
    data_input_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Export a EasyRec model on RTP."""
    try:
        from torch_fx_tool import ExportTorchFxTool
    except Exception as e:
        raise RuntimeError(
            "torch_fx_tool not exist. please install https://tzrec.oss-accelerate.aliyuncs.com/third_party/rtp/torch_fx_tool-0.0.1%2B20251201.8c109c4-py3-none-any.whl"
        ) from e

    device, _ = init_process_group()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_rank_zero = rank == 0
    is_local_rank_zero = local_rank == 0
    train_config = pipeline_config.train_config

    # RTP do not support fbgemm now. patch kt.regroup to slow path
    def _all_keys_used_once(
        keyed_tensors: List["KeyedTensor"], groups: List[List["str"]]
    ) -> bool:
        return False

    jagged_tensor._all_keys_used_once = _all_keys_used_once

    feature_to_embedding_info = _get_rtp_feature_to_embedding_info(model)

    graph_dir = os.path.join(save_dir, "graph")
    if is_rank_zero or (use_local_cache_dir and is_local_rank_zero):
        # when caching save_dir on local_rank, we need make save_dir
        # for each worker
        if not os.path.exists(save_dir):
            logger.info(f"make save dir {save_dir}")
            os.makedirs(save_dir)
    if is_rank_zero:
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
    dist.barrier()

    if not checkpoint_path:
        raise ValueError("checkpoint path should be specified.")

    # make dataparser to get user feats before create model
    data_config = copy.deepcopy(pipeline_config.data_config)
    features = cast(List[BaseFeature], model.features)
    data_config.num_workers = 1
    data_config.batch_size = acc_utils.get_max_export_batch_size()
    input_path = data_input_path or pipeline_config.train_input_path
    dataloader = create_dataloader(data_config, features, input_path, mode=Mode.PREDICT)
    batch = next(dataloader.get_iterator())  # pyre-ignore[16]
    data = batch.to(device).to_dict(sparse_dtype=torch.int64)

    # Build Sharded Model
    planner = create_planner(
        device=device,
        # pyre-ignore [16]
        batch_size=dataloader.dataset.sampled_batch_size,
        ckpt_plan_path=os.path.join(checkpoint_path, "plan")
        if checkpoint_path
        else None,
        global_constraints_cfg=train_config.global_embedding_constraints
        if train_config.HasField("global_embedding_constraints")
        else None,
        model=model,
    )
    sharders = get_default_sharders()
    plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)
    if is_rank_zero:
        logger.info(str(plan))

    dmp_model = DistributedModelParallel(
        module=model,
        sharders=sharders,
        device=device,
        plan=plan,
        init_parameters=False,
        init_data_parallel=False,
    )
    dmp_model.eval()

    # Trace Sharded Model
    unwrap_model = dmp_model.module
    tracer = Tracer(
        leaf_modules=_get_sharded_leaf_module_names(unwrap_model)
        + _get_dense_embedding_leaf_module_names(unwrap_model)
    )
    full_graph = tracer.trace(unwrap_model)  # , concrete_args=concrete_args)
    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_full.graph"), "w") as f:
            f.write(str(full_graph))

    def _seq_len_name(seq_name: str) -> str:
        return seq_name + "_sequence_length"

    def _seq_feat_name(seq_name: str) -> str:
        return seq_name + "_sequence"

    # Extract Sparse Model
    logger.info("exporting sparse model...")
    graph = copy.deepcopy(full_graph)
    for node in graph.nodes:
        if node.op == "output":
            graph.erase_node(node)
    outputs = {}
    output_attrs = {}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == fx_mark_keyed_tensor:
            name = node.args[0]
            node_kt = node.args[1]
            with graph.inserting_after(node_kt):
                outputs[name] = graph.call_method("values", args=(node_kt,))
                output_attrs[name + "__length_per_key"] = graph.call_method(
                    "length_per_key", args=(node_kt,)
                )
                output_attrs[name + "__keys"] = graph.call_method(
                    "keys", args=(node_kt,)
                )
        elif node.op == "call_function" and node.target == fx_mark_tensor:
            # query
            name = node.args[0]
            t = node.args[1]
            outputs[name] = t
        elif node.op == "call_function" and node.target == fx_mark_seq_tensor:
            # sequence
            seq_name = node.args[0]
            name = _seq_feat_name(seq_name)
            seq_node = node.args[1]
            assert node.kwargs["max_seq_len"] is not None, (
                f"[{node.kwargs['keys']}] should config sequence_length."
            )
            if node.kwargs["is_jagged_seq"]:
                assert data_config.batch_size == 1, (
                    "jagged sequence only support MAX_EXPORT_BATCH_SIZE=1 when export rtp."  # NOQA
                )
                with graph.inserting_after(seq_node):
                    seq_node = graph.call_function(torch.unsqueeze, args=(seq_node, 0))
            # rtp table_api always padding sequence to max_seq_len
            with graph.inserting_after(seq_node):
                seq_node = graph.call_function(
                    _rtp_pad_to_max_seq_len, args=(seq_node, node.kwargs["max_seq_len"])
                )
            outputs[name] = seq_node
        elif node.op == "call_function" and node.target == fx_mark_seq_len:
            # sequence length
            seq_name = node.args[0]
            name = _seq_len_name(seq_name)
            t = node.args[1]
            with graph.inserting_after(t):
                # RTP do not support rank=1 tensor
                unsqueeze_t = graph.call_function(torch.unsqueeze, args=(t, 1))
                outputs[name] = unsqueeze_t
    graph.output(tuple([outputs, output_attrs]))
    gm = torch.fx.GraphModule(unwrap_model, graph)
    gm.graph.eliminate_dead_code()
    gm = _prune_unused_param_and_buffer(gm)
    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_sparse.graph"), "w") as f:
            f.write(str(gm.graph))
    sparse_model = DistributedModelParallel(
        module=gm,
        sharders=sharders,
        device=device,
        plan=plan,
    )
    sparse_model.eval()
    checkpoint_util.restore_model(checkpoint_path, sparse_model)
    sparse_output, sparse_attrs = sparse_model(data, device=device)
    # Save Sparse Parameters
    logger.info("saving sparse parameters...")
    local_tensor, meta = _get_rtp_embedding_tensor(
        sparse_model, checkpoint_path, feature_to_embedding_info.values()
    )
    local_tensor_name = f"model-{rank:06d}-of-{world_size:06d}"
    local_tensor_path = os.path.join(save_dir, f"{local_tensor_name}.safetensors")
    logger.info(f"save safetensor to {local_tensor_path}")
    save_file(local_tensor, local_tensor_path)
    with open(os.path.join(save_dir, f"{local_tensor_name}.json"), "w") as f:
        json.dump(meta, f, indent=4)

    # Extract Dense Model
    logger.info("exporting dense model...")
    additional_fg = []

    graph = copy.deepcopy(full_graph)
    output_keys = []
    output_values = []
    mc_config = defaultdict()
    for node in graph.nodes:
        if node.op == "output":
            for k, v in sorted(node.args[0].items()):
                if k == TARGET_REPEAT_INTERLEAVE_KEY:
                    continue
                output_keys.append(k)
                output_values.append(v)
            graph.erase_node(node)
    input_node = next(node for node in graph.nodes if node.op == "placeholder")

    seq_len_nodes = {}
    for node in graph.nodes:
        if node.op == "call_function" and node.target == fx_mark_seq_len:
            # sequence_length
            seq_name = node.args[0]
            name = _seq_len_name(seq_name)
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                get_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                # rtp do not support RANK=1 tensor
                new_node = graph.call_function(torch.squeeze, args=(get_node, 1))
                seq_len_nodes[seq_name] = new_node
                # add sequence_length into fg
                additional_fg.append(
                    {
                        "feature_name": name,
                        "feature_type": "raw_feature",
                        "expression": f"user:{name}",
                    }
                )
                logger.info(f"You should add additional feature [{name}] into qinfo.")
                mc_config[name] = [name]
            node_t.replace_all_uses_with(new_node)

    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == fx_mark_keyed_tensor:
            name = node.args[0]
            node_kt = node.args[1]
            with graph.inserting_before(node_kt):
                getitem_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                new_node = graph.call_function(
                    KeyedTensor,
                    kwargs={
                        "keys": sparse_attrs[name + "__keys"],
                        "length_per_key": sparse_attrs[name + "__length_per_key"],
                        "values": getitem_node,
                    },
                )
                mc_config[name] = new_node.kwargs["keys"]
                node_kt.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target == fx_mark_tensor:
            # query
            name = node.args[0]
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                new_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                mc_config[name] = node.kwargs["keys"]
            node_t.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target == fx_mark_seq_tensor:
            # sequence
            seq_name = node.args[0]
            name = _seq_feat_name(seq_name)
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                new_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                new_node = graph.call_function(
                    _rtp_slice_with_seq_len,
                    args=(
                        new_node,
                        seq_len_nodes[seq_name],
                        node.kwargs["max_seq_len"],
                    ),
                )
                if node.kwargs["is_jagged_seq"]:
                    assert data_config.batch_size == 1, (
                        "jagged sequence only support MAX_EXPORT_BATCH_SIZE=1 when export rtp."  # NOQA
                    )
                    new_node = graph.call_function(torch.squeeze, args=(new_node, 0))
                mc_config[name] = node.kwargs["keys"]
            node_t.replace_all_uses_with(new_node)
        elif node.op == "call_function" and "fbgemm" in str(node.target):
            # rtp do not support fbgemm op now.
            if node.target in FBGEMM_RTP_TORCH_OP_MAPPING.keys():
                assert data_config.batch_size == 1, (
                    "{node.target} op only support MAX_EXPORT_BATCH_SIZE=1 when export rtp."  # NOQA
                )
                with graph.inserting_before(node):
                    new_node = graph.call_function(
                        FBGEMM_RTP_TORCH_OP_MAPPING[node.target],
                        args=node.args,
                        kwargs=node.kwargs,
                    )
                node.replace_all_uses_with(new_node)
            else:
                raise RuntimeError(f"{node.target} op is not supported by rtp")
    graph.output(tuple(output_values))
    gm = torch.fx.GraphModule(unwrap_model, graph)
    gm.graph.eliminate_dead_code()
    gm = _prune_unused_param_and_buffer(gm)
    init_parameters(gm, device)
    gm.to(device)
    checkpoint_util.restore_model(checkpoint_path, gm)

    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_dense.graph"), "w") as f:
            f.write(str(gm.graph))
        _ = gm(sparse_output)

        # Save Dense Model
        # when batch_size=1, we assume gr model.
        dynamic = data_config.batch_size > 1
        # remove device metadata assert to fix: torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function aten._assert_tensor_metadata.default*(FakeTensor(..., device-'cuda:0', size-(u1, 512)), None, None, torch. float32), **{'device': device(type-'cuda", index-0), 'layout': torch.strided}): got RuntimeError(Tensor device mismatch!')   # NOQA
        with torch._export.utils._disable_aten_to_metadata_assertions():
            fx_tool = ExportTorchFxTool(
                os.path.join(save_dir, "fx_user_model"), dynamic=dynamic
            )
            fx_tool.set_output_nodes_name(output_keys)
            fx_tool.export_fx_model(gm, sparse_output, mc_config)

        has_fg_asset = False
        if assets is not None:
            for asset in assets:
                if asset.endswith("fg.json"):
                    has_fg_asset = True
                shutil.copy(asset, save_dir)

        # Save FG
        if not has_fg_asset:
            logger.info("saving fg json...")
            fg_json = create_fg_json(features, asset_dir=save_dir)
            fg_json["features"].extend(additional_fg)
            _adjust_fg_json_for_rtp(fg_json, feature_to_embedding_info)
            with open(os.path.join(save_dir, "fg.json"), "w") as f:
                json.dump(fg_json, f, indent=4)


def _compute_seq_share_groups(
    features: List[BaseFeature],
    feature_groups: List[model_pb2.FeatureGroupConfig],
) -> Dict[str, str]:
    """Map ``{group_name}__sequence`` to a share_key.

    Feature groups whose first feature comes from the same parent
    SequenceFeature share per-sample lengths and therefore must share
    a torch.export.Dim on the lengths-derived axis.
    """
    from tzrec.protos.model_pb2 import FeatureGroupType

    feat_by_name = {f.name: f for f in features}
    share: Dict[str, str] = {}
    for fg in feature_groups:
        if fg.group_type not in (
            FeatureGroupType.SEQUENCE,
            FeatureGroupType.JAGGED_SEQUENCE,
        ):
            continue
        if not fg.feature_names:
            continue
        first = feat_by_name.get(fg.feature_names[0])
        if first is not None and getattr(first, "_is_grouped_seq", False):
            share_key = f"seq_{first.sequence_name}"
        else:
            share_key = f"fg_{fg.group_name}"
        share[f"{fg.group_name}__sequence"] = share_key
    return share


def split_model(
    data: Dict[str, torch.Tensor], model: BaseModule, save_dir: str
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    """Split an EasyRec model into sparse part and dense part."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0
    graph_dir = os.path.join(save_dir, "graph")
    if is_rank_zero:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

    model.eval()

    tracer = Tracer()
    full_graph = tracer.trace(model)  # , concrete_args=concrete_args)
    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_full.graph"), "w") as f:
            f.write(str(full_graph))

    def _seq_len_name(seq_name: str) -> str:
        return seq_name + "__sequence_length"

    def _seq_feat_name(seq_name: str) -> str:
        return seq_name + "__sequence"

    # Extract Sparse Model
    logger.info("exporting sparse model...")
    graph = copy.deepcopy(full_graph)
    for node in graph.nodes:
        if node.op == "output":
            graph.erase_node(node)
    outputs = {}
    output_attrs = {}
    jagged_seq_tensor_names = []
    seq_tensor_names = []
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == fx_mark_keyed_tensor:
            name = node.args[0]
            node_kt = node.args[1]
            with graph.inserting_after(node_kt):
                outputs[name] = graph.call_method("values", args=(node_kt,))
                output_attrs[name + "__length_per_key"] = graph.call_method(
                    "length_per_key", args=(node_kt,)
                )
                output_attrs[name + "__keys"] = graph.call_method(
                    "keys", args=(node_kt,)
                )
        elif node.op == "call_function" and node.target == fx_mark_tensor:
            # query
            name = node.args[0]
            t = node.args[1]
            outputs[name] = t
        elif node.op == "call_function" and node.target == fx_mark_seq_tensor:
            # sequence
            name = _seq_feat_name(node.args[0])
            t = node.args[1]
            outputs[name] = t
            if node.kwargs["is_jagged_seq"]:
                jagged_seq_tensor_names.append(name)
            else:
                seq_tensor_names.append(name)
        elif node.op == "call_function" and node.target == fx_mark_seq_len:
            # sequence length
            name = _seq_len_name(node.args[0])
            t = node.args[1]
            outputs[name] = t
    graph.output(tuple([outputs, output_attrs]))
    sparse_gm = torch.fx.GraphModule(model, graph)
    sparse_gm.graph.eliminate_dead_code()
    sparse_gm = _prune_unused_param_and_buffer(sparse_gm)
    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_sparse.graph"), "w") as f:
            f.write(str(sparse_gm.graph))
    _, sparse_attrs = sparse_gm(data, device=device)

    # Extract Dense Model
    logger.info("exporting dense model...")

    graph = copy.deepcopy(full_graph)
    input_node = next(node for node in graph.nodes if node.op == "placeholder")
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == fx_mark_keyed_tensor:
            name = node.args[0]
            node_kt = node.args[1]
            with graph.inserting_before(node_kt):
                getitem_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                new_node = graph.call_function(
                    KeyedTensor,
                    kwargs={
                        "keys": sparse_attrs[name + "__keys"],
                        "length_per_key": sparse_attrs[name + "__length_per_key"],
                        "values": getitem_node,
                    },
                )
                node_kt.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target in (
            fx_mark_tensor,
            fx_mark_seq_tensor,
        ):
            # sequence or query name
            name = (
                node.args[0]
                if node.target == fx_mark_tensor
                else _seq_feat_name(node.args[0])
            )
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                new_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
            node_t.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target == fx_mark_seq_len:
            # sequence_length
            name = _seq_len_name(node.args[0])
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                get_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )

            node_t.replace_all_uses_with(get_node)
    dense_gm = torch.fx.GraphModule(model, graph)
    dense_gm.graph.eliminate_dead_code()
    dense_gm = _prune_unused_param_and_buffer(dense_gm)

    seq_share_groups = _compute_seq_share_groups(
        features=cast(List[BaseFeature], model.features),
        feature_groups=model.feature_groups,
    )
    meta_info = {
        "seq_tensor_names": seq_tensor_names,
        "jagged_seq_tensor_names": jagged_seq_tensor_names,
        "seq_share_groups": seq_share_groups,
    }
    return sparse_gm, dense_gm, meta_info


def export_distributed_embedding(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
    use_local_cache_dir: bool = False,
    **kwargs: Any,
) -> None:
    """Export for online serving under distributed embedding mode."""
    if not _prepare_single_rank_distributed_embedding_export():
        return

    device, _ = init_process_group()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_rank_zero = rank == 0
    graph_dir = os.path.join(save_dir, "graph")
    if is_rank_zero:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

    if not checkpoint_path:
        raise ValueError("checkpoint path should be specified.")

    feature_to_embedding_bag_info, feature_to_embedding_info = (
        _get_sparse_feature_to_embedding_info(model)
    )

    # make dataparser to get user feats before create model
    data_config = copy.deepcopy(pipeline_config.data_config)
    features = cast(List[BaseFeature], model.features)
    data_config.num_workers = 1
    dataloader = create_dataloader(
        data_config, features, pipeline_config.train_input_path, mode=Mode.PREDICT
    )
    batch = next(iter(dataloader))
    data = batch.to(device).to_dict(sparse_dtype=torch.int64)

    train_config = pipeline_config.train_config

    model.set_is_inference(True)

    # Build Sharded Model
    planner = create_planner(
        device=device,
        # pyre-ignore [16]
        batch_size=dataloader.dataset.sampled_batch_size,
        ckpt_plan_path=os.path.join(checkpoint_path, "plan")
        if checkpoint_path
        else None,
        global_constraints_cfg=train_config.global_embedding_constraints
        if train_config.HasField("global_embedding_constraints")
        else None,
        model=model,
    )
    sharders = get_default_sharders()
    plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)
    dmp_model = DistributedModelParallel(
        module=model,
        sharders=sharders,
        device=device,
        plan=plan,
        init_parameters=True,
        init_data_parallel=False,
    )
    dmp_model.eval()

    # Materialize lazy modules before FX tracing. Some modules build submodules
    # from concrete tensor shapes on their first forward; during FX trace those
    # shapes become Proxy objects and cannot be used to construct Parameters.
    #
    # Keep the existing sparse/dense GraphModule restore steps below as the
    # final source of exported weights. This warm-up is only to make the module
    # structure traceable.
    logger.info("running pre-trace warm-up for distributed embedding export...")
    checkpoint_util.restore_model(checkpoint_path, dmp_model)
    dmp_model.to(device)
    with torch.no_grad():
        warmup_result = dmp_model(data, device=device)
    del warmup_result
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unwrap_model = dmp_model.module
    tracer = Tracer(leaf_modules=_get_sharded_leaf_module_names(unwrap_model))
    full_graph = tracer.trace(unwrap_model)  # , concrete_args=concrete_args)

    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_full.graph"), "w") as f:
            f.write(str(full_graph))

    # Extract Sparse Model
    logger.info("exporting sparse model...")
    graph = copy.deepcopy(full_graph)
    for node in graph.nodes:
        if node.op == "output":
            graph.erase_node(node)
    outputs = {}
    output_attrs = {}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == fx_mark_keyed_tensor:
            name = node.args[0]
            if node.kwargs.get("is_dense", False):
                continue
            node_kt = node.args[1]
            with graph.inserting_after(node_kt):
                kt_values = node_kt.kwargs.get("values")
                if (
                    _is_input_tile_user_keyed_tensor(name)
                    and getattr(kt_values, "op", None) == "call_method"
                    and kt_values.target == "tile"
                ):
                    # Serving supplies raw user-side embedding values with batch=1.
                    # Keep sparse output raw as well; the dense model owns tiling.
                    outputs[name] = kt_values.args[0]
                    output_attrs[name + "__length_per_key"] = node_kt.kwargs[
                        "length_per_key"
                    ]
                    output_attrs[name + "__keys"] = node_kt.kwargs["keys"]
                else:
                    outputs[name] = graph.call_method("values", args=(node_kt,))
                    output_attrs[name + "__length_per_key"] = graph.call_method(
                        "length_per_key", args=(node_kt,)
                    )
                    output_attrs[name + "__keys"] = graph.call_method(
                        "keys", args=(node_kt,)
                    )
        elif node.op == "call_function" and node.target == fx_mark_seq_ec_jt:
            name = node.args[0]
            node_jt = node.args[1]
            with graph.inserting_after(node_jt):
                outputs[name] = graph.call_method("values", args=(node_jt,))
                outputs[name + "__lengths"] = graph.call_method(
                    "lengths", args=(node_jt,)
                )
                output_attrs[name + "__lengths"] = graph.call_method(
                    "lengths", args=(node_jt,)
                )

    graph.output(tuple([outputs, output_attrs]))
    sparse_gm = torch.fx.GraphModule(unwrap_model, graph)
    sparse_gm.graph.eliminate_dead_code()
    sparse_gm = _prune_unused_param_and_buffer(sparse_gm)

    if is_rank_zero:
        with open(os.path.join(graph_dir, "gm_sparse.graph"), "w") as f:
            f.write(str(sparse_gm.graph))

    # `unwrap_model` has already been wrapped by DistributedModelParallel and
    # restored above. `sparse_gm` shares those sharded embedding modules; wrapping
    # the extracted sparse graph a second time can leave TorchRec's per-table
    # feature split state inconsistent for INPUT_TILE user towers.
    sparse_model = sparse_gm
    sparse_model.eval()
    sparse_output, sparse_attrs = sparse_model(data, device=device)
    # Save Sparse Parameters
    logger.info("saving sparse parameters...")

    local_tensor, dynamic_local_tensor, emb_meta, feat_meta = (
        _get_sparse_embedding_tensor(
            sparse_model,
            checkpoint_path,
            feature_to_embedding_info.values(),
            feature_to_embedding_bag_info.values(),
        )
    )
    local_tensor_name = f"sparse_embeddings-{rank:02d}-of-{world_size:02d}"
    save_dir_sparse = f"{save_dir}/sparse"
    if not os.path.exists(save_dir_sparse):
        os.makedirs(save_dir_sparse)
    local_tensor_path = os.path.join(save_dir_sparse, f"{local_tensor_name}.npz")
    logger.info(f"save sparse tensors to {local_tensor_path}")
    # np.savez(local_tensor_path, **local_tensor)
    # OSS mounted file system may have problem in file seek, so first
    # save to a temp file then move to target path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f:
        np.savez(f, **local_tensor)
        temp_path = f.name
    shutil.move(temp_path, local_tensor_path)

    if dynamic_local_tensor:
        dynamic_tensor_name = f"sparse_dynamic_embedding-{rank:02d}-of-{world_size:02d}"
        dynamic_tensor_path = os.path.join(
            save_dir_sparse, f"{dynamic_tensor_name}.npz"
        )
        logger.info(f"save dynamic sparse tensors to {dynamic_tensor_path}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as f:
            np.savez(f, **dynamic_local_tensor)
            temp_path = f.name
        shutil.move(temp_path, dynamic_tensor_path)

    with open(os.path.join(save_dir_sparse, f"{local_tensor_name}.json"), "w") as f:
        json.dump(emb_meta, f, indent=4)
    if is_rank_zero:
        with open(os.path.join(save_dir_sparse, "sparse_features.json"), "w") as f:
            json.dump(feat_meta, f, indent=4)

    # Extract Dense Model
    logger.info("exporting dense model...")
    graph = copy.deepcopy(full_graph)
    output_keys = []
    output_values = []
    dense_graph_config = defaultdict()
    for node in graph.nodes:
        if node.op == "output":
            for k, v in sorted(node.args[0].items()):
                if k == TARGET_REPEAT_INTERLEAVE_KEY:
                    continue
                output_keys.append(k)
                output_values.append(v)
            graph.erase_node(node)
    input_node = next(node for node in graph.nodes if node.op == "placeholder")

    dense_graph_config["sequence__ec"] = []
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == fx_mark_keyed_tensor:
            name = node.args[0]
            if node.kwargs.get("is_dense", False):
                continue
            node_kt = node.args[1]
            with graph.inserting_before(node_kt):
                getitem_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                values_node = getitem_node
                if _is_input_tile_user_keyed_tensor(name):
                    batch_size_node = graph.call_function(
                        operator.getitem, args=(input_node, "batch_size")
                    )
                    tile_size_node = graph.call_function(
                        _tile_size, args=(batch_size_node,)
                    )
                    values_node = graph.call_method(
                        "tile", args=(getitem_node, tile_size_node, 1)
                    )
                new_node = graph.call_function(
                    KeyedTensor,
                    kwargs={
                        "keys": sparse_attrs[name + "__keys"],
                        "length_per_key": sparse_attrs[name + "__length_per_key"],
                        "values": values_node,
                    },
                )
                dense_graph_config[name] = [
                    k + "__ebc" for k in new_node.kwargs["keys"]
                ]
                node_kt.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target == fx_mark_seq_ec_jt:
            name = node.args[0]
            node_jt = node.args[1]

            with graph.inserting_before(node_jt):
                getitem_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                getitem_lengths = graph.call_function(
                    operator.getitem, args=(input_node, name + "__lengths")
                )
                new_node = graph.call_function(
                    JaggedTensor,
                    kwargs={
                        "values": getitem_node,
                        "lengths": getitem_lengths,
                    },
                )
            node_jt.replace_all_uses_with(new_node)
            emb_name = name + "__ec"
            dense_graph_config["sequence__ec"].append(emb_name)
            dense_graph_config["sequence__ec"].append(name + "__lengths")

    graph.output(dict(zip(output_keys, output_values)))
    gm = torch.fx.GraphModule(unwrap_model, graph)
    gm.graph.eliminate_dead_code()
    gm = _prune_unused_param_and_buffer(gm)

    init_parameters(gm, device)
    gm.to(device)
    checkpoint_util.restore_model(checkpoint_path, gm)

    if is_rank_zero:
        with open(os.path.join(save_dir, "dense_meta.json"), "w") as f:
            json.dump(dense_graph_config, f, indent=4)

        with open(os.path.join(graph_dir, "gm_dense.graph"), "w") as f:
            f.write(str(gm.graph))
        data.update(sparse_output)
        _ = gm(data, device)

        dense_model_traced = symbolic_trace(gm)

        with open(os.path.join(save_dir, "gm_dense.code"), "w") as f:
            f.write(dense_model_traced.code)

        dense_model_scripted = torch.jit.script(dense_model_traced)
        dense_model_scripted.save(os.path.join(save_dir, "scripted_model.pt"))

        logger.info("saving pipeline.config...")
        feature_configs = create_feature_configs(features, asset_dir=save_dir)
        pipeline_config = copy.copy(pipeline_config)
        pipeline_config.ClearField("feature_configs")
        pipeline_config.feature_configs.extend(feature_configs)
        config_util.save_message(
            pipeline_config, os.path.join(save_dir, "pipeline.config")
        )

        with open(os.path.join(save_dir, "model_acc.json"), "w") as f:
            json.dump(acc_utils.export_acc_config(), f, indent=4)

        has_fg_asset = False
        if assets is not None:
            for asset in assets:
                if asset.endswith("fg.json"):
                    has_fg_asset = True
                shutil.copy(asset, save_dir)

        # Save FG
        if not has_fg_asset:
            logger.info("saving fg json...")
            fg_json = create_fg_json(features, asset_dir=save_dir)
            with open(os.path.join(save_dir, "fg.json"), "w") as f:
                json.dump(fg_json, f, indent=4)

    if is_rank_zero:
        # merge sharded sparse meta files
        emb_json_file_names = glob.glob(
            os.path.join(save_dir_sparse, "sparse_embeddings*.json")
        )
        emb_json_files = []
        for emb_j in emb_json_file_names:
            with open(f"{emb_j}", "r", encoding="utf-8") as f:
                data_j = json.load(f)
                emb_json_files.append(data_j)

        merged_emb_json = _merge_sharded_embedding_json(emb_json_files)
        with open(os.path.join(save_dir_sparse, "sparse_embedding.json"), "w") as f:
            json.dump(merged_emb_json, f, indent=4)


def _merge_sharded_embedding_json(
    emb_json_files: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge sharded embedding json files into one."""
    merged_json = {}
    for emb_json in emb_json_files:
        for emb_name, info in emb_json.items():
            if emb_name not in merged_json:
                merged_json[emb_name] = info
            else:
                merged_json[emb_name]["memory"] += info["memory"]
                merged_json[emb_name]["shape"][0] += info["shape"][0]
                if info.get("is_dynamic"):
                    # Dynamic npz entry names are uniform across ranks
                    # (e.g. always "user_id_emb.keys"). Serving iterates all
                    # sparse_dynamic_embedding-*.npz files with the same entry
                    # names, so keep the single key/value/score entry names.
                    for field in ("key_name", "value_name", "score_name"):
                        if merged_json[emb_name].get(field) != info.get(field):
                            raise ValueError(
                                f"dynamic embedding {emb_name} has inconsistent "
                                f"{field}: {merged_json[emb_name].get(field)} vs "
                                f"{info.get(field)}"
                            )
                    if merged_json[emb_name].get("score_dtype") != info.get(
                        "score_dtype"
                    ):
                        raise ValueError(
                            f"dynamic embedding {emb_name} has inconsistent "
                            f"score_dtype: {merged_json[emb_name].get('score_dtype')} "
                            f"vs {info.get('score_dtype')}"
                        )

    return merged_json


def _get_sparse_feature_to_embedding_info(
    model: nn.Module,
) -> Tuple[Dict[str, BaseEmbeddingConfig], Dict[str, BaseEmbeddingConfig]]:
    feature_to_embedding_bag_info = dict()
    feature_to_embedding_info = dict()
    q = Queue()
    q.put(("", model))
    while not q.empty():
        child_path, m = q.get()
        if isinstance(m, EmbeddingBagCollectionInterface):
            embedding_configs = m.embedding_bag_configs()
            for t in embedding_configs:
                for fname in t.feature_names:
                    feature_to_embedding_bag_info[fname] = t

        elif isinstance(m, EmbeddingCollectionInterface):
            embedding_configs = m.embedding_configs()
            for t in embedding_configs:
                for fname in t.feature_names:
                    feature_to_embedding_info[fname] = t
        else:
            for name, child in m.named_children():
                if child_path == "":
                    q.put((name, child))
                else:
                    q.put((f"{child_path}.{name}", child))

    return feature_to_embedding_bag_info, feature_to_embedding_info


_SPARSE_EC_ROLE = "ec"
_SPARSE_EBC_ROLE = "ebc"


def _build_sparse_export_name_map(
    embedding_infos: List[BaseEmbeddingConfig],
    embedding_bag_info: List[BaseEmbeddingConfig],
) -> Dict[Tuple[str, str], str]:
    """Build physical export names for sparse embedding tables.

    TorchRec keeps EmbeddingCollection and EmbeddingBagCollection as separate
    physical modules. A config may reuse the same embedding_name in both places,
    but those are still distinct checkpoint tensors. Keep the historical name
    unless it appears in multiple sparse collection kinds; then suffix the
    exported table name so serving can address the correct physical table.
    """
    roles_by_name = defaultdict(set)
    for emb_info in embedding_infos:
        roles_by_name[emb_info.name].add(_SPARSE_EC_ROLE)
    for emb_info in embedding_bag_info:
        roles_by_name[emb_info.name].add(_SPARSE_EBC_ROLE)

    used_names = set(roles_by_name.keys())
    export_name_by_role: Dict[Tuple[str, str], str] = {}
    for emb_name, roles in roles_by_name.items():
        if len(roles) == 1:
            role = next(iter(roles))
            export_name_by_role[(role, emb_name)] = emb_name
            continue

        for role in sorted(roles):
            base_candidate = f"{emb_name}__{role}"
            candidate = base_candidate
            suffix = 1
            while candidate in used_names:
                candidate = f"{base_candidate}_{suffix}"
                suffix += 1
            used_names.add(candidate)
            export_name_by_role[(role, emb_name)] = candidate
    return export_name_by_role


def _sparse_export_role_from_state_key(state_key: str) -> Optional[str]:
    if ".embedding_bags." in state_key:
        return _SPARSE_EBC_ROLE
    if ".embeddings." in state_key:
        return _SPARSE_EC_ROLE
    return None


def _sparse_export_role_from_dynamic_path(path: str) -> Optional[str]:
    parent = os.path.dirname(path)
    if (
        ".ec_dict" in parent
        or ".ec_list" in parent
        or ".mc_ec" in parent
        or ".ec." in parent
    ):
        return _SPARSE_EC_ROLE
    if ".ebc" in parent or ".mc_ebc" in parent:
        return _SPARSE_EBC_ROLE
    return None


def _resolve_sparse_export_name(
    export_name_by_role: Dict[Tuple[str, str], str],
    emb_name: str,
    role: Optional[str],
) -> str:
    if role is not None and (role, emb_name) in export_name_by_role:
        return export_name_by_role[(role, emb_name)]

    candidates = [
        export_name
        for (_role, name), export_name in export_name_by_role.items()
        if name == emb_name
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"sparse embedding {emb_name} is not in export metadata")
    raise ValueError(
        f"sparse embedding {emb_name} appears in multiple sparse collection "
        f"kinds; cannot resolve export name without role, got role={role}"
    )


def _get_sparse_embedding_tensor(
    model: nn.Module,
    checkpoint_path: str,
    embedding_infos: List[BaseEmbeddingConfig],
    embedding_bag_info: List[BaseEmbeddingConfig],
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Get Embedding Tensors for sparse part.

    Returns:
        out: regular sparse embedding tensors keyed by emb_name.
        dynamic_out: dynamicemb keys/values/scores keyed by composite names. Empty if
            no dynamic embedding tables exist.
        emb_meta: per-table meta (shape/dtype/memory) for ALL sparse tables.
        feat_meta: feature -> embedding_name / pooling mapping.
    """
    emb_name_to_emb_dim = dict()
    feat_name_to_pooling = dict()

    feat_name_impl_to_emb_name = dict()
    emb_name_to_feat_name_impl = dict()

    export_name_by_role = _build_sparse_export_name_map(
        embedding_infos, embedding_bag_info
    )

    for emb_info in embedding_infos:
        export_emb_name = _resolve_sparse_export_name(
            export_name_by_role, emb_info.name, _SPARSE_EC_ROLE
        )
        emb_name_to_emb_dim[export_emb_name] = emb_info.embedding_dim
        emb_name_to_feat_name_impl[export_emb_name] = []
        for feat_name in emb_info.feature_names:
            feat_name_impl = feat_name + "__ec"
            emb_name_to_feat_name_impl[export_emb_name].append(feat_name_impl)
            feat_name_impl_to_emb_name[feat_name_impl] = export_emb_name
            if hasattr(emb_info, "pooling"):
                feat_name_to_pooling[feat_name_impl] = str(emb_info.pooling).split(".")[
                    -1
                ]
            else:
                feat_name_to_pooling[feat_name_impl] = "NONE"
    for emb_info in embedding_bag_info:
        export_emb_name = _resolve_sparse_export_name(
            export_name_by_role, emb_info.name, _SPARSE_EBC_ROLE
        )
        emb_name_to_emb_dim[export_emb_name] = emb_info.embedding_dim
        emb_name_to_feat_name_impl[export_emb_name] = []
        for feat_name in emb_info.feature_names:
            feat_name_impl = feat_name + "__ebc"
            emb_name_to_feat_name_impl[export_emb_name].append(feat_name_impl)
            feat_name_impl_to_emb_name[feat_name_impl] = export_emb_name
            if hasattr(emb_info, "pooling"):
                feat_name_to_pooling[feat_name_impl] = str(emb_info.pooling).split(".")[
                    -1
                ]
            else:
                feat_name_to_pooling[feat_name_impl] = "NONE"

    def _remove_prefix(src: str, prefix: str = "torch.") -> str:
        if src.startswith(prefix):
            return src[len(prefix) :]
        return src

    out = {}
    # dynamicemb keys/values are saved into a separate npz, kept in this dict.
    dynamic_out = {}
    # shard_offsets = {}
    # per-table representative tensor used only for meta (shape/dtype/memory).
    # dynamicemb tables are stored in `dynamic_out` under composite keys, so we
    # track a separate mapping keyed by emb_name here.
    emb_name_to_meta_tensor = {}
    # set of emb_names that are dynamic embedding tables (loaded from
    # checkpoint's `dynamicemb/` directory rather than model state_dict).
    dynamic_emb_names = set()
    # per-emb_name npz entry names for dynamic tables on this rank.
    dynamic_key_names = defaultdict(list)
    dynamic_value_names = defaultdict(list)
    dynamic_score_names = defaultdict(list)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    for name, values in model.state_dict().items():
        emb_name = name.split(".")[-2]
        export_emb_name = _resolve_sparse_export_name(
            export_name_by_role,
            emb_name,
            _sparse_export_role_from_state_key(name),
        )
        # emb_impl_type = name.split(".")[2]  # 'emb_impls' or 'seq_emb_impls'

        # feat_name_impl_list = emb_name_to_feat_name_impl.get(emb_name, [])

        emb_dim = emb_name_to_emb_dim[export_emb_name]
        if isinstance(values, DTensor):
            raise ValueError("DTensors are not considered yet.")
        elif isinstance(values, ShardedTensor):
            _len_local_shards = len(values.local_shards())
            assert _len_local_shards in [0, 1], "other cases are not considered."

            if _len_local_shards == 1:
                for _idx, shards_meta in enumerate(values.metadata().shards_metadata):
                    placement = shards_meta.placement
                    assert placement is not None
                    if placement.rank() == rank:
                        # name = name + f"/part_{idx}_{num_shards}"
                        local_tensor = values.local_tensor().cpu().numpy()
                        if list(local_tensor.shape)[-1] == emb_dim:
                            # dynamicemb may have a dummy tensor in state_dict, skip it.
                            out[export_emb_name] = local_tensor
                            emb_name_to_meta_tensor[export_emb_name] = local_tensor
                            # shard_offsets[feat_name_impl] = shards_meta.shard_offsets
        elif list(values.shape)[-1] == emb_dim:
            # dynamicemb may have a dummy tensor in state_dict, skip it.
            local_tensor = values.detach().cpu().numpy()
            out[export_emb_name] = local_tensor
            emb_name_to_meta_tensor[export_emb_name] = local_tensor
            # shard_offsets[feat_name_impl] = shards_meta.shard_offsets

    dynamicemb_path = os.path.join(checkpoint_path, "dynamicemb")
    if os.path.exists(dynamicemb_path):
        key_files = sorted(
            glob.glob(os.path.join(dynamicemb_path, "*/*_emb_keys.rank_*.world_size_*"))
        )
        key_files = _dedup_key_files_by_realpath(key_files)
        key_pattern = re.compile(
            r"^(?P<emb_name>.+)_emb_keys\.rank_(?P<idx>\d+)\.world_size_(?P<num_shards>\d+)$"
        )
        key_files_by_emb = defaultdict(list)
        for key_file in key_files:
            path_parts = key_file.split(os.path.sep)
            match = key_pattern.match(path_parts[-1])
            if not match:
                continue
            emb_name = match.group("emb_name")
            export_emb_name = _resolve_sparse_export_name(
                export_name_by_role,
                emb_name,
                _sparse_export_role_from_dynamic_path(key_file),
            )
            ckpt_rank = int(match.group("idx"))
            ckpt_world_size = int(match.group("num_shards"))
            key_files_by_emb[export_emb_name].append(
                (emb_name, ckpt_rank, ckpt_world_size, key_file)
            )

        for emb_name, emb_key_files in key_files_by_emb.items():
            emb_dim = emb_name_to_emb_dim[emb_name]
            key_name = f"{emb_name}.keys"
            value_name = f"{emb_name}.values"
            score_name = f"{emb_name}.scores"
            keys_list = []
            values_list = []
            scores_list = []
            ckpt_world_sizes = {x[2] for x in emb_key_files}
            if len(ckpt_world_sizes) > 1:
                raise ValueError(
                    f"dynamic embedding {emb_name} has inconsistent checkpoint "
                    f"world_size values: {sorted(ckpt_world_sizes)}"
                )
            for ckpt_emb_name, ckpt_rank, ckpt_world_size, key_file in sorted(
                emb_key_files
            ):
                if ckpt_rank % world_size != rank:
                    continue
                with open(key_file, "rb") as f:
                    keys = torch.tensor(
                        np.fromfile(f, dtype=np.int64), dtype=torch.int64
                    )
                value_file = os.path.join(
                    os.path.dirname(key_file),
                    f"{ckpt_emb_name}_emb_values.rank_{ckpt_rank}.world_size_{ckpt_world_size}",
                )
                with open(value_file, "rb") as f:
                    values = torch.tensor(
                        np.fromfile(f, dtype=np.float32), dtype=torch.float32
                    )
                score_file = os.path.join(
                    os.path.dirname(key_file),
                    f"{ckpt_emb_name}_emb_scores.rank_{ckpt_rank}.world_size_{ckpt_world_size}",
                )
                if not os.path.exists(score_file):
                    raise FileNotFoundError(
                        f"dynamic embedding {emb_name} score file not found: "
                        f"{score_file}"
                    )
                with open(score_file, "rb") as f:
                    scores = torch.tensor(
                        np.fromfile(f, dtype=np.int64), dtype=torch.int64
                    )
                if keys.numel() != scores.numel():
                    raise ValueError(
                        f"dynamic embedding {emb_name} key/score row mismatch: "
                        f"keys={keys.numel()}, scores={scores.numel()}, "
                        f"key_file={key_file}, score_file={score_file}"
                    )
                if values.numel() != keys.numel() * emb_dim:
                    raise ValueError(
                        f"dynamic embedding {emb_name} value row mismatch: "
                        f"keys={keys.numel()}, value_elements={values.numel()}, "
                        f"embedding_dim={emb_dim}"
                    )
                keys_list.append(keys)
                values_list.append(values.view([-1, emb_dim]))
                scores_list.append(scores)

            if not keys_list:
                continue
            keys = torch.cat(keys_list)
            values_2d = torch.cat(values_list, dim=0)
            scores = torch.cat(scores_list)
            dynamic_out[key_name] = keys
            dynamic_out[value_name] = values_2d
            dynamic_out[score_name] = scores
            emb_name_to_meta_tensor[emb_name] = values_2d
            dynamic_emb_names.add(emb_name)
            dynamic_key_names[emb_name].append(key_name)
            dynamic_value_names[emb_name].append(value_name)
            dynamic_score_names[emb_name].append(score_name)

    # TODO(hongsheng.jhs): support mczch

    emb_meta = {}
    for emb_name, feat_name_impl_list in emb_name_to_feat_name_impl.items():
        if emb_name not in emb_name_to_meta_tensor:
            # table not present on this rank (e.g. sharded to other ranks)
            continue
        values = emb_name_to_meta_tensor[emb_name]
        dimension = list(values.shape)[-1]
        dtype = _remove_prefix(str(values.dtype))
        memory: int = int(values.nbytes)
        shape = list(values.shape)
        is_dynamic = emb_name in dynamic_emb_names
        t_meta = {
            "feat_name_impl": feat_name_impl_list,
            "dense": False,
            "is_dynamic": is_dynamic,
            "dimension": dimension,
            "dtype": dtype,
            "memory": memory,
            "shape": shape,
            # "shard_offsets": shard_offsets[name],
            # "pooling": feat_name_to_pooling[name],
        }
        if is_dynamic:
            # Entry names inside sparse_dynamic_embedding-*.npz. The serving
            # processor expects one key/value/score entry triplet and applies it
            # to all dynamic npz shards.
            t_meta["key_dtype"] = "int64"
            t_meta["score_dtype"] = "int64"
            key_names = list(dict.fromkeys(dynamic_key_names[emb_name]))
            value_names = list(dict.fromkeys(dynamic_value_names[emb_name]))
            score_names = list(dict.fromkeys(dynamic_score_names[emb_name]))
            if len(key_names) != 1 or len(value_names) != 1 or len(score_names) != 1:
                raise ValueError(
                    f"dynamic embedding {emb_name} expects one key/value/score "
                    f"npz entry triplet, got {key_names} / {value_names} / "
                    f"{score_names}"
                )
            t_meta["key_name"] = key_names[0]
            t_meta["value_name"] = value_names[0]
            t_meta["score_name"] = score_names[0]
        emb_meta[emb_name] = t_meta

    feat_meta = {}
    for feat_name_impl, emb_name in feat_name_impl_to_emb_name.items():
        t_meta = {
            "embedding_name": emb_name,
            "pooling": feat_name_to_pooling[feat_name_impl],
        }
        feat_meta[feat_name_impl] = t_meta
    return out, dynamic_out, emb_meta, feat_meta
