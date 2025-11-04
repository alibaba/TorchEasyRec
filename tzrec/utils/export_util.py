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
import json
import operator
import os
import shutil
from collections import OrderedDict, defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from safetensors.torch import save_file
from torch import distributed as dist
from torch import nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torchrec import KeyedTensor
from torchrec.distributed.train_pipeline.utils import Tracer, _get_leaf_module_names
from torchrec.inference.modules import quantize_embeddings
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse import jagged_tensor

from tzrec.acc import utils as acc_utils
from tzrec.acc.aot_utils import export_model_aot
from tzrec.acc.trt_utils import export_model_trt
from tzrec.constant import Mode
from tzrec.datasets.dataset import (
    create_dataloader,
)
from tzrec.features.feature import (
    BaseFeature,
    create_feature_configs,
    create_fg_json,
)
from tzrec.modules.utils import BaseModule
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import checkpoint_util, config_util
from tzrec.utils.dist_util import DistributedModelParallel, init_process_group
from tzrec.utils.fx_util import (
    fx_mark_keyed_tensor,
    fx_mark_seq_len,
    fx_mark_tensor,
    symbolic_trace,
)
from tzrec.utils.logging_util import logger
from tzrec.utils.plan_util import create_planner, get_default_sharders
from tzrec.utils.state_dict_util import fix_mch_state, init_parameters
from tzrec.utils.filesystem_util import url_to_fs


def export_model(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
) -> None:
    """Export a EasyRec model, may be a part of model in PipelineConfig."""
    use_rtp = os.environ.get("USE_RTP", "0") == "1"
    impl = export_rtp_model if use_rtp else export_model_normal
    fs, local_path = url_to_fs(save_dir)
    if fs is not None:
        # scripted model use io in cpp, so that we can not path to fsspec
        local_path = os.environ.get('LOCAL_CACHE_DIR', local_path)
    impl(
        pipeline_config=pipeline_config,
        model=model,
        checkpoint_path=checkpoint_path,
        save_dir=local_path,
        assets=assets,
    )
    if fs is not None and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"uploading {local_path} to {save_dir}.")
        fs.upload(local_path, save_dir, recursive=True)
        shutil.rmtree(local_path)



def export_model_normal(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
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
    data_config = pipeline_config.data_config
    features = cast(List[BaseFeature], model._features)
    if acc_utils.is_cuda_export():
        # export batch_size too large may OOM in compile phase
        max_batch_size = acc_utils.get_max_export_batch_size()
        data_config.batch_size = min(data_config.batch_size, max_batch_size)
        logger.info("using new batch_size: %s in export", data_config.batch_size)
    data_config.num_workers = 1
    dataloader = create_dataloader(
        data_config, features, pipeline_config.train_input_path, mode=Mode.PREDICT
    )

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
        model.set_is_inference(True)

        init_parameters(model, torch.device("cpu"))
        checkpoint_util.restore_model(
            checkpoint_path, model, ckpt_param_map_path=ckpt_param_map_path
        )
        # for mc modules, fix output_segments_tensor is a meta tensor.
        fix_mch_state(model)

        batch = next(iter(dataloader))

        if acc_utils.is_cuda_export():
            model = model.cuda()

        if acc_utils.is_quant() or acc_utils.is_ec_quant():
            logger.info("quantize embeddings...")
            additional_qconfig_spec_keys = []
            additional_mapping = {}
            if acc_utils.is_ec_quant():
                additional_qconfig_spec_keys.append(EmbeddingCollection)
                additional_mapping[EmbeddingCollection] = QuantEmbeddingCollection
            quantize_embeddings(
                model,
                dtype=acc_utils.quant_dtype(),
                inplace=True,
                additional_qconfig_spec_keys=additional_qconfig_spec_keys,
                additional_mapping=additional_mapping,
            )
            logger.info("finish quantize embeddings...")

        model.eval()

        data = batch.to_dict(sparse_dtype=torch.int64)
        if acc_utils.is_trt():
            data = OrderedDict(sorted(data.items()))
            result = model(data, "cuda:0")
            result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
            logger.info(f"Model Outputs: {result_info}")
            export_model_trt(model, data, save_dir)
        elif acc_utils.is_aot():
            data = OrderedDict(sorted(data.items()))
            result = model(data)
            result_info = {k: (v.size(), v.dtype) for k, v in result.items()}
            logger.info(f"Model Outputs: {result_info}")
            export_model_aot(model, data, save_dir)
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
            json.dump(acc_utils.export_acc_config(), f, indent=4)

        if assets is not None:
            for asset in assets:
                shutil.copy(asset, save_dir)


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
            # This is a top-level parameter or buffer
            param_path = node.target
            if param_path not in name_to_obj:
                path_parts = param_path.split(".")
                current_obj = gm
                for part in path_parts:
                    current_obj = getattr(current_obj, part)

                # We need to register it on the new_root correctly
                # For simplicity here, we assume it's a direct attribute.
                if isinstance(current_obj, nn.Parameter):
                    new_root.register_parameter(param_path, current_obj)
                elif isinstance(current_obj, torch.Tensor):  # It's a buffer
                    new_root.register_buffer(param_path, current_obj)
                name_to_obj[param_path] = current_obj

    new_gm = torch.fx.GraphModule(new_root, gm.graph)
    return new_gm


def _get_rtp_embedding_tensor(
    model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Get Embedding Tensors for RTP."""

    def _remove_prefix(src: str, prefix: str = "torch.") -> str:
        if src.startswith(prefix):
            return src[len(prefix) :]
        return src

    out = {}
    rank = int(os.environ.get("RANK", 0))
    for key, values in model.state_dict().items():
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
                        out[key + f"/part_{idx}_{num_shards}"] = values.local_tensor()
        else:
            out[key] = values

    meta = {}
    for key, values in out.items():
        dimension = list(values.shape)[-1]
        dtype = _remove_prefix(str(values.dtype))
        memory: int = int(values.nbytes)
        shape = list(values.shape)
        # TODO(hongsheng.jhs): support mczch and dynamicemb
        is_hashmap = False

        meta[key] = {
            "name": key,
            "dense": False,
            "dimension": dimension,
            "dtype": dtype,
            "memory": memory,
            "shape": shape,
            "is_hashmap": is_hashmap,
        }
        out[key] = values
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
            try:
                feature_name = feature["feature_name"]
            except Exception:
                breakpoint()
            embedding_info = feature_to_embedding_info.get(feature_name, None)
            _adjust_one_feature_for_rtp(feature, embedding_info)
        else:
            sequence_name = feature["sequence_name"]
            if "sequence_table" not in feature:
                feature["sequence_table"] = "item"
            for sub_feature in feature["features"]:
                feature_name = sub_feature["feature_name"]
                embedding_info = feature_to_embedding_info.get(
                    f"{sequence_name}__{feature_name}", None
                )
                _adjust_one_feature_for_rtp(sub_feature, embedding_info)


def export_rtp_model(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
) -> None:
    """Export a EasyRec model on RTP."""
    try:
        from torch_fx_tool import ExportTorchFxTool
    except Exception as e:
        raise RuntimeError(
            "torch_fx_tool not exist. please install https://tzrec.oss-accelerate.aliyuncs.com/third_party/rtp/torch_fx_tool-0.0.1%2B20251031.376cedb-py3-none-any.whl"
        ) from e
    assert os.environ["USE_FARM_HASH_TO_BUCKETIZE"] == "true", (
        "you should set USE_FARM_HASH_TO_BUCKETIZE=true for "
        "train/eval/export when use rtp for online inference."
    )

    device, _ = init_process_group()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_rank_zero = rank == 0
    train_config = pipeline_config.train_config

    # RTP do not support fbgemm now. patch kt.regroup to slow path
    def _all_keys_used_once(
        keyed_tensors: List["KeyedTensor"], groups: List[List["str"]]
    ) -> bool:
        return False

    jagged_tensor._all_keys_used_once = _all_keys_used_once

    feature_to_embedding_info = _get_rtp_feature_to_embedding_info(model)

    graph_dir = os.path.join(save_dir, "graph")
    if is_rank_zero:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
    dist.barrier()

    if not checkpoint_path:
        raise ValueError("checkpoint path should be specified.")

    # make dataparser to get user feats before create model
    data_config = pipeline_config.data_config
    features = cast(List[BaseFeature], model._features)
    data_config.num_workers = 1
    dataloader = create_dataloader(
        data_config, features, pipeline_config.train_input_path, mode=Mode.PREDICT
    )
    batch = next(iter(dataloader))
    data = batch.to(device).to_dict(sparse_dtype=torch.int64)

    model.set_is_inference(True)
    model.eval()

    # Build Sharded Model
    planner = create_planner(
        device=device,
        # pyre-ignore [16]
        batch_size=dataloader.dataset.sampled_batch_size,
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

    # Trace Sharded Model
    unwrap_model = dmp_model.module
    tracer = Tracer(leaf_modules=_get_leaf_module_names(unwrap_model))
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
            # sequence or query name
            name = node.args[0]
            t = node.args[1]
            outputs[name] = t
        elif node.op == "call_function" and node.target == fx_mark_seq_len:
            # sequence length
            name = node.args[0]
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
    checkpoint_util.restore_model(checkpoint_path, sparse_model)
    sparse_output, sparse_attrs = sparse_model(data, device=device)
    # Save Sparse Parameters
    logger.info("saving sparse parameters...")
    local_tensor, meta = _get_rtp_embedding_tensor(sparse_model)
    save_file(
        local_tensor,
        os.path.join(save_dir, f"model-{rank:06d}-of-{world_size:06d}.safetensors"),
    )
    with open(
        os.path.join(save_dir, f"model-{rank:06d}-of-{world_size:06d}.json"), "w"
    ) as f:
        json.dump(meta, f, indent=4)

    # Extract Dense Model
    logger.info("exporting dense model...")
    additional_fg = []
    seqname_mapping = {}
    for feature in features:
        if feature.is_grouped_sequence:
            # rtp do not use __ concat sequence_name and feature_name
            # pyre-ignore [16]
            seqname_mapping[feature.name] = (
                f"{feature.sequence_name}_{feature.config.feature_name}"
            )

    graph = copy.deepcopy(full_graph)
    output_keys = []
    output_values = []
    mc_config = defaultdict()
    for node in graph.nodes:
        if node.op == "output":
            for k, v in sorted(node.args[0].items()):
                output_keys.append(k)
                output_values.append(v)
            graph.erase_node(node)
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
                mc_config[name] = new_node.kwargs["keys"]
                node_kt.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target == fx_mark_tensor:
            # sequence or query name
            name = node.args[0]
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                new_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                keys = [
                    seqname_mapping[k] if k in seqname_mapping else k
                    for k in node.kwargs["keys"]
                ]
                mc_config[name] = keys
            node_t.replace_all_uses_with(new_node)
        elif node.op == "call_function" and node.target == fx_mark_seq_len:
            # sequence_length
            name = node.args[0]
            node_t = node.args[1]
            with graph.inserting_before(node_t):
                get_node = graph.call_function(
                    operator.getitem, args=(input_node, name)
                )
                # rtp do not support RANK=1 tensor
                new_node = graph.call_function(torch.squeeze, args=(get_node, 1))
                # add sequence_length into fg
                additional_fg.append(
                    {
                        "feature_name": name,
                        "feature_type": "raw_feature",
                        "expression": f"user:{name}",
                    }
                )
                mc_config[name] = [name]
            node_t.replace_all_uses_with(new_node)
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
        fx_tool = ExportTorchFxTool(os.path.join(save_dir, "fx_user_model"))
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
