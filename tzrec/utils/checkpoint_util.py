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

import glob
import json
import os
from dataclasses import replace
from typing import List, Optional, Tuple

from torch import nn, optim
from torch.distributed.checkpoint import (
    FileSystemReader,
    TensorStorageMetadata,
    load,
    save,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DTensor,
    LoadPlan,
    _create_read_items,
)

from tzrec.constant import EVAL_RESULT_FILENAME
from tzrec.protos import export_pb2
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.logging_util import logger


class PartialLoadPlanner(DefaultLoadPlanner):
    """Support restore partial states.

    Args:
        flatten_state_dict (bool): Handle state_dict with nested dicts.
        flatten_sharded_tensors (bool): For FSDP in 2D parallel mode.
        ckpt_param_map_path (str): parameter mapping for checkpoint.
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        ckpt_param_map_path: Optional[str] = None,
    ) -> None:
        super().__init__(flatten_state_dict, flatten_sharded_tensors)
        self._ckpt_param_map = dict()
        if ckpt_param_map_path:
            with open(ckpt_param_map_path) as f:
                for line in f.readlines():
                    cur_param_name, old_param_name = line.strip().split()
                    self._ckpt_param_map[cur_param_name] = old_param_name

    def create_local_plan(self) -> LoadPlan:
        """Create local load plan."""
        requests = []

        # mapping old __BASE__.ec_list.0 to new __BASE__.ec_dict.{dim}
        ec_compat_map = {}
        # pyre-ignore [16]
        for k, v in self.metadata.state_dict_metadata.items():
            if k.endswith(".weight") and isinstance(v, TensorStorageMetadata):
                for old_pattern, new_pattern in [
                    ("mc_ec_list", "mc_ec_dict"),
                    ("ec_list", "ec_dict"),
                ]:
                    if f".{old_pattern}." in k:
                        parts = k.split(".")
                        pattern_idx = parts.index(old_pattern)
                        dim = v.size[1]
                        ec_compat_map[
                            f"{parts[pattern_idx - 1]}.{new_pattern}.{dim}"
                        ] = f"{parts[pattern_idx - 1]}.{old_pattern}.{parts[pattern_idx + 1]}"  # NOQA

        # pyre-ignore [16]
        for fqn, obj in self.state_dict.items():
            meta_fqn = fqn

            fqn_remap_set = set()
            if fqn in self._ckpt_param_map:
                meta_fqn = self._ckpt_param_map[fqn]
                fqn_remap_set.add(fqn)
                logger.info(f"Remap restore state [{fqn}] from [{meta_fqn}]")

            for ec_new, ec_old in ec_compat_map.items():
                if ec_new in meta_fqn:
                    new_meta_fqn = meta_fqn
                    meta_fqn = new_meta_fqn.replace(ec_new, ec_old)
                    fqn_remap_set.add(fqn)
                    logger.warning(
                        f"Remap EmbeddingCollection state [{new_meta_fqn}] from old "
                        "[{meta_fqn}], will be deprecated when tzrec version >= 1.0.0"
                    )

            if meta_fqn in self.metadata.state_dict_metadata:
                md = self.metadata.state_dict_metadata[meta_fqn]
            else:
                logger.warning(f"Skip restore state [{fqn}]")
                continue

            read_items = []
            if isinstance(obj, DTensor):
                if obj.device_mesh.get_coordinate() is not None:
                    read_items = _create_read_items(meta_fqn, md, obj)
            else:
                read_items = _create_read_items(meta_fqn, md, obj)

            if fqn in fqn_remap_set:
                read_items = [
                    replace(x, dest_index=replace(x.dest_index, fqn=fqn))
                    for x in read_items
                ]
            requests += read_items

        plan = LoadPlan(requests)
        return plan


def _get_checkpoint_step(ckpt_path: str) -> int:
    """Get checkpoint step from ckpt_path.

    Args:
        ckpt_path: checkpoint path, such as xx/model.ckpt-2000.

    Return:
        ckpt_step: checkpoint step, such as 2000.
    """
    _, ckpt_name = os.path.split(ckpt_path)
    ckpt_name, ext = os.path.splitext(ckpt_name)
    if ext.startswith(".ckpt-"):
        ckpt_name = ext
    toks = ckpt_name.split("-")
    try:
        ckpt_step = int(toks[-1])
    except Exception:
        ckpt_step = 0
    return ckpt_step


def latest_checkpoint(model_dir: str) -> Tuple[Optional[str], int]:
    """Find latest checkpoint under a directory.

    Args:
        model_dir: model directory

    Return:
        latest_ckpt_path: latest checkpoint path.
        latest_step: step of the latest checkpoint
    """
    if "model.ckpt-" not in model_dir:
        ckpt_metas = glob.glob(os.path.join(model_dir, "model.ckpt-*"))
        if len(ckpt_metas) == 0:
            model_ckpt_dir = os.path.join(model_dir, "model")
            optim_ckpt_dir = os.path.join(model_dir, "optimizer")
            if os.path.exists(model_ckpt_dir) or os.path.exists(optim_ckpt_dir):
                return model_dir, 0
            else:
                return None, -1
        if len(ckpt_metas) > 1:
            ckpt_metas.sort(key=lambda x: _get_checkpoint_step(x))
        latest_ckpt_path = ckpt_metas[-1]
    else:
        latest_ckpt_path = model_dir
    return latest_ckpt_path, _get_checkpoint_step(latest_ckpt_path)


def best_checkpoint(
    model_dir: str,
    export_config: export_pb2.ExportConfig,
    eval_result_filename: str = EVAL_RESULT_FILENAME,
) -> Tuple[Optional[str], int]:
    """Find best checkpoint under a directory.

    Args:
        model_dir: model directory
        export_config: export_pb2.ExportConfig
        eval_result_filename: evaluation result filename

    Return:
        latest_ckpt_path: latest checkpoint path.
        latest_step: step of the latest checkpoint
    """
    eval_path = os.path.join(model_dir, eval_result_filename)
    metric_name = ""
    if export_config.HasField("best_exporter_metric"):
        metric_name = export_config.best_exporter_metric
    if export_config.HasField("tower_name"):
        metric_name = f"{export_config.tower_name}_{export_config}"
    if os.path.isfile(eval_path):
        step_metric = {}
        with open(eval_path, "r") as f:
            for line in f:
                if line:
                    step = int(line.split("step:")[0].strip())
                    metric = json.loads(line.split("step:")[-1].strip())
                    if len(metric) == 1:
                        step_metric[step] = metric.values()[0]
                    else:
                        if metric_name == "_" or metric_name == "":
                            raise ValueError(
                                f"please set export_config best_exporter_metric "
                                f"and tower_name, has mertic name: {metric.keys()}"
                            )
                        if metric_name not in metric:
                            raise ValueError(
                                f"checkpoint {eval_result_filename}"
                                f" not find {metric_name} metric."
                            )
                        step_metric[step] = metric[metric_name]
        if len(step_metric) < 1:
            logger.info(
                f"not find eval result in {eval_result_filename}, "
                f"will search latest checkpoint"
            )
            return latest_checkpoint(model_dir)
        sorted_mertic = sorted(step_metric.items(), key=lambda x: x[1], reverse=True)
        max_metric_step = sorted_mertic[0][0]
        best_ckpt_path = os.path.join(model_dir, f"model.ckpt-{max_metric_step}")
        if os.path.exists(best_ckpt_path):
            logger.info(f"find best checkpoint is {best_ckpt_path}")
            return best_ckpt_path, max_metric_step
        else:
            raise ValueError(
                f"find best metric is {max_metric_step} step,"
                f"but not find {best_ckpt_path}."
            )
    else:
        logger.info(f"not find {eval_result_filename}, will search latest checkpoint")
        return latest_checkpoint(model_dir)


def restore_model(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    ckpt_param_map_path: Optional[str] = None,
) -> None:
    """Restore model state.

    Args:
        checkpoint_dir (str): easyrec model checkpoint dir.
        model (nn.Module): a EasyRec model.
        optimizer (optim.Optimizer, optional): a optimizer.
        ckpt_param_map_path (str): parameter mapping for checkpoint.
    """
    is_local_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_local_rank_zero:
        logger.info(f"Restoring checkpoint from {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"checkpoint_dir[{checkpoint_dir}] not exists.")

    meta_path = os.path.join(checkpoint_dir, "meta")
    model_ckpt_path = os.path.join(checkpoint_dir, "model")
    optim_ckpt_path = os.path.join(checkpoint_dir, "optimizer")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    if not meta.get("load_model", True):
        pass
    elif os.path.exists(model_ckpt_path):
        if is_local_rank_zero:
            logger.info(f"Restoring model state from {model_ckpt_path}...")
        state_dict = model.state_dict()
        load(
            state_dict,
            checkpoint_id=model_ckpt_path,
            planner=PartialLoadPlanner(ckpt_param_map_path=ckpt_param_map_path),
        )
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError(f"model_ckpt_path[{model_ckpt_path}] not exists.")

    if optimizer:
        if not meta.get("load_optim", True):
            pass
        elif os.path.exists(optim_ckpt_path):
            if is_local_rank_zero:
                logger.info(f"Restoring optimizer state from {optim_ckpt_path}...")
            state_dict = optimizer.state_dict()
            load(
                state_dict,
                checkpoint_id=optim_ckpt_path,
                planner=PartialLoadPlanner(ckpt_param_map_path=ckpt_param_map_path),
            )
            optimizer.load_state_dict(state_dict)
        else:
            if is_local_rank_zero:
                logger.warning(f"optim_ckpt_path[{optim_ckpt_path}] not exists.")

    if has_dynamicemb:
        from dynamicemb.dump_load import DynamicEmbLoad

        logger.info(f"{os.environ.get('RANK', 0)} restoring dynamic embedding...")
        DynamicEmbLoad(
            os.path.join(checkpoint_dir, "dynamicemb"),
            model,
            table_names=meta.get("dynamicemb_load_table_names", None),
            optim=meta.get("dynamicemb_load_optim", optimizer is not None),
        )
        logger.info(f"{os.environ.get('RANK', 0)} restore dynamic embedding finished.")


def save_model(
    checkpoint_dir: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None
) -> None:
    """Save model state.

    Args:
        checkpoint_dir (str): easyrec model checkpoint dir.
        model (nn.Module): a EasyRec model.
        optimizer (optim.Optimizer, optional): a optimizer.
    """
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Saving checkpoint to {checkpoint_dir}...")
    save(model.state_dict(), checkpoint_id=os.path.join(checkpoint_dir, "model"))
    if optimizer:
        save(
            optimizer.state_dict(),
            checkpoint_id=os.path.join(checkpoint_dir, "optimizer"),
        )
    if has_dynamicemb:
        from dynamicemb.dump_load import DynamicEmbDump

        DynamicEmbDump(
            os.path.join(checkpoint_dir, "dynamicemb"),
            model,
            optim=optimizer is not None,
        )
    # save model plan
    if hasattr(model, "_plan") and model._plan is not None:
        if int(os.environ.get("RANK", 0)) == 0:
            plan = {}
            for module_path, module_plan in model._plan.plan.items():
                plan[module_path] = {}
                for param_name, param_sharding in module_plan.items():
                    plan[module_path][param_name] = {
                        "sharding_type": param_sharding.sharding_type,
                        "compute_kernel": param_sharding.compute_kernel,
                        "ranks": param_sharding.ranks,
                    }
            with open(os.path.join(checkpoint_dir, "plan"), "w") as f:
                json.dump(plan, f)


def list_distcp_param(checkpoint_dir: str) -> List[str]:
    """List."""
    meta_paths = []
    if os.path.exists(os.path.join(checkpoint_dir, ".metadata")):
        meta_paths.append(checkpoint_dir)
    else:
        if os.path.exists(os.path.join(checkpoint_dir, "model", ".metadata")):
            meta_paths.append(os.path.join(checkpoint_dir, "model"))
        if os.path.exists(os.path.join(checkpoint_dir, "optimizer", ".metadata")):
            meta_paths.append(os.path.join(checkpoint_dir, "optimizer"))
    if len(meta_paths) == 0:
        raise RuntimeError(f"Can't find distribute checkpoint in {checkpoint_dir}")

    param_names = []
    for meta_path in meta_paths:
        reader = FileSystemReader(path=meta_path)
        meta = reader.read_metadata()
        logger.info(f"Params in {meta_path}:")
        for k, v in meta.state_dict_metadata.items():
            if isinstance(v, TensorStorageMetadata):
                param_names.append(k)
                logger.info(f"{k}: {v.size}")
    return param_names
