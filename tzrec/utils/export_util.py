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
import os
import shutil
from collections import OrderedDict
from typing import List, Optional, cast

import torch
from torch import distributed as dist
from torchrec.inference.modules import quantize_embeddings
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)

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
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger
from tzrec.utils.state_dict_util import fix_mch_state, init_parameters


def export_model(
    pipeline_config: EasyRecConfig,
    model: BaseModule,
    checkpoint_path: Optional[str],
    save_dir: str,
    assets: Optional[List[str]] = None,
) -> None:
    """Export a EasyRec model, may be a part of model in PipelineConfig."""
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
