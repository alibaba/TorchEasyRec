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

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

import torch.nn.functional as F

from tzrec.models.model import ScriptWrapper, ScriptWrapperAOT
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.logging_util import logger

import torch
from torch.export import Dim

def export_model_aot(
    model: nn.Module, data: Dict[str, torch.Tensor], save_dir: str
) -> None:
    """Export aot model.

    Args:
        model (nn.Module): the model
        data (Dict[str, torch.Tensor]): the test data
        save_dir (str): model save dir
    """
    gm = symbolic_trace(model)
    with open(os.path.join(save_dir, "gm.code"), "w") as f:
        f.write(gm.code)
    # with open(os.path.join(save_dir, "gm.graph"), "w") as f:
    #     f.write(gm.graph.print_tabular())

    gm = gm.cuda()

    print(gm)

    batch = Dim("batch")
    dynamic_shapes = {}
    for key in data:
        if key.endswith('.lengths'):
            if data[key].shape[0] == 1:
                logger.info('uniq user sparse fea %s length=1' % key)
                dynamic_shapes[key] = {}
            else:
                dynamic_shapes[key] = {0: batch}
        elif key == 'batch_size':
            dynamic_shapes[key] = {}
        elif data[key].dtype == torch.float32 and '__' not in key:
            if data[key].shape[0] == 1:
                logger.info('uniq user dense_fea=%s shape=%s' % (key, data[key].shape))
                dynamic_shapes[key] = {}
            else:
                logger.info('dense_fea=%s shape=%s' % (key, data[key].shape))
                dynamic_shapes[key] = {0: batch}
        elif data[key].dtype == torch.float32 and '__' in key and data[key].shape[0] == 1:
            logger.info('uniq seq_dense_fea=%s shape=%s' % (key, data[key].shape))
            dynamic_shapes[key] = {}
        else:
            tmp_val_dim = Dim(key.replace('.', '__') + "__batch", min=0)
            # to handle torch.export 0/1 specialization problem
            if data[key].shape[0] < 2:
                data[key] = F.pad(data[key], 
                    [0, 2] + [0, 0] * (len(data[key].shape) - 1), mode='constant')
            dynamic_shapes[key] = {0: tmp_val_dim}

    exported_gm = torch.export.export(gm, args=(data, ), 
             dynamic_shapes=(dynamic_shapes, ))
    print(exported_gm)

    export_path = os.path.join(save_dir, 'exported_gm.code')
    with open(export_path, 'w') as fout:
        fout.write(str(exported_gm))

    exported_gm_path = os.path.join(save_dir, 'debug_exported_gm.py')
    with open(exported_gm_path, 'w') as fout:
        fout.write(str(exported_gm))

    output = exported_gm.module()(data)

    return exported_gm
