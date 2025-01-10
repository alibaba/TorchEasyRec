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



from typing import Dict

import torch
from torch import nn

from tzrec.acc.export_utils import export_pm


def export_model_aot(
    model: nn.Module, data: Dict[str, torch.Tensor], save_dir: str
) -> torch.export.ExportedProgram:
    """Export aot model.

    Args:
        model (nn.Module): the model
        data (Dict[str, torch.Tensor]): the test data
        save_dir (str): model save dir
    """
    exported_pg,data = export_pm(model, data, save_dir)
    
    # TODO(aot cmpile)
    return exported_pg
