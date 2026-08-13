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

from typing import Dict

import numpy as np

from tzrec.prompt.assembler import PromptAssembler
from tzrec.prompt.plan import CompiledPrompt


def assemble_into(
    prompt: CompiledPrompt,
    parsed: Dict[str, "np.ndarray"],
    ignore_index: int = -100,
) -> Dict[str, np.ndarray]:
    """Assemble one parsed batch with a temporary assembler.

    Args:
        prompt: the compiled prompt.
        parsed: ``{feature}.values`` / ``{feature}.lengths`` as the data parser
            emits them.
        ignore_index: label value outside the supervised span.

    Returns:
        The assembled streams keyed for ``additional_infos``.
    """
    return PromptAssembler(
        prompt.prompt_plan, prompt.sid_space, ignore_index
    ).assemble_batch(parsed)
