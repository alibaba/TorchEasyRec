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

import dataclasses
import json
import os
import unittest
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from google.protobuf import text_format
from tokenizers import Tokenizer, models, pre_tokenizers

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature, FgMode, create_features
from tzrec.main import _create_features, _create_model, export
from tzrec.models.model import TrainWrapper
from tzrec.prompt.assembler import PromptAssembler
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.types import CompiledPrompt
from tzrec.protos import feature_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.hf_export_util import write_hf_assets
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import create_tiny_causal_lm, make_test_dir


def create_prompt_tokenizer(path: str, words: Sequence[str]) -> str:
    """Write a word-level tokenizer used by prompt tests.

    Args:
        path: destination JSON path.
        words: vocabulary entries in token-id order.

    Returns:
        The destination path.
    """
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={word: i for i, word in enumerate(words)}, unk_token="<unk>"
        )
    )
    # pyrefly: ignore[read-only]
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(path)
    return path


def create_prompt_feature(text: str) -> BaseFeature:
    """Create one prompt test feature from text-format protobuf.

    Args:
        text: text-format ``FeatureConfig``.

    Returns:
        The created feature.
    """
    config = feature_pb2.FeatureConfig()
    text_format.Merge(text, config)
    return create_features([config], fg_mode=FgMode.FG_NONE)[0]


def offset_sid_codes(codes: Sequence[Any], codebook: Sequence[int]) -> np.ndarray:
    """Shift local SID codes into the flattened per-level space.

    Args:
        codes: local codes grouped by SID item.
        codebook: vocabulary size for each SID level.

    Returns:
        Flat offset codes in item-major order.
    """
    offsets = np.cumsum([0, *codebook[:-1]])
    return (np.asarray(codes).reshape(-1, len(codebook)) + offsets).reshape(-1)


def assemble_into(
    compiled_prompt: CompiledPrompt,
    parsed_features: Dict[str, "np.ndarray"],
) -> Dict[str, np.ndarray]:
    """Assemble one parsed batch with a temporary assembler.

    Args:
        compiled_prompt: the compiled prompt.
        parsed_features: ``{column}.values`` / ``{column}.lengths`` as the data
            parser emits them.

    Returns:
        The assembled streams keyed for ``additional_infos``.
    """
    return PromptAssembler(
        compiled_prompt.prompt_plan, compiled_prompt.sid_space
    ).forward(parsed_features)


_CODEBOOK = [4, 4, 4]
_WORDS = ["History", "Predict", ":", ".", "<unk>", "<|im_end|>"]
_HIST = 'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }'


def projected_feature(name: str, dim: int) -> str:
    """A PROJECTED slot member: a sequence id feature with an embedding."""
    return (
        f'sequence_id_feature {{ feature_name: "{name}" expression: "user:{name}" '
        f"num_buckets: 32 embedding_dim: {dim} sequence_length: 2 }}"
    )


class GenrecModelTestBase(unittest.TestCase):
    """Builds a real GenrecCausalLMModel over a tiny backbone and prompt."""

    def setUp(self) -> None:
        """Build the tiny backbone, tokenizer and compiled prompt."""
        self.test_dir = make_test_dir()
        self.backbone = os.path.join(self.test_dir, "backbone")
        create_tiny_causal_lm(64).save_pretrained(self.backbone)
        self.tok = create_prompt_tokenizer(
            os.path.join(self.test_dir, "tok.json"), _WORDS
        )
        self.features = [create_prompt_feature(_HIST)]
        self.compiled_prompt = self._compile(self.features)

    def _compile(self, features, template="History : {{hist}} . Predict :", **kwargs):
        kwargs.setdefault("response", "{{answer}}")
        cfg = PromptConfig(tokenizer_path=self.tok, prompt=template, **kwargs)
        cfg.sid_space.codebook.extend(_CODEBOOK)
        return compile_prompt(cfg, features, ["answer"])

    def _model(
        self,
        features=None,
        compiled_prompt=-1,
        beam_widths=(2, 2, 2),
        num_return_sequences=2,
        lm_parameter_dtype=None,
        hf_model_name_or_path=None,
    ):
        model_config = ModelConfig()
        lm_cfg = model_config.genrec_causal_lm_model
        lm_cfg.hf_model_name_or_path = hf_model_name_or_path or self.backbone
        lm_cfg.common.beam_widths.extend(beam_widths)
        lm_cfg.common.num_return_sequences = num_return_sequences
        if lm_parameter_dtype is not None:
            lm_cfg.common.lm_parameter_dtype = lm_parameter_dtype
        return _create_model(
            model_config,
            self.features if features is None else features,
            ["answer"],
            compiled_prompt=(
                self.compiled_prompt if compiled_prompt == -1 else compiled_prompt
            ),
        )

    def _batch(self, parsed, compiled_prompt=None, sparse=None):
        streams = assemble_into(compiled_prompt or self.compiled_prompt, parsed)
        batch = Batch(sparse_features={BASE_DATA_GROUP: sparse} if sparse else {})
        batch.additional_infos.update(
            {k: torch.from_numpy(np.asarray(v)) for k, v in streams.items()}
        )
        return batch


def write_genrec_checkpoint(model: torch.nn.Module, ckpt_dir: str) -> str:
    """Save a model the way a training run does, minus the dynamic-table dump.

    Args:
        model: the bare genrec model; wrapped and initialized here.
        ckpt_dir: the ``model.ckpt-N`` directory to write.

    Returns:
        ``ckpt_dir``.
    """
    from torch.distributed.checkpoint import save

    wrapped = TrainWrapper(model)
    init_parameters(wrapped, torch.device("cpu"))
    save(wrapped.state_dict(), checkpoint_id=os.path.join(ckpt_dir, "model"))
    write_hf_assets(wrapped, ckpt_dir)
    return ckpt_dir


@dataclasses.dataclass
class ExportedGenrec:
    """A tiny genrec model, its checkpoint and its HF export.

    Args:
        config: the pipeline config the export ran on.
        config_path: where it was written.
        features: the created features.
        compiled_prompt: the prompt as the export compiled it.
        checkpoint_dir: the checkpoint the export converted.
        export_dir: the export directory.
    """

    config: EasyRecConfig
    config_path: str
    features: List[BaseFeature]
    compiled_prompt: CompiledPrompt
    checkpoint_dir: str
    export_dir: str


def export_tiny_genrec(
    test_dir: str,
    projected: bool = True,
    bundle_uuid: Optional[str] = None,
    hidden_size: int = 32,
) -> ExportedGenrec:
    """Train nothing, save a checkpoint of a tiny genrec model, and export it.

    The prompt carries an INLINE SID history and, when ``projected``, one
    PROJECTED behaviour slot, which is what makes the export write a front-end
    with tables and a projection.

    Args:
        test_dir: scratch directory.
        projected: whether to add the projected slot.
        bundle_uuid: when given, a SID manifest carrying it is written and
            referenced, so the export records a bundle identity.
        hidden_size: the tiny backbone's hidden size.

    Returns:
        Everything a test needs to read the export back.
    """
    backbone = os.path.join(test_dir, "backbone")
    create_tiny_causal_lm(64).save_pretrained(backbone)
    tok = create_prompt_tokenizer(os.path.join(test_dir, "tok.json"), _WORDS)
    manifest = ""
    if bundle_uuid is not None:
        manifest = os.path.join(test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": _CODEBOOK, "bundle_uuid": bundle_uuid}, f)

    config = EasyRecConfig()
    text_format.Merge(
        f'''
train_input_path: "" eval_input_path: "" model_dir: "{test_dir}/train"
train_config {{
  sparse_optimizer {{ adagrad_optimizer {{ lr: 0.0 }} constant_learning_rate {{}} }}
  dense_optimizer {{ adam_optimizer {{ lr: 0.0001 }} constant_learning_rate {{}} }}
  num_epochs: 1
}}
data_config {{
  batch_size: 4 dataset_type: ParquetDataset fg_mode: FG_NONE
  label_fields: "answer" num_workers: 1
}}
feature_configs {{ {_HIST} }}
{"feature_configs { " + projected_feature("beh", 8) + " }" if projected else ""}
prompt_config {{
  tokenizer_path: "{tok}"
  prompt: "History : {{{{hist}}}} .{" {{beh}}" if projected else ""} Predict :"
  response: "{{{{answer}}}}"
  sid_space {{
    codebook: 4 codebook: 4 codebook: 4
    {f'manifest_path: "{manifest}"' if manifest else ""}
  }}
  max_length: 64
}}
model_config {{
  genrec_causal_lm_model {{
    hf_model_name_or_path: "{backbone}"
    common {{ beam_widths: 2 beam_widths: 2 beam_widths: 2 num_return_sequences: 2 }}
  }}
}}
''',
        config,
    )
    os.makedirs(config.model_dir)
    config_path = os.path.join(test_dir, "pipeline.config")
    with open(config_path, "w") as f:
        f.write(text_format.MessageToString(config))

    features = _create_features(list(config.feature_configs), config.data_config)
    compiled_prompt = compile_prompt(config.prompt_config, features, ["answer"])
    model = _create_model(
        config.model_config, features, ["answer"], compiled_prompt=compiled_prompt
    )
    checkpoint_dir = write_genrec_checkpoint(
        model, os.path.join(config.model_dir, "model.ckpt-1")
    )
    export_dir = os.path.join(test_dir, "export")
    export(config_path, export_dir, checkpoint_path=checkpoint_dir)
    return ExportedGenrec(
        config=config,
        config_path=config_path,
        features=features,
        compiled_prompt=compiled_prompt,
        checkpoint_dir=checkpoint_dir,
        export_dir=export_dir,
    )
