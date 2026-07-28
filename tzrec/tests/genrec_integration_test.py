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

import glob
import os
import random
import shutil
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from tzrec.tests import utils
from tzrec.utils import config_util
from tzrec.utils.test_util import make_test_dir

_MOCK_CONFIG = "tzrec/tests/configs/qwen2_rec_lm_mock.config"
# must match the mock config's `common.codebook`
_CODEBOOK = [4, 4, 4]


def _write_backbone(save_dir: str, vocab_size: int = 256) -> str:
    """Save a tiny but COMPLETE Qwen2 model dir so no test downloads a real one.

    Weights are required, not just ``config.json``: cold-start training calls
    ``init_from_pretrained`` -> ``from_pretrained``, which refuses a dir with no
    checkpoint. The SID offsets only need a tokenizer whose ``len()`` is stable
    and which has room to append the ``C*`` atoms, so word-level is enough.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    eos = "<|endoftext|>"
    vocab = {t: i for i, t in enumerate([eos, "<|im_start|>", "<|im_end|>"])}
    for i in range(vocab_size - len(vocab)):
        vocab[f"b{i}"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token=eos))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token=eos,
        eos_token=eos,
        pad_token=eos,
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    ).save_pretrained(save_dir)
    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).save_pretrained(save_dir)
    return save_dir


def _write_samples(save_dir: str, num_rows: int, seed: int = 0) -> str:
    """Write the two-column sample contract: history + answer, both list<int64>.

    Codes are local 1-based per-level values in ``[1, codebook[level]]`` and
    every row holds whole items in level order.
    """
    rnd = random.Random(seed)

    def _item():
        return [rnd.randint(1, size) for size in _CODEBOOK]

    schema = pa.schema(
        [
            pa.field("user_sequence", pa.list_(pa.int64()), nullable=False),
            pa.field("label", pa.list_(pa.int64()), nullable=False),
        ]
    )
    table = pa.table(
        {
            "user_sequence": [
                [c for _ in range(rnd.randint(1, 4)) for c in _item()]
                for _ in range(num_rows)
            ],
            "label": [_item() for _ in range(num_rows)],
        },
        schema=schema,
    )
    pq.write_table(table, os.path.join(save_dir, "part-0.parquet"))
    return os.path.join(save_dir, "*.parquet")


class GenRecIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir()
        # every rank builds its own backbone; one is enough for a mock run.
        patcher = mock.patch.dict(
            os.environ, {"TEST_NPROC_PER_NODE": "1", "HF_HUB_OFFLINE": "1"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if self.success and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _prepare_config(self, num_rows: int = 64) -> str:
        """Point the mock config at a local backbone and freshly written samples."""
        backbone = _write_backbone(os.path.join(self.test_dir, "backbone"))
        data_dir = os.path.join(self.test_dir, "genrec_data")
        os.makedirs(data_dir, exist_ok=True)
        data_glob = _write_samples(data_dir, num_rows)

        config = config_util.load_pipeline_config(_MOCK_CONFIG)
        config.train_input_path = data_glob
        config.eval_input_path = data_glob
        config.model_config.qwen2_rec_lm.hf_model_id = backbone
        config_path = os.path.join(self.test_dir, "genrec.config")
        config_util.save_message(config, config_path)
        return config_path

    def test_mock_config_builds_the_model_and_runs_a_batch(self) -> None:
        """The config -> model path: oneof dispatch and the derived contract.

        The sample contract is derived, not configured, so only a real
        ``ModelConfig`` proves that the history feature_group, the answer
        label_field and the SID vocab extension line up.
        """
        from tzrec.constant import Mode
        from tzrec.datasets.dataset import create_dataloader
        from tzrec.main import _create_features, _create_model

        config = config_util.load_pipeline_config(self._prepare_config())
        features = _create_features(list(config.feature_configs), config.data_config)
        model = _create_model(
            config.model_config, features, list(config.data_config.label_fields)
        )
        self.assertEqual(type(model).__name__, "Qwen2RecLM")
        self.assertEqual(model._history_group, "sids")
        self.assertEqual(model._input_name, "user_sequence")
        self.assertEqual(model._label_name, "label")
        self.assertEqual(model._num_levels, len(_CODEBOOK))
        # base vocab + sum(codebook) atoms, padded to vocab_pad_to_multiple_of
        self.assertGreaterEqual(
            model.lm.config.vocab_size, model._base_vocab + sum(_CODEBOOK)
        )
        self.assertEqual(model.lm.config.vocab_size % 128, 0)

        dataloader = create_dataloader(
            config.data_config, features, config.train_input_path, mode=Mode.TRAIN
        )
        model.train()
        predictions = model.predict(next(dataloader.get_iterator()))
        self.assertEqual(list(predictions), ["loss"])
        self.assertTrue(bool(predictions["loss"].isfinite()))
        self.success = True

    def test_qwen2_rec_lm_train_eval(self) -> None:
        """End-to-end train -> checkpoint, with HF assets co-located."""
        config_path = self._prepare_config()
        self.success = utils.test_train_eval(config_path, self.test_dir)
        self.assertTrue(self.success)
        ckpts = glob.glob(os.path.join(self.test_dir, "train", "model.ckpt-*"))
        self.assertTrue(ckpts, "no checkpoint persisted")
        # export_format: HF -> each checkpoint is convertible without the code
        for name in ("config.json", "tokenizer.json"):
            self.assertTrue(
                os.path.exists(os.path.join(ckpts[0], name)),
                f"{name} not co-located in {ckpts[0]}",
            )


if __name__ == "__main__":
    unittest.main()
