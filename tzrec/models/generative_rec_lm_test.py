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

import types
import unittest

import torch
from torch import nn

from tzrec.models.generative_rec_lm import GenerativeRecLM
from tzrec.models.model import BaseModel
from tzrec.models.qwen2_rec_lm import Qwen2RecLM


class _FakeJT:
    """Minimal stand-in for a TorchRec JaggedTensor (callable values/lengths)."""

    def __init__(self, values, lengths, dim2=False):
        v = torch.tensor(values, dtype=torch.float)  # TER delivers list<int64> as float
        self._v = v.unsqueeze(-1) if dim2 else v
        self._l = torch.tensor(lengths)

    def values(self):
        return self._v

    def lengths(self):
        return self._l


def _stub(num_levels=3, base_vocab=100, device="cpu"):
    """A Qwen2RecLM with the base data-prep state wired up, but no HF backbone.

    Exercises the architecture-agnostic base methods (inherited by every family)
    without downloading a model.
    """
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._base_vocab = base_vocab
    m._num_levels = num_levels
    m.lm = types.SimpleNamespace(device=torch.device(device))
    return m


class GenerativeRecLMTest(unittest.TestCase):
    def test_registry_dispatch(self) -> None:
        # importing qwen2_rec_lm auto-registers the family by class name
        self.assertIs(BaseModel.create_class("Qwen2RecLM"), Qwen2RecLM)
        self.assertTrue(issubclass(Qwen2RecLM, GenerativeRecLM))

    def test_resolve_pad_token_id(self) -> None:
        tok = types.SimpleNamespace
        # pad present -> pad
        self.assertEqual(
            GenerativeRecLM._resolve_pad_token_id(tok(pad_token_id=5, eos_token_id=9)),
            5,
        )
        # pad absent -> eos fallback
        self.assertEqual(
            GenerativeRecLM._resolve_pad_token_id(
                tok(pad_token_id=None, eos_token_id=9)
            ),
            9,
        )
        # neither -> a clear error, not an opaque int(None) TypeError
        with self.assertRaisesRegex(ValueError, "neither pad_token_id nor"):
            GenerativeRecLM._resolve_pad_token_id(
                tok(pad_token_id=None, eos_token_id=None)
            )

    def test_backbone_owned_by_family_proto(self) -> None:
        # the backbone lives on the family message (its architecture), NOT in
        # the shared common config; it defaults to the canonical Qwen2.5-0.5B.
        from tzrec.protos.models.generative_model_pb2 import (
            GenerativeRecLMConfig,
        )
        from tzrec.protos.models.generative_model_pb2 import (
            Qwen2RecLM as Qwen2RecLMProto,
        )

        self.assertEqual(Qwen2RecLMProto().hf_model_id, "Qwen/Qwen2.5-0.5B")
        common_fields = [f.name for f in GenerativeRecLMConfig.DESCRIPTOR.fields]
        self.assertNotIn("hf_model_id", common_fields)

    def test_abstract_hooks_raise(self) -> None:
        base = object.__new__(GenerativeRecLM)
        with self.assertRaises(NotImplementedError):
            base._build_prompt_tokens(None, None)
        with self.assertRaises(NotImplementedError):
            base.predict(None)

    def test_device_property(self) -> None:
        self.assertEqual(_stub(device="cpu").device, torch.device("cpu"))

    def test_tokenize_sids(self) -> None:
        m = _stub(base_vocab=100)  # token = sid + base - 1 = sid + 99
        out = m._tokenize_sids(torch.tensor([1, 2, 3]))
        self.assertEqual(out.tolist(), [100, 101, 102])
        self.assertEqual(out.dtype, torch.int64)
        # shape-agnostic: a 2-D batch maps elementwise
        out2 = m._tokenize_sids(torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(out2.tolist(), [[100, 101], [102, 103]])

    def test_sid_token_rows_split_and_cast(self) -> None:
        m = _stub(base_vocab=100)
        rows = m._sid_token_rows(_FakeJT([1, 2, 3, 4, 5], [3, 2]))
        self.assertEqual([r.tolist() for r in rows], [[100, 101, 102], [103, 104]])
        self.assertTrue(all(r.dtype == torch.int64 for r in rows))

    def test_sid_token_rows_squeezes_n1(self) -> None:
        m = _stub(base_vocab=100)
        rows = m._sid_token_rows(_FakeJT([1, 2, 3], [3], dim2=True))  # (N, 1)
        self.assertEqual([r.tolist() for r in rows], [[100, 101, 102]])

    def test_sid_token_rows_width_ok(self) -> None:
        m = _stub(base_vocab=100, num_levels=3)
        rows = m._sid_token_rows(_FakeJT([1, 2, 3, 4, 5, 6], [3, 3]), expected_width=3)
        self.assertEqual([r.tolist() for r in rows], [[100, 101, 102], [103, 104, 105]])

    def test_sid_token_rows_width_violation_raises(self) -> None:
        m = _stub(base_vocab=100, num_levels=3)
        with self.assertRaises(ValueError):
            # second row has 2 codes, not 3 -> anomalous sample
            m._sid_token_rows(_FakeJT([1, 2, 3, 4, 5], [3, 2]), expected_width=3)


if __name__ == "__main__":
    unittest.main()
