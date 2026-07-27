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


def _stub(codebook=None, base_vocab=100, device="cpu"):
    """A Qwen2RecLM with the base data-prep state wired up, but no HF backbone.

    Exercises the architecture-agnostic base methods (inherited by every family)
    without downloading a model.
    """
    codebook = codebook or [2, 3, 4]
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._base_vocab = base_vocab
    m._num_levels = len(codebook)
    m.lm = types.SimpleNamespace(device=torch.device(device))
    lo, hi = Qwen2RecLM._sid_level_bands(codebook)
    m.register_buffer("_level_offsets", lo - 1, persistent=False)
    m.register_buffer("_codebook_sizes", hi - lo + 1, persistent=False)
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

    def test_configurable_knob_defaults(self) -> None:
        # generated_sids_key / param_dtype are proto knobs whose defaults are the
        # previous class-constant values.
        from tzrec.protos.models.generative_model_pb2 import GenerativeRecLMConfig

        c = GenerativeRecLMConfig()
        self.assertEqual(c.generated_sids_key, "generated_sids")
        self.assertEqual(c.param_dtype, "float32")
        self.assertIs(Qwen2RecLM._DTYPE_BY_NAME["float32"], torch.float32)
        self.assertIs(Qwen2RecLM._DTYPE_BY_NAME["bfloat16"], torch.bfloat16)

    def test_read_common_config_reads_knobs(self) -> None:
        m = object.__new__(Qwen2RecLM)
        nn.Module.__init__(m)
        m._features = []
        # The history group + SID-column names are DERIVED, not configured: the
        # history is the single feature_group + its one member; the answer is the
        # first data_config.label_field.
        m._labels = ["label"]
        m._feature_groups = [
            types.SimpleNamespace(
                group_name="user_seq", feature_names=["user_sequence"]
            )
        ]
        common = types.SimpleNamespace(
            ignore_index=-100,
            generated_sids_key="my_sids",
            param_dtype="bfloat16",
            codebook=[2, 3, 4],
            vocab_pad_to_multiple_of=128,
            max_sequence_length=288,
        )
        sid_atoms = m._read_common_config(common)
        self.assertEqual(m._history_group, "user_seq")  # the single group
        self.assertEqual(m._input_name, "user_sequence")  # its one member
        self.assertEqual(m._label_name, "label")  # from label_fields[0]
        self.assertEqual(m._generated_sids_key, "my_sids")  # configurable
        self.assertIs(m._param_dtype, torch.bfloat16)  # name -> torch dtype
        self.assertEqual(m._max_seq_length, 288)  # model knob used
        self.assertEqual(sid_atoms, 9)
        self.assertEqual(m._level_offsets.tolist(), [0, 2, 5])
        self.assertEqual(m._codebook_sizes.tolist(), [2, 3, 4])
        self.assertNotIn("_level_offsets", m.state_dict())
        self.assertNotIn("_codebook_sizes", m.state_dict())
        # unknown dtype -> a clear error, not a KeyError
        common.param_dtype = "float64"
        with self.assertRaisesRegex(ValueError, "param_dtype must be one of"):
            m._read_common_config(common)

    def test_read_common_config_no_feature_group_raises(self) -> None:
        # the history is the single declared feature_group; none -> fail loudly.
        m = object.__new__(Qwen2RecLM)
        nn.Module.__init__(m)
        m._features = []
        m._labels = ["label"]
        m._feature_groups = []  # no group declared
        common = types.SimpleNamespace(
            ignore_index=-100,
            generated_sids_key="generated_sids",
            param_dtype="float32",
            codebook=[4, 4, 4],
            vocab_pad_to_multiple_of=128,
            max_sequence_length=0,
        )
        with self.assertRaisesRegex(ValueError, "no feature_group declared"):
            m._read_common_config(common)

    def test_max_sequence_length_model_knob(self) -> None:
        # _max_seq_length is the model knob; 0 = off (no feature fallback).
        def _common(max_seq):
            return types.SimpleNamespace(
                ignore_index=-100,
                generated_sids_key="generated_sids",
                param_dtype="float32",
                codebook=[4, 4, 4],
                vocab_pad_to_multiple_of=128,
                max_sequence_length=max_seq,
            )

        def _wired():
            m = object.__new__(Qwen2RecLM)
            nn.Module.__init__(m)
            m._features = []
            m._labels = ["label"]
            m._feature_groups = [
                types.SimpleNamespace(
                    group_name="user_seq", feature_names=["user_sequence"]
                )
            ]
            return m

        m = _wired()
        m._read_common_config(_common(128))
        self.assertEqual(m._max_seq_length, 128)  # model knob used
        m2 = _wired()
        m2._read_common_config(_common(0))
        self.assertEqual(m2._max_seq_length, 0)  # 0 = off, no fallback

    def test_abstract_hooks_raise(self) -> None:
        base = object.__new__(GenerativeRecLM)
        with self.assertRaises(NotImplementedError):
            base._build_prompt_tokens(None, None)
        with self.assertRaises(NotImplementedError):
            base.predict(None)

    def test_device_property(self) -> None:
        self.assertEqual(_stub(device="cpu").device, torch.device("cpu"))

    def test_tokenize_sids(self) -> None:
        m = _stub(base_vocab=100)
        codes = torch.tensor([[1, 1, 1], [2, 3, 4]])
        level_ids = torch.arange(3)
        out = m._tokenize_sids(codes, level_ids)
        self.assertEqual(out.tolist(), [[100, 102, 105], [101, 104, 108]])
        self.assertEqual(out.dtype, torch.int64)
        self.assertEqual(
            m._detokenize_sids(out, level_ids).tolist(),
            codes.tolist(),
        )

    def test_sid_token_bands_use_same_level_aware_mapping(self) -> None:
        m = _stub(base_vocab=100)
        lo, hi = m._sid_token_bands()
        self.assertEqual(lo.tolist(), [100, 102, 105])
        self.assertEqual(hi.tolist(), [101, 104, 108])

    def test_sid_token_rows_split_and_cast(self) -> None:
        m = _stub(base_vocab=100)
        jt = _FakeJT([1, 1, 1, 2, 3, 4, 2, 1, 3], [6, 3])
        rows = m._sid_token_rows(jt.values(), jt.lengths())
        self.assertEqual(
            [r.tolist() for r in rows],
            [[100, 102, 105, 101, 104, 108], [101, 102, 107]],
        )
        self.assertTrue(all(r.dtype == torch.int64 for r in rows))

    def test_sid_token_rows_squeezes_n1(self) -> None:
        m = _stub(base_vocab=100)
        jt = _FakeJT([1, 1, 1], [3], dim2=True)  # (N, 1)
        rows = m._sid_token_rows(jt.values(), jt.lengths())
        self.assertEqual([r.tolist() for r in rows], [[100, 102, 105]])

    def test_sid_token_rows_width_ok(self) -> None:
        m = _stub(base_vocab=100)
        jt = _FakeJT([1, 1, 1, 2, 3, 4], [3, 3])
        rows = m._sid_token_rows(jt.values(), jt.lengths(), expected_width=3)
        self.assertEqual(
            [r.tolist() for r in rows],
            [[100, 102, 105], [101, 104, 108]],
        )

    def test_sid_token_rows_width_violation_raises(self) -> None:
        m = _stub(base_vocab=100)
        with self.assertRaises(ValueError):
            # second row has 6 codes, not 3 -> anomalous answer sample
            jt = _FakeJT([1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 6])
            m._sid_token_rows(jt.values(), jt.lengths(), expected_width=3)

    def test_sid_token_rows_rejects_partial_items(self) -> None:
        m = _stub(base_vocab=100)
        with self.assertRaisesRegex(ValueError, "whole 3-level items"):
            # The total is divisible by 3, but neither row starts a whole item.
            jt = _FakeJT([1, 1, 1, 1, 1, 1], [2, 4])
            m._sid_token_rows(jt.values(), jt.lengths())

    def test_sid_token_rows_rejects_out_of_range_codes(self) -> None:
        m = _stub(base_vocab=100)
        for values in (
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [3, 1, 1],
            [1, 4, 1],
            [1, 1, 5],
            [1, 3, 6],  # legacy global 1-based representation
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "local 1-based"):
                    jt = _FakeJT(values, [3])
                    m._sid_token_rows(jt.values(), jt.lengths())

    def test_build_input_history_group_label_field(self) -> None:
        # history: EmbeddingGroup output keyed by GROUP name ("{group}.sequence"
        # / ".sequence_length"). answer: batch.jagged_labels[label_name]. Both
        # level-offset tokenized; returned dict keyed by FEATURE name.
        m = _stub(base_vocab=100)
        m._input_name, m._label_name = "user_sequence", "label"
        m._history_group = "user_seq"
        m._max_seq_length = 0
        m._is_inference = False  # train: the answer label_field is read too
        m.embedding_group = lambda b: {
            "user_seq.sequence": torch.tensor(
                [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 3.0]
            ),
            "user_seq.sequence_length": torch.tensor([6, 3]),
        }
        batch = types.SimpleNamespace(
            jagged_labels={"label": _FakeJT([2, 3, 4, 1, 2, 3], [3, 3])}
        )
        rows = m.build_input(batch)
        self.assertEqual(
            [r.tolist() for r in rows["user_sequence"]],
            [[100, 102, 105, 101, 104, 108], [101, 102, 107]],
        )
        self.assertEqual(
            [r.tolist() for r in rows["label"]],
            [[101, 104, 108], [100, 103, 107]],
        )

    def test_build_input_skips_label_in_inference(self) -> None:
        m = _stub(base_vocab=100)
        m._input_name, m._label_name = "user_sequence", "label"
        m._history_group = "user_seq"
        m._max_seq_length = 0
        m._is_inference = True  # inference: history only, no ground-truth label
        m.embedding_group = lambda b: {
            "user_seq.sequence": torch.tensor([1.0, 1.0, 1.0]),
            "user_seq.sequence_length": torch.tensor([3]),
        }
        # jagged_labels intentionally empty — the label is absent at inference
        rows = m.build_input(types.SimpleNamespace(jagged_labels={}))
        self.assertEqual([r.tolist() for r in rows["user_sequence"]], [[100, 102, 105]])
        self.assertNotIn("label", rows)


if __name__ == "__main__":
    unittest.main()
