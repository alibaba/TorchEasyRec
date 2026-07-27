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
from tzrec.protos import model_pb2


class _FakeJT:
    """Minimal stand-in for a TorchRec JaggedTensor."""

    def __init__(self, values, lengths, dim2=False):
        v = torch.tensor(values, dtype=torch.float)  # TER delivers list<int64> as float
        self._v = v.unsqueeze(-1) if dim2 else v
        self._l = torch.tensor(lengths)

    def values(self):
        return self._v

    def lengths(self):
        return self._l


def _stub(codebook=None, base_vocab=100, device="cpu"):
    """A Qwen2RecLM with the base data-prep state wired up, but no HF backbone."""
    codebook = codebook or [2, 3, 4]
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._base_vocab = base_vocab
    m._num_levels = len(codebook)
    m.lm = types.SimpleNamespace(device=torch.device(device))
    offsets, sizes = Qwen2RecLM._sid_level_layout(codebook)
    m.register_buffer("_level_offsets", offsets, persistent=False)
    m.register_buffer("_codebook_sizes", sizes, persistent=False)
    return m


class GenerativeRecLMTest(unittest.TestCase):
    def test_registry_dispatch(self) -> None:
        self.assertIs(BaseModel.create_class("Qwen2RecLM"), Qwen2RecLM)
        self.assertTrue(issubclass(Qwen2RecLM, GenerativeRecLM))

    def test_model_config_oneof_resolves_to_the_class(self) -> None:
        # the path _create_model takes: oneof -> message type name -> class.
        from tzrec.utils import config_util

        cfg = model_pb2.ModelConfig()
        cfg.qwen2_rec_lm.common.codebook.extend([2, 3])
        cfg.qwen2_rec_lm.common.max_sequence_length = 8  # required field
        self.assertEqual(config_util.which_msg(cfg, "model"), "Qwen2RecLM")
        self.assertIs(
            BaseModel.create_class(config_util.which_msg(cfg, "model")), Qwen2RecLM
        )

    def test_resolve_pad_token_id(self) -> None:
        tok = types.SimpleNamespace
        self.assertEqual(
            GenerativeRecLM._resolve_pad_token_id(tok(pad_token_id=5, eos_token_id=9)),
            5,
        )
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
        m._labels = ["label"]
        m._feature_groups = [
            types.SimpleNamespace(
                group_name="user_seq",
                feature_names=["user_sequence"],
                group_type=model_pb2.JAGGED_SEQUENCE,
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
        self.assertEqual(m._generated_sids_key, "my_sids")
        self.assertIs(m._param_dtype, torch.bfloat16)
        self.assertEqual(m._max_seq_length, 288)
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
        m = object.__new__(Qwen2RecLM)
        nn.Module.__init__(m)
        m._features = []
        m._labels = ["label"]
        m._feature_groups = []
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

    def test_read_common_config_rejects_bad_group_and_codebook(self) -> None:
        def _wired(group_type, feature_names=("user_sequence",)):
            m = object.__new__(Qwen2RecLM)
            nn.Module.__init__(m)
            m._features = []
            m._labels = ["label"]
            m._feature_groups = [
                types.SimpleNamespace(
                    group_name="user_seq",
                    feature_names=list(feature_names),
                    group_type=group_type,
                )
            ]
            return m

        def _common(codebook):
            return types.SimpleNamespace(
                ignore_index=-100,
                generated_sids_key="generated_sids",
                param_dtype="float32",
                codebook=codebook,
                vocab_pad_to_multiple_of=128,
                max_sequence_length=0,
            )

        # a SEQUENCE group emits the same key with padded-dense semantics.
        with self.assertRaisesRegex(ValueError, "must be JAGGED_SEQUENCE"):
            _wired(model_pb2.SEQUENCE)._read_common_config(_common([2, 3]))
        with self.assertRaisesRegex(ValueError, "has no feature_names"):
            _wired(model_pb2.JAGGED_SEQUENCE, ())._read_common_config(_common([2, 3]))
        with self.assertRaisesRegex(ValueError, "codebook must be non-empty"):
            _wired(model_pb2.JAGGED_SEQUENCE)._read_common_config(_common([]))
        with self.assertRaisesRegex(ValueError, "codebook size must be positive"):
            _wired(model_pb2.JAGGED_SEQUENCE)._read_common_config(_common([2, 0]))

    def test_vocab_pad_zero_disables_padding(self) -> None:
        m = object.__new__(Qwen2RecLM)
        nn.Module.__init__(m)
        m._features = []
        m._labels = ["label"]
        m._feature_groups = [
            types.SimpleNamespace(
                group_name="user_seq",
                feature_names=["user_sequence"],
                group_type=model_pb2.JAGGED_SEQUENCE,
            )
        ]
        m._read_common_config(
            types.SimpleNamespace(
                ignore_index=-100,
                generated_sids_key="generated_sids",
                param_dtype="float32",
                codebook=[2, 3],
                vocab_pad_to_multiple_of=0,
                max_sequence_length=0,
            )
        )
        self.assertEqual(m._vocab_pad_mult, 0)  # not silently rewritten to 128

    def test_max_sequence_length_model_knob(self) -> None:
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
                    group_name="user_seq",
                    feature_names=["user_sequence"],
                    group_type=model_pb2.JAGGED_SEQUENCE,
                )
            ]
            return m

        m = _wired()
        m._read_common_config(_common(128))
        self.assertEqual(m._max_seq_length, 128)
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
            [1, 3, 6],  # global cross-level codes, not local per-level
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "local 1-based"):
                    jt = _FakeJT(values, [3])
                    m._sid_token_rows(jt.values(), jt.lengths())

    def test_sid_token_rows_recency_clip(self) -> None:
        m = _stub(base_vocab=100)
        items = [(1, 1, 1), (2, 3, 4), (1, 2, 3), (2, 1, 4), (1, 3, 2)]
        toks = [
            [100, 102, 105],
            [101, 104, 108],
            [100, 103, 107],
            [101, 102, 108],
            [100, 104, 106],
        ]
        values = torch.tensor(items, dtype=torch.float).flatten()
        lengths = torch.tensor([values.numel()])

        # 15 codes (5 items), cap 9 -> keep the most recent three whole items.
        rows = m._sid_token_rows(values, lengths, max_codes=9)
        self.assertEqual(rows[0].tolist(), sum(toks[2:], []))
        # item-aligned: cap 10 still keeps 9 (3 whole items), never cuts mid-item
        rows = m._sid_token_rows(values, lengths, max_codes=10)
        self.assertEqual(rows[0].tolist(), sum(toks[2:], []))
        # within cap -> untouched
        rows = m._sid_token_rows(values[:6], torch.tensor([6]), max_codes=9)
        self.assertEqual(rows[0].tolist(), sum(toks[:2], []))
        # disabled (0/None) -> no clip
        rows = m._sid_token_rows(values, lengths, max_codes=0)
        self.assertEqual(rows[0].tolist(), sum(toks, []))

    def test_validate_sid_candidates_groups_batch_major(self) -> None:
        m = _stub(base_vocab=100)
        # decoders emit rows batch-major: [b0_c0, b0_c1, b1_c0, b1_c1]
        tokens = torch.tensor(
            [[100, 102, 105], [101, 104, 108], [101, 103, 106], [100, 104, 107]]
        )
        sids = m._validate_sid_candidates(tokens, batch_size=2)
        self.assertEqual(tuple(sids.shape), (2, 2, 3))
        self.assertEqual(sids[0].tolist(), [[1, 1, 1], [2, 3, 4]])
        self.assertEqual(sids[1].tolist(), [[2, 2, 2], [1, 3, 3]])

    def test_build_input_history_group_label_field(self) -> None:
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
        rows = m.build_input(types.SimpleNamespace(jagged_labels={}))
        self.assertEqual([r.tolist() for r in rows["user_sequence"]], [[100, 102, 105]])
        self.assertNotIn("label", rows)


if __name__ == "__main__":
    unittest.main()
