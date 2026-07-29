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
from google.protobuf import text_format
from torch import nn

from tzrec.features.feature import create_features
from tzrec.models.generative_model import BaseGenerativeModel
from tzrec.models.generative_qwen import GenerativeQwen
from tzrec.models.model import BaseModel
from tzrec.protos import feature_pb2, model_pb2
from tzrec.protos.models import generative_model_pb2


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


def _sid_feature(name="user_sequence", codebook=(2, 3, 4), prefix_text=""):
    """A real SidFeature -- the model dispatches on the type, not on duck-typing."""
    fc = feature_pb2.FeatureConfig()
    text_format.Merge(
        f'sequence_sid_feature {{ feature_name: "{name}" expression: "user:{name}" '
        + " ".join(f"codebook: {c}" for c in codebook)
        + f' prefix_text: "{prefix_text}" }}',
        fc,
    )
    return create_features([fc])[0]


def _common(**overrides):
    """A fake ``GenerativeModelConfig`` -- only the fields the base actually reads."""
    fields = {
        "ignore_index": -100,
        "generated_sids_key": "generated_sids",
        "param_dtype": generative_model_pb2.FP32,
        "vocab_pad_to_multiple_of": 128,
        "max_sequence_length": 0,
        "beam_widths": [],
        "num_return_sequences": 50,
    }
    return types.SimpleNamespace(**{**fields, **overrides})


def _wired(features=None, group_type=model_pb2.JAGGED_SEQUENCE, members=None):
    """Pre-``__init__`` state: the features/labels/groups the config-time code reads."""
    m = object.__new__(GenerativeQwen)
    nn.Module.__init__(m)
    m._features = [_sid_feature()] if features is None else features
    m._labels = ["label"]
    m._feature_groups = [
        types.SimpleNamespace(
            group_name="user_seq",
            feature_names=["user_sequence"] if members is None else list(members),
            group_type=group_type,
        )
    ]
    return m


def _stub(codebook=None, base_vocab=100, device="cpu"):
    """A GenerativeQwen with the base data-prep state wired up, but no HF backbone."""
    codebook = codebook or [2, 3, 4]
    m = object.__new__(GenerativeQwen)
    nn.Module.__init__(m)
    m._base_vocab = base_vocab
    m._num_levels = len(codebook)
    m.lm = types.SimpleNamespace(device=torch.device(device))
    sizes = torch.tensor(codebook, dtype=torch.long)
    m.register_buffer("_codebook_sizes", sizes, persistent=False)
    m.register_buffer(
        "_level_offsets", torch.cumsum(sizes, 0) - sizes, persistent=False
    )
    return m


class BaseGenerativeModelTest(unittest.TestCase):
    def test_registry_dispatch(self) -> None:
        self.assertIs(BaseModel.create_class("GenerativeQwen"), GenerativeQwen)
        self.assertTrue(issubclass(GenerativeQwen, BaseGenerativeModel))

    def test_model_config_oneof_resolves_to_the_class(self) -> None:
        # the path _create_model takes: oneof -> message type name -> class.
        from tzrec.utils import config_util

        cfg = model_pb2.ModelConfig()
        cfg.generative_qwen.common.max_sequence_length = 8  # required field
        self.assertEqual(config_util.which_msg(cfg, "model"), "GenerativeQwen")
        self.assertIs(
            BaseModel.create_class(config_util.which_msg(cfg, "model")), GenerativeQwen
        )

    def test_resolve_pad_token_id(self) -> None:
        tok = types.SimpleNamespace
        self.assertEqual(
            BaseGenerativeModel._resolve_pad_token_id(
                tok(pad_token_id=5, eos_token_id=9)
            ),
            5,
        )
        self.assertEqual(
            BaseGenerativeModel._resolve_pad_token_id(
                tok(pad_token_id=None, eos_token_id=9)
            ),
            9,
        )
        # neither -> a clear error, not an opaque int(None) TypeError
        with self.assertRaisesRegex(ValueError, "neither pad_token_id nor"):
            BaseGenerativeModel._resolve_pad_token_id(
                tok(pad_token_id=None, eos_token_id=None)
            )

    def test_backbone_owned_by_family_proto(self) -> None:
        from tzrec.protos.models.generative_model_pb2 import (
            GenerativeModelConfig,
        )
        from tzrec.protos.models.generative_model_pb2 import (
            GenerativeQwen as GenerativeQwenProto,
        )

        self.assertEqual(GenerativeQwenProto().hf_model_id, "Qwen/Qwen2.5-0.5B")
        common_fields = [f.name for f in GenerativeModelConfig.DESCRIPTOR.fields]
        self.assertNotIn("hf_model_id", common_fields)

    def test_configurable_knob_defaults(self) -> None:
        from tzrec.protos.models.generative_model_pb2 import GenerativeModelConfig

        c = GenerativeModelConfig()
        self.assertEqual(c.generated_sids_key, "generated_sids")
        self.assertEqual(c.param_dtype, generative_model_pb2.FP32)
        self.assertIs(
            GenerativeQwen._PARAM_DTYPE[generative_model_pb2.FP32], torch.float32
        )
        self.assertIs(
            GenerativeQwen._PARAM_DTYPE[generative_model_pb2.BF16], torch.bfloat16
        )

    def test_read_common_config_reads_knobs(self) -> None:
        m = _wired()
        sid_atoms = m._read_common_config(
            _common(
                max_sequence_length=288,
                generated_sids_key="my_sids",
                param_dtype=generative_model_pb2.BF16,
            )
        )
        self.assertEqual(m._label_name, "label")  # from label_fields[0]
        self.assertEqual(m._generated_sids_key, "my_sids")
        self.assertIs(m._param_dtype, torch.bfloat16)
        self.assertEqual(m._max_seq_length, 288)
        self.assertEqual(sid_atoms, 9)
        self.assertEqual(m._level_offsets.tolist(), [0, 2, 5])
        self.assertEqual(m._codebook_sizes.tolist(), [2, 3, 4])
        self.assertNotIn("_level_offsets", m.state_dict())
        self.assertNotIn("_codebook_sizes", m.state_dict())
        # the enum is closed: protobuf itself rejects an unlisted value
        cfg = generative_model_pb2.GenerativeModelConfig()
        with self.assertRaises(ValueError):
            cfg.param_dtype = 99

    def test_read_common_config_tolerates_no_feature_group(self) -> None:
        # group validation is prompt-driven, so it lives in _resolve_prompt_slots
        m = _wired()
        m._feature_groups = []
        m._read_common_config(_common())
        self.assertEqual(m._num_levels, 3)

    def test_max_sequence_length_below_one_item_raises(self) -> None:
        # a budget under num_levels floors to zero whole items, which would
        # silently leave the history uncapped instead of capping it.
        for cap in (1, 2):
            with self.subTest(cap=cap):
                with self.assertRaisesRegex(ValueError, "cannot hold one 3-level"):
                    _wired()._read_common_config(_common(max_sequence_length=cap))
        # 0 disables the budget; num_levels is the smallest meaningful cap
        for cap in (0, 3):
            with self.subTest(cap=cap):
                m = _wired()
                m._read_common_config(_common(max_sequence_length=cap))
                self.assertEqual(m._max_seq_length, cap)

    def test_one_feature_claimed_by_two_groups_raises(self) -> None:
        # keyed by feature, so a second claim would otherwise just overwrite
        m = _wired()
        m._feature_groups.append(
            types.SimpleNamespace(
                group_name="user_seq_dup",
                feature_names=["user_sequence"],
                group_type=model_pb2.JAGGED_SEQUENCE,
            )
        )
        with self.assertRaisesRegex(ValueError, "claimed by both"):
            m._slot_group_names()

    def test_slot_group_and_shared_codebook_validation(self) -> None:
        # a SEQUENCE group emits the same key with padded-dense semantics.
        with self.assertRaisesRegex(ValueError, "must be JAGGED_SEQUENCE"):
            _wired(group_type=model_pb2.SEQUENCE)._slot_group_names()
        with self.assertRaisesRegex(ValueError, "exactly one feature"):
            _wired(members=())._slot_group_names()
        # a codebook the SID features disagree on is the model's business
        m = _wired(
            features=[
                _sid_feature("user_sequence", (2, 3)),
                _sid_feature("other_seq", (4, 4)),
            ]
        )
        with self.assertRaisesRegex(ValueError, "must share one"):
            m._shared_sid_space()
        m._features = []
        with self.assertRaisesRegex(ValueError, "no SID feature declares"):
            m._shared_sid_space()

    def test_vocab_pad_zero_disables_padding(self) -> None:
        m = _wired()
        m._read_common_config(_common(vocab_pad_to_multiple_of=0))
        self.assertEqual(m._vocab_pad_mult, 0)  # not silently rewritten to 128

    def test_max_sequence_length_model_knob(self) -> None:
        m = _wired()
        m._read_common_config(_common(max_sequence_length=128))
        self.assertEqual(m._max_seq_length, 128)
        m2 = _wired()
        m2._read_common_config(_common(max_sequence_length=0))
        self.assertEqual(m2._max_seq_length, 0)  # 0 = off, no fallback

    def test_resolve_prompt_slots_splits_and_records(self) -> None:
        m = _wired()
        gaps, features = m._resolve_prompt_slots("A{{user_sequence}}B")
        self.assertEqual(gaps, ["A", "B"])  # N slots -> N+1 gaps
        self.assertEqual([f.name for f in features], ["user_sequence"])
        # recorded here, not in the family hook -- build_input reads them
        self.assertEqual(m._slot_names, ["user_sequence"])
        self.assertEqual(m._slot_groups, ["user_seq"])

    def test_resolve_prompt_slots_rejects_a_mismatched_template(self) -> None:
        from tzrec.features.feature import create_features

        other = _wired(
            features=[
                _sid_feature("user_sequence"),
                _sid_feature("other_seq"),
            ]
        )
        other._feature_groups.append(
            types.SimpleNamespace(
                group_name="other_seq_group",
                feature_names=["other_seq"],
                group_type=model_pb2.JAGGED_SEQUENCE,
            )
        )
        raw = feature_pb2.FeatureConfig()
        text_format.Merge(
            'id_feature { feature_name: "user_sequence" expression: "user:x" '
            "num_buckets: 8 embedding_dim: 4 }",
            raw,
        )
        not_a_sid = _wired(features=create_features([raw]))

        for model, template, msg in (
            (_wired(), "no slot at all", "at least one"),
            (_wired(), "{{nope}} x", "names no feature_group"),
            (other, "{{user_sequence}}", "never referenced"),
            (not_a_sid, "{{user_sequence}}", "only a SidFeature"),
        ):
            with self.subTest(template=template):
                with self.assertRaisesRegex(ValueError, msg):
                    model._resolve_prompt_slots(template)

        # a group whose feature_config vanished: reachable only out of sync
        orphan = _wired()
        orphan._features = []
        with self.assertRaisesRegex(ValueError, "no feature_config"):
            orphan._resolve_prompt_slots("{{user_sequence}}")

    def test_beam_config_defaults_and_validation(self) -> None:
        def read(widths, num_return, levels=3):
            m = object.__new__(GenerativeQwen)
            m._num_levels = levels
            m._read_beam_config(
                types.SimpleNamespace(
                    beam_widths=widths, num_return_sequences=num_return
                )
            )
            return m

        # empty -> flat DEFAULT_BEAM_WIDTH per level; anything else verbatim
        self.assertEqual(read([], 50)._beam_widths, [50, 50, 50])
        self.assertEqual(read([100, 200, 400], 400)._beam_widths, [100, 200, 400])
        with self.assertRaisesRegex(ValueError, "one width per level"):
            read([50, 50], 50)
        with self.assertRaisesRegex(ValueError, "must not exceed the final"):
            read([50, 50, 50], 80)

    def test_abstract_hooks_raise(self) -> None:
        base = object.__new__(BaseGenerativeModel)
        with self.assertRaises(NotImplementedError):
            base._build_prompt_tokens(None, None)
        with self.assertRaises(NotImplementedError):
            base.predict(None)

    def test_device_property(self) -> None:
        self.assertEqual(_stub(device="cpu").device, torch.device("cpu"))

    def test_tokenize_sids(self) -> None:
        m = _stub(base_vocab=100)
        # 0-based codes: the flat index IS the atom index, so no bridging shift.
        flat = torch.tensor([[0, 2, 5], [1, 4, 8]])  # offsets [0, 2, 5]
        level_ids = torch.arange(3)
        out = m._tokenize_sids(flat)
        self.assertEqual(out.tolist(), [[100, 102, 105], [101, 104, 108]])
        self.assertEqual(out.dtype, torch.int64)
        # decode goes the whole way back to per-level codes
        self.assertEqual(
            m._detokenize_sids(out, level_ids).tolist(), [[0, 0, 0], [1, 2, 3]]
        )

    def test_sid_token_bands_use_same_level_aware_mapping(self) -> None:
        m = _stub(base_vocab=100)
        lo, hi = m._sid_token_bands()
        self.assertEqual(lo.tolist(), [100, 102, 105])
        self.assertEqual(hi.tolist(), [101, 104, 108])

    def test_sid_token_rows_shifts_and_splits(self) -> None:
        # values arrive FLAT from SidFeature._parse; the model adds base_vocab
        m = _stub(base_vocab=100)
        jt = _FakeJT([0, 2, 5, 1, 4, 8, 0, 3, 6], [6, 3])
        rows = m._sid_token_rows(jt.values(), jt.lengths())
        self.assertEqual(
            [r.tolist() for r in rows],
            [[100, 102, 105, 101, 104, 108], [100, 103, 106]],
        )
        self.assertTrue(all(r.dtype == torch.int64 for r in rows))

    def test_sid_token_rows_squeezes_n1(self) -> None:
        m = _stub(base_vocab=100)
        jt = _FakeJT([0, 2, 5], [3], dim2=True)  # (N, 1)
        rows = m._sid_token_rows(jt.values(), jt.lengths())
        self.assertEqual([r.tolist() for r in rows], [[100, 102, 105]])

    def test_sid_token_rows_recency_clip(self) -> None:
        m = _stub(base_vocab=100)
        flat = [0, 2, 5, 1, 3, 6, 0, 4, 7, 1, 2, 8, 0, 3, 5]  # 5 whole items
        values = torch.tensor(flat, dtype=torch.float)
        lengths = torch.tensor([values.numel()])
        tail = [100 + v for v in flat[6:]]  # last three items, +base_vocab
        # cap 9 -> keep the most recent three whole items
        self.assertEqual(
            m._sid_token_rows(values, lengths, max_codes=9)[0].tolist(), tail
        )
        # item-aligned: cap 10 still keeps 9, never cuts mid-item
        self.assertEqual(
            m._sid_token_rows(values, lengths, max_codes=10)[0].tolist(), tail
        )
        # disabled -> untouched
        self.assertEqual(
            m._sid_token_rows(values, lengths, max_codes=0)[0].tolist(),
            [100 + v for v in flat],
        )

    def test_answer_token_rows_folds_offsets_and_validates(self) -> None:
        # the answer is a label_field, not a feature, so the model still owns
        # the per-level fold-in for it.
        m = _stub(base_vocab=100)
        jt = _FakeJT([0, 0, 0, 1, 2, 3], [3, 3])
        rows = m._answer_token_rows(jt.values(), jt.lengths())
        self.assertEqual([r.tolist() for r in rows], [[100, 102, 105], [101, 104, 108]])
        with self.assertRaisesRegex(ValueError, "each answer must be"):
            bad = _FakeJT([0, 0, 0, 0, 0, 0], [2, 4])
            m._answer_token_rows(bad.values(), bad.lengths())
        with self.assertRaisesRegex(ValueError, "local 0-based"):
            bad = _FakeJT([0, 3, 0], [3])  # 3 == codebook[1]
            m._answer_token_rows(bad.values(), bad.lengths())

    def test_build_input_history_group_label_field(self) -> None:
        m = _stub(base_vocab=100)
        m._label_name = "label"
        m._slot_names = ["user_sequence"]
        m._slot_groups = ["user_seq"]
        m._max_seq_length = 0
        m._is_inference = False  # train: the answer label_field is read too
        m.embedding_group = lambda b: {
            # flat indices, as SidFeature._parse emits them
            "user_seq.sequence": torch.tensor(
                [0.0, 2.0, 5.0, 1.0, 4.0, 8.0, 1.0, 2.0, 7.0]
            ),
            "user_seq.sequence_length": torch.tensor([6, 3]),
        }
        batch = types.SimpleNamespace(
            jagged_labels={"label": _FakeJT([1, 2, 3, 0, 1, 2], [3, 3])}
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
        m._label_name = "label"
        m._slot_names = ["user_sequence"]
        m._slot_groups = ["user_seq"]
        m._max_seq_length = 0
        m._is_inference = True  # inference: history only, no ground-truth label
        m.embedding_group = lambda b: {
            "user_seq.sequence": torch.tensor([0.0, 2.0, 5.0]),
            "user_seq.sequence_length": torch.tensor([3]),
        }
        rows = m.build_input(types.SimpleNamespace(jagged_labels={}))
        self.assertEqual([r.tolist() for r in rows["user_sequence"]], [[100, 102, 105]])
        self.assertNotIn("label", rows)


if __name__ == "__main__":
    unittest.main()
