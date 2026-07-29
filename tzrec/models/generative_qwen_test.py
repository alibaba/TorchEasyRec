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
from unittest import mock

import torch
from parameterized import parameterized
from torch import nn

from tzrec.models.generative_qwen import GenerativeQwen
from tzrec.utils.test_util import create_tiny_causal_lm, parameterized_name_func


def _stub(codebook=None, base_vocab=100, pad_id=9, device="cpu"):
    """A GenerativeQwen with the splice-relevant state wired up, no HF backbone.

    The non-uniform default codebook makes incorrect ``level * uniform_size``
    offset arithmetic visible: sizes=[2,3,4], offsets=[0,2,5].
    """
    codebook = codebook or [2, 3, 4]
    m = object.__new__(GenerativeQwen)
    nn.Module.__init__(m)
    m._ignore_index = -100
    m._num_levels = len(codebook)
    m._base_vocab = base_vocab
    m._pad_token_id = pad_id
    m._num_return = 2
    m._beam_widths = [2] * m._num_levels
    m._max_seq_length = 0
    m._slot_names = ["user_sequence"]
    m._generated_sids_key = "generated_sids"
    m.lm = types.SimpleNamespace(device=torch.device(device))
    for name, vals in {
        "tpl_gap_0": [10, 11, 12],
        "tpl_gap_1": [13, 14],
        "tpl_asst_suffix": [15],
        "tpl_eos": [9],
    }.items():
        m.register_buffer(name, torch.tensor(vals, dtype=torch.long), persistent=False)
    sizes = torch.tensor(codebook, dtype=torch.long)
    m.register_buffer("_codebook_sizes", sizes, persistent=False)
    m.register_buffer(
        "_level_offsets", torch.cumsum(sizes, 0) - sizes, persistent=False
    )
    return m


def _real_lm_stub(codebook=None, base_vocab=20, beam_width=2):
    """A GenerativeQwen carrying a real (tiny, random) Qwen2 backbone.

    Needed wherever the real forward runs: the training objective and the
    end-to-end band-restricted decode; the other tests mock the kernel.
    """
    codebook = codebook or [2, 3, 4]
    m = object.__new__(GenerativeQwen)
    nn.Module.__init__(m)
    m._num_levels = len(codebook)
    m._base_vocab = base_vocab
    m._beam_widths = [beam_width] * m._num_levels
    m.lm = create_tiny_causal_lm(base_vocab + sum(codebook))
    sizes = torch.tensor(codebook, dtype=torch.long)
    m.register_buffer("_codebook_sizes", sizes, persistent=False)
    m.register_buffer(
        "_level_offsets", torch.cumsum(sizes, 0) - sizes, persistent=False
    )
    return m


def _wire_slots(m, prefix_text="", suffix_text=""):
    """Minimal _features/_feature_groups so the base slot resolver can run."""
    from google.protobuf import text_format

    from tzrec.features.feature import create_features
    from tzrec.protos import feature_pb2, model_pb2

    fc = feature_pb2.FeatureConfig()
    text_format.Merge(
        'sequence_sid_feature { feature_name: "user_sequence" '
        'expression: "user:user_sequence" codebook: 2 codebook: 3 codebook: 4 '
        f'prefix_text: "{prefix_text}" suffix_text: "{suffix_text}" }}',
        fc,
    )
    m._features = create_features([fc])
    m._feature_groups = [
        types.SimpleNamespace(
            group_name="sids",
            feature_names=["user_sequence"],
            group_type=model_pb2.JAGGED_SEQUENCE,
        )
    ]


def _train_stub(max_total_len):
    """A ``_predict_train``-ready stub; returns ``(model, spliced T per step)``."""
    m = _stub()
    m._is_inference = False  # not inference + nn.Module.training=True -> is_train
    m._label_name = "label"
    m._slot_names = ["user_sequence"]
    m._max_total_len = max_total_len
    m._pool_warmed = False
    seen_lens = []

    def fwd(input_ids, labels, attention_mask):
        seen_lens.append(input_ids.shape[1])
        return {"loss": torch.tensor(0.0)}

    m.build_input = lambda b: {
        "user_sequence": [torch.tensor([100, 101, 102])],
        m._label_name: [torch.tensor([200, 201, 202])],
    }
    m._forward_loss = fwd
    return m, seen_lens


class GenerativeQwenTest(unittest.TestCase):
    def test_splice_layout_and_labels(self) -> None:
        m = _stub()
        u = [torch.tensor([100, 101, 102])]
        a = [torch.tensor([200, 201, 202])]  # 3 codes = num_levels
        ids, labels, mask = m._splice_input_ids([u], a)
        # head | history | tail | answer | asst_suffix | eos
        self.assertEqual(
            ids[0].tolist(), [10, 11, 12, 100, 101, 102, 13, 14, 200, 201, 202, 15, 9]
        )
        # only the answer (cols 8-10) and the trailing eos (col 12) are supervised
        self.assertEqual(
            labels[0].tolist(),
            [-100] * 8 + [200, 201, 202, -100, 9],
        )
        self.assertEqual(mask[0].tolist(), [1] * 13)

    def test_left_padding_varied_lengths(self) -> None:
        m = _stub()
        u = [torch.tensor([100, 101, 102, 103]), torch.tensor([100])]
        a = [torch.tensor([200, 201, 202]), torch.tensor([207, 208, 209])]
        ids, labels, mask = m._splice_input_ids([u], a)
        T = ids.shape[1]
        n1 = 2 + 1 + 1 + 1 + 1 + 3 + 1 + 1  # shorter row's real length
        self.assertEqual(ids[1, : T - n1].tolist(), [m._pad_token_id] * (T - n1))
        self.assertEqual(mask[1].tolist(), [0] * (T - n1) + [1] * n1)
        self.assertEqual(labels[1, : T - n1].tolist(), [-100] * (T - n1))
        # the trailing eos is supervised in every row
        self.assertEqual(labels[:, -1].tolist(), [9, 9])

    def test_mask_keeps_trailing_eos_when_pad_equals_eos(self) -> None:
        m = _stub(pad_id=9)  # tpl_eos == 9 too
        ids, _, mask = m._splice_input_ids(
            [[torch.tensor([100])]], [torch.tensor([200, 201, 202])]
        )
        self.assertEqual(int(ids[0, -1]), 9)
        self.assertEqual(int(mask[0, -1]), 1)
        self.assertEqual(mask[0].tolist(), [1] * ids.shape[1])

    def test_predict_routes_on_inference_flag(self) -> None:
        m = _stub()
        m._predict_train = lambda b: {"branch": "train"}
        m._is_inference = False  # train / eval
        self.assertEqual(GenerativeQwen.predict(m, object())["branch"], "train")
        m._is_inference = True  # inference
        sentinel = torch.zeros(1, 2, 3)
        with mock.patch(
            "tzrec.models.generative_qwen._fx_wrapped_generate", return_value=sentinel
        ):
            out = GenerativeQwen.predict(m, object())
        self.assertIs(out["generated_sids"], sentinel)

    def test_predict_survives_fx_tracing(self) -> None:
        """TorchRec's predict pipeline FX-traces the model before running it.

        The decode is un-traceable by construction (per-row python lists,
        data-dependent beam widths), so it sits behind one ``torch.fx.wrap``
        leaf. Without it ``tzrec.predict`` dies inside TorchRec's
        ``_rewrite_model`` with "Proxy object cannot be iterated" -- a failure no
        unit test saw because train and eval take the un-traced
        ``TrainPipelineBase`` path (this model has no ShardedModule).
        """
        m = _stub(base_vocab=100)
        m._is_inference = True

        class _Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, batch):
                return self.inner.predict(batch)

        gm = torch.fx.symbolic_trace(_Wrapper(m))
        leaves = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and "generate" in str(n.target)
        ]
        self.assertEqual(len(leaves), 1, f"expected one opaque decode node: {gm.graph}")

    # (generated tail -> decoded SIDs). sizes [2,3,4] -> offsets [0,2,5], so
    # [100,102,105] / [101,104,108] are each level's min/max token and local
    # codes [0,0,0] / [1,2,3]; every malformed row collapses to -1.
    @parameterized.expand(
        [
            [[[100, 102, 105], [101, 104, 108]], [[0, 0, 0], [1, 2, 3]]],
            [
                [
                    [100, 102, 105],  # valid -> local [0, 0, 0]
                    [101, 104, 108],  # valid -> local [1, 2, 3]
                    [102, 102, 105],  # pos0 above level-0 band
                    [100, 101, 105],  # pos1 below level-1 band
                    [100, 102, 109],  # pos2 above level-2 band
                    [100, 104, 9],  # pos2 = eos/pad token (sid -90) -> invalid
                ],
                [[0, 0, 0], [1, 2, 3]] + [[-1, -1, -1]] * 4,
            ],
            # early EOS: a tail narrower than num_levels still reshapes cleanly,
            # and the missing 3rd atom stays -1 -> out of band -> candidate -1
            [[[100, 102], [101, 104]], [[-1, -1, -1], [-1, -1, -1]]],
        ],
        name_func=parameterized_name_func,
    )
    def test_generate_maps_tokens_to_sids(self, tail, expected) -> None:
        m = _stub(base_vocab=100)
        m._slot_names = ["user_sequence"]
        m._num_return = len(tail)
        m._beam_widths = [len(tail)] * m._num_levels
        # build_input is mocked, so the batch is opaque to this decoding test.
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 102, 105])]}
        with mock.patch(
            "tzrec.models.generative_qwen.dynamic_beam_search",
            return_value=torch.tensor(tail),
        ):
            sids = m._generate(object())
        # (B, num_return, num_levels)
        self.assertEqual(tuple(sids.shape), (1, len(tail), 3))
        self.assertEqual(sids[0].tolist(), expected)

    def test_generate_trims_to_num_return_keeping_the_best(self) -> None:
        # the kernel returns score-ordered best-first, so the trim is a prefix
        m = _stub(base_vocab=100)
        m._slot_names = ["user_sequence"]
        m._beam_widths = [4] * m._num_levels
        m._num_return = 2
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 102, 105])]}
        four = torch.tensor(
            [[100, 102, 105], [101, 104, 108], [100, 103, 106], [101, 102, 107]]
        )
        with mock.patch(
            "tzrec.models.generative_qwen.dynamic_beam_search", return_value=four
        ):
            sids = m._generate(object())
        self.assertEqual(tuple(sids.shape), (1, 2, 3))
        self.assertEqual(sids[0].tolist(), [[0, 0, 0], [1, 2, 3]])

    def test_generate_hands_the_kernel_prompt_bands_and_schedule(self) -> None:
        m = _stub(base_vocab=100)
        m._slot_names = ["user_sequence"]
        m._beam_widths = [5 * 2 ** (j + 1) for j in range(m._num_levels)]
        m._num_return = m._beam_widths[-1]
        seen = {}

        def fake_kernel(lm, input_ids, attention_mask, *, beam_widths, lo_tok, hi_tok):
            seen.update(
                lm=lm,
                ids=input_ids[0].tolist(),
                mask=attention_mask[0].tolist(),
                widths=list(beam_widths),
                lo=lo_tok.tolist(),
                hi=hi_tok.tolist(),
            )
            return torch.tensor([[100, 102, 105], [101, 104, 108]])

        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 102, 105])]}
        with mock.patch(
            "tzrec.models.generative_qwen.dynamic_beam_search", side_effect=fake_kernel
        ):
            sids = m._generate(object())
        self.assertIs(seen["lm"], m.lm)
        self.assertEqual(seen["ids"], [10, 11, 12, 100, 102, 105, 13, 14])
        self.assertEqual(seen["mask"], [1] * 8)
        # the schedule reaches the kernel verbatim
        self.assertEqual(seen["widths"], [10, 20, 40])
        # per-level bands, base_vocab-shifted: sizes [2,3,4] -> offsets [0,2,5]
        self.assertEqual(seen["lo"], [100, 102, 105])
        self.assertEqual(seen["hi"], [101, 104, 108])
        self.assertEqual(tuple(sids.shape), (1, 2, 3))
        self.assertEqual(sids[0].tolist(), [[0, 0, 0], [1, 2, 3]])

    def _prompt_tokenizer(self):
        # encode -> [len(text)] makes each buffer a fingerprint of its fragment
        return types.SimpleNamespace(
            eos_token_id=99,
            encode=lambda text, add_special_tokens=False: [len(text)],
        )

    def test_build_prompt_tokens_splits_the_template_around_the_slot(self) -> None:
        m = object.__new__(GenerativeQwen)
        nn.Module.__init__(m)
        _wire_slots(m, prefix_text="PRE", suffix_text="SUF")
        cfg = types.SimpleNamespace(prompt_template="A{{user_sequence}}B")
        m._build_prompt_tokens(self._prompt_tokenizer(), cfg)
        for name in ["tpl_gap_0", "tpl_gap_1", "tpl_asst_suffix", "tpl_eos"]:
            buf = getattr(m, name)
            self.assertIsInstance(buf, torch.Tensor)
            self.assertEqual(buf.dtype, torch.int64)
        tpl = GenerativeQwen.CHAT_TEMPLATE
        # head = user_prefix + before + feature.prefix_text
        self.assertEqual(m.tpl_gap_0.tolist(), [len(tpl["user_prefix"] + "A" + "PRE")])
        # tail = feature.suffix_text + after + user_suffix + asst_prefix
        self.assertEqual(
            m.tpl_gap_1.tolist(),
            [len("SUF" + "B" + tpl["user_suffix"] + tpl["asst_prefix"])],
        )
        self.assertEqual(m.tpl_eos.tolist(), [99])  # eos cached for supervision

    def test_compute_max_total_length(self) -> None:
        m = _stub()
        # frame = 3 head + 2 tail + 1 asst_suffix + 1 eos = 7
        m._max_seq_length = 300
        self.assertEqual(m._compute_max_total_length(), 7 + 300 + 3)
        m._max_seq_length = 0  # pre-allocation disabled
        self.assertEqual(m._compute_max_total_length(), 0)

    def test_first_step_pads_to_max_then_actual_length(self) -> None:
        m, seen_lens = _train_stub(max_total_len=50)
        m._predict_train(object())  # first step: pre-size to worst case
        m._predict_train(object())  # subsequent step: natural length
        self.assertEqual(seen_lens[0], 50)
        self.assertLess(seen_lens[1], 50)
        self.assertTrue(m._pool_warmed)

    def test_no_forced_padding_when_disabled(self) -> None:
        m, seen_lens = _train_stub(max_total_len=0)  # pre-allocation off
        m._predict_train(object())
        self.assertLess(seen_lens[0], 50)
        self.assertFalse(m._pool_warmed)

    def test_splice_row_count_mismatch_raises(self) -> None:
        m = _stub()
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            m._splice_input_ids([[torch.tensor([100])]], [])


class GenerativeQwenLossTest(unittest.TestCase):
    """The training objective, run for real against a tiny Qwen backbone."""

    def _model(self, ignore_index=-100):
        m = _real_lm_stub(codebook=[2, 3, 4], base_vocab=20, beam_width=2)
        m._ignore_index = ignore_index
        m._pad_token_id = 0
        for name, vals in {
            "tpl_gap_0": [1, 2, 3],
            "tpl_gap_1": [4, 5],
            "tpl_asst_suffix": [6],
            "tpl_eos": [7],
        }.items():
            m.register_buffer(
                name, torch.tensor(vals, dtype=torch.long), persistent=False
            )
            m._slot_names = ["user_sequence"]
        m._suffix_keep = 6  # num_levels 3 + asst_suffix 1 + trailing eos + HF shift
        return m

    def _rows(self):
        # ragged histories so left padding is exercised
        u = [torch.tensor([20, 22, 25]), torch.tensor([21, 23, 26, 20, 24, 27])]
        a = [torch.tensor([20, 22, 25]), torch.tensor([21, 24, 28])]
        return [u], a

    def test_suffix_slice_matches_full_sequence_loss(self) -> None:
        # the fixed-width suffix slice must give the same CE as full-T logits
        m = self._model()
        ids, labels, mask = m._splice_input_ids(*self._rows())
        with torch.no_grad():
            got = m._forward_loss(ids, labels, mask)["loss"]
            full = m.lm(input_ids=ids, attention_mask=mask).logits
            ref = m.lm.loss_function(
                logits=full, labels=labels, vocab_size=m.lm.config.vocab_size
            )
        self.assertTrue(torch.allclose(got, ref, atol=1e-6))

    def test_loss_is_invariant_to_extra_left_padding(self) -> None:
        # the pool-warmup pad_to must not perturb the objective
        m = self._model()
        u, a = self._rows()
        with torch.no_grad():
            base = m._forward_loss(*m._splice_input_ids(u, a))["loss"]
            padded = m._forward_loss(*m._splice_input_ids(u, a, pad_to=40))["loss"]
        self.assertTrue(torch.allclose(base, padded, atol=1e-6))

    def test_forward_loss_honours_configured_ignore_index(self) -> None:
        # ignore_index must reach loss_function or every pad slot is supervised
        m = self._model(ignore_index=-100)
        u, a = self._rows()
        with torch.no_grad():
            default = m._forward_loss(*m._splice_input_ids(u, a))["loss"]
        m._ignore_index = -7
        with torch.no_grad():
            ids, labels, mask = m._splice_input_ids(u, a)
            custom = m._forward_loss(ids, labels, mask)["loss"]
        self.assertEqual(int((labels == -7).sum() > 0), 1)
        self.assertTrue(torch.allclose(default, custom, atol=1e-6))

    def test_forward_loss_returns_only_the_loss(self) -> None:
        # returned logits would stay alive across the next step's fwd/bwd
        m = self._model()
        with torch.no_grad():
            out = m._forward_loss(*m._splice_input_ids(*self._rows()))
        self.assertEqual(list(out), ["loss"])


if __name__ == "__main__":
    unittest.main()
