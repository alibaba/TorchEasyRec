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

from tzrec.models.generative_rec_lm_test import _FakeJT
from tzrec.models.qwen2_rec_lm import Qwen2RecLM


def _stub(codebook=None, base_vocab=100, pad_id=9, device="cpu"):
    """A Qwen2RecLM with the splice-relevant state wired up, no HF backbone.

    The non-uniform default codebook makes incorrect ``level * uniform_size``
    offset arithmetic visible: sizes=[2,3,4], offsets=[0,2,5].
    """
    codebook = codebook or [2, 3, 4]
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._ignore_index = -100
    m._num_levels = len(codebook)
    m._base_vocab = base_vocab
    m._pad_token_id = pad_id
    m._dynamic_beam = False
    m._max_seq_length = 0
    m._generated_sids_key = "generated_sids"
    m.lm = types.SimpleNamespace(device=torch.device(device))
    for name, vals in {
        "tpl_system": [10, 11],
        "tpl_user_prefix": [12],
        "tpl_user_suffix": [13],
        "tpl_asst_prefix": [14],
        "tpl_asst_suffix": [15],
        "tpl_eos": [9],
    }.items():
        m.register_buffer(name, torch.tensor(vals, dtype=torch.long), persistent=False)
    offsets, sizes = Qwen2RecLM._sid_level_layout(codebook)
    m.register_buffer("_level_offsets", offsets, persistent=False)
    m.register_buffer("_codebook_sizes", sizes, persistent=False)
    return m


def _gen_batch():
    """An opaque one-row batch; the ``_generate`` tests mock ``build_input``."""
    return types.SimpleNamespace(
        sequence_dense_features={"user_sequence": _FakeJT([1, 2, 3], [3])}
    )


def _first_non_neg_index(labels):
    """First supervised label column, minimized over rows."""
    tmp = (labels >= 0).cumsum(dim=-1)
    return int((tmp == 1).float().argmax(dim=-1).min().item())


class Qwen2RecLMTest(unittest.TestCase):
    def test_splice_layout_and_labels(self) -> None:
        m = _stub()
        u = [torch.tensor([100, 101, 102])]
        a = [torch.tensor([200, 201, 202])]  # 3 codes = num_levels
        ids, labels, mask = m._splice_input_ids(u, a)
        # sys|user_prefix|history|user_suffix|asst_prefix|answer|asst_suffix|eos
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
        ids, labels, mask = m._splice_input_ids(u, a)
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
            [torch.tensor([100])], [torch.tensor([200, 201, 202])]
        )
        self.assertEqual(int(ids[0, -1]), 9)
        self.assertEqual(int(mask[0, -1]), 1)
        self.assertEqual(mask[0].tolist(), [1] * ids.shape[1])

    def test_suffix_keep_matches_dynamic_slice(self) -> None:
        # the constant suffix width must cover every supervised column.
        m = _stub()  # num_levels=3, asst_suffix=[15] (numel 1) -> suffix_keep=6
        suffix_keep = m._num_levels + m.tpl_asst_suffix.numel() + 2
        _, labels, _ = m._splice_input_ids(
            [torch.tensor([100, 101, 102])], [torch.tensor([200, 201, 202])]
        )
        self.assertEqual(
            suffix_keep, labels.shape[1] - _first_non_neg_index(labels) + 1
        )
        self.assertTrue(bool((labels[:, :-suffix_keep] < 0).all()))

    def test_splice_prompt_ids(self) -> None:
        m = _stub()
        ids, mask = m._splice_prompt_ids([torch.tensor([100, 101, 102])])
        # [system | user_prefix | history | user_suffix | asst_prefix], no answer
        self.assertEqual(ids[0].tolist(), [10, 11, 12, 100, 101, 102, 13, 14])
        self.assertEqual(mask[0].tolist(), [1] * 8)

    def test_predict_routes_on_inference_flag(self) -> None:
        m = _stub()
        m._predict_train = lambda b: {"branch": "train"}
        m._generate = lambda b: {"branch": "generate"}
        m._is_inference = False  # train / eval
        self.assertEqual(Qwen2RecLM.predict(m, object())["branch"], "train")
        m._is_inference = True  # inference
        self.assertEqual(Qwen2RecLM.predict(m, object())["branch"], "generate")

    def test_generate_maps_tokens_to_sids(self) -> None:
        m = _stub(base_vocab=100)
        m._input_name = "user_sequence"
        m._num_beams = m._num_return = 2

        def fake_generate(
            input_ids,
            attention_mask,
            max_new_tokens,
            num_beams,
            num_return_sequences,
            do_sample,
            pad_token_id,
        ):
            prompt = input_ids.repeat_interleave(num_return_sequences, dim=0)
            # offsets [0,2,5]: local [1,1,1]/[2,3,4] -> each level's min/max token
            new = torch.tensor([[100, 102, 105], [101, 104, 108]])
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate
        # build_input is mocked, so the batch is opaque to this generation test.
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 102, 105])]}
        sids = m._generate(_gen_batch())["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 2, 3))  # (B, num_return, num_levels)
        self.assertEqual(sids[0].tolist(), [[1, 1, 1], [2, 3, 4]])

    def test_generate_rejects_malformed_candidates(self) -> None:
        # every malformed candidate collapses to the -1 sentinel, in place.
        m = _stub(base_vocab=100)
        m._input_name = "user_sequence"
        m._num_beams = m._num_return = 6

        def fake_generate(
            input_ids,
            attention_mask,
            max_new_tokens,
            num_beams,
            num_return_sequences,
            do_sample,
            pad_token_id,
        ):
            prompt = input_ids.repeat_interleave(num_return_sequences, dim=0)
            new = torch.tensor(
                [
                    [100, 102, 105],  # valid -> local [1, 1, 1]
                    [101, 104, 108],  # valid -> local [2, 3, 4]
                    [102, 102, 105],  # pos0 above level-0 band
                    [100, 101, 105],  # pos1 below level-1 band
                    [100, 102, 109],  # pos2 above level-2 band
                    [100, 104, 9],  # pos2 = eos/pad token (sid -90) -> invalid
                ]
            )
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 102, 105])]}
        sids = m._generate(_gen_batch())["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 6, 3))
        self.assertEqual(
            sids[0].tolist(),
            [
                [1, 1, 1],
                [2, 3, 4],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
        )

    def test_generate_narrow_tail_no_crash(self) -> None:
        # early EOS -> a tail narrower than num_levels must still reshape cleanly
        m = _stub(base_vocab=100)
        m._input_name = "user_sequence"
        m._num_beams = m._num_return = 2

        def fake_generate(
            input_ids,
            attention_mask,
            max_new_tokens,
            num_beams,
            num_return_sequences,
            do_sample,
            pad_token_id,
        ):
            prompt = input_ids.repeat_interleave(num_return_sequences, dim=0)
            new = torch.tensor([[100, 102], [101, 104]])  # width 2 < num_levels 3
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 102, 105])]}
        sids = m._generate(_gen_batch())["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 2, 3))
        # the missing 3rd atom stays -1 -> out of band -> whole candidate -1
        self.assertEqual(sids[0].tolist(), [[-1, -1, -1], [-1, -1, -1]])

    def test_build_prompt_tokens_registers_buffers(self) -> None:
        m = object.__new__(Qwen2RecLM)
        nn.Module.__init__(m)
        tok = types.SimpleNamespace(
            eos_token_id=99,
            encode=lambda text, add_special_tokens=False: [len(text)],
        )
        cfg = types.SimpleNamespace(
            system_instruction="", user_prefix_text="", user_suffix_text=""
        )
        m._build_prompt_tokens(tok, cfg)
        for name in [
            "tpl_system",
            "tpl_user_prefix",
            "tpl_user_suffix",
            "tpl_asst_prefix",
            "tpl_asst_suffix",
            "tpl_eos",
        ]:
            buf = getattr(m, name)
            self.assertIsInstance(buf, torch.Tensor)
            self.assertEqual(buf.dtype, torch.int64)
        self.assertEqual(m.tpl_eos.tolist(), [99])  # eos cached for supervision

    def test_sid_token_rows_recency_clip(self) -> None:
        m = _stub(base_vocab=100)
        values = torch.tensor(
            [
                1,
                1,
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                2,
                1,
                4,
                1,
                3,
                2,
            ],
            dtype=torch.float,
        )

        def _vl():
            return values, torch.tensor([values.numel()])

        # 15 codes (5 items), cap 9 -> keep the most recent three whole items.
        expected_tail = [100, 103, 107, 101, 102, 108, 100, 104, 106]
        rows = m._sid_token_rows(*_vl(), max_codes=9)
        self.assertEqual(rows[0].tolist(), expected_tail)
        # item-aligned: cap 10 still keeps 9 (3 whole items), never cuts mid-item
        rows = m._sid_token_rows(*_vl(), max_codes=10)
        self.assertEqual(rows[0].tolist(), expected_tail)
        # within cap -> untouched
        rows = m._sid_token_rows(values[:6], torch.tensor([6]), max_codes=9)
        self.assertEqual(rows[0].tolist(), [100, 102, 105, 101, 104, 108])
        # disabled (0/None) -> no clip
        rows = m._sid_token_rows(*_vl(), max_codes=0)
        self.assertEqual(
            rows[0].tolist(),
            [
                100,
                102,
                105,
                101,
                104,
                108,
                100,
                103,
                107,
                101,
                102,
                108,
                100,
                104,
                106,
            ],
        )

    def test_compute_max_total_length(self) -> None:
        m = _stub()
        # frame = 2 system + 1 each user_pfx/sfx, asst_pfx/sfx, eos = 7
        m._max_seq_length = 300
        self.assertEqual(m._compute_max_total_length(), 7 + 300 + 3)
        m._max_seq_length = 0  # pre-allocation disabled
        self.assertEqual(m._compute_max_total_length(), 0)

    def test_first_step_pads_to_max_then_actual_length(self) -> None:
        m = _stub()
        m._is_inference = False  # not inference + nn.Module.training=True -> is_train
        m._input_name, m._label_name = "user_sequence", "label"
        m._max_total_len = 50
        m._pool_warmed = False
        seen_lens = []

        def fwd(i, lbl, a):
            seen_lens.append(i.shape[1])
            return {"loss": torch.tensor(0.0)}

        m.build_input = lambda b: {
            m._input_name: [torch.tensor([100, 101, 102])],
            m._label_name: [torch.tensor([200, 201, 202])],
        }
        m._forward_loss = fwd
        batch = object()
        m._predict_train(batch)  # first step: pre-size to worst case
        m._predict_train(batch)  # subsequent step: natural length
        self.assertEqual(seen_lens[0], 50)
        self.assertLess(seen_lens[1], 50)
        self.assertTrue(m._pool_warmed)

    def test_no_forced_padding_when_disabled(self) -> None:
        m = _stub()
        m._is_inference = False
        m._input_name, m._label_name = "user_sequence", "label"
        m._max_total_len = 0  # max_seq_length unset -> pre-allocation off
        m._pool_warmed = False
        seen_lens = []

        def fwd(i, lbl, a):
            seen_lens.append(i.shape[1])
            return {"loss": torch.tensor(0.0)}

        m.build_input = lambda b: {
            m._input_name: [torch.tensor([100, 101, 102])],
            m._label_name: [torch.tensor([200, 201, 202])],
        }
        m._forward_loss = fwd
        batch = object()
        m._predict_train(batch)
        self.assertLess(seen_lens[0], 50)
        self.assertFalse(m._pool_warmed)

    def test_splice_row_count_mismatch_raises(self) -> None:
        m = _stub()
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            m._splice_input_ids([torch.tensor([100])], [])


class Qwen2ForwardLossTest(unittest.TestCase):
    """The training objective, run for real against a tiny Qwen2 backbone."""

    def _model(self, ignore_index=-100):
        m = _real_lm_stub(codebook=[2, 3, 4], base_vocab=20, num_beams=2)
        m._ignore_index = ignore_index
        m._pad_token_id = 0
        for name, vals in {
            "tpl_system": [1, 2],
            "tpl_user_prefix": [3],
            "tpl_user_suffix": [4],
            "tpl_asst_prefix": [5],
            "tpl_asst_suffix": [6],
            "tpl_eos": [7],
        }.items():
            m.register_buffer(
                name, torch.tensor(vals, dtype=torch.long), persistent=False
            )
        m._suffix_keep = m._num_levels + m.tpl_asst_suffix.numel() + 2
        return m

    def _rows(self):
        # ragged histories so left padding is exercised
        u = [torch.tensor([20, 22, 25]), torch.tensor([21, 23, 26, 20, 24, 27])]
        a = [torch.tensor([20, 22, 25]), torch.tensor([21, 24, 28])]
        return u, a

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


def _real_lm_stub(codebook=None, base_vocab=20, num_beams=2):
    """A Qwen2RecLM carrying a real (tiny, random) Qwen2 backbone.

    Needed by the dynamic-beam tests, which exercise the real KV-cached forward
    and cache-reorder path; the other tests mock ``lm.generate``.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    codebook = codebook or [2, 3, 4]
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._num_levels = len(codebook)
    m._base_vocab = base_vocab
    m._num_beams = num_beams
    cfg = Qwen2Config(
        vocab_size=base_vocab + sum(codebook),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    m.lm = Qwen2ForCausalLM(cfg).eval()
    offsets, sizes = Qwen2RecLM._sid_level_layout(codebook)
    m.register_buffer("_level_offsets", offsets, persistent=False)
    m.register_buffer("_codebook_sizes", sizes, persistent=False)
    return m


class Qwen2DynamicBeamTest(unittest.TestCase):
    def test_width_schedule_and_final_count(self) -> None:
        m = _real_lm_stub(codebook=[8, 7, 6], base_vocab=20, num_beams=2)
        ids = torch.tensor([[1, 2, 3, 4]])
        new = m._dynamic_beam_search(ids, torch.ones_like(ids))
        # base 2: widths [4, 8, 16], with enough combinations at each level.
        self.assertEqual(tuple(new.shape), (2 * 2**3, 3))

    def test_every_candidate_is_in_band(self) -> None:
        # band masking guarantees well-formed SIDs: validate -> no -1 sentinels.
        codebook = [2, 3, 4]
        m = _real_lm_stub(codebook=codebook, base_vocab=20, num_beams=2)
        lo, hi = m._sid_token_bands()
        self.assertEqual(lo.tolist(), [20, 22, 25])
        self.assertEqual(hi.tolist(), [21, 24, 28])
        ids = torch.tensor([[5, 6, 7]])
        new = m._dynamic_beam_search(ids, torch.ones_like(ids))
        sids = m._validate_sid_candidates(new, batch_size=1)
        self.assertEqual(tuple(sids.shape), (1, new.shape[0], 3))
        for level, size in enumerate(codebook):
            self.assertTrue(bool((sids[..., level] >= 1).all()))
            self.assertTrue(bool((sids[..., level] <= size).all()))

    def test_left_padding_two_rows(self) -> None:
        # ragged batch (row 1 left-padded): both rows yield valid, full beam sets.
        codebook = [2, 3]
        m = _real_lm_stub(codebook=codebook, base_vocab=20, num_beams=2)
        ids = torch.tensor([[5, 6, 7, 8], [0, 0, 9, 10]])
        am = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        new = m._dynamic_beam_search(ids, am)
        # The requested width is 8, but only 2*3=6 distinct pairs exist.
        self.assertEqual(tuple(new.shape), (2 * 6, 2))
        sids = m._validate_sid_candidates(new, batch_size=2)
        self.assertEqual(tuple(sids.shape), (2, 6, 2))
        for level, size in enumerate(codebook):
            self.assertTrue(bool((sids[..., level] >= 1).all()))
            self.assertTrue(bool((sids[..., level] <= size).all()))

    def test_exhaustive_matches_bruteforce_topk(self) -> None:
        # with no pruning the beam is EXACT: every SID combo, ordered by the
        # true (full-recompute) cumulative log-prob.
        codebook, base = [2, 3], 20
        m = _real_lm_stub(codebook=codebook, base_vocab=base, num_beams=3)
        # widths [min(6,2)=2, min(12,6)=6] -> exhaustive over all 2*3 SIDs
        ids = torch.tensor([[5, 6, 7, 8]])
        am = torch.ones_like(ids)
        got = [tuple(r) for r in m._dynamic_beam_search(ids, am).tolist()]
        self.assertEqual(len(got), codebook[0] * codebook[1])
        lm = m.lm
        lo0, lo1 = base, base + codebook[0]
        ref = {}
        with torch.no_grad():
            logp0 = torch.log_softmax(
                lm(ids, attention_mask=am).logits[0, -1].float(), -1
            )
            for t0 in range(lo0, lo0 + codebook[0]):
                s2 = torch.cat([ids, torch.tensor([[t0]])], 1)
                logp1 = torch.log_softmax(
                    lm(s2, attention_mask=torch.ones_like(s2)).logits[0, -1].float(), -1
                )
                for t1 in range(lo1, lo1 + codebook[1]):
                    ref[(t0, t1)] = (logp0[t0] + logp1[t1]).item()
        self.assertEqual(set(got), set(ref))
        # ordered best-first by the true score (tolerant of float-noise ties)
        s = [ref[c] for c in got]
        self.assertTrue(all(s[i] >= s[i + 1] - 1e-4 for i in range(len(s) - 1)))
        self.assertEqual(got[0], max(ref, key=ref.get))  # top-1 is the global best
        decoded = m._validate_sid_candidates(torch.tensor(got), batch_size=1)[0]
        self.assertEqual(
            {tuple(row) for row in decoded.tolist()},
            {(a, b) for a in range(1, 3) for b in range(1, 4)},
        )


if __name__ == "__main__":
    unittest.main()
