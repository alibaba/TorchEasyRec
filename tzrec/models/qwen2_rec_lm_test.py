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


def _stub(num_levels=3, base_vocab=100, pad_id=9, device="cpu", per_level=4):
    """A Qwen2RecLM with the splice-relevant state wired up, no HF backbone.

    Template buffers use tiny placeholder ids so the spliced layout is easy to
    read; real buffers come from ``_build_prompt_tokens`` at init time.

    ``per_level`` sets the (uniform) codebook size used to derive the per-level
    SID validity bands the inference gate checks — real bands come from
    ``_read_common_config``. With ``per_level=4`` / ``num_levels=3`` the bands
    are lo=[1,5,9], hi=[4,8,12] (level j -> sid in [j*4+1, (j+1)*4]).
    """
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._ignore_index = -100
    m._num_levels = num_levels
    m._base_vocab = base_vocab
    m._pad_token_id = pad_id
    m._dynamic_beam = False  # default = HF fixed-width beam path
    m._max_seq_length = 0  # no recency clip by default in unit stubs
    m._generated_sids_key = "generated_sids"  # configurable; default key
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
    lo, hi = Qwen2RecLM._sid_level_bands([per_level] * num_levels)
    m.register_buffer("_sid_lvl_lo", lo, persistent=False)
    m.register_buffer("_sid_lvl_hi", hi, persistent=False)
    return m


def _gen_batch():
    """A one-row inference batch (history SIDs [1, 2, 3]) for ``_generate`` tests."""
    return types.SimpleNamespace(
        sequence_dense_features={"user_sequence": _FakeJT([1, 2, 3], [3])}
    )


def _first_non_neg_index(labels):
    """The per-step suffix bound _forward_loss's cached _suffix_keep replaces."""
    tmp = (labels >= 0).cumsum(dim=-1)
    return int((tmp == 1).float().argmax(dim=-1).min().item())


class Qwen2RecLMTest(unittest.TestCase):
    def test_splice_layout_and_labels(self) -> None:
        m = _stub()
        u = [torch.tensor([100, 101, 102])]
        a = [torch.tensor([200, 201, 202])]  # 3 codes = num_levels
        ids, labels, mask = m._splice_input_ids(u, a)
        # [system | user_prefix | history | user_suffix |
        #  asst_prefix | answer | asst_suffix | eos]
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
        # shorter row is left-padded: pad at the front, content right-aligned
        self.assertEqual(ids[1, : T - n1].tolist(), [m._pad_token_id] * (T - n1))
        self.assertEqual(mask[1].tolist(), [0] * (T - n1) + [1] * n1)
        self.assertEqual(labels[1, : T - n1].tolist(), [-100] * (T - n1))
        # every row's trailing eos is supervised and the answer ends just before
        self.assertEqual(labels[:, -1].tolist(), [9, 9])

    def test_mask_keeps_trailing_eos_when_pad_equals_eos(self) -> None:
        # pad_id == eos value: the mask must NOT mask the real trailing eos
        m = _stub(pad_id=9)  # tpl_eos == 9 too
        ids, _, mask = m._splice_input_ids(
            [torch.tensor([100])], [torch.tensor([200, 201, 202])]
        )
        self.assertEqual(int(ids[0, -1]), 9)
        self.assertEqual(int(mask[0, -1]), 1)
        self.assertEqual(mask[0].tolist(), [1] * ids.shape[1])

    def test_suffix_keep_matches_dynamic_slice(self) -> None:
        # _forward_loss caches self._suffix_keep instead of recomputing the suffix
        # bound per step (two GPU->CPU syncs); prove the constant equals the old
        # dynamic computation and that the slice drops nothing supervised.
        m = _stub()  # num_levels=3, asst_suffix=[15] (numel 1) -> suffix_keep=6
        suffix_keep = m._num_levels + m.tpl_asst_suffix.numel() + 2
        _, labels, _ = m._splice_input_ids(
            [torch.tensor([100, 101, 102])], [torch.tensor([200, 201, 202])]
        )
        # the constant matches the per-step bound it replaces
        self.assertEqual(
            suffix_keep, labels.shape[1] - _first_non_neg_index(labels) + 1
        )
        # everything before the kept suffix is unsupervised (-100): nothing dropped
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
        m._is_inference = True  # inference (set_is_inference in main.py)
        self.assertEqual(Qwen2RecLM.predict(m, object())["branch"], "generate")

    def test_generate_maps_tokens_to_sids(self) -> None:
        m = _stub(base_vocab=100)  # sid = token - base + 1 = token - 99
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
            # 2 beams x 3 codes, every atom INSIDE its level's band
            # (bands lo=[1,5,9] hi=[4,8,12]; token = sid + 99):
            #   pos0 token in [100,103], pos1 in [104,107], pos2 in [108,111].
            new = torch.tensor([[100, 104, 108], [103, 107, 111]])
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate
        # build_input (mocked) supplies the tokenized history rows; the batch is
        # opaque to it. SIDs [1,2,3] tokenize to [100,101,102] at base_vocab=100.
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 101, 102])]}
        sids = m._generate(_gen_batch())["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 2, 3))  # (B, num_return, num_levels)
        self.assertEqual(sids[0].tolist(), [[1, 5, 9], [4, 8, 12]])

    def test_generate_rejects_malformed_candidates(self) -> None:
        # Layer-A gate: every malformed candidate -> the -1 sentinel, in place.
        # bands lo=[1,5,9] hi=[4,8,12]; token = sid + 99.
        m = _stub(base_vocab=100)
        m._input_name = "user_sequence"
        m._num_beams = m._num_return = 4

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
                    [100, 104, 108],  # all in-band -> valid -> [1, 5, 9]
                    [100, 104, 9],  # pos2 = eos/pad token (sid -90) -> invalid
                    [
                        108,
                        104,
                        100,
                    ],  # wrong-level scramble (pos0 = lvl-2 code) -> invalid
                    [100, 104, 112],  # pos2 sid 13 > band hi 12 -> invalid
                ]
            )
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate
        # build_input (mocked) supplies the tokenized history rows; the batch is
        # opaque to it. SIDs [1,2,3] tokenize to [100,101,102] at base_vocab=100.
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 101, 102])]}
        sids = m._generate(_gen_batch())["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 4, 3))
        # valid candidate kept at its rank; every malformed one -> all -1 (in place)
        self.assertEqual(
            sids[0].tolist(),
            [[1, 5, 9], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        )

    def test_generate_narrow_tail_no_crash(self) -> None:
        # every beam emits EOS before num_levels -> generate() returns a tail
        # narrower than num_levels; the canvas keeps the reshape rectangular.
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
            new = torch.tensor([[100, 104], [103, 107]])  # width 2 < num_levels 3
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate
        # build_input (mocked) supplies the tokenized history rows; the batch is
        # opaque to it. SIDs [1,2,3] tokenize to [100,101,102] at base_vocab=100.
        m.build_input = lambda b: {"user_sequence": [torch.tensor([100, 101, 102])]}
        sids = m._generate(_gen_batch())["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 2, 3))  # rectangular, no crash
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

    def test_input_sequence_length_from_feature(self) -> None:
        m = object.__new__(Qwen2RecLM)
        m._input_name = "user_sequence"
        f_user = types.SimpleNamespace(
            config=types.SimpleNamespace(feature_name="user_sequence"),
            sequence_length=300,
        )
        f_label = types.SimpleNamespace(
            config=types.SimpleNamespace(feature_name="label"), sequence_length=32
        )
        m._features = [f_label, f_user]
        self.assertEqual(m._input_sequence_length(), 300)  # truncation length
        m._features = [f_label]  # user-sequence feature absent
        self.assertEqual(m._input_sequence_length(), 0)
        f_user.sequence_length = None  # no length cap -> pre-allocation disabled
        m._features = [f_user]
        self.assertEqual(m._input_sequence_length(), 0)

    def test_sid_token_rows_recency_clip(self) -> None:
        m = _stub(num_levels=3, base_vocab=100)  # token = sid + base - 1 = sid + 99

        def _vl(n):  # one row of n codes (values 1..n) as flat (values, lengths)
            return torch.arange(1, n + 1, dtype=torch.float), torch.tensor([n])

        # 15 codes (5 items), cap 9 -> keep last 9 (items 3-5 = codes 7..15)
        rows = m._sid_token_rows(*_vl(15), max_codes=9)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(7, 16)])
        # item-aligned: cap 10 still keeps 9 (3 whole items), never cuts mid-item
        rows = m._sid_token_rows(*_vl(15), max_codes=10)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(7, 16)])
        # within cap -> untouched
        rows = m._sid_token_rows(*_vl(6), max_codes=9)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(1, 7)])
        # disabled (0/None) -> no clip
        rows = m._sid_token_rows(*_vl(15), max_codes=0)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(1, 16)])

    def test_compute_max_total_length(self) -> None:
        m = _stub(num_levels=3)
        # frame = |system|2 + |user_prefix|1 + |user_suffix|1 + |asst_prefix|1
        #         + |asst_suffix|1 + |eos|1 = 7; + max_history + answer(num_levels)
        m._max_seq_length = 300
        self.assertEqual(m._compute_max_total_length(), 7 + 300 + 3)
        m._max_seq_length = 0  # pre-allocation disabled
        self.assertEqual(m._compute_max_total_length(), 0)

    def test_first_step_pads_to_max_then_actual_length(self) -> None:
        m = _stub()
        m._is_inference = False  # not inference + nn.Module.training=True -> is_train
        m._smoke_log_once = False
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
        # one-shot: first step left-pads to _max_total_len, latched by
        # _pool_warmed; later steps use the actual (shorter) length.
        self.assertEqual(seen_lens[0], 50)
        self.assertLess(seen_lens[1], 50)
        self.assertTrue(m._pool_warmed)

    def test_no_forced_padding_when_disabled(self) -> None:
        m = _stub()
        m._is_inference = False
        m._smoke_log_once = False
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
        # disabled: natural length, never forced to max; flag stays unlatched.
        self.assertLess(seen_lens[0], 50)
        self.assertFalse(m._pool_warmed)


def _real_lm_stub(num_levels=3, base_vocab=20, per_level=4, num_beams=2):
    """A Qwen2RecLM carrying a real (tiny, random) Qwen2 backbone.

    Needed by the dynamic-beam tests, which exercise the actual KV-cached
    forward / cache-reorder path (the other tests mock ``lm.generate``). The SID
    atoms occupy the last ``num_levels * per_level`` token ids, matching the
    base-vocab + appended-codebook layout.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._num_levels = num_levels
    m._base_vocab = base_vocab
    m._num_beams = num_beams
    cfg = Qwen2Config(
        vocab_size=base_vocab + per_level * num_levels,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    m.lm = Qwen2ForCausalLM(cfg).eval()
    lo, hi = Qwen2RecLM._sid_level_bands([per_level] * num_levels)
    m.register_buffer("_sid_lvl_lo", lo, persistent=False)
    m.register_buffer("_sid_lvl_hi", hi, persistent=False)
    return m


class Qwen2DynamicBeamTest(unittest.TestCase):
    def test_width_schedule_and_final_count(self) -> None:
        # widths double per level, returning num_beams * 2**num_levels candidates.
        m = _real_lm_stub(num_levels=3, base_vocab=20, per_level=8, num_beams=2)
        ids = torch.tensor([[1, 2, 3, 4]])
        new = m._dynamic_beam_search(ids, torch.ones_like(ids))
        # base 2: widths [4, 8, 16] (none capped: per_level 8 is roomy) -> 16 final
        self.assertEqual(tuple(new.shape), (2 * 2**3, 3))

    def test_every_candidate_is_in_band(self) -> None:
        # band masking guarantees well-formed SIDs: validate -> no -1 sentinels.
        m = _real_lm_stub(num_levels=3, base_vocab=20, per_level=4, num_beams=2)
        ids = torch.tensor([[5, 6, 7]])
        new = m._dynamic_beam_search(ids, torch.ones_like(ids))
        sids = m._validate_sid_candidates(new, batch_size=1)
        self.assertEqual(tuple(sids.shape), (1, new.shape[0], 3))
        self.assertFalse(bool((sids < 0).any()))  # every candidate well-formed

    def test_left_padding_two_rows(self) -> None:
        # ragged batch (row 1 left-padded): both rows yield valid, full beam sets.
        m = _real_lm_stub(num_levels=2, base_vocab=20, per_level=4, num_beams=2)
        ids = torch.tensor([[5, 6, 7, 8], [0, 0, 9, 10]])
        am = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        new = m._dynamic_beam_search(ids, am)
        self.assertEqual(tuple(new.shape), (2 * 2 * 2**2, 2))  # B=2, 2*2^2=8 each
        sids = m._validate_sid_candidates(new, batch_size=2)
        self.assertEqual(tuple(sids.shape), (2, 8, 2))
        self.assertFalse(bool((sids < 0).any()))

    def test_exhaustive_matches_bruteforce_topk(self) -> None:
        # When the schedule covers the whole tree (no pruning) the escalating
        # beam is EXACT: its candidate set must equal all SID combos and be
        # ordered by true (full-recompute) cumulative log-prob. This validates
        # cache-stepping, band masking, scoring, and ordering end-to-end.
        per, base = 3, 20
        m = _real_lm_stub(num_levels=2, base_vocab=base, per_level=per, num_beams=3)
        # widths [min(6,3)=3, min(12,9)=9] -> exhaustive over all 3*3=9 SIDs
        ids = torch.tensor([[5, 6, 7, 8]])
        am = torch.ones_like(ids)
        got = [tuple(r) for r in m._dynamic_beam_search(ids, am).tolist()]
        self.assertEqual(len(got), per * per)
        lm = m.lm
        lo0, lo1 = base, base + per  # level token bands [20,22] and [23,25]
        ref = {}
        with torch.no_grad():
            logp0 = torch.log_softmax(
                lm(ids, attention_mask=am).logits[0, -1].float(), -1
            )
            for t0 in range(lo0, lo0 + per):
                s2 = torch.cat([ids, torch.tensor([[t0]])], 1)
                logp1 = torch.log_softmax(
                    lm(s2, attention_mask=torch.ones_like(s2)).logits[0, -1].float(), -1
                )
                for t1 in range(lo1, lo1 + per):
                    ref[(t0, t1)] = (logp0[t0] + logp1[t1]).item()
        # 1. exhaustive: the returned set is exactly every SID combination
        self.assertEqual(set(got), set(ref))
        # 2. ordered best-first by the true score (tolerant of float-noise ties)
        s = [ref[c] for c in got]
        self.assertTrue(all(s[i] >= s[i + 1] - 1e-4 for i in range(len(s) - 1)))
        self.assertEqual(got[0], max(ref, key=ref.get))  # top-1 is the global best


if __name__ == "__main__":
    unittest.main()
