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

from tzrec.models.qwen2_rec_lm import Qwen2RecLM


def _stub(num_levels=3, base_vocab=100, pad_id=9, device="cpu"):
    """A Qwen2RecLM with the splice-relevant state wired up, no HF backbone.

    Template buffers use tiny placeholder ids so the spliced layout is easy to
    read; real buffers come from ``_build_prompt_tokens`` at init time.
    """
    m = object.__new__(Qwen2RecLM)
    nn.Module.__init__(m)
    m._ignore_index = -100
    m._num_levels = num_levels
    m._base_vocab = base_vocab
    m._pad_token_id = pad_id
    m._max_seq_length = 0  # no recency clip by default in unit stubs
    m.lm = types.SimpleNamespace(device=torch.device(device))
    for name, vals in {
        "tpl_system": [10, 11], "tpl_user_prefix": [12], "tpl_user_suffix": [13],
        "tpl_asst_prefix": [14], "tpl_asst_suffix": [15], "tpl_eos": [9],
    }.items():
        m.register_buffer(name, torch.tensor(vals, dtype=torch.long), persistent=False)
    return m


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

    def test_min_first_non_neg_index(self) -> None:
        labels = torch.tensor([[-100, -100, 5, 6], [-100, 7, 8, 9]])
        self.assertEqual(Qwen2RecLM._min_first_non_neg_index(labels), 1)

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

        def fake_generate(input_ids, attention_mask, max_new_tokens,
                          num_beams, num_return_sequences, do_sample, pad_token_id):
            prompt = input_ids.repeat_interleave(num_return_sequences, dim=0)
            new = torch.tensor([[200, 201, 202], [203, 204, 205]])  # 2 beams x 3 codes
            return torch.cat([prompt, new], dim=1)

        m.lm.generate = fake_generate

        class _JT:
            def values(self):
                return torch.tensor([1, 2, 3], dtype=torch.float)

            def lengths(self):
                return torch.tensor([3])

        batch = types.SimpleNamespace(sequence_dense_features={"user_sequence": _JT()})
        sids = m._generate(batch)["generated_sids"]
        self.assertEqual(tuple(sids.shape), (1, 2, 3))  # (B, num_return, num_levels)
        self.assertEqual(sids[0].tolist(), [[101, 102, 103], [104, 105, 106]])

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
            "tpl_system", "tpl_user_prefix", "tpl_user_suffix",
            "tpl_asst_prefix", "tpl_asst_suffix", "tpl_eos",
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

        def _jt(n):  # one row of n codes: values 1..n
            return types.SimpleNamespace(
                values=lambda: torch.arange(1, n + 1, dtype=torch.float),
                lengths=lambda: torch.tensor([n]),
            )

        # 15 codes (5 items), cap 9 -> keep last 9 (items 3-5 = codes 7..15)
        rows = m._sid_token_rows(_jt(15), max_codes=9)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(7, 16)])
        # item-aligned: cap 10 still keeps 9 (3 whole items), never cuts mid-item
        rows = m._sid_token_rows(_jt(15), max_codes=10)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(7, 16)])
        # within cap -> untouched
        rows = m._sid_token_rows(_jt(6), max_codes=9)
        self.assertEqual(rows[0].tolist(), [c + 99 for c in range(1, 7)])
        # disabled (0/None) -> no clip
        rows = m._sid_token_rows(_jt(15), max_codes=0)
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

        m._sid_token_rows = lambda jt, expected_width=None, max_codes=None: [
            torch.tensor([100, 101, 102])
        ]
        m._forward_loss = fwd
        batch = types.SimpleNamespace(
            sequence_dense_features={"user_sequence": None, "label": None}
        )
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

        m._sid_token_rows = lambda jt, expected_width=None, max_codes=None: [
            torch.tensor([100, 101, 102])
        ]
        m._forward_loss = fwd
        batch = types.SimpleNamespace(
            sequence_dense_features={"user_sequence": None, "label": None}
        )
        m._predict_train(batch)
        # disabled: natural length, never forced to max; flag stays unlatched.
        self.assertLess(seen_lens[0], 50)
        self.assertFalse(m._pool_warmed)


if __name__ == "__main__":
    unittest.main()
