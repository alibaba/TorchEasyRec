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

import ast
import itertools
import pathlib
import unittest
from typing import Any, Dict, List, Tuple

import torch
from parameterized import parameterized

from tzrec.modules import escalating_beam
from tzrec.modules.escalating_beam import escalating_beam_search
from tzrec.utils.test_util import parameterized_name_func


def _tiny_lm(vocab_size, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).eval()


def _bands(pairs):
    lo = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    hi = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return lo, hi


class _RowSpy:
    """Duck-typed backbone recording the row count of every ``.model`` call.

    The kernel touches only ``.model`` and ``.lm_head``, so the per-level beam
    width -- otherwise a local inside the kernel -- becomes observable.
    """

    def __init__(self, lm) -> None:
        self.lm_head = lm.lm_head
        self.rows: List[int] = []
        self._lm = lm

    def model(self, **kwargs: Any) -> Any:
        self.rows.append(kwargs["input_ids"].shape[0])
        return self._lm.model(**kwargs)


def _bruteforce_scores(lm, input_ids, attention_mask, pairs):
    """Exact cumulative log-prob of every band combination, by full recompute."""
    ref: Dict[Tuple[int, ...], float] = {}
    with torch.no_grad():
        for combo in itertools.product(*[range(lo, hi + 1) for lo, hi in pairs]):
            seq, mask, total = input_ids, attention_mask, 0.0
            for tok in combo:
                logits = lm(input_ids=seq, attention_mask=mask).logits[0, -1].float()
                total += float(torch.log_softmax(logits, dim=-1)[tok])
                seq = torch.cat([seq, torch.tensor([[tok]])], dim=1)
                mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
            ref[combo] = total
    return ref


class EscalatingBeamSearchTest(unittest.TestCase):
    def test_module_declares_no_tzrec_imports(self) -> None:
        # the "torch-only, no tzrec deps" docstring is what makes it liftable.
        tree = ast.parse(pathlib.Path(escalating_beam.__file__).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        self.assertEqual(
            {m for m in mods if m == "tzrec" or m.startswith("tzrec.")}, set()
        )

    @parameterized.expand(
        [
            # uncapped doubling: 2 -> 4 -> 8 -> 16
            [2, [(20, 27), (28, 34), (35, 40)], [4, 8, 16]],
            # capped by band x surviving prefixes, not by the doubling
            [3, [(20, 21), (22, 24)], [2, 6]],
            [1, [(20, 21), (22, 24), (25, 28)], [2, 4, 8]],
            # a width-1 band collapses level 0 to a single beam
            [2, [(20, 20), (21, 23)], [1, 3]],
        ],
        name_func=parameterized_name_func,
    )
    def test_width_schedule(self, num_beams, pairs, expected_widths) -> None:
        spy = _RowSpy(_tiny_lm(vocab_size=48))
        lo, hi = _bands(pairs)
        ids = torch.tensor([[5, 6, 7, 8]])
        out = escalating_beam_search(
            spy, ids, torch.ones_like(ids), num_beams=num_beams, lo_tok=lo, hi_tok=hi
        )
        # calls: [prompt (1 row)] + one per level>0, each carrying widths[j-1] rows
        widths = spy.rows[1:] + [out.shape[0]]
        self.assertEqual(spy.rows[0], 1)
        self.assertEqual(widths, expected_widths)
        self.assertEqual(tuple(out.shape), (expected_widths[-1], len(pairs)))

    def test_tokens_stay_inside_arbitrary_bands(self) -> None:
        # bands the Qwen2RecLM caller can never produce: descending, disjoint,
        # unequal width -- the kernel's contract is per-level (lo, hi), not a
        # contiguous codebook layout.
        pairs = [(5, 6), (20, 24), (11, 13)]
        lo, hi = _bands(pairs)
        ids = torch.tensor([[1, 2, 3, 4]])
        out = escalating_beam_search(
            _tiny_lm(vocab_size=30),
            ids,
            torch.ones_like(ids),
            num_beams=2,
            lo_tok=lo,
            hi_tok=hi,
        )
        self.assertEqual(tuple(out.shape), (min(2 * 2**3, 2 * 5 * 3), 3))
        for level, (lo_j, hi_j) in enumerate(pairs):
            col = out[:, level]
            self.assertTrue(bool((col >= lo_j).all()))
            self.assertTrue(bool((col <= hi_j).all()))
        self.assertEqual(len({tuple(r) for r in out.tolist()}), out.shape[0])

    @parameterized.expand(
        [[0, 1], [1, 3], [2, 16], [0, 16]],
        name_func=parameterized_name_func,
    )
    def test_left_padding_matches_unpadded(self, seed, n_pad) -> None:
        pairs = [(20, 21), (22, 24), (25, 28)]
        lo, hi = _bands(pairs)
        lm = _tiny_lm(vocab_size=30, seed=seed)
        short = torch.tensor([[5, 6, 7]])
        pad = torch.cat([torch.zeros(1, n_pad, dtype=torch.long), short], dim=1)
        am_pad = torch.cat(
            [torch.zeros(1, n_pad, dtype=torch.long), torch.ones_like(short)], dim=1
        )
        plain = escalating_beam_search(
            lm, short, torch.ones_like(short), num_beams=2, lo_tok=lo, hi_tok=hi
        )
        padded = escalating_beam_search(
            lm, pad, am_pad, num_beams=2, lo_tok=lo, hi_tok=hi
        )
        self.assertTrue(torch.equal(plain, padded))

    def test_ragged_batch_rows_match_solo_runs(self) -> None:
        # every row of a ragged batch must decode exactly as if run alone.
        pairs = [(20, 21), (22, 24), (25, 28)]
        lo, hi = _bands(pairs)
        lm = _tiny_lm(vocab_size=30)
        row0, row1 = torch.tensor([[5, 6, 7, 8]]), torch.tensor([[9, 10, 11]])
        ids = torch.tensor([[5, 6, 7, 8], [0, 9, 10, 11]])
        am = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        out = escalating_beam_search(lm, ids, am, num_beams=2, lo_tok=lo, hi_tok=hi)
        width = out.shape[0] // 2
        for i, solo_ids in enumerate([row0, row1]):
            solo = escalating_beam_search(
                lm,
                solo_ids,
                torch.ones_like(solo_ids),
                num_beams=2,
                lo_tok=lo,
                hi_tok=hi,
            )
            self.assertTrue(torch.equal(out[i * width : (i + 1) * width], solo))

    def test_exhaustive_matches_bruteforce_topk(self) -> None:
        # widths [2, 6, 12] over 2*3*2 = 12 combinations -> no pruning at any
        # level, so the beam must reproduce the exact full-recompute ranking.
        pairs = [(20, 21), (22, 24), (25, 26)]
        lo, hi = _bands(pairs)
        lm = _tiny_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        am = torch.ones_like(ids)
        out = escalating_beam_search(lm, ids, am, num_beams=6, lo_tok=lo, hi_tok=hi)
        got = [tuple(r) for r in out.tolist()]
        ref = _bruteforce_scores(lm, ids, am, pairs)
        self.assertEqual(set(got), set(ref))
        self.assertEqual(len(got), len(ref))
        scores = [ref[c] for c in got]
        self.assertTrue(
            all(scores[i] >= scores[i + 1] - 1e-4 for i in range(len(scores) - 1))
        )
        self.assertEqual(got[0], max(ref, key=ref.get))


if __name__ == "__main__":
    unittest.main()
