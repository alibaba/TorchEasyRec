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

import itertools
import unittest
from typing import Any, Dict, List, Tuple

import torch
from parameterized import parameterized

from tzrec.modules.dynamic_beam import capped_beam_widths, dynamic_beam_search
from tzrec.utils.test_util import create_tiny_causal_lm, parameterized_name_func


def _decode(lm, ids, pairs, width=8, beam_widths=None, attention_mask=None):
    """Run the kernel over ``pairs`` of inclusive per-level (lo, hi) band edges.

    Defaults to a FLAT schedule: the kernel is policy-free, so only the cases
    about scheduling spell one out. The kernel takes already-capped widths, so
    the helper caps them the way the model does.
    """
    if beam_widths is None:
        beam_widths = [width] * len(pairs)
    return dynamic_beam_search(
        lm,
        lm.get_input_embeddings()(ids),
        torch.ones_like(ids) if attention_mask is None else attention_mask,
        capped_widths=capped_beam_widths(beam_widths, pairs),
        bands=pairs,
    )


class _RowSpy:
    """Backbone wrapper recording the row count of every forward call.

    The kernel calls the model itself, so the per-level beam width -- otherwise
    a local inside the kernel -- becomes observable.
    """

    def __init__(self, lm) -> None:
        self.rows: List[int] = []
        self._lm = lm

    def get_input_embeddings(self):
        return self._lm.get_input_embeddings()

    def __call__(self, **kwargs: Any) -> Any:
        first = kwargs.get("input_ids")
        if first is None:
            first = kwargs["inputs_embeds"]
        self.rows.append(first.shape[0])
        return self._lm(**kwargs)


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


class DynamicBeamSearchTest(unittest.TestCase):
    @parameterized.expand(
        [
            # every request satisfiable -> the schedule is honoured verbatim
            [[4, 8, 16], [(20, 27), (28, 34), (35, 40)], [4, 8, 16]],
            # capped by band x surviving prefixes, not by what was asked
            [[6, 12], [(20, 21), (22, 24)], [2, 6]],
            # a width-1 band collapses level 0 and bounds everything after it
            [[4, 8], [(20, 20), (21, 23)], [1, 3]],
            # a flat (non-doubling) schedule is just as valid to the kernel
            [[3, 3, 3], [(20, 27), (28, 34), (35, 40)], [3, 3, 3]],
        ],
        name_func=parameterized_name_func,
    )
    def test_width_schedule(self, beam_widths, pairs, expected_widths) -> None:
        spy = _RowSpy(create_tiny_causal_lm(vocab_size=48))
        ids = torch.tensor([[5, 6, 7, 8]])
        out = _decode(spy, ids, pairs, beam_widths=beam_widths)
        # calls: [prompt (1 row)] + one per level>0, each carrying widths[j-1] rows
        widths = spy.rows[1:] + [out.shape[0]]
        self.assertEqual(spy.rows[0], 1)
        self.assertEqual(widths, expected_widths)
        self.assertEqual(tuple(out.shape), (expected_widths[-1], len(pairs)))

    def test_ragged_batch_rows_match_solo_runs(self) -> None:
        # every row of a ragged batch must decode exactly as if run alone.
        pairs = [(20, 21), (22, 24), (25, 28)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8], [0, 9, 10, 11]])
        am = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        out = _decode(lm, ids, pairs, attention_mask=am)
        width = out.shape[0] // 2
        solos = [torch.tensor([[5, 6, 7, 8]]), torch.tensor([[9, 10, 11]])]
        for i, solo_ids in enumerate(solos):
            solo = _decode(lm, solo_ids, pairs)
            self.assertTrue(torch.equal(out[i * width : (i + 1) * width], solo))

    def test_exhaustive_matches_bruteforce_topk(self) -> None:
        # widths [2, 6, 12] over 2*3*2 = 12 combinations -> no pruning at any
        # level. Disjoint, non-monotonic bands also verify that each level uses
        # its own bounds rather than assuming one contiguous SID layout.
        pairs = [(20, 21), (5, 7), (25, 26)]
        lm = create_tiny_causal_lm(vocab_size=30)
        ids = torch.tensor([[5, 6, 7, 8]])
        am = torch.ones_like(ids)
        out = _decode(lm, ids, pairs, width=12)
        got = [tuple(r) for r in out.tolist()]
        ref = _bruteforce_scores(lm, ids, am, pairs)
        self.assertEqual(set(got), set(ref))
        self.assertEqual(len(got), len(ref))
        scores = [ref[c] for c in got]
        self.assertTrue(
            all(scores[i] >= scores[i + 1] - 1e-4 for i in range(len(scores) - 1))
        )
        self.assertEqual(got[0], max(ref, key=lambda combo: ref[combo]))


if __name__ == "__main__":
    unittest.main()
