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

import unittest

import torch
from torch import nn

from tzrec.modules.sid.residual_quantizer import (
    ResidualQuantizer,
    normalize_n_embed,
)
from tzrec.modules.sid.types import QuantizeOutput


class NormalizeNEmbedTest(unittest.TestCase):
    def test_scalar_broadcasts(self) -> None:
        self.assertEqual(normalize_n_embed(256, 3), [256, 256, 256])

    def test_list_passes_through(self) -> None:
        self.assertEqual(normalize_n_embed([8, 4, 2], 3), [8, 4, 2])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(AssertionError):
            normalize_n_embed([8, 4], 3)


class ResidualQuantizerBaseTest(unittest.TestCase):
    """The abstract base owns shared state but not the backend primitives."""

    def test_shared_state_and_output_dim(self) -> None:
        rq = ResidualQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertEqual(rq.output_dim(), 4)
        self.assertEqual(rq.n_embed_list, [8, 8])
        self.assertEqual(len(rq.layers), 0)  # subclasses populate this

    def test_abstract_primitives_raise(self) -> None:
        rq = ResidualQuantizer(embed_dim=4, n_layers=2)
        x = torch.randn(3, 4)
        with self.assertRaises(NotImplementedError):
            rq.forward(x)
        with self.assertRaises(NotImplementedError):
            rq.get_codes(x)
        with self.assertRaises(NotImplementedError):
            rq.get_codebook_embeddings(0)
        # decode_codes is concrete but delegates to the abstract _lookup_code.
        with self.assertRaises(NotImplementedError):
            rq.decode_codes(torch.zeros(3, 2, dtype=torch.long))


class _FakeQuantizer(ResidualQuantizer):
    """Minimal concrete subclass to exercise the base residual walk.

    Implements only the two per-layer primitives over a learnable codebook,
    so the base's _residual_pass / get_codes / decode_codes can be tested
    without pulling in the K-Means or VQ backends.
    """

    def __init__(
        self,
        embed_dim,
        n_layers,
        n_embed=5,
        normalize_residuals=False,
        candidate_output_config=None,
    ):
        super().__init__(
            embed_dim,
            n_layers,
            n_embed,
            normalize_residuals,
            candidate_output_config=candidate_output_config,
        )
        self.books = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.n_embed_list[i], embed_dim))
                for i in range(n_layers)
            ]
        )

    def _quantize_layer(self, layer_idx, residual, topk=1):
        scores = residual.detach() @ self.books[layer_idx].t()
        topk_scores, topk_ids = torch.topk(scores, k=topk, dim=-1)
        codes = topk_ids[:, 0]
        return QuantizeOutput(
            embeddings=self.books[layer_idx][codes],
            ids=codes,
            scores=topk_scores[:, 0],
            topk_ids=topk_ids,
            topk_scores=topk_scores,
        )

    def _lookup_code(self, layer_idx, code_idx):
        return self.books[layer_idx][code_idx]

    def forward(self, input):
        return self._residual_pass(input)

    def get_codebook_embeddings(self, layer_idx):
        return self.books[layer_idx]


class ResidualQuantizerWalkTest(unittest.TestCase):
    """Exercise the concrete residual walk the base owns (via a fake backend)."""

    def test_residual_pass_shapes_and_aggregate(self) -> None:
        torch.manual_seed(0)
        fq = _FakeQuantizer(embed_dim=4, n_layers=3, n_embed=5)
        x = torch.randn(6, 4)
        ids, agg, cum, candidate_codes, candidate_scores = fq._residual_pass(x)
        self.assertEqual(ids.shape, (6, 3))
        self.assertEqual(fq.get_codes(x).shape, (6, 3))
        manual = sum(fq._lookup_code(i, ids[:, i]) for i in range(3))
        torch.testing.assert_close(agg, manual)  # aggregated == Σ quantized_i
        self.assertTrue(torch.equal(cum[-1], agg))
        self.assertIsNone(candidate_codes)
        self.assertIsNone(candidate_scores)

    def test_detach_invariant(self) -> None:
        torch.manual_seed(0)
        fq = _FakeQuantizer(embed_dim=4, n_layers=2, n_embed=5)
        x = torch.randn(5, 4, requires_grad=True)
        _, agg, _, _, _ = fq._residual_pass(x)
        # Codebook grad flows, but the residual chain is detached, so the
        # input receives no gradient.
        self.assertTrue(agg.requires_grad)
        agg.sum().backward()
        self.assertIsNotNone(fq.books[0].grad)
        self.assertIsNone(x.grad)

    def test_normalize_residuals_changes_assignment(self) -> None:
        # Same input and same codebook (re-seeded before each build), so the
        # only difference is the normalize_residuals branch — it must change
        # the residual a later layer sees and hence the codes it assigns.
        x = torch.randn(8, 4)
        torch.manual_seed(1)
        fq_off = _FakeQuantizer(embed_dim=4, n_layers=2, n_embed=6)
        torch.manual_seed(1)
        fq_on = _FakeQuantizer(
            embed_dim=4, n_layers=2, n_embed=6, normalize_residuals=True
        )
        ids_off, _, _, _, _ = fq_off._residual_pass(x)
        ids_on, _, _, _, _ = fq_on._residual_pass(x)
        self.assertEqual(ids_on.shape, (8, 2))
        self.assertFalse(torch.equal(ids_off, ids_on))

    def test_decode_codes_sum_and_dtype(self) -> None:
        torch.manual_seed(0)
        fq = _FakeQuantizer(embed_dim=4, n_layers=3, n_embed=5)
        codes = torch.randint(0, 5, (6, 3))
        recon = fq.decode_codes(codes)
        self.assertEqual(recon.shape, (6, 4))
        manual = sum(fq.books[i][codes[:, i]] for i in range(3))
        torch.testing.assert_close(recon, manual)
        # device/dtype follow the codebook (regression for the fp32-pin fix).
        fq16 = _FakeQuantizer(embed_dim=4, n_layers=2, n_embed=5).to(torch.bfloat16)
        recon16 = fq16.decode_codes(torch.randint(0, 5, (3, 2)))
        self.assertEqual(recon16.dtype, torch.bfloat16)


def _candidate_config(topk=2, strategy="last_layer_knn") -> dict:
    return {
        "enabled": True,
        "topk": topk,
        "strategy": strategy,
    }


class ResidualQuantizerCandidateConfigTest(unittest.TestCase):
    """Validation done by _init_candidate_output_config at construction."""

    def test_disabled_config_leaves_candidates_off(self) -> None:
        # enabled=False must short-circuit before topk/strategy are even read.
        fq = _FakeQuantizer(
            embed_dim=3,
            n_layers=2,
            n_embed=4,
            candidate_output_config={"enabled": False},
        )
        self.assertFalse(fq._candidate_output_enabled)

    def test_topk_below_one_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "topk must be >= 1"):
            _FakeQuantizer(
                embed_dim=3,
                n_layers=2,
                n_embed=4,
                candidate_output_config=_candidate_config(topk=0),
            )

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "last_layer_knn"):
            _FakeQuantizer(
                embed_dim=3,
                n_layers=2,
                n_embed=4,
                candidate_output_config=_candidate_config(strategy="random"),
            )

    def test_topk_exceeds_last_layer_codebook_raises(self) -> None:
        # n_embed=4 -> last-layer codebook size is 4; topk=5 is out of range.
        with self.assertRaisesRegex(ValueError, "must be <= target"):
            _FakeQuantizer(
                embed_dim=3,
                n_layers=2,
                n_embed=4,
                candidate_output_config=_candidate_config(topk=5),
            )


class ResidualQuantizerCandidateWalkTest(unittest.TestCase):
    """Candidate machinery exercised backend-agnostically via _FakeQuantizer."""

    def test_candidate_output_requires_eval_mode(self) -> None:
        # Inconsistent mode: candidate output is requested (is_inference + config
        # enabled) yet the module is still in .train(); _residual_pass must reject
        # it via the training guard.
        fq = _FakeQuantizer(
            embed_dim=3,
            n_layers=2,
            n_embed=4,
            candidate_output_config=_candidate_config(topk=2),
        )
        fq.set_is_inference(True)
        fq.train()
        with self.assertRaisesRegex(
            RuntimeError, "candidate SID output requires eval/inference mode."
        ):
            fq._residual_pass(torch.randn(3, 3))

    def test_build_candidates(self) -> None:
        # Last-layer codes are the raw top-k neighbors (candidate[:, 0] is the
        # top-scored one, == greedy here).
        torch.manual_seed(0)
        fq = _FakeQuantizer(
            embed_dim=3,
            n_layers=2,
            n_embed=4,
            candidate_output_config=_candidate_config(topk=2),
        )
        fq.eval()
        fq.set_is_inference(True)
        x = torch.randn(5, 3)
        _, _, _, cand_codes, cand_scores = fq._residual_pass(x)
        self.assertEqual(cand_codes.shape, (5, 2, 2))  # (B, topk, n_layers)
        self.assertEqual(cand_scores.shape, (5, 2))
        torch.testing.assert_close(cand_codes[:, 0, :], fq.get_codes(x))
        # The greedy prefix (all but the last layer) is shared by every candidate.
        self.assertTrue(
            torch.equal(cand_codes[:, :, 0], cand_codes[:, :1, 0].expand(-1, 2))
        )

    def test_build_candidates_topk_one(self) -> None:
        # topk==1: the single candidate is the greedy SID.
        torch.manual_seed(0)
        fq = _FakeQuantizer(
            embed_dim=3,
            n_layers=2,
            n_embed=4,
            candidate_output_config=_candidate_config(topk=1),
        )
        fq.eval()
        fq.set_is_inference(True)
        x = torch.randn(5, 3)
        _, _, _, cand_codes, cand_scores = fq._residual_pass(x)
        self.assertEqual(cand_codes.shape, (5, 1, 2))
        self.assertEqual(cand_scores.shape, (5, 1))
        torch.testing.assert_close(cand_codes[:, 0, :], fq.get_codes(x))


if __name__ == "__main__":
    unittest.main()
