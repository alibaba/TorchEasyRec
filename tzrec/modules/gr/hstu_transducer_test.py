# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit + integration tests for ``HSTUTransducer``."""

import unittest
from typing import Dict, List
from unittest.mock import patch

import torch
from parameterized import parameterized

from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.ops import Kernel
from tzrec.ops.hstu_attention_utils import compute_stu_truncation_plan
from tzrec.utils.test_util import reference_stu_truncation


class _StubInputPreprocessor(torch.nn.Module):
    """Stub preprocessor returning a synthetic 8-tuple from recorded inputs.

    Avoids the full ContentEncoder / ActionEncoder / MLP chain that the
    real ContextualInterleavePreprocessor requires.
    """

    def __init__(
        self,
        lengths: List[int],
        targets: List[int],
        embedding_dim: int,
        contextual_seq_len: int,
        interleave_targets: bool = False,
    ) -> None:
        super().__init__()
        self._lengths = lengths
        self._targets = targets
        self._embedding_dim = embedding_dim
        self._contextual_seq_len = contextual_seq_len
        self._interleave_targets = interleave_targets

    def contextual_seq_len(self) -> int:
        return self._contextual_seq_len

    def interleave_targets(self) -> bool:
        return self._interleave_targets

    def forward(self, grouped_features):
        seq_lengths = torch.tensor(self._lengths, dtype=torch.int64)
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        total = int(seq_offsets[-1].item())
        num_targets = torch.tensor(self._targets, dtype=torch.int64)
        total_targets = int(num_targets.sum().item())
        return (
            int(seq_lengths.max().item()),
            total - total_targets,
            total_targets,
            seq_lengths,
            seq_offsets,
            torch.arange(total, dtype=torch.float32),
            torch.randn(total, self._embedding_dim),
            num_targets,
        )


class _StubOutputPostprocessor(torch.nn.Module):
    def forward(
        self, seq_embeddings: torch.Tensor, seq_timestamps: torch.Tensor
    ) -> torch.Tensor:
        return seq_embeddings


class HSTUTransducerTest(unittest.TestCase):
    # ---------- _replay_truncation_state unit tests ----------

    def _replay_setup(self):
        lengths = [10, 14, 6]
        targets = [2, 3, 1]
        ctx, tail = 2, 4
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(lengths, dtype=torch.int64)
        )
        total = int(offsets[-1].item())
        seq_lengths = offsets[1:] - offsets[:-1]
        seq_timestamps = torch.arange(total, dtype=torch.float32) * 7.0
        plan = compute_stu_truncation_plan(
            x_offsets=offsets,
            num_targets=torch.tensor(targets, dtype=torch.int64),
            max_seq_len=int(seq_lengths.max().item()),
            truncate_tail_len=tail,
            contextual_seq_len=ctx,
        )
        return {
            "lengths": lengths,
            "targets": targets,
            "ctx": ctx,
            "tail": tail,
            "offsets": offsets,
            "seq_lengths": seq_lengths,
            "seq_timestamps": seq_timestamps,
            "plan": plan,
            "max_seq_len": int(seq_lengths.max().item()),
            "total_uih_len": int(seq_lengths.sum().item()) - sum(targets),
            "total_targets": sum(targets),
        }

    def test_replay_truncation_state_pass_through_when_plan_is_none(self) -> None:
        s = self._replay_setup()
        out = HSTUTransducer._replay_truncation_state(
            seq_timestamps=s["seq_timestamps"],
            seq_lengths=s["seq_lengths"],
            seq_offsets=s["offsets"],
            max_seq_len=s["max_seq_len"],
            total_uih_len=s["total_uih_len"],
            total_targets=s["total_targets"],
            post_stu_seq_offsets=s["offsets"],
            post_stu_max_seq_len=s["max_seq_len"],
            plan=None,
            kernel=Kernel.PYTORCH,
        )
        out_ts, out_lens, out_offsets, out_max, out_total_uih = out
        self.assertIs(out_ts, s["seq_timestamps"])
        self.assertIs(out_lens, s["seq_lengths"])
        self.assertIs(out_offsets, s["offsets"])
        self.assertEqual(out_max, s["max_seq_len"])
        self.assertEqual(out_total_uih, s["total_uih_len"])

    def test_replay_truncation_state_replays_plan_and_assigns_correct_fields(
        self,
    ) -> None:
        s = self._replay_setup()
        post_stu_seq_offsets = s["plan"].new_x_offsets
        post_stu_max_seq_len = s["plan"].new_max_seq_len
        out = HSTUTransducer._replay_truncation_state(
            seq_timestamps=s["seq_timestamps"],
            seq_lengths=s["seq_lengths"],
            seq_offsets=s["offsets"],
            max_seq_len=s["max_seq_len"],
            total_uih_len=s["total_uih_len"],
            total_targets=s["total_targets"],
            post_stu_seq_offsets=post_stu_seq_offsets,
            post_stu_max_seq_len=post_stu_max_seq_len,
            plan=s["plan"],
            kernel=Kernel.PYTORCH,
        )
        out_ts, out_lens, out_offsets, out_max, out_total_uih = out

        # Timestamps truncated against an independent reference.
        ref_ts, _ = reference_stu_truncation(
            s["seq_timestamps"].unsqueeze(-1),
            s["offsets"],
            s["targets"],
            truncate_tail_len=s["tail"],
            contextual_seq_len=s["ctx"],
        )
        torch.testing.assert_close(out_ts, ref_ts.squeeze(-1))
        # seq_lengths must come from plan.new_lengths (not new_x_offsets).
        torch.testing.assert_close(out_lens, s["plan"].new_lengths)
        self.assertEqual(out_lens.shape[0], len(s["lengths"]))
        self.assertEqual(s["plan"].new_x_offsets.shape[0], len(s["lengths"]) + 1)
        # seq_offsets / max_seq_len come from the post-STU outputs.
        self.assertIs(out_offsets, post_stu_seq_offsets)
        self.assertEqual(out_max, post_stu_max_seq_len)
        expected_total_uih = s["plan"].total_kept - s["total_targets"]
        self.assertEqual(out_total_uih, expected_total_uih)

    # ---------- forward() end-to-end integration tests ----------

    def _build_transducer(
        self,
        attn_truncation_split_layer: int,
        attn_truncation_tail_len: int,
        lengths: List[int],
        targets: List[int],
        contextual_seq_len: int = 0,
        interleave_targets: bool = False,
        embedding_dim: int = 16,
        attn_num_layers: int = 3,
    ) -> HSTUTransducer:
        """Construct a transducer with stubbed preprocessor / postprocessor.

        The full ``ContextualInterleavePreprocessor`` requires an
        action_encoder + content_encoder + MLP chain; for an integration
        test focused on the truncation glue we stub those out via
        :func:`unittest.mock.patch.object` over the factory functions.
        """
        stub_pre = _StubInputPreprocessor(
            lengths=lengths,
            targets=targets,
            embedding_dim=embedding_dim,
            contextual_seq_len=contextual_seq_len,
            interleave_targets=interleave_targets,
        )
        stub_post = _StubOutputPostprocessor()
        with (
            patch(
                "tzrec.modules.gr.hstu_transducer.create_input_preprocessor",
                return_value=stub_pre,
            ),
            patch(
                "tzrec.modules.gr.hstu_transducer.create_output_postprocessor",
                return_value=stub_post,
            ),
        ):
            transducer = HSTUTransducer(
                uih_embedding_dim=embedding_dim,
                target_embedding_dim=embedding_dim,
                stu={
                    "embedding_dim": embedding_dim,
                    "num_heads": 2,
                    "hidden_dim": 32,
                    "attention_dim": 32,
                    "output_dropout_ratio": 0.0,
                    "causal": True,
                    "target_aware": True,
                },
                attn_num_layers=attn_num_layers,
                input_preprocessor={"contextual_preprocessor": {}},
                output_postprocessor={"l2norm_postprocessor": {}},
                attn_truncation_split_layer=attn_truncation_split_layer,
                attn_truncation_tail_len=attn_truncation_tail_len,
                is_inference=False,
            )
        # Set kernel on the whole transducer so split_2D_jagged calls in
        # _postprocess (which read self.kernel()) also dispatch to PyTorch.
        transducer.set_kernel(Kernel.PYTORCH)
        return transducer

    @parameterized.expand(
        [
            # split, tail, interleave, ctx, lengths,         targets
            [0, 0, False, 0, [12, 20, 5, 30], [2, 3, 1, 2]],  # disabled baseline
            [1, 4, False, 0, [12, 20, 5, 30], [2, 3, 1, 2]],  # truncation, plain
            [1, 4, True, 0, [12, 20, 5, 30], [2, 4, 2, 2]],  # interleave (T=10, even)
            [1, 4, False, 3, [15, 22, 8, 30], [2, 3, 1, 2]],  # contextual prefix
            [1, 4, True, 3, [15, 22, 8, 30], [2, 4, 2, 2]],  # interleave + ctx
        ]
    )
    def test_forward_end_to_end(
        self,
        split: int,
        tail: int,
        interleave: bool,
        ctx: int,
        lengths: List[int],
        targets: List[int],
    ) -> None:
        """Full ``forward`` runs cleanly across the (interleave, ctx) matrix.

        Catches: wrong field assignment in ``_replay_truncation_state``,
        num_targets / seq_lengths mismatch in ``_postprocess`` after
        truncation, kernel not threading through, shape mismatch on the
        seq_timestamps unsqueeze/squeeze round-trip, the
        ``view(-1, 2)`` candidate reshape under interleave, and the
        3-op contextual-prefix path through ``apply_stu_truncation_plan``.
        """
        transducer = self._build_transducer(
            attn_truncation_split_layer=split,
            attn_truncation_tail_len=tail,
            lengths=lengths,
            targets=targets,
            contextual_seq_len=ctx,
            interleave_targets=interleave,
        )
        # grouped_features is consumed by the stubbed preprocessor only.
        grouped_features: Dict[str, torch.Tensor] = {}
        encoded_candidate_embeddings, encoded_embeddings = transducer(grouped_features)
        self.assertTrue(torch.isfinite(encoded_candidate_embeddings).all())
        # Under interleave, candidates are reshaped (T, D) -> (T // 2, D).
        expected_rows = sum(targets) // (2 if interleave else 1)
        self.assertEqual(encoded_candidate_embeddings.size(0), expected_rows)
        # return_full_embeddings=False (default), so encoded_embeddings is None.
        self.assertIsNone(encoded_embeddings)


if __name__ == "__main__":
    unittest.main()
