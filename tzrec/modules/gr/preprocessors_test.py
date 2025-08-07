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

import unittest

import torch
from hypothesis import Verbosity, given
from hypothesis import strategies as st

from tzrec.utils.test_util import gpu_unavailable
from tzrec.utils.test_util import hypothesis_settings as settings


class PreprocessorTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        enable_interleaving=st.sampled_from([True, False]),
        enable_pmlp=st.sampled_from([True, False]),
        is_train=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_contextual_interleave_preprocessor(
        self,
        enable_interleaving: bool,
        enable_pmlp: bool,
        is_train: bool,
    ) -> None:
        from tzrec.modules.gr.preprocessors import ContextualInterleavePreprocessor

        device = torch.device("cuda")

        input_embedding_dim = 64
        output_embedding_dim = 32
        action_embedding_dim = 16
        action_encoder_hidden_dim = 256
        content_encoder_hidden_dim = 128
        contextual_len = 3

        action_embedding_dim = 32
        action_weights = [1, 2, 4, 8, 16]
        watchtime_to_action_thresholds = [30, 60, 100]
        watchtime_to_action_weights = [32, 64, 128]

        preprocessor = ContextualInterleavePreprocessor(
            input_embedding_dim=input_embedding_dim,
            output_embedding_dim=output_embedding_dim,
            contextual_feature_to_max_length={"c_0": 1, "c_1": 2},
            contextual_feature_to_min_uih_length={"c_1": 4},
            pmlp_contextual_dropout_ratio=0.2,
            content_encoder=dict(
                additional_content_features={
                    "a0": input_embedding_dim,
                    "a1": input_embedding_dim,
                },
                target_enrich_features={
                    "t0": input_embedding_dim,
                    "t1": input_embedding_dim,
                },
            ),
            content_mlp=dict(
                parameterized_mlp=dict(hidden_dim=content_encoder_hidden_dim)
            )
            if enable_pmlp
            else dict(simple_mlp=dict(hidden_dim=content_encoder_hidden_dim)),
            action_encoder=dict(
                watchtime_feature_name="watchtimes",
                action_feature_name="actions",
                action_weights=action_weights,
                watchtime_to_action_thresholds=watchtime_to_action_thresholds,
                watchtime_to_action_weights=watchtime_to_action_weights,
                action_embedding_dim=action_embedding_dim,
            ),
            action_mlp=dict(
                parameterized_mlp=dict(hidden_dim=action_encoder_hidden_dim)
            )
            if enable_pmlp
            else dict(simple_mlp=dict(hidden_dim=action_encoder_hidden_dim)),
            enable_interleaving=enable_interleaving,
            is_inference=False,
        ).to(device)
        if not is_train:
            preprocessor.eval()

        # inputs
        seq_lengths = [6, 3]
        num_targets = [2, 1]
        seq_embeddings = torch.rand(
            (sum(seq_lengths), input_embedding_dim),
            device=device,
        )
        seq_timestamps = torch.tensor(
            [1, 2, 3, 4, 5, 6, 10, 20, 30],
            device=device,
        )
        watchtimes = [40, 20, 110, 31, 26, 55]
        actions = [1, 3, 26, 30, 6, 4]
        (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            _,
        ) = preprocessor(
            max_uih_len=4,
            max_targets=2,
            total_uih_len=sum(seq_lengths) - sum(num_targets),
            total_targets=sum(num_targets),
            seq_lengths=torch.tensor(seq_lengths, device=device),
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            seq_payloads={
                # contextual
                "c_0": torch.rand((2, input_embedding_dim), device=device),
                "c_0_offsets": torch.tensor([0, 1, 1], device=device),
                "c_1": torch.rand((4, input_embedding_dim), device=device),
                "c_1_offsets": torch.tensor([0, 2, 3], device=device),
                # action
                "watchtimes": torch.tensor(watchtimes, device=device),
                "actions": torch.tensor(actions, device=device),
                # content
                "a0": torch.rand_like(seq_embeddings).requires_grad_(True),
                "a1": torch.rand_like(seq_embeddings).requires_grad_(True),
                "t0": torch.rand(
                    sum(num_targets), input_embedding_dim, device=device
                ).requires_grad_(True),
                "t1": torch.rand(
                    sum(num_targets), input_embedding_dim, device=device
                ).requires_grad_(True),
            },
            num_targets=torch.tensor(num_targets, device=device),
        )
        if enable_interleaving:
            if is_train:
                expected_output_seq_lengths = [
                    2 * s + contextual_len for s in seq_lengths
                ]
                expected_max_seq_len = max(expected_output_seq_lengths)
                expected_output_num_targets = [2 * s for s in num_targets]
                expected_seq_embedding_size = (
                    sum(expected_output_seq_lengths),
                    output_embedding_dim,
                )
                expected_seq_timestamps_size = (sum(expected_output_seq_lengths),)
                expected_output_seq_timestamps = [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    0,
                    0,
                    0,
                    10,
                    10,
                    20,
                    20,
                    30,
                    30,
                ]
            else:
                expected_output_seq_lengths = [
                    2 * s - n + contextual_len for s, n in zip(seq_lengths, num_targets)
                ]
                expected_max_seq_len = max(expected_output_seq_lengths)
                expected_output_num_targets = num_targets
                expected_seq_embedding_size = (
                    sum(expected_output_seq_lengths),
                    output_embedding_dim,
                )
                expected_seq_timestamps_size = (sum(expected_output_seq_lengths),)
                expected_output_seq_timestamps = [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    6,
                    0,
                    0,
                    0,
                    10,
                    10,
                    20,
                    20,
                    30,
                ]
        else:
            expected_output_seq_lengths = [s + contextual_len for s in seq_lengths]
            expected_max_seq_len = max(expected_output_seq_lengths)
            expected_output_num_targets = num_targets
            expected_seq_embedding_size = (
                sum(expected_output_seq_lengths),
                output_embedding_dim,
            )
            expected_seq_timestamps_size = (sum(expected_output_seq_lengths),)
            expected_output_seq_timestamps = [
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                0,
                0,
                10,
                20,
                30,
            ]

        self.assertEqual(output_max_seq_len, expected_max_seq_len)
        self.assertEqual(output_seq_lengths.tolist(), expected_output_seq_lengths)
        torch.testing.assert_close(
            torch.ops.fbgemm.asynchronous_complete_cumsum(output_seq_lengths),
            output_seq_offsets,
        )
        self.assertEqual(output_num_targets.tolist(), expected_output_num_targets)
        self.assertEqual(
            output_seq_embeddings.size(),
            expected_seq_embedding_size,
        )
        self.assertEqual(
            output_seq_timestamps.size(),
            expected_seq_timestamps_size,
        )
        self.assertEqual(
            output_seq_timestamps.tolist(),
            expected_output_seq_timestamps,
        )


if __name__ == "__main__":
    unittest.main()
