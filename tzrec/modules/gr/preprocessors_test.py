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
        contextual_feature_dim=st.sampled_from([32, 64]),
        enable_interleaving=st.sampled_from([True, False]),
        enable_pmlp=st.sampled_from([True, False]),
        is_train=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_contextual_interleave_preprocessor(
        self,
        contextual_feature_dim: int,
        enable_interleaving: bool,
        enable_pmlp: bool,
        is_train: bool,
    ) -> None:
        from tzrec.modules.gr.preprocessors import ContextualInterleavePreprocessor

        device = torch.device("cuda")

        uih_embedding_dim = 64
        target_embedding_dim = 128
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
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            output_embedding_dim=output_embedding_dim,
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=contextual_len,
            content_encoder=dict(slice_content_encoder=dict()),
            content_mlp=dict(
                parameterized_mlp=dict(hidden_dim=content_encoder_hidden_dim)
            )
            if enable_pmlp
            else dict(simple_mlp=dict(hidden_dim=content_encoder_hidden_dim)),
            action_encoder=dict(
                simple_action_encoder=dict(
                    action_weights=action_weights,
                    watchtime_to_action_thresholds=watchtime_to_action_thresholds,
                    watchtime_to_action_weights=watchtime_to_action_weights,
                    action_embedding_dim=action_embedding_dim,
                )
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
        uih_lengths = [4, 2]
        num_targets = [2, 1]
        uih_timestamps = [1, 2, 3, 4, 10, 20]
        candidate_timestamp = [5, 6, 30]
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
        ) = preprocessor(
            grouped_features={
                "uih.sequence": torch.rand(
                    (sum(uih_lengths), uih_embedding_dim), device=device
                ),
                "uih.sequence_length": torch.tensor(uih_lengths, device=device),
                "candidate.sequence": torch.rand(
                    (sum(num_targets), target_embedding_dim), device=device
                ),
                "candidate.sequence_length": torch.tensor(num_targets, device=device),
                "contextual": torch.rand(
                    (len(uih_lengths), contextual_len * contextual_feature_dim),
                    device=device,
                ),
                "uih_action.sequence": torch.tensor(actions, device=device).unsqueeze(
                    1
                ),
                "uih_watchtime.sequence": torch.tensor(
                    watchtimes, device=device
                ).unsqueeze(1),
                "uih_timestamp.sequence": torch.tensor(
                    uih_timestamps, device=device
                ).unsqueeze(1),
                "candidate_timestamp.sequence": torch.tensor(
                    candidate_timestamp, device=device
                ).unsqueeze(1),
            }
        )
        seq_lengths = [6, 3]
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
