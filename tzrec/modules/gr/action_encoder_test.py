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
from parameterized import parameterized

from tzrec.modules.gr.action_encoder import SimpleActionEncoder
from tzrec.utils.test_util import TestGraphType, create_test_module, gpu_unavailable


class ActionEncoderTest(unittest.TestCase):
    @parameterized.expand([[TestGraphType.NORMAL], [TestGraphType.FX_TRACE]])
    @unittest.skipIf(*gpu_unavailable)
    def test_simple_action_encoder(self, graph_type) -> None:
        device = torch.device("cuda")
        action_embedding_dim = 32
        action_weights = [1, 2, 4, 8, 16]
        watchtime_to_action_thresholds = [30, 60, 100]
        watchtime_to_action_weights = [32, 64, 128]
        num_action_types = len(action_weights) + len(watchtime_to_action_thresholds)
        combined_action_weights = action_weights + watchtime_to_action_weights
        enabled_actions = [
            [0],
            [0, 1],
            [1, 3, 4],
            [1, 2, 3, 4],
            [1, 2],
            [2],
        ]
        watchtimes = [40, 20, 110, 31, 26, 55]
        for i, wt in enumerate(watchtimes):
            for j, w in enumerate(
                zip(watchtime_to_action_thresholds, watchtime_to_action_weights)
            ):
                if wt > w[0]:
                    enabled_actions[i].append(j + len(action_weights))
        actions = [
            sum([combined_action_weights[t] for t in x]) for x in enabled_actions
        ]

        encoder = SimpleActionEncoder(
            action_weights=action_weights,
            watchtime_to_action_thresholds=watchtime_to_action_thresholds,
            watchtime_to_action_weights=watchtime_to_action_weights,
            action_embedding_dim=action_embedding_dim,
            is_inference=False,
        ).to(device)
        encoder = create_test_module(encoder, graph_type)

        seq_lengths = [6, 3]
        seq_offsets = [0, 6, 9]
        num_targets = [2, 1]
        uih_offsets = [0, 4, 6]
        target_offsets = [0, 2, 3]
        action_embeddings = encoder(
            seq_actions=torch.tensor(actions, device=device),
            max_uih_len=4,
            max_targets=2,
            uih_offsets=torch.tensor(uih_offsets, device=device),
            target_offsets=torch.tensor(target_offsets, device=device),
            total_uih_len=6,
            total_targets=3,
            seq_watchtimes=torch.tensor(watchtimes, device=device),
        )
        self.assertEqual(
            action_embeddings.shape, (9, action_embedding_dim * num_action_types)
        )
        for b in range(len(seq_lengths)):
            b_start = seq_offsets[b]
            b_end = seq_offsets[b + 1]
            u_start = uih_offsets[b]
            for j in range(b_start, b_end):
                embedding = action_embeddings[j].view(num_action_types, -1)
                for atype in range(num_action_types):
                    if b_end - j <= num_targets[b]:
                        torch.testing.assert_close(
                            embedding[atype],
                            encoder._target_action_embedding_table.view(
                                num_action_types, -1
                            )[atype],
                        )
                    else:
                        if atype in enabled_actions[j - b_start + u_start]:
                            torch.testing.assert_close(
                                embedding[atype],
                                encoder._action_embedding_table[atype],
                            )
                        else:
                            torch.testing.assert_close(
                                embedding[atype], torch.zeros_like(embedding[atype])
                            )
        if graph_type == TestGraphType.NORMAL:
            action_embeddings.sum().backward()


if __name__ == "__main__":
    unittest.main()
