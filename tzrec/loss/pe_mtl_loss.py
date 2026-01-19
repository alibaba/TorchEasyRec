# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import numpy as np
import torch
from scipy.optimize import minimize, nnls


class ParetoEfficientMultiTaskLoss(torch.nn.Module):
    """Dynamic loss weights based on the Pareto-Efficient for multi-task learning."""

    def __init__(self, min_c: list[float]) -> None:
        super().__init__()
        assert 0 <= sum(min_c) < 1.0, (
            "all pareto_min_loss_weight sum should be in [0, 1), not {}".format(
                sum(min_c)
            )
        )
        self._c = np.array(min_c).reshape([-1, 1])

    def _pareto_step(self, W: np.ndarray, C: np.ndarray, G: np.ndarray) -> np.array:
        """Ref:http://ofey.me/papers/Pareto.pdf.

        Args:
                W: dimension (K,1)
                C: dimension (K,1)
                G: multi loss grad. dimension (K,M)
        """
        GGT = np.matmul(G, G.T)  # (K, K)
        e = np.mat(np.ones(np.shape(W)))  # (K, 1)
        m_up = np.hstack((GGT, e))  # (K, K+1)
        m_down = np.hstack(
            (e.T, np.mat(np.zeros([1, 1], dtype=np.float32)))
        )  # (1, K+1)
        M = np.vstack((m_up, m_down))  # (K+1, K+1)
        z = np.vstack((-np.matmul(GGT, C), 1 - np.sum(C)))  # (K+1, 1)
        hat_w = np.matmul(np.matmul(np.linalg.inv(np.matmul(M.T, M)), M), z)  # (K+1, 1)
        hat_w = np.asarray(hat_w[:-1])  # (k,1)
        hat_w = np.reshape(hat_w, (hat_w.shape[0],))
        c = np.reshape(np.array(C), (C.shape[0],))  # (K,)
        new_w = self._asm(hat_w, c)
        return new_w

    def _asm(self, hat_w: np.array, c: np.array) -> np.array:
        """Ref: http://ofey.me/papers/Pareto.pdf.

        Args:
                hat_w: dimension (K)
                c: dimension (K)
        """
        A = np.array(
            [[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))]
        )
        b = hat_w
        x0, _ = nnls(A, b)

        def _fn(x, A, b):
            return np.linalg.norm(A.dot(x) - b)

        cons = {"type": "eq", "fun": lambda x: np.sum(x) + np.sum(c) - 1}
        bounds = [[0.0, None] for _ in range(len(hat_w))]
        min_out = minimize(
            _fn, x0, args=(A, b), method="SLSQP", bounds=bounds, constraints=cons
        )
        new_w = min_out.x + c
        return new_w

    def forward(
        self, losses: Dict[str, torch.Tensor], model: torch.nn.Module
    ) -> torch.Tensor:
        """Compute pareto front of losses for weight."""
        grads = []

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        for loss in losses.values():
            loss_sum = torch.sum(loss, dim=0)
            gradients = torch.autograd.grad(
                loss_sum,
                trainable_params,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            grad_flattened = []
            for grad, param in zip(gradients, trainable_params):
                if grad is not None:
                    grad_flattened.append(grad.view(-1))
                else:
                    grad_flattened.append(torch.zeros_like(param).view(-1))

            all_grads = torch.cat(grad_flattened)
            grads.append(all_grads)
        # grads.append(all_grads.cpu().numpy())
        grads = torch.stack(grads).detach().data.cpu().numpy()
        init_weight = 1 / len(losses)
        w = np.array([init_weight] * len(losses), dtype=np.float32).reshape([-1, 1])
        new_w = self._pareto_step(w, self._c, grads)
        new_loss = []
        for i, loss in enumerate(losses.values()):
            new_loss.append(loss * new_w[i])
        losses = torch.stack(new_loss).sum()
        return losses
