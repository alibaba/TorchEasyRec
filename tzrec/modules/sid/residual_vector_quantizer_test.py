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

from tzrec.modules.sid.residual_vector_quantizer import faiss_residual_kmeans


class FaissResidualKmeansTest(unittest.TestCase):
    """Tests for the FAISS residual K-Means warm-start helper."""

    def test_faiss_residual_kmeans_per_layer_centers(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        samples = torch.randn(512, 6)
        centers = faiss_residual_kmeans(
            samples, [8, 4], {"niter": 5, "verbose": False, "seed": 1}
        )
        self.assertEqual(len(centers), 2)
        self.assertEqual(centers[0].shape, (8, 6))
        self.assertEqual(centers[1].shape, (4, 6))
        self.assertTrue(torch.isfinite(centers[0]).all())
        # Centroids come back on the input device (CPU fit, device-preserving).
        self.assertEqual(centers[0].device, samples.device)


if __name__ == "__main__":
    unittest.main()
