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

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tzrec.modules.sid.residual_quantizer import ResidualQuantizer
from tzrec.modules.sid.residual_vector_quantizer import (
    ResidualVectorQuantizer,
    faiss_residual_kmeans,
)
from tzrec.modules.sid.types import ResidualQuantizerOutput
from tzrec.utils import misc_util


class GumbelResidualVQTest(unittest.TestCase):
    """Gumbel-Softmax forward_mode: grad reaches encoder + codebook (not STE)."""

    def test_default_gumbel_config_disables_sinkhorn(self) -> None:
        # use_sinkhorn defaults True; gumbel must auto-disable it (not crash).
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=3,
            n_embed=16,
            forward_mode="gumbel_softmax",
            use_sinkhorn=True,
            kmeans_init=False,
        )
        self.assertTrue(all(not layer.use_sinkhorn for layer in rvq.layers))

    def test_gumbel_grad_flows_via_soft_assignment(self) -> None:
        # The fix: the gradient from the reconstruction path (no commitment
        # loss) must reach BOTH the encoder input and the codebook through the
        # soft gumbel embedding. Under the old code it reached neither (the soft
        # embedding was discarded), so gumbel silently trained like STE.
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=3,
            n_embed=16,
            forward_mode="gumbel_softmax",
            use_sinkhorn=False,
            kmeans_init=False,
        )
        rvq.train()
        z = torch.randn(32, 8, requires_grad=True)
        rvq(z).quantized_embeddings.sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.0)
        cb_grad = rvq.layers[0].embedding.weight.grad
        self.assertIsNotNone(cb_grad)
        self.assertGreater(cb_grad.abs().sum().item(), 0.0)

    def test_ste_codebook_grad_is_detached_on_recon_path(self) -> None:
        # Contrast: STE detaches the aggregate, so the recon path gives the
        # codebook no gradient (it trains via the commitment loss instead).
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=2,
            n_embed=16,
            forward_mode="ste",
            use_sinkhorn=False,
            kmeans_init=False,
        )
        rvq.train()
        z = torch.randn(16, 8, requires_grad=True)
        rvq(z).quantized_embeddings.sum().backward()
        cb_grad = rvq.layers[0].embedding.weight.grad
        self.assertTrue(cb_grad is None or cb_grad.abs().sum().item() == 0.0)

    def test_ste_codebook_grad_flows_via_commitment_latents(self) -> None:
        # The codebook trains via the commitment loss, which consumes ``latents``,
        # so backward through latents MUST reach the codebook. Regression: a
        # per-layer STE wrap once detached the codebook from latents, freezing it
        # at init (commitment loss then grew unbounded while recon stayed fine).
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=2,
            n_embed=16,
            forward_mode="ste",
            use_sinkhorn=False,
            kmeans_init=False,
        )
        rvq.train()
        rvq(torch.randn(16, 8)).latents.sum().backward()
        cb_grad = rvq.layers[0].embedding.weight.grad
        self.assertIsNotNone(cb_grad)
        self.assertGreater(cb_grad.abs().sum().item(), 0.0)


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

    def test_raises_on_too_few_points(self) -> None:
        # Gained from the shared faiss_kmeans_fit primitive: a clear N>=K error
        # before faiss's opaque C++ throw.
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        with self.assertRaisesRegex(RuntimeError, "need >= 8 points"):
            faiss_residual_kmeans(torch.randn(4, 6), [8])


class ResidualVQBranchTest(unittest.TestCase):
    """Coverage for the rotation-trick STE branch and the kmeans-init guard."""

    def test_rotation_trick_rotates_gradient(self) -> None:
        # The rotation trick keeps the STE forward value but ROTATES the input
        # gradient. Plain STE also yields a finite non-zero grad, so a regression
        # that reverted the Householder branch to ordinary STE would pass a smoke
        # test. Pin the distinguishing property: same forward output, different
        # input gradient.
        def run(rotation_trick: bool):
            torch.manual_seed(0)  # identical codebook init
            rvq = ResidualVectorQuantizer(
                embed_dim=8,
                n_layers=2,
                n_embed=16,
                forward_mode="ste",
                rotation_trick=rotation_trick,
                use_sinkhorn=False,
                kmeans_init=False,
            )
            rvq.train()
            torch.manual_seed(1)  # identical input
            z = torch.randn(16, 8, requires_grad=True)
            out = rvq(z).quantized_embeddings
            out.sum().backward()
            return out.detach(), z.grad

        out_rot, grad_rot = run(True)
        out_ste, grad_ste = run(False)
        self.assertTrue(torch.isfinite(grad_rot).all())
        self.assertGreater(grad_rot.abs().sum().item(), 0.0)
        # Forward value is identical (the trick only changes the backward).
        torch.testing.assert_close(out_rot, out_ste)
        # ...but it genuinely rotates the gradient, unlike plain STE.
        self.assertFalse(torch.allclose(grad_rot, grad_ste))

    def test_kmeans_init_too_small_batch_raises(self) -> None:
        # kmeans_init needs N >= max(codebook). A too-small first batch must
        # raise a clear error (broadcast so all ranks abort together under DDP),
        # not hang the non-rank-0 ranks on the centroid broadcast.
        rvq = ResidualVectorQuantizer(
            embed_dim=4,
            n_layers=2,
            n_embed=8,
            kmeans_init=True,
            use_sinkhorn=False,
        )
        rvq.train()
        with self.assertRaisesRegex(RuntimeError, "fewer rows than the largest"):
            rvq(torch.randn(4, 4))  # 4 < max(codebook)=8


class ResidualVectorQuantizerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.rvq = ResidualVectorQuantizer(
            embed_dim=8, n_layers=3, n_embed=16, kmeans_init=False
        )

    def test_is_subclass(self) -> None:
        self.assertIsInstance(self.rvq, ResidualQuantizer)

    def test_forward_output(self) -> None:
        self.rvq.train()
        out = self.rvq(torch.randn(5, 8))
        self.assertIsInstance(out, ResidualQuantizerOutput)
        self.assertEqual(out.cluster_ids.shape, (5, 3))
        self.assertEqual(out.quantized_embeddings.shape, (5, 8))
        # latents: per-layer cumulative quantized vectors (B, n_layers, D).
        self.assertEqual(out.latents.shape, (5, 3, 8))
        self.assertTrue(torch.isfinite(out.latents).all())

    def test_decode_codes_shared_base(self) -> None:
        codes = torch.randint(0, 16, (5, 3))
        recon = self.rvq.decode_codes(codes)
        self.assertEqual(recon.shape, (5, 8))

    def test_get_codes_no_grad(self) -> None:
        codes = self.rvq.get_codes(torch.randn(4, 8))
        self.assertEqual(codes.shape, (4, 3))

    def test_forward_get_codes_consistent_eval(self) -> None:
        """get_codes (shared base walk) matches forward's ids in eval."""
        self.rvq.eval()
        x = torch.randn(6, 8)
        fwd_ids = self.rvq(x).cluster_ids
        gc_ids = self.rvq.get_codes(x)
        self.assertFalse(gc_ids.requires_grad)
        torch.testing.assert_close(gc_ids, fwd_ids)

    def test_faiss_kmeans_init_seeds_codebook(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8, n_layers=2, n_embed=16, kmeans_init=True
        )
        self.assertFalse(bool(rvq.initted.item()))
        rvq.train()
        # First training forward triggers the FAISS warm-start.
        rvq(torch.randn(512, 8))
        self.assertTrue(bool(rvq.initted.item()))
        for layer in rvq.layers:
            self.assertTrue(torch.isfinite(layer.embedding.weight).all())
            self.assertGreater(layer.embedding.weight.abs().sum().item(), 0.0)


# --- Multi-process test for ResidualVectorQuantizer FAISS kmeans-init. ---
# Validates the DDP path of ``init_embed_``: the codebook is fit on rank 0 only
# and broadcast, so every rank ends with a bit-identical warm start. (The
# previous behavior averaged permutation-misaligned per-rank centroids, which the
# review flagged as a near-random init.) Uses NCCL on GPU when >=2 devices are
# available, else gloo/CPU.

WORLD_SIZE = 2


def _init(rank: int, world_size: int, port: int) -> torch.device:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    if use_cuda:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return torch.device("cpu")


def _init_embed_worker(rank: int, world_size: int, port: int) -> None:
    device = _init(rank, world_size, port)
    # Rank-distinct data: a per-rank average/init would diverge; only a
    # broadcast-from-rank0 init yields identical codebooks.
    torch.manual_seed(rank)
    rvq = ResidualVectorQuantizer(
        embed_dim=8, n_layers=2, n_embed=16, kmeans_init=True
    ).to(device)
    rvq.train()
    rvq(torch.randn(512, 8, device=device))  # first forward triggers init_embed_
    assert bool(rvq.initted.item()), f"rank{rank}: not initialized"

    for layer in rvq.layers:
        w = layer.embedding.weight.detach().clone()
        wmin, wmax = w.clone(), w.clone()
        dist.all_reduce(wmin, op=dist.ReduceOp.MIN)
        dist.all_reduce(wmax, op=dist.ReduceOp.MAX)
        assert torch.allclose(wmin, wmax), (
            f"rank{rank}: codebook differs across ranks (init not broadcast)"
        )
    dist.destroy_process_group()


class ResidualVectorQuantizerDistTest(unittest.TestCase):
    """2-rank test for the FAISS kmeans-init broadcast."""

    def test_init_embed_broadcast(self) -> None:
        port = misc_util.get_free_port()
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(WORLD_SIZE):
            p = ctx.Process(target=_init_embed_worker, args=(rank, WORLD_SIZE, port))
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed (exitcode={p.exitcode}).")


if __name__ == "__main__":
    unittest.main()
