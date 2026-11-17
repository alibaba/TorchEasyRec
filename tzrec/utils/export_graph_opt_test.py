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
import shutil
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import torch
import torch.nn.functional as F
from parameterized import parameterized
from torch import distributed as dist
from torch import nn
from torchrec import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from tzrec.datasets.data_parser import _tile_size
from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.modules.activation import Dice
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import Perceptron
from tzrec.modules.utils import Transpose
from tzrec.protos import feature_pb2, model_pb2
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import checkpoint_util, export_util, misc_util
from tzrec.utils.export_graph_opt import (
    _infer_tile_size,
    _tzrec_eval_bn,
    _tzrec_eval_bn_dice,
    optimize_fx_for_export,
)
from tzrec.utils.export_util import (
    build_dense_graph_module,
    finalize_dense_export,
)
from tzrec.utils.fx_util import fx_mark_keyed_tensor, symbolic_trace
from tzrec.utils.test_util import (
    gpu_unavailable,
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)

torch.fx.wrap(fx_mark_keyed_tensor)
torch.fx.wrap(_tile_size)


def _count_call_function(script_module: torch.jit.ScriptModule, fn_suffix: str) -> int:
    count = 0
    for node in script_module.graph.nodes():
        if node.kind() != "prim::CallFunction":
            continue
        fn_node = node.inputsAt(0).node()
        if fn_node.kind() == "prim::Constant" and fn_node.hasAttribute("name"):
            name = fn_node.s("name")
            if name == fn_suffix or name.endswith("." + fn_suffix):
                count += 1
    return count


def _init_bn_stats(bn: nn.BatchNorm1d, seed: int = 0) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            bn.running_mean.normal_(0.0, 0.5)
            bn.running_var.uniform_(0.25, 2.0)
            if bn.weight is not None:
                bn.weight.normal_(1.0, 0.25)
            if bn.bias is not None:
                bn.bias.normal_(0.0, 0.25)


class _FactoredRsubDice(nn.Module):
    def __init__(self, hidden_size: int, dim: int = 2) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)
        self.alpha = nn.Parameter(torch.linspace(-0.2, 0.3, hidden_size))
        self.dim = dim
        _init_bn_stats(self.bn, seed=11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 3:
            x_t = x.transpose(1, 2)
            xp = torch.sigmoid(self.bn(x_t))
            one_minus_xp = torch.rsub(xp, 1.0)
            out = (xp + one_minus_xp * self.alpha.unsqueeze(1)) * x_t
            return out.transpose(1, 2)
        xp = torch.sigmoid(self.bn(x))
        one_minus_xp = torch.rsub(xp, 1.0)
        return (xp + one_minus_xp * self.alpha) * x


class _DenseTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p2 = Perceptron(
            8, 4, activation="Dice", use_bn=True, dropout_ratio=0.2, dim=2
        )
        self.p3 = Perceptron(
            6, 4, activation="Dice", use_bn=True, dropout_ratio=0.1, dim=3
        )
        self.bn2 = nn.BatchNorm1d(4)
        self.t1 = Transpose(1, 2)
        self.bn3 = nn.BatchNorm1d(4)
        self.t2 = Transpose(1, 2)
        self.id_layer = nn.Identity()
        _init_bn_stats(self.p2.perceptron[1], seed=1)
        _init_bn_stats(self.p2.perceptron[2].bn, seed=2)
        self.p2.perceptron[2].alpha.data.uniform_(-0.3, 0.3)
        _init_bn_stats(self.p3.perceptron[2], seed=3)
        _init_bn_stats(self.p3.perceptron[4].bn, seed=4)
        self.p3.perceptron[4].alpha.data.uniform_(-0.3, 0.3)
        _init_bn_stats(self.bn2, seed=5)
        _init_bn_stats(self.bn3, seed=6)

    def forward(self, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        h2 = self.bn2(self.p2(x2))
        h3 = self.t2(self.bn3(self.t1(self.p3(x3))))
        return self.id_layer(h2) + h3.sum(dim=1)


class ExportGraphOptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def test_pass1_identities_and_regroup_preservation(self) -> None:
        class RegroupModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.drop = nn.Dropout(0.3)
                self.drop1d = nn.Dropout1d(0.2)
                self.ident = nn.Identity()
                self.lin = nn.Linear(6, 4)

            def forward(
                self, v1: torch.Tensor, v2: torch.Tensor
            ) -> Dict[str, torch.Tensor]:
                kt1 = KeyedTensor(keys=["a", "b"], length_per_key=[2, 2], values=v1)
                kt2 = KeyedTensor(keys=["c"], length_per_key=[2], values=v2)
                grouped = KeyedTensor.regroup_as_dict(
                    [kt1, kt2], [["a", "c"], ["b"]], ["g_ac", "g_b"]
                )
                h = self.drop1d(self.ident(self.drop(grouped["g_ac"])))
                h = F.dropout(h, p=0.5, training=self.training)
                return {"out": self.lin(torch.cat([h, grouped["g_b"]], dim=-1))}

        mod = RegroupModule().eval()
        gm = symbolic_trace(mod)
        opt_gm = optimize_fx_for_export(gm, emit_jit_anchors=True)
        self.assertNotIn("Dropout", opt_gm.code)
        self.assertNotIn("Identity", opt_gm.code)
        self.assertNotIn("dropout", opt_gm.code)
        self.assertNotIn("_all_keys_used_once", opt_gm.code)
        self.assertIn("_regroup_keyed_tensors", opt_gm.code)

        scripted = torch.jit.script(opt_gm)
        self.assertEqual(_count_call_function(scripted, "_regroup_keyed_tensors"), 1)
        v1 = torch.randn(3, 4)
        v2 = torch.randn(3, 2)
        torch.testing.assert_close(
            scripted(v1, v2)["out"], mod(v1, v2)["out"], atol=1e-5, rtol=1e-5
        )

    def test_pass1_skips_training_dropout(self) -> None:
        class TrainDrop(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.drop = nn.Dropout(0.5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.drop(x)

        mod = TrainDrop().train()
        gm = symbolic_trace(mod)
        opt_gm = optimize_fx_for_export(gm)
        self.assertIn("self.drop", opt_gm.code)
        self.assertTrue(mod.drop.training)

    def test_pass2_linear_bn_2d_and_3d_including_leaf_transpose_and_no_bias(
        self,
    ) -> None:
        class LeafTransposeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin2 = nn.Linear(5, 4, bias=False)
                self.bn2 = nn.BatchNorm1d(4)
                self.lin2_aff_false = nn.Linear(4, 4, bias=True)
                self.bn2_aff_false = nn.BatchNorm1d(4, affine=False)
                self.lin3 = nn.Linear(5, 4)
                self.t1 = Transpose(1, 2)
                self.bn3 = nn.BatchNorm1d(4)
                self.t2 = Transpose(2, 1)
                _init_bn_stats(self.bn2, seed=21)
                _init_bn_stats(self.bn2_aff_false, seed=22)
                _init_bn_stats(self.bn3, seed=23)

            def forward(
                self, x2: torch.Tensor, x3: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                y2 = self.bn2_aff_false(self.lin2_aff_false(self.bn2(self.lin2(x2))))
                y3 = self.t2(self.bn3(self.t1(self.lin3(x3))))
                return y2, y3

        mod = LeafTransposeModel().eval()
        # Trace with Transpose kept as leaf module to test call_module Transpose path.
        gm = symbolic_trace(mod, leaf_modules=["Transpose"])
        self.assertIn("self.t1", gm.code)
        self.assertIn("self.bn3", gm.code)

        opt_gm = optimize_fx_for_export(gm, emit_jit_anchors=True)
        self.assertNotIn("BatchNorm1d", opt_gm.code)
        self.assertNotIn("transpose", opt_gm.code)
        self.assertNotIn("self.t1", opt_gm.code)
        self.assertNotIn("self.t2", opt_gm.code)
        self.assertNotIn("_tzrec_eval_bn", opt_gm.code)

        x2 = torch.randn(3, 5)
        x3 = torch.randn(3, 4, 5)
        ref2, ref3 = mod(x2, x3)
        out2, out3 = opt_gm(x2, x3)
        torch.testing.assert_close(out2, ref2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out3, ref3, atol=1e-5, rtol=1e-5)

        scripted = torch.jit.script(opt_gm)
        s2, s3 = scripted(x2, x3)
        torch.testing.assert_close(s2, ref2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s3, ref3, atol=1e-5, rtol=1e-5)

    def test_pass3_dice_variants_and_skip_connection_guard(self) -> None:
        class DiceGallery(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d2 = Dice(4, dim=2)
                self.d3 = Dice(4, dim=3)
                self.rsub2 = _FactoredRsubDice(4, dim=2)
                self.rsub3 = _FactoredRsubDice(4, dim=3)
                self.d3_shared = Dice(4, dim=3)
                _init_bn_stats(self.d2.bn, seed=31)
                _init_bn_stats(self.d3.bn, seed=32)
                _init_bn_stats(self.d3_shared.bn, seed=33)
                self.d2.alpha.data.uniform_(-0.3, 0.3)
                self.d3.alpha.data.uniform_(-0.3, 0.3)
                self.d3_shared.alpha.data.uniform_(-0.3, 0.3)

            def forward(
                self, x2: torch.Tensor, x3: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                a2 = self.rsub2(self.d2(x2))
                a3 = self.rsub3(self.d3(x3))
                # Shared transpose cannot drop t1 since skip branch consumes x_t.
                x_t = x3.transpose(1, 2)
                xp = torch.sigmoid(self.d3_shared.bn(x_t))
                out_t = self.d3_shared.alpha.unsqueeze(1) * (1 - xp) * x_t + xp * x_t
                skip = (out_t + x_t).transpose(1, 2)
                return a2, a3, skip

        mod = DiceGallery().eval()
        gm = symbolic_trace(mod)
        opt_gm = optimize_fx_for_export(gm, emit_jit_anchors=True)
        self.assertNotIn("BatchNorm1d", opt_gm.code)
        self.assertNotIn("sigmoid", opt_gm.code)

        # Verify anchor args (5 explicit positional args -> 6 TorchScript inputs).
        dice_nodes = [
            n
            for n in opt_gm.graph.nodes
            if n.op == "call_function" and n.target is _tzrec_eval_bn_dice
        ]
        self.assertEqual(len(dice_nodes), 5)
        for node in dice_nodes:
            self.assertEqual(len(node.args), 5)
            self.assertEqual(node.kwargs, {})
        channel_last_flags = [node.args[4] for node in dice_nodes]
        # d2=False, rsub2=False, d3=True, rsub3=True, d3_shared=False
        self.assertEqual(channel_last_flags, [False, False, True, True, False])

        # Verify buffers are 1D FP32 contiguous non-trainable.
        for name, buf in opt_gm.named_buffers():
            self.assertEqual(buf.dtype, torch.float32, name)
            self.assertEqual(buf.dim(), 1, name)
            self.assertTrue(buf.is_contiguous(), name)
            self.assertFalse(buf.requires_grad, name)

        scripted = torch.jit.script(opt_gm)
        self.assertEqual(_count_call_function(scripted, "_tzrec_eval_bn_dice"), 5)
        for node in scripted.graph.nodes():
            if node.kind() == "prim::CallFunction":
                self.assertEqual(len(list(node.inputs())), 6)

        x2 = torch.randn(3, 4)
        x3 = torch.randn(3, 5, 4)
        ref = mod(x2, x3)
        out = scripted(x2, x3)
        for r, o in zip(ref, out):
            torch.testing.assert_close(o, r, atol=1e-5, rtol=1e-5)

    def test_pass4_standalone_bn_2d_3d_and_non_paired_3d(self) -> None:
        class StandaloneBnModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn2 = nn.BatchNorm1d(4)
                self.bn2_noaff = nn.BatchNorm1d(4, affine=False)
                self.bn3_paired = nn.BatchNorm1d(4)
                self.bn3_raw = nn.BatchNorm1d(4)
                _init_bn_stats(self.bn2, seed=41)
                _init_bn_stats(self.bn2_noaff, seed=42)
                _init_bn_stats(self.bn3_paired, seed=43)
                _init_bn_stats(self.bn3_raw, seed=44)

            def forward(
                self, x2: torch.Tensor, x3: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                y2 = self.bn2_noaff(self.bn2(x2))
                y3_paired = self.bn3_paired(x3.transpose(1, 2)).transpose(1, 2)
                y3_raw = self.bn3_raw(x3.transpose(1, 2))
                return y2, y3_paired, y3_raw

        mod = StandaloneBnModel().eval()
        gm = symbolic_trace(mod)
        opt_gm = optimize_fx_for_export(gm, emit_jit_anchors=True)
        bn_nodes = [
            n
            for n in opt_gm.graph.nodes
            if n.op == "call_function" and n.target is _tzrec_eval_bn
        ]
        self.assertEqual(len(bn_nodes), 4)
        for node in bn_nodes:
            self.assertEqual(len(node.args), 4)
            self.assertEqual(node.kwargs, {})
        self.assertEqual([n.args[3] for n in bn_nodes], [False, False, True, False])

        scripted = torch.jit.script(opt_gm)
        self.assertEqual(_count_call_function(scripted, "_tzrec_eval_bn"), 4)
        for node in scripted.graph.nodes():
            if node.kind() == "prim::CallFunction":
                self.assertEqual(len(list(node.inputs())), 5)

        x2 = torch.randn(3, 4)
        x3 = torch.randn(3, 5, 4)
        for r, o in zip(mod(x2, x3), scripted(x2, x3)):
            torch.testing.assert_close(o, r, atol=1e-5, rtol=1e-5)

    @parameterized.expand(
        [
            ("cpu_fp32_jit", "cpu", torch.float32, True),
            ("cpu_fp32_aten", "cpu", torch.float32, False),
            ("cpu_bf16_jit", "cpu", torch.bfloat16, True),
            ("cpu_bf16_aten", "cpu", torch.bfloat16, False),
        ],
        name_func=parameterized_name_func,
    )
    def test_cpu_dtypes_and_emit_jit_anchors_modes(
        self, _name: str, device_str: str, dtype: torch.dtype, emit_jit_anchors: bool
    ) -> None:
        self._assert_dense_tower_numerics(
            torch.device(device_str), dtype, emit_jit_anchors
        )

    @parameterized.expand(
        [
            ("cuda_fp32_jit", torch.float32, True),
            ("cuda_fp32_aten", torch.float32, False),
            ("cuda_fp16_jit", torch.float16, True),
            ("cuda_fp16_aten", torch.float16, False),
            ("cuda_bf16_jit", torch.bfloat16, True),
            ("cuda_bf16_aten", torch.bfloat16, False),
        ],
        name_func=parameterized_name_func,
    )
    @unittest.skipIf(*gpu_unavailable)
    @mark_ci_scope("gpu")
    def test_cuda_dtypes_and_emit_jit_anchors_modes(
        self, _name: str, dtype: torch.dtype, emit_jit_anchors: bool
    ) -> None:
        self._assert_dense_tower_numerics(
            torch.device("cuda:0"), dtype, emit_jit_anchors
        )

    def _assert_dense_tower_numerics(
        self, device: torch.device, dtype: torch.dtype, emit_jit_anchors: bool
    ) -> None:
        mod = _DenseTower().to(device=device, dtype=dtype).eval()
        gm = symbolic_trace(mod)
        opt_gm = optimize_fx_for_export(gm, emit_jit_anchors=emit_jit_anchors)

        self.assertNotIn("BatchNorm1d", opt_gm.code)
        self.assertNotIn("Dropout", opt_gm.code)
        if emit_jit_anchors:
            self.assertIn("_tzrec_eval_bn_dice", opt_gm.code)
            self.assertIn("_tzrec_eval_bn", opt_gm.code)
        else:
            self.assertNotIn("_tzrec_eval_bn_dice", opt_gm.code)
            self.assertNotIn("_tzrec_eval_bn", opt_gm.code)
            self.assertNotIn("_infer_tile_size", opt_gm.code)

        for name, buf in opt_gm.named_buffers():
            self.assertEqual(buf.dtype, torch.float32, name)
            self.assertEqual(buf.dim(), 1, name)
            self.assertEqual(buf.device, device, name)
            self.assertTrue(buf.is_contiguous(), name)
            self.assertFalse(buf.requires_grad, name)

        x2 = torch.randn(4, 8, device=device, dtype=dtype)
        x3 = torch.randn(4, 5, 6, device=device, dtype=dtype)
        ref = mod(x2, x3)
        out = opt_gm(x2, x3)
        self.assertEqual(out.dtype, dtype)

        scripted = torch.jit.script(opt_gm)
        s_out = scripted(x2, x3)
        self.assertEqual(s_out.dtype, dtype)

        tol = (
            (1e-5, 1e-5)
            if dtype == torch.float32
            else ((5e-3, 5e-3) if dtype == torch.float16 else (2e-2, 2e-2))
        )
        torch.testing.assert_close(out, ref, atol=tol[0], rtol=tol[1])
        torch.testing.assert_close(s_out, ref, atol=tol[0], rtol=tol[1])

    def test_pass5_infer_tile_size_all_boundary_cases_and_fallback(self) -> None:
        class TileModel(nn.Module):
            def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
                # Batch size read before user/item keys, matching DataParser order.
                ts = _tile_size(data["batch_size"])
                u = data["u__ebc_user"].tile(ts, 1)
                it = data["it__ebc"]
                return u + it

        gm = symbolic_trace(TileModel().eval())
        opt_gm = optimize_fx_for_export(
            gm,
            emit_jit_anchors=True,
            user_input_keys={"u__ebc_user"},
            item_input_keys={"it__ebc"},
        )
        self.assertNotIn("data_parser__tile_size", opt_gm.code)
        self.assertFalse(any(n.target is _tile_size for n in opt_gm.graph.nodes))
        self.assertTrue(any(n.target is _infer_tile_size for n in opt_gm.graph.nodes))
        self.assertNotIn('"batch_size"', opt_gm.code)
        self.assertIn("_infer_tile_size", opt_gm.code)

        scripted = torch.jit.script(opt_gm)
        self.assertEqual(_count_call_function(scripted, "_infer_tile_size"), 1)
        self.assertNotIn("aten::_local_scalar_dense", str(scripted.graph))

        cases = [
            ("online_1_vs_B", 1, 4, 4),
            ("online_1_vs_1", 1, 1, 1),
            ("online_1_vs_0", 1, 0, 0),
            ("offline_B_vs_B", 4, 4, 1),
        ]
        for label, u_rows, it_rows, expected_ts in cases:
            with self.subTest(label=label):
                u = torch.randn(u_rows, 3)
                it = torch.randn(it_rows, 3)
                out = scripted({"u__ebc_user": u, "it__ebc": it})
                expected = u.tile(expected_ts, 1) + it
                torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

        # Fallback: missing user or item keys leaves _tile_size untouched.
        opt_fallback = optimize_fx_for_export(
            gm,
            emit_jit_anchors=True,
            user_input_keys={"u__ebc_user"},
            item_input_keys=set(),
        )
        self.assertIn("_tile_size", opt_fallback.code)
        self.assertNotIn("_infer_tile_size", opt_fallback.code)

    @unittest.skipIf(*gpu_unavailable)
    @mark_ci_scope("gpu")
    def test_pass5_infer_tile_size_cuda_zero_sync(self) -> None:
        class TileModel(nn.Module):
            def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
                ts = _tile_size(data["batch_size"])
                return data["u"].tile(ts, 1) + data["it"]

        gm = symbolic_trace(TileModel().eval())
        opt_gm = optimize_fx_for_export(
            gm, user_input_keys={"u"}, item_input_keys={"it"}
        )
        scripted = torch.jit.script(opt_gm).to("cuda:0")
        for u_rows, it_rows, exp_ts in [(1, 4, 4), (1, 1, 1), (1, 0, 0), (4, 4, 1)]:
            u = torch.randn(u_rows, 3, device="cuda:0")
            it = torch.randn(it_rows, 3, device="cuda:0")
            torch.testing.assert_close(
                scripted({"u": u, "it": it}), u.tile(exp_ts, 1) + it
            )

    def test_disable_env_var_returns_original_gm(self) -> None:
        mod = _DenseTower().eval()
        gm = symbolic_trace(mod)
        with mock.patch.dict(os.environ, {"TZREC_DISABLE_EXPORT_GRAPH_OPT": "1"}):
            opt_gm = optimize_fx_for_export(gm)
        self.assertIs(opt_gm, gm)

    def test_non_mutation_and_multi_round_online_dense_export(self) -> None:
        tables = [
            EmbeddingBagConfig(
                name=f"{name}_emb",
                embedding_dim=4,
                num_embeddings=8,
                feature_names=[name],
            )
            for name in ("item_a", "item_b")
        ]
        user_tables = [
            EmbeddingBagConfig(
                name="user_a_emb",
                embedding_dim=4,
                num_embeddings=8,
                feature_names=["user_a"],
            )
        ]

        class DenseOptServingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features: List[Any] = []
                self.ebc = EmbeddingBagCollection(tables, device=torch.device("cpu"))
                self.ebc_user = EmbeddingBagCollection(
                    user_tables, device=torch.device("cpu")
                )
                self.p2 = Perceptron(8, 4, activation="Dice", use_bn=True, dim=2)
                self.p3 = Perceptron(
                    4, 4, activation="Dice", use_bn=True, dropout_ratio=0.1, dim=3
                )
                self.t1 = Transpose(1, 2)
                self.bn_solo = nn.BatchNorm1d(4)
                self.t2 = Transpose(1, 2)
                self.out_lin = nn.Linear(4, 1)
                _init_bn_stats(self.p2.perceptron[1], seed=101)
                _init_bn_stats(self.p2.perceptron[2].bn, seed=102)
                self.p2.perceptron[2].alpha.data.fill_(0.15)
                _init_bn_stats(self.p3.perceptron[2], seed=103)
                _init_bn_stats(self.p3.perceptron[4].bn, seed=104)
                self.p3.perceptron[4].alpha.data.fill_(-0.2)
                _init_bn_stats(self.bn_solo, seed=105)

            def set_is_inference(self, is_inference: bool) -> None:
                self.is_inference = is_inference

            def forward(
                self,
                data: Dict[str, torch.Tensor],
                device: Optional[torch.device] = None,
            ) -> Dict[str, torch.Tensor]:
                kt_item = self.ebc(data["x"])
                fx_mark_keyed_tensor("grp__ebc", kt_item)
                raw_user = self.ebc_user(data["x_user"])
                kt_user = KeyedTensor(
                    keys=raw_user.keys(),
                    length_per_key=raw_user.length_per_key(),
                    values=raw_user.values().tile(2, 1),
                )
                fx_mark_keyed_tensor("grp__ebc_user", kt_user)
                grouped = KeyedTensor.regroup_as_dict(
                    [kt_item, kt_user],
                    [["item_a", "user_a"], ["item_b"]],
                    ["g2", "g3"],
                )
                h2 = self.p2(grouped["g2"])
                h3_in = grouped["g3"].unsqueeze(1).expand(-1, 3, -1)
                h3 = self.t2(self.bn_solo(self.t1(self.p3(h3_in)))).mean(dim=1)
                return {"score": self.out_lin(h2 + h3).squeeze(-1)}

        warmup_data = {
            "x": KeyedJaggedTensor.from_lengths_sync(
                keys=["item_a", "item_b"],
                values=torch.zeros(4, dtype=torch.int64),
                lengths=torch.ones(4, dtype=torch.int64),
            ),
            "x_user": KeyedJaggedTensor.from_lengths_sync(
                keys=["user_a"],
                values=torch.zeros(1, dtype=torch.int64),
                lengths=torch.ones(1, dtype=torch.int64),
            ),
            "batch_size": torch.tensor(2),
        }

        model = DenseOptServingModel().eval()
        gm, full_graph, dense_graph_config = build_dense_graph_module(
            model, warmup_data, torch.device("cpu")
        )
        # Keep requires_grad=True on one linear to prove non-mutation of gm flags.
        gm.out_lin.weight.requires_grad_(True)
        orig_gm_keys = list(gm.state_dict().keys())
        orig_req_grad = {k: p.requires_grad for k, p in gm.named_parameters()}

        round1_dir = os.path.join(self.test_dir, "round1")
        round2_dir = os.path.join(self.test_dir, "round2")
        dense_model_traced = finalize_dense_export(
            model,
            full_graph,
            gm,
            warmup_data,
            torch.device("cpu"),
            round1_dir,
            dense_graph_config,
        )
        orig_traced_keys = list(dense_model_traced.state_dict().keys())
        orig_traced_code = dense_model_traced.code

        self.assertEqual(list(gm.state_dict().keys()), orig_gm_keys)
        self.assertEqual(
            {k: p.requires_grad for k, p in gm.named_parameters()}, orig_req_grad
        )

        # Round 2: update weights in place and forbid symbolic_trace on reuse path.
        snap2 = {
            k: (v + 0.5 if v.is_floating_point() else v.clone())
            for k, v in gm.state_dict().items()
        }
        gm.load_state_dict(snap2)
        with mock.patch(
            "tzrec.utils.export_util.symbolic_trace",
            side_effect=AssertionError("reuse path must not call symbolic_trace"),
        ):
            returned_traced = finalize_dense_export(
                model,
                full_graph,
                gm,
                warmup_data,
                torch.device("cpu"),
                round2_dir,
                dense_graph_config,
                dense_model_traced=dense_model_traced,
            )

        self.assertIs(returned_traced, dense_model_traced)
        self.assertEqual(list(gm.state_dict().keys()), orig_gm_keys)
        self.assertEqual(list(dense_model_traced.state_dict().keys()), orig_traced_keys)
        self.assertEqual(dense_model_traced.code, orig_traced_code)
        self.assertEqual(
            {k: p.requires_grad for k, p in gm.named_parameters()}, orig_req_grad
        )

        # Verify IR invariants in gm_dense_opt.code and scripted_model.pt.
        with open(os.path.join(round2_dir, "gm_dense_opt.code")) as f:
            opt_code = f.read()
        self.assertNotIn("BatchNorm1d", opt_code)
        self.assertNotIn("Dropout", opt_code)
        self.assertNotIn("transpose(1, 2)", opt_code)
        self.assertNotIn("data_parser__tile_size", opt_code)
        self.assertIn("_regroup_keyed_tensors", opt_code)
        self.assertIn("_tzrec_eval_bn_dice", opt_code)
        self.assertIn("_tzrec_eval_bn", opt_code)
        self.assertIn("_infer_tile_size", opt_code)

        s1 = torch.jit.load(os.path.join(round1_dir, "scripted_model.pt"))
        s2 = torch.jit.load(os.path.join(round2_dir, "scripted_model.pt"))
        self.assertEqual(_count_call_function(s2, "_regroup_keyed_tensors"), 1)
        self.assertEqual(_count_call_function(s2, "_tzrec_eval_bn_dice"), 2)
        self.assertEqual(_count_call_function(s2, "_tzrec_eval_bn"), 1)
        self.assertEqual(_count_call_function(s2, "_infer_tile_size"), 1)
        self.assertNotIn("aten::batch_norm", str(s2.graph))
        self.assertNotIn("aten::_local_scalar_dense", str(s2.graph))

        online_in = {
            "grp__ebc": torch.randn(3, 8),
            "grp__ebc_user": torch.randn(1, 4),
            "batch_size": torch.tensor(3),
        }
        offline_in = {
            "grp__ebc": online_in["grp__ebc"],
            "grp__ebc_user": online_in["grp__ebc_user"].tile(3, 1),
            "batch_size": torch.tensor(1),
        }
        with torch.no_grad():
            ref2_out = gm(online_in)["score"]
        s1_out = s1(online_in)["score"]
        s2_out = s2(online_in)["score"]
        s2_offline_out = s2(offline_in)["score"]

        self.assertFalse(torch.allclose(s1_out, s2_out))
        torch.testing.assert_close(s2_out, ref2_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s2_offline_out, ref2_out, atol=1e-5, rtol=1e-5)

    def test_export_model_normal_input_tile_wiring(self) -> None:
        feat_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    embedding_dim=4,
                    num_buckets=8,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="item_id",
                    expression="item:item_id",
                    embedding_dim=4,
                    num_buckets=8,
                )
            ),
        ]
        features = create_features(feat_cfgs)
        fg_cfg = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["user_id", "item_id"],
                group_type=model_pb2.DEEP,
            )
        ]

        class TinyNormalTileModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = features
                self.feature_groups = fg_cfg
                self.embedding_group = EmbeddingGroup(
                    features, fg_cfg, device=torch.device("cpu")
                )
                self.mlp = Perceptron(8, 4, activation="Dice", use_bn=True, dim=2)
                self.head = nn.Linear(4, 1)
                _init_bn_stats(self.mlp.perceptron[1], seed=201)
                _init_bn_stats(self.mlp.perceptron[2].bn, seed=202)

            def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
                grouped = self.embedding_group(batch)
                return {"score": self.head(self.mlp(grouped["deep"])).squeeze(-1)}

        with mock.patch.dict(
            os.environ,
            {
                "INPUT_TILE": "3",
                "USE_DISTRIBUTED_EMBEDDING": "0",
                "USE_RTP": "0",
                "QUANT_EMB": "0",
                "QUANT_EC_EMB": "0",
                "ENABLE_AOT": "0",
                "ENABLE_TRT": "0",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
        ):
            from tzrec.models.model import ScriptWrapper

            wrapped = ScriptWrapper(TinyNormalTileModel()).eval()
            online_in = {
                "user_id.values": torch.tensor([1], dtype=torch.int64),
                "user_id.lengths": torch.tensor([1], dtype=torch.int64),
                "item_id.values": torch.tensor([2, 3, 4], dtype=torch.int64),
                "item_id.lengths": torch.tensor([1, 1, 1], dtype=torch.int64),
                "batch_size": torch.tensor(3),
            }
            offline_in = {
                "user_id.values": torch.tensor([1, 1, 1], dtype=torch.int64),
                "user_id.lengths": torch.tensor([1, 1, 1], dtype=torch.int64),
                "item_id.values": torch.tensor([2, 3, 4], dtype=torch.int64),
                "item_id.lengths": torch.tensor([1, 1, 1], dtype=torch.int64),
                "batch_size": torch.tensor(1),
            }
            batch = Batch(
                dense_features={},
                sparse_features={
                    BASE_DATA_GROUP + "_user": KeyedJaggedTensor.from_lengths_sync(
                        keys=["user_id"],
                        values=torch.tensor([1], dtype=torch.int64),
                        lengths=torch.tensor([1], dtype=torch.int64),
                    ),
                    BASE_DATA_GROUP: KeyedJaggedTensor.from_lengths_sync(
                        keys=["item_id"],
                        values=torch.tensor([2, 3, 4], dtype=torch.int64),
                        lengths=torch.tensor([1, 1, 1], dtype=torch.int64),
                    ),
                },
                labels={},
                tile_size=3,
            )

            class FakeLoader:
                def get_iterator(self) -> Any:
                    return iter([batch])

            ckpt_dir = os.path.join(self.test_dir, "ckpt")
            save_dir = os.path.join(self.test_dir, "normal_export")
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg = EasyRecConfig(train_input_path="unused")
            port = misc_util.get_free_port()
            dist.init_process_group(
                "gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
            self.addCleanup(dist.destroy_process_group)
            with (
                mock.patch.object(
                    export_util, "create_dataloader", return_value=FakeLoader()
                ),
                mock.patch.object(checkpoint_util, "restore_model"),
            ):
                export_util.export_model_normal(cfg, wrapped, ckpt_dir, save_dir)

            with open(os.path.join(save_dir, "gm_opt.code")) as f:
                opt_code = f.read()
            self.assertNotIn("data_parser__tile_size", opt_code)
            self.assertNotIn("BatchNorm1d", opt_code)
            self.assertIn("_infer_tile_size", opt_code)
            self.assertIn("_tzrec_eval_bn_dice", opt_code)

            scripted = torch.jit.load(os.path.join(save_dir, "scripted_model.pt"))
            self.assertNotIn("aten::_local_scalar_dense", str(scripted.graph))
            torch.testing.assert_close(
                scripted(online_in)["score"],
                wrapped(online_in)["score"],
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                scripted(offline_in)["score"],
                wrapped(online_in)["score"],
                atol=1e-5,
                rtol=1e-5,
            )

    def test_torch_transpose_torch_export_training_guard_and_key_safety(self) -> None:
        class FnTransposeAndTrainBn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(5, 4)
                self.bn_eval = nn.BatchNorm1d(4)
                self.bn_train = nn.BatchNorm1d(4)
                _init_bn_stats(self.bn_eval, seed=301)
                _init_bn_stats(self.bn_train, seed=302)
                self.bn_eval.eval()
                self.bn_train.train()

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                h_eval = torch.transpose(
                    self.bn_eval(torch.transpose(self.lin(x), 1, 2)), 2, 1
                )
                h_train = self.bn_train(h_eval.transpose(1, 2)).transpose(1, 2)
                return h_eval, h_train

        mod = FnTransposeAndTrainBn()
        gm = symbolic_trace(mod)
        opt_gm = optimize_fx_for_export(gm, emit_jit_anchors=False)
        self.assertIn("self.bn_train", opt_gm.code)
        self.assertNotIn("self.bn_eval", opt_gm.code)
        self.assertTrue(mod.bn_train.training)

        # Standard ATen emit_jit_anchors=False tower compiles via torch.export.export.
        tower = _DenseTower().eval()
        tower_opt = optimize_fx_for_export(
            symbolic_trace(tower), emit_jit_anchors=False
        )
        x2 = torch.randn(4, 8)
        x3 = torch.randn(4, 5, 6)
        ep = torch.export.export(tower_opt, (x2, x3))
        torch.testing.assert_close(
            ep.module()(x2, x3), tower(x2, x3), atol=1e-5, rtol=1e-5
        )

        # Unconfigured RawFeature (no expression) must never be misclassified.
        ambiguous_feats = create_features(
            [
                feature_pb2.FeatureConfig(
                    raw_feature=feature_pb2.RawFeature(feature_name="raw_x")
                )
            ]
        )
        with mock.patch.dict(os.environ, {"INPUT_TILE": "3"}):
            u_keys, it_keys = export_util._collect_normal_export_tile_input_keys(
                ambiguous_feats
            )
            self.assertEqual(u_keys, set())
            self.assertEqual(it_keys, set())


if __name__ == "__main__":
    unittest.main()
