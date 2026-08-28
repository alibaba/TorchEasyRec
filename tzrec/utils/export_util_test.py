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


import copy
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import distributed as dist
from torchrec import KeyedJaggedTensor, KeyedTensor
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.utils import Tracer
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    LFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)

from tzrec.acc import utils as acc_utils
from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.deepfm import DeepFM
from tzrec.models.model import ScriptWrapper
from tzrec.modules.dense_embedding_collection import (
    AutoDisEmbeddingConfig,
    DenseEmbeddingCollection,
    MLPDenseEmbeddingConfig,
)
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import rank_model_pb2
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import checkpoint_util, config_util, export_util, misc_util
from tzrec.utils.export_util import (
    _add_module_by_dotted_path,
    _canonicalize_keyed_tensor_attrs,
    _dedup_key_files_by_realpath,
    _get_dense_embedding_leaf_module_names,
    _get_embedding_bag_configs,
    _get_sparse_embedding_tensor,
    _infer_keyed_tensor_attrs_from_module,
    _isolate_kafka_export_group,
    _merge_sharded_embedding_json,
    _permute_keyed_tensor_values,
    _prepare_single_rank_distributed_embedding_export,
    _prune_unused_param_and_buffer,
    _shrink_sparse_embedding_tables,
    build_dense_graph_module,
    create_dense_export_warmup_data,
    export_dense_model_cpu,
    export_distributed_embedding,
    finalize_dense_export,
)
from tzrec.utils.fx_util import fx_mark_keyed_tensor
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import make_test_dir

# register the mark for fx tracing from this module's call sites, as
# tzrec/modules/embedding.py does for its own
torch.fx.wrap(fx_mark_keyed_tensor)


def _restore_env(old_env):
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _dequant_quint8_rowwise_f16(values: np.ndarray, emb_dim: int) -> np.ndarray:
    q = values[:, :emb_dim].astype(np.float32)
    scale = np.ascontiguousarray(values[:, emb_dim : emb_dim + 2]).view(np.float16)
    offset = np.ascontiguousarray(values[:, emb_dim + 2 : emb_dim + 4]).view(np.float16)
    dequant = q * scale.astype(np.float32).reshape(-1, 1)
    dequant += offset.astype(np.float32).reshape(-1, 1)
    return dequant.astype(np.float16).astype(np.float32)


class ExportUtilTest(unittest.TestCase):
    def test_distributed_sparse_quant_env(self) -> None:
        old_env = {
            "DIST_QUANT": os.environ.get("DIST_QUANT"),
            "QUANT": os.environ.get("QUANT"),
            "USE_DISTRIBUTED_EMBEDDING": os.environ.get("USE_DISTRIBUTED_EMBEDDING"),
        }
        try:
            os.environ.pop("USE_DISTRIBUTED_EMBEDDING", None)
            os.environ["QUANT"] = "INT8"
            os.environ.pop("DIST_QUANT", None)
            self.assertFalse(acc_utils.is_distributed_sparse_quant())
            acc_config = acc_utils.export_acc_config()
            self.assertNotIn("DIST_QUANT", acc_config)
            self.assertNotIn("QUANT", acc_config)
            os.environ.pop("QUANT", None)

            for value in (None, "", "0", "NONE", "none"):
                if value is None:
                    os.environ.pop("DIST_QUANT", None)
                else:
                    os.environ["DIST_QUANT"] = value
                self.assertFalse(acc_utils.is_distributed_sparse_quant())
                self.assertEqual(acc_utils.distributed_sparse_quant_format(), "")
                self.assertNotIn("DIST_QUANT", acc_utils.export_acc_config())

            os.environ["DIST_QUANT"] = "INT8"
            self.assertTrue(acc_utils.is_distributed_sparse_quant())
            self.assertEqual(
                acc_utils.distributed_sparse_quant_format(), "QUint8RowwiseF16"
            )
            self.assertNotIn("DIST_QUANT", acc_utils.export_acc_config())

            os.environ["USE_DISTRIBUTED_EMBEDDING"] = "1"
            self.assertEqual(acc_utils.export_acc_config()["DIST_QUANT"], "INT8")

            os.environ["DIST_QUANT"] = "FP16"
            with self.assertRaisesRegex(ValueError, "Unsupported DIST_QUANT"):
                acc_utils.is_distributed_sparse_quant()
        finally:
            _restore_env(old_env)

    def test_dedup_key_files_by_realpath_preserves_first_physical_file(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dedup_key_files_")
        try:
            real_dir = os.path.join(tmp, "real")
            alias_dir = os.path.join(tmp, "alias")
            other_dir = os.path.join(tmp, "other")
            os.makedirs(real_dir)
            os.makedirs(alias_dir)
            os.makedirs(other_dir)

            key_file = os.path.join(real_dir, "table_emb_keys.rank_0.world_size_1")
            alias_file = os.path.join(alias_dir, "table_emb_keys.rank_0.world_size_1")
            other_file = os.path.join(other_dir, "table_emb_keys.rank_0.world_size_1")
            with open(key_file, "wb") as f:
                f.write(b"key")
            os.symlink(key_file, alias_file)
            with open(other_file, "wb") as f:
                f.write(b"other")

            self.assertEqual(
                _dedup_key_files_by_realpath([alias_file, key_file, other_file]),
                [alias_file, other_file],
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_distributed_embedding_export_forces_rank_zero_single_process(self) -> None:
        """Rank 0 export should be normalized to a single logical GPU."""
        old_env = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        }
        try:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "2"
            os.environ["WORLD_SIZE"] = "4"
            os.environ["LOCAL_WORLD_SIZE"] = "4"

            self.assertTrue(_prepare_single_rank_distributed_embedding_export())
            self.assertEqual(os.environ["RANK"], "0")
            self.assertEqual(os.environ["LOCAL_RANK"], "0")
            self.assertEqual(os.environ["WORLD_SIZE"], "1")
            self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "1")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_distributed_embedding_export_skips_nonzero_rank_before_pg_init(
        self,
    ) -> None:
        """Non-zero ranks should exit before creating a process group."""
        old_env = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        }
        try:
            os.environ["RANK"] = "1"
            os.environ["LOCAL_RANK"] = "1"
            os.environ["WORLD_SIZE"] = "2"
            os.environ["LOCAL_WORLD_SIZE"] = "2"

            with mock.patch("tzrec.utils.export_util.init_process_group") as init_pg:
                export_distributed_embedding(None, None, None, "/tmp/unused_export")
                init_pg.assert_not_called()
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_distributed_embedding_export_uses_export_overrides(self) -> None:
        class FakeBatch:
            def to(self, device):  # type: ignore[no-untyped-def]
                return self

            def to_dict(self, sparse_dtype):  # type: ignore[no-untyped-def]
                return {"x": torch.ones(1)}

        class FakeDataloader:
            dataset = SimpleNamespace(sampled_batch_size=1)

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter([FakeBatch()])

        class TinyModel(torch.nn.Module):
            def __init__(self):  # type: ignore[no-untyped-def]
                super().__init__()
                self.features = []

            def set_is_inference(self, is_inference):  # type: ignore[no-untyped-def]
                self.is_inference = is_inference

            def forward(self, data, device=None):  # type: ignore[no-untyped-def]
                return {"score": data["x"] + 1}

        class FakeDMP(torch.nn.Module):
            def __init__(self, module, *args, **kwargs):  # type: ignore[no-untyped-def]
                super().__init__()
                self.module = module

            def forward(self, data, device=None):  # type: ignore[no-untyped-def]
                return self.module(data, device=device)

        tmp = tempfile.mkdtemp(prefix="tzrec_export_dist_overrides_")
        old_env = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        }
        try:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            pipeline_config = EasyRecConfig(
                train_input_path="train_input",
                eval_input_path="eval_input",
                model_dir="model_dir",
            )
            model_acc = {"SPARSE_INT64": "1", "cand_seq_pk": "cand_seq"}
            fake_scripted = mock.Mock()

            with (
                mock.patch(
                    "tzrec.utils.export_util.init_process_group",
                    return_value=(torch.device("cpu"), None),
                ),
                mock.patch(
                    "tzrec.utils.export_util._get_sparse_table_to_embedding_info",
                    return_value=({}, {}),
                ),
                mock.patch(
                    "tzrec.utils.export_util.create_dataloader",
                    return_value=FakeDataloader(),
                ) as create_dataloader_mock,
                mock.patch(
                    "tzrec.utils.export_util.create_planner",
                    return_value=SimpleNamespace(collective_plan=lambda *args: None),
                ),
                mock.patch(
                    "tzrec.utils.export_util.get_default_sharders", return_value=[]
                ),
                mock.patch(
                    "tzrec.utils.export_util.DistributedModelParallel",
                    side_effect=lambda *args, **kwargs: FakeDMP(kwargs["module"]),
                ),
                mock.patch("tzrec.utils.export_util.checkpoint_util.restore_model"),
                mock.patch("tzrec.utils.export_util.init_parameters"),
                mock.patch(
                    "tzrec.utils.export_util._get_sparse_embedding_tensor",
                    return_value=({}, {}, {}, {}),
                ),
                mock.patch("tzrec.utils.export_util.config_util.save_message"),
                mock.patch(
                    "tzrec.utils.export_util.create_fg_json",
                    return_value={"features": []},
                ),
                mock.patch(
                    "tzrec.utils.export_util.symbolic_trace",
                    return_value=SimpleNamespace(code="def forward(self):\n    pass\n"),
                ),
                mock.patch(
                    "tzrec.utils.export_util.torch.jit.script",
                    return_value=fake_scripted,
                ),
                mock.patch(
                    "tzrec.utils.export_util.acc_utils.export_acc_config",
                    return_value=model_acc,
                ) as export_acc_config_mock,
            ):
                export_distributed_embedding(
                    pipeline_config,
                    TinyModel(),
                    "checkpoint_dir",
                    tmp,
                    additional_export_config={"cand_seq_pk": "cand_seq"},
                    data_input_path="override_input",
                )

            create_dataloader_mock.assert_called_once()
            self.assertEqual(create_dataloader_mock.call_args.args[2], "override_input")
            export_acc_config_mock.assert_called_once_with(
                additional_export_config={"cand_seq_pk": "cand_seq"}
            )
            with open(os.path.join(tmp, "model_acc.json")) as f:
                self.assertEqual(json.load(f), model_acc)
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_distributed_embedding_export_uses_overrides_and_preserves_config(
        self,
    ) -> None:
        class FakeBatch:
            def to(self, device):  # type: ignore[no-untyped-def]
                return self

            def to_dict(self, sparse_dtype):  # type: ignore[no-untyped-def]
                return {"x": torch.ones(1)}

        class FakeDataloader:
            dataset = SimpleNamespace(sampled_batch_size=1)

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter([FakeBatch()])

        class TinyModel(torch.nn.Module):
            def __init__(self):  # type: ignore[no-untyped-def]
                super().__init__()
                self.features = []

            def set_is_inference(self, is_inference):  # type: ignore[no-untyped-def]
                self.is_inference = is_inference

            def forward(self, data, device=None):  # type: ignore[no-untyped-def]
                return {"score": data["x"] + 1}

        class FakeDMP(torch.nn.Module):
            def __init__(self, module, *args, **kwargs):  # type: ignore[no-untyped-def]
                super().__init__()
                self.module = module

            def forward(self, data, device=None):  # type: ignore[no-untyped-def]
                return self.module(data, device=device)

        tmp = tempfile.mkdtemp(prefix="tzrec_export_dist_overrides_")
        old_env = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        }
        try:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            pipeline_config = EasyRecConfig(
                train_input_path="train_input",
                eval_input_path="eval_input",
                model_dir="model_dir",
            )
            dump_config = pipeline_config.train_config.delta_embedding_dump_config
            feature_store_config = dump_config.feature_store_config
            feature_store_config.region = "cn-test"
            feature_store_config.project_name = "project_a"
            feature_store_config.feature_view_name = "shared_embeddings"
            feature_store_config.version = "model_a@export_1"
            model_acc = {"SPARSE_INT64": "1", "cand_seq_pk": "cand_seq"}
            fake_scripted = mock.Mock()

            with (
                mock.patch(
                    "tzrec.utils.export_util.init_process_group",
                    return_value=(torch.device("cpu"), None),
                ),
                mock.patch(
                    "tzrec.utils.export_util._get_sparse_table_to_embedding_info",
                    return_value=({}, {}),
                ),
                mock.patch(
                    "tzrec.utils.export_util.create_dataloader",
                    return_value=FakeDataloader(),
                ) as create_dataloader_mock,
                mock.patch(
                    "tzrec.utils.export_util.create_planner",
                    return_value=SimpleNamespace(collective_plan=lambda *args: None),
                ),
                mock.patch(
                    "tzrec.utils.export_util.get_default_sharders", return_value=[]
                ),
                mock.patch(
                    "tzrec.utils.export_util.DistributedModelParallel",
                    side_effect=lambda *args, **kwargs: FakeDMP(kwargs["module"]),
                ),
                mock.patch("tzrec.utils.export_util.checkpoint_util.restore_model"),
                mock.patch("tzrec.utils.export_util.init_parameters"),
                mock.patch(
                    "tzrec.utils.export_util._get_sparse_embedding_tensor",
                    return_value=({}, {}, {}, {}),
                ),
                mock.patch(
                    "tzrec.utils.export_util.create_fg_json",
                    return_value={"features": []},
                ),
                mock.patch(
                    "tzrec.utils.export_util.symbolic_trace",
                    return_value=SimpleNamespace(code="def forward(self):\n    pass\n"),
                ),
                mock.patch(
                    "tzrec.utils.export_util.torch.jit.script",
                    return_value=fake_scripted,
                ),
                mock.patch(
                    "tzrec.utils.export_util.acc_utils.export_acc_config",
                    return_value=model_acc,
                ) as export_acc_config_mock,
            ):
                export_distributed_embedding(
                    pipeline_config,
                    TinyModel(),
                    "checkpoint_dir",
                    tmp,
                    additional_export_config={"cand_seq_pk": "cand_seq"},
                    data_input_path="override_input",
                )

            create_dataloader_mock.assert_called_once()
            self.assertEqual(create_dataloader_mock.call_args.args[2], "override_input")
            export_acc_config_mock.assert_called_once_with(
                additional_export_config={"cand_seq_pk": "cand_seq"}
            )
            with open(os.path.join(tmp, "model_acc.json")) as f:
                self.assertEqual(json.load(f), model_acc)
            pipeline_config_path = os.path.join(tmp, "pipeline.config")
            exported_config = config_util.load_pipeline_config(pipeline_config_path)
            exported_dump_config = (
                exported_config.train_config.delta_embedding_dump_config
            )
            exported_feature_store_config = exported_dump_config.feature_store_config
            self.assertEqual(exported_feature_store_config.project_name, "project_a")
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_distributed_embedding_export_bakes_config_order_keyed_tensor(
        self,
    ) -> None:
        """The dense export must bake config-order keys, not sharded runtime order.

        A sharded EBC emits its KeyedTensor grouped by sharding type / compute
        kernel (e.g. dynamicemb tables move to the tail), while the online
        dense export derives keys statically in config order; baking the
        runtime order makes the serving processor reject every hot dense
        update with dense_meta_mismatch. The fake sharded EBC below outputs
        keys rotated to (f_b, f_c, f_a) whereas its ``_embedding_bag_configs``
        order is (f_a, f_b, f_c).
        """
        tables = [
            EmbeddingBagConfig(
                name="t_a", embedding_dim=4, num_embeddings=10, feature_names=["f_a"]
            ),
            EmbeddingBagConfig(
                name="t_b", embedding_dim=8, num_embeddings=10, feature_names=["f_b"]
            ),
            EmbeddingBagConfig(
                name="t_c", embedding_dim=4, num_embeddings=10, feature_names=["f_c"]
            ),
        ]

        class FakeShardedEBC(ShardedModule):
            def __init__(self, tables):  # type: ignore[no-untyped-def]
                super().__init__()
                self._embedding_bag_configs = tables

            def forward(self, features):  # type: ignore[no-untyped-def]
                batch_size = features.size(0)
                values = torch.cat(
                    [
                        torch.full((batch_size, 8), 2.0),
                        torch.full((batch_size, 4), 3.0),
                        torch.full((batch_size, 4), 1.0),
                    ],
                    dim=1,
                )
                return KeyedTensor(
                    keys=["f_b", "f_c", "f_a"],
                    length_per_key=[8, 4, 4],
                    values=values,
                )

            def create_context(self):  # type: ignore[no-untyped-def]
                return None

            def input_dist(self, ctx, *inputs, **kwargs):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def compute(self, ctx, dist_input):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def output_dist(self, ctx, output):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            @property
            def unsharded_module_type(self):  # type: ignore[no-untyped-def]
                return EmbeddingBagCollection

        class TinyModel(torch.nn.Module):
            def __init__(self):  # type: ignore[no-untyped-def]
                super().__init__()
                self.features = []
                self.ebc = FakeShardedEBC(tables)

            def set_is_inference(self, is_inference):  # type: ignore[no-untyped-def]
                self.is_inference = is_inference

            def forward(self, data, device=None):  # type: ignore[no-untyped-def]
                kt = self.ebc(data["x"])
                fx_mark_keyed_tensor("grp__ebc", kt)
                grouped = KeyedTensor.regroup_as_dict(
                    [kt], [["f_a"], ["f_b", "f_c"]], ["y_a", "y_bc"]
                )
                return {
                    "y_a": grouped["y_a"].sum(dim=1),
                    "y_bc": grouped["y_bc"].sum(dim=1),
                }

        # the same dict flows through warm-up, sparse run and the dense
        # smoke run, so it exposes the permuted smoke-run input afterwards
        data = {"x": torch.ones(2, 3)}

        class FakeBatch:
            def to(self, device):  # type: ignore[no-untyped-def]
                return self

            def to_dict(self, sparse_dtype):  # type: ignore[no-untyped-def]
                return data

        class FakeDataloader:
            dataset = SimpleNamespace(sampled_batch_size=2)

            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter([FakeBatch()])

        class FakeDMP(torch.nn.Module):
            def __init__(self, module, *args, **kwargs):  # type: ignore[no-untyped-def]
                super().__init__()
                self.module = module

            def forward(self, data, device=None):  # type: ignore[no-untyped-def]
                return self.module(data, device=device)

        tmp = make_test_dir()
        old_env = {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        }
        try:
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            pipeline_config = EasyRecConfig(
                train_input_path="train_input",
                eval_input_path="eval_input",
                model_dir="model_dir",
            )
            captured = {}

            def _capture_symbolic_trace(gm):  # type: ignore[no-untyped-def]
                captured["gm"] = gm
                return SimpleNamespace(code="def forward(self):\n    pass\n")

            with (
                mock.patch(
                    "tzrec.utils.export_util.init_process_group",
                    return_value=(torch.device("cpu"), None),
                ),
                mock.patch(
                    "tzrec.utils.export_util._get_sparse_table_to_embedding_info",
                    return_value=({}, {}),
                ),
                mock.patch(
                    "tzrec.utils.export_util.create_dataloader",
                    return_value=FakeDataloader(),
                ),
                mock.patch(
                    "tzrec.utils.export_util.create_planner",
                    return_value=SimpleNamespace(collective_plan=lambda *args: None),
                ),
                mock.patch(
                    "tzrec.utils.export_util.get_default_sharders", return_value=[]
                ),
                mock.patch(
                    "tzrec.utils.export_util.DistributedModelParallel",
                    side_effect=lambda *args, **kwargs: FakeDMP(kwargs["module"]),
                ),
                mock.patch("tzrec.utils.export_util.checkpoint_util.restore_model"),
                mock.patch("tzrec.utils.export_util.init_parameters"),
                mock.patch(
                    "tzrec.utils.export_util._get_sparse_embedding_tensor",
                    return_value=({}, {}, {}, {}),
                ),
                mock.patch("tzrec.utils.export_util.config_util.save_message"),
                mock.patch(
                    "tzrec.utils.export_util.create_fg_json",
                    return_value={"features": []},
                ),
                mock.patch(
                    "tzrec.utils.export_util.symbolic_trace",
                    side_effect=_capture_symbolic_trace,
                ),
                mock.patch(
                    "tzrec.utils.export_util.torch.jit.script",
                    return_value=mock.Mock(),
                ),
                mock.patch(
                    "tzrec.utils.export_util.acc_utils.export_acc_config",
                    return_value={},
                ),
            ):
                export_distributed_embedding(
                    pipeline_config, TinyModel(), "checkpoint_dir", tmp
                )

            # dense_meta.json lists the marked group in config order
            with open(os.path.join(tmp, "dense_meta.json")) as f:
                dense_meta = json.load(f)
            self.assertEqual(
                dense_meta["grp__ebc"], ["f_a__ebc", "f_b__ebc", "f_c__ebc"]
            )
            self.assertEqual(dense_meta["sequence__ec"], [])

            # the dense graph bakes the canonical keys/length_per_key
            gm = captured["gm"]
            kt_nodes = [
                node
                for node in gm.graph.nodes
                if node.op == "call_function" and node.target is KeyedTensor
            ]
            self.assertEqual(len(kt_nodes), 1)
            self.assertEqual(kt_nodes[0].kwargs["keys"], ["f_a", "f_b", "f_c"])
            self.assertEqual(kt_nodes[0].kwargs["length_per_key"], [4, 8, 4])

            # the smoke-run input was permuted from the runtime layout
            # (f_b=2.0 x8, f_c=3.0 x4, f_a=1.0 x4) to the canonical layout
            canonical_values = torch.cat(
                [
                    torch.full((2, 4), 1.0),
                    torch.full((2, 8), 2.0),
                    torch.full((2, 4), 3.0),
                ],
                dim=1,
            )
            torch.testing.assert_close(data["grp__ebc"], canonical_values)

            # the exported dense graph regroups a canonical-layout input by
            # key name into the right slices
            out = gm({"grp__ebc": canonical_values.clone()}, torch.device("cpu"))
            torch.testing.assert_close(out["y_a"], torch.full((2,), 4.0))
            torch.testing.assert_close(out["y_bc"], torch.full((2,), 28.0))
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_dynamic_embedding_export_concats_training_shards(self) -> None:
        """Single-rank export must not drop multi-GPU dynamicemb checkpoint shards."""
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_")
        old_rank = os.environ.get("RANK")
        old_world_size = os.environ.get("WORLD_SIZE")
        old_quant = os.environ.get("DIST_QUANT")
        try:
            ckpt_dir = os.path.join(tmp, "model.ckpt-1")
            dy_dir = os.path.join(
                ckpt_dir,
                "dynamicemb",
                "model.model.embedding_group.emb_impls.__BASE__.ebc",
            )
            os.makedirs(dy_dir)

            def write_shard(rank: int, keys: np.ndarray, values: np.ndarray) -> None:
                keys.astype(np.int64).tofile(
                    os.path.join(
                        dy_dir, f"user_id_emb_emb_keys.rank_{rank}.world_size_2"
                    )
                )
                values.astype(np.float32).tofile(
                    os.path.join(
                        dy_dir, f"user_id_emb_emb_values.rank_{rank}.world_size_2"
                    )
                )
                (keys + 100).astype(np.int64).tofile(
                    os.path.join(
                        dy_dir, f"user_id_emb_emb_scores.rank_{rank}.world_size_2"
                    )
                )

            write_shard(
                0,
                np.array([0, 2]),
                np.array([[0.0, 0.1], [2.0, 2.1]], dtype=np.float32),
            )
            write_shard(
                1,
                np.array([1, 3]),
                np.array([[1.0, 1.1], [3.0, 3.1]], dtype=np.float32),
            )

            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ.pop("DIST_QUANT", None)
            table_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc."
                "embedding_bags.user_id_emb"
            )
            embedding_bag_info = {
                table_fqn: SimpleNamespace(
                    name="user_id_emb",
                    embedding_dim=2,
                    feature_names=["user_id"],
                    pooling="SUM",
                )
            }

            _, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                {},
                embedding_bag_info,
            )

            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.keys"], torch.tensor([0, 2, 1, 3])
            )
            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.scores"],
                torch.tensor([100, 102, 101, 103]),
            )
            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.values"],
                torch.tensor([[0.0, 0.1], [2.0, 2.1], [1.0, 1.1], [3.0, 3.1]]),
            )
            self.assertEqual(emb_meta[table_fqn]["shape"], [4, 2])
            self.assertEqual(emb_meta[table_fqn]["key_name"], f"{table_fqn}.keys")
            self.assertEqual(emb_meta[table_fqn]["value_name"], f"{table_fqn}.values")
            self.assertEqual(emb_meta[table_fqn]["score_name"], f"{table_fqn}.scores")
            self.assertEqual(
                feat_meta["user_id__ebc"],
                {"embedding_name": table_fqn, "pooling": "SUM"},
            )
        finally:
            if old_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = old_rank
            if old_world_size is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = old_world_size
            if old_quant is None:
                os.environ.pop("DIST_QUANT", None)
            else:
                os.environ["DIST_QUANT"] = old_quant
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_dynamic_embedding_quant_export(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_quant_")
        old_env = {
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "DIST_QUANT": os.environ.get("DIST_QUANT"),
        }
        try:
            ckpt_dir = os.path.join(tmp, "model.ckpt-1")
            dy_dir = os.path.join(
                ckpt_dir,
                "dynamicemb",
                "model.model.embedding_group.emb_impls.__BASE__.ebc",
            )
            os.makedirs(dy_dir)

            keys = np.array([0, 1], dtype=np.int64)
            values = np.array([[-2.0, 2.0], [-1.0, 1.0]], dtype=np.float32)
            keys.tofile(
                os.path.join(dy_dir, "user_id_emb_emb_keys.rank_0.world_size_1")
            )
            values.tofile(
                os.path.join(dy_dir, "user_id_emb_emb_values.rank_0.world_size_1")
            )
            (keys + 100).tofile(
                os.path.join(dy_dir, "user_id_emb_emb_scores.rank_0.world_size_1")
            )

            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["DIST_QUANT"] = "INT8"
            table_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc."
                "embedding_bags.user_id_emb"
            )
            embedding_bag_info = {
                table_fqn: SimpleNamespace(
                    name="user_id_emb",
                    embedding_dim=2,
                    feature_names=["user_id"],
                    pooling="SUM",
                )
            }

            _, dynamic_out, emb_meta, _ = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                {},
                embedding_bag_info,
            )

            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.keys"], torch.tensor([0, 1])
            )
            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.scores"], torch.tensor([100, 101])
            )
            self.assertEqual(dynamic_out[f"{table_fqn}.values"].dtype, np.uint8)
            self.assertEqual(dynamic_out[f"{table_fqn}.values"].shape, (2, 6))
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(
                    dynamic_out[f"{table_fqn}.values"], emb_dim=2
                ),
                values,
                atol=5e-3,
            )
            self.assertEqual(emb_meta[table_fqn]["dtype"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta[table_fqn]["shape"], [2, 2])
            self.assertEqual(emb_meta[table_fqn]["storage_shape"], [2, 6])
            self.assertEqual(emb_meta[table_fqn]["row_bytes"], 6)
            self.assertEqual(emb_meta[table_fqn]["quant"]["format"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta[table_fqn]["value_name"], f"{table_fqn}.values")
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_export_converts_zch_tables_to_dynamic(self) -> None:
        """ZCH tables serve by raw id, so they export as dynamic tables."""
        invalid_raw_id = torch.iinfo(torch.int64).max

        def _make_mch(zch_size, eviction_policy):  # type: ignore[no-untyped-def]
            return MCHManagedCollisionModule(
                zch_size=zch_size,
                device=torch.device("cpu"),
                eviction_policy=eviction_policy,
                eviction_interval=2,
            )

        def _set_mch_state(mch, raw_ids, remapped_ids, metadata):  # type: ignore[no-untyped-def]
            mch._buffers["_mch_sorted_raw_ids"].copy_(torch.tensor(raw_ids))
            mch._buffers["_mch_remapped_ids_mapping"].copy_(torch.tensor(remapped_ids))
            for name, value in metadata.items():
                mch._buffers[name].copy_(torch.tensor(value))

        model = torch.nn.Module()

        user_id_config = EmbeddingBagConfig(
            name="user_id_emb",
            embedding_dim=2,
            num_embeddings=4,
            feature_names=["user_id"],
        )
        plain_config = EmbeddingBagConfig(
            name="plain_emb",
            embedding_dim=2,
            num_embeddings=2,
            feature_names=["plain_id"],
        )
        seq_config = EmbeddingConfig(
            name="seq_emb",
            embedding_dim=2,
            num_embeddings=3,
            feature_names=["click_seq__cate"],
        )

        mc_ebc_weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        for path in ("mc_ebc", "mc_ebc_user"):
            mc_ebc = ManagedCollisionEmbeddingBagCollection(
                EmbeddingBagCollection([user_id_config], device=torch.device("cpu")),
                ManagedCollisionCollection(
                    {"user_id_emb": _make_mch(4, LFU_EvictionPolicy())},
                    [user_id_config],
                ),
            )
            mc_ebc._embedding_module.embedding_bags["user_id_emb"].weight.data.copy_(
                mc_ebc_weight
            )
            _set_mch_state(
                mc_ebc._managed_collision_collection._managed_collision_modules[
                    "user_id_emb"
                ],
                raw_ids=[101, 202, 303, invalid_raw_id],
                remapped_ids=[2, 0, 1, 3],
                metadata={"_mch_counts": [5, 7, 9, 0]},
            )
            _add_module_by_dotted_path(
                model, f"model.embedding_group.emb_impls.__BASE__.{path}", mc_ebc
            )

        plain_ebc = EmbeddingBagCollection([plain_config], device=torch.device("cpu"))
        plain_ebc.embedding_bags["plain_emb"].weight.data.copy_(
            torch.tensor([[7.0, 7.1], [8.0, 8.1]])
        )
        _add_module_by_dotted_path(
            model, "model.embedding_group.emb_impls.__BASE__.ebc", plain_ebc
        )

        mc_ec = ManagedCollisionEmbeddingCollection(
            EmbeddingCollection([seq_config], device=torch.device("cpu")),
            ManagedCollisionCollection(
                {"seq_emb": _make_mch(3, DistanceLFU_EvictionPolicy())}, [seq_config]
            ),
        )
        mc_ec._embedding_module.embeddings["seq_emb"].weight.data.copy_(
            torch.tensor([[4.0, 4.1], [5.0, 5.1], [6.0, 6.1]])
        )
        _set_mch_state(
            mc_ec._managed_collision_collection._managed_collision_modules["seq_emb"],
            raw_ids=[11, 22, invalid_raw_id],
            remapped_ids=[1, 0, 2],
            metadata={"_mch_counts": [3, 4, 0], "_mch_last_access_iter": [30, 40, 0]},
        )
        _add_module_by_dotted_path(
            model, "model.embedding_group.seq_emb_impls.__BASE__.mc_ec_dict.2", mc_ec
        )

        zch_ebc_fqn = (
            "model.embedding_group.emb_impls.__BASE__.mc_ebc."
            "_embedding_module.embedding_bags.user_id_emb"
        )
        zch_ebc_user_fqn = zch_ebc_fqn.replace("mc_ebc.", "mc_ebc_user.")
        plain_fqn = (
            "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.plain_emb"
        )
        zch_ec_fqn = (
            "model.embedding_group.seq_emb_impls.__BASE__.mc_ec_dict.2."
            "_embedding_module.embeddings.seq_emb"
        )

        tmp = tempfile.mkdtemp(prefix="tzrec_export_zch_")
        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        try:
            os.environ.pop("DIST_QUANT", None)
            with mock.patch.object(export_util, "logger") as m_logger:
                out, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                    model,
                    tmp,
                    {
                        zch_ec_fqn: SimpleNamespace(
                            name="seq_emb",
                            embedding_dim=2,
                            feature_names=["click_seq__cate"],
                        )
                    },
                    {
                        zch_ebc_fqn: SimpleNamespace(
                            name="user_id_emb",
                            embedding_dim=2,
                            feature_names=["user_id"],
                            pooling="SUM",
                        ),
                        zch_ebc_user_fqn: SimpleNamespace(
                            name="user_id_emb",
                            embedding_dim=2,
                            feature_names=["user_id"],
                            pooling="SUM",
                        ),
                        plain_fqn: SimpleNamespace(
                            name="plain_emb",
                            embedding_dim=2,
                            feature_names=["plain_id"],
                            pooling="SUM",
                        ),
                    },
                )

            self.assertEqual(sorted(out.keys()), [plain_fqn])
            np.testing.assert_array_equal(
                out[plain_fqn], np.array([[7.0, 7.1], [8.0, 8.1]], dtype=np.float32)
            )
            self.assertFalse(emb_meta[plain_fqn]["is_dynamic"])
            self.assertNotIn(zch_ebc_user_fqn, emb_meta)

            torch.testing.assert_close(
                dynamic_out[f"{zch_ebc_fqn}.keys"], torch.tensor([101, 202, 303])
            )
            np.testing.assert_array_equal(
                dynamic_out[f"{zch_ebc_fqn}.values"],
                np.array([[2.0, 2.1], [0.0, 0.1], [1.0, 1.1]], dtype=np.float32),
            )
            torch.testing.assert_close(
                dynamic_out[f"{zch_ebc_fqn}.scores"], torch.tensor([5, 7, 9])
            )
            # zch sends every id it does not hold to its reserved last row.
            np.testing.assert_array_equal(
                dynamic_out[f"{zch_ebc_fqn}.default_value"],
                np.array([[3.0, 3.1]], dtype=np.float32),
            )
            self.assertTrue(emb_meta[zch_ebc_fqn]["is_dynamic"])
            self.assertEqual(emb_meta[zch_ebc_fqn]["shape"], [3, 2])
            self.assertEqual(emb_meta[zch_ebc_fqn]["key_dtype"], "int64")
            self.assertEqual(emb_meta[zch_ebc_fqn]["score_dtype"], "int64")
            self.assertEqual(emb_meta[zch_ebc_fqn]["key_name"], f"{zch_ebc_fqn}.keys")
            self.assertEqual(
                emb_meta[zch_ebc_fqn]["value_name"], f"{zch_ebc_fqn}.values"
            )
            self.assertEqual(
                emb_meta[zch_ebc_fqn]["score_name"], f"{zch_ebc_fqn}.scores"
            )
            self.assertEqual(
                emb_meta[zch_ebc_fqn]["default_value_name"],
                f"{zch_ebc_fqn}.default_value",
            )
            self.assertEqual(
                feat_meta["user_id__ebc"],
                {"embedding_name": zch_ebc_fqn, "pooling": "SUM"},
            )

            torch.testing.assert_close(
                dynamic_out[f"{zch_ec_fqn}.keys"], torch.tensor([11, 22])
            )
            np.testing.assert_array_equal(
                dynamic_out[f"{zch_ec_fqn}.values"],
                np.array([[5.0, 5.1], [4.0, 4.1]], dtype=np.float32),
            )
            # DistanceLFU keeps counts and last access iter, recency is the score.
            torch.testing.assert_close(
                dynamic_out[f"{zch_ec_fqn}.scores"], torch.tensor([30, 40])
            )
            np.testing.assert_array_equal(
                dynamic_out[f"{zch_ec_fqn}.default_value"],
                np.array([[6.0, 6.1]], dtype=np.float32),
            )
            self.assertTrue(emb_meta[zch_ec_fqn]["is_dynamic"])

            zch_logs = [
                call.args[0]
                for call in m_logger.info.call_args_list
                if "convert zch table" in call.args[0]
            ]
            self.assertEqual(
                {log.split(" to dynamic")[0] for log in zch_logs},
                {
                    f"convert zch table {zch_ebc_fqn}",
                    f"convert zch table {zch_ec_fqn}",
                },
            )
            self.assertTrue(
                any("3 of 3 ids exported" in log for log in zch_logs), zch_logs
            )
            self.assertTrue(
                any("2 of 2 ids exported" in log for log in zch_logs), zch_logs
            )
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_export_quantizes_zch_default_value(self) -> None:
        """The default row is quantized like any other row of the table."""
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        mc_ebc_config = EmbeddingBagConfig(
            name="user_id_emb",
            embedding_dim=2,
            num_embeddings=4,
            feature_names=["user_id"],
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            EmbeddingBagCollection([mc_ebc_config], device=torch.device("cpu")),
            ManagedCollisionCollection(
                {
                    "user_id_emb": MCHManagedCollisionModule(
                        zch_size=4,
                        device=torch.device("cpu"),
                        eviction_policy=LFU_EvictionPolicy(),
                        eviction_interval=2,
                    )
                },
                [mc_ebc_config],
            ),
        )
        mc_ebc._embedding_module.embedding_bags["user_id_emb"].weight.data.copy_(weight)
        mch = mc_ebc._managed_collision_collection._managed_collision_modules[
            "user_id_emb"
        ]
        mch._buffers["_mch_sorted_raw_ids"].copy_(
            torch.tensor([101, 202, 303, torch.iinfo(torch.int64).max])
        )
        mch._buffers["_mch_remapped_ids_mapping"].copy_(torch.tensor([2, 0, 1, 3]))
        mch._buffers["_mch_counts"].copy_(torch.tensor([5, 7, 9, 0]))

        model = torch.nn.Module()
        _add_module_by_dotted_path(
            model, "model.embedding_group.emb_impls.__BASE__.mc_ebc", mc_ebc
        )
        table_fqn = (
            "model.embedding_group.emb_impls.__BASE__.mc_ebc."
            "_embedding_module.embedding_bags.user_id_emb"
        )

        tmp = tempfile.mkdtemp(prefix="tzrec_export_zch_quant_")
        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        try:
            os.environ["DIST_QUANT"] = "INT8"
            _, dynamic_out, emb_meta, _ = _get_sparse_embedding_tensor(
                model,
                tmp,
                {},
                {
                    table_fqn: SimpleNamespace(
                        name="user_id_emb",
                        embedding_dim=2,
                        feature_names=["user_id"],
                        pooling="SUM",
                    )
                },
            )

            default_value = dynamic_out[f"{table_fqn}.default_value"]
            self.assertEqual(default_value.dtype, np.uint8)
            self.assertEqual(default_value.shape, (1, 6))
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(default_value, emb_dim=2),
                weight[3:].numpy(),
                atol=5e-3,
            )
            self.assertEqual(
                emb_meta[table_fqn]["default_value_name"], f"{table_fqn}.default_value"
            )
            self.assertEqual(emb_meta[table_fqn]["shape"], [3, 2])
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_export_zch_without_eviction_metadata_emits_zero_scores(
        self,
    ) -> None:
        """A zch table with no eviction metadata exports zero scores, not an error."""
        invalid_raw_id = torch.iinfo(torch.int64).max
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        mc_ebc_config = EmbeddingBagConfig(
            name="user_id_emb",
            embedding_dim=2,
            num_embeddings=4,
            feature_names=["user_id"],
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            EmbeddingBagCollection([mc_ebc_config], device=torch.device("cpu")),
            ManagedCollisionCollection(
                {
                    "user_id_emb": MCHManagedCollisionModule(
                        zch_size=4,
                        device=torch.device("cpu"),
                        eviction_policy=LFU_EvictionPolicy(),
                        eviction_interval=2,
                    )
                },
                [mc_ebc_config],
            ),
        )
        mc_ebc._embedding_module.embedding_bags["user_id_emb"].weight.data.copy_(weight)
        mch = mc_ebc._managed_collision_collection._managed_collision_modules[
            "user_id_emb"
        ]
        mch._buffers["_mch_sorted_raw_ids"].copy_(
            torch.tensor([101, 202, 303, invalid_raw_id])
        )
        mch._buffers["_mch_remapped_ids_mapping"].copy_(torch.tensor([2, 0, 1, 3]))
        # No eviction metadata buffer: export must fall back to zero scores.
        for buffer_name in ("_mch_counts", "_mch_last_access_iter"):
            mch._buffers.pop(buffer_name, None)

        model = torch.nn.Module()
        _add_module_by_dotted_path(
            model, "model.embedding_group.emb_impls.__BASE__.mc_ebc", mc_ebc
        )
        table_fqn = (
            "model.embedding_group.emb_impls.__BASE__.mc_ebc."
            "_embedding_module.embedding_bags.user_id_emb"
        )

        tmp = tempfile.mkdtemp(prefix="tzrec_export_zch_noscore_")
        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        try:
            os.environ.pop("DIST_QUANT", None)
            _, dynamic_out, emb_meta, _ = _get_sparse_embedding_tensor(
                model,
                tmp,
                {},
                {
                    table_fqn: SimpleNamespace(
                        name="user_id_emb",
                        embedding_dim=2,
                        feature_names=["user_id"],
                        pooling="SUM",
                    )
                },
            )

            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.keys"], torch.tensor([101, 202, 303])
            )
            np.testing.assert_array_equal(
                dynamic_out[f"{table_fqn}.values"],
                np.array([[2.0, 2.1], [0.0, 0.1], [1.0, 1.1]], dtype=np.float32),
            )
            torch.testing.assert_close(
                dynamic_out[f"{table_fqn}.scores"], torch.tensor([0, 0, 0])
            )
            np.testing.assert_array_equal(
                dynamic_out[f"{table_fqn}.default_value"],
                np.array([[3.0, 3.1]], dtype=np.float32),
            )
            self.assertTrue(emb_meta[table_fqn]["is_dynamic"])
            self.assertEqual(emb_meta[table_fqn]["score_dtype"], "int64")
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_export_disambiguates_ec_ebc_embedding_name_collision(
        self,
    ) -> None:
        """EC and EBC may use the same config name but hold different tensors."""

        class SparseCollisionModel(torch.nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
                    "model.embedding_group.emb_impls.__BASE__.ebc_user."
                    "embedding_bags.shared_emb.weight": torch.tensor(
                        [[9.0, 9.1], [9.2, 9.3]]
                    ),
                    "model.embedding_group.emb_impls.__BASE__.ebc."
                    "embedding_bags.shared_emb.weight": torch.tensor(
                        [[1.0, 1.1], [1.2, 1.3]]
                    ),
                    "model.embedding_group.seq_emb_impls.__BASE__.ec_dict.2."
                    "embeddings.shared_emb.weight": torch.tensor(
                        [[2.0, 2.1], [2.2, 2.3]]
                    ),
                }

        tmp = tempfile.mkdtemp(prefix="tzrec_export_sparse_collision_")
        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        try:
            os.environ.pop("DIST_QUANT", None)
            ebc_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.shared_emb"
            )
            ebc_user_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc_user."
                "embedding_bags.shared_emb"
            )
            ec_fqn = (
                "model.embedding_group.seq_emb_impls.__BASE__.ec_dict.2."
                "embeddings.shared_emb"
            )
            out, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                SparseCollisionModel(),
                tmp,
                {
                    ec_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["seq_feat"],
                    )
                },
                {
                    ebc_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["id_feat"],
                        pooling="SUM",
                    ),
                    ebc_user_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["user_feat"],
                        pooling="SUM",
                    ),
                },
            )

            self.assertEqual(dynamic_out, {})
            self.assertNotIn("shared_emb", out)
            self.assertNotIn(ebc_user_fqn, out)
            self.assertNotIn(ebc_user_fqn, emb_meta)
            np.testing.assert_array_equal(
                out[ec_fqn],
                np.array([[2.0, 2.1], [2.2, 2.3]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                out[ebc_fqn],
                np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            )
            self.assertEqual(emb_meta[ec_fqn]["feat_name_impl"], ["seq_feat__ec"])
            self.assertEqual(
                emb_meta[ebc_fqn]["feat_name_impl"],
                ["id_feat__ebc", "user_feat__ebc"],
            )
            self.assertEqual(
                feat_meta["seq_feat__ec"],
                {"embedding_name": ec_fqn, "pooling": "NONE"},
            )
            self.assertEqual(
                feat_meta["id_feat__ebc"],
                {"embedding_name": ebc_fqn, "pooling": "SUM"},
            )
            self.assertEqual(
                feat_meta["user_feat__ebc"],
                {"embedding_name": ebc_fqn, "pooling": "SUM"},
            )
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_export_quantizes_ec_and_ebc_weights(self) -> None:
        class SparseCollisionModel(torch.nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
                    "model.embedding_group.emb_impls.__BASE__.ebc_user."
                    "embedding_bags.shared_emb.weight": torch.tensor(
                        [[-9.0, 9.0], [-10.0, 10.0]]
                    ),
                    "model.embedding_group.emb_impls.__BASE__.ebc."
                    "embedding_bags.shared_emb.weight": torch.tensor(
                        [[-1.0, 1.0], [-2.0, 2.0]]
                    ),
                    "model.embedding_group.seq_emb_impls.__BASE__.ec_dict.2."
                    "embeddings.shared_emb.weight": torch.tensor(
                        [[-3.0, 3.0], [-4.0, 4.0]]
                    ),
                }

        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        tmp = tempfile.mkdtemp(prefix="tzrec_export_sparse_quant_")
        try:
            os.environ["DIST_QUANT"] = "INT8"
            ebc_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.shared_emb"
            )
            ebc_user_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc_user."
                "embedding_bags.shared_emb"
            )
            ec_fqn = (
                "model.embedding_group.seq_emb_impls.__BASE__.ec_dict.2."
                "embeddings.shared_emb"
            )
            out, dynamic_out, emb_meta, _ = _get_sparse_embedding_tensor(
                SparseCollisionModel(),
                tmp,
                {
                    ec_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["seq_feat"],
                    )
                },
                {
                    ebc_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["id_feat"],
                        pooling="SUM",
                    ),
                    ebc_user_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["user_feat"],
                        pooling="SUM",
                    ),
                },
            )

            self.assertEqual(dynamic_out, {})
            self.assertNotIn(ebc_user_fqn, out)
            self.assertNotIn(ebc_user_fqn, emb_meta)
            self.assertEqual(out[ec_fqn].dtype, np.uint8)
            self.assertEqual(out[ebc_fqn].dtype, np.uint8)
            self.assertEqual(out[ec_fqn].shape, (2, 6))
            self.assertEqual(out[ebc_fqn].shape, (2, 6))
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(out[ec_fqn], emb_dim=2),
                np.array([[-3.0, 3.0], [-4.0, 4.0]], dtype=np.float32),
                atol=5e-3,
            )
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(out[ebc_fqn], emb_dim=2),
                np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32),
                atol=5e-3,
            )
            self.assertEqual(emb_meta[ec_fqn]["dtype"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta[ec_fqn]["shape"], [2, 2])
            self.assertEqual(emb_meta[ec_fqn]["storage_shape"], [2, 6])
            self.assertEqual(emb_meta[ec_fqn]["row_bytes"], 6)
            self.assertEqual(emb_meta[ebc_fqn]["dtype"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta[ebc_fqn]["shape"], [2, 2])
            self.assertEqual(emb_meta[ebc_fqn]["storage_shape"], [2, 6])
            self.assertEqual(emb_meta[ebc_fqn]["row_bytes"], 6)
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_quant_rejects_odd_embedding_dim(self) -> None:
        class OddDimModel(torch.nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
                    "model.embedding_group.emb_impls.__BASE__.ebc."
                    "embedding_bags.user_id_emb.weight": torch.ones(2, 3)
                }

        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        tmp = tempfile.mkdtemp(prefix="tzrec_export_sparse_quant_odd_")
        try:
            os.environ["DIST_QUANT"] = "INT8"
            with self.assertRaisesRegex(
                ValueError,
                "user_id_emb.*embedding_dim \\+ 4 = 3 \\+ 4 = 7.*"
                "change the table's embedding_dim to an even value.*"
                "DIST_QUANT=0/NONE",
            ):
                table_fqn = (
                    "model.embedding_group.emb_impls.__BASE__.ebc."
                    "embedding_bags.user_id_emb"
                )
                _get_sparse_embedding_tensor(
                    OddDimModel(),
                    tmp,
                    {},
                    {
                        table_fqn: SimpleNamespace(
                            name="user_id_emb",
                            embedding_dim=3,
                            feature_names=["user_id"],
                            pooling="SUM",
                        )
                    },
                )
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_merge_sharded_embedding_json_quant_meta(self) -> None:
        left = {
            "user_id_emb": {
                "feat_name_impl": ["user_id__ebc"],
                "dense": False,
                "is_dynamic": False,
                "dimension": 2,
                "dtype": "QUint8RowwiseF16",
                "storage_dtype": "uint8",
                "storage_shape": [2, 6],
                "row_bytes": 6,
                "memory": 12,
                "shape": [2, 2],
                "quant": {
                    "enabled": True,
                    "format": "QUint8RowwiseF16",
                    "scale_offset_dtype": "float16",
                    "output_dtype": "float16",
                },
            }
        }
        right = {
            "user_id_emb": {
                "feat_name_impl": ["user_id__ebc"],
                "dense": False,
                "is_dynamic": False,
                "dimension": 2,
                "dtype": "QUint8RowwiseF16",
                "storage_dtype": "uint8",
                "storage_shape": [3, 6],
                "row_bytes": 6,
                "memory": 18,
                "shape": [3, 2],
                "quant": {
                    "enabled": True,
                    "format": "QUint8RowwiseF16",
                    "scale_offset_dtype": "float16",
                    "output_dtype": "float16",
                },
            }
        }

        merged = _merge_sharded_embedding_json([left, right])
        self.assertEqual(merged["user_id_emb"]["shape"], [5, 2])
        self.assertEqual(merged["user_id_emb"]["storage_shape"], [5, 6])
        self.assertEqual(merged["user_id_emb"]["memory"], 30)

        bad_right = copy.deepcopy(right)
        bad_right["user_id_emb"]["row_bytes"] = 11
        with self.assertRaisesRegex(ValueError, "row_bytes"):
            _merge_sharded_embedding_json([left, bad_right])

    def test_dense_embedding_restore_survives_fx_flatten(self) -> None:
        """AutoDis/MLP params must restore after the RTP FX flatten.

        Their split-name ``state_dict`` only round-trips if the module class
        survives tracing as a leaf; otherwise restore skips them and leaves
        uninitialized memory. See ``export_rtp_model``.
        """
        configs = [
            AutoDisEmbeddingConfig(16, 3, 0.1, 0.8, ["dense_1", "dense_2"]),
            MLPDenseEmbeddingConfig(8, ["dense_3"]),
        ]
        ec = DenseEmbeddingCollection(configs)
        # state_dict returns parameter views; clone before mutating params.
        ref_state_dict = {k: v.detach().clone() for k, v in ec.state_dict().items()}

        leaf_names = _get_dense_embedding_leaf_module_names(ec)
        self.assertEqual(len([n for n in leaf_names if n.startswith("dense_embs.")]), 2)

        # Trace + flatten as export_rtp_model does.
        tracer = Tracer(leaf_modules=leaf_names)
        graph = tracer.trace(ec)
        gm = torch.fx.GraphModule(ec, graph)
        gm.graph.eliminate_dead_code()
        gm = _prune_unused_param_and_buffer(gm)

        # Garbage-fill to mimic init_parameters, then restore from checkpoint.
        with torch.no_grad():
            for param in gm.parameters():
                param.fill_(float("nan"))
        gm.load_state_dict(ref_state_dict)

        restored = gm.state_dict()
        self.assertEqual(sorted(restored.keys()), sorted(ref_state_dict.keys()))
        for name, ref in ref_state_dict.items():
            self.assertFalse(
                torch.isnan(restored[name]).any(), f"{name} was not restored"
            )
            torch.testing.assert_close(restored[name], ref)

    def test_infer_keyed_tensor_attrs_from_module_matches_ebc(self) -> None:
        """Inferred attrs must equal the EBC's runtime KeyedTensor attrs.

        Covers merged tables (one table serving multiple features) and shared
        features (one feature across tables, which get ``@table`` suffixed
        keys), and the MC-EBC duck-typing fallback via ``_embedding_module``.
        """
        tables = [
            EmbeddingBagConfig(
                name="uid_emb",
                embedding_dim=8,
                num_embeddings=100,
                feature_names=["uid"],
            ),
            EmbeddingBagConfig(
                name="pid_emb",
                embedding_dim=4,
                num_embeddings=100,
                feature_names=["pid", "cid"],
            ),
            EmbeddingBagConfig(
                name="pid_emb_shared",
                embedding_dim=16,
                num_embeddings=100,
                feature_names=["pid"],
            ),
        ]
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("cpu"))

        attrs = _infer_keyed_tensor_attrs_from_module(ebc)
        self.assertIsNotNone(attrs)
        keys, length_per_key = attrs
        self.assertEqual(keys, ebc._embedding_names)
        self.assertEqual(length_per_key, ebc._lengths_per_embedding)
        self.assertEqual(keys, ["uid", "pid@pid_emb", "cid", "pid@pid_emb_shared"])
        self.assertEqual(length_per_key, [8, 4, 4, 16])

        mc_like = torch.nn.Module()
        mc_like._embedding_module = ebc
        self.assertEqual(
            _infer_keyed_tensor_attrs_from_module(mc_like), (keys, length_per_key)
        )

    def test_get_embedding_bag_configs_reads_private_and_nested_attrs(self) -> None:
        """Sharded modules expose configs only via private attributes.

        ``ShardedEmbeddingBagCollection`` keeps the original-order configs in
        ``_embedding_bag_configs`` without a public accessor, and sharded
        MC-EBC modules nest that holder under ``_embedding_module``.
        """
        configs = [
            EmbeddingBagConfig(
                name="t_a", embedding_dim=4, num_embeddings=10, feature_names=["f_a"]
            ),
            EmbeddingBagConfig(
                name="t_b", embedding_dim=8, num_embeddings=10, feature_names=["f_b"]
            ),
        ]

        sharded_ebc_like = SimpleNamespace(_embedding_bag_configs=configs)
        self.assertEqual(_get_embedding_bag_configs(sharded_ebc_like), configs)

        sharded_mc_ebc_like = SimpleNamespace(
            _embedding_module=SimpleNamespace(_embedding_bag_configs=configs)
        )
        self.assertEqual(_get_embedding_bag_configs(sharded_mc_ebc_like), configs)

        ebc = EmbeddingBagCollection(tables=configs, device=torch.device("cpu"))
        self.assertEqual(_get_embedding_bag_configs(ebc), configs)

        self.assertIsNone(_get_embedding_bag_configs(torch.nn.Module()))

    def _make_canonicalize_test_ebc(self) -> EmbeddingBagCollection:
        tables = [
            EmbeddingBagConfig(
                name="t_a", embedding_dim=4, num_embeddings=10, feature_names=["f_a"]
            ),
            EmbeddingBagConfig(
                name="t_b", embedding_dim=8, num_embeddings=10, feature_names=["f_b"]
            ),
            EmbeddingBagConfig(
                name="t_c", embedding_dim=4, num_embeddings=10, feature_names=["f_c"]
            ),
        ]
        return EmbeddingBagCollection(tables=tables, device=torch.device("cpu"))

    def test_canonicalize_keyed_tensor_attrs_returns_config_order(self) -> None:
        """Sharding-plan-dependent runtime key order maps back to config order."""
        ebc = self._make_canonicalize_test_ebc()
        keys, length_per_key = _canonicalize_keyed_tensor_attrs(
            ebc, "grp__ebc", ["f_b", "f_c", "f_a"], [8, 4, 4]
        )
        self.assertEqual(keys, ["f_a", "f_b", "f_c"])
        self.assertEqual(length_per_key, [4, 8, 4])

    def test_canonicalize_keyed_tensor_attrs_rejects_runtime_mismatch(self) -> None:
        ebc = self._make_canonicalize_test_ebc()
        # a runtime dim that disagrees with the config
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            _canonicalize_keyed_tensor_attrs(
                ebc, "grp__ebc", ["f_b", "f_c", "f_a"], [8, 4, 8]
            )
        # a renamed runtime key
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            _canonicalize_keyed_tensor_attrs(
                ebc, "grp__ebc", ["f_b", "f_c", "f_x"], [8, 4, 4]
            )
        # a missing runtime key
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            _canonicalize_keyed_tensor_attrs(ebc, "grp__ebc", ["f_b", "f_c"], [8, 4])

    def test_canonicalize_keyed_tensor_attrs_raises_without_module(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "statically infer"):
            _canonicalize_keyed_tensor_attrs(
                None, "grp__ebc", ["f_b", "f_c", "f_a"], [8, 4, 4]
            )
        # a resolved module without embedding bag configs must also fail fast
        with self.assertRaisesRegex(RuntimeError, "statically infer"):
            _canonicalize_keyed_tensor_attrs(
                torch.nn.Linear(2, 2), "grp__ebc", ["f_b", "f_c", "f_a"], [8, 4, 4]
            )

    def test_permute_keyed_tensor_values_reorders_dim1_blocks(self) -> None:
        values = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 16)
        permuted = _permute_keyed_tensor_values(
            values, ["f_b", "f_c", "f_a"], [8, 4, 4], ["f_a", "f_b", "f_c"]
        )
        # src layout: f_b = cols 0:8, f_c = cols 8:12, f_a = cols 12:16
        expected = torch.cat([values[:, 12:16], values[:, 0:8], values[:, 8:12]], dim=1)
        self.assertEqual(permuted.shape, values.shape)
        torch.testing.assert_close(permuted, expected)

    def test_isolate_kafka_export_group_swaps_group_id(self) -> None:
        """Isolate the export Kafka consumer from the live training group."""
        from tzrec.datasets.kafka_dataset import _parse_kafka_uri

        uri = "kafka://broker:9092/topic?group.id=training&auto.offset.reset=earliest"
        isolated = _isolate_kafka_export_group(uri)
        topic, params, _ = _parse_kafka_uri(isolated)
        self.assertEqual(topic, "topic")
        self.assertEqual(params["group.id"], "training__dense_export")
        self.assertEqual(params.get("auto.offset.reset"), "earliest")
        # non-kafka inputs pass through unchanged
        self.assertEqual(
            _isolate_kafka_export_group("hdfs://path/to/file"),
            "hdfs://path/to/file",
        )
        # kafka without group.id is left untouched
        self.assertEqual(
            _isolate_kafka_export_group("kafka://broker:9092/topic?foo=bar"),
            "kafka://broker:9092/topic?foo=bar",
        )

    def test_export_dense_model_cpu_end_to_end(self) -> None:
        """Warm-up, strict restore, sanity run and scripting on a real model."""
        test_dir = make_test_dir()
        try:
            feature_cfgs = [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_a", embedding_dim=16, num_buckets=100
                    )
                ),
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_b", embedding_dim=16, num_buckets=1000
                    )
                ),
                feature_pb2.FeatureConfig(
                    raw_feature=feature_pb2.RawFeature(feature_name="int_a")
                ),
            ]
            features = create_features(feature_cfgs)
            model_config = model_pb2.ModelConfig(
                feature_groups=[
                    model_pb2.FeatureGroupConfig(
                        group_name="wide",
                        feature_names=["cat_a", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.WIDE,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="fm",
                        feature_names=["cat_a", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="deep",
                        feature_names=["cat_a", "cat_b", "int_a"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                ],
                deepfm=rank_model_pb2.DeepFM(
                    deep=module_pb2.MLP(hidden_units=[8, 4]),
                    final=module_pb2.MLP(hidden_units=[2]),
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            )

            def _build_model() -> DeepFM:
                return DeepFM(
                    model_config=model_config, features=features, labels=["label"]
                )

            def _build_wrapped_model() -> ScriptWrapper:
                model = _build_model()
                init_parameters(model, device=torch.device("cpu"))
                return ScriptWrapper(model)

            batch = Batch(
                dense_features={
                    BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
                    )
                },
                # First id is an out-of-range dynamicemb-style 64-bit FG hash
                # to guard the warm-up zeroing in export_dense_model_cpu:
                # without it F.embedding_bag's strict CPU range-check raises.
                sparse_features={
                    BASE_DATA_GROUP: KeyedJaggedTensor.from_lengths_sync(
                        keys=["cat_a", "cat_b"],
                        values=torch.tensor([2100765614044343531, 2, 3, 4, 5, 6, 7]),
                        lengths=torch.tensor([1, 2, 1, 3]),
                    )
                },
                labels={},
            )

            pipeline_config = EasyRecConfig()
            pipeline_config.train_input_path = "unused-mocked"

            ckpt_dir = os.path.join(test_dir, "model.ckpt-0")
            export_dir = os.path.join(test_dir, "dense_export")
            port = misc_util.get_free_port()
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
            try:
                with (
                    mock.patch("tzrec.utils.checkpoint_util.has_dynamicemb", False),
                    mock.patch(
                        "tzrec.utils.export_util.create_dataloader",
                        return_value=iter([batch]),
                    ),
                ):
                    checkpoint_util.save_model(ckpt_dir, _build_wrapped_model())
                    # pass meta embeddings to exercise in-function init_parameters
                    export_dense_model_cpu(
                        pipeline_config=pipeline_config,
                        model=ScriptWrapper(_build_model()),
                        checkpoint_path=ckpt_dir,
                        save_dir=export_dir,
                    )
            finally:
                dist.destroy_process_group()

            with open(os.path.join(export_dir, "dense_meta.json")) as f:
                dense_meta = json.load(f)
            ebc_groups = {k: v for k, v in dense_meta.items() if k != "sequence__ec"}
            all_emb_names = [n for names in ebc_groups.values() for n in names]
            self.assertTrue(all_emb_names)
            for emb_name in all_emb_names:
                self.assertIn(emb_name.split("@")[0], {"cat_a", "cat_b"})
            # never bare table names (cat_a_emb / cat_b_emb): the old
            # table-name inference emitted those instead of feature names
            self.assertNotIn("cat_a_emb__ebc", all_emb_names)
            self.assertNotIn("cat_b_emb__ebc", all_emb_names)
            # cat_a/cat_b are shared by the wide and fm/deep tables, so the
            # shared-feature @table form must appear
            self.assertTrue(any("@" in n for n in all_emb_names))
            self.assertEqual(dense_meta["sequence__ec"], [])

            scripted = torch.jit.load(os.path.join(export_dir, "scripted_model.pt"))
            serving_data = dict(batch.to_dict())
            for group_name, names in ebc_groups.items():
                # wide tables use the 4-dim wide embedding, others 16
                dims = [4 if "_wide" in n else 16 for n in names]
                serving_data[group_name] = torch.rand(2, sum(dims))
            serving_data["batch_size"] = torch.tensor(2)
            predictions = scripted(serving_data)
            self.assertEqual(predictions["logits"].size(), (2,))
            self.assertEqual(predictions["probs"].size(), (2,))
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_in_process_dense_export_matches_checkpoint_export(self) -> None:
        """Build + in-memory weight load + finalize must match the checkpoint path.

        The in-process online export hot-swaps weights gathered from the live
        model into a resident dense graph instead of restoring a checkpoint;
        given the same weights, both paths must script identical predictions.
        """
        test_dir = make_test_dir()
        try:
            feature_cfgs = [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_a", embedding_dim=16, num_buckets=100
                    )
                ),
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_b", embedding_dim=16, num_buckets=1000
                    )
                ),
                feature_pb2.FeatureConfig(
                    raw_feature=feature_pb2.RawFeature(feature_name="int_a")
                ),
            ]
            features = create_features(feature_cfgs)
            model_config = model_pb2.ModelConfig(
                feature_groups=[
                    model_pb2.FeatureGroupConfig(
                        group_name="wide",
                        feature_names=["cat_a", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.WIDE,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="fm",
                        feature_names=["cat_a", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="deep",
                        feature_names=["cat_a", "cat_b", "int_a"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                ],
                deepfm=rank_model_pb2.DeepFM(
                    deep=module_pb2.MLP(hidden_units=[8, 4]),
                    final=module_pb2.MLP(hidden_units=[2]),
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            )

            def _build_model() -> DeepFM:
                return DeepFM(
                    model_config=model_config, features=features, labels=["label"]
                )

            batch = Batch(
                dense_features={
                    BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
                    )
                },
                sparse_features={
                    BASE_DATA_GROUP: KeyedJaggedTensor.from_lengths_sync(
                        keys=["cat_a", "cat_b"],
                        values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                        lengths=torch.tensor([1, 2, 1, 3]),
                    )
                },
                labels={},
            )

            pipeline_config = EasyRecConfig()
            pipeline_config.train_input_path = "unused-mocked"

            device = torch.device("cpu")
            live_model = ScriptWrapper(_build_model())
            init_parameters(live_model, device=device)

            ckpt_dir = os.path.join(test_dir, "model.ckpt-0")
            ckpt_export_dir = os.path.join(test_dir, "dense_export_ckpt")
            inproc_export_dir = os.path.join(test_dir, "dense_export_inproc")
            port = misc_util.get_free_port()
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
            try:
                with (
                    mock.patch("tzrec.utils.checkpoint_util.has_dynamicemb", False),
                    # a fresh single-batch iterator per dataloader creation
                    mock.patch(
                        "tzrec.utils.export_util.create_dataloader",
                        side_effect=lambda *args, **kwargs: iter([batch]),
                    ),
                ):
                    checkpoint_util.save_model(ckpt_dir, live_model)
                    export_dense_model_cpu(
                        pipeline_config=pipeline_config,
                        model=ScriptWrapper(_build_model()),
                        checkpoint_path=ckpt_dir,
                        save_dir=ckpt_export_dir,
                    )
                    # in-process path: gather the weights from the live
                    # model's own state_dict (single process => no sharding,
                    # every source is a plain replicated tensor), load them
                    # into the resident graph and finalize.
                    warmup_data = create_dense_export_warmup_data(
                        pipeline_config, live_model, device
                    )
                    gm, full_graph, dense_graph_config = build_dense_graph_module(
                        live_model, warmup_data, device
                    )
                    live_state = live_model.state_dict()
                    snapshot = {
                        key: live_state[
                            key
                            if key in live_state
                            else checkpoint_util.remap_input_tile_user_key(
                                key, live_state
                            )
                        ]
                        .detach()
                        .cpu()
                        for key in sorted(gm.state_dict().keys())
                    }
                    gm.load_state_dict(snapshot)
                    dense_model_traced = finalize_dense_export(
                        live_model,
                        full_graph,
                        gm,
                        warmup_data,
                        device,
                        inproc_export_dir,
                        dense_graph_config,
                    )
            finally:
                dist.destroy_process_group()

            scripted_ckpt = torch.jit.load(
                os.path.join(ckpt_export_dir, "scripted_model.pt")
            )
            scripted_inproc = torch.jit.load(
                os.path.join(inproc_export_dir, "scripted_model.pt")
            )
            with open(os.path.join(ckpt_export_dir, "dense_meta.json")) as f:
                dense_meta = json.load(f)
            ebc_groups = {k: v for k, v in dense_meta.items() if k != "sequence__ec"}
            serving_data = dict(batch.to_dict())
            for group_name, names in ebc_groups.items():
                dims = [4 if "_wide" in n else 16 for n in names]
                serving_data[group_name] = torch.rand(2, sum(dims))
            serving_data["batch_size"] = torch.tensor(2)
            out_ckpt = scripted_ckpt(serving_data)
            out_inproc = scripted_inproc(serving_data)
            self.assertEqual(set(out_ckpt.keys()), set(out_inproc.keys()))
            for key in out_ckpt:
                torch.testing.assert_close(out_ckpt[key], out_inproc[key])

            # The online export traces once (captured above) and reuses the
            # traced module across versions instead of re-tracing on the worker.
            # Reusing it must not call symbolic_trace at all: a regression that
            # re-traces on the reuse path would re-open the worker-thread race
            # this PR fixes while still passing a mere output-equivalence check,
            # so patch symbolic_trace to raise under the reuse calls.
            with mock.patch(
                "tzrec.utils.export_util.symbolic_trace",
                side_effect=AssertionError("reuse path must not call symbolic_trace"),
            ):
                reuse_dir = os.path.join(test_dir, "dense_export_reuse")
                finalize_dense_export(
                    live_model,
                    full_graph,
                    gm,
                    warmup_data,
                    device,
                    reuse_dir,
                    dense_graph_config,
                    dense_model_traced=dense_model_traced,
                )
                out_reuse = torch.jit.load(
                    os.path.join(reuse_dir, "scripted_model.pt")
                )(serving_data)
                for key in out_inproc:
                    torch.testing.assert_close(out_reuse[key], out_inproc[key])

                # A weight reload into gm (whose parameters the traced module
                # shares) must be reflected by the reused traced module without
                # re-tracing, matching a fresh internal-trace finalize on the
                # same reloaded weights.
                reloaded = {
                    key: value + 1.0 if value.is_floating_point() else value
                    for key, value in gm.state_dict().items()
                }
                gm.load_state_dict(reloaded)
                reload_reuse_dir = os.path.join(test_dir, "dense_export_reload_reuse")
                finalize_dense_export(
                    live_model,
                    full_graph,
                    gm,
                    warmup_data,
                    device,
                    reload_reuse_dir,
                    dense_graph_config,
                    dense_model_traced=dense_model_traced,
                )
            # The fresh internal-trace path (no cached module) still traces.
            reload_fresh_dir = os.path.join(test_dir, "dense_export_reload_fresh")
            finalize_dense_export(
                live_model,
                full_graph,
                gm,
                warmup_data,
                device,
                reload_fresh_dir,
                dense_graph_config,
            )
            out_reload_reuse = torch.jit.load(
                os.path.join(reload_reuse_dir, "scripted_model.pt")
            )(serving_data)
            out_reload_fresh = torch.jit.load(
                os.path.join(reload_fresh_dir, "scripted_model.pt")
            )(serving_data)
            for key in out_reload_fresh:
                torch.testing.assert_close(out_reload_reuse[key], out_reload_fresh[key])
            self.assertTrue(
                any(
                    not torch.allclose(out_reload_reuse[key], out_reuse[key])
                    for key in out_reload_reuse
                ),
                "weight reload was not reflected by the reused traced module",
            )
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_shrink_sparse_embedding_tables(self) -> None:
        """EBC / EC tables and zch buffers shrink to one row in place.

        Covers plain EBC, sequence EC, and the MC-EBC wrapper whose inner
        EBC is reached through ``_embedding_module``; meta device must be
        preserved so materialization still happens later, at 1 row.
        """
        ebc_config = EmbeddingBagConfig(
            name="huge_emb",
            embedding_dim=16,
            num_embeddings=10_000_000,
            feature_names=["huge"],
        )
        ebc = EmbeddingBagCollection(tables=[ebc_config], device=torch.device("meta"))
        ec = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name="seq_emb",
                    embedding_dim=8,
                    num_embeddings=5_000_000,
                    feature_names=["seq_feat"],
                )
            ],
            device=torch.device("meta"),
        )
        zch_config = EmbeddingBagConfig(
            name="zch_emb",
            embedding_dim=8,
            num_embeddings=1000,
            feature_names=["zch_feat"],
        )
        mch = MCHManagedCollisionModule(
            zch_size=1000,
            device=torch.device("meta"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=5,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            EmbeddingBagCollection(tables=[zch_config], device=torch.device("meta")),
            ManagedCollisionCollection({"zch_emb": mch}, [zch_config]),
        )
        root = torch.nn.ModuleDict({"ebc": ebc, "ec": ec, "mc_ebc": mc_ebc})

        # the zch-sized buffers shrink selects by name
        mch_buffer_names = {"_mch_sorted_raw_ids", "_mch_remapped_ids_mapping"} | {
            f"_mch_{name}" for name in mch._mch_metadata
        }
        # (1,)-shaped sentinels that a shape heuristic would mishandle:
        # _mch_slots carries the *value* zch_size - 1, _delimiter iinfo.max
        sentinel_names = ["_mch_slots", "_delimiter", "_output_segments_tensor"]
        sentinels_before = {name: mch._buffers[name] for name in sentinel_names}

        _shrink_sparse_embedding_tables(root)

        self.assertEqual(ebc.embedding_bags["huge_emb"].weight.shape, (1, 16))
        self.assertTrue(ebc.embedding_bags["huge_emb"].weight.is_meta)
        self.assertEqual(ec.embeddings["seq_emb"].weight.shape, (1, 8))
        self.assertEqual(
            mc_ebc._embedding_module.embedding_bags["zch_emb"].weight.shape,
            (1, 8),
        )
        self.assertEqual(mch._zch_size, 1)
        for name in mch_buffer_names:
            self.assertEqual(mch._buffers[name].shape[0], 1)
        # _mch_metadata caches references to the eviction buffers; without
        # a refresh it would keep pointing at the old zch-sized tensors.
        for metadata_name in mch._mch_metadata:
            self.assertIs(
                mch._mch_metadata[metadata_name],
                mch._buffers[f"_mch_{metadata_name}"],
            )
        for name in sentinel_names:
            self.assertIs(mch._buffers[name], sentinels_before[name])

    def test_shrink_sparse_embedding_tables_preserves_sentinels_at_zch_size_2(
        self,
    ) -> None:
        """zch_size=2: (1,)-shaped sentinel buffers must survive untouched.

        With zch_size=2 every (1,)-shaped buffer's shape[0] equals
        zch_size - 1, so a shape-based match would zero the sentinels
        (_mch_slots=[1], _delimiter=iinfo.max) while the name-based match
        only shrinks the zch-sized remap buffers.
        """
        zch_config = EmbeddingBagConfig(
            name="zch_emb", embedding_dim=8, num_embeddings=2, feature_names=["f"]
        )
        mch = MCHManagedCollisionModule(
            zch_size=2,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=5,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            EmbeddingBagCollection(tables=[zch_config], device=torch.device("cpu")),
            ManagedCollisionCollection({"zch_emb": mch}, [zch_config]),
        )

        _shrink_sparse_embedding_tables(mc_ebc)

        self.assertEqual(mch._zch_size, 1)
        self.assertEqual(mch._buffers["_mch_sorted_raw_ids"].shape[0], 1)
        self.assertEqual(mch._buffers["_mch_remapped_ids_mapping"].shape[0], 1)
        self.assertEqual(mch._buffers["_mch_slots"].item(), 1)
        self.assertEqual(
            mch._buffers["_delimiter"].item(), torch.iinfo(torch.int64).max
        )

    def test_export_dense_model_cpu_with_zch_end_to_end(self) -> None:
        """A zch (MC-EBC) model traces, prunes and scripts after shrinking.

        Locks the MCH branch of the shrink: the zeroed warm-up batch must
        trace through the MC remap, the sparse lookups and their zch buffers
        must be pruned out of the dense graph, and the sanitized module must
        script -- i.e. the MCH shrink keeps the export runnable end to end.
        """
        test_dir = make_test_dir()
        try:
            feature_cfgs = [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_zch",
                        embedding_dim=16,
                        num_buckets=10_000,
                        zch=feature_pb2.ZeroCollisionHash(
                            zch_size=1000,
                            eviction_interval=5,
                            lfu=feature_pb2.LFU_EvictionPolicy(),
                        ),
                    )
                ),
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_b", embedding_dim=16, num_buckets=100
                    )
                ),
                feature_pb2.FeatureConfig(
                    raw_feature=feature_pb2.RawFeature(feature_name="int_a")
                ),
            ]
            features = create_features(feature_cfgs)
            model_config = model_pb2.ModelConfig(
                feature_groups=[
                    model_pb2.FeatureGroupConfig(
                        group_name="wide",
                        feature_names=["cat_zch", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.WIDE,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="fm",
                        feature_names=["cat_zch", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="deep",
                        feature_names=["cat_zch", "cat_b", "int_a"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                ],
                deepfm=rank_model_pb2.DeepFM(
                    deep=module_pb2.MLP(hidden_units=[8, 4]),
                    final=module_pb2.MLP(hidden_units=[2]),
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            )

            def _build_model() -> DeepFM:
                return DeepFM(
                    model_config=model_config, features=features, labels=["label"]
                )

            batch = Batch(
                dense_features={
                    BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
                    )
                },
                sparse_features={
                    BASE_DATA_GROUP: KeyedJaggedTensor.from_lengths_sync(
                        keys=["cat_zch", "cat_b"],
                        values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                        lengths=torch.tensor([1, 2, 1, 3]),
                    )
                },
                labels={},
            )
            # mirror create_dense_export_warmup_data: zero the sparse ids so
            # the shrunken tables' only valid index (0) is the one looked up.
            for kjt in batch.sparse_features.values():
                kjt.values().zero_()
            data = batch.to_dict(sparse_dtype=torch.int64)

            model = ScriptWrapper(_build_model())
            mch_modules = [
                m for m in model.modules() if isinstance(m, MCHManagedCollisionModule)
            ]
            self.assertTrue(mch_modules)
            gm, full_graph, dense_graph_config = build_dense_graph_module(
                model, data, torch.device("cpu")
            )
            # build_dense_graph_module shrank the model's zch tables in place.
            for mch_module in mch_modules:
                self.assertEqual(mch_module._zch_size, 1)
            # the sparse lookups and their zch buffers are pruned out of the
            # dense graph; no _mch_* state may survive into gm.
            self.assertTrue(gm.state_dict())
            for key in gm.state_dict():
                self.assertNotIn("_mch_", key)

            export_dir = os.path.join(test_dir, "dense_export")
            finalize_dense_export(
                model,
                full_graph,
                gm,
                data,
                torch.device("cpu"),
                export_dir,
                dense_graph_config,
            )
            scripted = torch.jit.load(os.path.join(export_dir, "scripted_model.pt"))
            with open(os.path.join(export_dir, "dense_meta.json")) as f:
                dense_meta = json.load(f)
            ebc_groups = {k: v for k, v in dense_meta.items() if k != "sequence__ec"}
            serving_data = dict(batch.to_dict())
            for group_name, names in ebc_groups.items():
                dims = [4 if "_wide" in n else 16 for n in names]
                serving_data[group_name] = torch.rand(2, sum(dims))
            serving_data["batch_size"] = torch.tensor(2)
            predictions = scripted(serving_data)
            self.assertEqual(predictions["logits"].size(), (2,))
            self.assertEqual(predictions["probs"].size(), (2,))
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_export_dense_model_cpu_materializes_only_shrunken_tables(self) -> None:
        """Sparse tables must be shrunk before any parameter materialization.

        Regression for the online dense export OOM kill: init_parameters
        materializes every meta parameter at full shape, so a dynamicemb
        table at its max_capacity row count exhausts host memory before the
        sparse pruning ever runs. Guard init_parameters and fail the moment
        a sparse table reaches it with more than one row.
        """
        test_dir = make_test_dir()
        try:
            feature_cfgs = [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_a",
                        embedding_dim=16,
                        num_buckets=1_000_000,
                    )
                ),
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="cat_b", embedding_dim=16, num_buckets=1000
                    )
                ),
                feature_pb2.FeatureConfig(
                    raw_feature=feature_pb2.RawFeature(feature_name="int_a")
                ),
            ]
            features = create_features(feature_cfgs)
            model_config = model_pb2.ModelConfig(
                feature_groups=[
                    model_pb2.FeatureGroupConfig(
                        group_name="wide",
                        feature_names=["cat_a", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.WIDE,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="fm",
                        feature_names=["cat_a", "cat_b"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                    model_pb2.FeatureGroupConfig(
                        group_name="deep",
                        feature_names=["cat_a", "cat_b", "int_a"],
                        group_type=model_pb2.FeatureGroupType.DEEP,
                    ),
                ],
                deepfm=rank_model_pb2.DeepFM(
                    deep=module_pb2.MLP(hidden_units=[8, 4]),
                    final=module_pb2.MLP(hidden_units=[2]),
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            )

            def _build_model() -> DeepFM:
                return DeepFM(
                    model_config=model_config, features=features, labels=["label"]
                )

            def _build_wrapped_model() -> ScriptWrapper:
                model = _build_model()
                init_parameters(model, device=torch.device("cpu"))
                return ScriptWrapper(model)

            batch = Batch(
                dense_features={
                    BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                        keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
                    )
                },
                sparse_features={
                    BASE_DATA_GROUP: KeyedJaggedTensor.from_lengths_sync(
                        keys=["cat_a", "cat_b"],
                        values=torch.tensor([2100765614044343531, 2, 3, 4, 5, 6, 7]),
                        lengths=torch.tensor([1, 2, 1, 3]),
                    )
                },
                labels={},
            )

            pipeline_config = EasyRecConfig()
            pipeline_config.train_input_path = "unused-mocked"

            real_init_parameters = export_util.init_parameters

            def guarded_init_parameters(module, device):
                for sub in module.modules():
                    tables = None
                    if isinstance(sub, EmbeddingBagCollection):
                        tables = list(sub.embedding_bags.values())
                    elif isinstance(sub, EmbeddingCollection):
                        tables = list(sub.embeddings.values())
                    for table in tables or []:
                        self.assertEqual(
                            table.weight.shape[0],
                            1,
                            "sparse table not shrunk before materialization",
                        )
                real_init_parameters(module, device)

            ckpt_dir = os.path.join(test_dir, "model.ckpt-0")
            export_dir = os.path.join(test_dir, "dense_export")
            port = misc_util.get_free_port()
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
            try:
                with (
                    mock.patch("tzrec.utils.checkpoint_util.has_dynamicemb", False),
                    mock.patch(
                        "tzrec.utils.export_util.create_dataloader",
                        return_value=iter([batch]),
                    ),
                    mock.patch.object(
                        export_util,
                        "init_parameters",
                        side_effect=guarded_init_parameters,
                    ),
                ):
                    checkpoint_util.save_model(ckpt_dir, _build_wrapped_model())
                    export_dense_model_cpu(
                        pipeline_config=pipeline_config,
                        model=ScriptWrapper(_build_model()),
                        checkpoint_path=ckpt_dir,
                        save_dir=export_dir,
                    )
            finally:
                dist.destroy_process_group()

            self.assertTrue(
                os.path.exists(os.path.join(export_dir, "scripted_model.pt"))
            )
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
