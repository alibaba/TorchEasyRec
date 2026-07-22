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
from torchrec.distributed.train_pipeline.utils import Tracer
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

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
from tzrec.utils import checkpoint_util, misc_util
from tzrec.utils.export_util import (
    _dedup_key_files_by_realpath,
    _get_dense_embedding_leaf_module_names,
    _get_sparse_embedding_tensor,
    _infer_keyed_tensor_attrs_from_module,
    _isolate_kafka_export_group,
    _merge_sharded_embedding_json,
    _prepare_single_rank_distributed_embedding_export,
    _prune_unused_param_and_buffer,
    export_dense_model_cpu,
    export_distributed_embedding,
)
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import make_test_dir


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
                    "tzrec.utils.export_util._get_sparse_feature_to_embedding_info",
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
            embedding_bag_info = [
                SimpleNamespace(
                    name="user_id_emb",
                    embedding_dim=2,
                    feature_names=["user_id"],
                    pooling="SUM",
                )
            ]

            _, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                [],
                embedding_bag_info,
            )

            torch.testing.assert_close(
                dynamic_out["user_id_emb.keys"], torch.tensor([0, 2, 1, 3])
            )
            torch.testing.assert_close(
                dynamic_out["user_id_emb.scores"], torch.tensor([100, 102, 101, 103])
            )
            torch.testing.assert_close(
                dynamic_out["user_id_emb.values"],
                torch.tensor([[0.0, 0.1], [2.0, 2.1], [1.0, 1.1], [3.0, 3.1]]),
            )
            self.assertEqual(emb_meta["user_id_emb"]["shape"], [4, 2])
            self.assertEqual(emb_meta["user_id_emb"]["key_name"], "user_id_emb.keys")
            self.assertEqual(
                emb_meta["user_id_emb"]["value_name"], "user_id_emb.values"
            )
            self.assertEqual(
                emb_meta["user_id_emb"]["score_name"], "user_id_emb.scores"
            )
            self.assertEqual(
                feat_meta["user_id__ebc"],
                {"embedding_name": "user_id_emb", "pooling": "SUM"},
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
            embedding_bag_info = [
                SimpleNamespace(
                    name="user_id_emb",
                    embedding_dim=2,
                    feature_names=["user_id"],
                    pooling="SUM",
                )
            ]

            _, dynamic_out, emb_meta, _ = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                [],
                embedding_bag_info,
            )

            torch.testing.assert_close(
                dynamic_out["user_id_emb.keys"], torch.tensor([0, 1])
            )
            torch.testing.assert_close(
                dynamic_out["user_id_emb.scores"], torch.tensor([100, 101])
            )
            self.assertEqual(dynamic_out["user_id_emb.values"].dtype, np.uint8)
            self.assertEqual(dynamic_out["user_id_emb.values"].shape, (2, 6))
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(
                    dynamic_out["user_id_emb.values"], emb_dim=2
                ),
                values,
                atol=5e-3,
            )
            self.assertEqual(emb_meta["user_id_emb"]["dtype"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta["user_id_emb"]["shape"], [2, 2])
            self.assertEqual(emb_meta["user_id_emb"]["storage_shape"], [2, 6])
            self.assertEqual(emb_meta["user_id_emb"]["row_bytes"], 6)
            self.assertEqual(
                emb_meta["user_id_emb"]["quant"]["format"], "QUint8RowwiseF16"
            )
            self.assertEqual(
                emb_meta["user_id_emb"]["value_name"], "user_id_emb.values"
            )
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
            out, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                SparseCollisionModel(),
                tmp,
                [
                    SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["seq_feat"],
                    )
                ],
                [
                    SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["id_feat"],
                        pooling="SUM",
                    )
                ],
            )

            self.assertEqual(dynamic_out, {})
            self.assertNotIn("shared_emb", out)
            np.testing.assert_array_equal(
                out["shared_emb__ec"],
                np.array([[2.0, 2.1], [2.2, 2.3]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                out["shared_emb__ebc"],
                np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            )
            self.assertEqual(
                emb_meta["shared_emb__ec"]["feat_name_impl"], ["seq_feat__ec"]
            )
            self.assertEqual(
                emb_meta["shared_emb__ebc"]["feat_name_impl"], ["id_feat__ebc"]
            )
            self.assertEqual(
                feat_meta["seq_feat__ec"],
                {"embedding_name": "shared_emb__ec", "pooling": "NONE"},
            )
            self.assertEqual(
                feat_meta["id_feat__ebc"],
                {"embedding_name": "shared_emb__ebc", "pooling": "SUM"},
            )
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_export_quantizes_ec_and_ebc_weights(self) -> None:
        class SparseCollisionModel(torch.nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
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
            out, dynamic_out, emb_meta, _ = _get_sparse_embedding_tensor(
                SparseCollisionModel(),
                tmp,
                [
                    SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["seq_feat"],
                    )
                ],
                [
                    SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["id_feat"],
                        pooling="SUM",
                    )
                ],
            )

            self.assertEqual(dynamic_out, {})
            self.assertEqual(out["shared_emb__ec"].dtype, np.uint8)
            self.assertEqual(out["shared_emb__ebc"].dtype, np.uint8)
            self.assertEqual(out["shared_emb__ec"].shape, (2, 6))
            self.assertEqual(out["shared_emb__ebc"].shape, (2, 6))
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(out["shared_emb__ec"], emb_dim=2),
                np.array([[-3.0, 3.0], [-4.0, 4.0]], dtype=np.float32),
                atol=5e-3,
            )
            np.testing.assert_allclose(
                _dequant_quint8_rowwise_f16(out["shared_emb__ebc"], emb_dim=2),
                np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32),
                atol=5e-3,
            )
            self.assertEqual(emb_meta["shared_emb__ec"]["dtype"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta["shared_emb__ec"]["shape"], [2, 2])
            self.assertEqual(emb_meta["shared_emb__ec"]["storage_shape"], [2, 6])
            self.assertEqual(emb_meta["shared_emb__ec"]["row_bytes"], 6)
            self.assertEqual(emb_meta["shared_emb__ebc"]["dtype"], "QUint8RowwiseF16")
            self.assertEqual(emb_meta["shared_emb__ebc"]["shape"], [2, 2])
            self.assertEqual(emb_meta["shared_emb__ebc"]["storage_shape"], [2, 6])
            self.assertEqual(emb_meta["shared_emb__ebc"]["row_bytes"], 6)
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
                _get_sparse_embedding_tensor(
                    OddDimModel(),
                    tmp,
                    [],
                    [
                        SimpleNamespace(
                            name="user_id_emb",
                            embedding_dim=3,
                            feature_names=["user_id"],
                            pooling="SUM",
                        )
                    ],
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


if __name__ == "__main__":
    unittest.main()
