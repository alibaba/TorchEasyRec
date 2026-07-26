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
from typing import Optional
from unittest import mock

import numpy as np
import torch
from parameterized import parameterized
from torchrec.distributed.train_pipeline.utils import Tracer
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)

from tzrec.acc import utils as acc_utils
from tzrec.modules.dense_embedding_collection import (
    AutoDisEmbeddingConfig,
    DenseEmbeddingCollection,
    MLPDenseEmbeddingConfig,
)
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils.export_util import (
    _dedup_key_files_by_realpath,
    _get_dense_embedding_leaf_module_names,
    _get_sparse_embedding_tensor,
    _get_sparse_table_to_embedding_info,
    _merge_sharded_embedding_json,
    _prepare_single_rank_distributed_embedding_export,
    _prune_unused_param_and_buffer,
    export_distributed_embedding,
)
from tzrec.utils.test_util import parameterized_name_func


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


def _write_dynamic_shard(
    directory: str,
    emb_name: str = "shared_emb",
    keys: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
) -> None:
    if keys is None:
        keys = np.array([1, 2], dtype=np.int64)
    if values is None:
        values = np.array([[1.0, 1.1], [2.0, 2.1]], dtype=np.float32)
    os.makedirs(directory, exist_ok=True)
    keys.astype(np.int64).tofile(
        os.path.join(directory, f"{emb_name}_emb_keys.rank_0.world_size_1")
    )
    values.astype(np.float32).tofile(
        os.path.join(directory, f"{emb_name}_emb_values.rank_0.world_size_1")
    )
    (keys + 100).astype(np.int64).tofile(
        os.path.join(directory, f"{emb_name}_emb_scores.rank_0.world_size_1")
    )


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

    def test_sparse_table_info_is_keyed_by_owner_fqn(self) -> None:
        class SparseCollections(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = EmbeddingBagCollection(
                    [
                        EmbeddingBagConfig(
                            num_embeddings=4,
                            embedding_dim=2,
                            name="shared",
                            feature_names=["left_feat"],
                        )
                    ],
                    device=torch.device("cpu"),
                )
                self.right = EmbeddingBagCollection(
                    [
                        EmbeddingBagConfig(
                            num_embeddings=4,
                            embedding_dim=2,
                            name="shared",
                            feature_names=["right_feat"],
                        )
                    ],
                    device=torch.device("cpu"),
                )
                self.sequence = EmbeddingCollection(
                    [
                        EmbeddingConfig(
                            num_embeddings=4,
                            embedding_dim=2,
                            name="shared",
                            feature_names=["sequence_feat"],
                        )
                    ],
                    device=torch.device("cpu"),
                )

        embedding_bag_info, embedding_info = _get_sparse_table_to_embedding_info(
            SparseCollections()
        )

        self.assertEqual(
            set(embedding_bag_info),
            {
                "left.embedding_bags.shared",
                "right.embedding_bags.shared",
            },
        )
        self.assertEqual(
            set(embedding_info),
            {"sequence.embeddings.shared"},
        )

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

    def test_sparse_dynamic_export_disambiguates_owner_fqns(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_fqn_")
        old_env = {
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "DIST_QUANT": os.environ.get("DIST_QUANT"),
        }
        try:
            ckpt_dir = os.path.join(tmp, "model.ckpt-1")
            ebc_dir = os.path.join(
                ckpt_dir,
                "dynamicemb",
                "model.model.left.ebc",
            )
            ec_dir = os.path.join(
                ckpt_dir,
                "dynamicemb",
                "model.model.right.ec_dict.2",
            )
            os.makedirs(ebc_dir)
            os.makedirs(ec_dir)
            for directory, key, values in (
                (ebc_dir, 1, np.array([[1.0, 1.1]], dtype=np.float32)),
                (ec_dir, 2, np.array([[2.0, 2.1]], dtype=np.float32)),
            ):
                np.array([key], dtype=np.int64).tofile(
                    os.path.join(directory, "shared_emb_emb_keys.rank_0.world_size_1")
                )
                values.tofile(
                    os.path.join(directory, "shared_emb_emb_values.rank_0.world_size_1")
                )
                np.array([key + 100], dtype=np.int64).tofile(
                    os.path.join(directory, "shared_emb_emb_scores.rank_0.world_size_1")
                )

            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ.pop("DIST_QUANT", None)
            ebc_fqn = "model.left.ebc.embedding_bags.shared_emb"
            ec_fqn = "model.right.ec_dict.2.embeddings.shared_emb"

            _, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                {
                    ec_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["sequence_feat"],
                    )
                },
                {
                    ebc_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["pooled_feat"],
                        pooling="SUM",
                    )
                },
            )

            torch.testing.assert_close(
                dynamic_out[f"{ebc_fqn}.values"],
                torch.tensor([[1.0, 1.1]]),
            )
            torch.testing.assert_close(
                dynamic_out[f"{ec_fqn}.values"],
                torch.tensor([[2.0, 2.1]]),
            )
            self.assertEqual(set(emb_meta), {ebc_fqn, ec_fqn})
            self.assertEqual(feat_meta["pooled_feat__ebc"]["embedding_name"], ebc_fqn)
            self.assertEqual(feat_meta["sequence_feat__ec"]["embedding_name"], ec_fqn)
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_dynamic_export_maps_input_tile_table_fqns(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_input_tile_")
        old_env = {
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "DIST_QUANT": os.environ.get("DIST_QUANT"),
        }
        try:
            ckpt_dir = os.path.join(tmp, "model.ckpt-1")
            base_dir = os.path.join(
                ckpt_dir,
                "dynamicemb",
                "model.model.embedding_group.emb_impls.__BASE__.ebc",
            )
            user_dir = os.path.join(
                ckpt_dir,
                "dynamicemb",
                "model.model.embedding_group.emb_impls.__BASE__.ebc_user",
            )
            _write_dynamic_shard(base_dir)
            _write_dynamic_shard(
                user_dir,
                keys=np.array([9, 10], dtype=np.int64),
                values=np.array([[9.0, 9.1], [10.0, 10.1]], dtype=np.float32),
            )

            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ.pop("DIST_QUANT", None)
            base_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.shared_emb"
            )
            user_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc_user."
                "embedding_bags.shared_emb"
            )

            _, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                {},
                {
                    base_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["item_feat"],
                        pooling="SUM",
                    ),
                    user_fqn: SimpleNamespace(
                        name="shared_emb",
                        embedding_dim=2,
                        feature_names=["user_feat"],
                        pooling="SUM",
                    ),
                },
            )

            torch.testing.assert_close(
                dynamic_out[f"{base_fqn}.keys"], torch.tensor([1, 2])
            )
            self.assertEqual(
                set(dynamic_out),
                {
                    f"{base_fqn}.keys",
                    f"{base_fqn}.values",
                    f"{base_fqn}.scores",
                },
            )
            self.assertEqual(set(emb_meta), {base_fqn})
            self.assertNotIn(user_fqn, emb_meta)
            self.assertEqual(feat_meta["item_feat__ebc"]["embedding_name"], base_fqn)
            self.assertEqual(feat_meta["user_feat__ebc"]["embedding_name"], base_fqn)
            self.assertEqual(feat_meta["item_feat__ebc"]["pooling"], "SUM")
            self.assertEqual(feat_meta["user_feat__ebc"]["pooling"], "SUM")
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    @parameterized.expand(
        [
            ("ec_dict", "ec_dict.2", "ec_dict_user.2", "embeddings", "__ec"),
            ("ec_list", "ec_list.0", "ec_list_user.0", "embeddings", "__ec"),
            (
                "mc_ec_dict",
                "mc_ec_dict.2._embedding_module",
                "mc_ec_dict_user.2._embedding_module",
                "embeddings",
                "__ec",
            ),
            (
                "mc_ec_list",
                "mc_ec_list.0._embedding_module",
                "mc_ec_list_user.0._embedding_module",
                "embeddings",
                "__ec",
            ),
            (
                "mc_ebc",
                "mc_ebc._embedding_module",
                "mc_ebc_user._embedding_module",
                "embedding_bags",
                "__ebc",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_sparse_dynamic_export_canonicalizes_input_tile_aliases(
        self,
        _case: str,
        base_owner: str,
        user_owner: str,
        table_segment: str,
        feature_suffix: str,
    ) -> None:
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_input_tile_alias_")
        old_env = {
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "DIST_QUANT": os.environ.get("DIST_QUANT"),
        }
        try:
            ckpt_dir = os.path.join(tmp, "model.ckpt-1")
            module_prefix = "model.embedding_group.emb_impls.__BASE__"
            base_module = f"{module_prefix}.{base_owner}"
            user_module = f"{module_prefix}.{user_owner}"
            _write_dynamic_shard(
                os.path.join(ckpt_dir, "dynamicemb", f"model.{base_module}")
            )
            _write_dynamic_shard(
                os.path.join(ckpt_dir, "dynamicemb", f"model.{user_module}"),
                keys=np.array([9, 10], dtype=np.int64),
                values=np.array([[9.0, 9.1], [10.0, 10.1]], dtype=np.float32),
            )

            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ.pop("DIST_QUANT", None)
            base_fqn = f"{base_module}.{table_segment}.shared_emb"
            user_fqn = f"{user_module}.{table_segment}.shared_emb"
            base_info = SimpleNamespace(
                name="shared_emb",
                embedding_dim=2,
                feature_names=["base_feat"],
                pooling="SUM",
                data_type="FP32",
            )
            user_info = SimpleNamespace(
                name="shared_emb",
                embedding_dim=2,
                feature_names=["user_feat"],
                pooling="SUM",
                data_type="FP32",
            )
            embedding_infos = {}
            embedding_bag_info = {}
            target_infos = (
                embedding_infos if table_segment == "embeddings" else embedding_bag_info
            )
            target_infos[base_fqn] = base_info
            target_infos[user_fqn] = user_info

            out, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                torch.nn.Module(),
                ckpt_dir,
                embedding_infos,
                embedding_bag_info,
            )

            self.assertEqual(out, {})
            torch.testing.assert_close(
                dynamic_out[f"{base_fqn}.keys"], torch.tensor([1, 2])
            )
            self.assertEqual(
                set(dynamic_out),
                {
                    f"{base_fqn}.keys",
                    f"{base_fqn}.values",
                    f"{base_fqn}.scores",
                },
            )
            self.assertEqual(set(emb_meta), {base_fqn})
            self.assertNotIn(user_fqn, emb_meta)
            expected_pooling = "NONE" if feature_suffix == "__ec" else "SUM"
            for feature_name in ("base_feat", "user_feat"):
                self.assertEqual(
                    feat_meta[f"{feature_name}{feature_suffix}"],
                    {
                        "embedding_name": base_fqn,
                        "pooling": expected_pooling,
                    },
                )
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    @parameterized.expand(
        [("float", None), ("quantized", "INT8")],
        name_func=parameterized_name_func,
    )
    def test_sparse_static_export_canonicalizes_input_tile_aliases(
        self, _case: str, quantization: Optional[str]
    ) -> None:
        base_fqn = (
            "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.shared_emb"
        )
        user_fqn = (
            "model.embedding_group.emb_impls.__BASE__.ebc_user."
            "embedding_bags.shared_emb"
        )
        base_values = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
        user_values = torch.tensor([[9.0, 9.1], [10.0, 10.1]])

        class InputTileAliasModel(torch.nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
                    f"{user_fqn}.weight": user_values,
                    f"{base_fqn}.weight": base_values,
                }

        tmp = tempfile.mkdtemp(prefix="tzrec_export_static_input_tile_alias_")
        old_env = {"DIST_QUANT": os.environ.get("DIST_QUANT")}
        try:
            if quantization is None:
                os.environ.pop("DIST_QUANT", None)
            else:
                os.environ["DIST_QUANT"] = quantization
            embedding_bag_info = {
                base_fqn: SimpleNamespace(
                    name="shared_emb",
                    embedding_dim=2,
                    feature_names=["item_feat"],
                    pooling="SUM",
                    data_type="FP32",
                ),
                user_fqn: SimpleNamespace(
                    name="shared_emb",
                    embedding_dim=2,
                    feature_names=["user_feat"],
                    pooling="SUM",
                    data_type="FP32",
                ),
            }

            out, dynamic_out, emb_meta, feat_meta = _get_sparse_embedding_tensor(
                InputTileAliasModel(),
                tmp,
                {},
                embedding_bag_info,
            )

            self.assertEqual(dynamic_out, {})
            self.assertEqual(set(out), {base_fqn})
            self.assertEqual(set(emb_meta), {base_fqn})
            self.assertNotIn(user_fqn, out)
            if quantization is None:
                np.testing.assert_array_equal(out[base_fqn], base_values.numpy())
            else:
                self.assertEqual(out[base_fqn].dtype, np.uint8)
                np.testing.assert_allclose(
                    _dequant_quint8_rowwise_f16(out[base_fqn], emb_dim=2),
                    base_values.numpy(),
                    atol=5e-3,
                )
            self.assertEqual(
                emb_meta[base_fqn]["feat_name_impl"],
                ["item_feat__ebc", "user_feat__ebc"],
            )
            for feature_name in ("item_feat", "user_feat"):
                self.assertEqual(
                    feat_meta[f"{feature_name}__ebc"],
                    {"embedding_name": base_fqn, "pooling": "SUM"},
                )
        finally:
            _restore_env(old_env)
            shutil.rmtree(tmp, ignore_errors=True)

    @parameterized.expand(
        [
            ("embedding_dim", "embedding_dim", 3, "embedding_dim"),
            ("dtype", "data_type", "FP16", "dtype"),
            ("pooling", "pooling", "MEAN", "pooling"),
        ],
        name_func=parameterized_name_func,
    )
    def test_sparse_export_rejects_incompatible_input_tile_alias_configs(
        self,
        _case: str,
        field: str,
        incompatible_value: object,
        expected_field: str,
    ) -> None:
        base_fqn = "model.group.ebc.embedding_bags.shared_emb"
        user_fqn = "model.group.ebc_user.embedding_bags.shared_emb"
        base_info = {
            "name": "shared_emb",
            "embedding_dim": 2,
            "feature_names": ["item_feat"],
            "pooling": "SUM",
            "data_type": "FP32",
        }
        user_info = {
            "name": "shared_emb",
            "embedding_dim": 2,
            "feature_names": ["user_feat"],
            "pooling": "SUM",
            "data_type": "FP32",
        }
        user_info[field] = incompatible_value

        with self.assertRaisesRegex(
            ValueError,
            f"{base_fqn} and {user_fqn}.*incompatible {expected_field}",
        ):
            _get_sparse_embedding_tensor(
                torch.nn.Module(),
                "",
                {},
                {
                    base_fqn: SimpleNamespace(**base_info),
                    user_fqn: SimpleNamespace(**user_info),
                },
            )

    def test_sparse_export_rejects_input_tile_alias_table_kind_mismatch(
        self,
    ) -> None:
        base_fqn = "model.group.ebc.embedding_bags.shared_emb"
        user_fqn = "model.group.ebc_user.embedding_bags.shared_emb"
        info = SimpleNamespace(
            name="shared_emb",
            embedding_dim=2,
            feature_names=["feat"],
            pooling="SUM",
            data_type="FP32",
        )

        with self.assertRaisesRegex(
            ValueError,
            f"{user_fqn} and {base_fqn}.*incompatible table_type",
        ):
            _get_sparse_embedding_tensor(
                torch.nn.Module(),
                "",
                {user_fqn: info},
                {base_fqn: info},
            )

    def test_sparse_export_rejects_input_tile_alias_tensor_mismatch(self) -> None:
        base_fqn = "model.group.ebc.embedding_bags.shared_emb"
        user_fqn = "model.group.ebc_user.embedding_bags.shared_emb"

        class InputTileAliasModel(torch.nn.Module):
            def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return {
                    f"{base_fqn}.weight": torch.ones(2, 2),
                    f"{user_fqn}.weight": torch.ones(2, 2, dtype=torch.float64),
                }

        info = SimpleNamespace(
            name="shared_emb",
            embedding_dim=2,
            feature_names=["feat"],
            pooling="SUM",
            data_type="FP32",
        )
        with self.assertRaisesRegex(
            ValueError,
            f"{base_fqn} and {user_fqn}.*incompatible tensor shape/dtype",
        ):
            _get_sparse_embedding_tensor(
                InputTileAliasModel(),
                "",
                {},
                {base_fqn: info, user_fqn: info},
            )

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
            ebc_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.shared_emb"
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
                    )
                },
            )

            self.assertEqual(dynamic_out, {})
            self.assertNotIn("shared_emb", out)
            np.testing.assert_array_equal(
                out[ec_fqn],
                np.array([[2.0, 2.1], [2.2, 2.3]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                out[ebc_fqn],
                np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            )
            self.assertEqual(emb_meta[ec_fqn]["feat_name_impl"], ["seq_feat__ec"])
            self.assertEqual(emb_meta[ebc_fqn]["feat_name_impl"], ["id_feat__ebc"])
            self.assertEqual(
                feat_meta["seq_feat__ec"],
                {"embedding_name": ec_fqn, "pooling": "NONE"},
            )
            self.assertEqual(
                feat_meta["id_feat__ebc"],
                {"embedding_name": ebc_fqn, "pooling": "SUM"},
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
            ebc_fqn = (
                "model.embedding_group.emb_impls.__BASE__.ebc.embedding_bags.shared_emb"
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
                    )
                },
            )

            self.assertEqual(dynamic_out, {})
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


if __name__ == "__main__":
    unittest.main()
