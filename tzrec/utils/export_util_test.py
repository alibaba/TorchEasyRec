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
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torchrec.distributed.train_pipeline.utils import Tracer

from tzrec.acc import utils as acc_utils
from tzrec.modules.dense_embedding_collection import (
    AutoDisEmbeddingConfig,
    DenseEmbeddingCollection,
    MLPDenseEmbeddingConfig,
)
from tzrec.utils.export_util import (
    _dedup_key_files_by_realpath,
    _get_dense_embedding_leaf_module_names,
    _get_sparse_embedding_tensor,
    _merge_sharded_embedding_json,
    _prepare_single_rank_distributed_embedding_export,
    _prune_unused_param_and_buffer,
    export_distributed_embedding,
)


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
        old_env = {"QUANT": os.environ.get("QUANT")}
        try:
            for value in (None, "", "0", "NONE", "none"):
                if value is None:
                    os.environ.pop("QUANT", None)
                else:
                    os.environ["QUANT"] = value
                self.assertFalse(acc_utils.is_distributed_sparse_quant())
                self.assertEqual(acc_utils.distributed_sparse_quant_format(), "")
                self.assertNotIn("QUANT", acc_utils.export_acc_config())

            os.environ["QUANT"] = "INT8"
            self.assertTrue(acc_utils.is_distributed_sparse_quant())
            self.assertEqual(
                acc_utils.distributed_sparse_quant_format(), "QUint8RowwiseF16"
            )
            self.assertEqual(acc_utils.export_acc_config()["QUANT"], "INT8")

            os.environ["QUANT"] = "FP16"
            with self.assertRaisesRegex(ValueError, "Unsupported QUANT"):
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

    def test_sparse_dynamic_embedding_export_concats_training_shards(self) -> None:
        """Single-rank export must not drop multi-GPU dynamicemb checkpoint shards."""
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_")
        old_rank = os.environ.get("RANK")
        old_world_size = os.environ.get("WORLD_SIZE")
        old_quant = os.environ.get("QUANT")
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
            os.environ.pop("QUANT", None)
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
                os.environ.pop("QUANT", None)
            else:
                os.environ["QUANT"] = old_quant
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sparse_dynamic_embedding_quant_export(self) -> None:
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_quant_")
        old_env = {
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "QUANT": os.environ.get("QUANT"),
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
            os.environ["QUANT"] = "INT8"
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
        old_env = {"QUANT": os.environ.get("QUANT")}
        try:
            os.environ.pop("QUANT", None)
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

        old_env = {"QUANT": os.environ.get("QUANT")}
        tmp = tempfile.mkdtemp(prefix="tzrec_export_sparse_quant_")
        try:
            os.environ["QUANT"] = "INT8"
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

        old_env = {"QUANT": os.environ.get("QUANT")}
        tmp = tempfile.mkdtemp(prefix="tzrec_export_sparse_quant_odd_")
        try:
            os.environ["QUANT"] = "INT8"
            with self.assertRaisesRegex(
                ValueError,
                "user_id_emb.*embedding_dim \\+ 4 = 3 \\+ 4 = 7.*"
                "change the table's embedding_dim to an even value.*"
                "QUANT=0/NONE",
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


if __name__ == "__main__":
    unittest.main()
