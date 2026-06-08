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
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torchrec.distributed.train_pipeline.utils import Tracer

from tzrec.modules.dense_embedding_collection import (
    AutoDisEmbeddingConfig,
    DenseEmbeddingCollection,
    MLPDenseEmbeddingConfig,
)
from tzrec.utils.export_util import (
    _get_dense_embedding_leaf_module_names,
    _get_sparse_embedding_tensor,
    _prune_unused_param_and_buffer,
)


class ExportUtilTest(unittest.TestCase):
    def test_sparse_dynamic_embedding_export_concats_training_shards(self) -> None:
        """Single-rank export must not drop multi-GPU dynamicemb checkpoint shards."""
        tmp = tempfile.mkdtemp(prefix="tzrec_export_dynemb_")
        old_rank = os.environ.get("RANK")
        old_world_size = os.environ.get("WORLD_SIZE")
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
            shutil.rmtree(tmp, ignore_errors=True)

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
