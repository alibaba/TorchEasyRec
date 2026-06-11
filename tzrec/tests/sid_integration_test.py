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

import glob
import json
import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tzrec.tests import utils
from tzrec.utils import config_util


class SidIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
        os.chmod(self.test_dir, 0o755)
        # SidRqkmeans is single-process; pin nproc=1 (the CI harness defaults
        # to 2, which would trip the world_size>1 guard).
        patcher = mock.patch.dict(os.environ, {"TEST_NPROC_PER_NODE": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if self.success and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _prepare_config(self, num_rows: int, dim: int) -> str:
        """Write an embedding parquet + a SID config pointed at it.

        Single dense ``embedding`` column, no labels — SID reads the item
        embedding straight from the batch. Returns the saved config path.
        """
        data_dir = os.path.join(self.test_dir, "sid_data")
        os.makedirs(data_dir, exist_ok=True)
        emb = np.random.rand(num_rows, dim).astype(np.float32)
        pq.write_table(
            pa.table({"embedding": pa.array(list(emb))}),
            os.path.join(data_dir, "part-0.parquet"),
        )
        data_glob = os.path.join(data_dir, "*.parquet")

        # train_input_path set -> load_config_for_test uses it as-is (the
        # FG_DAG auto-mock path is match-model-specific; SID is single-table).
        config = config_util.load_pipeline_config(
            "tzrec/tests/configs/sid_rqkmeans_mock.config"
        )
        config.train_input_path = data_glob
        config.eval_input_path = data_glob
        config_path = os.path.join(self.test_dir, "sid.config")
        config_util.save_message(config, config_path)
        return config_path

    @unittest.skipIf(
        torch.cuda.is_available(),
        "SidRqkmeans is CPU-only; this end-to-end test runs on the CPU CI job. "
        "Forcing CPU on a CUDA-built (GPU) image is unreliable.",
    )
    def test_sid_rqkmeans_train_eval(self):
        """End-to-end train -> on_train_end FAISS fit -> checkpoint -> eval.

        Locks down the load-bearing path: the codebook exists only after
        ``on_train_end``, which forces the final checkpoint; the post-fit eval
        then reports finite reconstruction metrics.
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        config_path = self._prepare_config(num_rows=2048, dim=16)

        self.success = utils.test_train_eval(config_path, self.test_dir)
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        self.assertTrue(self.success)
        # on_train_end fitted the codebook and forced a final checkpoint.
        self.assertTrue(
            glob.glob(os.path.join(self.test_dir, "train", "model.ckpt-*")),
            "no checkpoint persisted after on_train_end",
        )
        # A fitted codebook yields finite metrics; a degenerate/unfitted one
        # never exposes x_hat -> metrics stay NaN. So assert finiteness, plus
        # rel_loss < 1.0 (all-zero baseline ~ 1.0) and nonzero SID variety.
        result_path = os.path.join(self.test_dir, "train", "eval_result.txt")
        self.assertTrue(os.path.exists(result_path), "no eval_result.txt produced")
        with open(result_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        self.assertTrue(lines, "eval_result.txt is empty")
        metrics = json.loads(lines[-1])
        for key in ("mse", "rel_loss", "unique_sid_ratio"):
            self.assertIn(key, metrics)
            self.assertTrue(
                math.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"
            )
        self.assertLess(metrics["rel_loss"], 1.0)
        self.assertGreater(metrics["unique_sid_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
