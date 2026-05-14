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

import json
import os
import shutil
import tempfile
import unittest

from tzrec.tests import utils
from tzrec.utils.test_util import gpu_unavailable, mark_ci_scope


class MatchIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
        os.chmod(self.test_dir, 0o755)

    def tearDown(self):
        if self.success:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)

    def _test_match_train_eval_export(
        self,
        pipeline_config_path: str,
        **train_eval_kwargs: str,
    ) -> None:
        """Common match-model train -> eval -> export flow (non-AOT).

        Non-AOT export produces ``scripted_model.pt`` in each tower's
        export dir.

        Downstream predict / faiss / hitrate coverage already lives in
        ``test_dssm_with_fg_train_eval_export``; kept out of this helper
        to keep the matrix lean.
        """
        self.success = utils.test_train_eval(
            pipeline_config_path, self.test_dir, **train_eval_kwargs
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        self.assertTrue(self.success)
        for side in ("user", "item"):
            self.assertTrue(
                os.path.exists(
                    os.path.join(self.test_dir, f"export/{side}/scripted_model.pt")
                ),
                f"missing scripted model for {side} tower",
            )

    def _test_match_train_eval_export_aot(
        self,
        pipeline_config_path: str,
        **train_eval_kwargs: str,
    ) -> None:
        """Match-model train -> eval -> AOT (two-stage) export flow.

        Exercises ``_compute_seq_share_groups`` and
        ``_build_dynamic_shapes`` on the tower-wrapper path. Two-stage
        AOT export produces ``scripted_sparse_model.pt`` plus a
        ``model_acc.json`` recording ``ENABLE_AOT=1`` in each tower's
        export dir.
        """
        self.success = utils.test_train_eval(
            pipeline_config_path, self.test_dir, **train_eval_kwargs
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                self.test_dir,
                env_str="ENABLE_AOT=1",
            )
        self.assertTrue(self.success)
        for side in ("user", "item"):
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        self.test_dir, f"export/{side}/scripted_sparse_model.pt"
                    )
                ),
                f"missing AOT sparse model for {side} tower",
            )
            with open(
                os.path.join(self.test_dir, f"export/{side}/model_acc.json")
            ) as f:
                self.assertEqual(json.load(f).get("ENABLE_AOT"), "1")

    def test_dssm_nofg_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/dssm_mock.config", self.test_dir, item_id="item_id"
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export/item"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "predict_result"),
                reserved_columns="item_id",
                output_columns="item_tower_emb",
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/model.ckpt-63"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/user/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/item/scripted_model.pt"))
        )

    def test_dssm_nofg_variational_dropout(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/dssm_variational_dropout_mock.config",
            self.test_dir,
            item_id="item_id",
        )
        if self.success:
            self.success = utils.test_feature_selection(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/model.ckpt-63"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "output_dir/pipeline.config"))
        )

    def test_dssm_with_fg_train_eval_export(self):
        self._test_match_train_eval_export(
            "tzrec/tests/configs/dssm_fg_mock.config",
            user_id="user_id",
            item_id="item_id",
        )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export/item"),
                predict_input_path=os.path.join(self.test_dir, r"item_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "item_emb"),
                reserved_columns="item_id",
                output_columns="item_tower_emb",
                test_dir=self.test_dir,
            )
        if self.success:
            self.success = utils.test_create_faiss_index(
                embedding_input_path=os.path.join(
                    self.test_dir, r"item_emb/\*.parquet"
                ),
                index_output_dir=os.path.join(self.test_dir, "export/user"),
                id_field="item_id",
                embedding_field="item_tower_emb",
                test_dir=self.test_dir,
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export/user"),
                predict_input_path=os.path.join(self.test_dir, r"user_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "user_emb"),
                reserved_columns="user_id,click_50_seq__item_id",
                output_columns="user_tower_emb",
                test_dir=self.test_dir,
            )
        if self.success:
            self.success = utils.test_hitrate(
                user_gt_input=os.path.join(self.test_dir, r"user_emb/\*.parquet"),
                item_embedding_input=os.path.join(
                    self.test_dir, r"item_emb/\*.parquet"
                ),
                total_hitrate_output=os.path.join(self.test_dir, "total_hitrate"),
                hitrate_details_output=os.path.join(self.test_dir, "hitrate_details"),
                request_id_field="user_id",
                gt_items_field="click_50_seq__item_id",
                test_dir=self.test_dir,
            )
        if self.success:
            self.success = utils.test_create_fg_json(
                os.path.join(self.test_dir, "pipeline.config"),
                fg_output_dir=os.path.join(self.test_dir, "fg_output"),
                reserves="clk",
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/user/faiss_index"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/user/id_mapping"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "fg_output/fg.json"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.test_dir,
                    "fg_output/tokenizer_b2faab7921bbfb593973632993ca4c85.json",
                )
            )
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_dssm_with_fg_train_eval_export_aot(self):
        # AOT variant exercises TowerWrapper through the helpers refactored
        # to take feature_groups (_compute_seq_share_groups +
        # _build_dynamic_shapes).
        self._test_match_train_eval_export_aot(
            "tzrec/tests/configs/dssm_fg_mock.config",
            user_id="user_id",
            item_id="item_id",
        )

    def test_dssm_hard_negative_with_fg_train_eval_export(self):
        # Downstream predict / faiss / hitrate coverage already lives in
        # test_dssm_with_fg_train_eval_export.
        self._test_match_train_eval_export(
            "tzrec/tests/configs/dssm_fg_hard_negative_mock.config",
            user_id="user_id",
            item_id="item_id",
        )

    def test_dssm_v2_with_fg_train_eval_export(self):
        self._test_match_train_eval_export(
            "tzrec/tests/configs/dssm_v2_fg_mock.config",
            user_id="user_id",
            item_id="item_id",
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_dssm_v2_with_fg_train_eval_export_aot(self):
        # AOT variant exercises TowerWoEGWrapper through the helpers
        # refactored to take feature_groups.
        self._test_match_train_eval_export_aot(
            "tzrec/tests/configs/dssm_v2_fg_mock.config",
            user_id="user_id",
            item_id="item_id",
        )

    def test_dssm_v2_mlp_emb_with_fg_train_eval_export(self):
        self._test_match_train_eval_export(
            "tzrec/tests/configs/dssm_v2_mlpemb_fg_mock.config",
            user_id="user_id",
            item_id="item_id",
        )

    def test_tdm_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/tdm_fg_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
            cate_id="cate_id",
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                self.test_dir,
                asset_files=os.path.join(self.test_dir, "init_tree/serving_tree"),
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export/embedding"),
                predict_input_path=os.path.join(self.test_dir, r"item_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "item_emb"),
                reserved_columns="item_id,cate_id,id_4,id_5,raw_1,raw_2",
                output_columns="item_emb",
                test_dir=self.test_dir,
            )
        if self.success:
            self.success = utils.test_tdm_cluster_train_eval(
                pipeline_config_path=os.path.join(self.test_dir, "pipeline.config"),
                test_dir=self.test_dir,
                item_input_path=os.path.join(self.test_dir, r"item_emb/\*.parquet"),
                item_id="item_id",
                embedding_field="item_emb",
            )
        if self.success:
            self.success = utils.test_tdm_retrieval(
                scripted_model_path=os.path.join(self.test_dir, "export/model"),
                eval_data_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                retrieval_output_path=os.path.join(self.test_dir, "retrieval_result"),
                reserved_columns="user_id,item_id",
                test_dir=self.test_dir,
            )

        self.assertTrue(self.success)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "learnt_tree")))
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "export/embedding/scripted_model.pt")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "export/model/scripted_model.pt")
            )
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/model/serving_tree"))
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "retrieval_result")))

    def test_dat_train_eval_export(self):
        self._test_match_train_eval_export(
            "tzrec/tests/configs/dat_mock.config",
            item_id="item_id",
        )

    def test_mind_train_eval_export(self):
        self._test_match_train_eval_export(
            "tzrec/tests/configs/mind_mock.config",
            item_id="item_id",
        )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export/item"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "predict_result"),
                reserved_columns="item_id",
                output_columns="item_tower_emb",
                test_dir=self.test_dir,
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export/user"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "user_predict_result"),
                reserved_columns="user_id,click_50_seq__item_id_1,click_50_seq__item_id_2",
                output_columns="user_tower_emb",
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)

    @mark_ci_scope("h20")
    @unittest.skipIf(*gpu_unavailable)
    def test_hstu_with_fg_train_eval(self):
        # Train + eval end-to-end on real data; verifies the full HSTUMatch
        # pipeline (UIHPreprocessor -> HSTUPositionalEncoder -> STUStack ->
        # OutputPostprocessor + the row-(B-1) suffix candidate scoring).
        # Export + predict coverage lives in hstu_test (FX_TRACE / JIT_SCRIPT
        # graph types) because sequence_id_feature parsing requires a
        # sampler-fed delimited string and Mode.PREDICT has no sampler.
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/hstu_kuairand_1k.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        self.assertTrue(self.success)


if __name__ == "__main__":
    unittest.main()
