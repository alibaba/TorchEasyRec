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

import torch
from pyarrow import dataset as ds

from tzrec.constant import Mode
from tzrec.main import _create_features, _get_dataloader
from tzrec.tests import utils
from tzrec.utils import config_util


class TrainEvalExportTest(unittest.TestCase):
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
        os.environ.pop("QUANT_EMB", None)
        os.environ.pop("INPUT_TILE", None)

    def test_multi_tower_din_fg_encoded_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_mock.config", self.test_dir
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
                scripted_model_path=os.path.join(self.test_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "predict_result"),
                reserved_columns="clk",
                output_columns="probs",
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )

    def test_multi_tower_din_fg_encoded_finetune(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_mock.config",
            os.path.join(self.test_dir, "1"),
        )
        if self.success:
            self.success = utils.test_train_eval(
                "tzrec/tests/configs/multi_tower_din_mock.config",
                os.path.join(self.test_dir, "2"),
                f"--fine_tune_checkpoint {os.path.join(self.test_dir, '1')}",
            )
        self.assertTrue(self.success)

    def test_dssm_fg_encoded_train_eval_export(self):
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

    def test_dssm_fg_encoded_variational_dropout(self):
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

    def test_multi_tower_din_with_fg_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_fg_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
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
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )

        pipeline_config = config_util.load_pipeline_config(
            os.path.join(self.test_dir, "pipeline.config")
        )
        features = _create_features(
            pipeline_config.feature_configs, pipeline_config.data_config
        )
        dataloader = _get_dataloader(
            pipeline_config.data_config,
            features,
            pipeline_config.train_input_path,
            mode=Mode.PREDICT,
        )
        iterator = iter(dataloader)
        data = next(iterator)
        device = "cpu"
        model_cpu = torch.jit.load(
            os.path.join(self.test_dir, "export/scripted_model.pt"), map_location=device
        )
        result_cpu = model_cpu(data.to_dict(sparse_dtype=torch.int64))
        device = "cuda:0"
        model_gpu = torch.jit.load(
            os.path.join(self.test_dir, "export/scripted_model.pt"), map_location=device
        )
        result_gpu = model_gpu(data.to_dict(sparse_dtype=torch.int32), device)
        for k, v in result_gpu.items():
            torch.testing.assert_close(
                result_cpu[k].to(device), v, rtol=5e-3, atol=1e-5
            )

    def test_dssm_with_fg_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/dssm_fg_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
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
            os.path.exists(os.path.join(self.test_dir, "export/user/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/user/faiss_index"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/user/id_mapping"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/item/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "fg_output/fg.json"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_dir, "fg_output/item_title_tokenizer.json")
            )
        )

    def test_dssm_v2_with_fg_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/dssm_v2_fg_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
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
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/user/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/item/scripted_model.pt"))
        )

    def test_multi_tower_din_with_fg_train_eval_export_input_tile(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_fg_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )

        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )

        no_quant_dir = os.path.join(self.test_dir, "no_quant")
        input_tile_dir = os.path.join(self.test_dir, "input_tile")
        input_tile_dir_emb = os.path.join(self.test_dir, "input_tile_emb")
        pred_output = os.path.join(self.test_dir, "predict_result")
        tile_pred_output = os.path.join(self.test_dir, "predict_result_tile")

        # quant and no-input-tile
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(self.test_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=pred_output,
                reserved_columns="user_id,item_id,clk",
                output_columns="probs",
                test_dir=self.test_dir,
            )

        # no-quant and no-input-tile
        if self.success:
            os.environ["QUANT_EMB"] = "0"
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"), no_quant_dir
            )

        # quant and input-tile
        if self.success:
            os.environ["QUANT_EMB"] = "1"
            os.environ["INPUT_TILE"] = "2"
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"), input_tile_dir
            )
        if self.success:
            # we should not set INPUT_TILE env when predict
            os.environ.pop("QUANT_EMB", None)
            os.environ.pop("INPUT_TILE", None)
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(input_tile_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=tile_pred_output,
                reserved_columns="user_id,item_id,clk",
                output_columns="probs",
                test_dir=input_tile_dir,
            )
            # compare INPUT_TILE and no INPUT_TILE result consistency
            df = ds.dataset(pred_output, format="parquet").to_table().to_pandas()
            df_t = ds.dataset(tile_pred_output, format="parquet").to_table().to_pandas()
            df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
            df_t = df_t.sort_values(by=list(df_t.columns)).reset_index(drop=True)
            self.assertTrue(df.equals(df_t))

        # quant and input-tile emb
        if self.success:
            os.environ["QUANT_EMB"] = "1"
            os.environ["INPUT_TILE"] = "3"
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"), input_tile_dir_emb
            )

        self.assertTrue(self.success)

        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(no_quant_dir, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(input_tile_dir, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(input_tile_dir_emb, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(input_tile_dir_emb, "export/model_acc.json"))
        )
        with open(os.path.join(input_tile_dir_emb, "export/model_acc.json")) as f:
            acc_cfg = json.load(f)
            self.assertEqual(acc_cfg["QUANT_EMB"], "1")
            self.assertEqual(acc_cfg["INPUT_TILE"], "3")

        pipeline_config = config_util.load_pipeline_config(
            os.path.join(self.test_dir, "pipeline.config")
        )
        features = _create_features(
            pipeline_config.feature_configs, pipeline_config.data_config
        )
        utils.create_predict_data(
            pipeline_config_path=os.path.join(self.test_dir, "pipeline.config"),
            batch_size=512,
            item_id="item_id",
            output_dir=os.path.join(self.test_dir, "predict"),
        )
        dataloader = _get_dataloader(
            pipeline_config.data_config,
            features,
            os.path.join(self.test_dir, "predict", "*.parquet"),
            mode=Mode.PREDICT,
        )

        os.environ["INPUT_TILE"] = "1"
        iterator = iter(dataloader)
        data = next(iterator)
        device = "cpu"

        # quant and no-input-tile
        model_cpu = torch.jit.load(
            os.path.join(self.test_dir, "export/scripted_model.pt"), map_location=device
        )
        result_cpu = model_cpu(data.to_dict(sparse_dtype=torch.int64))
        device = "cuda:0"
        model_gpu = torch.jit.load(
            os.path.join(self.test_dir, "export/scripted_model.pt"), map_location=device
        )
        result_gpu = model_gpu(data.to_dict(sparse_dtype=torch.int32), device)
        result_dict_json_path = os.path.join(self.test_dir, "result_gpu.json")
        utils.save_predict_result_json(result_gpu, result_dict_json_path)
        for k, v in result_gpu.items():
            torch.testing.assert_close(
                result_cpu[k].to(device), v, rtol=5e-3, atol=1e-5
            )

        # no-quant and no-input-tile
        model_gpu_no_quant = torch.jit.load(
            os.path.join(self.test_dir, "no_quant/export/scripted_model.pt"),
            map_location=device,
        )
        result_gpu_no_quant = model_gpu_no_quant(
            data.to_dict(sparse_dtype=torch.int32), device
        )
        result_dict_json_path = os.path.join(self.test_dir, "result_gpu_no_quant.json")
        utils.save_predict_result_json(result_gpu_no_quant, result_dict_json_path)

        # quant and input-tile
        model_gpu_input_tile = torch.jit.load(
            os.path.join(self.test_dir, "input_tile/export/scripted_model.pt"),
            map_location=device,
        )
        os.environ["INPUT_TILE"] = "2"
        dataloader = _get_dataloader(
            pipeline_config.data_config,
            features,
            os.path.join(self.test_dir, "predict", "*.parquet"),
            mode=Mode.PREDICT,
        )
        iterator_input_tile = iter(dataloader)
        data_input_tile = next(iterator_input_tile)
        result_gpu_input_tile = model_gpu_input_tile(
            data_input_tile.to(device=device).to_dict(sparse_dtype=torch.int32), device
        )
        result_dict_json_path = os.path.join(
            self.test_dir, "result_gpu_input_tile.json"
        )
        utils.save_predict_result_json(result_gpu_input_tile, result_dict_json_path)

        # quant and input-tile emb
        model_gpu_input_tile_emb = torch.jit.load(
            os.path.join(self.test_dir, "input_tile_emb/export/scripted_model.pt"),
            map_location=device,
        )
        result_gpu_input_tile_emb = model_gpu_input_tile_emb(
            data_input_tile.to(device=device).to_dict(sparse_dtype=torch.int32), device
        )

        # tile is all same sa no-tile
        for k, v in result_gpu.items():
            torch.testing.assert_close(result_gpu_input_tile[k].to(device), v)

        # tile emb is all same sa no-tile
        for k, v in result_gpu.items():
            torch.testing.assert_close(result_gpu_input_tile_emb[k].to(device), v)

        # tile is all same, the atol is because the quant
        for k, v in result_gpu_no_quant.items():
            torch.testing.assert_close(
                result_gpu_input_tile[k].to(device), v, rtol=1e-4, atol=1e-4
            )

    def test_dbmtl_has_sequence_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/dbmtl_has_sequence_mock.config",
            self.test_dir,
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
                scripted_model_path=os.path.join(self.test_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "predict_result"),
                reserved_columns="clk,buy",
                output_columns="probs_ctr,probs_cvr",
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )

    def test_dbmtl_has_sequence_variational_dropout_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/dbmtl_has_sequence_variational_dropout_mock.config",
            self.test_dir,
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
            self.success = utils.test_feature_selection(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )

        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "output_dir/pipeline.config"))
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
            with open(os.path.join(self.test_dir, "init_tree/node_table.txt")) as f:
                for line_number, line in enumerate(f):
                    if line_number == 1:
                        root_id = int(line.split("\t")[0])
                        break
            self.success = utils.test_tdm_retrieval(
                scripted_model_path=os.path.join(self.test_dir, "export"),
                eval_data_path=os.path.join(self.test_dir, "eval_data/*.parquet"),
                retrieval_output_path=os.path.join(self.test_dir, "retrieval_result"),
                item_id="item_id",
                gt_item_id="gt_item_id",
                root_id=root_id,
                reserved_columns="user_id,gt_item_id",
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
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/serving_tree"))
        )
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "retrieval_result")))


if __name__ == "__main__":
    unittest.main()
