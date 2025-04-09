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
from tzrec.utils.test_util import dfs_are_close, gpu_unavailable, nv_gpu_unavailable


class RankIntegrationTest(unittest.TestCase):
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
        os.environ.pop("ENABLE_TRT", None)

    def _test_rank_nofg(self, pipeline_config_path, reserved_columns, output_columns):
        self.success = utils.test_train_eval(pipeline_config_path, self.test_dir)
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
                reserved_columns=reserved_columns,
                output_columns=output_columns,
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_aot_export(self):
        pipeline_config_path = "tzrec/tests/configs/multi_tower_din_fg_mock.config"
        self.success = utils.test_train_eval(
            pipeline_config_path, self.test_dir, user_id="user_id", item_id="item_id"
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
        input_tile_dir = os.path.join(self.test_dir, "input_tile")
        input_tile_dir_emb = os.path.join(self.test_dir, "input_tile_emb")
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                input_tile_dir,
                env_str="INPUT_TILE=2 ENABLE_AOT=1",
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                input_tile_dir_emb,
                env_str="INPUT_TILE=3 ENABLE_AOT=1",
            )
        self.assertTrue(self.success)

    def test_multi_tower_din_fg_encoded_train_eval_export(self):
        self._test_rank_nofg(
            "tzrec/tests/configs/multi_tower_din_mock.config",
            reserved_columns="clk",
            output_columns="probs",
        )

    def test_dbmtl_has_sequence_fg_encoded_train_eval_export(self):
        self._test_rank_nofg(
            "tzrec/tests/configs/dbmtl_has_sequence_mock.config",
            reserved_columns="clk,buy",
            output_columns="probs_ctr,probs_cvr",
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

    def _test_rank_with_fg(self, pipeline_config_path, comp_cpu_gpu_pred_result=False):
        self.success = utils.test_train_eval(
            pipeline_config_path,
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
                scripted_model_path=os.path.join(self.test_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "predict_result"),
                reserved_columns="user_id,item_id",
                output_columns="",
                test_dir=self.test_dir,
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )
        if comp_cpu_gpu_pred_result and torch.cuda.is_available():
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
                os.path.join(self.test_dir, "export/scripted_model.pt"),
                map_location=device,
            )
            result_cpu = model_cpu(data.to_dict(sparse_dtype=torch.int64))
            device = "cuda:0"
            model_gpu = torch.jit.load(
                os.path.join(self.test_dir, "export/scripted_model.pt"),
                map_location=device,
            )
            result_gpu = model_gpu(data.to_dict(sparse_dtype=torch.int64), device)
            for k, v in result_gpu.items():
                torch.testing.assert_close(
                    result_cpu[k].to(device), v, rtol=5e-3, atol=1e-5
                )

    def _test_rank_with_fg_quant(self, pipeline_config_path):
        self.success = utils.test_train_eval(
            pipeline_config_path,
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
        )
        for quant_emb in ["FP32", "FP16", "INT8", "INT4", "INT2"]:
            test_dir = os.path.join(self.test_dir, f"quant_{quant_emb.lower()}")
            if self.success:
                self.success = utils.test_export(
                    os.path.join(self.test_dir, "pipeline.config"),
                    test_dir,
                    env_str=f"QUANT_EMB={quant_emb}",
                )
            if self.success:
                self.success = utils.test_predict(
                    scripted_model_path=os.path.join(test_dir, "export"),
                    predict_input_path=os.path.join(
                        self.test_dir, r"eval_data/\*.parquet"
                    ),
                    predict_output_path=os.path.join(test_dir, "predict_result"),
                    reserved_columns="user_id,item_id",
                    output_columns="",
                    test_dir=test_dir,
                )
            self.assertTrue(self.success)
            self.assertTrue(
                os.path.exists(os.path.join(test_dir, "export/scripted_model.pt"))
            )

    def _test_rank_with_fg_input_tile(
        self,
        pipeline_config_path,
    ):
        self.success = utils.test_train_eval(
            pipeline_config_path,
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
        tile_pred_output_emb = os.path.join(self.test_dir, "predict_result_tile_emb")

        # export quant and no-input-tile
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

        # export no-quant and no-input-tile
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                no_quant_dir,
                env_str="QUANT_EMB=0",
            )

        # export quant and input-tile
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                input_tile_dir,
                env_str="QUANT_EMB=1 INPUT_TILE=2",
            )

        # predict quant and input-tile
        if self.success:
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

        # export quant and input-tile emb
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                input_tile_dir_emb,
                env_str="QUANT_EMB=1 INPUT_TILE=3",
            )

        # predict quant and input-tile emb
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(input_tile_dir_emb, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=tile_pred_output_emb,
                reserved_columns="user_id,item_id,clk",
                output_columns="probs",
                test_dir=input_tile_dir_emb,
            )
            # compare INPUT_TILE and no INPUT_TILE result consistency
            df = ds.dataset(pred_output, format="parquet").to_table().to_pandas()
            df_t = (
                ds.dataset(tile_pred_output_emb, format="parquet")
                .to_table()
                .to_pandas()
            )
            df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
            df_t = df_t.sort_values(by=list(df_t.columns)).reset_index(drop=True)
            self.assertTrue(df.equals(df_t))

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

        if torch.cuda.is_available():
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
                os.path.join(self.test_dir, "export/scripted_model.pt"),
                map_location=device,
            )
            result_cpu = model_cpu(data.to_dict(sparse_dtype=torch.int64))
            device = "cuda:0"
            model_gpu = torch.jit.load(
                os.path.join(self.test_dir, "export/scripted_model.pt"),
                map_location=device,
            )
            result_gpu = model_gpu(data.to_dict(sparse_dtype=torch.int64), device)
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
                data.to_dict(sparse_dtype=torch.int64), device
            )
            result_dict_json_path = os.path.join(
                self.test_dir, "result_gpu_no_quant.json"
            )
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
                data_input_tile.to(device=device).to_dict(sparse_dtype=torch.int64),
                device,
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
                data_input_tile.to(device=device).to_dict(sparse_dtype=torch.int64),
                device,
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

    def _test_rank_with_fg_trt(self, pipeline_config_path, predict_columns):
        self.success = utils.test_train_eval(
            pipeline_config_path,
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
        trt_dir = os.path.join(self.test_dir, "trt")
        input_tile_trt_dir = os.path.join(self.test_dir, "input_tile_trt")
        input_tile_emb_trt_dir = os.path.join(self.test_dir, "input_tile_emb_trt")

        pred_output = os.path.join(self.test_dir, "predict_result")
        trt_pred_output = os.path.join(self.test_dir, "predict_result_trt")
        tile_trt_pred_output = os.path.join(self.test_dir, "predict_result_tile_trt")
        tile_trt_pred_output_emb = os.path.join(
            self.test_dir, "predict_result_tile_emb_trt"
        )

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

        # quant and and trt
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                trt_dir,
                env_str="ENABLE_TRT=1 DEBUG_TRT=1",
            )

        # predict quant and trt
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(trt_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=trt_pred_output,
                reserved_columns="user_id,item_id,clk",
                output_columns="probs",
                test_dir=trt_dir,
            )
            # compare TRT and origin result consistency
            df = ds.dataset(pred_output, format="parquet").to_table().to_pandas()
            df_t = ds.dataset(trt_pred_output, format="parquet").to_table().to_pandas()
            df = df.sort_values(by=predict_columns).reset_index(drop=True)
            df_t = df_t.sort_values(by=predict_columns).reset_index(drop=True)
            self.assertTrue(dfs_are_close(df, df_t, 1e-6))

        # quant and input-tile and trt
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                input_tile_trt_dir,
                env_str="ENABLE_TRT=1 DEBUG_TRT=1 INPUT_TILE=2",
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(input_tile_trt_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=tile_trt_pred_output,
                reserved_columns="user_id,item_id,clk",
                output_columns="probs",
                test_dir=input_tile_trt_dir,
            )
            # compare INPUT_TILE+TRT and origin result consistency
            df = ds.dataset(pred_output, format="parquet").to_table().to_pandas()
            df_t = (
                ds.dataset(tile_trt_pred_output, format="parquet")
                .to_table()
                .to_pandas()
            )
            df = df.sort_values(by=predict_columns).reset_index(drop=True)
            df_t = df_t.sort_values(by=predict_columns).reset_index(drop=True)
            self.assertTrue(dfs_are_close(df, df_t, 1e-6))

        # quant and input-tile emb and trt
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                input_tile_emb_trt_dir,
                env_str="ENABLE_TRT=1 DEBUG_TRT=1 INPUT_TILE=3",
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=os.path.join(input_tile_emb_trt_dir, "export"),
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=tile_trt_pred_output_emb,
                reserved_columns="user_id,item_id,clk",
                output_columns="probs",
                test_dir=input_tile_emb_trt_dir,
            )
            # compare INPUT_TILE_EMB+TRT and origin result consistency
            df = ds.dataset(pred_output, format="parquet").to_table().to_pandas()
            df_t = (
                ds.dataset(tile_trt_pred_output_emb, format="parquet")
                .to_table()
                .to_pandas()
            )
            df = df.sort_values(by=predict_columns).reset_index(drop=True)
            df_t = df_t.sort_values(by=predict_columns).reset_index(drop=True)
            self.assertTrue(dfs_are_close(df, df_t, 1e-6))

        self.assertTrue(self.success)

        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "export/scripted_model.pt"))
        )

        self.assertTrue(
            os.path.exists(os.path.join(input_tile_trt_dir, "export/scripted_model.pt"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(input_tile_emb_trt_dir, "export/scripted_model.pt")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(input_tile_emb_trt_dir, "export/model_acc.json")
            )
        )
        with open(os.path.join(input_tile_emb_trt_dir, "export/model_acc.json")) as f:
            acc_cfg = json.load(f)
            self.assertEqual(acc_cfg["ENABLE_TRT"], "1")
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

        # quant and no-input-tile
        device = "cuda:0"
        model_gpu = torch.jit.load(
            os.path.join(self.test_dir, "export/scripted_model.pt"), map_location=device
        )
        result_gpu = model_gpu(data.to_dict(sparse_dtype=torch.int64), device)
        result_dict_json_path = os.path.join(self.test_dir, "result_gpu.json")
        utils.save_predict_result_json(result_gpu, result_dict_json_path)

        # quant and trt
        model_gpu_trt = torch.jit.load(
            os.path.join(self.test_dir, "trt/export/scripted_model.pt"),
            map_location=device,
        )
        result_gpu_trt = model_gpu_trt(data.to_dict(sparse_dtype=torch.int64))
        result_dict_json_path = os.path.join(self.test_dir, "result_gpu_trt.json")
        utils.save_predict_result_json(result_gpu_trt, result_dict_json_path)

        # quant and input-tile and trt
        model_gpu_input_tile = torch.jit.load(
            os.path.join(self.test_dir, "input_tile_trt/export/scripted_model.pt"),
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
            data_input_tile.to(device=device).to_dict(sparse_dtype=torch.int64)
        )
        result_dict_json_path = os.path.join(
            self.test_dir, "result_gpu_input_tile.json"
        )
        utils.save_predict_result_json(result_gpu_input_tile, result_dict_json_path)

        # quant and input-tile emb
        model_gpu_input_tile_emb = torch.jit.load(
            os.path.join(self.test_dir, "input_tile_emb_trt/export/scripted_model.pt"),
            map_location=device,
        )
        result_gpu_input_tile_emb = model_gpu_input_tile_emb(
            data_input_tile.to(device=device).to_dict(sparse_dtype=torch.int64)
        )

        # trt is all same sa no-trt
        for k, v in result_gpu.items():
            torch.testing.assert_close(
                result_gpu_trt[k].to(device), v, rtol=1e-6, atol=1e-6
            )

        # tile & trt is all same sa no-tile-trt
        for k, v in result_gpu.items():
            torch.testing.assert_close(
                result_gpu_input_tile[k].to(device), v, rtol=1e-6, atol=1e-6
            )

        # tile emb & trt is all same sa no-tile-trt
        for k, v in result_gpu.items():
            torch.testing.assert_close(
                result_gpu_input_tile_emb[k].to(device), v, rtol=1e-6, atol=1e-6
            )

    def test_multi_tower_din_with_fg_train_eval_export(self):
        self._test_rank_with_fg(
            "tzrec/tests/configs/multi_tower_din_fg_mock.config",
            comp_cpu_gpu_pred_result=True,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_multi_tower_din_zch_with_fg_train_eval_export(self):
        self._test_rank_with_fg(
            "tzrec/tests/configs/multi_tower_din_zch_fg_mock.config",
            comp_cpu_gpu_pred_result=True,
        )

    def test_multi_tower_din_with_fg_export_quant(self):
        self._test_rank_with_fg_quant(
            "tzrec/tests/configs/multi_tower_din_fg_mock.config"
        )

    def test_multi_tower_din_with_fg_train_eval_export_input_tile(self):
        self._test_rank_with_fg_input_tile(
            "tzrec/tests/configs/multi_tower_din_fg_mock.config"
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_multi_tower_din_zch_with_fg_train_eval_export_input_tile(self):
        self._test_rank_with_fg_input_tile(
            "tzrec/tests/configs/multi_tower_din_zch_fg_mock.config"
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

    @unittest.skipIf(*nv_gpu_unavailable)
    def test_multi_tower_with_fg_train_eval_export_trt(self):
        self._test_rank_with_fg_trt(
            "tzrec/tests/configs/multi_tower_din_trt_fg_mock.config",
            predict_columns=["user_id", "item_id", "clk", "probs"],
        )

    @unittest.skipIf(*nv_gpu_unavailable)
    def test_multi_tower_zch_with_fg_train_eval_export_trt(self):
        self._test_rank_with_fg_trt(
            "tzrec/tests/configs/multi_tower_din_zch_trt_fg_mock.config",
            predict_columns=["user_id", "item_id", "clk", "probs"],
        )


if __name__ == "__main__":
    unittest.main()
