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


import math
import multiprocessing as mp
import os
import tempfile
import time
import unittest

import numpy as np
import pyarrow as pa
from torch import distributed as dist

from tzrec.datasets.sampler import (
    HardNegativeSampler,
    HardNegativeSamplerV2,
    NegativeSampler,
    NegativeSamplerV2,
    TDMPredictSampler,
    TDMSampler,
    _to_arrow_array,
)
from tzrec.protos import sampler_pb2
from tzrec.utils import misc_util


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self._temp_files = []

    def tearDown(self):
        for f in self._temp_files:
            f.close()

    def _create_item_gl_data(self):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(100):
            f.write(f"{i}\t{1}\t{i}:{i+1000}:我们{i}\n")
        f.flush()
        return f

    def _create_user_gl_data(self):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\n")
        for i in range(100):
            f.write(f"{i}\t{1}\n")
        f.flush()
        return f

    def _create_clk_edge_gl_data(self):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("userid:int64\titemid:int64\tweight:float\n")
        for i in range(100):
            f.write(f"{i}\t{i}\t{1}\n")
        f.flush()
        return f

    def _create_noclk_edge_gl_data(self):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("userid:int64\titemid:int64\tweight:float\n")
        for i in range(100):
            f.write(f"{i}\t{99-i}\t{1}\n")
        f.flush()
        return f

    def _create_item_gl_data_for_tdm(self):
        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("id:int64\tweight:float\tattrs:string\n")
        for i in range(63):
            f.write(f"{i}\t{1}\t{int(math.log(i+1,2))}:{i}:{i+1000}:我们{i}\n")
        f.flush()
        return f

    def _create_edge_gl_data_for_tdm(self):
        def _ancesstor(code):
            ancs = []
            while code > 0:
                code = int((code - 1) / 2)
                ancs.append(code)
            return ancs

        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("src_id:int64\tdst_id:int\tweight:float\n")
        for i in range(31, 63):
            for anc in _ancesstor(i):
                f.write(f"{i}\t{anc}\t{1.0}\n")
        f.flush()
        return f

    def _create_predict_edge_gl_data_for_tdm(self):
        def _childern(code):
            return [2 * code + 1, 2 * code + 2]

        f = tempfile.NamedTemporaryFile("w")
        self._temp_files.append(f)
        f.write("src_id:int64\tdst_id:int\tweight:float\n")
        for i in range(7, 15):
            f.write(f"0\t{i}\t{1}\n")
        for i in range(7, 31):
            for child in _childern(i):
                f.write(f"{i}\t{child}\t{1}\n")
        f.flush()
        return f

    def test_to_arrow_array_with_list_type(self):
        x = _to_arrow_array(np.array(["1\x1d2", "3", ""]), pa.list_(pa.int32()))
        self.assertEqual(x, pa.array([[1, 2], [3], None], type=pa.list_(pa.int32())))

    def test_to_arrow_array_with_map_type(self):
        x = _to_arrow_array(
            np.array(["a:1\x1db:2", "c:3", ""]), pa.map_(pa.string(), pa.int32())
        )
        self.assertEqual(
            x,
            pa.MapArray.from_arrays(
                offsets=pa.array([0, 2, 3, 3]),
                keys=pa.array(["a", "b", "c"]),
                items=pa.array([1, 2, 3], type=pa.int32()),
                type=pa.map_(pa.string(), pa.int32()),
            ),
        )

    def test_negative_sampler(self):
        f = self._create_item_gl_data()

        def _sampler_worker(res):
            config = sampler_pb2.NegativeSampler(
                input_path=f.name,
                num_sample=8,
                attr_fields=["int_a", "float_b", "str_c"],
                item_id_field="item_id",
            )
            sampler = NegativeSampler(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster()
            sampler.launch_server()
            sampler.init()
            res.update(sampler.get({"item_id": pa.array([0, 1, 2, 3])}))

        res = mp.Manager().dict()
        p = mp.Process(target=_sampler_worker, args=(res,))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("worker failed.")
        self.assertEqual(len(res["int_a"]), 8)
        self.assertEqual(len(res["float_b"]), 8)
        self.assertEqual(len(res["str_c"]), 8)

    def test_negative_sampler_multi_node(self):
        f = self._create_item_gl_data()

        def _launch_client(
            sampler,
        ):
            sampler.init()
            res = sampler.get({"item_id": pa.array([0, 1, 2, 3])})
            assert len(res["int_a"]) == 8
            assert len(res["float_b"]) == 8
            assert len(res["str_c"]) == 8

        def _sampler_worker(
            rank,
            local_rank,
            group_rank,
            local_world_size,
            world_size,
            gl_num_client,
            port,
        ):
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["GROUP_RANK"] = str(group_rank)
            os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group(backend="gloo")

            config = sampler_pb2.NegativeSampler(
                input_path=f.name,
                num_sample=8,
                attr_fields=["int_a", "float_b", "str_c"],
                item_id_field="item_id",
            )
            sampler = NegativeSampler(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster(gl_num_client)
            sampler.launch_server()
            procs = []
            for _ in range(gl_num_client):
                p = mp.Process(target=_launch_client, args=(sampler,))
                p.start()
                # on ecs, popen too fast may become <defunct> process
                time.sleep(1)
                procs.append(p)
            for i, p in enumerate(procs):
                p.join()
                print(f"{local_rank}, {group_rank} done.")
                if p.exitcode != 0:
                    raise RuntimeError(f"client-{i} of worker-{rank} failed.")

        num_node = 2
        num_proc_per_node = 3
        num_dl_per_proc = 2
        procs = []
        port = misc_util.get_free_port()
        for i in range(num_node):
            for j in range(num_proc_per_node):
                rank = i * num_proc_per_node + j
                world_size = num_node * num_proc_per_node
                p = mp.Process(
                    target=_sampler_worker,
                    args=(
                        rank,
                        j,
                        i,
                        num_proc_per_node,
                        world_size,
                        num_dl_per_proc,
                        port,
                    ),
                )
                p.start()
                procs.append(p)

        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed.")

    def test_negative_sampler_v2(self):
        f_user = self._create_user_gl_data()
        f_item = self._create_item_gl_data()
        f_clk_edge = self._create_clk_edge_gl_data()

        def _sampler_worker(res):
            config = sampler_pb2.NegativeSamplerV2(
                user_input_path=f_user.name,
                item_input_path=f_item.name,
                pos_edge_input_path=f_clk_edge.name,
                num_sample=8,
                attr_fields=["int_a", "float_b", "str_c"],
                item_id_field="item_id",
                user_id_field="user_id",
            )
            sampler = NegativeSamplerV2(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster()
            sampler.launch_server()
            sampler.init()
            res.update(
                sampler.get(
                    {
                        "user_id": pa.array([0, 1, 2, 3]),
                        "item_id": pa.array([0, 1, 2, 3]),
                    }
                )
            )

        res = mp.Manager().dict()
        p = mp.Process(target=_sampler_worker, args=(res,))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("worker failed.")
        self.assertEqual(len(res["int_a"]), 8)
        self.assertEqual(len(res["float_b"]), 8)
        self.assertEqual(len(res["str_c"]), 8)

    def test_hard_negative_sampler(self):
        f_user = self._create_user_gl_data()
        f_item = self._create_item_gl_data()
        f_noclk_edge = self._create_noclk_edge_gl_data()

        def _sampler_worker(res):
            config = sampler_pb2.HardNegativeSampler(
                user_input_path=f_user.name,
                item_input_path=f_item.name,
                hard_neg_edge_input_path=f_noclk_edge.name,
                num_sample=8,
                num_hard_sample=8,
                attr_fields=["int_a", "float_b", "str_c"],
                item_id_field="item_id",
                user_id_field="user_id",
            )
            sampler = HardNegativeSampler(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster()
            sampler.launch_server()
            sampler.init()
            res.update(
                sampler.get(
                    {
                        "user_id": pa.array([0, 1, 2, 3]),
                        "item_id": pa.array([0, 1, 2, 3]),
                    }
                )
            )

        res = mp.Manager().dict()
        p = mp.Process(target=_sampler_worker, args=(res,))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("worker failed.")
        self.assertGreater(len(res["int_a"]), 8)
        self.assertGreater(len(res["float_b"]), 8)
        self.assertGreater(len(res["str_c"]), 8)

    def test_hard_negative_sampler_v2(self):
        f_user = self._create_user_gl_data()
        f_item = self._create_item_gl_data()
        f_clk_edge = self._create_clk_edge_gl_data()
        f_noclk_edge = self._create_noclk_edge_gl_data()

        def _sampler_worker(res):
            config = sampler_pb2.HardNegativeSamplerV2(
                user_input_path=f_user.name,
                item_input_path=f_item.name,
                pos_edge_input_path=f_clk_edge.name,
                hard_neg_edge_input_path=f_noclk_edge.name,
                num_sample=8,
                num_hard_sample=8,
                attr_fields=["int_a", "float_b", "str_c"],
                item_id_field="item_id",
                user_id_field="user_id",
            )
            sampler = HardNegativeSamplerV2(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster()
            sampler.launch_server()
            sampler.init()
            res.update(
                sampler.get(
                    {
                        "user_id": pa.array([0, 1, 2, 3]),
                        "item_id": pa.array([0, 1, 2, 3]),
                    }
                )
            )

        res = mp.Manager().dict()
        p = mp.Process(target=_sampler_worker, args=(res,))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("worker failed.")
        self.assertGreater(len(res["int_a"]), 8)
        self.assertGreater(len(res["float_b"]), 8)
        self.assertGreater(len(res["str_c"]), 8)

    def test_tdm_sampler(self):
        f_item = self._create_item_gl_data_for_tdm()
        f_edge = self._create_edge_gl_data_for_tdm()
        f_predict_edge = self._create_predict_edge_gl_data_for_tdm()

        def _sampler_worker(pos_res, neg_res):
            config = sampler_pb2.TDMSampler(
                item_input_path=f_item.name,
                edge_input_path=f_edge.name,
                predict_edge_input_path=f_predict_edge.name,
                attr_fields=["tree_level", "int_a", "float_b", "str_c"],
                item_id_field="item_id",
                layer_num_sample=[0, 1, 2, 3, 4, 5],
            )
            sampler = TDMSampler(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster()
            sampler.launch_server()
            sampler.init()
            pos_res.update(
                sampler.get(
                    {
                        "item_id": pa.array([31, 41, 51, 61]),
                    }
                )[0]
            )
            neg_res.update(
                sampler.get(
                    {
                        "item_id": pa.array([31, 41, 51, 61]),
                    }
                )[1]
            )

        pos_res = mp.Manager().dict()
        neg_res = mp.Manager().dict()
        p = mp.Process(
            target=_sampler_worker,
            args=(
                pos_res,
                neg_res,
            ),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("worker failed.")
        self.assertEqual(len(pos_res["int_a"]), 4 * 5)
        self.assertEqual(len(pos_res["float_b"]), 4 * 5)
        self.assertEqual(len(pos_res["str_c"]), 4 * 5)
        self.assertEqual(len(neg_res["int_a"]), 4 * 15)
        self.assertEqual(len(neg_res["float_b"]), 4 * 15)
        self.assertEqual(len(neg_res["str_c"]), 4 * 15)

    def test_tdm_predict_sampler(self):
        f_item = self._create_item_gl_data_for_tdm()
        f_edge = self._create_edge_gl_data_for_tdm()
        f_predict_edge = self._create_predict_edge_gl_data_for_tdm()

        def _sampler_worker(res):
            config = sampler_pb2.TDMSampler(
                item_input_path=f_item.name,
                edge_input_path=f_edge.name,
                predict_edge_input_path=f_predict_edge.name,
                attr_fields=["tree_level", "int_a", "float_b", "str_c"],
                item_id_field="item_id",
                layer_num_sample=[0, 1, 2, 3, 4, 5],
            )
            sampler = TDMPredictSampler(
                config=config,
                fields=[
                    pa.field(name="int_a", type=pa.int64()),
                    pa.field(name="float_b", type=pa.float64()),
                    pa.field(name="str_c", type=pa.string()),
                ],
                batch_size=4,
            )
            sampler.init_cluster()
            sampler.launch_server()
            sampler.init()
            sampler.init_sampler(2)
            res.update(
                sampler.get(
                    pa.array([21, 22, 23, 24]),
                )
            )

        res = mp.Manager().dict()
        p = mp.Process(
            target=_sampler_worker,
            args=(res,),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("worker failed.")
        self.assertEqual(len(res["int_a"]), 4 * 2)
        self.assertEqual(len(res["float_b"]), 4 * 2)
        self.assertEqual(len(res["str_c"]), 4 * 2)


if __name__ == "__main__":
    unittest.main()
