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
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import pyarrow as pa
from parameterized import parameterized

from tzrec.protos.train_pb2 import FeatureStoreConfig
from tzrec.utils import feature_store_export_uploader, quant_util
from tzrec.utils.dynamic_embedding_export import get_dynamic_embedding_export
from tzrec.utils.feature_store_delta_uploader import (
    FEATURE_STORE_EMBEDDING_TYPE_UINT8,
    FeatureStoreDeltaUploader,
    FeatureStoreUploadError,
)
from tzrec.utils.feature_store_delta_uploader_test import (
    _FakeClient,
    _FakeProject,
    _FakeView,
)
from tzrec.utils.feature_store_export_uploader import FeatureStoreExportUploader
from tzrec.utils.npz_util import savez_streaming
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

_TABLE_NAME = "model.ebc.embedding_bags.user_emb"


class FeatureStoreExportUploaderTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(
            FeatureStoreDeltaUploader, "_create_credentials_client"
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _uploader(self, views, dimensions=None, **kwargs):
        config = FeatureStoreConfig(
            region="cn-test",
            project_name="project_a",
            feature_view_name="embeddings",
            version="export_1",
            upload_batch_size=kwargs.pop("upload_batch_size", 1000),
            max_retries=kwargs.pop("max_retries", 1),
            retry_backoff_secs=kwargs.pop("retry_backoff_secs", 0),
            upload_format=kwargs.pop("upload_format", "ARROW"),
        )
        uploader = FeatureStoreExportUploader(
            config, dimensions or {_TABLE_NAME: 2}, **kwargs
        )
        uploader._writer._create_client = mock.Mock(
            side_effect=[_FakeClient(_FakeProject(view), {}) for view in views]
        )
        self.addCleanup(uploader.close, raise_on_error=False)
        return uploader

    @parameterized.expand([("ARROW",), ("JSON",)], name_func=parameterized_name_func)
    def test_splits_rows_and_drains_chunk(self, upload_format):
        view = _FakeView()
        uploader = self._uploader(
            [view], max_in_flight_batches=2, upload_format=upload_format
        )
        keys = np.arange(2501, dtype=np.int64)
        values = np.arange(5002, dtype=np.float32).reshape(-1, 2)
        uploader.start(total_records=len(keys))
        with mock.patch.object(
            uploader._writer,
            "_validate_delta_batch",
            wraps=uploader._writer._validate_delta_batch,
        ) as validate:
            uploader.write(_TABLE_NAME, keys, values)
        self.assertEqual(validate.call_count, 3)
        self.assertEqual(view.flush_calls, [[1000, 1000], [501]])
        self.assertEqual(uploader._window, [])
        self.assertIsNone(uploader._writer._worker)
        rows = [row for call in view.calls for row in call["data"]]
        np.testing.assert_array_equal([row["key_id"] for row in rows], keys)
        np.testing.assert_array_equal([row["embedding"] for row in rows], values)
        self.assertEqual({call["write_mode"] for call in view.calls}, {"OVERWRITE"})
        self.assertEqual({call["version"] for call in view.calls}, {"export_1"})
        uploader.close()
        self.assertEqual(view.closed, [True])

    def test_byte_budget_splits_wide_rows(self):
        view = _FakeView()
        uploader = self._uploader(
            [view], dimensions={_TABLE_NAME: 512}, max_in_flight_batches=2
        )
        uploader.start(total_records=3)
        with mock.patch.object(feature_store_export_uploader, "_MAX_BATCH_BYTES", 8192):
            uploader.write(
                _TABLE_NAME,
                np.arange(3, dtype=np.int64),
                np.zeros((3, 512), dtype=np.float32),
            )
        self.assertEqual(view.flush_calls, [[1, 1], [1]])

    def test_row_exceeding_byte_budget_is_rejected(self):
        view = _FakeView()
        uploader = self._uploader([view], dimensions={_TABLE_NAME: 2048})
        uploader.start(total_records=1)
        with mock.patch.object(feature_store_export_uploader, "_MAX_BATCH_BYTES", 8192):
            with self.assertRaisesRegex(ValueError, "one embedding row"):
                uploader.write(
                    _TABLE_NAME,
                    np.arange(1, dtype=np.int64),
                    np.zeros((1, 2048), dtype=np.float32),
                )
        self.assertEqual(view.calls, [])

    def test_uint8_preserves_bytes_and_remaps_table(self):
        table_name = "model.ebc_user.embedding_bags.user_emb"
        view = _FakeView(embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8)
        uploader = self._uploader(
            [view],
            dimensions={table_name: 2},
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8,
        )
        values = np.array([[0, 255, 12, 128, 254, 17]], dtype=np.uint8)
        uploader.start(total_records=1)
        uploader.write(table_name, np.array([7], dtype=np.int64), values)
        batch = view.arrow_calls[0]["batch"]
        self.assertEqual(batch.column("embedding").type, pa.list_(pa.uint8()))
        np.testing.assert_array_equal(batch.column("embedding").to_pylist(), values)
        self.assertEqual(batch.column("embedding_name").to_pylist(), [_TABLE_NAME])

    def test_quantized_checkpoint_upload_matches_npz(self):
        test_dir = make_test_dir("quantized_export_upload_")
        self.addCleanup(shutil.rmtree, test_dir)
        directory = os.path.join(test_dir, "dynamicemb", "model.ebc")
        os.makedirs(directory)
        keys = np.array([-7, 2**60 + 1, 0], dtype=np.int64)
        values = np.array([[-2.0, 2.0], [-1.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        for rank, rows in enumerate((slice(0, 2), slice(2, 3))):
            for field, array in (
                ("keys", keys),
                ("values", values),
                ("scores", np.arange(3, dtype=np.int64)),
            ):
                array[rows].tofile(
                    os.path.join(
                        directory, f"user_emb_emb_{field}.rank_{rank}.world_size_2"
                    )
                )
        view = _FakeView(embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8)
        uploader = self._uploader(
            [view], embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8
        )
        arrays, metadata = get_dynamic_embedding_export(
            test_dir,
            {_TABLE_NAME: 2},
            read_chunk_bytes=16,
            quant_format="QUint8RowwiseF16",
            on_values=uploader.write,
        )
        uploader.start(total_records=metadata[_TABLE_NAME]["shape"][0])
        npz_path = os.path.join(test_dir, "export.npz")
        savez_streaming(npz_path, arrays)
        uploader.close()
        self.assertEqual(len(view.arrow_calls), 3)
        for call in view.arrow_calls:
            self.assertEqual(
                call["batch"].column("embedding").type, pa.list_(pa.uint8())
            )
            self.assertEqual(call["write_mode"], "OVERWRITE")
            self.assertEqual(call["version"], "export_1")
        uploaded_keys = [
            key
            for call in view.arrow_calls
            for key in call["batch"].column("key_id").to_pylist()
        ]
        uploaded_values = np.array(
            [
                row
                for call in view.arrow_calls
                for row in call["batch"].column("embedding").to_pylist()
            ],
            dtype=np.uint8,
        )
        with np.load(npz_path) as exported:
            np.testing.assert_array_equal(
                uploaded_keys, exported[f"{_TABLE_NAME}.keys"]
            )
            np.testing.assert_array_equal(
                uploaded_values, exported[f"{_TABLE_NAME}.values"]
            )
        np.testing.assert_array_equal(uploaded_keys, keys)
        self.assertEqual(uploaded_values.shape, (3, 6))
        np.testing.assert_allclose(
            quant_util.dequantize_quint8_rowwise_f16(uploaded_values, emb_dim=2),
            values,
            atol=5e-3,
        )
        self.assertEqual(uploader._uploaded_records, 3)

    @parameterized.expand(
        [
            (
                np.array([-1], dtype=np.int64),
                np.ones((1, 2), dtype=np.float32),
                "reserved",
            ),
            (
                np.array([1], dtype=np.int64),
                np.array([[np.nan, 0]], dtype=np.float32),
                "NaN",
            ),
            (
                np.array([1], dtype=np.int64),
                np.ones((1, 3), dtype=np.float32),
                "dimension",
            ),
            (np.array([1], dtype=np.int32), np.ones((1, 2), dtype=np.float32), "int64"),
            (np.array([1], dtype=np.int64), np.ones((1, 2), dtype=np.float64), "dtype"),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_rows_are_not_submitted(self, keys, values, error):
        view = _FakeView()
        uploader = self._uploader([view])
        uploader.start(total_records=len(keys))
        with self.assertRaisesRegex(ValueError, error):
            uploader.write(_TABLE_NAME, keys, values)
        self.assertEqual(view.calls, [])

    def test_empty_chunk_does_not_submit(self):
        view = _FakeView()
        uploader = self._uploader([view])
        uploader.start(total_records=0)
        uploader.write(
            _TABLE_NAME, np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        )
        with self.assertLogs("tzrec", level="INFO") as logs:
            uploader.close()
        self.assertIn("records=0/0 progress=100.00%", logs.output[-1])
        self.assertEqual(view.calls, [])
        self.assertEqual(view.flush_calls, [])

    def test_progress_reports_total_rows_and_percentage(self):
        view = _FakeView()
        with (
            mock.patch.object(
                feature_store_export_uploader.time, "monotonic", return_value=0
            ) as clock,
            self.assertLogs("tzrec", level="INFO") as logs,
        ):
            uploader = self._uploader([view])
            uploader.start(total_records=5)
            for now, keys in ((31, [0, 1]), (32, [2, 3]), (62, [4])):
                clock.return_value = now
                uploader.write(
                    _TABLE_NAME,
                    np.array(keys, dtype=np.int64),
                    np.ones((len(keys), 2), dtype=np.float32),
                )
            uploader.close()
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(
            messages,
            [
                "Dynemb upload to FeatureStore started: "
                "version=export_1 total_records=5",
                "Dynemb upload to FeatureStore progress: "
                "version=export_1 records=2/5 progress=40.00%",
                "Dynemb upload to FeatureStore progress: "
                "version=export_1 records=5/5 progress=100.00%",
                "Dynemb upload to FeatureStore completed: "
                "version=export_1 records=5/5 progress=100.00%",
            ],
        )

    def test_retry_replays_only_failed_window_after_draining(self):
        first = _FakeView()
        second = _FakeView()
        flush = first.write_flush

        def fail_second_window():
            summary = flush()
            if len(first.flush_calls) == 2:
                summary["success_records"] -= 1
                summary["failed_records"] += 1
            return summary

        first.write_flush = fail_second_window
        uploader = self._uploader(
            [first, second], upload_batch_size=1, max_in_flight_batches=2, max_retries=2
        )
        create_client = uploader._writer._create_client

        def get_client():
            if create_client.call_count:
                self.assertEqual(first.closed, [True])
            return create_client()

        uploader._writer._create_client = get_client
        uploader.start(total_records=5)
        uploader.write(
            _TABLE_NAME, np.arange(5, dtype=np.int64), np.ones((5, 2), dtype=np.float32)
        )
        self.assertEqual(
            [call["data"][0]["key_id"] for call in second.calls], [2, 3, 4]
        )
        self.assertEqual(
            [call["ts"] for call in first.calls[2:]],
            [call["ts"] for call in second.calls[:2]],
        )
        self.assertEqual(first.flush_calls, [[1, 1], [1, 1]])
        self.assertEqual(second.flush_calls, [[1, 1], [1]])
        with self.assertLogs("tzrec", level="INFO") as logs:
            uploader.close()
        self.assertIn("records=5/5 progress=100.00%", logs.output[-1])

    def test_incomplete_summary_propagates_on_write_and_close(self):
        views = [_FakeView(summaries=[{}]), _FakeView(summaries=[{}])]
        uploader = self._uploader(views, max_retries=2)
        uploader.start(total_records=1)
        with self.assertRaisesRegex(FeatureStoreUploadError, "2 attempts"):
            uploader.write(
                _TABLE_NAME,
                np.array([1], dtype=np.int64),
                np.ones((1, 2), dtype=np.float32),
            )
        self.assertEqual(uploader._writer._create_client.call_count, 2)
        with self.assertRaises(FeatureStoreUploadError):
            uploader.close()
        uploader.close(raise_on_error=False)

    def test_close_failure_stops_retries(self):
        upload_error = RuntimeError("upload failed")
        first = _FakeView(close_error=RuntimeError("close failed"))
        first.write_flush = mock.Mock(side_effect=upload_error)
        uploader = self._uploader([first, _FakeView()], max_retries=2)
        uploader.start(total_records=1)
        with self.assertRaisesRegex(
            FeatureStoreUploadError, "could not drain"
        ) as error:
            uploader.write(
                _TABLE_NAME,
                np.array([1], dtype=np.int64),
                np.ones((1, 2), dtype=np.float32),
            )
        self.assertEqual(uploader._writer._create_client.call_count, 1)
        self.assertIs(error.exception.__cause__, upload_error)

    @parameterized.expand(
        [("upload", FeatureStoreUploadError), ("validation", ValueError)],
        name_func=parameterized_name_func,
    )
    def test_close_preserves_flush_error_when_sdk_close_fails(self, _name, error_type):
        view = _FakeView(close_error=RuntimeError("close failed"))
        uploader = self._uploader([view])
        uploader.start(total_records=1)
        flush_error = error_type("original flush failure")
        with (
            mock.patch.object(uploader, "flush", side_effect=flush_error),
            self.assertRaises(error_type) as error,
        ):
            uploader.close()
        self.assertIs(error.exception, flush_error)
        self.assertTrue(uploader._closed)
        self.assertEqual(view.closed, [True])

    def test_close_surfaces_sdk_failure_after_successful_flush(self):
        view = _FakeView(close_error=RuntimeError("close failed"))
        uploader = self._uploader([view])
        uploader.start(total_records=1)
        with self.assertRaisesRegex(FeatureStoreUploadError, "writer close failed"):
            uploader.close()

    def test_blocking_flush_applies_backpressure(self):
        view = _FakeView()
        uploader = self._uploader([view], upload_batch_size=1, max_in_flight_batches=2)
        flush = view.write_flush
        entered = threading.Event()
        release = threading.Event()

        def blocking_flush():
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test did not release flush")
            return flush()

        view.write_flush = blocking_flush
        uploader.start(total_records=5)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                uploader.write,
                _TABLE_NAME,
                np.arange(5, dtype=np.int64),
                np.ones((5, 2), dtype=np.float32),
            )
            try:
                self.assertTrue(entered.wait(timeout=10))
                self.assertFalse(future.done())
                self.assertEqual(len(view.calls), 2)
            finally:
                release.set()
            future.result(timeout=10)
        self.assertEqual(view.flush_calls, [[1, 1], [1, 1], [1]])

    def test_start_checks_schema_before_data(self):
        view = _FakeView(embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8)
        uploader = self._uploader([view])
        with self.assertRaisesRegex(RuntimeError, "type mismatch"):
            uploader.start(total_records=1)
        self.assertEqual(view.closed, [True])
        self.assertEqual(view.calls, [])
        with self.assertNoLogs("tzrec", level="INFO"):
            uploader.close()

    @parameterized.expand(
        [
            ("zero_window", {_TABLE_NAME: 2}, 0, "max_in_flight_batches"),
            ("negative_window", {_TABLE_NAME: 2}, -1, "max_in_flight_batches"),
            ("empty_name", {"": 2}, 1, "names must be nonempty"),
            ("zero_dimension", {_TABLE_NAME: 0}, 1, "dimensions positive"),
            ("negative_dimension", {_TABLE_NAME: -1}, 1, "dimensions positive"),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_constructor_contract(self, _name, dimensions, window, error):
        with self.assertRaisesRegex(ValueError, error):
            self._uploader(
                [_FakeView()], dimensions=dimensions, max_in_flight_batches=window
            )

    @parameterized.expand(
        [
            (
                "write_before_start",
                False,
                False,
                "write",
                RuntimeError,
                "started and open",
            ),
            (
                "write_after_close",
                True,
                True,
                "write",
                RuntimeError,
                "started and open",
            ),
            ("start_after_close", True, True, "start", RuntimeError, "already closed"),
            ("unknown_table", True, False, "unknown", ValueError, "absent from model"),
        ],
        name_func=parameterized_name_func,
    )
    def test_lifecycle_guards(self, _name, start, close, operation, error_type, error):
        view = _FakeView()
        uploader = self._uploader([view])
        if start:
            uploader.start(total_records=1)
        if close:
            uploader.close()
        with self.assertRaisesRegex(error_type, error):
            if operation == "start":
                uploader.start(total_records=1)
            else:
                uploader.write(
                    "unknown" if operation == "unknown" else _TABLE_NAME,
                    np.array([1], dtype=np.int64),
                    np.ones((1, 2), dtype=np.float32),
                )
        self.assertEqual(view.calls, [])

    def test_retry_backoff_scales_with_attempt(self):
        views = [_FakeView(summaries=[{}]), _FakeView(summaries=[{}]), _FakeView()]
        uploader = self._uploader(views, max_retries=3, retry_backoff_secs=1)
        uploader.start(total_records=1)
        with mock.patch.object(feature_store_export_uploader.time, "sleep") as sleep:
            uploader.write(
                _TABLE_NAME,
                np.array([1], dtype=np.int64),
                np.ones((1, 2), dtype=np.float32),
            )
        self.assertEqual(sleep.call_args_list, [mock.call(1), mock.call(2)])
        self.assertEqual(uploader._uploaded_records, 1)
        self.assertEqual(uploader._writer._create_client.call_count, 3)


if __name__ == "__main__":
    unittest.main()
