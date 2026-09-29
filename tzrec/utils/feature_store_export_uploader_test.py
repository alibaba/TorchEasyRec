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

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import pyarrow as pa
from parameterized import parameterized

from tzrec.protos.train_pb2 import FeatureStoreConfig
from tzrec.utils import feature_store_export_uploader
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
from tzrec.utils.test_util import parameterized_name_func

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
            retry_backoff_secs=0,
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
        uploader.start()
        uploader.write(_TABLE_NAME, keys, values)
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
        uploader.start()
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
        uploader.start()
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
        uploader.start()
        uploader.write(table_name, np.array([7], dtype=np.int64), values)
        batch = view.arrow_calls[0]["batch"]
        self.assertEqual(batch.column("embedding").type, pa.list_(pa.uint8()))
        np.testing.assert_array_equal(batch.column("embedding").to_pylist(), values)
        self.assertEqual(batch.column("embedding_name").to_pylist(), [_TABLE_NAME])

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
        uploader.start()
        with self.assertRaisesRegex(ValueError, error):
            uploader.write(_TABLE_NAME, keys, values)
        self.assertEqual(view.calls, [])

    def test_empty_chunk_does_not_submit(self):
        view = _FakeView()
        uploader = self._uploader([view])
        uploader.start()
        uploader.write(
            _TABLE_NAME, np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        )
        uploader.close()
        self.assertEqual(view.calls, [])
        self.assertEqual(view.flush_calls, [])

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
        uploader.start()
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

    def test_incomplete_summary_propagates_on_write_and_close(self):
        views = [_FakeView(summaries=[{}]), _FakeView(summaries=[{}])]
        uploader = self._uploader(views, max_retries=2)
        uploader.start()
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
        first = _FakeView(summaries=[{}], close_error=RuntimeError("close failed"))
        uploader = self._uploader([first, _FakeView()], max_retries=2)
        uploader.start()
        with self.assertRaisesRegex(FeatureStoreUploadError, "could not drain"):
            uploader.write(
                _TABLE_NAME,
                np.array([1], dtype=np.int64),
                np.ones((1, 2), dtype=np.float32),
            )
        self.assertEqual(uploader._writer._create_client.call_count, 1)

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
        uploader.start()
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
            uploader.start()
        self.assertEqual(view.closed, [True])
        self.assertEqual(view.calls, [])


if __name__ == "__main__":
    unittest.main()
