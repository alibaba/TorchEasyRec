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

import json
import os
import shutil
import unittest
from unittest import mock

from parameterized import param, parameterized

from tzrec.protos.train_pb2 import TrainConfig
from tzrec.utils.online_learning_util import (
    OnlineLearningHandover,
    create_online_learning_handover,
)
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class OnlineLearningHandoverTest(unittest.TestCase):
    """Tests for the stream lineage hand-over progress check."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def _handover(self, successor_progress, tolerance=0.0):
        own_dir = os.path.join(self.test_dir, "own")
        successor_dir = os.path.join(self.test_dir, "successor")
        os.makedirs(own_dir)
        os.makedirs(successor_dir)
        if successor_progress is not None:
            with open(os.path.join(successor_dir, "progress.json"), "w") as f:
                if isinstance(successor_progress, str):
                    f.write(successor_progress)
                else:
                    json.dump(successor_progress, f)
        return OnlineLearningHandover(own_dir, successor_dir, tolerance, 0.5), own_dir

    @parameterized.expand(
        [
            param("ahead", own=1000.0, successor=2000.0, tolerance=0.0, expected=True),
            param("equal", own=1000.0, successor=1000.0, tolerance=0.0, expected=True),
            param(
                "behind", own=2000.0, successor=1000.0, tolerance=0.0, expected=False
            ),
            param(
                "within_tolerance",
                own=2000.0,
                successor=1500.0,
                tolerance=500.0,
                expected=True,
            ),
            param(
                "beyond_tolerance",
                own=2000.0,
                successor=1499.0,
                tolerance=500.0,
                expected=False,
            ),
            param(
                "successor_missing",
                own=1000.0,
                successor=None,
                tolerance=0.0,
                expected=False,
            ),
            param(
                "sentinel", own=1000.0, successor=-1.0, tolerance=1e9, expected=False
            ),
            param(
                "nan", own=1000.0, successor=float("nan"), tolerance=0.0, expected=False
            ),
            param(
                "inf", own=1000.0, successor=float("inf"), tolerance=0.0, expected=False
            ),
            param(
                "non_numeric",
                own=1000.0,
                successor="2000",
                tolerance=0.0,
                expected=False,
            ),
            param("bool", own=0.5, successor=True, tolerance=1.0, expected=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_update_and_check(self, _name, own, successor, tolerance, expected) -> None:
        progress = None if successor is None else {"data_ts": successor}
        handover, own_dir = self._handover(progress, tolerance)
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            self.assertIs(handover.update_and_check(7, own), expected)
        with open(os.path.join(own_dir, "progress.json")) as f:
            self.assertEqual(json.load(f), {"step": 7, "data_ts": own})

    @parameterized.expand(
        [
            param("corrupt", content="{not json"),
            param("non_object", content='["2000"]'),
        ],
        name_func=parameterized_name_func,
    )
    def test_update_and_check_unreadable_successor(self, _name, content) -> None:
        handover, _ = self._handover(content, tolerance=1e9)
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            self.assertFalse(handover.update_and_check(1, 1000.0))

    def test_update_and_check_without_data_ts(self) -> None:
        """No event-time: no progress written, no hand-over, one warning."""
        handover, own_dir = self._handover({"data_ts": 1000.0}, tolerance=1e9)
        with (
            mock.patch.dict(os.environ, {"RANK": "0"}),
            mock.patch("tzrec.utils.online_learning_util.logger") as mock_logger,
        ):
            self.assertFalse(handover.update_and_check(1, -1.0))
            self.assertFalse(handover.update_and_check(2, -1.0))
        self.assertEqual(mock_logger.warning.call_count, 1)
        self.assertFalse(os.path.exists(os.path.join(own_dir, "progress.json")))

    def test_create_online_learning_handover(self) -> None:
        model_dir = os.path.join(self.test_dir, "own")
        successor_dir = os.path.join(self.test_dir, "successor")
        train_config = TrainConfig()
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("ONLINE_LEARNING_SUCCESSOR_DIR", None)
            os.environ.pop("ONLINE_LEARNING_HANDOVER_TOLERANCE_SECS", None)
            self.assertIsNone(create_online_learning_handover(model_dir, train_config))
            # no online dense export needed
            os.environ["ONLINE_LEARNING_SUCCESSOR_DIR"] = successor_dir
            handover = create_online_learning_handover(model_dir, train_config)
            self.assertEqual(handover.successor_model_dir, successor_dir)
            self.assertEqual(handover._tolerance_secs, 300.0)
            os.environ["ONLINE_LEARNING_HANDOVER_TOLERANCE_SECS"] = "120"
            handover = create_online_learning_handover(model_dir, train_config)
            self.assertEqual(handover._tolerance_secs, 120.0)
            for bad in ("-1", "nan", "abc"):
                os.environ["ONLINE_LEARNING_HANDOVER_TOLERANCE_SECS"] = bad
                with self.assertRaises(ValueError):
                    create_online_learning_handover(model_dir, train_config)
            os.environ.pop("ONLINE_LEARNING_HANDOVER_TOLERANCE_SECS")
            os.environ["ONLINE_LEARNING_SUCCESSOR_DIR"] = model_dir + "/"
            with self.assertRaisesRegex(RuntimeError, "must differ"):
                create_online_learning_handover(model_dir, train_config)


if __name__ == "__main__":
    unittest.main()
