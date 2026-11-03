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
import math
import os
from typing import Any, List, Optional

import torch.distributed as dist

from tzrec.protos.train_pb2 import TrainConfig
from tzrec.utils import checkpoint_util
from tzrec.utils.logging_util import logger

_HANDOVER_PROGRESS_FILENAME = "progress.json"
# Covers Kafka partition skew and progress write delay between the two lineages.
_HANDOVER_DEFAULT_TOLERANCE_SECS = 300.0


def _valid_data_ts(ts: Any) -> Optional[float]:
    """Return ts as a float if it is a valid event-time, else None."""
    # bool is an int subclass; -1.0 is the "no timestamp" sentinel
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    return float(ts) if math.isfinite(ts) and ts > 0 else None


class OnlineLearningHandover:
    """Retire a stream lineage once the next one has consumed data as fresh.

    Every lineage writes its quorum consumed event-time to
    ``<model_dir>/progress.json``; the predecessor compares it with the
    successor's and leaves training when the successor trails by at most the
    tolerance.

    Args:
        model_dir: this run's model_dir.
        successor_model_dir: the successor lineage's model_dir.
        tolerance_secs: allowed lag of the successor in seconds.
        quorum: fraction of workers that must have reached the event-time.
    """

    def __init__(
        self,
        model_dir: str,
        successor_model_dir: str,
        tolerance_secs: float,
        quorum: float,
    ) -> None:
        self.successor_model_dir = successor_model_dir
        self._progress_path = os.path.join(model_dir, _HANDOVER_PROGRESS_FILENAME)
        self._successor_progress_path = os.path.join(
            successor_model_dir, _HANDOVER_PROGRESS_FILENAME
        )
        self._tolerance_secs = tolerance_secs
        self._quorum = quorum
        self._warned_no_data_ts = False

    def _read_successor_ts(self) -> Optional[float]:
        """Read the successor's progress; None while absent or unreadable."""
        try:
            with open(self._successor_progress_path) as f:
                progress = json.load(f)
        except (OSError, ValueError):
            return None
        return (
            _valid_data_ts(progress.get("data_ts"))
            if isinstance(progress, dict)
            else None
        )

    def update_and_check(self, step: int, data_timestamp: float) -> bool:
        """Publish this run's progress, read the successor's, decide hand-over.

        A collective over the default process group: every rank must call it
        at the same step, or the gather/broadcast inside will hang. All ranks
        gather their consumed event-times and reconcile them with the quorum.
        Rank zero then writes the result with ``step`` to this run's
        progress.json, reads the successor's progress.json, compares the two
        against the tolerance and broadcasts the decision, so every rank gets
        the same answer. Nothing is written while no quorum event-time exists.

        Args:
            step: current global step.
            data_timestamp: this rank's consumed event-time, -1.0 if none.

        Returns:
            True if this run should hand over to the successor.
        """
        if dist.is_initialized():
            worker_ts: List[float] = [0.0] * dist.get_world_size()
            dist.all_gather_object(worker_ts, data_timestamp)
        else:
            worker_ts = [data_timestamp]
        data_ts = _valid_data_ts(
            checkpoint_util.quorum_event_time(worker_ts, self._quorum)
        )
        decision = [False]
        if int(os.environ.get("RANK", 0)) == 0:
            if data_ts is None:
                if not self._warned_no_data_ts:
                    logger.warning(
                        "no consumed event-time yet, so the lineage hand-over "
                        "cannot progress; it requires a data source with "
                        "timestamps such as KafkaDataset."
                    )
                    self._warned_no_data_ts = True
            else:
                tmp_path = f"{self._progress_path}.tmp.{os.getpid()}"
                with open(tmp_path, "w") as f:
                    json.dump({"step": step, "data_ts": data_ts}, f)
                os.replace(tmp_path, self._progress_path)
                successor_ts = self._read_successor_ts()
                if successor_ts is not None:
                    lag = data_ts - successor_ts
                    decision[0] = lag <= self._tolerance_secs
                    logger.info(
                        f"successor lineage lags {lag:.0f}s behind this run "
                        f"(tolerance {self._tolerance_secs:.0f}s)."
                    )
        if dist.is_initialized():
            dist.broadcast_object_list(decision, src=0)
        return decision[0]


def create_online_learning_handover(
    model_dir: str, train_config: TrainConfig
) -> Optional[OnlineLearningHandover]:
    """Build the lineage hand-over from ONLINE_LEARNING_SUCCESSOR_DIR.

    Args:
        model_dir: this run's model_dir.
        train_config: this run's train config.

    Returns:
        The hand-over, or None when no successor is configured.
    """
    successor_model_dir = os.environ.get("ONLINE_LEARNING_SUCCESSOR_DIR")
    if not successor_model_dir:
        return None
    if os.path.abspath(successor_model_dir) == os.path.abspath(model_dir):
        raise RuntimeError(
            "ONLINE_LEARNING_SUCCESSOR_DIR must differ from this run's model_dir, "
            f"got {successor_model_dir}."
        )
    # parsed on every rank so a bad value fails fast instead of on rank 0 only
    tolerance_env = os.environ.get("ONLINE_LEARNING_HANDOVER_TOLERANCE_SECS")
    tolerance_secs = (
        float(tolerance_env) if tolerance_env else _HANDOVER_DEFAULT_TOLERANCE_SECS
    )
    if not (math.isfinite(tolerance_secs) and tolerance_secs >= 0):
        raise ValueError(
            "ONLINE_LEARNING_HANDOVER_TOLERANCE_SECS must be a non-negative "
            f"number, got {tolerance_env}."
        )
    return OnlineLearningHandover(
        model_dir,
        successor_model_dir,
        tolerance_secs,
        train_config.save_checkpoints_timestamp_quorum,
    )
