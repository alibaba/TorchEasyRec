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

"""Test tooling: tzrec.train_eval that re-saves the checkpoint it restored.

Run like ``tzrec/train_eval.py --continue_train``. Right after the restore (and
before any training step) the restored state is saved as
``$RESAVE_MODEL_DIR/model.ckpt-<step>`` and the process exits, so a test can
compare a checkpoint with its save -> restore -> save round trip.
"""

import os
import runpy
import sys

from tzrec.utils import checkpoint_util

_dataloader_state = {}
_restore_dataloader_state = checkpoint_util.CheckpointManager.restore_dataloader_state
_restore = checkpoint_util.CheckpointManager.restore


def _capture_dataloader_state(self, ckpt_path):
    state = _restore_dataloader_state(self, ckpt_path)
    _dataloader_state.update(state or {})
    return state


def _restore_then_save(self, ckpt_path, model, optimizer=None, *args, **kwargs):
    _restore(self, ckpt_path, model, optimizer, *args, **kwargs)
    step = int(ckpt_path.rstrip("/").rsplit("-", 1)[1])
    checkpoint_util.CheckpointManager(os.environ["RESAVE_MODEL_DIR"]).save(
        step,
        model,
        optimizer,
        dict(_dataloader_state),
        kwargs.get("dense_ema"),
        lr_schedulers=kwargs.get("lr_schedulers"),
    )
    raise SystemExit(0)


checkpoint_util.CheckpointManager.restore_dataloader_state = _capture_dataloader_state
checkpoint_util.CheckpointManager.restore = _restore_then_save

sys.argv = ["tzrec.train_eval"] + sys.argv[1:]
runpy.run_module("tzrec.train_eval", run_name="__main__")
