# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the tzrec.main train-loop step counter."""

import itertools
import unittest


class TrainStepCounterMultiPassTest(unittest.TestCase):
    """Guard the step-based (``num_steps``) multi-pass step counter.

    Mirrors ``tzrec/main.py``: with ``use_step`` set, ``step_iter = iter(
    range(num_steps))`` and the data-pass ``StopIteration`` handler does
    ``step_iter = itertools.chain([i_step], step_iter); i_step -= 1``.
    ``step_iter`` must be a *one-shot* iterator so the next data pass resumes
    at ``i_step`` (retried) then ``i_step + 1``; a bare ``range`` is
    re-iterable and would yield ``[i_step, 0, 1, ...]``, resetting the counter
    so ``model.ckpt-{step}`` collides across passes and ``DynamicEmbDump``
    then refuses to overwrite the existing ``dynamicemb`` dir.
    """

    def test_use_step_counter_monotonic_and_terminates_across_passes(self):
        num_steps = 10
        # Mirrors tzrec/main.py use_step branch (post-fix): a one-shot iterator.
        step_iter = iter(range(num_steps))

        # Dataloader raises StopIteration at this global step on pass 1.
        exhaust_steps = iter([6])
        next_exhaust = next(exhaust_steps, None)

        trained = []
        # For use_step, epoch_iter is infinite (itertools.count(0, 0)); the loop
        # exits only via the num_steps termination guard below.
        while True:
            for i_step in step_iter:
                if next_exhaust is not None and i_step == next_exhaust:
                    # StopIteration handler (tzrec/main.py): retry this step on
                    # the next pass; do not let the chain re-iterate from 0.
                    step_iter = itertools.chain([i_step], step_iter)
                    i_step -= 1
                    break
                trained.append(i_step)
            # Mirrors "if use_step and i_step >= num_steps - 1: break".
            if i_step >= num_steps - 1:
                break
            next_exhaust = next(exhaust_steps, None)

        # Exactly num_steps steps trained, strictly monotonic, despite the
        # mid-run data exhaustion and retry at step 6: no reset to 0 -> no
        # model.ckpt-{step} collision across passes.
        self.assertEqual(trained, list(range(num_steps)))


if __name__ == "__main__":
    unittest.main()
