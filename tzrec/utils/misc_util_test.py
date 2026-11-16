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

import random
import unittest

from tzrec.utils.misc_util import random_name


class MiscUtilTest(unittest.TestCase):
    def test_random_name_preserves_global_rng(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        names = [random_name() for _ in range(32)]
        self.assertEqual(state, random.getstate())
        self.assertEqual(len({s.lower() for s in names}), len(names))
        self.assertTrue(all(len(s) == 8 and s.isascii() and s.isalnum() for s in names))
        self.assertEqual(len(random_name(16)), 16)
