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

import unittest

from tzrec.ops.utils import prev_power_of_2


class OpUtilsTest(unittest.TestCase):
    def test_prev_power_of_2(self):
        self.assertEqual(prev_power_of_2(30), 16)
        self.assertEqual(prev_power_of_2(31), 16)
        self.assertEqual(prev_power_of_2(32), 32)
        self.assertEqual(prev_power_of_2(33), 32)
        self.assertEqual(prev_power_of_2(63), 32)
        self.assertEqual(prev_power_of_2(64), 64)


if __name__ == "__main__":
    unittest.main()
