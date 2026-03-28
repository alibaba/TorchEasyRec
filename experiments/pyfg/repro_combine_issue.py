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

r"""Minimal reproduction of pyfg combine_feature combiner issue.

When combine_feature is used in sequence mode with multi-value input
(separated by \x1d) and a combiner (sum/mean), the framework expands
multi-values into separate sequence positions BEFORE the CombineOP can
aggregate them. The combiner never gets to run on the multi-values.

Expected: combiner aggregates \x1d-separated values within each
sequence position, producing one combined value per position.

Actual: each \x1d-separated value becomes a separate sequence position,
bypassing the combiner entirely.
"""

import pyarrow as pa
import pyfg


def test_standalone_sequence_combine():
    """Standalone sequence_combine_feature with multi-value input."""
    config = {
        "features": [
            {
                "feature_type": "sequence_combine_feature",
                "feature_name": "combine_feat",
                "expression": "user:input",
                "num_buckets": 10,
                "combiner": "sum",
                "value_map": {"tag1": 1.0, "tag2": 2.0},
                "default_value": "0",
                "sequence_length": 50,
                "sequence_delim": ";",
            }
        ]
    }
    handler = pyfg.FgArrowHandler(config, 1)

    # Input: position 1 has multi-value "tag1\x1dtag2", position 2 has "tag1"
    input_data = {
        "input": pa.array(["tag1\x1dtag2;tag1", "tag2", ""]),
    }
    result, status = handler.process_arrow(input_data)
    output = result["combine_feat"]

    print("=== Standalone sequence_combine_feature ===")
    print("Input: ['tag1\\x1dtag2;tag1', 'tag2', '']")
    print("value_map: tag1->1.0, tag2->2.0, combiner: sum")
    print(f"Actual values:      {output.np_values}")
    print(f"Actual key_lengths: {output.np_key_lengths}")
    print(f"Actual lengths:     {output.np_lengths}")
    print("Expected values:      [3 1 2 0]  (pos1: sum(1,2)=3, pos2: 1)")
    print("Expected key_lengths: [1 1 1 1]")
    print("Expected lengths:     [2 1 1]    (2 positions for sample 1, not 3)")
    print()


def test_grouped_sequence_combine():
    """Grouped sequence with combine_feature sub-feature."""
    config = {
        "features": [
            {
                "sequence_name": "click_seq",
                "sequence_length": 50,
                "sequence_delim": ";",
                "sequence_pk": "user:click_seq",
                "features": [
                    {
                        "feature_type": "combine_feature",
                        "feature_name": "combine_feat",
                        "expression": "user:input",
                        "num_buckets": 10,
                        "combiner": "sum",
                        "value_map": {"tag1": 1.0, "tag2": 2.0},
                        "value_type": "int64",
                        "default_value": "0",
                    }
                ],
            }
        ]
    }
    handler = pyfg.FgArrowHandler(config, 1)

    input_data = {
        "click_seq__input": pa.array(["tag1\x1dtag2;tag1", "tag2", ""]),
    }
    result, status = handler.process_arrow(input_data)
    output = result["click_seq__combine_feat"]

    print("=== Grouped sequence combine_feature ===")
    print("Input: ['tag1\\x1dtag2;tag1', 'tag2', '']")
    print("value_map: tag1->1.0, tag2->2.0, combiner: sum")
    print(f"Actual values:      {output.np_values}")
    print(f"Actual key_lengths: {output.np_key_lengths}")
    print(f"Actual lengths:     {output.np_lengths}")
    print("Expected values:      [3 1 2 0]  (pos1: sum(1,2)=3, pos2: 1)")
    print("Expected key_lengths: [1 1 1 1]")
    print("Expected lengths:     [2 1 1]    (2 positions for sample 1, not 3)")
    print()


if __name__ == "__main__":
    print(f"pyfg version: {pyfg.__version__}\n")
    test_standalone_sequence_combine()
    test_grouped_sequence_combine()

    print("Root cause: the framework expands \\x1d multi-values into separate")
    print("sequence positions BEFORE CombineOP runs. CombineOP.Process()")
    print("(combine_feature.cc:302-358) correctly handles per-position combiner")
    print("aggregation when it receives strings with \\x1d intact, but the")
    print("framework pre-flattens them so the combiner sees single values only.")
