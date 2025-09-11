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

import traceback
import unittest


class DimensionInferenceTest(unittest.TestCase):
    """Test class for DIN automatic dimension inference functionality."""

    def test_din_module_import(self):
        """Test DIN module import functionality."""
        try:
            from tzrec.utils.load_class import load_torch_layer

            # Test loading DINEncoder
            din_cls, is_customize = load_torch_layer("DIN")
            self.assertEqual(
                is_customize, True, "DINEncoder should be a customized class"
            )

            self.assertIsNotNone(din_cls, "DINEncoder should not be None")

            # Check parameters of DINEncoder
            import inspect

            sig = inspect.signature(din_cls.__init__)
            self.assertEqual(
                len(sig.parameters), 7, "DINEncoder should have 7 parameters"
            )
            self.assertEqual(
                list(sig.parameters.keys()),
                [
                    "self",
                    "sequence_dim",
                    "query_dim",
                    "input",
                    "attn_mlp",
                    "max_seq_length",
                    "kwargs",
                ],
            )

        except Exception as e:
            self.fail(f"Error importing DINEncoder: {e}")
            traceback.print_exc()

    def test_dimension_inference(self):
        """Test dimension inference functionality."""

        try:
            from tzrec.modules.sequence import DINEncoder
            from tzrec.utils.dimension_inference import (
                DimensionInferenceEngine,
                DimensionInfo,
            )

            # Create a dimension inference engine
            engine = DimensionInferenceEngine()

            # Create a DINEncoder (provide necessary parameters)
            din = DINEncoder(
                sequence_dim=128,
                query_dim=96,
                input="seq",
                attn_mlp={"hidden_units": [256, 64]},
                max_seq_length=100,
            )

            self.assertEqual(din.output_dim(), 128)

            # Test input dimension info
            input_total_dim = 224
            input_dim_info = DimensionInfo(
                dim=input_total_dim,
                shape=(32, input_total_dim),
                feature_dim=input_total_dim,
            )

            # Infer output dimension
            output_dim_info = engine.infer_layer_output_dim(din, input_dim_info)

            # Validate inference result
            expected_output_dim = 128
            actual_output_dim = output_dim_info.get_feature_dim()
            self.assertEqual(
                actual_output_dim,
                expected_output_dim,
                f"Expected output dim {expected_output_dim}, got {actual_output_dim}",
            )

        except Exception as e:
            self.fail(f"Dimension inference failed: {e}")
            traceback.print_exc()

    def test_automatic_dimension_inference(self):
        """Test automatic dimension inference (simulate backbone scenario)."""
        try:
            import inspect

            from tzrec.modules.sequence import DINEncoder

            # Simulate the process of automatic dimension inference
            din_cls = DINEncoder
            sig = inspect.signature(din_cls.__init__)

            self.assertEqual(
                [p for p in sig.parameters.keys() if p != "self"],
                [
                    "sequence_dim",
                    "query_dim",
                    "input",
                    "attn_mlp",
                    "max_seq_length",
                    "kwargs",
                ],
            )

            # Simulate kwargs dictionary (result of proto configuration parsing)
            kwargs = {
                "input": "seq",
                "attn_mlp": {"hidden_units": [256, 64]},
                "max_seq_length": 100,
            }

            # Simulate logic for automatic dimension inference
            if "sequence_dim" not in kwargs:
                kwargs["sequence_dim"] = 128

            if "query_dim" not in kwargs:
                kwargs["query_dim"] = 96

            self.assertEqual(kwargs["sequence_dim"], 128)
            self.assertEqual(kwargs["query_dim"], 96)
            self.assertEqual(kwargs["input"], "seq")
            self.assertEqual(kwargs["attn_mlp"], {"hidden_units": [256, 64]})
            self.assertEqual(kwargs["max_seq_length"], 100)

            # Create DINEncoder instance
            din = din_cls(**kwargs)
            self.assertEqual(din.output_dim(), 128)

        except Exception as e:
            self.fail(f"Automatic dimension inference failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    unittest.main()
