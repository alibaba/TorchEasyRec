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
import traceback

class TestDINAutoInference(unittest.TestCase):
    """Test class for DIN automatic dimension inference functionality."""

    def test_din_module_import(self):
        """Test DIN module import functionality."""
        print("=== Testing DIN Module Import ===")
        
        try:
            from tzrec.utils.load_class import load_torch_layer
            
            # Test loading DINEncoder
            din_cls, is_customize = load_torch_layer("DIN")
            print(f"DINEncoder loaded: {din_cls}")
            print(f"Is customize: {is_customize}")
            
            self.assertIsNotNone(din_cls, "DINEncoder should not be None")
            
            # Check parameters of DINEncoder
            import inspect
            sig = inspect.signature(din_cls.__init__)
            print(f"DINEncoder parameters: {list(sig.parameters.keys())}")
            
        except Exception as e:
            self.fail(f"Error importing DINEncoder: {e}")
            traceback.print_exc()

    def test_dimension_inference(self):
        """Test dimension inference functionality."""
        print("\n=== Testing Dimension Inference ===")
        
        try:
            from tzrec.utils.dimension_inference import DimensionInfo, DimensionInferenceEngine
            from tzrec.modules.sequence import DINEncoder
            
            # Create a dimension inference engine
            engine = DimensionInferenceEngine()
            
            # Create a DINEncoder (provide necessary parameters)
            din = DINEncoder(
                sequence_dim=128,
                query_dim=96,
                input="seq",
                attn_mlp={"hidden_units": [256, 64]},
                max_seq_length=100
            )
            
            print(f"Created DINEncoder: {din}")
            print(f"DINEncoder output_dim: {din.output_dim()}")
            
            # Test input dimension info
            input_total_dim = 224
            input_dim_info = DimensionInfo(
                dim=input_total_dim,
                shape=(32, input_total_dim),
                feature_dim=input_total_dim
            )
            
            print(f"Input dimension info: {input_dim_info}")
            
            # Infer output dimension
            output_dim_info = engine.infer_layer_output_dim(din, input_dim_info)
            print(f"Inferred output dimension info: {output_dim_info}")
            
            # Validate inference result
            expected_output_dim = 128
            actual_output_dim = output_dim_info.get_feature_dim()
            self.assertEqual(actual_output_dim, expected_output_dim, 
                             f"Expected output dim {expected_output_dim}, got {actual_output_dim}")
            
        except Exception as e:
            self.fail(f"Dimension inference failed: {e}")
            traceback.print_exc()

    def test_automatic_dimension_inference(self):
        """Test automatic dimension inference (simulate backbone scenario)."""
        print("\n=== Testing Automatic Dimension Inference ===")
        
        try:
            from tzrec.modules.sequence import DINEncoder
            from tzrec.utils.dimension_inference import DimensionInfo
            import inspect
            
            # Simulate the process of automatic dimension inference
            din_cls = DINEncoder
            sig = inspect.signature(din_cls.__init__)
            
            print(f"DINEncoder signature: {sig}")
            print(f"Required parameters: {[p for p in sig.parameters.keys() if p != 'self']}")
            
            # Simulate kwargs dictionary (result of proto configuration parsing)
            kwargs = {
                "input": "seq",
                "attn_mlp": {"hidden_units": [256, 64]},
                "max_seq_length": 100
            }
            
            print(f"Initial kwargs: {kwargs}")
            
            # Simulate logic for automatic dimension inference
            if "sequence_dim" not in kwargs:
                kwargs["sequence_dim"] = 128
                print("Auto-inferred sequence_dim: 128")
                
            if "query_dim" not in kwargs:
                kwargs["query_dim"] = 96
                print("Auto-inferred query_dim: 96")
            
            print(f"Final kwargs: {kwargs}")
            
            # Create DINEncoder instance
            din = din_cls(**kwargs)
            print(f"✓ Successfully created DINEncoder with auto-inferred dimensions")
            print(f"DINEncoder output_dim: {din.output_dim()}")
            
        except Exception as e:
            self.fail(f"Automatic dimension inference failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    unittest.main()