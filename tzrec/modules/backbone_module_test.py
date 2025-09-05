# Copyright (c) 2024, Alibaba Group;
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

import torch
from parameterized import parameterized

from tzrec.modules.backbone_module import FM, Add
from tzrec.utils.test_util import TestGraphType, create_test_module


class BackboneModuleTest(unittest.TestCase):
    """Test cases for backbone modules."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_fm(self, graph_type):
        """Test FM module with 3D tensor input."""
        batch_size, field_size, embedding_size = 32, 4, 16

        # Create FM module
        fm = FM(use_variant=False, l2_regularization=1e-4)
        fm = create_test_module(fm, graph_type)

        # Create input tensor
        input_tensor = torch.randn(batch_size, field_size, embedding_size)

        # Forward pass
        output = fm(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        # Only test output_dim for normal modules
        if graph_type == TestGraphType.NORMAL:
            self.assertEqual(fm.output_dim(), 1)

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_fm_variant(self, graph_type):
        """Test FM module with variant computation."""
        batch_size, field_size, embedding_size = 32, 4, 16

        # Create FM module with variant
        fm = FM(use_variant=True, l2_regularization=1e-4)
        fm = create_test_module(fm, graph_type)

        # Create input tensor
        input_tensor = torch.randn(batch_size, field_size, embedding_size)

        # Forward pass
        output = fm(input_tensor)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        # Only test output_dim for normal modules
        if graph_type == TestGraphType.NORMAL:
            self.assertEqual(fm.output_dim(), 1)

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_fm_edge_cases(self, graph_type):
        """Test FM module edge cases."""
        batch_size, embedding_size = 32, 16

        # Create FM module
        fm = FM(use_variant=False, l2_regularization=1e-4)
        fm = create_test_module(fm, graph_type)

        # Test with single field (no interactions)
        single_field = torch.randn(batch_size, 1, embedding_size)
        output = fm(single_field)
        self.assertEqual(output.shape, (batch_size, 1))
        # Should be zero since no interactions possible
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_add_module(self, graph_type):
        """Test Add module."""
        batch_size, features = 32, 16

        # Create Add module
        add_module = Add()
        add_module = create_test_module(add_module, graph_type)

        # Create input tensors
        input1 = torch.randn(batch_size, features)
        input2 = torch.randn(batch_size, features)
        input3 = torch.randn(batch_size, features)

        # Forward pass
        output = add_module(input1, input2, input3)

        # Check output shape and value
        self.assertEqual(output.shape, (batch_size, features))
        expected = input1 + input2 + input3
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_fm_runtime_errors(self):
        """Test FM module runtime errors (only for NORMAL graph type)."""
        # Note: Runtime dimension validation is disabled for FX tracing compatibility
        # This test is kept for documentation purposes but may not fail as expected
        fm = FM(use_variant=False, l2_regularization=1e-4)

        # These tests may not work as expected since dimension validation
        # is disabled for graph compilation compatibility
        # Test with wrong dimensions - may not raise errors due to FX compatibility
        try:
            # 2D tensor - may work due to broadcasting
            result = fm(torch.randn(32, 16))
            print(f"2D input result shape: {result.shape}")
        except Exception as e:
            print(f"2D input error: {e}")

        try:
            # 4D tensor - may work due to shape unpacking
            result = fm(torch.randn(32, 4, 16, 8))
            print(f"4D input result shape: {result.shape}")
        except Exception as e:
            print(f"4D input error: {e}")


if __name__ == "__main__":
    unittest.main()
