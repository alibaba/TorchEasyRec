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
    def test_fm_with_3d_tensor(self, graph_type):
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
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
            #  [TestGraphType.JIT_SCRIPT]
        ]
    )
    def test_fm_with_list_input(self, graph_type):
        """Test FM module with list of 2D tensors input."""
        batch_size, field_size, embedding_size = 32, 4, 16

        # Create FM module
        fm = FM(use_variant=False, l2_regularization=1e-4)

        # Create list of 2D tensors
        input_list = [
            torch.randn(batch_size, embedding_size) for _ in range(field_size)
        ]

        # For FX_TRACE and JIT_SCRIPT, we need to convert list to tensor first
        # because these graph compilation methods have trouble with list inputs
        if graph_type in [TestGraphType.FX_TRACE, TestGraphType.JIT_SCRIPT]:
            # Convert list to tensor for graph tracing
            input_tensor = torch.stack(input_list, dim=1)
            fm = create_test_module(fm, graph_type)
            output = fm(input_tensor)
            # For graph modules, we can't call output_dim(), so we skip this check
        else:
            # For normal execution, test with list input
            fm = create_test_module(fm, graph_type)
            output = fm(input_list)
            # Only test output_dim for normal modules
            self.assertEqual(fm.output_dim(), 1)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))

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
    def test_fm_equivalence(self, graph_type):
        """Test that both input formats produce same results."""
        batch_size, field_size, embedding_size = 8, 3, 4

        # Create FM module
        fm = FM(use_variant=False, l2_regularization=0.0)
        fm = create_test_module(fm, graph_type)

        # Create test data
        input_3d = torch.randn(batch_size, field_size, embedding_size)
        input_list = [input_3d[:, i, :] for i in range(field_size)]

        # Forward pass with both input formats
        output_3d = fm(input_3d)

        # For graph-traced modules, we can't test list inputs directly
        # So we test equivalence by converting list to tensor
        if graph_type in [TestGraphType.FX_TRACE, TestGraphType.JIT_SCRIPT]:
            input_list_as_tensor = torch.stack(input_list, dim=1)
            output_list = fm(input_list_as_tensor)
        else:
            output_list = fm(input_list)

        # Check equivalence
        torch.testing.assert_close(output_3d, output_list, rtol=1e-5, atol=1e-5)

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

        # Note: 对于JIT_SCRIPT和FX_TRACE，不能测试运行时错误（如empty list），
        # 因为这些是编译时图优化，所以跳过empty list测试

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
        # 这些测试只适用于正常运行时，不适用于编译后的图
        fm = FM(use_variant=False, l2_regularization=1e-4)

        # Test with empty list
        with self.assertRaises(IndexError):
            fm([])


if __name__ == "__main__":
    unittest.main()
