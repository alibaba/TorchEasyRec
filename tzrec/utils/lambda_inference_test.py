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

"""Test lambda layer dimension inference in backbone."""

import unittest
import torch
import logging
from tzrec.utils.dimension_inference import DimensionInfo
from tzrec.modules.backbone import LambdaWrapper
from tzrec.utils.lambda_inference import LambdaOutputDimInferrer

logging.basicConfig(level=logging.DEBUG)


class TestLambdaDimensionInference(unittest.TestCase):
    """Test the dimension inference function of the lambda module."""
    
    def test_lambda_wrapper_simple(self):
        """Testing simple lambda expressions"""
        # create input dimension info
        input_dim = DimensionInfo(16, shape=(32, 16))
        
        # create lambda wrapper
        lambda_wrapper = LambdaWrapper("lambda x: x", "identity")
        
        # infer output dimension
        output_dim = lambda_wrapper.infer_output_dim(input_dim)
        
        print(f"Output dim: {output_dim}")
        
        self.assertEqual(output_dim.get_feature_dim(), 16)
    
    def test_lambda_wrapper_sum(self):
        """Testing the lambda expression for the sum operation."""
        # 3D tensor
        input_dim = DimensionInfo(16, shape=(32, 10, 16))  # batch_size=32, seq_len=10, feature_dim=16
        
        # create lambda wrapper - Summing over the sequence dimension
        lambda_wrapper = LambdaWrapper("lambda x: x.sum(dim=1)", "sum_seq")
        
        # infer output dimension
        output_dim = lambda_wrapper.infer_output_dim(input_dim)
        
        print(f"Input dim: {input_dim}")
        print(f"Output dim: {output_dim}")
        
        # sum over the sequence dimension, should get (32, 16)
        self.assertEqual(output_dim.get_feature_dim(), 16)
        self.assertEqual(output_dim.shape, (32, 16))
    
    def test_lambda_wrapper_list_conversion(self):
        """测试转换为list的lambda表达式"""
        # 创建输入维度信息
        input_dim = DimensionInfo(16, shape=(32, 16))
        
        # 创建lambda wrapper - 转换为list
        lambda_wrapper = LambdaWrapper("lambda x: [x]", "to_list")
        
        # 推断输出维度
        output_dim = lambda_wrapper.infer_output_dim(input_dim)
        
        print(f"Input dim: {input_dim}")
        print(f"Output dim: {output_dim}")
        
        # 转换为list后，维度应该保持，但标记为list类型
        self.assertEqual(output_dim.get_feature_dim(), 16)
        self.assertTrue(output_dim.is_list)
    
    def test_lambda_wrapper_execution(self):
        """Test the execution function of the lambda wrapper."""
        # create lambda wrapper
        lambda_wrapper = LambdaWrapper("lambda x: x * 2", "multiply")
        
        # create test input
        test_input = torch.randn(4, 8)
        
        # execute
        output = lambda_wrapper(test_input)
        
        # expected output
        expected = test_input * 2
        torch.testing.assert_close(output, expected)
    
    def test_direct_inferrer(self):
        """Testing LambdaOutputDimInferrer"""
        # create inferrer
        inferrer = LambdaOutputDimInferrer()
        
        # create input dimension info
        input_dim = DimensionInfo(16, shape=(32, 16))
        
        test_cases = [
            ("lambda x: x", 16),
            ("lambda x: x.sum(dim=-1)", 32),
            ("lambda x: x.sum(dim=-1, keepdim=True)", 1),
            ("lambda x: [x]", 16),
        ]
        
        for lambda_expr, expected_feature_dim in test_cases:
            with self.subTest(lambda_expr=lambda_expr):
                output_dim = inferrer.infer_output_dim(input_dim, lambda_expr)
                print(f"Lambda: {lambda_expr}")
                print(f"Input: {input_dim}")
                print(f"Output: {output_dim}")
                print(f"Expected feature dim: {expected_feature_dim}")
                print("---")
                
                if expected_feature_dim is not None:
                    self.assertEqual(output_dim.get_feature_dim(), expected_feature_dim)


if __name__ == "__main__":
    unittest.main()
