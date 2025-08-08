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

"""Lambda expression dimension inference module."""

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from tzrec.layers.dimension_inference import DimensionInfo


class LambdaOutputDimInferrer:
    """Lambda表达式输出维度推断器

    通过创建dummy tensor并执行lambda表达式来推断输出维度
    """

    def __init__(self, safe_mode: bool = True):
        """Args:
        safe_mode: 安全模式，在安全模式下会进行额外的检查和错误处理
        """
        self.safe_mode = safe_mode
        self.logger = logging.getLogger(__name__)

    def infer_output_dim(
        self,
        input_dim_info: DimensionInfo,
        lambda_fn_str: str,
        dummy_batch_size: int = 2,
        dummy_seq_len: Optional[int] = None,
    ) -> DimensionInfo:
        """推断lambda表达式的输出维度

        Args:
            input_dim_info: 输入维度信息
            lambda_fn_str: lambda表达式字符串，如 "lambda x: x.sum(dim=1)"
            dummy_batch_size: 用于创建dummy tensor的batch size
            dummy_seq_len: 用于创建dummy tensor的序列长度（可选）

        Returns:
            推断出的输出维度信息
        """
        try:
            # 1. 创建dummy tensor
            dummy_tensor = self._create_dummy_tensor(
                input_dim_info, dummy_batch_size, dummy_seq_len
            )

            # 2. 编译lambda函数
            lambda_fn = self._compile_lambda_function(lambda_fn_str)

            # 3. 执行lambda函数
            with torch.no_grad():  # 不需要梯度计算
                output_tensor = lambda_fn(dummy_tensor)

            # 4. 分析输出并创建DimensionInfo
            return self._analyze_output(output_tensor, input_dim_info)

        except Exception as e:
            self.logger.error(
                f"Failed to infer output dim for lambda '{lambda_fn_str}': {e}"
            )
            if self.safe_mode:
                # 安全模式下返回输入维度
                self.logger.warning("Falling back to input dimension")
                return input_dim_info
            else:
                raise

    def _create_dummy_tensor(
        self,
        input_dim_info: DimensionInfo,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """创建用于测试的dummy tensor"""
        if input_dim_info.shape is not None:
            # 如果有完整的shape信息，使用它
            shape = input_dim_info.shape
            # 替换第一个维度为dummy_batch_size
            if len(shape) > 0:
                shape = (batch_size,) + shape[1:]
        else:
            # 根据特征维度估算shape
            feature_dim = input_dim_info.get_feature_dim()

            if seq_len is not None:
                # 3D: (batch_size, seq_len, feature_dim)
                shape = (batch_size, seq_len, feature_dim)
            else:
                # 2D: (batch_size, feature_dim)
                shape = (batch_size, feature_dim)

        # 创建随机tensor
        dummy_tensor = torch.randn(shape, dtype=torch.float32)
        self.logger.debug(f"Created dummy tensor with shape: {shape}")
        return dummy_tensor

    def _compile_lambda_function(self, lambda_fn_str: str) -> Callable:
        """编译lambda函数字符串"""
        try:
            # 清理字符串
            lambda_fn_str = lambda_fn_str.strip()

            # 安全检查
            if self.safe_mode:
                self._validate_lambda_safety(lambda_fn_str)

            # 编译lambda函数
            # 为了安全起见，我们限制可用的全局变量
            safe_globals = {
                "torch": torch,
                "__builtins__": {},
                # 添加常用的torch函数
                "cat": torch.cat,
                "stack": torch.stack,
                "sum": torch.sum,
                "mean": torch.mean,
                "max": torch.max,
                "min": torch.min,
            }

            lambda_fn = eval(lambda_fn_str, safe_globals, {})

            if not callable(lambda_fn):
                raise ValueError(
                    f"Lambda expression does not evaluate to a callable: {lambda_fn_str}"
                )

            return lambda_fn

        except Exception as e:
            self.logger.error(
                f"Failed to compile lambda function '{lambda_fn_str}': {e}"
            )
            raise ValueError(f"Invalid lambda expression: {lambda_fn_str}") from e

    def _validate_lambda_safety(self, lambda_fn_str: str) -> None:
        """验证lambda表达式的安全性"""
        # 检查危险的关键词
        dangerous_keywords = [
            "import",
            "exec",
            "eval",
            "open",
            "file",
            "__import__",
            "getattr",
            "setattr",
            "delattr",
            "globals",
            "locals",
            "vars",
            "dir",
            "compile",
            "reload",
        ]

        lambda_lower = lambda_fn_str.lower()
        for keyword in dangerous_keywords:
            if keyword in lambda_lower:
                raise ValueError(
                    f"Potentially unsafe lambda expression contains '{keyword}': {lambda_fn_str}"
                )

        # 检查是否是有效的lambda表达式格式
        if not lambda_fn_str.strip().startswith("lambda"):
            raise ValueError(f"Expression must be a lambda function: {lambda_fn_str}")

    def _analyze_output(
        self, output_tensor: torch.Tensor, input_dim_info: DimensionInfo
    ) -> DimensionInfo:
        """分析输出tensor并创建DimensionInfo"""
        if isinstance(output_tensor, (list, tuple)):
            # 如果输出是list/tuple
            if len(output_tensor) == 0:
                return DimensionInfo(0, is_list=True)

            # 分析list中每个元素的维度
            dims = []
            shapes = []
            for item in output_tensor:
                if isinstance(item, torch.Tensor):
                    dims.append(item.shape[-1] if len(item.shape) > 0 else 1)
                    shapes.append(item.shape)
                else:
                    # 非tensor元素
                    dims.append(1)
                    shapes.append((1,))

            return DimensionInfo(
                dim=dims,
                shape=shapes[0]
                if len(set(shapes)) == 1
                else None,  # 如果所有shape相同则保留
                is_list=True,
                feature_dim=sum(dims),
            )

        elif isinstance(output_tensor, torch.Tensor):
            # 标准tensor输出
            output_shape = tuple(output_tensor.shape)
            feature_dim = output_shape[-1] if len(output_shape) > 0 else 1

            return DimensionInfo(
                dim=feature_dim, shape=output_shape, feature_dim=feature_dim
            )

        else:
            # 其他类型的输出
            self.logger.warning(f"Unexpected output type: {type(output_tensor)}")
            return DimensionInfo(1, feature_dim=1)


class LambdaLayer(nn.Module):
    """Lambda表达式层，提供output_dim方法"""

    def __init__(
        self,
        lambda_fn_str: str,
        input_dim_info: Optional[DimensionInfo] = None,
        name: str = "lambda_layer",
    ):
        """Args:
        lambda_fn_str: lambda表达式字符串
        input_dim_info: 输入维度信息（用于推断输出维度）
        name: 层的名称
        """
        super().__init__()
        self.lambda_fn_str = lambda_fn_str
        self.name = name
        self._input_dim_info = input_dim_info
        self._output_dim_info = None
        self._lambda_fn = None

        # 编译lambda函数
        self._compile_function()

        # 如果有输入维度信息，立即推断输出维度
        if input_dim_info is not None:
            self._infer_output_dim()

    def _compile_function(self):
        """编译lambda函数"""
        inferrer = LambdaOutputDimInferrer(safe_mode=True)
        self._lambda_fn = inferrer._compile_lambda_function(self.lambda_fn_str)

    def _infer_output_dim(self):
        """推断输出维度"""
        if self._input_dim_info is None:
            raise ValueError(
                "Cannot infer output dimension without input dimension info"
            )

        inferrer = LambdaOutputDimInferrer(safe_mode=True)
        self._output_dim_info = inferrer.infer_output_dim(
            self._input_dim_info, self.lambda_fn_str
        )

    def set_input_dim_info(self, input_dim_info: DimensionInfo):
        """设置输入维度信息并推断输出维度"""
        self._input_dim_info = input_dim_info
        self._infer_output_dim()

    def output_dim(self) -> int:
        """获取输出维度，类似MLP.output_dim()"""
        if self._output_dim_info is None:
            raise ValueError(
                f"Output dimension not available for {self.name}. "
                "Make sure to set input_dim_info first."
            )
        return self._output_dim_info.get_feature_dim()

    def get_output_dim_info(self) -> DimensionInfo:
        """获取完整的输出维度信息"""
        if self._output_dim_info is None:
            raise ValueError(
                f"Output dimension not available for {self.name}. "
                "Make sure to set input_dim_info first."
            )
        return self._output_dim_info

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list, tuple]:
        """前向传播"""
        if self._lambda_fn is None:
            raise ValueError("Lambda function not compiled")
        return self._lambda_fn(x)

    def __repr__(self):
        return f"LambdaLayer(name={self.name}, lambda_fn='{self.lambda_fn_str}')"


def create_lambda_layer_from_input_fn(
    input_fn_str: str, input_dim_info: DimensionInfo, name: str = "input_fn_layer"
) -> LambdaLayer:
    """从input_fn字符串创建Lambda层

    这个函数可以用于将backbone配置中的input_fn转换为具有output_dim方法的层
    """
    return LambdaLayer(
        lambda_fn_str=input_fn_str, input_dim_info=input_dim_info, name=name
    )


# 便捷函数
def infer_lambda_output_dim(
    input_dim_info: DimensionInfo, lambda_fn_str: str, safe_mode: bool = True
) -> DimensionInfo:
    """便捷函数：推断lambda表达式的输出维度"""
    inferrer = LambdaOutputDimInferrer(safe_mode=safe_mode)
    return inferrer.infer_output_dim(input_dim_info, lambda_fn_str)
