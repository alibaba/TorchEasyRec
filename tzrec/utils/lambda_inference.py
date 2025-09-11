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
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from tzrec.utils.dimension_inference import DimensionInfo


class LambdaOutputDimInferrer:
    """Lambda expression output dimension inferer.

    Infer the output dimensions by creating a dummy tensor and
    executing the lambda expression.
    """

    def __init__(self) -> None:
        """Initialize the Lambda output dimension inferrer."""
        self.logger = logging.getLogger(__name__)

    def infer_output_dim(
        self,
        input_dim_info: DimensionInfo,
        lambda_fn_str: str,
        dummy_batch_size: int = 2,
        dummy_seq_len: Optional[int] = None,
    ) -> DimensionInfo:
        """Infer the output dimensions of a lambda expression.

        Args:
            input_dim_info: The input dimension information.
            lambda_fn_str: The lambda expression string, such as "lambda x: x.sum".
            dummy_batch_size: The batch size used to create a dummy tensor.
            dummy_seq_len: The sequence length used to create a dummy tensor (optional).

        Returns:
            The inferred output dimension information.
        """
        # If the first dimension of input_dim_info.shape
        # is not None, use it as batch_size
        if (
            input_dim_info.shape[0] is not None and len(input_dim_info.shape) > 0
        ):  # pyre-ignore[6]
            dummy_batch_size = input_dim_info.shape[0]
        try:
            # 1. Create a dummy tensor
            dummy_tensor = self._create_dummy_tensor(
                input_dim_info, dummy_batch_size, dummy_seq_len
            )

            # 2. Compile the Lambda function
            lambda_fn = self._compile_lambda_function(lambda_fn_str)

            # 3. Execute the Lambda function
            with torch.no_grad():  # No gradient computation needed
                output_tensor = lambda_fn(dummy_tensor)

            # 4. Parse the output and create a DimensionInfo
            return self._analyze_output(output_tensor, input_dim_info)

        except Exception as e:
            self.logger.error(
                f"Failed to infer output dim for lambda '{lambda_fn_str}': {e}"
            )
            # Return the input dimension as fallback on error
            self.logger.warning("Falling back to input dimension")
            return input_dim_info

    def _create_dummy_tensor(
        self,
        input_dim_info: DimensionInfo,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Create a dummy tensor for testing."""
        if input_dim_info.shape is not None:
            # if there is full shape info, use it
            shape = input_dim_info.shape
            # replace the first dimension with dummy_batch_size
            if len(shape) > 0:
                shape = (batch_size,) + shape[1:]
        else:
            # compute shape based on feature dimension
            feature_dim = input_dim_info.get_feature_dim()

            if seq_len is not None:
                # 3D: (batch_size, seq_len, feature_dim)
                shape = (batch_size, seq_len, feature_dim)
            else:
                # 2D: (batch_size, feature_dim)
                shape = (batch_size, feature_dim)

        dummy_tensor = torch.randn(shape, dtype=torch.float32)  # pyre-ignore[7]
        self.logger.debug(f"Created dummy tensor with shape: {shape}")
        return dummy_tensor

    def _compile_lambda_function(
        self, lambda_fn_str: str
    ) -> Union[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[Iterable[torch.Tensor]], torch.Tensor],
    ]:
        """Compile lambda function string."""
        try:
            lambda_fn_str = lambda_fn_str.strip()

            lambda_fn = eval(lambda_fn_str)

            if not callable(lambda_fn):
                raise ValueError(
                    f"Lambda expression does not evaluate to a callable: "
                    f"{lambda_fn_str}"
                )

            return lambda_fn  # pyre-ignore[7]

        except Exception as e:
            self.logger.error(
                f"Failed to compile lambda function '{lambda_fn_str}': {e}"
            )
            raise ValueError(f"Invalid lambda expression: {lambda_fn_str}") from e

    def _analyze_output(
        self, output_tensor: torch.Tensor, input_dim_info: DimensionInfo
    ) -> DimensionInfo:
        """Analyze the output tensor and create DimensionInfo."""
        if isinstance(output_tensor, (list, tuple)):
            # if the output is list/tuple
            if len(output_tensor) == 0:
                return DimensionInfo(0, is_list=True)

            # analyze the dimension of each element in the list
            dims = []
            shapes = []
            for item in output_tensor:
                if isinstance(item, torch.Tensor):
                    dims.append(item.shape[-1] if len(item.shape) > 0 else 1)
                    shapes.append(item.shape)
                else:
                    # not a tensor
                    dims.append(1)
                    shapes.append((1,))

            return DimensionInfo(
                dim=dims,
                shape=shapes[0] if len(set(shapes)) == 1 else None,
                is_list=True,
                feature_dim=sum(dims),
            )

        elif isinstance(output_tensor, torch.Tensor):
            # Standard tensor output
            output_shape = tuple(output_tensor.shape)
            feature_dim = output_shape[-1] if len(output_shape) > 0 else 1

            return DimensionInfo(
                dim=feature_dim, shape=output_shape, feature_dim=feature_dim
            )

        else:
            # other types of output
            self.logger.warning(f"Unexpected output type: {type(output_tensor)}")
            return DimensionInfo(1, feature_dim=1)


class LambdaLayer(nn.Module):
    """Lambda expression layer, providing output_dim method."""

    def __init__(
        self,
        lambda_fn_str: str,
        input_dim_info: DimensionInfo,
        name: str = "lambda_layer",
    ) -> None:
        """Initialize the Lambda layer.

        Args:
            lambda_fn_str: lambda expression string
            input_dim_info: Input dimension information (used to infer output dimension)
            name: Layer name
        """
        super().__init__()
        self.lambda_fn_str = lambda_fn_str
        self.name = name
        self._input_dim_info = input_dim_info
        self._output_dim_info = None
        self._lambda_fn = None

        # compile the lambda function
        self._compile_function()

        # if there is input dimension info, infer output dimension immediately
        if input_dim_info is not None:
            self._infer_output_dim()

    def _compile_function(self) -> None:
        """Compile lambda function."""
        inferrer = LambdaOutputDimInferrer()
        self._lambda_fn = inferrer._compile_lambda_function(self.lambda_fn_str)

    def _infer_output_dim(self) -> None:
        """Infer output dimension."""
        if self._input_dim_info is None:
            raise ValueError(
                "Cannot infer output dimension without input dimension info"
            )

        inferrer = LambdaOutputDimInferrer()
        self._output_dim_info = inferrer.infer_output_dim(
            self._input_dim_info, self.lambda_fn_str
        )

    def set_input_dim_info(self, input_dim_info: DimensionInfo) -> None:
        """Set input dimension info and re-infer output dimension."""
        self._input_dim_info = input_dim_info
        self._infer_output_dim()

    def output_dim(self) -> int:
        """Get the output feature dimension."""
        if self._output_dim_info is None:
            raise ValueError(
                f"Output dimension not available for {self.name}. "
                "Make sure to set input_dim_info first."
            )
        return self._output_dim_info.get_feature_dim()

    def get_output_dim_info(self) -> DimensionInfo:
        """Get the output dimension info."""
        if self._output_dim_info is None:
            raise ValueError(
                f"Output dimension not available for {self.name}. "
                "Make sure to set input_dim_info first."
            )
        return self._output_dim_info

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Forward."""
        if self._lambda_fn is None:
            raise ValueError("Lambda function not compiled")
        return self._lambda_fn(x)

    def __repr__(self) -> str:
        return f"LambdaLayer(name={self.name}, lambda_fn='{self.lambda_fn_str}')"


def create_lambda_layer_from_input_fn(
    input_fn_str: str, input_dim_info: DimensionInfo, name: str = "input_fn_layer"
) -> LambdaLayer:
    """Create a Lambda layer from an input_fn string.

    Convert the input_fn in the backbone configuration
    into a layer with an output_dim method.
    """
    return LambdaLayer(
        lambda_fn_str=input_fn_str, input_dim_info=input_dim_info, name=name
    )


def infer_lambda_output_dim(
    input_dim_info: DimensionInfo, lambda_fn_str: str
) -> DimensionInfo:
    """Infer the output dimensions of a lambda expression."""
    inferrer = LambdaOutputDimInferrer()
    return inferrer.infer_output_dim(input_dim_info, lambda_fn_str)
