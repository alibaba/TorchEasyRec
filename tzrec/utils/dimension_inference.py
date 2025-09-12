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

"""Enhanced dimension inference utilities for backbone blocks."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn

from tzrec.modules.embedding import EmbeddingGroup


class DimensionInfo:
    """Class representing dimension information."""

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...]],
        shape: Optional[Tuple[int, ...]] = None,
        is_list: bool = False,
        feature_dim: Optional[int] = None,
    ) -> None:
        """Initialize DimensionInfo.

        Args:
            dim: Dimension information, int (single dim) or a list/tuple (multiple dim).
            shape: The complete tensor shape information (if available).
            is_list: Indicates whether the output is of a list type.
            feature_dim: Explicitly specified feature dime to override inference.
        """
        self.dim = dim
        self.shape = shape
        self.is_list = is_list
        self._feature_dim = feature_dim

    def __repr__(self) -> str:
        return (
            f"DimensionInfo(dim={self.dim}, shape={self.shape}, "
            f"is_list={self.is_list}, feature_dim={self._feature_dim})"
        )

    def get_feature_dim(self) -> Union[int, List[int], Tuple[int, ...]]:
        """Get feature dimension (last dimension)."""
        # Prefer explicitly specified feature dimensions
        if self._feature_dim is not None:
            return self._feature_dim

        if isinstance(self.dim, (list, tuple)):
            if self.is_list:
                # If list type, return the sum of all dimensions
                return sum(self.dim)
            else:
                # If tensor, return the last dimension
                return self.dim[-1] if self.dim else 0
        return self.dim

    def get_total_dim(self) -> Union[int, List[int], Tuple[int, ...]]:
        """Get the total dimension (for operations such as concat)."""
        if isinstance(self.dim, (list, tuple)):
            return sum(self.dim)
        return self.dim

    def to_list(self) -> List[int]:
        """Convert to list format."""
        if isinstance(self.dim, (list, tuple)):
            return list(self.dim)
        return [self.dim]

    def with_shape(self, shape: Tuple[int, ...]) -> "DimensionInfo":
        """Returns a new DimensionInfo with the specified shape information."""
        feature_dim = shape[-1] if shape else self.get_feature_dim()
        return DimensionInfo(
            dim=self.dim, shape=shape, is_list=self.is_list, feature_dim=feature_dim
        )  # pyre-ignore [7]

    def estimate_shape(
        self, batch_size: Optional[int] = None, seq_len: Optional[int] = None
    ) -> Tuple[int, ...]:
        """Estimate shape based on known information.

        Args:
            batch_size: The batch size.
            seq_len: The sequence length (if applicable).

        Returns:
            The estimated shape as a tuple.
        """
        if self.shape is not None:
            return self.shape

        feature_dim = self.get_feature_dim()

        # 2D  (batch_size, feature_dim)
        if batch_size is not None:
            if seq_len is not None:
                # 3D (batch_size, seq_len, feature_dim)
                return (batch_size, seq_len, feature_dim)  # pyre-ignore [7]
            else:
                # 2D (batch_size, feature_dim)
                return (batch_size, feature_dim)  # pyre-ignore [7]
        else:
            # Only feature dimensions are returned
            return (feature_dim,)


class DimensionInferenceEngine:
    """Dimension inference engine, manages and infers dim information between blocks."""

    def __init__(self) -> None:
        self.block_input_dims: Dict[str, DimensionInfo] = {}
        self.block_output_dims: Dict[str, DimensionInfo] = {}
        self.block_layers: Dict[str, nn.Module] = {}
        self.logger = logging.getLogger(__name__)

    def register_input_dim(self, block_name: str, dim_info: DimensionInfo) -> None:
        """Register the input dimension of the block."""
        self.block_input_dims[block_name] = dim_info
        logging.debug(f"Registered input dim for {block_name}: {dim_info}")

    def register_output_dim(self, block_name: str, dim_info: DimensionInfo) -> None:
        """Register the output dimension of the block."""
        self.block_output_dims[block_name] = dim_info
        logging.debug(f"Registered output dim for {block_name}: {dim_info}")

    def register_layer(self, block_name: str, layer: nn.Module) -> None:
        """Register the layer corresponding to the block."""
        self.block_layers[block_name] = layer

    def get_output_dim(self, block_name: str) -> DimensionInfo:
        """Get the output dimension of the block."""
        return self.block_output_dims.get(block_name)

    def infer_layer_output_dim(
        self, layer: nn.Module, input_dim: DimensionInfo
    ) -> DimensionInfo:
        """Infer the output dimensions of a layer."""
        if hasattr(layer, "output_dim") and callable(layer.output_dim):
            # If the layer has an output_dim method, call it directly
            try:
                output_dim = layer.output_dim()
                # Estimating output shape
                input_shape = input_dim.shape
                if input_shape is not None:
                    output_shape = input_shape[:-1] + (output_dim,)
                else:
                    output_shape = input_dim.estimate_shape()
                    if output_shape:
                        output_shape = output_shape[:-1] + (output_dim,)
                    else:
                        output_shape = None

                return DimensionInfo(
                    dim=output_dim, shape=output_shape, feature_dim=output_dim
                )
            except Exception as e:
                logging.warning(
                    f"Failed to call output_dim on {type(layer).__name__}: {e}"
                )

        # try:
        #     return create_dimension_info_from_layer_output(layer, input_dim)
        # except Exception:
        #     # failed
        #     pass

        # Inferring output dimensions based on layer type
        layer_type = type(layer).__name__

        # if layer_type == "MLP":
        #     if hasattr(layer, "hidden_units") and layer.hidden_units:
        #         output_dim = layer.hidden_units[-1]
        #         return DimensionInfo(output_dim, feature_dim=output_dim)
        #     elif hasattr(layer, "out_features"):
        #         output_dim = layer.out_features
        #         return DimensionInfo(output_dim, feature_dim=output_dim)

        # elif layer_type in ["Linear", "LazyLinear"]:
        #     if hasattr(layer, "out_features"):
        #         output_dim = layer.out_features
        #         return DimensionInfo(output_dim, feature_dim=output_dim)

        # elif layer_type == "DIN":
        #     # DIN
        #     if hasattr(layer, "_sequence_dim") and layer._sequence_dim is not None:
        #         # If it has been initialized, return sequence_dim directly
        #         output_dim = layer._sequence_dim
        #         return DimensionInfo(output_dim, feature_dim=output_dim)
        #     else:
        #         # not initialized yet, infer from input
        #         if isinstance(input_dim, DimensionInfo):
        #             # input is [sequence_features, query_features]concat
        #             # The output dimension is equal to sequence_dim
        #             total_dim = input_dim.get_feature_dim()
        #             if total_dim > 0:
        #                 sequence_dim = total_dim // 2
        #                 logging.info(
        #                     f"DIN output dimension inferred as {sequence_dim} "
        #                     f"(half of input {total_dim})"
        #                 )
        #                 return DimensionInfo(sequence_dim, feature_dim=sequence_dim)

        #         # If inference cannot be made, return the input dimensions
        #         logging.warning(
        #             "Cannot infer DIN output dimension, using input dimension"
        #         )
        #         return input_dim

        # elif layer_type == "DINEncoder":
        #     # DINEncoder
        #     if hasattr(layer, "_sequence_dim") and layer._sequence_dim is not None:
        #         output_dim = layer._sequence_dim
        #         return DimensionInfo(output_dim, feature_dim=output_dim)
        #     elif hasattr(layer, "output_dim") and callable(layer.output_dim):
        #         # use output_dim method
        #         try:
        #             output_dim = layer.output_dim()
        #             return DimensionInfo(output_dim, feature_dim=output_dim)
        #         except Exception:
        #             pass

        #     # If it cannot be obtained from the layer, infer it from the input
        #     if isinstance(input_dim, DimensionInfo):
        #         total_dim = input_dim.get_feature_dim()
        #         if total_dim > 0:
        #             sequence_dim = total_dim // 2
        #             logging.info(
        #                 f"DINEncoder output dimension inferred as {sequence_dim}"
        #             )
        #             return DimensionInfo(sequence_dim, feature_dim=sequence_dim)

        #     # If inference cannot be made, return the input dimensions
        #     logging.warning(
        #         "Cannot infer DINEncoder output dimension, using input dimension"
        #     )
        #     return input_dim

        # elif layer_type in [
        #     "BatchNorm1d",
        #     "LayerNorm",
        #     "Dropout",
        #     "ReLU",
        #     "GELU",
        #     "Tanh",
        # ]:
        #     # These layers do not change the dimensions
        #     return input_dim

        # elif layer_type == "Sequential":
        #     current_dim = input_dim
        #     for sublayer in layer:
        #         current_dim = self.infer_layer_output_dim(sublayer, current_dim)
        #     return current_dim

        # Default: output dimension is the same as input dimension
        logging.warning(
            f"Unknown layer type {layer_type}, assuming output dim == input dim"
        )
        return input_dim

    def apply_input_transforms(
        self,
        input_dim: DimensionInfo,
        input_fn: Optional[str] = None,
        input_slice: Optional[str] = None,
    ) -> DimensionInfo:
        """input_fn and input_slice transforms."""
        current_dim = input_dim

        # use input_slice
        if input_slice is not None:
            current_dim = self._apply_input_slice(current_dim, input_slice)

        # use input_fn
        if input_fn is not None:
            current_dim = self._apply_input_fn(current_dim, input_fn)

        return current_dim

    def _apply_input_slice(
        self, dim_info: DimensionInfo, input_slice: str
    ) -> DimensionInfo:
        """Use input_slice."""
        try:
            # Parsing slice expressions
            slice_expr = eval(
                f"slice{input_slice}"
                if input_slice.startswith("[") and input_slice.endswith("]")
                else input_slice
            )

            if isinstance(slice_expr, int):
                # Single index
                if isinstance(dim_info.dim, (list, tuple)):
                    new_dim = dim_info.dim[slice_expr]
                    return DimensionInfo(new_dim)
                else:
                    raise ValueError(
                        f"Cannot apply index {slice_expr} to scalar dimension "
                        f"{dim_info.dim}"
                    )

            elif isinstance(slice_expr, slice):
                # slice
                if isinstance(dim_info.dim, (list, tuple)):
                    new_dim = dim_info.dim[slice_expr]
                    return DimensionInfo(new_dim, is_list=True)
                else:
                    raise ValueError(
                        f"Cannot apply slice {slice_expr} to scalar dimension "
                        f"{dim_info.dim}"
                    )

            else:
                logging.warning(f"Unsupported slice expression: {input_slice}")
                return dim_info

        except Exception as e:
            logging.error(f"Failed to apply input_slice {input_slice}: {e}")
            return dim_info

    def _apply_input_fn(self, dim_info: DimensionInfo, input_fn: str) -> DimensionInfo:
        """Use input_fn transform - Prioritize using dummy tensor inference."""
        try:
            # First try to use dummy tensor for inference
            try:
                from tzrec.utils.lambda_inference import infer_lambda_output_dim

                result = infer_lambda_output_dim(dim_info, input_fn)
                self.logger.info(
                    f"Successfully inferred output dim using dummy tensor for "
                    f"'{input_fn}': {result}"
                )
                return result
            except Exception as e:
                self.logger.debug(
                    f"Dummy tensor inference failed for '{input_fn}': {e}, "
                    f"falling back to pattern matching"
                )

        except Exception as e:
            logging.error(f"Failed to apply input_fn {input_fn}: {e}")
            return dim_info

    def merge_input_dims(
        self, input_dims: List[DimensionInfo], merge_mode: str = "concat"
    ) -> DimensionInfo:
        """Merge multiple input dimensions."""
        if not input_dims:
            raise ValueError("No input dimensions to merge")

        if len(input_dims) == 1:
            return input_dims[0]

        if merge_mode == "concat":
            # Splicing mode: Dimension addition
            total_dim = sum(dim_info.get_total_dim() for dim_info in input_dims)
            return DimensionInfo(total_dim)

        elif merge_mode == "list":
            # List mode: Keep as list
            dims = []
            for dim_info in input_dims:
                if dim_info is not None:
                    dims.extend(dim_info.to_list())
            return DimensionInfo(dims, is_list=True)

        elif merge_mode == "stack":
            # Stacked Mode: Adding a Dimension
            if not all(
                dim_info.get_feature_dim() == input_dims[0].get_feature_dim()
                for dim_info in input_dims
            ):
                raise ValueError(
                    "All inputs must have same feature dimension for stacking"
                )
            feature_dim = input_dims[0].get_feature_dim()
            return DimensionInfo(feature_dim)

        else:
            raise ValueError(f"Unsupported merge mode: {merge_mode}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about dimension inference."""
        return {
            "total_blocks": len(self.block_output_dims),
            "input_dims": {
                name: str(dim) for name, dim in self.block_input_dims.items()
            },
            "output_dims": {
                name: str(dim) for name, dim in self.block_output_dims.items()
            },
        }


def create_dimension_info_from_embedding(
    embedding_group: EmbeddingGroup, group_name: str, batch_size: Optional[int] = None
) -> DimensionInfo:
    """Create dimension information from an embedding group.

    Args:
        embedding_group: The embedding group object.
        group_name: The name of the group.
        batch_size: The batch size (optional, used for estimating the full shape).

    Returns:
        A DimensionInfo object containing feature dimension information.
    """
    try:
        total_dim = embedding_group.group_total_dim(group_name)

        # Estimate shape information
        if batch_size is not None:
            estimated_shape = (batch_size, total_dim)
        else:
            estimated_shape = None

        return DimensionInfo(
            dim=total_dim,
            shape=estimated_shape,
            feature_dim=total_dim,  # Explicitly specify the feature dimension
        )
    except Exception as e:
        logging.error(f"Failed to get dimension from embedding group {group_name}: {e}")
        return DimensionInfo(0, feature_dim=0)
