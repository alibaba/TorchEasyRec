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
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn


class DimensionInfo:
    """表示维度信息的类，支持多种维度表示方式."""

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...]],
        shape: Optional[Tuple[int, ...]] = None,
        is_list: bool = False,
        feature_dim: Optional[int] = None,
    ):
        """Initialize DimensionInfo.

        Args:
            dim: 维度信息，可以是int（单一维度）或list/tuple（多个维度）
            shape: 完整的tensor shape信息（如果可用）
            is_list: 是否表示list类型的输出
            feature_dim: 显式指定的特征维度，用于覆盖自动推断
        """
        self.dim = dim
        self.shape = shape
        self.is_list = is_list
        self._feature_dim = feature_dim

    def __repr__(self):
        return (
            f"DimensionInfo(dim={self.dim}, shape={self.shape}, "
            f"is_list={self.is_list}, feature_dim={self._feature_dim})"
        )

    def get_feature_dim(self) -> int:
        """获取特征维度（最后一个维度）."""
        # 优先使用显式指定的特征维度
        if self._feature_dim is not None:
            return self._feature_dim

        if isinstance(self.dim, (list, tuple)):
            if self.is_list:
                # 如果是list类型，返回所有维度之和
                return sum(self.dim)
            else:
                # 如果是tensor，返回最后一个维度
                return self.dim[-1] if self.dim else 0
        return self.dim

    def get_total_dim(self) -> int:
        """获取总维度（用于concat等操作）."""
        if isinstance(self.dim, (list, tuple)):
            return sum(self.dim)
        return self.dim

    def to_list(self) -> List[int]:
        """转换为list形式的维度表示."""
        if isinstance(self.dim, (list, tuple)):
            return list(self.dim)
        return [self.dim]

    def with_shape(self, shape: Tuple[int, ...]) -> "DimensionInfo":
        """返回带有指定shape信息的新DimensionInfo."""
        feature_dim = shape[-1] if shape else self.get_feature_dim()
        return DimensionInfo(
            dim=self.dim, shape=shape, is_list=self.is_list, feature_dim=feature_dim
        )

    def estimate_shape(
        self, batch_size: int = None, seq_len: int = None
    ) -> Tuple[int, ...]:
        """基于已知信息估算shape.

        Args:
            batch_size: 批次大小
            seq_len: 序列长度（如果适用）

        Returns:
            估算的shape tuple
        """
        if self.shape is not None:
            return self.shape

        feature_dim = self.get_feature_dim()

        # 基本的2D形状 (batch_size, feature_dim)
        if batch_size is not None:
            if seq_len is not None:
                # 3D形状 (batch_size, seq_len, feature_dim)
                return (batch_size, seq_len, feature_dim)
            else:
                # 2D形状 (batch_size, feature_dim)
                return (batch_size, feature_dim)
        else:
            # 只返回特征维度
            return (feature_dim,)


class DimensionInferenceEngine:
    """维度推断引擎，负责管理和推断block之间的维度信息."""

    def __init__(self):
        self.block_input_dims: Dict[str, DimensionInfo] = {}
        self.block_output_dims: Dict[str, DimensionInfo] = {}
        self.block_layers: Dict[str, nn.Module] = {}
        self.logger = logging.getLogger(__name__)

    def register_input_dim(self, block_name: str, dim_info: DimensionInfo):
        """注册block的输入维度."""
        self.block_input_dims[block_name] = dim_info
        logging.debug(f"Registered input dim for {block_name}: {dim_info}")

    def register_output_dim(self, block_name: str, dim_info: DimensionInfo):
        """注册block的输出维度."""
        self.block_output_dims[block_name] = dim_info
        logging.debug(f"Registered output dim for {block_name}: {dim_info}")

    def register_layer(self, block_name: str, layer: nn.Module):
        """注册block对应的layer."""
        self.block_layers[block_name] = layer

    def get_output_dim(self, block_name: str) -> Optional[DimensionInfo]:
        """获取block的输出维度."""
        return self.block_output_dims.get(block_name)

    def infer_layer_output_dim(
        self, layer: nn.Module, input_dim: DimensionInfo
    ) -> DimensionInfo:
        """推断layer的输出维度."""
        if hasattr(layer, "output_dim") and callable(layer.output_dim):
            # 如果layer有output_dim方法，直接调用
            try:
                output_dim = layer.output_dim()
                # 估算输出shape
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

        # 使用专门的辅助函数
        try:
            return create_dimension_info_from_layer_output(layer, input_dim)
        except Exception:
            # 如果辅助函数失败，回退到原始逻辑
            pass

        # 根据layer类型推断输出维度
        layer_type = type(layer).__name__

        if layer_type == "MLP":
            if hasattr(layer, "hidden_units") and layer.hidden_units:
                output_dim = layer.hidden_units[-1]
                return DimensionInfo(output_dim, feature_dim=output_dim)
            elif hasattr(layer, "out_features"):
                output_dim = layer.out_features
                return DimensionInfo(output_dim, feature_dim=output_dim)

        elif layer_type in ["Linear", "LazyLinear"]:
            if hasattr(layer, "out_features"):
                output_dim = layer.out_features
                return DimensionInfo(output_dim, feature_dim=output_dim)

        elif layer_type == "DIN":
            # DIN模块的输出维度推断
            if hasattr(layer, "_sequence_dim") and layer._sequence_dim is not None:
                # 如果已经初始化，直接返回sequence_dim
                output_dim = layer._sequence_dim
                return DimensionInfo(output_dim, feature_dim=output_dim)
            else:
                # 未初始化时，尝试从输入维度推断
                if isinstance(input_dim, DimensionInfo):
                    # 假设输入是[sequence_features, query_features]的concat
                    # 输出维度等于sequence_dim，通常是输入维度的一半
                    total_dim = input_dim.get_feature_dim()
                    if total_dim > 0:
                        sequence_dim = total_dim // 2  # 简化假设
                        logging.info(
                            f"DIN output dimension inferred as {sequence_dim} "
                            f"(half of input {total_dim})"
                        )
                        return DimensionInfo(sequence_dim, feature_dim=sequence_dim)

                # 如果无法推断，返回输入维度
                logging.warning(
                    "Cannot infer DIN output dimension, using input dimension"
                )
                return input_dim

        elif layer_type == "DINEncoder":
            # DINEncoder的输出维度推断
            if hasattr(layer, "_sequence_dim") and layer._sequence_dim is not None:
                # 如果已经初始化，直接返回sequence_dim
                output_dim = layer._sequence_dim
                return DimensionInfo(output_dim, feature_dim=output_dim)
            elif hasattr(layer, "output_dim") and callable(layer.output_dim):
                # 使用DINEncoder的output_dim方法
                try:
                    output_dim = layer.output_dim()
                    return DimensionInfo(output_dim, feature_dim=output_dim)
                except Exception:
                    pass

            # 如果无法从layer获取，从输入推断
            if isinstance(input_dim, DimensionInfo):
                total_dim = input_dim.get_feature_dim()
                if total_dim > 0:
                    # DINEncoder的输出维度通常等于sequence_dim
                    # 如果无法明确确定，假设为输入维度的一半
                    sequence_dim = total_dim // 2
                    logging.info(
                        f"DINEncoder output dimension inferred as {sequence_dim}"
                    )
                    return DimensionInfo(sequence_dim, feature_dim=sequence_dim)

            # 如果无法推断，返回输入维度
            logging.warning(
                "Cannot infer DINEncoder output dimension, using input dimension"
            )
            return input_dim

        elif layer_type in [
            "BatchNorm1d",
            "LayerNorm",
            "Dropout",
            "ReLU",
            "GELU",
            "Tanh",
        ]:
            # 这些层不改变维度
            return input_dim

        elif layer_type == "Sequential":
            # 对于Sequential，需要递归推断
            current_dim = input_dim
            for sublayer in layer:
                current_dim = self.infer_layer_output_dim(sublayer, current_dim)
            return current_dim

        elif layer_type in ["Conv1d", "Conv2d"]:
            if hasattr(layer, "out_channels"):
                # 对于卷积层，输出通道数作为特征维度
                output_dim = layer.out_channels
                return DimensionInfo(output_dim, feature_dim=output_dim)

        # 默认情况：输出维度与输入维度相同
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
        """应用input_fn和input_slice变换."""
        current_dim = input_dim

        # 先应用input_slice
        if input_slice is not None:
            current_dim = self._apply_input_slice(current_dim, input_slice)

        # 再应用input_fn
        if input_fn is not None:
            current_dim = self._apply_input_fn(current_dim, input_fn)

        return current_dim

    def _apply_input_slice(
        self, dim_info: DimensionInfo, input_slice: str
    ) -> DimensionInfo:
        """应用input_slice变换."""
        try:
            # 解析slice表达式
            slice_expr = eval(
                f"slice{input_slice}"
                if input_slice.startswith("[") and input_slice.endswith("]")
                else input_slice
            )

            if isinstance(slice_expr, int):
                # 单个索引
                if isinstance(dim_info.dim, (list, tuple)):
                    new_dim = dim_info.dim[slice_expr]
                    return DimensionInfo(new_dim)
                else:
                    raise ValueError(
                        f"Cannot apply index {slice_expr} to scalar dimension "
                        f"{dim_info.dim}"
                    )

            elif isinstance(slice_expr, slice):
                # 切片
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
        """应用input_fn变换 - 改进版本，优先使用dummy tensor推断."""
        try:
            # 首先尝试使用dummy tensor进行精确推断
            try:
                from tzrec.layers.lambda_inference import infer_lambda_output_dim

                result = infer_lambda_output_dim(dim_info, input_fn, safe_mode=True)
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

            # 如果dummy tensor推断失败，回退到原来的模式匹配方法
            return self._apply_input_fn_pattern_matching(dim_info, input_fn)

        except Exception as e:
            logging.error(f"Failed to apply input_fn {input_fn}: {e}")
            return dim_info

    def _apply_input_fn_pattern_matching(
        self, dim_info: DimensionInfo, input_fn: str
    ) -> DimensionInfo:
        """应用input_fn变换 - 模式匹配版本（作为fallback）."""
        try:
            # 常见的input_fn模式匹配

            # lambda x: [x] - 转换为list
            if "lambda x: [x]" in input_fn.strip():
                return DimensionInfo(dim_info.to_list(), is_list=True)

            # lambda x: x.sum(dim=...) - 求和操作
            sum_pattern = (
                r"lambda\s+x:\s+x\.sum\s*\(\s*dim\s*=\s*(-?\d+)"
                r"(?:\s*,\s*keepdim\s*=\s*(True|False))?\s*\)"
            )
            match = re.search(sum_pattern, input_fn)
            if match:
                dim = int(match.group(1))
                keepdim = match.group(2) == "True" if match.group(2) else False

                if dim_info.shape is not None:
                    # 有完整shape信息，精确计算
                    new_shape = list(dim_info.shape)
                    if keepdim:
                        new_shape[dim] = 1
                    else:
                        del new_shape[dim]
                    feature_dim = new_shape[-1] if new_shape else 1
                    return DimensionInfo(feature_dim, shape=tuple(new_shape))
                else:
                    # 只有特征维度信息，基于常见模式推断
                    feature_dim = dim_info.get_feature_dim()

                    if dim == -1 or dim == 1:
                        # 通常是在序列维度或特征维度上求和
                        if dim == -1:  # 在最后一个维度求和
                            # 假设是在特征维度求和，输出为1维或保持原维度
                            new_feature_dim = 1 if keepdim else feature_dim
                        else:  # dim == 1，通常是序列维度
                            # 在序列维度求和，特征维度保持不变
                            new_feature_dim = feature_dim

                        # 估算新的shape
                        if keepdim:
                            estimated_shape = dim_info.estimate_shape()
                            new_shape = list(estimated_shape)
                            if dim < len(new_shape):
                                new_shape[dim] = 1
                            estimated_shape = tuple(new_shape)
                        else:
                            # 不保持维度，简化处理
                            estimated_shape = (new_feature_dim,)

                        return DimensionInfo(
                            new_feature_dim,
                            shape=estimated_shape,
                            feature_dim=new_feature_dim,
                        )
                    else:
                        # 其他维度的求和，保守处理
                        logging.warning(
                            f"Sum on dim={dim} with limited shape info, "
                            f"assuming feature dim unchanged"
                        )
                        return dim_info

            # lambda x: x.mean(dim=...) - 均值操作，类似于sum
            mean_pattern = (
                r"lambda\s+x:\s+x\.mean\s*\(\s*dim\s*=\s*(-?\d+)"
                r"(?:\s*,\s*keepdim\s*=\s*(True|False))?\s*\)"
            )
            match = re.search(mean_pattern, input_fn)
            if match:
                # 均值操作的维度变化与sum相同
                return self._apply_input_fn(dim_info, input_fn.replace(".mean", ".sum"))

            # lambda x: torch.cat([...], dim=-1) - 拼接操作
            if "torch.cat" in input_fn and "dim=-1" in input_fn:
                # 这种情况通常是在多个输入之间进行拼接，维度会增加
                # 但具体增加多少需要根据上下文确定，这里暂时返回原维度
                logging.info(f"Detected concatenation in input_fn: {input_fn}")
                return dim_info

            # lambda x: x.view(...) or x.reshape(...) - 重塑操作
            reshape_pattern = r"lambda\s+x:\s+x\.(view|reshape)\s*\(\s*([^)]+)\s*\)"
            match = re.search(reshape_pattern, input_fn)
            if match:
                reshape_args = match.group(2).strip()
                # 尝试解析简单的reshape参数
                if reshape_args == "-1" or reshape_args == "(-1,)":
                    # 展平操作
                    feature_dim = dim_info.get_total_dim()
                    return DimensionInfo(feature_dim, shape=(feature_dim,))
                elif reshape_args.startswith("-1,") or reshape_args.startswith("(-1,"):
                    # 部分展平，如view(-1, feature_dim)
                    try:
                        # 简单解析最后一个维度
                        last_dim_match = re.search(r",\s*(\d+)\s*\)?$", reshape_args)
                        if last_dim_match:
                            last_dim = int(last_dim_match.group(1))
                            return DimensionInfo(last_dim, feature_dim=last_dim)
                    except Exception:
                        pass

                logging.warning(
                    f"Complex reshape operation: {input_fn}, cannot infer exact shape"
                )
                return dim_info

            # lambda x: x.squeeze(...) - 压缩维度
            squeeze_pattern = r"lambda\s+x:\s+x\.squeeze\s*\(\s*(-?\d+)?\s*\)"
            match = re.search(squeeze_pattern, input_fn)
            if match:
                squeeze_dim = match.group(1)
                if squeeze_dim is not None:
                    squeeze_dim = int(squeeze_dim)
                    # 压缩指定维度
                    if dim_info.shape is not None:
                        new_shape = list(dim_info.shape)
                        if squeeze_dim < len(new_shape) and new_shape[squeeze_dim] == 1:
                            del new_shape[squeeze_dim]
                        feature_dim = (
                            new_shape[-1] if new_shape else dim_info.get_feature_dim()
                        )
                        return DimensionInfo(feature_dim, shape=tuple(new_shape))
                    else:
                        # 没有shape信息，假设特征维度不变
                        return dim_info
                else:
                    # squeeze()压缩所有size=1的维度
                    logging.warning(
                        "squeeze() without specific dim, assuming feature dim unchanged"
                    )
                    return dim_info

            # lambda x: x.unsqueeze(...) - 增加维度
            unsqueeze_pattern = r"lambda\s+x:\s+x\.unsqueeze\s*\(\s*(-?\d+)\s*\)"
            match = re.search(unsqueeze_pattern, input_fn)
            if match:
                unsqueeze_dim = int(match.group(1))
                if dim_info.shape is not None:
                    new_shape = list(dim_info.shape)
                    new_shape.insert(unsqueeze_dim, 1)
                    feature_dim = new_shape[-1]
                    return DimensionInfo(feature_dim, shape=tuple(new_shape))
                else:
                    # 没有shape信息，估算新shape
                    feature_dim = dim_info.get_feature_dim()
                    if unsqueeze_dim == 0:
                        new_shape = (1, feature_dim)
                    elif unsqueeze_dim == -1 or unsqueeze_dim == 1:
                        new_shape = (feature_dim, 1)
                    else:
                        new_shape = dim_info.estimate_shape()
                        new_shape = list(new_shape)
                        new_shape.insert(unsqueeze_dim, 1)
                        new_shape = tuple(new_shape)

                    return DimensionInfo(feature_dim, shape=new_shape)

            # lambda x: x.transpose(...) - 转置操作
            if "transpose" in input_fn:
                # 转置通常不改变特征维度，只改变维度顺序
                logging.info(
                    f"Transpose operation detected: {input_fn}, assuming "
                    f"feature dim unchanged"
                )
                return dim_info

            # 其他复杂的lambda表达式暂时不支持自动推断
            logging.warning(f"Unsupported input_fn pattern: {input_fn}")
            return dim_info

        except Exception as e:
            logging.error(f"Failed to apply input_fn {input_fn}: {e}")
            return dim_info

    def merge_input_dims(
        self, input_dims: List[DimensionInfo], merge_mode: str = "concat"
    ) -> DimensionInfo:
        """合并多个输入维度."""
        if not input_dims:
            raise ValueError("No input dimensions to merge")

        if len(input_dims) == 1:
            return input_dims[0]

        if merge_mode == "concat":
            # 拼接模式：维度相加
            total_dim = sum(dim_info.get_total_dim() for dim_info in input_dims)
            return DimensionInfo(total_dim)

        elif merge_mode == "list":
            # 列表模式：保持为列表
            dims = []
            for dim_info in input_dims:
                dims.extend(dim_info.to_list())
            return DimensionInfo(dims, is_list=True)

        elif merge_mode == "stack":
            # 堆叠模式：增加一个维度
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

    def validate_dimension_compatibility(
        self, layer: nn.Module, input_dim: DimensionInfo
    ) -> bool:
        """验证layer与输入维度的兼容性."""
        try:
            layer_type = type(layer).__name__

            if layer_type in ["Linear", "LazyLinear"] and hasattr(layer, "in_features"):
                expected_dim = layer.in_features
                actual_dim = input_dim.get_feature_dim()
                if (
                    expected_dim != -1 and expected_dim != actual_dim
                ):  # -1表示LazyLinear未初始化
                    logging.warning(
                        f"Dimension mismatch for {layer_type}: expected "
                        f"{expected_dim}, got {actual_dim}"
                    )
                    return False

            elif layer_type == "MLP" and hasattr(layer, "in_features"):
                expected_dim = layer.in_features
                actual_dim = input_dim.get_feature_dim()
                if expected_dim != actual_dim:
                    logging.warning(
                        f"Dimension mismatch for MLP: expected {expected_dim}, "
                        f"got {actual_dim}"
                    )
                    return False

            return True

        except Exception as e:
            logging.error(f"Failed to validate dimension compatibility: {e}")
            return True  # 验证失败时默认兼容

    def get_summary(self) -> Dict[str, Any]:
        """获取维度推断的摘要信息."""
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
    embedding_group, group_name: str, batch_size: int = None
) -> DimensionInfo:
    """从embedding group创建维度信息.

    Args:
        embedding_group: embedding组对象
        group_name: 组名
        batch_size: 批次大小（可选，用于估算完整shape）

    Returns:
        DimensionInfo对象，包含特征维度信息
    """
    try:
        total_dim = embedding_group.group_total_dim(group_name)

        # 估算shape信息
        if batch_size is not None:
            estimated_shape = (batch_size, total_dim)
        else:
            estimated_shape = None

        return DimensionInfo(
            dim=total_dim,
            shape=estimated_shape,
            feature_dim=total_dim,  # 明确指定特征维度
        )
    except Exception as e:
        logging.error(f"Failed to get dimension from embedding group {group_name}: {e}")
        return DimensionInfo(0, feature_dim=0)


def create_dimension_info_from_layer_output(
    layer: nn.Module, input_dim_info: DimensionInfo
) -> DimensionInfo:
    """从layer和输入维度信息创建输出维度信息.

    这是一个辅助函数，用于更准确地推断layer的输出维度.
    """
    layer_type = type(layer).__name__

    # MLP层的特殊处理
    if layer_type == "MLP":
        if hasattr(layer, "hidden_units") and layer.hidden_units:
            output_dim = layer.hidden_units[-1]
        elif hasattr(layer, "out_features"):
            output_dim = layer.out_features
        else:
            # 如果无法确定输出维度，使用输入维度
            output_dim = input_dim_info.get_feature_dim()
            logging.warning(
                f"Cannot determine MLP output dimension, using input dim: {output_dim}"
            )

        # 估算输出shape
        input_shape = input_dim_info.shape
        if input_shape is not None:
            output_shape = input_shape[:-1] + (
                output_dim,
            )  # 保持除最后一维外的所有维度
        else:
            output_shape = input_dim_info.estimate_shape()
            if output_shape:
                output_shape = output_shape[:-1] + (output_dim,)
            else:
                output_shape = None

        return DimensionInfo(dim=output_dim, shape=output_shape, feature_dim=output_dim)

    # Linear层的处理
    elif layer_type in ["Linear", "LazyLinear"]:
        if hasattr(layer, "out_features"):
            output_dim = layer.out_features

            # 估算输出shape
            input_shape = input_dim_info.shape
            if input_shape is not None:
                output_shape = input_shape[:-1] + (output_dim,)
            else:
                output_shape = input_dim_info.estimate_shape()
                if output_shape:
                    output_shape = output_shape[:-1] + (output_dim,)
                else:
                    output_shape = None

            return DimensionInfo(
                dim=output_dim, shape=output_shape, feature_dim=output_dim
            )

    # DIN层的处理
    elif layer_type == "DIN":
        if hasattr(layer, "_sequence_dim") and layer._sequence_dim is not None:
            # 已初始化的DIN，直接使用sequence_dim
            output_dim = layer._sequence_dim
        else:
            # 未初始化的DIN，从输入维度推断
            # DIN通常接收[sequence_features, query_features]的concatenation
            # 输出维度等于sequence_dim
            total_dim = input_dim_info.get_feature_dim()
            if total_dim > 0:
                # 假设sequence_dim = total_dim / 2 (简化处理)
                # 实际项目中应该从feature group配置获取更准确的维度信息
                output_dim = total_dim // 2
                logging.info(
                    f"DIN output dimension inferred as {output_dim} "
                    f"from input {total_dim}"
                )
            else:
                output_dim = input_dim_info.get_feature_dim()
                logging.warning(
                    f"Cannot infer DIN sequence dimension, using input dim: "
                    f"{output_dim}"
                )

        # 估算输出shape
        input_shape = input_dim_info.shape
        if input_shape is not None:
            output_shape = input_shape[:-1] + (output_dim,)
        else:
            output_shape = input_dim_info.estimate_shape()
            if output_shape:
                output_shape = output_shape[:-1] + (output_dim,)
            else:
                output_shape = None

        return DimensionInfo(dim=output_dim, shape=output_shape, feature_dim=output_dim)

    # DINEncoder层的处理
    elif layer_type == "DINEncoder":
        if hasattr(layer, "_sequence_dim") and layer._sequence_dim is not None:
            # 已初始化的DINEncoder，直接使用sequence_dim
            output_dim = layer._sequence_dim
        elif hasattr(layer, "output_dim") and callable(layer.output_dim):
            # 使用DINEncoder的output_dim方法
            try:
                output_dim = layer.output_dim()
            except Exception:
                output_dim = input_dim_info.get_feature_dim()
        else:
            # 未初始化的DINEncoder，使用sequence_dim（如果有的话）
            if hasattr(layer, "sequence_dim"):
                output_dim = layer.sequence_dim
            else:
                # 从输入维度推断
                total_dim = input_dim_info.get_feature_dim()
                output_dim = total_dim // 2 if total_dim > 0 else total_dim
                logging.info(f"DINEncoder output dimension inferred as {output_dim}")

        # 估算输出shape
        input_shape = input_dim_info.shape
        if input_shape is not None:
            output_shape = input_shape[:-1] + (output_dim,)
        else:
            output_shape = input_dim_info.estimate_shape()
            if output_shape:
                output_shape = output_shape[:-1] + (output_dim,)
            else:
                output_shape = None

        return DimensionInfo(dim=output_dim, shape=output_shape, feature_dim=output_dim)

    # 其他情况回退到通用方法
    engine = DimensionInferenceEngine()
    return engine.infer_layer_output_dim(layer, input_dim_info)
