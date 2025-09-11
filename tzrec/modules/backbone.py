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

import inspect
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import torch
from networkx.drawing.nx_agraph import to_agraph
from torch import nn

from tzrec.features.feature import BaseFeature
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.protos import backbone_pb2
from tzrec.protos.model_pb2 import FeatureGroupConfig
from tzrec.utils.backbone_utils import Parameter
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.dimension_inference import (
    DimensionInferenceEngine,
    DimensionInfo,
    create_dimension_info_from_embedding,
)
from tzrec.utils.lambda_inference import LambdaOutputDimInferrer
from tzrec.utils.load_class import load_torch_layer
from tzrec.utils.logging_util import logger

# Constants for auto-inferred parameters
# Input dimension related parameters
INPUT_DIM_PARAMS = ["in_features", "input_dim", "feature_dim", "mask_input_dim"]

# Sequence dimension related parameters
SEQUENCE_QUERY_PARAMS = ["sequence_dim", "query_dim"]

# All parameters that support automatic inference
AUTO_INFER_PARAMS = INPUT_DIM_PARAMS + SEQUENCE_QUERY_PARAMS


class LambdaWrapper(nn.Module):
    """Lambda expression wrapper for dimension inference and execution."""

    def __init__(self, expression: str, name: str = "lambda_wrapper") -> None:
        super().__init__()
        self.expression = expression
        self.name = name
        self._lambda_fn = None
        self._compile_function()

    def _compile_function(self) -> None:
        """Compiling Lambda Functions."""
        try:
            self._lambda_fn = eval(self.expression)
            if not callable(self._lambda_fn):
                raise ValueError(
                    f"Expression does not evaluate to callable: {self.expression}"
                )
        except Exception as e:
            logger.error(f"Failed to compile lambda function '{self.expression}': {e}")
            raise

    def forward(
        self, x: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Executing lambda expressions."""
        if self._lambda_fn is None:
            raise ValueError("Lambda function not compiled")
        return self._lambda_fn(x)

    def infer_output_dim(self, input_dim_info: DimensionInfo) -> DimensionInfo:
        """Inferring output dims using LambdaOutputDimInferrer."""
        try:
            inferrer = LambdaOutputDimInferrer()
            output_dim_info = inferrer.infer_output_dim(input_dim_info, self.expression)
            logger.debug(
                f"Lambda wrapper {self.name} inferred output dim: {output_dim_info}"
            )
            return output_dim_info
        except Exception as e:
            logger.warning(
                f"Failed to infer output dim for lambda {self.name}: {e}, using input dim"  # NOQA
            )
            return input_dim_info

    def __repr__(self) -> str:
        return f"LambdaWrapper(name={self.name}, expression='{self.expression}')"


class Package(nn.Module):
    """A sub DAG for reuse."""

    __packages = {}

    @staticmethod
    def has_backbone_block(name: str) -> bool:
        """Return True if the backbone block with the given name exists."""
        if "backbone" not in Package.__packages:
            return False
        backbone = Package.__packages["backbone"]
        return backbone.has_block(name)

    @staticmethod
    def backbone_block_outputs(
        name: str,
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """Get the outputs of a backbone block by name.

        Args:
            name (str): The name of the backbone block to retrieve outputs for.

        Returns:
            Any: The output of the specified backbone block, or None if the backbone
                package doesn't exist or the block is not found.
        """
        if "backbone" not in Package.__packages:
            return None
        backbone = Package.__packages["backbone"]
        return backbone.block_outputs(name)

    def __init__(
        self,
        config: backbone_pb2.BlockPackage,
        features: List[BaseFeature],
        embedding_group: EmbeddingGroup,
        feature_groups: List[FeatureGroupConfig],
        wide_embedding_dim: Optional[int] = None,
        wide_init_fn: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._features = features
        self._embedding_group = embedding_group
        self._feature_groups = feature_groups
        self._wide_embedding_dim = wide_embedding_dim
        self._wide_init_fn = wide_init_fn
        # build DAG using networkx DiGraph
        self.G = nx.DiGraph()
        self._name_to_blocks = {}

        self._name_to_layer = nn.ModuleDict()  # Layer corresponding to each Block name
        self._name_to_customize = {}  # Whether each Block is a custom implementation

        # Dimension inference engine
        self.dim_engine = DimensionInferenceEngine()

        self._name_to_output_dim = {}
        self._name_to_input_dim = {}

        self.reset_input_config(None)
        self._block_outputs = {}
        self._package_input = None
        self._feature_group_inputs = {}
        input_feature_groups = self._feature_group_inputs

        # ======= step 1: Register all nodes =======
        for block in config.blocks:
            if len(block.inputs) == 0:
                raise ValueError("block takes at least one input: %s" % block.name)
            self._name_to_blocks[block.name] = block
            self.G.add_node(block.name)

        # ======= step 2: Complete all DAG edges ========
        for block in config.blocks:
            name = block.name
            for input_node in block.inputs:
                input_type = input_node.WhichOneof(
                    "name"
                )  # feature_group_name / block_name
                input_name = getattr(input_node, input_type)
                if input_type == "feature_group_name":
                    # If not registered, register it as an input node.
                    # "feature_group_name" requires adding a new DAG node.
                    if input_name not in self._name_to_blocks:
                        new_block = backbone_pb2.Block()
                        new_block.name = input_name
                        input_cfg = backbone_pb2.Input()
                        input_cfg.feature_group_name = input_name
                        new_block.inputs.append(input_cfg)
                        new_block.input_layer.CopyFrom(backbone_pb2.InputLayer())
                        self._name_to_blocks[input_name] = new_block
                        self.G.add_node(input_name)
                        self.G.add_edge(input_name, name)
                elif input_type == "package_name":
                    # The package is the sub-DAG as the input of the Block
                    raise NotImplementedError
                else:
                    # block-to-block
                    if input_name in self._name_to_blocks:
                        self.G.add_edge(input_name, name)
                    else:
                        raise KeyError(
                            f"input name `{input_name}` not found in blocks/feature_groups"  # NOQA
                        )
        # ========== step 3: After topological sorting, define_layer in order ==========
        self.topo_order = nx.topological_sort(self.G)
        self.topo_order_list = list(self.topo_order)
        A = to_agraph(self.G)
        A.layout("dot")
        import hashlib
        import time

        config_info = f"{config.name}_{len(config.blocks)}_{len(self._name_to_layer)}"
        config_hash = hashlib.md5(config_info.encode()).hexdigest()[:8]
        timestamp = int(time.time())

        dag_filename = f"dag_{config.name}_{config_hash}_{timestamp}.png"
        A.draw(dag_filename)
        for block_name in self.topo_order_list:
            block = self._name_to_blocks[block_name]
            layer = block.WhichOneof("layer")
            if layer in {"input_layer", "raw_input", "embedding_layer"}:
                # Register input-related layer, needs 1 input
                if len(block.inputs) != 1:
                    raise ValueError(
                        "input layer `%s` takes only one input" % block.name
                    )
                one_input = block.inputs[0]
                name = one_input.WhichOneof("name")
                if name != "feature_group_name":
                    raise KeyError(
                        "`feature_group_name` should be set for input layer: "
                        + block.name
                    )
                group = one_input.feature_group_name

                if group in input_feature_groups:
                    # Already exists, do not register again
                    if layer == "input_layer":
                        logger.warning(
                            "input `%s` already exists in other block" % group
                        )
                    elif layer == "raw_input":
                        raise NotImplementedError
                    elif layer == "embedding_layer":
                        raise NotImplementedError
                else:
                    input_fn = EmbeddingGroup(
                        features=self._features,
                        feature_groups=self._feature_groups,
                        wide_embedding_dim=self._wide_embedding_dim,
                        wide_init_fn=self._wide_init_fn,
                    )
                    if layer == "input_layer":
                        # Use dimension inference engine
                        dim_info = create_dimension_info_from_embedding(
                            input_fn,
                            group,
                            batch_size=None,
                        )
                        self.dim_engine.register_output_dim(block.name, dim_info)
                        self._name_to_output_dim[block.name] = (
                            dim_info.get_feature_dim()
                        )

                        input_feature_groups[group] = (
                            embedding_group  # not a layer is a dim
                        )
                    elif layer == "raw_input":
                        raise NotImplementedError
                    else:  # embedding_layer
                        raise NotImplementedError
                    self._name_to_layer[block.name] = input_fn
            # If module is None, it may be a sequential module
            elif layer is not None:
                # Use the dimension inference engine to handle multiple input dimensions
                input_dim_infos = []

                for input_node in block.inputs:
                    if (len(block.inputs)) > 1:
                        logger.debug(
                            f"Processing multiple inputs for block {block.name}: {[getattr(n, n.WhichOneof('name')) for n in block.inputs]}"  # NOQA
                        )
                    input_type = input_node.WhichOneof("name")
                    input_name = getattr(input_node, input_type)
                    # Parse input_fn & input_slice
                    input_fn = getattr(input_node, "input_fn", None)
                    input_slice = getattr(input_node, "input_slice", None)

                    if input_type == "package_name":
                        # package is a sub-DAG as input to Block
                        raise NotImplementedError
                    else:  # block_name or feature_group_name
                        # Get input dimension info from dimension inference engine
                        input_dim_info = self.dim_engine.get_output_dim(input_name)

                        # If it is a recurrent or repeat layer
                        # To ensure the latest output dimensions,
                        # need to do some processing first.
                        if input_name in self._name_to_blocks:
                            input_block = self._name_to_blocks[input_name]
                            input_layer_type = input_block.WhichOneof("layer")
                            if input_layer_type in ["recurrent", "repeat"]:
                                # Get the latest output dimension
                                if input_name in self._name_to_output_dim:
                                    latest_output_dim = self._name_to_output_dim[
                                        input_name
                                    ]
                                    latest_dim_info = DimensionInfo(latest_output_dim)
                                    logger.info(
                                        f"Overriding dim_engine cache for {input_layer_type} layer {input_name}: {latest_output_dim}"  # NOQA
                                    )
                                    # Updated dimension inference engine
                                    self.dim_engine.register_output_dim(
                                        input_name, latest_dim_info
                                    )
                                    input_dim_info = latest_dim_info
                                else:
                                    logger.warning(
                                        f"{input_layer_type} layer {input_name} not found in _name_to_output_dim"  # NOQA
                                    )
                        # Apply input_fn and input_slice transformations
                        if input_fn or input_slice:
                            input_dim_info = self.dim_engine.apply_input_transforms(
                                input_dim_info, input_fn, input_slice
                            )

                        input_dim_infos.append(input_dim_info)

                # Merge dimension info of multiple inputs
                if len(input_dim_infos) == 1:
                    merged_input_dim = input_dim_infos[0]
                else:
                    # Determine the merging method based on block configuration
                    merge_mode = (
                        "list"
                        if getattr(block, "merge_inputs_into_list", False)
                        else "concat"
                    )
                    merged_input_dim = self.dim_engine.merge_input_dims(
                        input_dim_infos, merge_mode
                    )

                # Register input dimension
                self.dim_engine.register_input_dim(block.name, merged_input_dim)
                self._name_to_input_dim[block.name] = merged_input_dim.get_total_dim()

                # Add debug info
                logger.info(
                    f"Block {block.name} input dimensions: merged_input_dim={merged_input_dim}, total_dim={merged_input_dim.get_total_dim()}"  # NOQA
                )
                if merged_input_dim.is_list:
                    logger.info(
                        f"  - is_list=True, dims_list={merged_input_dim.to_list()}"
                    )
                else:
                    logger.info(
                        f"  - is_list=False, feature_dim={merged_input_dim.get_feature_dim()}"  # NOQA
                    )

                # define layer
                self.define_layers(layer, block, block.name)

                # Register the layer to the dimension inference engine
                if block.name in self._name_to_layer:
                    layer_obj = self._name_to_layer[block.name]
                    self.dim_engine.register_layer(block.name, layer_obj)

                    # Lambda module require dimension inference
                    if isinstance(layer_obj, LambdaWrapper):
                        output_dim_info = layer_obj.infer_output_dim(merged_input_dim)
                        logger.info(
                            f"Lambda layer {block.name} inferred output dim: {output_dim_info}"  # NOQA
                        )
                    else:
                        # Check if it is already a recurrent or repeat layer
                        # if so skip output dimension inference
                        if layer in {"recurrent", "repeat"}:
                            # Output dimension is already set in define_layers,
                            # no need to infer again
                            output_dim_info = self.dim_engine.get_output_dim(block.name)
                            if output_dim_info is None:
                                # If not in dimension inference engine,
                                # get from self._name_to_output_dim
                                if block.name in self._name_to_output_dim:
                                    output_dim = self._name_to_output_dim[block.name]
                                    output_dim_info = DimensionInfo(output_dim)
                                    self.dim_engine.register_output_dim(
                                        block.name, output_dim_info
                                    )
                                    logger.info(
                                        f"{layer.capitalize()} layer {block.name} output dim restored from compatibility field: {output_dim}"  # NOQA
                                    )
                                else:
                                    raise ValueError(
                                        f"{layer.capitalize()} layer {block.name} missing output dimension"  # NOQA
                                    )
                            else:
                                logger.info(
                                    f"{layer.capitalize()} layer {block.name} output dim already set: {output_dim_info}"  # NOQA
                                )
                        else:
                            # Inferred output dimensions
                            output_dim_info = self.dim_engine.infer_layer_output_dim(
                                layer_obj, merged_input_dim
                            )

                    self.dim_engine.register_output_dim(block.name, output_dim_info)
                    self._name_to_output_dim[block.name] = (
                        output_dim_info.get_feature_dim()
                    )

                    logger.info(
                        f"Block {block.name} output dimensions: output_dim_info={output_dim_info}, feature_dim={output_dim_info.get_feature_dim()}"  # NOQA
                    )
                else:
                    # Check if it is a recurrent or repeat layer, and if so,
                    # do not overwrite the set output dimension.
                    layer_type = layer
                    if layer_type in ["recurrent", "repeat"]:
                        # The output dimensions of the recurrent layer have been set
                        # in define_layers and are no need to overwrite.
                        existing_output_dim_info = self.dim_engine.get_output_dim(
                            block.name
                        )
                        existing_output_dim = self._name_to_output_dim.get(block.name)
                        logger.info(
                            f"[SKIP OVERRIDE] {layer_type.capitalize()} layer {block.name} - keeping existing output dim: engine={existing_output_dim_info}, compat={existing_output_dim}"  # NOQA
                        )
                        logger.info(
                            f"Skipping override for {layer_type} layer {block.name} - keeping existing output dimensions"  # NOQA
                        )
                    else:
                        # Use input dimensions as output dimensions
                        self.dim_engine.register_output_dim(
                            block.name, merged_input_dim
                        )
                        self._name_to_output_dim[block.name] = (
                            merged_input_dim.get_feature_dim()
                        )

                        logger.info(
                            f"Block {block.name} (no layer) output dimensions: output_dim_info={merged_input_dim}, feature_dim={merged_input_dim.get_feature_dim()}"  # NOQA
                        )
            else:  # layer is None, e.g. sequential
                if len(block.inputs) == 0:
                    input_dim_info = self.dim_engine.get_output_dim(input_name)
                    # sequential block without inputs, use input_dim_info
                    raise ValueError(
                        f"Sequential block {block.name} has no input dimensions registered"  # NOQA
                    )
                else:
                    # sequential block with inputs, use merged input dimensions
                    for input_node in block.inputs:
                        input_type = input_node.WhichOneof("name")
                        input_name = getattr(input_node, input_type)
                        # Parsing input_fn & input_slice does
                        # not support input_fn & input_slice in sequential
                        input_fn = getattr(input_node, "input_fn", None)
                        input_slice = getattr(input_node, "input_slice", None)

                        if input_type == "package_name":
                            # The package is the sub-DAG as the input of the Block
                            # Nested packages in sequential modules
                            input_dim_info = self.dim_engine.get_output_dim(input_name)
                            raise NotImplementedError
                        else:  # block_name or feature_group_name
                            # Get input dimension info from dimension inference engine
                            input_dim_info = self.dim_engine.get_output_dim(input_name)
                # Dimension inference for sequential layers
                prev_output_dim_info = input_dim_info
                prev_output_dim = input_dim_info.get_feature_dim()
                last_output_dim_info = None
                last_output_dim = None
                for i, layer_cnf in enumerate(block.layers):
                    layer = layer_cnf.WhichOneof("layer")
                    name_i = "%s_l%d" % (block.name, i)  # e.g. block1_l0
                    # Register input dimension
                    self.dim_engine.register_input_dim(name_i, prev_output_dim_info)
                    self._name_to_input_dim[name_i] = prev_output_dim
                    # Define layer
                    self.define_layers(layer, layer_cnf, name_i)
                    # Register layer to dimension inference engine
                    if name_i in self._name_to_layer:
                        layer_obj = self._name_to_layer[name_i]
                        self.dim_engine.register_layer(name_i, layer_obj)
                        # Infer output dimension
                        if isinstance(layer_obj, LambdaWrapper):
                            output_dim_info = layer_obj.infer_output_dim(
                                prev_output_dim_info
                            )
                        else:
                            output_dim_info = self.dim_engine.infer_layer_output_dim(
                                layer_obj, prev_output_dim_info
                            )
                        self.dim_engine.register_output_dim(name_i, output_dim_info)
                        self._name_to_output_dim[name_i] = (
                            output_dim_info.get_feature_dim()
                        )
                        # Update prev to current output
                        prev_output_dim_info = output_dim_info
                        prev_output_dim = output_dim_info.get_feature_dim()
                        last_output_dim_info = output_dim_info
                        last_output_dim = output_dim_info.get_feature_dim()
                    else:
                        raise ValueError(
                            f"Sequential layer {name_i} not found in _name_to_layer"
                        )
                # The block output dimension is the last layer output
                if last_output_dim_info is not None:
                    self.dim_engine.register_output_dim(
                        block.name, last_output_dim_info
                    )
                    self._name_to_output_dim[block.name] = last_output_dim
                    logger.info(
                        f"Sequential block {block.name} output dim set to {last_output_dim}"  # NOQA
                    )
                else:
                    raise ValueError(
                        f"Cannot determine output dimension for sequential block {block.name}"  # NOQA
                    )

        # ======= Post-processing, output node inference =======
        input_feature_groups = self._feature_group_inputs
        num_groups = len(input_feature_groups)  # Number of input_feature_groups
        # Subtract the number of input feature groups,
        # blocks contain feature_groups e.g. feature group user
        num_blocks = len(self._name_to_blocks) - num_groups
        assert num_blocks > 0, "there must be at least one block in backbone"
        # num_pkg_input = 0
        # Processing multiple pkgs is not yet supported
        # Optional: Check package inputs

        # If concat_blocks is not configured,
        # automatically concatenate all leaf nodes of the DAG and output
        if len(config.concat_blocks) == 0 and len(config.output_blocks) == 0:
            # Get all leaf nodes
            leaf = [node for node in self.G.nodes() if self.G.out_degree(node) == 0]
            logger.warning(
                (
                    f"{config.name} has no `concat_blocks` or `output_blocks`, "
                    f"try to concat all leaf blocks: {','.join(leaf)}"
                )
            )
            self._config.concat_blocks.extend(leaf)

        Package.__packages[self._config.name] = self

        # Output dimension inference summary
        dim_summary = self.dim_engine.get_summary()
        logger.info(f"{config.name} dimension inference summary: {dim_summary}")

        # Output detailed dimension info for all blocks
        logger.info("=== Final dimension summary ===")
        for block_name in self.topo_order_list:
            if block_name in self._name_to_input_dim:
                input_dim = self._name_to_input_dim[block_name]
                output_dim = self._name_to_output_dim.get(block_name, "N/A")
                dim_engine_output = self.dim_engine.get_output_dim(block_name)
                logger.info(
                    f"Block {block_name}: input_dim={input_dim}, output_dim={output_dim}, dim_engine={dim_engine_output}"  # NOQA
                )

        logger.info(
            "%s layers: %s" % (config.name, ",".join(self._name_to_layer.keys()))
        )

    def get_output_block_names(self) -> List[str]:
        """Returns the final output block name list (prefer concat_blocks, otherwise output_blocks)."""  # NOQA
        blocks = list(getattr(self._config, "concat_blocks", []))
        if not blocks:
            blocks = list(getattr(self._config, "output_blocks", []))
        return blocks

    def get_dimension_summary(self) -> Dict[str, Any]:
        """Get detailed summary information of dimension inference."""
        summary = self.dim_engine.get_summary()
        summary.update(
            {
                "config_name": self._config.name,
                "total_layers": len(self._name_to_layer),
                "output_blocks": list(getattr(self._config, "output_blocks", [])),
                "concat_blocks": list(getattr(self._config, "concat_blocks", [])),
                "final_output_dims": self.output_block_dims(),
                "total_output_dim": self.total_output_dim(),
            }
        )
        return summary

    def output_block_dims(self) -> List[int]:
        """Return a list of dimensions of the final output blocks, e.g. [160, 96]."""
        blocks = self.get_output_block_names()
        dims = []
        for block in blocks:
            dim_info = self.dim_engine.get_output_dim(block)
            logger.info(f"Output block `{block}` dimension info: {dim_info}")
            if dim_info is not None:
                dims.append(dim_info.get_feature_dim())
            elif block in self._name_to_output_dim:
                dims.append(self._name_to_output_dim[block])
            else:
                raise ValueError(f"block `{block}` not in output dims")
        return dims

    def total_output_dim(self) -> int:
        """Return the total dimension of the final output after concatenation."""
        return sum(self.output_block_dims())

    def define_layers(
        self, layer: str, layer_cnf: backbone_pb2.Block, name: str
    ) -> None:
        """Define layers.

        Args:
            layer (str): the type of layer, e.g., 'module', 'recurrent', 'repeat'.
            layer_cnf (backbone_pb2.Block): the configuration of the layer.
              class_name: "MLP" mlp {
                hidden_units: 512
                hidden_units: 256
                hidden_units: 128
                activation: "nn.ReLU"
                }
            name (str): the name of the layer. e.g., 'user_mlp'.
        """
        if layer == "module":
            layer_cls, customize = self.load_torch_layer(
                layer_cnf.module, name, self._name_to_input_dim.get(name, None)
            )
            self._name_to_layer[name] = layer_cls
            self._name_to_customize[name] = customize
        elif layer == "recurrent":
            torch_layer = layer_cnf.recurrent.module
            # Get the input dimension info of the parent layer,
            # used for child layer dimension inference
            parent_input_dim_info = self.dim_engine.block_input_dims.get(name)
            parent_input_dim = self._name_to_input_dim.get(name, None)

            # Check if there is a fixed_input_index configuration
            fixed_input_index = getattr(layer_cnf.recurrent, "fixed_input_index", None)

            # If fixed_input_index exists and parent_input_dim_info is a list,
            # special handling is needed
            child_input_dim_info = parent_input_dim_info
            child_input_dim = parent_input_dim

            if fixed_input_index is not None and parent_input_dim_info is not None:
                if parent_input_dim_info.is_list:
                    # Take the dimension specified by fixed_input_index from the list
                    dims_list = parent_input_dim_info.to_list()
                    if fixed_input_index < len(dims_list):
                        fixed_dim = dims_list[fixed_input_index]
                        child_input_dim_info = DimensionInfo(fixed_dim)
                        child_input_dim = fixed_dim
                        logger.info(
                            f"Recurrent layer {name} using fixed_input_index={fixed_input_index}, child input_dim={fixed_dim}"  # NOQA
                        )
                    else:
                        logger.warning(
                            f"fixed_input_index={fixed_input_index} out of range for input dims: {dims_list}"  # NOQA
                        )

            # record the output dimension of the last child layer
            last_output_dim_info = None
            last_output_dim = None

            for i in range(layer_cnf.recurrent.num_steps):
                name_i = "%s_%d" % (name, i)

                # Register input dimension info for each child layer
                if child_input_dim_info is not None:
                    self.dim_engine.register_input_dim(name_i, child_input_dim_info)
                if child_input_dim is not None:
                    self._name_to_input_dim[name_i] = child_input_dim

                # Load the child layer, passing the correct input_dim parameter
                layer_obj, customize = self.load_torch_layer(
                    torch_layer, name_i, child_input_dim
                )
                self._name_to_layer[name_i] = layer_obj
                self._name_to_customize[name_i] = customize

                # Register the child layer with the dimension inference engine
                self.dim_engine.register_layer(name_i, layer_obj)

                # Infer the output dimension of the child layer
                if child_input_dim_info is not None:
                    if isinstance(layer_obj, LambdaWrapper):
                        output_dim_info = layer_obj.infer_output_dim(
                            child_input_dim_info
                        )
                    else:
                        output_dim_info = self.dim_engine.infer_layer_output_dim(
                            layer_obj, child_input_dim_info
                        )

                    self.dim_engine.register_output_dim(name_i, output_dim_info)
                    self._name_to_output_dim[name_i] = output_dim_info.get_feature_dim()

                    # Record the output dimension of the last child layer
                    last_output_dim_info = output_dim_info
                    last_output_dim = output_dim_info.get_feature_dim()
                else:
                    raise ValueError(
                        f"Cannot determine output dimension for layer {name_i}"
                    )

            # Set the output dimension of the parent layer (recurrent layer) to
            # the output dimension of the last child layer
            if last_output_dim_info is not None:
                # Updates the dimension inference engine and self._name_to_output_dim
                self.dim_engine.register_output_dim(name, last_output_dim_info)
                self._name_to_output_dim[name] = last_output_dim
                logger.info(
                    f"Recurrent layer {name} output dim set to {last_output_dim} (from last child layer)"  # NOQA
                )
                logger.info(f"  - last_output_dim_info: {last_output_dim_info}")
                logger.info(
                    f"  - Updated _name_to_output_dim[{name}]: {self._name_to_output_dim[name]}"  # NOQA
                )

                # Verify that the update was successful
                updated_dim_info = self.dim_engine.get_output_dim(name)
                logger.info(
                    f"[VERIFY] Updated dim_engine output for {name}: {updated_dim_info}"
                )
            else:
                raise ValueError(f"Cannot determine input dimension for layer {name}")
        elif layer == "repeat":
            torch_layer = layer_cnf.repeat.module
            # Get the input dimension information of the parent layer
            # for dimension inference of the child layer
            parent_input_dim_info = self.dim_engine.block_input_dims.get(name)
            parent_input_dim = self._name_to_input_dim.get(name, None)

            # Used to record the output dimension of the last child layer
            last_output_dim_info = None
            last_output_dim = None

            for i in range(layer_cnf.repeat.num_repeat):
                name_i = "%s_%d" % (name, i)

                # Register input dimension info for each child layer
                if parent_input_dim_info is not None:
                    self.dim_engine.register_input_dim(name_i, parent_input_dim_info)
                if parent_input_dim is not None:
                    self._name_to_input_dim[name_i] = parent_input_dim

                # Load the child layer,
                # passing the correct input_dim parameter
                layer_obj, customize = self.load_torch_layer(
                    torch_layer, name_i, parent_input_dim
                )
                self._name_to_layer[name_i] = layer_obj
                self._name_to_customize[name_i] = customize

                # Register child layer to dimension inference engine
                self.dim_engine.register_layer(name_i, layer_obj)

                # Infer the output dimension of the child layer
                if parent_input_dim_info is not None:
                    if isinstance(layer_obj, LambdaWrapper):
                        output_dim_info = layer_obj.infer_output_dim(
                            parent_input_dim_info
                        )
                    else:
                        output_dim_info = self.dim_engine.infer_layer_output_dim(
                            layer_obj, parent_input_dim_info
                        )

                    self.dim_engine.register_output_dim(name_i, output_dim_info)
                    self._name_to_output_dim[name_i] = output_dim_info.get_feature_dim()

                    # Record the output dimension of the last child layer
                    last_output_dim_info = output_dim_info
                    last_output_dim = output_dim_info.get_feature_dim()
                else:
                    raise ValueError(
                        f"Cannot determine output dimension for layer {name_i}"
                    )

            # Calculate the output dimension of the parent layer (repeat layer),
            # taking into account the output_concat_axis configuration
            if last_output_dim_info is not None:
                final_output_dim_info = last_output_dim_info
                final_output_dim = last_output_dim

                # Check if output_concat_axis is configured
                #
                # e.g., repeat maskblock 2 times and concatenate in
                # the last dimension (output_concat_axis: -1).
                # Equivalent to: [maskblock1, maskblock2] in the last dimension cat
                if (
                    hasattr(layer_cnf.repeat, "output_concat_axis")
                    and layer_cnf.repeat.output_concat_axis is not None
                ):
                    axis = layer_cnf.repeat.output_concat_axis
                    num_repeat = layer_cnf.repeat.num_repeat

                    # IF in the last dimension splicing (axis=-1),
                    # you need to multiply the dimension by the number of repeats
                    if axis == -1:
                        # The output dimension of a single child layer
                        # multiplied by repeat times
                        if isinstance(last_output_dim, int):
                            final_output_dim = last_output_dim * num_repeat
                        final_output_dim_info = DimensionInfo(final_output_dim)
                        logger.info(
                            f"Repeat layer {name} with output_concat_axis={axis}: "
                            f"single_output_dim={last_output_dim} * num_repeat={num_repeat} = {final_output_dim}"  # NOQA
                        )
                    else:
                        # For the splicing of other axes, remain unchanged for now
                        # and require more complex dimension inference logic.
                        logger.warning(
                            f"Repeat layer {name} with output_concat_axis={axis}: "
                            f"non-last axis concatenation not fully supported, using single layer output dim={last_output_dim}"  # NOQA
                        )
                else:
                    # If output_concat_axis is not configured, return as list format
                    num_repeat = layer_cnf.repeat.num_repeat
                    # Create dimension information in list format,
                    # containing num_repeat identical sub-layer output dimensions
                    list_dims = [last_output_dim] * num_repeat
                    final_output_dim_info = DimensionInfo(list_dims, is_list=True)

                    # final_output_dim, by default uses the total dimension of the list
                    # In actual use, the correct dimension information should
                    # be obtained through the dimension inference engine
                    final_output_dim = sum(list_dims)

                    logger.info(
                        f"Repeat layer {name} without output_concat_axis: returns list of {num_repeat} outputs, "  # NOQA
                        f"each with dim={last_output_dim}, list_dims={list_dims}"
                    )

                self.dim_engine.register_output_dim(name, final_output_dim_info)
                self._name_to_output_dim[name] = final_output_dim
                logger.info(
                    f"Repeat layer {name} final output dim set to {final_output_dim}"
                )
            else:
                raise ValueError(f"Cannot determine output dimension for layer {name}")
        elif layer == "lambda":
            expression = getattr(layer_cnf, "lambda").expression
            lambda_layer = LambdaWrapper(expression, name=name)
            self._name_to_layer[name] = lambda_layer
            self._name_to_customize[name] = True

    def load_torch_layer(self, layer_conf, name, input_dim=None):
        """Dynamically load and initialize a torch layer based on configuration.

        Args:
            layer_conf: Layer configuration containing class name and parameters.
            name (str): Name of the layer to be created.
            input_dim (int, optional): Input dimension for the layer.

        Returns:
            tuple: A tuple containing (layer_instance, customize_flag) where
                layer_instance is the initialized layer object and customize_flag
                indicates if it's a custom implementation.

        Raises:
            ValueError: If the layer class name is invalid or layer creation fails.
        """
        # customize indicates whether it is a custom implementation
        layer_cls, customize = load_torch_layer(layer_conf.class_name)
        if layer_cls is None:
            raise ValueError("Invalid torch layer class name: " + layer_conf.class_name)
        param_type = layer_conf.WhichOneof("params")
        # st_params is a parameter configured
        # in the google.protobuf.Struct object format;
        # can also pass parameters to the loaded Layer object
        # in a custom protobuf message format.
        if customize:
            if param_type is None:  # No additional parameters
                # Get the constructor signature
                sig = inspect.signature(layer_cls.__init__)
                kwargs = {}
            elif param_type == "st_params":
                params = Parameter(layer_conf.st_params, True)
                sig = inspect.signature(layer_cls.__init__)
                kwargs = config_to_kwargs(params)
            # If param_type points to some other field in oneof,
            # the code dynamically gets the value of that field via getattr,
            # assuming it is a Protocol Buffer message (is_struct=False).
            else:
                pb_params = getattr(layer_conf, param_type)
                params = Parameter(pb_params, False)
                sig = inspect.signature(layer_cls.__init__)
                kwargs = config_to_kwargs(params)

            # Check if you need to automatically infer the input dimension parameters
            input_dim_params_in_sig = [
                param for param in INPUT_DIM_PARAMS if param in sig.parameters
            ]
            if input_dim_params_in_sig:
                input_dim_params_missing = [
                    param for param in INPUT_DIM_PARAMS if param not in kwargs
                ]
                if input_dim_params_missing:
                    # Get input dimensions from the dimension inference engine
                    input_dim_info = self.dim_engine.block_input_dims.get(name)
                    if input_dim_info is not None:
                        # For modules that receive multiple independent tensors,
                        # check whether sum operation should be avoided
                        should_use_single_dim = False

                        # Check method: whether the forward method
                        # accepts multiple tensor parameters
                        if hasattr(layer_cls, "forward"):
                            try:
                                forward_sig = inspect.signature(layer_cls.forward)
                                forward_params = [
                                    p
                                    for p in forward_sig.parameters.keys()
                                    if p != "self"
                                ]
                                # If forward method has 2 or more non-self parameters,
                                # it may be multiple tensor inputs
                                if len(forward_params) >= 2:
                                    should_use_single_dim = True
                                    logger.info(
                                        f"Detected multi-tensor input module {layer_cls.__name__} with {len(forward_params)} forward parameters"  # NOQA
                                    )
                            except Exception as err:
                                raise ValueError(
                                    f"Failed to inspect forward method of {layer_cls.__name__} for dimension inference"  # NOQA
                                ) from err
                        if (
                            should_use_single_dim
                            and input_dim_info.is_list
                            and isinstance(input_dim_info.dim, (list, tuple))
                        ):
                            # For forward modules that require multiple tensor inputs,
                            # use the dimensions in list format.
                            for idx, param_name in enumerate(input_dim_params_in_sig):
                                kwargs[param_name] = input_dim_info.dim[idx]
                                logger.info(
                                    f"Layer {name} ({layer_cls.__name__}) auto-inferred {param_name}={input_dim_info.dim[idx]} from input dim list"  # NOQA
                                )
                        else:
                            # For other modules, use the total dimension
                            feature_dim = input_dim_info.get_feature_dim()
                            for param_name in input_dim_params_in_sig:
                                kwargs[param_name] = feature_dim
                                logger.info(
                                    f"Layer {name} ({layer_cls.__name__}) auto-inferred {param_name}={feature_dim} from dim_engine"  # NOQA
                                )
                    else:
                        logger.error(
                            f"Layer {name} ({layer_cls.__name__}) dimension inference failed - no input_dim available"  # NOQA
                        )
                        logger.error(
                            f"  - input_dim_info from dim_engine: {input_dim_info}"
                        )
                        logger.error(f"  - input_dim: {input_dim}")
                        logger.error(
                            f"  - block_input_dims keys: {list(self.dim_engine.block_input_dims.keys())}"  # NOQA
                        )
                        if name in self._name_to_input_dim:
                            logger.error(
                                f"  - _name_to_input_dim[{name}]: {self._name_to_input_dim[name]}"  # NOQA
                            )
                        raise ValueError(
                            f"Cannot automatically infer {', '.join(missing_params)} for {layer_cls.__name__} {name}. "  # NOQA
                            "Please ensure correct input feature groups are configured or manually specify these parameters."  # NOQA
                        )

            # sequence_dim and query_dim are automatically inferred
            sequence_dim_missing = (
                SEQUENCE_QUERY_PARAMS[0] in sig.parameters
                and SEQUENCE_QUERY_PARAMS[0] not in kwargs
            )
            query_dim_missing = (
                SEQUENCE_QUERY_PARAMS[1] in sig.parameters
                and SEQUENCE_QUERY_PARAMS[1] not in kwargs
            )

            if sequence_dim_missing or query_dim_missing:
                # Get the input information of the current block
                block_config = self._name_to_blocks[name]
                input_dims = self._infer_sequence_query_dimensions(block_config, name)

                if input_dims:
                    sequence_dim, query_dim = input_dims
                    if sequence_dim_missing:
                        kwargs[SEQUENCE_QUERY_PARAMS[0]] = sequence_dim
                    if query_dim_missing:
                        kwargs[SEQUENCE_QUERY_PARAMS[1]] = query_dim
                    logger.info(
                        f"Auto-inferred dimensions for {layer_cls.__name__} {name}: "  # NOQA
                        f"{SEQUENCE_QUERY_PARAMS[0]}={sequence_dim if sequence_dim_missing else 'provided'}, "  # NOQA
                        f"{SEQUENCE_QUERY_PARAMS[1]}={query_dim if query_dim_missing else 'provided'}"  # NOQA
                    )
                else:
                    missing_params = []
                    if sequence_dim_missing:
                        missing_params.append(SEQUENCE_QUERY_PARAMS[0])
                    if query_dim_missing:
                        missing_params.append(SEQUENCE_QUERY_PARAMS[1])
                    raise ValueError(
                        f"Cannot automatically infer {', '.join(missing_params)} for {layer_cls.__name__} {name}. "  # NOQA
                        "Please ensure correct input feature groups are configured or manually specify these parameters."  # NOQA
                    )
            layer = layer_cls(**kwargs)
            return layer, customize
        elif param_type is None:  # internal torch layer
            layer = layer_cls()
            return layer, customize
        else:  # st_params parameter
            assert param_type == "st_params", (
                "internal torch layer only support st_params as parameters"
            )
            try:
                kwargs = convert_to_dict(layer_conf.st_params)
                logger.info(
                    "call %s layer with params %r" % (layer_conf.class_name, kwargs)
                )
                layer = layer_cls(**kwargs)
            except TypeError as e:
                logger.warning(e)
                args = map(format_value, layer_conf.st_params.values())
                logger.info(
                    "try to call %s layer with params %r"
                    % (layer_conf.class_name, args)
                )
                layer = layer_cls(*args, name=name)
            return layer, customize

    def reset_input_config(self, config):
        """Reset the input configuration for this package.

        Args:
            config: The new input configuration to set.
        """
        self.input_config = config

    def _infer_sequence_query_dimensions(self, block_config, block_name):
        """Inference module sequence_dim and query_dim.

        e.g. infer DINEncoder's sequence_dim and query_dim
        Args:
            block_config: Block configuration
            block_name: Block name

        Returns:
            tuple: (sequence_dim, query_dim) or None if inference fails
        """
        sequence_dim = None
        query_dim = None

        # Analyze the input and infer the dimension based on feature_group_name
        for input_node in block_config.inputs:
            input_type = input_node.WhichOneof("name")
            input_name = getattr(input_node, input_type)

            if input_type == "feature_group_name":
                # get the sequence and query dimensions from the embedding group
                dims = self._try_get_sequence_query_dims_from_group(input_name)
                if dims:
                    sequence_dim, query_dim = dims
                    logger.info(
                        f"Auto-inferred dimensions from {input_name}: "
                        f"sequence_dim={sequence_dim}, query_dim={query_dim}"
                    )
                    return sequence_dim, query_dim
            else:
                raise NotImplementedError

        # Check the inference results
        if sequence_dim is not None and query_dim is not None:
            return sequence_dim, query_dim
        else:
            logger.warning(
                f"Could not infer sequence/query dimensions for {block_name}: "
                f"sequence_dim={sequence_dim}, query_dim={query_dim}"
            )
            return None

    def _try_get_sequence_query_dims_from_group(self, group_name):
        """Get the sequence and query dimensions from the embedding group.

        Args:
            group_name: embedding group name

        Returns:
            tuple: (sequence_dim, query_dim) or None if failed
        """
        # Check if group exists
        if group_name not in self._name_to_layer:
            logger.debug(f"Group {group_name} not found in _name_to_layer")
            return None

        layer = self._name_to_layer[group_name]

        # Check if there is a group_total_dim method
        if not hasattr(layer, "group_total_dim"):
            logger.debug(f"Group {group_name} does not have group_total_dim method")
            return None

        # Trying to get the dimensions of .sequence and .query subgroups
        sequence_group_name = f"{group_name}.sequence"
        query_group_name = f"{group_name}.query"

        try:
            sequence_dim = layer.group_total_dim(sequence_group_name)
            query_dim = layer.group_total_dim(query_group_name)
            return sequence_dim, query_dim
        except (KeyError, AttributeError, ValueError) as e:
            logger.debug(
                f"Could not get .sequence/.query dimensions for {group_name}: {type(e).__name__}: {e}"  # NOQA
            )
            return None
        except Exception as e:
            logger.warning(
                f"Unexpected error getting dimensions for {group_name}: {type(e).__name__}: {e}"  # NOQA
            )
            return None

    def set_package_input(self, pkg_input):
        """Set the package input for this package.

        Args:
            pkg_input: The input data to be used by this package.
        """
        self._package_input = pkg_input

    def has_block(self, name) -> bool:
        """Check if a block with the given name exists in this package.

        Args:
            name (str): The name of the block to check for.

        Returns:
            bool: True if the block exists, False otherwise.
        """
        return name in self._name_to_blocks

    def block_outputs(self, name):
        """Get the output of a specific block by name.

        Args:
            name (str): The name of the block to retrieve outputs for.

        Returns:
            Any: The output of the specified block, or None if not found.
        """
        return self._block_outputs.get(name, None)

    def block_input(self, config, block_outputs, **kwargs):
        """Process and merge inputs for a block based on its configuration.

        Args:
            config: Block configuration containing input specifications.
            block_outputs (dict): Dictionary of outputs from previously executed blocks.
            **kwargs: Additional keyword arguments passed to downstream components.

        Returns:
            torch.Tensor or list: Processed and merged input data ready for the block.
        """
        inputs = []
        # Traverse each input node configured by config.inputs
        for input_node in config.inputs:
            input_type = input_node.WhichOneof("name")
            input_name = getattr(input_node, input_type)

            if input_type == "use_package_input":
                input_feature = self._package_input
                input_name = "package_input"

            elif input_type == "package_name":
                if input_name not in Package.__packages:
                    raise KeyError(f"package name `{input_name}` does not exist")
                package = Package.__packages[input_name]
                if input_node.HasField("reset_input"):
                    package.reset_input_config(input_node.reset_input)
                if input_node.HasField("package_input"):
                    pkg_input_name = input_node.package_input
                    if pkg_input_name in block_outputs:
                        pkg_input = block_outputs[pkg_input_name]
                    else:
                        if pkg_input_name not in Package.__packages:
                            raise KeyError(
                                f"package name `{pkg_input_name}` does not exist"
                            )
                        inner_package = Package.__packages[pkg_input_name]
                        pkg_input = inner_package()
                    if input_node.HasField("package_input_fn"):
                        fn = eval(input_node.package_input_fn)
                        pkg_input = fn(pkg_input)
                    package.set_package_input(pkg_input)
                input_feature = package(**kwargs)

            elif input_name in block_outputs:
                input_feature = block_outputs[input_name]

            else:
                input_feature = Package.backbone_block_outputs(input_name)

            if input_feature is None:
                raise KeyError(f"input name `{input_name}` does not exist")

            if getattr(input_node, "ignore_input", False):
                continue

            # Get an element of the input tuple/list as input through slice syntax
            if input_node.HasField("input_slice"):
                fn = eval("lambda x: x" + input_node.input_slice.strip())
                input_feature = fn(input_feature)

            if input_node.HasField("input_fn"):
                # Specify a lambda function to perform transformation on the input.
                # e.g.,input_fn: 'lambda x: [x]'
                fn = eval(input_node.input_fn)
                input_feature = fn(input_feature)
                # Need to recalculate input_dim
            inputs.append(input_feature)

        # merge inputs
        if getattr(config, "merge_inputs_into_list", False):
            output = inputs
        else:
            try:
                # merge_inputs need self define，e.g. torch.cat
                # Assuming config.input_concat_axis is defined, usually 1
                output = merge_inputs(
                    inputs,
                    axis=getattr(config, "input_concat_axis", 1),
                    msg=config.name,
                )
            except ValueError as e:
                msg = getattr(e, "message", str(e))
                logger.error(f"merge inputs of block {config.name} failed: {msg}")
                raise e
        # To perform additional transformations on the merged multi-channel
        # input results, you need to configure it in the format of a lambda function.
        if config.HasField("extra_input_fn"):
            fn = eval(config.extra_input_fn)
            output = fn(output)

        return output

    def forward(self, batch=None, **kwargs):
        """Execute forward pass through the package DAG.

        Args:
            batch (Any, optional): Input batch data. Defaults to None.
            **kwargs: Additional keyword arguments passed to layers.

        Returns:
            torch.Tensor or List[torch.Tensor]: Output tensor(s) from the package.

        Raises:
            ValueError: If required output blocks are not found.
            KeyError: If input names are invalid or not found.
        """
        block_outputs = {}
        self._block_outputs = block_outputs  # reset
        blocks = self.topo_order_list
        logger.info(self._config.name + " topological order: " + ",".join(blocks))

        for block in blocks:  # Traverse blocks
            if block not in self._name_to_blocks:
                # package block
                assert block in Package.__packages, "invalid block: " + block
                continue
            config = self._name_to_blocks[block]
            # Case 1: sequential layers
            if hasattr(config, "layers") and config.layers:
                logger.info("call sequential %d layers" % len(config.layers))
                output = self.block_input(config, block_outputs, **kwargs)
                for i, layer in enumerate(config.layers):
                    name_i = "%s_l%d" % (block, i)
                    output = self.call_layer(output, layer, name_i, **kwargs)
                block_outputs[block] = output
                continue

            # Case 2: single layer  just one of layer
            layer_type = config.WhichOneof("layer")
            if layer_type is None:  # identity layer
                output = self.block_input(config, block_outputs, **kwargs)
                block_outputs[block] = output
            elif layer_type == "raw_input":
                block_outputs[block] = self._name_to_layer[block]
            elif layer_type == "input_layer":
                if (
                    block in self._name_to_layer
                    and self._name_to_layer[block] is not None
                ):
                    input_fn = self._name_to_layer[block]  # embedding group
                else:
                    input_fn = self._embedding_group
                # no block input itself
                input_config = config.input_layer
                if self.input_config is not None:
                    input_config = self.input_config
                    if hasattr(input_fn, "reset"):
                        input_fn.reset(input_config)
                if batch is not None:
                    embedding_outputs = input_fn(batch)
                    if (
                        isinstance(embedding_outputs, dict)
                        and block in embedding_outputs
                    ):
                        block_outputs[block] = embedding_outputs[block]
                    else:
                        # If the returned value is not a dictionary or does not
                        # have a corresponding key, use the entire output.
                        block_outputs[block] = embedding_outputs
                    if isinstance(block_outputs[block], torch.Tensor):
                        logger.info(f"block_outputs[{block}]shape: {block_outputs[block].shape}")
                    else:
                        logger.info(f"block_outputs[{block}] type: {type(block_outputs[block])}")
                else:
                    embedding_outputs = input_fn(input_config)
                    if (
                        isinstance(embedding_outputs, dict)
                        and block in embedding_outputs
                    ):
                        block_outputs[block] = embedding_outputs[block]
                    else:
                        block_outputs[block] = embedding_outputs
            elif layer_type == "embedding_layer":
                input_fn = self._name_to_layer[block]
                feature_group = config.inputs[0].feature_group_name
                inputs, _, weights = self._feature_group_inputs[feature_group]
                block_outputs[block] = input_fn([inputs, weights])
            else:
                # Custom module, e.g. mlp
                inputs = self.block_input(config, block_outputs, **kwargs)
                output = self.call_layer(inputs, config, block, **kwargs)
                block_outputs[block] = output

        # Collect outputs
        outputs = []
        for output in getattr(self._config, "output_blocks", []):
            if output in block_outputs:
                outputs.append(block_outputs[output])
            else:
                raise ValueError("No output `%s` of backbone to be concat" % output)
        if outputs:
            return outputs

        for output in getattr(self._config, "concat_blocks", []):
            if output in block_outputs:
                outputs.append(block_outputs[output])
            else:
                raise ValueError("No output `%s` of backbone to be concat" % output)

        try:
            logger.info(f"Number of outputs to merge: {len(outputs)}")
            # Log each output's shape
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    logger.info(f"Output {i} shape: {out.shape}")
                elif isinstance(out, (list, tuple)):
                    logger.info(f"Output {i} is a list/tuple with {len(out)} elements.")
                else:
                    logger.info(f"Output {i} is of type {type(out)}")
            # merge_inputs
            output = merge_inputs(outputs, msg="backbone")
        except Exception as e:
            logger.error("merge backbone's output failed: %s", str(e))
            raise e
        return output

    def _determine_input_format(self, layer_obj, inputs):
        """Determine the input format required by the module.

        Args:
            layer_obj: The layer object to call
            inputs: Input data (may be a tensor dict or a single tensor)

        Returns:
            Input suitable for this layer
        """
        try:
            # Check the module's forward method signature
            if hasattr(layer_obj, "forward"):
                sig = inspect.signature(layer_obj.forward)
                params = list(sig.parameters.keys())
                if "self" in params:
                    params.remove("self")

                # If the forward method has multiple parameters,
                # it may require a dictionary input
                if len(params) > 1:
                    logger.debug(
                        f"Layer {layer_obj.__class__.__name__} has multiple forward parameters: {params}"  # NOQA
                    )
                    # Check if a specific parameter name implies
                    # that a dictionary input is required
                    dict_indicators = [
                        "grouped_features",
                        "feature_dict",
                        "inputs_dict",
                        "batch",
                    ]
                    if any(indicator in params for indicator in dict_indicators):
                        logger.info(
                            f"Layer {layer_obj.__class__.__name__} likely needs dict input"  # NOQA
                        )
                        return inputs  # Return to original dictionary format

                # Check whether it is a sequence-related module
                class_name = layer_obj.__class__.__name__
                sequence_modules = [
                    "DINEncoder",
                    "SimpleAttention",
                    "PoolingEncoder",
                    "DIN",
                ]
                if any(seq_name in class_name for seq_name in sequence_modules):
                    logger.info(
                        f"Layer {class_name} is a sequence module, using dict input"
                    )
                    return inputs  # Sequence modules usually require a dictionary input

                # check if need dict format input
                dict_attributes = SEQUENCE_QUERY_PARAMS + ["attention"]
                if any(hasattr(layer_obj, attr) for attr in dict_attributes):
                    logger.info(
                        f"Layer {class_name} has sequence attributes, using dict input"
                    )
                    return inputs

                # Default: If inputs is a dictionary and has only one value,
                # extract that value
                if isinstance(inputs, dict):
                    if len(inputs) == 1:
                        single_key = list(inputs.keys())[0]
                        single_value = inputs[single_key]
                        logger.debug(
                            f"Extracting single tensor from dict for {layer_obj.__class__.__name__}"  # NOQA
                        )
                        return single_value
                    else:
                        # In the case of multiple values, try concatenation
                        logger.debug(
                            f"Multiple values in dict, trying to concatenate for {layer_obj.__class__.__name__}"  # NOQA
                        )
                        tensor_list = list(inputs.values())
                        if all(isinstance(t, torch.Tensor) for t in tensor_list):
                            try:
                                # Check if all tensors have
                                # the same number of dimensions
                                # except the last dimension
                                first_shape = tensor_list[0].shape
                                batch_size = first_shape[0]

                                # If the number of dimensions is different,
                                # try flattening and then concatenating
                                flattened_tensors = []
                                for t in tensor_list:
                                    if len(t.shape) != len(first_shape):
                                        # Flatten all dimensions except
                                        # the batch dimension
                                        flattened = t.view(batch_size, -1)
                                        flattened_tensors.append(flattened)
                                    else:
                                        # If the number of dimensions is the same
                                        # but the shape is different, flatten it
                                        if t.shape[:-1] != first_shape[:-1]:
                                            flattened = t.view(batch_size, -1)
                                            flattened_tensors.append(flattened)
                                        else:
                                            flattened_tensors.append(t)

                                result = torch.cat(flattened_tensors, dim=-1)
                                logger.debug(
                                    f"Successfully concatenated tensors, final shape: {result.shape}"  # NOQA
                                )
                                return result
                            except Exception as e:
                                logger.debug(
                                    f"Failed to concatenate tensors: {e}, "
                                    f"using first tensor"
                                )
                                return tensor_list[0]
                        else:
                            # If the concatenation cannot be done,
                            # return the original dictionary.
                            # If it is not a dictionary, return it directly.
                            return inputs
            return inputs

        except Exception as e:
            logger.warning(
                f"Error determining input format for "
                f"{layer_obj.__class__.__name__}: {e}"
            )
            return inputs  # Returns the original input on error

    def call_torch_layer(self, inputs, name, **kwargs):
        """Call predefined torch Layer."""
        layer = self._name_to_layer[name]
        cls = layer.__class__.__name__

        # Determine input format
        processed_inputs = self._determine_input_format(layer, inputs)

        # First try the processed input format
        if self._try_call_layer(layer, processed_inputs, name, cls):
            return self._last_output

        # If that fails and the input format has been modified,
        # try the original input format
        if processed_inputs is not inputs:
            logger.info(f"Retrying {name} with original input format")
            if self._try_call_layer(layer, inputs, name, cls):
                logger.info(f"Successfully called {name} with original input format")
                return self._last_output
            else:
                logger.error(f"Both input formats failed for {name}")
                raise RuntimeError(
                    f"Layer {name} failed with both processed and original input formats"  # NOQA
                )
        else:
            # If the input format has not changed,
            # throw an exception directly
            raise RuntimeError(f"Layer {name} ({cls}) failed to execute")

    def _try_call_layer(self, layer, inputs, name, cls):
        """Attempt to call the layer.

        Args:
            layer: the layer object to call
            inputs: input tensor data
            name: layer name
            cls: layer class name

        Returns:
            bool: Returns True on success, False on failure
        """
        try:
            # Check the module's forward method signature
            # to determine how to pass parameters
            if hasattr(layer, "forward"):
                sig = inspect.signature(layer.forward)
                params = list(sig.parameters.keys())
                # parameters without default values
                required_params = [
                    p
                    for p in sig.parameters.values()
                    if p.default == inspect.Parameter.empty and p.name != "self"
                ]
                if "self" in params:
                    params.remove("self")

                # If inputs is a list/tuple and the layer expects
                # multiple arguments, try spreading it out.
                if (
                    isinstance(inputs, (list, tuple))
                    and len(params) > 1
                    and (
                        len(inputs) == len(params)
                        or len(required_params) >= len(inputs)
                    )
                ):
                    self._last_output = layer(*inputs)
                    logger.debug(
                        f"Layer {name} ({cls}) called successfully with {len(inputs)} separate arguments"  # NOQA
                    )
                else:
                    # Default: single parameter passing
                    self._last_output = layer(inputs)
                    logger.debug(
                        f"Layer {name} ({cls}) called successfully with input type: {type(inputs)}"  # NOQA
                    )
            else:
                # no forward method, directly use
                self._last_output = layer(inputs)
                logger.debug(
                    f"Layer {name} ({cls}) called successfully with input type: {type(inputs)}"  # NOQA
                )
            return True
        except Exception as e:
            msg = getattr(e, "message", str(e))
            logger.error(f"Call layer {name} ({cls}) failed: {msg}")
            return False

    def call_layer(self, inputs, config, name, **kwargs):
        """Call a layer based on its configuration type.

        Args:
            inputs: Input data to be processed by the layer.
            config: Layer configuration containing layer type and parameters.
            name (str): Name of the layer to be called.
            **kwargs: Additional keyword arguments passed to the layer.

        Returns:
            Output from the called layer.

        Raises:
            NotImplementedError: If the layer type is not supported.
        """
        layer_name = config.WhichOneof("layer")
        if layer_name == "module":
            return self.call_torch_layer(inputs, name, **kwargs)
        elif layer_name == "recurrent":
            return self._call_recurrent_layer(inputs, config, name, **kwargs)
        elif layer_name == "repeat":
            return self._call_repeat_layer(inputs, config, name, **kwargs)
        elif layer_name == "lambda":
            if name in self._name_to_layer and isinstance(
                self._name_to_layer[name], LambdaWrapper
            ):
                lambda_wrapper = self._name_to_layer[name]
                return lambda_wrapper(inputs)
            else:
                # execution lambda expression
                conf = getattr(config, "lambda")
                fn = eval(conf.expression)
                return fn(inputs)
        raise NotImplementedError("Unsupported backbone layer:" + layer_name)

    def _call_recurrent_layer(self, inputs, config, name, **kwargs):
        """Call recurrent layer by iterating through all steps.

        Args:
            inputs: Input data to be processed by the recurrent layer.
            config: Recurrent layer configuration.
            name (str): Name of the recurrent layer.
            **kwargs: Additional keyword arguments passed to sub-layers.

        Returns:
            Output from the last step of the recurrent layer.
        """
        recurrent_config = config.recurrent

        # Fixed import index, default -1, display missing fixed import
        fixed_input_index = -1
        if hasattr(recurrent_config, "fixed_input_index"):
            fixed_input_index = recurrent_config.fixed_input_index

        # If there is a fixed input index, the input must be a list or tuple.
        if fixed_input_index >= 0:
            assert isinstance(inputs, (tuple, list)), (
                f"{name} inputs must be a list when using fixed_input_index"
            )
        # Initialize output to input
        output = inputs
        for i in range(recurrent_config.num_steps):
            name_i = f"{name}_{i}"
            if name_i in self._name_to_layer:
                # Calling child layer
                output_i = self.call_torch_layer(output, name_i, **kwargs)

                if fixed_input_index >= 0:
                    # In case of fixed input index:
                    # update all inputs except the fixed index
                    j = 0
                    for idx in range(len(output)):
                        if idx == fixed_input_index:
                            continue  # Skip fixed input index

                        if isinstance(output_i, (tuple, list)):
                            output[idx] = output_i[j]
                        else:
                            output[idx] = output_i
                        j += 1
                else:
                    # without fixed input index: directly replace the entire output
                    output = output_i
            else:
                logger.warning(f"Recurrent sub-layer {name_i} not found, skipping")

        if fixed_input_index >= 0:
            # Delete the element corresponding to the fixed input index
            output = list(output)
            del output[fixed_input_index]

            if len(output) == 1:
                return output[0]
            return output

        return output

    def _call_repeat_layer(self, inputs, config, name, **kwargs):
        """Call repeat layer by iterating through all repetitions.

        Args:
            inputs: Input data to be processed by the repeat layer.
            config: Repeat layer configuration.
            name (str): Name of the repeat layer.
            **kwargs: Additional keyword arguments passed to sub-layers.

        Returns:
            Output based on configuration: single tensor, concatenated tensor, or
        list of tensors.
        """
        repeat_config = config.repeat
        n_loop = repeat_config.num_repeat
        outputs = []

        # execute repeat
        for i in range(n_loop):
            name_i = f"{name}_{i}"
            ly_inputs = inputs

            # Processing input_slice configuration
            if hasattr(repeat_config, "input_slice") and repeat_config.input_slice:
                fn = eval("lambda x, i: x" + repeat_config.input_slice.strip())
                ly_inputs = fn(ly_inputs, i)

            # Processing input_fn configuration
            if hasattr(repeat_config, "input_fn") and repeat_config.input_fn:
                fn = eval(repeat_config.input_fn)
                ly_inputs = fn(ly_inputs, i)

            # Calling child layer
            if name_i in self._name_to_layer:
                output = self.call_torch_layer(ly_inputs, name_i, **kwargs)
                outputs.append(output)
            else:
                logger.warning(f"Repeat sub-layer {name_i} not found, skipping")

        # Output format determined by configuration
        if len(outputs) == 1:
            return outputs[0]

        if (
            hasattr(repeat_config, "output_concat_axis")
            and repeat_config.output_concat_axis is not None
        ):
            axis = repeat_config.output_concat_axis
            return torch.cat(outputs, dim=axis)

        return outputs


class Backbone(nn.Module):
    """Configurable Backbone Network."""

    def __init__(
        self,
        config: backbone_pb2.BackboneTower,
        features: List[BaseFeature],
        embedding_group: Any,
        feature_groups: List[FeatureGroupConfig],
        wide_embedding_dim: Optional[int] = None,
        wide_init_fn: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._config = config
        main_pkg = backbone_pb2.BlockPackage()
        main_pkg.name = "backbone"
        main_pkg.blocks.MergeFrom(config.blocks)
        # If concat_blocks is not configured,
        # concatenate all leaf nodes of the DAG and output them.
        if config.concat_blocks:
            main_pkg.concat_blocks.extend(config.concat_blocks)
        if config.output_blocks:
            # If the output of multiple blocks does not need
            # to be concat together, but as a list type
            # Use output_blocks instead of concat_blocks
            main_pkg.output_blocks.extend(config.output_blocks)

        self._main_pkg = Package(
            main_pkg,
            features,
            embedding_group,
            feature_groups,
            wide_embedding_dim,
            wide_init_fn,
        )
        for pkg in config.packages:
            Package(pkg, features, embedding_group)  # Package is a sub-DAG

        # initial top_mlp
        self._top_mlp = None
        if self._config.HasField("top_mlp"):
            params = Parameter.make_from_pb(self._config.top_mlp)

            # Get total output dimensions from main_pkg
            total_output_dim = self._main_pkg.total_output_dim()

            kwargs = config_to_kwargs(params)
            self._top_mlp = MLP(in_features=total_output_dim, **kwargs)

    def forward(self, batch=None, **kwargs):
        """Forward pass through the backbone network.

        Args:
            batch (Any, optional): Input batch data. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor from the backbone network.
        """
        output = self._main_pkg(batch, **kwargs)

        if hasattr(self, "_top_mlp") and self._top_mlp is not None:
            if isinstance(output, (list, tuple)):
                output = torch.cat(output, dim=-1)
            output = self._top_mlp(output)
        return output

    def output_dim(self):
        """Get the final output dimension, taking into account of top_mlp."""
        if hasattr(self, "_top_mlp") and self._top_mlp is not None:
            if hasattr(self._top_mlp, "output_dim"):
                return self._top_mlp.output_dim()
            elif hasattr(self._top_mlp, "hidden_units") and self._top_mlp.hidden_units:
                # Returns the hidden_units of the last layer
                return self._top_mlp.hidden_units[-1]
            else:
                # Trying to get the output dimension of the last layer from mlp
                if hasattr(self._top_mlp, "mlp") and len(self._top_mlp.mlp) > 0:
                    last_layer = self._top_mlp.mlp[-1]
                    if hasattr(last_layer, "perceptron"):
                        # Get the output dimension of the last Perceptron linear layer
                        linear_layers = [
                            module
                            for module in last_layer.perceptron
                            if isinstance(module, nn.Linear)
                        ]
                        if linear_layers:
                            return linear_layers[-1].out_features
                    elif isinstance(last_layer, nn.Linear):
                        return last_layer.out_features

        # If there is no top_mlp, return the output dimensions of main_pkg
        return self._main_pkg.total_output_dim()

    @classmethod
    def wide_embed_dim(cls, config):
        """Get wide embedding dimension from config."""
        raise NotImplementedError


def merge_inputs(inputs, axis=-1, msg=""):
    """Merge multiple inputs and apply different logic based on input types and count.

    Args:
        inputs (list): Inputs to merge; can be a list of lists or a list of tensors.
            - If all elements are lists, merged into a single list.
            - If elements are a mix of lists and non-list items,
            non-list items are wrapped into single-element lists before merging.
            - If all tensors, they are concatenated along the specified axis.
        axis (int): Axis along which to concatenate tensors,
        effective only when inputs are tensors. Default is -1.
            - If axis = -1, concatenation is along the last dimension.
            - If inputs are lists, this parameter is ignored.
        msg (str): Additional log message to identify the context of the operation.
        Default is an empty string.

    Returns:
        list or torch.Tensor:
            - lists, returns the merged list.
            - tensors, returns the tensor concatenated along the specified axis.
            - If inputs contain only one element, returns that element (no merge).

    Raises:
        ValueError: If inputs is an empty list (length 0),
        indicating there are no inputs to merge.
    """
    if len(inputs) == 0:
        raise ValueError("no inputs to be concat:" + msg)
    if len(inputs) == 1:
        return inputs[0]
    from functools import reduce

    if all(isinstance(x, list) for x in inputs):
        # merge multiple lists into a list
        return reduce(lambda x, y: x + y, inputs)

    if any(isinstance(x, list) for x in inputs):
        logger.warning("%s: try to merge inputs into list" % msg)
        return reduce(
            lambda x, y: x + y, [e if isinstance(e, list) else [e] for e in inputs]
        )

    if axis != -1:
        logger.info("concat inputs %s axis=%d" % (msg, axis))
    return torch.cat(inputs, dim=axis)


def format_value(value):
    """Format the input value based on its type.

    Args:
        value: The value to format.

    Returns:
        The formatted value.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        int_v = int(value)
        return int_v if int_v == value else value
    if isinstance(value, list):
        return [format_value(v) for v in value]
    if isinstance(value, dict):
        return convert_to_dict(value)
    return value


def convert_to_dict(struct):
    """Convert a struct_pb2.Struct object to a Python dictionary.

    Args:
        struct: A struct_pb2.Struct object.

    Returns:
        dict: The converted Python dictionary.
    """
    kwargs = {}
    for key, value in struct.items():
        kwargs[str(key)] = format_value(value)
    return kwargs
