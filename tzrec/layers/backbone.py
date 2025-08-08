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
import logging
from typing import Any, Dict

import networkx as nx
import torch
from networkx.drawing.nx_agraph import to_agraph
from torch import nn

from tzrec.layers.dimension_inference import (
    DimensionInferenceEngine,
    DimensionInfo,
    create_dimension_info_from_embedding,
)
from tzrec.layers.lambda_inference import LambdaOutputDimInferrer
from tzrec.layers.utils import Parameter
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.protos import backbone_pb2
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.dag import DAG
from tzrec.utils.load_class import load_torch_layer


class LambdaWrapper(nn.Module):
    """Lambda expression wrapper for dimension inference and execution."""

    def __init__(self, expression: str, name: str = "lambda_wrapper"):
        super().__init__()
        self.expression = expression
        self.name = name
        self._lambda_fn = None
        self._compile_function()

    def _compile_function(self):
        """Compiling Lambda Functions"""
        try:
            # Creating a secure execution environment
            safe_globals = {
                "torch": torch,
                "__builtins__": {},
                "cat": torch.cat,
                "stack": torch.stack,
                "sum": torch.sum,
                "mean": torch.mean,
                "max": torch.max,
                "min": torch.min,
            }
            self._lambda_fn = eval(self.expression, safe_globals, {})
            if not callable(self._lambda_fn):
                raise ValueError(
                    f"Expression does not evaluate to callable: {self.expression}"
                )
        except Exception as e:
            logging.error(f"Failed to compile lambda function '{self.expression}': {e}")
            raise

    def forward(self, x):
        """Executing lambda expressions"""
        if self._lambda_fn is None:
            raise ValueError("Lambda function not compiled")
        return self._lambda_fn(x)

    def infer_output_dim(self, input_dim_info: DimensionInfo) -> DimensionInfo:
        """Inferring output dims using LambdaOutputDimInferrer."""
        try:
            inferrer = LambdaOutputDimInferrer(safe_mode=True)
            output_dim_info = inferrer.infer_output_dim(input_dim_info, self.expression)
            logging.debug(
                f"Lambda wrapper {self.name} inferred output dim: {output_dim_info}"
            )
            return output_dim_info
        except Exception as e:
            logging.warning(
                f"Failed to infer output dim for lambda {self.name}: {e}, using input dim"
            )
            return input_dim_info

    def __repr__(self):
        return f"LambdaWrapper(name={self.name}, expression='{self.expression}')"


class Package(nn.Module):
    """A sub DAG of tf ops for reuse."""

    __packages = {}

    @staticmethod
    def has_backbone_block(name):
        """Return True if the backbone block with the given name exists."""
        if "backbone" not in Package.__packages:
            return False
        backbone = Package.__packages["backbone"]
        return backbone.has_block(name)

    @staticmethod
    def backbone_block_outputs(name):
        if "backbone" not in Package.__packages:
            return None
        backbone = Package.__packages["backbone"]
        return backbone.block_outputs(name)

    def __init__(
        self,
        config,
        features,
        embedding_group,
        feature_groups,
        wide_embedding_dim=None,
        wide_init_fn=None,
        input_layer=None,
        l2_reg=None,
    ):
        super().__init__()
        # self._base_model_config = config
        self._config = config
        self._features = features
        self._embedding_group = embedding_group
        self._feature_groups = feature_groups
        self._wide_embedding_dim = wide_embedding_dim
        self._wide_init_fn = wide_init_fn
        self._input_layer = input_layer
        self._l2_reg = l2_reg
        self._dag = DAG()
        # build DAG
        self.G = nx.DiGraph()
        self._name_to_blocks = {}

        self._name_to_layer = nn.ModuleDict()  # Layer corresponding to each Block name
        self._name_to_customize = {}  # 存储每个Block是否是自定义实现

        # 使用新的维度推断引擎
        self.dim_engine = DimensionInferenceEngine()

        # 保留兼容性的旧字段
        self._name_to_output_dim = {}  # 存储每个Block的输出维度  e.g. {'user': 160, 'item': 96}
        self._name_to_input_dim = {}  # 存储每个Block的输入维度

        self.reset_input_config(None)
        self._block_outputs = {}
        self._package_input = None
        self._feature_group_inputs = {}
        reuse = None
        input_feature_groups = self._feature_group_inputs

        # ======= step 1: 注册所有节点 =======
        for block in config.blocks:
            if len(block.inputs) == 0:
                raise ValueError("block takes at least one input: %s" % block.name)
            self._name_to_blocks[block.name] = block
            self._dag.add_node(block.name)
            self.G.add_node(block.name)

        # ======= step 2: 补全所有DAG边 ========
        for block in config.blocks:
            name = block.name
            for input_node in block.inputs:
                input_type = input_node.WhichOneof(
                    "name"
                )  # feature_group_name / block_name
                input_name = getattr(input_node, input_type)
                if input_type == "feature_group_name":
                    # 未注册则补注册成输入节点 这部分需要新增DAG节点
                    if input_name not in self._name_to_blocks:
                        # 补注册
                        new_block = backbone_pb2.Block()
                        new_block.name = input_name
                        input_cfg = backbone_pb2.Input()
                        input_cfg.feature_group_name = input_name
                        new_block.inputs.append(input_cfg)
                        new_block.input_layer.CopyFrom(backbone_pb2.InputLayer())
                        self._name_to_blocks[input_name] = new_block
                        self._dag.add_node(input_name)
                        self.G.add_node(input_name)
                        self._dag.add_edge(input_name, name)
                        self.G.add_edge(input_name, name)
                elif input_type == "package_name":
                    # package 为子DAG 作为 Block 的输入 | block package可以打包一组block，构成一个可被复用的子网络，即被打包的子网络以共享参数的方式在同一个模型中调用多次
                    raise NotImplementedError
                    self._dag.add_node_if_not_exists(input_name)
                    self._dag.add_edge(input_name, name)
                    if input_node.HasField("package_input"):
                        pkg_input_name = input_node.package_input
                        self._dag.add_node_if_not_exists(pkg_input_name)
                        self._dag.add_edge(pkg_input_name, input_name)
                elif input_type == "use_package_input":  # delete
                    continue  # 特殊处理
                else:
                    # block-to-block
                    if input_name in self._name_to_blocks:
                        self._dag.add_edge(input_name, name)
                        self.G.add_edge(input_name, name)
                    else:
                        raise KeyError(
                            f"input name `{input_name}` not found in blocks/feature_groups"
                        )
        # ========== step 3: topo排序后依次define_layer ============
        # self.G拓扑排序 输出图片
        # self.G.topological_sort()
        # conda install -c conda-forge pygraphviz
        self.topo_order = nx.topological_sort(self.G)  # 迭代器
        self.topo_order_list = list(self.topo_order)  # list
        A = to_agraph(self.G)
        A.layout("dot")  # 用 graphviz 的 dot 布局
        A.draw("dag.png")  # 输出图片文件
        # self._dag.topological_sort()
        for block_name in self.topo_order_list:
            block = self._name_to_blocks[block_name]
            layer = block.WhichOneof("layer")
            if layer in {"input_layer", "raw_input", "embedding_layer"}:
                # raise NotImplementedError
                # 注册输入相关层 需要1个输入
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
                # 计算output_dim
                # self._name_to_output_dim[block_name] = self._embedding_group.group_total_dim(group) # 计算input_layer的输出维度

                if group in input_feature_groups:
                    # 已有，不重复注册
                    if layer == "input_layer":
                        logging.warning(
                            "input `%s` already exists in other block" % group
                        )
                    elif layer == "raw_input":
                        raise NotImplementedError
                        input_fn = input_feature_groups[group]
                        self._name_to_layer[block.name] = input_fn
                    elif layer == "embedding_layer":
                        raise NotImplementedError
                        inputs, vocab, weights = input_feature_groups[group]
                        block.embedding_layer.vocab_size = vocab
                        params = Parameter.make_from_pb(block.embedding_layer)
                        input_fn = EmbeddingLayer(params, block.name)
                        self._name_to_layer[block.name] = input_fn
                else:
                    input_fn = EmbeddingGroup(
                        features=self._features,
                        feature_groups=self._feature_groups,
                        wide_embedding_dim=self._wide_embedding_dim,
                        wide_init_fn=self._wide_init_fn,
                    )
                    if layer == "input_layer":
                        # 使用改进的维度推断引擎，支持batch_size估算
                        dim_info = create_dimension_info_from_embedding(
                            input_fn,
                            group,
                            batch_size=None,  # 可以在实际使用时传入batch_size
                        )
                        self.dim_engine.register_output_dim(block.name, dim_info)

                        # 保留兼容性
                        self._name_to_output_dim[block.name] = (
                            dim_info.get_feature_dim()
                        )

                        input_feature_groups[group] = (
                            embedding_group  # not a layer is a dim
                        )
                    elif layer == "raw_input":
                        raise NotImplementedError
                        input_fn = self._input_layer.get_raw_features(
                            self._features, group
                        )
                        input_feature_groups[group] = input_fn
                    else:  # embedding_layer
                        raise NotImplementedError
                        inputs, vocab, weights = (
                            self._input_layer.get_bucketized_features(
                                self._features, group
                            )
                        )
                        block.embedding_layer.vocab_size = vocab
                        params = Parameter.make_from_pb(block.embedding_layer)
                        input_fn = EmbeddingLayer(params, block.name)
                        input_feature_groups[group] = (inputs, vocab, weights)
                        logging.info(
                            "add an embedding layer %s with vocab size %d",
                            block.name,
                            vocab,
                        )
                    self._name_to_layer[block.name] = input_fn
            else:  # module
                # 使用新的维度推断引擎处理多输入维度
                input_dim_infos = []

                for input_node in block.inputs:
                    input_type = input_node.WhichOneof("name")
                    input_name = getattr(input_node, input_type)
                    # 解析input_fn & input_slice
                    input_fn = getattr(input_node, "input_fn", None)
                    input_slice = getattr(input_node, "input_slice", None)

                    if input_type == "package_name":
                        # package 为子DAG 作为 Block 的输入
                        raise NotImplementedError
                    else:  # block_name 或者 feature_group_name 的情况
                        # 从维度推断引擎获取输入维度信息
                        input_dim_info = self.dim_engine.get_output_dim(input_name)

                        if input_dim_info is None:
                            # fallback到旧的方式
                            if input_name in self._name_to_output_dim:
                                output_dim = self._name_to_output_dim[input_name]
                                input_dim_info = DimensionInfo(output_dim)
                            else:
                                raise KeyError(
                                    f"input name `{input_name}` not found in blocks/feature_groups"
                                )

                        # 应用input_fn和input_slice变换
                        if input_fn or input_slice:
                            input_dim_info = self.dim_engine.apply_input_transforms(
                                input_dim_info, input_fn, input_slice
                            )

                        input_dim_infos.append(input_dim_info)

                # 合并多个输入的维度信息
                if len(input_dim_infos) == 1:
                    merged_input_dim = input_dim_infos[0]
                else:
                    # 根据block配置决定合并方式
                    merge_mode = (
                        "list"
                        if getattr(block, "merge_inputs_into_list", False)
                        else "concat"
                    )
                    merged_input_dim = self.dim_engine.merge_input_dims(
                        input_dim_infos, merge_mode
                    )

                # 注册输入维度
                self.dim_engine.register_input_dim(block.name, merged_input_dim)

                # 保留兼容性
                self._name_to_input_dim[block.name] = merged_input_dim.get_total_dim()

                # 定义layer
                self.define_layers(layer, block, block.name, reuse)

                # 注册layer到维度推断引擎
                if block.name in self._name_to_layer:
                    layer_obj = self._name_to_layer[block.name]
                    self.dim_engine.register_layer(block.name, layer_obj)

                    # Lambda层需要特殊处理维度推断
                    if isinstance(layer_obj, LambdaWrapper):
                        # 使用LambdaWrapper的infer_output_dim方法
                        output_dim_info = layer_obj.infer_output_dim(merged_input_dim)
                        logging.info(
                            f"Lambda layer {block.name} inferred output dim: {output_dim_info}"
                        )
                    else:
                        # 验证维度兼容性
                        if not self.dim_engine.validate_dimension_compatibility(
                            layer_obj, merged_input_dim
                        ):
                            logging.warning(
                                f"Dimension compatibility check failed for block {block.name}"
                            )

                        # 推断输出维度 - 使用改进的方法
                        output_dim_info = self.dim_engine.infer_layer_output_dim(
                            layer_obj, merged_input_dim
                        )

                    self.dim_engine.register_output_dim(block.name, output_dim_info)

                    # 保留兼容性
                    self._name_to_output_dim[block.name] = (
                        output_dim_info.get_feature_dim()
                    )
                else:
                    # 如果没有layer，使用输入维度作为输出维度
                    self.dim_engine.register_output_dim(block.name, merged_input_dim)
                    self._name_to_output_dim[block.name] = (
                        merged_input_dim.get_feature_dim()
                    )

        # ======= 后处理、输出节点推断 =======
        input_feature_groups = self._feature_group_inputs
        num_groups = len(input_feature_groups)  # input_feature_groups的数量
        num_blocks = (
            len(self._name_to_blocks) - num_groups
        )  # 减去输入特征组的数量,blocks里包含了 feature_groups e.g. feature group user
        assert num_blocks > 0, "there must be at least one block in backbone"
        #
        num_pkg_input = 0
        # 可选: 检查package输入
        # 如果不配置concat_blocks，框架会自动拼接DAG的所有叶子节点并输出
        if len(config.concat_blocks) == 0 and len(config.output_blocks) == 0:
            leaf = self._dag.all_leaves()
            logging.warning(
                (
                    f"{config.name} has no `concat_blocks` or `output_blocks`, "
                    f"try to concat all leaf blocks: {','.join(leaf)}"
                )
            )
            self._config.concat_blocks.extend(leaf)

        Package.__packages[self._config.name] = self

        # 输出维度推断摘要
        dim_summary = self.dim_engine.get_summary()
        logging.info(f"{config.name} dimension inference summary: {dim_summary}")

        logging.info(
            "%s layers: %s" % (config.name, ",".join(self._name_to_layer.keys()))
        )

    def get_output_block_names(self):
        """返回最终作为输出的 block 名字列表（优先 concat_blocks，否则 output_blocks）。"""
        blocks = list(getattr(self._config, "concat_blocks", []))
        if not blocks:
            blocks = list(getattr(self._config, "output_blocks", []))
        return blocks

    def get_dimension_summary(self) -> Dict[str, Any]:
        """获取维度推断的详细摘要信息"""
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

    def validate_all_dimensions(self) -> bool:
        """验证所有block的维度兼容性"""
        all_valid = True
        for block_name, layer in self._name_to_layer.items():
            input_dim_info = self.dim_engine.block_input_dims.get(block_name)
            if input_dim_info is not None:
                if not self.dim_engine.validate_dimension_compatibility(
                    layer, input_dim_info
                ):
                    logging.error(
                        f"Dimension validation failed for block: {block_name}"
                    )
                    all_valid = False
        return all_valid

    def output_block_dims(self):
        """返回最终输出 block 的维度组成的 list，比如 [160, 96]"""
        blocks = self.get_output_block_names()
        # import pdb; pdb.set_trace()
        dims = []
        for block in blocks:
            # 优先使用新的维度推断引擎
            dim_info = self.dim_engine.get_output_dim(block)
            print(f"Output block `{block}` dimension info: {dim_info}")
            if dim_info is not None:
                dims.append(dim_info.get_feature_dim())
            elif block in self._name_to_output_dim:
                dims.append(self._name_to_output_dim[block])
            else:
                raise ValueError(f"block `{block}` not in output dims")
        return dims

    def total_output_dim(self):
        """返回拼接后最终输出的总维度"""
        return sum(self.output_block_dims())

    def define_layers(self, layer, layer_cnf, name, reuse):
        """得到layer

        Args:
            layer (str): the type of layer, e.g., 'module', 'recurrent', 'repeat'.
            layer_cnf (backbone_pb2.LayerConfig): the configuration of the layer.
              class_name: "MLP" mlp {
                hidden_units: 512
                hidden_units: 256
                hidden_units: 128
                activation: "nn.ReLU"
                }
            name (str): the name of the layer. e.g., 'user_mlp'.
            reuse (bool): whether to reuse the layer.
        """
        if layer == "module":
            layer_cls, customize = self.load_torch_layer(
                layer_cnf.module, name, reuse, self._name_to_input_dim.get(name, None)
            )
            self._name_to_layer[name] = layer_cls
            self._name_to_customize[name] = customize
        elif layer == "recurrent":
            keras_layer = layer_cnf.recurrent.module
            for i in range(layer_cnf.recurrent.num_steps):
                name_i = "%s_%d" % (name, i)
                layer_obj = self.load_torch_layer(keras_layer, name_i, reuse)
                self._name_to_layer[name_i] = layer_obj
        elif layer == "repeat":
            keras_layer = layer_cnf.repeat.module
            for i in range(layer_cnf.repeat.num_repeat):
                name_i = "%s_%d" % (name, i)
                layer_obj = self.load_torch_layer(keras_layer, name_i, reuse)
                self._name_to_layer[name_i] = layer_obj
        elif layer == "lambda":
            expression = getattr(layer_cnf, "lambda").expression
            lambda_layer = LambdaWrapper(expression, name=name)
            self._name_to_layer[name] = lambda_layer
            self._name_to_customize[name] = True

    # 用于动态加载  层并根据配置初始化
    def load_torch_layer(self, layer_conf, name, reuse=None, input_dim=None):
        # customize 表示是否是自定义实现
        layer_cls, customize = load_torch_layer(layer_conf.class_name)
        if layer_cls is None:
            raise ValueError("Invalid keras layer class name: " + layer_conf.class_name)
        param_type = layer_conf.WhichOneof("params")
        # st_params是以google.protobuf.Struct对象格式配置的参数；
        # 还可以用自定义的protobuf message的格式传递参数给加载的Layer对象。
        if customize:
            # 代码假定 layer_conf.st_params 是一个结构化参数（is_struct=True），并使用它来创建一个 Parameter 对象，同时传递 L2 正则化参数。
            if param_type is None:  # 没有额外的参数
                layer = layer_cls()
                return layer, customize
            elif param_type == "st_params":
                params = Parameter(layer_conf.st_params, True, l2_reg=self._l2_reg)
            # 如果 param_type 指向 oneof 中的其他字段，代码通过 getattr 动态获取该字段的值，并假定它是一个 Protocol Buffer 消息（is_struct=False）。
            else:
                pb_params = getattr(layer_conf, param_type)
                params = Parameter(pb_params, False, l2_reg=self._l2_reg)
            has_reuse = False
            try:
                # 使用标准库 inspect.signature 获取构造函数的签名
                sig = inspect.signature(layer_cls.__init__)
                has_reuse = "reuse" in inspect.signature(layer_cls.__init__).parameters
            except Exception as e:
                # 如果出现异常，记录警告信息
                logging.warning(f"Failed to inspect function signature: {e}")
            if has_reuse:
                # layer = layer_cls(params, name=name, reuse=reuse)
                raise NotImplementedError
            else:
                kwargs = config_to_kwargs(params)

                # 检查是否需要自动推断 in_features 或 input_dim【改进版本】
                if "in_features" in sig.parameters or "input_dim" in sig.parameters:
                    if "in_features" not in kwargs and "input_dim" not in kwargs:
                        # 从维度推断引擎获取输入维度
                        input_dim_info = self.dim_engine.block_input_dims.get(name)
                        if input_dim_info is not None:
                            feature_dim = input_dim_info.get_feature_dim()
                            # 兼容不同实现风格
                            if "in_features" in sig.parameters:
                                kwargs["in_features"] = feature_dim
                            elif "input_dim" in sig.parameters:
                                kwargs["input_dim"] = feature_dim
                        elif input_dim is not None:
                            # fallback到传入的input_dim参数
                            feature_dim = (
                                input_dim
                                if isinstance(input_dim, int)
                                else (
                                    sum(input_dim)
                                    if isinstance(input_dim, (list, tuple))
                                    else input_dim
                                )
                            )
                            if "in_features" in sig.parameters:
                                kwargs["in_features"] = feature_dim
                            elif "input_dim" in sig.parameters:
                                kwargs["input_dim"] = feature_dim
                        else:
                            raise ValueError(
                                f"{layer_cls.__name__} 需要 in_features 或 input_dim, "
                                "但参数未给定，且无法自动推断。请检查维度推断配置。"
                            )

                # 【新增】通用的sequence_dim和query_dim自动推断
                sequence_dim_missing = (
                    "sequence_dim" in sig.parameters and "sequence_dim" not in kwargs
                )
                query_dim_missing = (
                    "query_dim" in sig.parameters and "query_dim" not in kwargs
                )

                if sequence_dim_missing or query_dim_missing:
                    # Get the input information of the current block
                    block_config = self._name_to_blocks[name]
                    input_dims = self._infer_sequence_query_dimensions(
                        block_config, name
                    )

                    if input_dims:
                        sequence_dim, query_dim = input_dims
                        if sequence_dim_missing:
                            kwargs["sequence_dim"] = sequence_dim
                        if query_dim_missing:
                            kwargs["query_dim"] = query_dim
                        logging.info(
                            f"Auto-inferred dimensions for {layer_cls.__name__} {name}: "
                            f"sequence_dim={sequence_dim if sequence_dim_missing else 'provided'}, "
                            f"query_dim={query_dim if query_dim_missing else 'provided'}"
                        )
                    else:
                        missing_params = []
                        if sequence_dim_missing:
                            missing_params.append("sequence_dim")
                        if query_dim_missing:
                            missing_params.append("query_dim")
                        raise ValueError(
                            f"无法为 {layer_cls.__name__} {name} 自动推断 {', '.join(missing_params)}。"
                            "请确保配置了正确的输入 feature groups 或手动指定这些参数。"
                        )

                layer = layer_cls(
                    **kwargs
                )  # 比如layer_cls是MLP,现在不知道in_features是多少
            return layer, customize
        elif param_type is None:  # internal keras layer 内置 nn.module
            layer = layer_cls(name=name)
            return layer, customize
        else:  # st_params 参数
            assert param_type == "st_params", (
                "internal keras layer only support st_params"
            )
            try:
                kwargs = convert_to_dict(layer_conf.st_params)
                logging.info(
                    "call %s layer with params %r" % (layer_conf.class_name, kwargs)
                )
                layer = layer_cls(name=name, **kwargs)
            except TypeError as e:
                logging.warning(e)
                args = map(format_value, layer_conf.st_params.values())
                logging.info(
                    "try to call %s layer with params %r"
                    % (layer_conf.class_name, args)
                )
                layer = layer_cls(*args, name=name)
            return layer, customize

    def reset_input_config(self, config):
        self.input_config = config

    def _infer_sequence_query_dimensions(self, block_config, block_name):
        """Inference module sequence_dim and query_dim

        适用于任何需要序列和查询维度的模块（如DINEncoder等）

        Args:
            block_config: Block的配置信息
            block_name: Block的名称

        Returns:
            tuple: (sequence_dim, query_dim) 或 None 如果推断失败
        """
        try:
            sequence_dim = None
            query_dim = None

            # 分析输入，根据feature_group_name推断维度
            for input_node in block_config.inputs:
                input_type = input_node.WhichOneof("name")
                input_name = getattr(input_node, input_type)

                # 只处理feature_group_name类型的输入
                if input_type == "feature_group_name":
                    group_name = input_name

                    # 尝试获取.sequence和.query子组的维度
                    try:
                        sequence_group_name = f"{group_name}.sequence"
                        query_group_name = f"{group_name}.query"
                        # 检查是否存在这些子组
                        if hasattr(self._name_to_layer[group_name], "group_total_dim"):
                            try:
                                test_seq_dim = self._name_to_layer[
                                    group_name
                                ].group_total_dim(sequence_group_name)
                                test_query_dim = self._name_to_layer[
                                    group_name
                                ].group_total_dim(query_group_name)

                                # 如果能成功获取维度，说明这是正确的格式
                                sequence_dim = test_seq_dim
                                query_dim = test_query_dim

                                logging.info(
                                    f"Auto-inferred dimensions from {group_name}: "
                                    f"sequence_dim={sequence_dim} (from {sequence_group_name}), "
                                    f"query_dim={query_dim} (from {query_group_name})"
                                )

                                return sequence_dim, query_dim

                            except Exception:
                                # 如果无法获取子组维度，继续尝试其他方式
                                logging.debug(
                                    f"Could not get .sequence/.query dimensions for {group_name}"
                                )
                                continue
                    except Exception as e:
                        logging.debug(
                            f"Error accessing embedding group dimensions: {e}"
                        )
                        continue

                elif input_type == "block_name":
                    # 从其他block获取维度作为fallback
                    dim_info = self.dim_engine.get_output_dim(input_name)
                    if dim_info is not None:
                        dim = dim_info.get_feature_dim()
                        # 如果还没有找到sequence_dim，使用这个作为sequence_dim
                        if sequence_dim is None:
                            sequence_dim = dim
                            logging.info(
                                f"Using block {input_name} output as sequence with dim {dim}"
                            )
                        # 如果还没有找到query_dim，使用这个作为query_dim
                        elif query_dim is None:
                            query_dim = dim
                            logging.info(
                                f"Using block {input_name} output as query with dim {dim}"
                            )

            if sequence_dim is not None and query_dim is not None:
                return sequence_dim, query_dim
            else:
                logging.warning(
                    f"Could not infer sequence/query dimensions for {block_name}: "
                    f"sequence_dim={sequence_dim}, query_dim={query_dim}"
                )
                return None

        except Exception as e:
            logging.error(
                f"Error inferring sequence/query dimensions for {block_name}: {e}"
            )
            return None

    def set_package_input(self, pkg_input):
        self._package_input = pkg_input

    def has_block(self, name):
        return name in self._name_to_blocks

    def block_outputs(self, name):
        return self._block_outputs.get(name, None)

    def block_input(self, config, block_outputs, training=None, **kwargs):
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
                        pkg_input = inner_package(training)
                    if input_node.HasField("package_input_fn"):
                        fn = eval(input_node.package_input_fn)
                        pkg_input = fn(pkg_input)
                    package.set_package_input(pkg_input)
                input_feature = package(training, **kwargs)

            elif input_name in block_outputs:
                input_feature = block_outputs[input_name]

            else:
                input_feature = Package.backbone_block_outputs(input_name)

            if input_feature is None:
                raise KeyError(f"input name `{input_name}` does not exist")

            if getattr(input_node, "ignore_input", False):
                continue

            if input_node.HasField(
                "input_slice"
            ):  # 通过python切片语法获取到输入元组的某个元素作为输入
                # input_slice例子："[..., :10]"
                fn = eval("lambda x: x" + input_node.input_slice.strip())
                input_feature = fn(input_feature)

            if input_node.HasField(
                "input_fn"
            ):  # 指定一个lambda函数对输入做一些简单的变换。比如配置input_fn: 'lambda x: [x]'可以把输入变成列表格式。
                # 没有tf.name_scope，直接调用
                fn = eval(input_node.input_fn)
                input_feature = fn(input_feature)
                # 需要重新计算input_dim

            inputs.append(input_feature)

        # 合并输入
        if getattr(config, "merge_inputs_into_list", False):
            output = inputs
        else:
            try:
                # merge_inputs需要你自定义，例如用torch.cat
                # 假设config.input_concat_axis有定义，通常是1
                output = merge_inputs(
                    inputs,
                    axis=getattr(config, "input_concat_axis", 1),
                    msg=config.name,
                )
            except ValueError as e:
                msg = getattr(e, "message", str(e))
                logging.error(f"merge inputs of block {config.name} failed: {msg}")
                raise e

        if config.HasField(
            "extra_input_fn"
        ):  # 来对合并后的多路输入结果做一些额外的变换，需要配置成lambda函数的格式。
            fn = eval(config.extra_input_fn)
            output = fn(output)

        return output

    def forward(self, is_training, batch=None, **kwargs):
        block_outputs = {}
        self._block_outputs = block_outputs  # reset
        blocks = self.topo_order_list
        blocks = self._dag.topological_sort()  # 拓扑排序
        logging.info(self._config.name + " topological order: " + ",".join(blocks))

        for block in blocks:  # 遍历每个block
            if block not in self._name_to_blocks:
                # package block
                assert block in Package.__packages, "invalid block: " + block
                continue
            config = self._name_to_blocks[block]
            # Case 1: sequential layers
            if hasattr(config, "layers") and config.layers:
                logging.info("call sequential %d layers" % len(config.layers))
                output = self.block_input(config, block_outputs, is_training, **kwargs)
                for i, layer in enumerate(config.layers):
                    name_i = "%s_l%d" % (block, i)
                    output = self.call_layer(output, layer, name_i, **kwargs)
                block_outputs[block] = output
                continue

            # Case 2: single layer  just one of layer
            layer_type = config.WhichOneof("layer")
            if layer_type is None:  # identity layer
                output = self.block_input(config, block_outputs, is_training, **kwargs)
                block_outputs[block] = output
            elif layer_type == "raw_input":
                block_outputs[block] = self._name_to_layer[block]
            elif layer_type == "input_layer":
                # 如果self._name_to_layer有block属性且不为None
                # 直接调用 self._name_to_layer[block]，否则调用 embedding group
                if (
                    block in self._name_to_layer
                    and self._name_to_layer[block] is not None
                ):
                    input_fn = self._name_to_layer[block]  # embedding group
                else:
                    input_fn = self._embedding_group
                # 本身没有block input 了
                input_config = config.input_layer
                if self.input_config is not None:
                    input_config = self.input_config
                    if hasattr(input_fn, "reset"):
                        input_fn.reset(input_config, is_training)
                # block_outputs[block] = input_fn(input_config, is_training)
                # block_outputs[block] = input_fn(input_config) # embedding group 没有is training 参数
                if batch is not None:
                    embedding_outputs = input_fn(
                        batch
                    )  # input_fn(batch) 是 tensor dict
                    if (
                        isinstance(embedding_outputs, dict)
                        and block in embedding_outputs
                    ):
                        block_outputs[block] = embedding_outputs[block]
                    else:
                        # 如果返回的不是字典或没有对应的key，直接使用整个输出
                        block_outputs[block] = embedding_outputs
                    if isinstance(block_outputs[block], torch.Tensor):
                        print(
                            f"block_outputs[{block}] shape: {block_outputs[block].shape}"
                        )
                    else:
                        print(
                            f"block_outputs[{block}] type: {type(block_outputs[block])}"
                        )
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
                block_outputs[block] = input_fn([inputs, weights], is_training)
            else:
                # module  Custom layer 一些自定义的层  例如 mlp
                inputs = self.block_input(config, block_outputs, is_training, **kwargs)
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
                # print(f"Adding output block: {output} with shape {block_outputs[output].shape}") 不一定是tensor 有可能是tensor list 不一定能.shape
                outputs.append(block_outputs[output])
            else:
                raise ValueError("No output `%s` of backbone to be concat" % output)

        try:
            print(f"Number of outputs to merge: {len(outputs)}")
            # 打印每个output的shape
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"Output {i} shape: {out.shape}")
                elif isinstance(out, (list, tuple)):
                    print(f"Output {i} is a list/tuple with {len(out)} elements.")
                else:
                    print(f"Output {i} is of type {type(out)}")
            # merge_inputs需自定义为torch的concatenate等
            output = merge_inputs(outputs, msg="backbone")
        except Exception as e:
            logging.error("merge backbone's output failed: %s", str(e))
            raise e
        return output

    def _determine_input_format(self, layer_obj, inputs):
        """智能判断模块需要的输入格式

        Args:
            layer_obj: 要调用的层对象
            inputs: 输入数据（可能是tensor dict或单个tensor）

        Returns:
            适合该层的输入格式
        """
        try:
            # 检查layer的forward方法签名
            if hasattr(layer_obj, "forward"):
                sig = inspect.signature(layer_obj.forward)
                params = list(sig.parameters.keys())

                # 排除self参数
                if "self" in params:
                    params.remove("self")

                # 如果forward方法有多个参数，可能需要字典输入
                if len(params) > 1:
                    logging.debug(
                        f"Layer {layer_obj.__class__.__name__} has multiple forward parameters: {params}"
                    )
                    # 检查是否有特定的参数名暗示需要字典输入
                    dict_indicators = [
                        "grouped_features",
                        "feature_dict",
                        "inputs_dict",
                        "batch",
                    ]
                    if any(indicator in params for indicator in dict_indicators):
                        logging.info(
                            f"Layer {layer_obj.__class__.__name__} likely needs dict input"
                        )
                        return inputs  # 返回原始字典格式

                # 检查是否是序列相关的模块
                class_name = layer_obj.__class__.__name__
                sequence_modules = [
                    "DINEncoder",
                    "AttentionLayer",
                    "SequenceLayer",
                    "DIN",
                ]
                if any(seq_name in class_name for seq_name in sequence_modules):
                    logging.info(
                        f"Layer {class_name} is a sequence module, using dict input"
                    )
                    return inputs  # 序列模块通常需要字典输入

                # 检查模块是否有特定的属性暗示需要字典输入
                dict_attributes = ["sequence_dim", "query_dim", "attention"]
                if any(hasattr(layer_obj, attr) for attr in dict_attributes):
                    logging.info(
                        f"Layer {class_name} has sequence attributes, using dict input"
                    )
                    return inputs

                # 默认情况：如果inputs是字典且只有一个值，提取该值
                if isinstance(inputs, dict):
                    if len(inputs) == 1:
                        single_key = list(inputs.keys())[0]
                        single_value = inputs[single_key]
                        logging.debug(
                            f"Extracting single tensor from dict for {layer_obj.__class__.__name__}"
                        )
                        return single_value
                    else:
                        # 多个值的情况，尝试拼接
                        logging.debug(
                            f"Multiple values in dict, trying to concatenate for {layer_obj.__class__.__name__}"
                        )
                        tensor_list = list(inputs.values())
                        if all(isinstance(t, torch.Tensor) for t in tensor_list):
                            try:
                                # 检查所有tensor是否有相同的维度数（除了最后一维）
                                first_shape = tensor_list[0].shape
                                batch_size = first_shape[0]

                                # 如果维度数不同，尝试展平后拼接
                                flattened_tensors = []
                                for t in tensor_list:
                                    if len(t.shape) != len(first_shape):
                                        # 展平除了batch维度外的所有维度
                                        flattened = t.view(batch_size, -1)
                                        flattened_tensors.append(flattened)
                                    else:
                                        # 如果维度数相同但shape不同，也展平
                                        if t.shape[:-1] != first_shape[:-1]:
                                            flattened = t.view(batch_size, -1)
                                            flattened_tensors.append(flattened)
                                        else:
                                            flattened_tensors.append(t)

                                result = torch.cat(flattened_tensors, dim=-1)
                                logging.debug(
                                    f"Successfully concatenated tensors, final shape: {result.shape}"
                                )
                                return result
                            except Exception as e:
                                logging.debug(
                                    f"Failed to concatenate tensors: {e}, using first tensor"
                                )
                                return tensor_list[0]
                        else:
                            return inputs  # 如果不能拼接，返回原字典            # 如果不是字典，直接返回
            return inputs

        except Exception as e:
            logging.warning(
                f"Error determining input format for {layer_obj.__class__.__name__}: {e}"
            )
            return inputs  # 出错时返回原始输入

    def call_keras_layer(self, inputs, name, **kwargs):
        """Call predefined torch Layer, which can be reused."""
        layer = self._name_to_layer[name]
        customize = self._name_to_customize.get(name, False)
        cls = layer.__class__.__name__

        # 判断输入格式
        processed_inputs = self._determine_input_format(layer, inputs)

        if customize:
            try:
                output = layer(processed_inputs)
                logging.debug(
                    f"Custom layer {name} ({cls}) called successfully with input type: {type(processed_inputs)}"
                )
            except Exception as e:
                msg = getattr(e, "message", str(e))
                logging.error("call torch layer %s (%s) failed: %s" % (name, cls, msg))
                # 尝试使用原始输入格式
                if processed_inputs is not inputs:
                    logging.info(f"Retrying {name} with original input format")
                    try:
                        output = layer(inputs)
                        logging.info(
                            f"Successfully called {name} with original input format"
                        )
                    except Exception as e2:
                        logging.error(f"Both input formats failed for {name}: {e2}")
                        raise e
                else:
                    raise e
        else:
            try:
                output = layer(processed_inputs)
                if cls == "BatchNormalization":
                    raise NotImplementedError
                    add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
            except TypeError:
                output = layer(processed_inputs)
            except Exception as e:
                # 尝试使用原始输入格式
                if processed_inputs is not inputs:
                    logging.info(
                        f"Retrying internal layer {name} with original input format"
                    )
                    try:
                        output = layer(inputs)
                    except Exception as e2:
                        logging.error(
                            f"Both input formats failed for internal layer {name}: {e2}"
                        )
                        raise e
                else:
                    raise e
        return output

    def call_layer(self, inputs, config, name, **kwargs):
        layer_name = config.WhichOneof("layer")
        if layer_name == "module":
            return self.call_keras_layer(inputs, name, **kwargs)
        elif layer_name == "lambda":
            # 优先使用注册的LambdaWrapper，如果存在的话
            if name in self._name_to_layer and isinstance(
                self._name_to_layer[name], LambdaWrapper
            ):
                lambda_wrapper = self._name_to_layer[name]
                return lambda_wrapper(inputs)
            else:
                # fallback到直接执行lambda表达式
                conf = getattr(config, "lambda")
                fn = eval(conf.expression)
                return fn(inputs)
        raise NotImplementedError("Unsupported backbone layer:" + layer_name)


class Backbone(nn.Module):
    """Configurable Backbone Network."""

    def __init__(
        self,
        config,
        features,
        embedding_group,
        feature_groups,
        wide_embedding_dim=None,
        wide_init_fn=None,
        input_layer=None,
        l2_reg=None,
    ):
        super().__init__()
        self._config = config
        self._l2_reg = l2_reg
        main_pkg = backbone_pb2.BlockPackage()
        main_pkg.name = "backbone"
        main_pkg.blocks.MergeFrom(config.blocks)
        if (
            config.concat_blocks
        ):  # 如果不配置concat_blocks，框架会自动拼接DAG的所有叶子节点并输出。
            main_pkg.concat_blocks.extend(config.concat_blocks)
        if config.output_blocks:  # 如果多个block的输出不需要 concat 在一起，而是作为一个list类型（下游对接多目标学习的tower）可以用output_blocks代替concat_blocks
            main_pkg.output_blocks.extend(config.output_blocks)

        self._main_pkg = Package(
            main_pkg,
            features,
            embedding_group,
            feature_groups,
            wide_embedding_dim,
            wide_init_fn,
            input_layer,
            l2_reg,
        )  # input_layer目前没有用到
        for pkg in config.packages:
            Package(
                pkg, features, embedding_group, input_layer, l2_reg
            )  # Package是一个子DAG

        # 初始化 top_mlp 目前top_mlp也会改变输出维度，暂未修复
        self._top_mlp = None
        if self._config.HasField("top_mlp"):
            params = Parameter.make_from_pb(self._config.top_mlp)
            params.l2_regularizer = self._l2_reg

            # 从main_pkg获取总输出维度
            total_output_dim = self._main_pkg.total_output_dim()

            kwargs = config_to_kwargs(params)
            self._top_mlp = MLP(in_features=total_output_dim, **kwargs)

    def forward(self, is_training, batch=None, **kwargs):
        output = self._main_pkg(is_training, batch, **kwargs)

        if hasattr(self, "_top_mlp") and self._top_mlp is not None:
            if isinstance(output, (list, tuple)):
                output = torch.cat(output, dim=-1)
            output = self._top_mlp(output)
        return output

    def get_final_output_dim(self):
        """获取最终输出维度，考虑top_mlp的影响"""
        if hasattr(self, "_top_mlp") and self._top_mlp is not None:
            # 如果有top_mlp，返回top_mlp的输出维度
            if hasattr(self._top_mlp, "output_dim"):
                return self._top_mlp.output_dim()
            elif hasattr(self._top_mlp, "hidden_units") and self._top_mlp.hidden_units:
                # 返回最后一层的hidden_units
                return self._top_mlp.hidden_units[-1]
            else:
                # 尝试从MLP的mlp模块列表中获取最后一层的输出维度
                if hasattr(self._top_mlp, "mlp") and len(self._top_mlp.mlp) > 0:
                    last_layer = self._top_mlp.mlp[-1]
                    if hasattr(last_layer, "perceptron"):
                        # 获取最后一个Perceptron的线性层输出维度
                        linear_layers = [
                            module
                            for module in last_layer.perceptron
                            if isinstance(module, nn.Linear)
                        ]
                        if linear_layers:
                            return linear_layers[-1].out_features
                    elif isinstance(last_layer, nn.Linear):
                        return last_layer.out_features

        # 如果没有top_mlp，返回main_pkg的输出维度
        return self._main_pkg.total_output_dim()

    @classmethod
    def wide_embed_dim(cls, config):
        wide_embed_dim = None
        raise NotImplementedError


def merge_inputs(inputs, axis=-1, msg=""):
    """合并多个输入，根据输入类型和数量执行不同的逻辑处理。

    参数:
    inputs (list): 待合并的输入，可以是列表或张量的列表。
                   - 如果所有元素是列表，则合并为一个列表。
                   - 如果元素既有列表又有非列表类型，
                     则将非列表类型转换为单元素列表后合并。
                   - 如果所有元素是张量，则沿指定轴进行拼接。
    axis (int): 指定张量拼接的维度，仅在输入为张量时有效。默认值为 -1。
                - 如果 axis=-1 表示沿最后一个维度拼接。
                - 如果输入是列表，此参数无效。
    msg (str): 附加的日志信息，用于标识当前操作的上下文。默认值为空字符串。

    返回:
    list 或 torch.Tensor:
        - 如果输入是列表，返回合并后的列表。
        - 如果输入是张量，返回沿指定轴拼接后的张量。
        - 如果输入只有一个元素，直接返回该元素（无合并操作）。

    异常:
    ValueError: 如果 inputs 为空列表（长度为 0)抛出异常 提示没有输入可供合并。
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
        logging.warning("%s: try to merge inputs into list" % msg)
        return reduce(
            lambda x, y: x + y, [e if isinstance(e, list) else [e] for e in inputs]
        )

    if axis != -1:
        logging.info("concat inputs %s axis=%d" % (msg, axis))
    # for i, x in enumerate(inputs): print(f"fzcccccc{i}: {x.shape}")
    return torch.cat(inputs, dim=axis)


# 根据输入值的类型对其进行格式化处理
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
    if isinstance(value, list):  # 替换 struct_pb2.ListValue 为普通列表支持
        return [format_value(v) for v in value]
    if isinstance(value, dict):  # 替换 struct_pb2.Struct 为普通字典支持
        return convert_to_dict(value)
    return value


# 将 struct_pb2.Struct 类型的对象转换为 Python 字典
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
