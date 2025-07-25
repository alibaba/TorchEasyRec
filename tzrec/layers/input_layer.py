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

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class VariationalDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask


class InputLayer(nn.Module):
    def __init__(
        self,
        features: List[Any],  # 特征对象列表
        feature_groups: List[Any],  # 每个 group 有 group_name, feature_names
        embedding_reg: Optional[nn.Module] = None,
        kernel_reg: Optional[nn.Module] = None,
        variational_dropout_p: float = 0.0,
        group_special_ops: Optional[Dict[str, nn.Module]] = None,
        seq_attention: Optional[Dict[str, nn.Module]] = None,
        seq_textcnn: Optional[Dict[str, nn.Module]] = None,
        training: bool = True,
    ):
        super().__init__()
        self.training = training
        self.variational_dropout_p = variational_dropout_p
        self.embedding_reg = embedding_reg
        self.kernel_reg = kernel_reg
        self.group_special_ops = group_special_ops or {}
        self.seq_attention = seq_attention or {}
        self.seq_textcnn = seq_textcnn or {}

        self.group_features = {}
        name2feat = {f.name: f for f in features}
        for g in feature_groups:
            group_name = g.group_name if hasattr(g, "group_name") else g["group_name"]
            feature_names = (
                g.feature_names if hasattr(g, "feature_names") else g["feature_names"]
            )
            self.group_features[group_name] = [
                name2feat[n] for n in feature_names if n in name2feat
            ]

        self.embeddings = nn.ModuleDict()
        for f in features:
            if getattr(f, "has_embedding", False):
                if f.name not in self.embeddings:
                    self.embeddings[f.name] = nn.Embedding(
                        f.num_embeddings, f.output_dim
                    )

        self.vdrop = (
            VariationalDropout(variational_dropout_p)
            if variational_dropout_p > 0
            else None
        )

    def apply_regularization(self, weight_list, reg_module):
        if reg_module is None or not weight_list:
            return 0
        return sum(reg_module(w) for w in weight_list)

    def forward(
        self,
        batch,  # 你的 Batch对象
        group_name: str,  # 需要哪个 group
        mode: str = "concat",  # "concat"|"list"|"dict"
        return_reg_loss: bool = False,
    ):
        assert group_name in self.group_features
        feats = self.group_features[group_name]
        tensors = []
        tensor_dict = {}
        emb_reg_list = []
        kernel_reg_list = []

        for f in feats:
            # 稀疏、序列稀疏
            if getattr(f, "is_sparse", False) or getattr(f, "is_sequence", False):
                # 稀疏特征 (非序列)
                if getattr(f, "is_sparse", False) and not getattr(
                    f, "is_sequence", False
                ):
                    kjt: KeyedJaggedTensor = batch.sparse_features.get(group_name)
                    assert kjt is not None, f"No sparse_features[{group_name}] in batch"
                    values = kjt.values(f.name)
                    emb = self.embeddings[f.name](values)
                    # pooling: sum/mean等
                    pooled = emb
                    if hasattr(f, "pooling") and f.pooling == "mean":
                        pooled = emb.mean(dim=1) if emb.dim() > 2 else emb
                    tensors.append(pooled)
                    tensor_dict[f.name] = pooled
                    emb_reg_list.append(self.embeddings[f.name].weight)
                # 序列特征
                elif getattr(f, "is_sequence", False):
                    kjt: KeyedJaggedTensor = batch.sparse_features.get(group_name)
                    if kjt is None:
                        kjt = batch.sequence_mulval_lengths.get(group_name)
                    assert kjt is not None, (
                        f"No sequence/mulval_features[{group_name}] in batch"
                    )
                    values = kjt.values(f.name)
                    emb = self.embeddings[f.name](values)
                    lengths = kjt.lengths(f.name)
                    if f.name in self.seq_attention:
                        pooled = self.seq_attention[f.name](emb, lengths)
                    elif f.name in self.seq_textcnn:
                        pooled = self.seq_textcnn[f.name](emb, lengths)
                    else:  # mean pooling
                        mask = (
                            torch.arange(emb.shape[1], device=emb.device)[None, :]
                            < lengths[:, None]
                        )
                        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp(
                            min=1
                        ).unsqueeze(-1)
                    tensors.append(pooled)
                    tensor_dict[f.name] = pooled
                    emb_reg_list.append(self.embeddings[f.name].weight)
            else:
                # 稠密特征
                kt: KeyedTensor = batch.dense_features.get(group_name)
                assert kt is not None, f"No dense_features[{group_name}] in batch"
                x = kt.values(f.name)
                tensors.append(x)
                tensor_dict[f.name] = x
                kernel_reg_list.append(x)

        # group级特殊操作（如归一化/交互/BN/高阶交互/特征交叉）
        if group_name in self.group_special_ops:
            group_tensor = torch.cat(tensors, dim=-1)
            group_tensor = self.group_special_ops[group_name](group_tensor)
            tensors = [group_tensor]

        # variational dropout
        if self.vdrop:
            out_tensor = self.vdrop(torch.cat(tensors, dim=-1))
        else:
            out_tensor = torch.cat(tensors, dim=-1)

        # 多模式输出
        if mode == "concat":
            out = out_tensor
        elif mode == "list":
            out = tensors
        elif mode == "dict":
            out = tensor_dict
        else:
            raise ValueError(f"Unknown mode: {mode}")
        reg_loss = self.apply_regularization(
            emb_reg_list, self.embedding_reg
        ) + self.apply_regularization(kernel_reg_list, self.kernel_reg)
        if return_reg_loss:
            return out, reg_loss
        return out

    def add_attention(self, feat_name, attn_module):
        self.seq_attention[feat_name] = attn_module

    def add_textcnn(self, feat_name, cnn_module):
        self.seq_textcnn[feat_name] = cnn_module

    def add_special_op(self, group_name, op):
        self.group_special_ops[group_name] = op
