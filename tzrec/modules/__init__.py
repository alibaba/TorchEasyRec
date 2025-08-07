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

from .mlp import MLP
from .backbone_module import Add,FM
from .sequence import DINEncoder as DIN
from .mmoe import MMoE
# from .fm import FactorizationMachine as FM
__all__ = ["MLP","Add","FM","DIN","MMoE"]
