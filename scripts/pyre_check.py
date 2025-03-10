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

import io
import subprocess

IGNORE_PATTERNS = [
    "Missing attribute annotation [4]",
    "Missing global annotation [5]",
    # expected `List[str]` but got `List[_ScalarV]`
    # `google.protobuf.internal.containers._ScalarV` has no attribute `__add__`
    "_ScalarV",
    # Invalid type parameters [24]: Generic type `_Mapping` expects 2 type parameters.
    "_pb2.pyi:",
    "Annotation `pa.Array` is not defined as a type",
    "Annotation `pa.Field` is not defined as a type",
    "Annotation `pa.DataType` is not defined as a type",
    "Annotation `pa.RecordBatch` is not defined as a type",
    "Annotation `pa.Scalar` is not defined as a type",
    "Annotation `parquet.FileMetaData` is not defined as a type",
    "Annotation `faiss.Index` is not defined as a type.",
    "Undefined attribute [16]: Module `pyarrow` has no attribute",
    "Undefined attribute [16]: Module `pyarrow.compute` has no attribute",
    "Undefined attribute [16]: Module `pyarrow.csv` has no attribute",
    # type-safety of torch.nn.Module instances
    # https://github.com/pytorch/pytorch/issues/81462
    # Call error [29]: `typing.Union[nn.modules.module.Module, torch._tensor.Tensor]` is
    # not a function.
    "Union[nn.modules.module.Module, torch._tensor.Tensor]",
    "Union[torch._tensor.Tensor, torch.nn.modules.module.Module]",
    "Union[torch._tensor.Tensor, nn.modules.module.Module]",
    "Union[Module, Tensor]",
]

if __name__ == "__main__":
    result = subprocess.run(["pyre", "check"], stdout=subprocess.PIPE)
    errors = io.StringIO(result.stdout.decode("utf-8"))
    count = 0
    for line in errors.readlines():
        ignore = False
        for pattern in IGNORE_PATTERNS:
            if pattern in line:
                ignore = True
        if ignore:
            continue
        count += 1
        print(line)
    if count > 0:
        print(f"Found {count} critical type errors.")
        exit(1)
    else:
        print("Found no critical type errors.")
