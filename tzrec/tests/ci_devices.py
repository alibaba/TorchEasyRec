# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enumerate accelerators and validate the images used by test workers."""

import json

PROBE = """import ctypes,json,os,datetime
import torch
import torch.distributed as dist
rank=int(os.environ["LOCAL_RANK"])
assert torch.cuda.device_count()==2
lib=ctypes.CDLL("libcudart.so")
buses=[]
for i in range(2):
 b=ctypes.create_string_buffer(64)
 assert lib.cudaDeviceGetPCIBusId(b,64,i)==0
 buses.append(":".join(b.value.decode().upper().split(":")[-2:]))
assert sorted(buses)==sorted(json.loads(os.environ["TZCI_PROBE_BUSES"]))
torch.cuda.set_device(rank)
dist.init_process_group("nccl",timeout=datetime.timedelta(seconds=45))
t=torch.tensor([rank+1.0],device="cuda")
dist.all_reduce(t)
assert t.item()==3.0
dist.barrier()
if rank==0:
 print("TZCI_PROBE "+json.dumps({"buses":buses,"torch":torch.__version__}),flush=True)
dist.destroy_process_group()
"""


def visible_buses() -> list[str]:
    """Read PCI identities in the parent's current CUDA-compatible device view."""
    import ctypes

    import torch

    runtime = ctypes.CDLL("libcudart.so")
    buses = []
    for index in range(torch.cuda.device_count()):
        value = ctypes.create_string_buffer(64)
        if runtime.cudaDeviceGetPCIBusId(value, 64, index):
            raise RuntimeError("could not identify a visible accelerator")
        buses.append(":".join(value.value.decode().upper().split(":")[-2:]))
    return buses


def probe_command(pair: list[dict]) -> list[str]:
    """Build the two-rank collective probe with the assigned PCI identities."""
    command = (
        "import pathlib,sys,os; "
        "p='/tmp/torcheasyrec-device-probe-'+str(os.getpid())+'.py'; "
        f"pathlib.Path(p).write_text({PROBE!r}); "
        "os.execv(sys.executable,[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',p])"
    )
    return [
        "env",
        "OMP_NUM_THREADS=1",
        "TZCI_PROBE_BUSES=" + json.dumps([x["bus"] for x in pair]),
        "python",
        "-c",
        command,
    ]


def check_probe(output: str, pair: list[dict], hardware: str) -> None:
    """Require the expected physical devices and matching PyTorch version."""
    row = next(
        json.loads(line.split("TZCI_PROBE ", 1)[1])
        for line in output.splitlines()
        if "TZCI_PROBE " in line
    )
    if sorted(row["buses"]) != sorted(x["bus"] for x in pair):
        raise ValueError("probe reported the wrong devices")
    expected = "2.14.0" if hardware == "h20" else "2.13.0"
    if row["torch"].split("+")[0] != expected:
        raise ValueError("probe image has an incompatible PyTorch version")
