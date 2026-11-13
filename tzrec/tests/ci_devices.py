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

import csv
import json
import re
import subprocess

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


def inventory(hardware: str) -> list[dict]:
    """Enumerate only the expected accelerator family, using live identities."""
    if hardware == "h20":
        data = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
        )
        result = []
        for index, uid, name, bus in csv.reader(
            data.splitlines(), skipinitialspace=True
        ):
            if "H20" not in name:
                raise ValueError("unexpected NVIDIA GPU family")
            result.append(
                dict(
                    index=int(index), id=uid, bus=":".join(bus.upper().split(":")[-2:])
                )
            )
        return result
    if hardware != "ppu":
        raise ValueError("unsupported hardware")
    output = subprocess.check_output(
        ["/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi"], text=True
    )
    found = re.findall(r"\|\s*(\d+)\s+PPU-\S+\s+N/A\s*\|\s*([A-Fa-f0-9:.]+)", output)
    result = [
        dict(index=int(i), id=i, bus=":".join(bus.upper().split(":")[-2:]))
        for i, bus in found
    ]
    if not result:
        raise ValueError("could not enumerate PPU PCI identities")
    return result


CPU_PROBE = """import os,glob,torch,urllib.request,urllib.error
assert torch.__version__.split('+')[0]=='2.14.0'
assert not torch.cuda.is_available()
assert not glob.glob('/dev/nvidia[0-9]*') and not glob.glob('/dev/alixpu_ppu*')
assert not os.path.exists('/var/run/docker.sock')
assert not any(os.environ.get(k) for k in [
 'ALIBABA_CLOUD_ACCESS_KEY_ID','ALIBABA_CLOUD_CREDENTIALS_URI',
 'ALIBABA_CLOUD_SECURITY_TOKEN'])
assert torch.ones(4).sum().item()==4
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for method,path,headers in [
 ('GET','/latest/meta-data/instance-id',{}),
 ('GET','/latest/meta-data/ram/security-credentials/',{}),
 ('PUT','/latest/api/token',{'X-aliyun-ecs-metadata-token-ttl-seconds':'60'})]:
 try:
  response=opener.open(urllib.request.Request('http://100.100.100.200'+path,method=method,headers=headers),timeout=3)
 except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError):continue
 else:
  response.close()
  raise RuntimeError('metadata endpoint is reachable from a test container')
"""


def probe_command(pair: list[dict]) -> list[str]:
    """Build the two-rank collective probe with the assigned PCI identities."""
    command = (
        "import pathlib,sys,os; p='/tmp/torcheasyrec-device-probe.py'; "
        f"pathlib.Path(p).write_text({PROBE!r}); "
        "os.execv(sys.executable,[sys.executable,'-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=29500','--nproc-per-node=2',p])"
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
