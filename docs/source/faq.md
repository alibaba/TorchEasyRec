# FAQ

**Q1: CUDA版本冲突**

**报错信息：**

```
ImportError: /root/miniconda3/envs/tzrec/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
```

**原因：** 环境里面已经有了不同版本的cuda。

**解决方法：** 清空环境变量LD_LIBRARY_PATH即可，通过设置LD_LIBRARY_PATH=来使用当前conda环境的cuda。

______________________________________________________________________

**Q2: 缺少libidn**

**报错信息：**

```
libidn.so.11: cannot open shared object file: No such file or directory
```

**原因：** 系统中缺少libidn.so库文件。

**解决方法：** Centos运行yum install libidn来安装所需的库；Ubuntu可以下载https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/libidn11_1.33-2.2ubuntu2_amd64.deb，apt-get install ./libidn11_1.33-2.2ubuntu2_amd64.deb来安装。

______________________________________________________________________

**Q3: 未检测到GPU**

**报错信息：**

```
libnvidia-ml.so.1: cannot open shared object file: No such file or directory
```

**原因：** 系统未检测到GPU。

**解决方法：** 请确保在具备GPU支持的环境中运行该命令。可以通过`nvidia-smi`命令是否能正常运行来判断是否在GPU环境中。如果是在容器环境中，可以检查容器启动命令中是否包含`--gpus all`。

______________________________________________________________________

**Q4: 训练命令多卡参数与GPU卡数不匹配**

**报错信息：**

```
RuntimeError: CUDA error: invalid device ordinal
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

**原因：** 训练使用的是多卡配置，而环境可能只有单卡或无法找到指定的CUDA设备。

**解决方法：** 请确保训练代码中使用的GPU卡数`--nproc-per-node`与执行环境中的可用GPU卡数相匹配。

______________________________________________________________________

**Q5: pipeline.config文件未找到**

**报错信息：**

```
FileNotFoundError: [Errno 2] No such file or directory: 'experiments/multi_tower_din_taobao_local/pipeline.config'
```

**原因：** pipeline.config文件未找到，可能是训练任务未正常运行导致。

**解决方法：** 检查pipeline.config文件是否存在于指定路径。

______________________________________________________________________

**Q6: 离线预测报 RuntimeError: KeyError: batch_size**

**报错信息：**

```
Trackback of TorchScript, original code (most recent all last):
def forward(self, data : typing_Dict[str,torch.Tensor], device: device = 'cpu') -> typing_Dict[str,torch.Tensor]:
     device_1 = device
     getitem = data['batch_size']
               ~~~~~~~~~~~~~~~~~ <--- HERE
     item = getitem.item(); getitem = None
RuntimeError: KeyError: batch_size
```

**原因：** 离线预测不能支持INPUT_TILE模式导出的模型

**解决方法：** 去掉这个INPUT_TILE环境变量，另外导出个模型

______________________________________________________________________

**Q7: 如何解决FG训练任务错误日志无法在dlc日志中显示的问题**

**原因：** GLOG日志默认存储在/tmp目录下

**解决方法：** 可以在训练时设置环境变量GLOG_logtostderr=1

______________________________________________________________________

**Q8: 如何解决mc读session失效问题**

**报错信息：**

```
pyarrow.lib.ArrowInvalid: Expected to read 538970747 metadata bytes, but only read 122
```

**原因：** 这是因为链接mc的session过期失效了，目前session过期时间为1天

**解决方法：** 使用离线FG + 训练 或 多机多卡加速训练，压缩训练时间在1天内

______________________________________________________________________

**Q9: SEQUENCE特征的特征组配置问题**

**报错信息：**

```
in _regroup_keyed_tensors  KeyError: 'YOU SEQUENCE FEATURE NAME'
```

**原因：** 将序列特征配置在了group_type为DEEP的特征组里，序列是三维的tensor，普通特征是两维的tensor，无法拼到一起。

**解决方法：** 将sequence特征放在group_type为SEQUENCE的组里

______________________________________________________________________

**Q10: Dataloader OOM(out-of-memory)**

**报错信息：**

```
RuntimeError: DataLoader worker (pid 1327) is killed by signal: Killed.
```

**原因：** Dataloader OOM(out-of-memory) 导致 dataloader进程被系统kill。

**解决方法：** 减少data_config中的batch_size或者num_workers

______________________________________________________________________

**Q11: 离线预测写表报schema不对**

**报错信息：**

```
Write data failed - The data stream you provided was not well-formed or did not validate against schema. WriteRecordBatch failed. Data invalid: ODPS-0010000:InvalidArgument:table: xxx, partitions: [] int64 is not equal to stringODPS-0422224: RequestId: xxx
Tag: TUNNEL Endpoint: http://dt.cn-shanghai-vpc.maxcompute.aliyun-inc.com
```

**原因：** 离线预测输出表已存在，并且schema不正确

**解决方法：** 删除已存在的输出表或修改输出表名

______________________________________________________________________

**Q11: fbgemm的embedding lookup op的EmbeddingBoundsCheck error**

**报错信息：** fbgemm的embedding lookup op报错：

```
EmbeddingBoundsCheck (VBE false): (at least one) Out of bounds access for batch: 12, table: 2, bag element: 0, idx: 3, num_rows: 3, indices_start: 1815, indices_end: 1816, T: 244, B: 67, b_t: 1955. Setting idx to zero.
```

**原因：** 第2个embedding table只有3行embedding（num_rows: 3)，但是传入的id是3（idx: 3），越界了

**解决方法：** 只通过报错日志很难直接确定第2个embedding table是关联哪一个特征。需设置环境变量`LOG_LEVEL=INFO`或`LOG_LEVEL=DEBUG`重新执行训练命令，可以看到训练日志中包含如下内容`[TBE=xxx] Contents: ['id_3_emb', 'lookup_2_emb', 'lookup_3_emb', ...`，就可以得知`lookup_3`这个特征的输入值存在问题需要进一步检查输入数据。
