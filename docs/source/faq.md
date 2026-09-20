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

**解决方法：** Centos运行yum install libidn来安装所需的库；Ubuntu可以下载https://tzrec.oss-accelerate.aliyuncs.com/third_party/libidn11_1.33-2.2ubuntu2_amd64.deb，apt-get install ./libidn11_1.33-2.2ubuntu2_amd64.deb来安装。

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

**解决方法：** 升级torcheasyrec版本>=0.7.5

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

______________________________________________________________________

**Q12: CUDA initialization error**

**报错信息：**

```
>>> torch.cuda.is_available()
/opt/conda/lib/python3.11/site-packages/torch/cuda/__init__.py:129: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
False
```

**原因：** 部分显卡（如：RTX系列的）与torcheasyrec cuda镜像中的cuda-compat library不兼容

**解决方法：** 在运行命令前增加环境变量LD_LIBRARY_PATH=，例如

```bash
LD_LIBRARY_PATH= torchrun --master_addr=localhost --master_port=32555 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path multi_tower_din_taobao_local.config
```

______________________________________________________________________

**Q13: kv特征的key包含":"导致报错**

**报错信息：**

```
[rank0]: Original Traceback (most recent call last):
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
[rank0]:     data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 42, in fetch
[rank0]:     data = next(self.dataset_iter)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/tzrec/datasets/dataset.py", line 311, in __iter__
[rank0]:     yield self._build_batch(input_data)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/tzrec/datasets/dataset.py", line 376, in _build_batch
[rank0]:     sampled = self._sampler.get(input_data)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/tzrec/datasets/sampler.py", line 423, in get
[rank0]:     features = self._parse_nodes(nodes)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/tzrec/datasets/sampler.py", line 323, in _parse_nodes
[rank0]:     feature = _to_arrow_array(feature, attr_type)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/tzrec/datasets/sampler.py", line 160, in _to_arrow_array
[rank0]:     items = kv_list.take(list(range(1, len(kv_list), 2))).cast(
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "pyarrow/array.pxi", line 1000, in pyarrow.lib.Array.cast
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/pyarrow/compute.py", line 405, in cast
[rank0]:     return call_function("cast", [arr], options, memory_pool)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "pyarrow/_compute.pyx", line 590, in pyarrow._compute.call_function
[rank0]:   File "pyarrow/_compute.pyx", line 385, in pyarrow._compute.Function.call
[rank0]:   File "pyarrow/error.pxi", line 155, in pyarrow.lib.pyarrow_internal_check_status
[rank0]:   File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
[rank0]: pyarrow.lib.ArrowInvalid: Failed to parse string: 'false}' as a scalar of type float

```

**原因：** kv特征的key包含":"导致报错

**解决方法：** 检查特征表string类型字段是否包含":"，进行数据清洗。

______________________________________________________________________

**Q14: 版本升级后序列特征vocab_list行为变化**

**问题描述：** 从0.9.x或1.0.3-1.1.4升级后，序列特征（如SequenceIdFeature）的vocab_list首元素发生变化。

**原因：** 在1.0.3版本中，`SequenceIdFeature`被合并到`IdFeature`（通过设置`sequence_length`），但`IdFeature`的proto `default_value`默认值为`""`，与原`SequenceIdFeature`的默认值`"0"`不同，导致vocab_list首元素从`"0"`变为`""`。1.1.5版本修复了此问题。

**升级指南：**

- **1.0.2及更早版本（含0.9.x）用户：** 可直接升级到1.0.18或1.1.5+，序列特征的`default_value`行为与旧版本一致，无需额外修改。

- **1.0.3至1.1.4用户：** 升级到1.1.5+后，未显式设置`default_value`的序列特征的vocab_list首元素将从`""`变为`"0"`。如需保持与旧版本完全一致的行为，可通过显式设置`default_bucketize_value`来自行控制vocab_list。示例如下：

  旧版本（1.0.3-1.1.4）配置，未显式设置`default_value`时，内部自动生成的vocab_list为 `["", "<OOV>", "cat", "dog", "bird"]`，其中index0为空字符串`""`，index1为`"<OOV>"`：

  ```protobuf
  feature_config {
    id_feature {
      feature_name: "item_id"
      expression: "item:item_id"
      embedding_dim: 16
      vocab_list: ["cat", "dog", "bird"]
      sequence_length: 50
      sequence_delim: ";"
    }
  }
  ```

  升级到1.1.5+后，如需保持与旧版本一致的vocab_list映射（即index0为`""`，index1为`"<OOV>"`），需将`""`和`"<OOV>"`手动加入vocab_list，并设置`default_bucketize_value: 1`：

  ```protobuf
  feature_config {
    id_feature {
      feature_name: "item_id"
      expression: "item:item_id"
      embedding_dim: 16
      vocab_list: ["", "<OOV>", "cat", "dog", "bird"]
      default_bucketize_value: 1
      sequence_length: 50
      sequence_delim: ";"
    }
  }
  ```

  设置`default_bucketize_value`后，系统不再自动在vocab_list前插入`default_value`和`"<OOV>"`，而是直接使用用户配置的vocab_list，OOV值将映射到`default_bucketize_value`指定的索引。

**版本查询方式：**

```bash
pip index versions tzrec -f http://tzrec.oss-accelerate.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-accelerate.aliyuncs.com
```

______________________________________________________________________

**Q15: Hopper GPU上HSTU注意力反向传播autotuning崩溃**

**报错信息：**

```
File "/opt/conda/lib/python3.11/site-packages/triton/testing.py", line 150, in do_bench
    di.synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**原因：** Triton 3.7.1 自带的 ptxas 12.8.93 在 Hopper（如 H20，sm_90）上会误编译 HSTU 的 WGMMA kernel：`_hstu_attn_bwd` 内核在 autotuning 阶段的共享内存访问越界（约 246 KiB，超出 H20 的 228 KiB 上限），触发非法内存访问（compute-sanitizer 下可见 `Invalid __shared__ read`）。该问题由 Triton `release/3.7` 分支的 commit `6c96454f2f` 将内置 ptxas 从 12.9.86 降级到 12.8.93 引入；换用 ptxas 12.9.86 即可修复（与 MMA v3 代码生成、triton-lang/triton#9514 均无关——经 H20 实测单独打入 #9514 仍报越界）。

**解决方法：** 1.4.0 镜像已内置修复——将官方 triton 3.7.1 wheel 中的 `backends/nvidia/bin/ptxas` 替换为 12.9.86 后重新打包，默认即生效，无需 `DISABLE_MMA_V3`，也不损失 Hopper v3 性能。如需在其它环境手动安装该修复版 wheel：

```bash
pip install --force-reinstall --no-deps \
  https://tzrec.oss-accelerate.aliyuncs.com/third_party/triton/triton-3.7.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

______________________________________________________________________

**Q16: DlrmHSTU/UltraHSTU AOTI sample-input autotune导出报错 `_assert_scalar(u? <= K)`**

**报错信息：**

```
torch._inductor.exc.InductorError: RuntimeError: Runtime assertion failed for expression u0 <= 2002 on node 'le_5'
While executing %_assert_scalar_1 = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le, "Runtime assertion failed for expression u0 <= 2002 on node 'le_5'"), kwargs = {})
```

**原因：** 开启`AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS=1`后，AOTI在`extract_autotune_inputs`中通过`torch.fx.Interpreter`走一次sample-input图，会触达由torch.export插入的`_assert_scalar(u_i <= K)`节点。这里：

- `u0/u2` 是 `max(uih_seq_lengths).item() / max(num_targets).item()`等data-dependent unbacked SymInt（见`tzrec/modules/gr/preprocessors.py:375-381`）；
- `tzrec/ops/utils.py:74` 的 `torch._check(max_len >= runtime_max_seq_len)` 在 `runtime_max_seq_len = max_uih_len + max_targets + max_contextual_seq_len`（见`preprocessors.py:283 + :325`）替换之后，给出 sum 上界 `u0 + u2 ≤ max_seq_len − max_contextual_seq_len`；sympy 进一步用 `u2 ≥ 1` 推出 per-symbol 上界 `u0 ≤ max_seq_len − max_contextual_seq_len − 1`；
- PyTorch对unbacked SymInt的`node.hint`在创建时一次性赋值（约等于`max_seq_len`），后续`torch._check`只更新value-range而不回填hint；当hint大于 per-symbol 上界时 sample-input 走graph会触发 assert。

**解决方法：** 调大`model_config.max_seq_len`使派生上界跳过hint，同时通过新增的`stu.scaling_seqlen`把attention scaling分母固定为原`max_seq_len`，保持推理跟训练时计算完全一致。例：原`max_seq_len: 2048`、`max_contextual_seq_len = 45`，per-symbol 上界 `2048 − 45 − 1 = 2002` 被 hint=2048 打破；改为：

```protobuf
model_config {
  dlrm_hstu {
    hstu {
      stu {
        # ...其余字段不变...
        scaling_seqlen: 2048   # 与训练时相同；保持attention scaling
      }
    }
    max_seq_len: 2112          # 原2048 + 余量；新 per-symbol 上界 2112−45−1 = 2066 > hint
  }
}
```

具体余量需大于`max_contextual_seq_len + 1`（`max_contextual_seq_len` = contextual feature group的特征数）。

______________________________________________________________________

**Q17: MaxCompute Storage API读表失败**

**报错信息：** 图采样（负采样/TDM）加载图数据时报错，最终sampler server退出，训练任务失败：

```
E0824 16:36:02.523523 163241 storage_api_file_system.cc:309] Fail to create read session:
E0824 16:36:02.523869 163241 edge_loader.cc:99] Try to read next edge file failed, Internal:Failed to create read session.
E0824 16:36:06.686076 163234 storage_api_file_system.cc:309] Fail to create read session: Read error
E0824 16:36:04.137389 163237 storage_api_file_system.cc:372] Reach the end of OdpsTable
F0824 16:36:33.027580 163127 server_impl.cc:172] Server load data failed: Internal:Failed to create read session.
```

**原因：** graphlearn 1.3.8及以前版本和pyodps 0.12.x的storage api读表实现有缺陷：session创建的瞬时失败不重试，读流中断还会被当作读完，导致数据静默截断。

**解决方法：** 升级依赖graphlearn>=1.3.9、pyodps>=0.13.1，TorchEasyRec 1.3.19之后的版本已默认包含。已有环境可手动升级（`cp311`需替换为实际的python版本）：

```bash
pip install --force-reinstall --no-deps \
  https://tzrec.oss-accelerate.aliyuncs.com/third_party/graphlearn/graphlearn-1.3.9-cp311-cp311-linux_x86_64.whl
pip install -U "pyodps>=0.13.1"
```

**排查与调优：** graphlearn>=1.3.9新增了以下环境变量，默认值一般无需修改。如果升级后读表仍然失败，优先设置`STORAGE_API_LOG_LEVEL=DEBUG`，SDK默认只按ERROR级别打日志，看不到失败请求的HTTP状态码和响应体，DEBUG可以打印出来（输出到stdout）。

| 环境变量                           | 默认值         | 作用                                                                    |
| ---------------------------------- | -------------- | ----------------------------------------------------------------------- |
| `STORAGE_API_LOG_LEVEL`            | SDK默认(ERROR) | 设为`DEBUG`打印失败请求的HTTP状态码与响应体，可选`DEBUG`/`INFO`/`ERROR` |
| `STORAGE_API_CONNECT_TIMEOUT`      | 180（秒）      | socket连接超时                                                          |
| `STORAGE_API_SOCKET_TIMEOUT`       | 300（秒）      | socket读写超时                                                          |
| `STORAGE_API_RETRY_TIMES`          | 5              | create read session的重试次数（指数退避+抖动）                          |
| `STORAGE_API_READ_RETRY_TIMES`     | 5              | 读流中断后从首个未消费行续读的重试次数                                  |
| `STORAGE_API_SDK_RETRY_TIMES`      | 5              | SDK内部重试次数（仅对连接错误生效）                                     |
| `STORAGE_API_SESSION_POLL_TIMEOUT` | 600（秒）      | 等待read session离开INIT状态的超时时间                                  |
| `STORAGE_API_COMPRESSION`          | `LZ4_FRAME`    | 传输压缩方式，可选`ZSTD`/`UNCOMPRESSED`                                 |

**Q18: 如何复现实验**

**解决方法：** TorchEasyRec通过环境变量控制随机性，不需要在模型或启动脚本里自己调用`torch.manual_seed`：

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHON_RANDOM_SEED=100007
export NUMPY_MANUAL_SEED=100007
export TORCH_MANUAL_SEED=100007          # 同时会设置所有CUDA设备的种子
export USE_DETERMINISTIC_ALGORITHMS=1    # 已包含cudnn的确定性行为
```

______________________________________________________________________

**Q19: tokenize_feature如何截断文本并保留EOS token**

`tokenize_feature`不会自动添加EOS等特殊token。这里有三个机制容易混淆：**按字符截断发生在分词之前，tokenizer的truncation发生在分词之后，EOS在哪一步加入决定了它会不会被截掉**。如果希望文本截断后仍然以EOS结尾，需要根据截断方式选择不同的方案。

**先确定需要哪种截断方式**

| 需求                                       | 推荐方案                                                   | 说明                                                                  |
| ------------------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------- |
| 只需要限制token数量，不需要EOS             | 在`tokenizer.json`中配置`truncation`，用`direction: Right` | 直接按token数截断，不需要`regex_replace_feature`                      |
| 需要EOS，并保留文本开头                    | 在分词前用`regex_replace_feature`按字符截断并追加EOS       | 推荐方案；不要再配置`direction: Right`的tokenizer truncation          |
| 需要EOS，可以丢弃文本开头                  | 上游追加EOS，再配置`direction: Left`                       | Left truncation保留文本末尾，因此EOS不会被截掉                        |
| 既要精确的token数量上限，又要EOS且保留开头 | 当前配置方式无法同时严格保证                               | 可以按字符数保守截断；如果再用Right truncation兜底，超长样本仍会丢EOS |

**1. 推荐方案：分词前截断文本并追加EOS**

例如，需要：

```
原始 title
    ↓ 最多保留前200个字符
截断后的 title + <|im_end|>
    ↓ tokenize
title_token
```

可以通过`regex_replace_feature`和`tokenize_feature`串联实现：

```
feature_configs {
    regex_replace_feature {
        feature_name: "title_eos"
        expression: "item:title"
        regex_pattern: "(?s)^(.{0,200}).*$"
        replacement: "\\1<|im_end|>"
        replace_all: false
        stub_type: true
    }
}
feature_configs {
    tokenize_feature {
        feature_name: "title_token"
        expression: "feature:title_eos"
        vocab_file: "tokenizer.json"
        embedding_dim: 128
        tokens_as_sequence: true
        sequence_length: 64
    }
}
```

这里：

- `regex_replace_feature`先截取最多200个字符，再追加`<|im_end|>`。`.`按字符（UTF-8）计数，不是字节也不是token；`(?s)`让`.`可以匹配换行符；`$`匹配的是文本结尾而不是行结尾，配合`replace_all: false`保证只追加一个EOS
- `stub_type: true`表示`title_eos`只是FG的中间结果，不会作为特征输出给模型
- `tokenize_feature`通过`feature:title_eos`消费上一步的结果
- 特征之间通过`feature:`输入域串联，因此`data_config.fg_mode`需要配置为`FG_DAG`

**2. EOS token的注意事项**

- EOS字面量必须已经存在于`tokenizer.json`的`added_tokens`中，例如Qwen的`<|im_end|>`，否则会被BPE拆成多个token
- `tokenizer_type: sentencepiece`不支持上述方式
- 如果已经用`regex_replace_feature`在文本末尾追加了EOS，就不要再在`tokenizer.json`中配置`direction: Right`的truncation，tokenizer的截断发生在分词之后，会把末尾的EOS再截掉
- 输入为空时，可以不给`regex_replace_feature`配置`default_value`，由后面的`tokenize_feature.default_value`兜底

**3. 序列特征的多段文本**

对于分组序列特征，也可以用相同的方式：

```
feature_configs {
    sequence_feature {
        sequence_name: "click_50_seq"
        sequence_length: 50
        sequence_delim: ";"
        features {
            regex_replace_feature {
                feature_name: "title_eos"
                expression: "item:title"
                regex_pattern: "(?s)^(.{0,200}).*$"
                replacement: "\\1<|im_end|>"
                replace_all: false
                stub_type: true
            }
        }
        features {
            tokenize_feature {
                feature_name: "title_token"
                expression: "feature:title_eos"
                sequence_fields: ["title_eos"]
                vocab_file: "tokenizer.json"
                embedding_dim: 128
            }
        }
    }
}
```

这里需要用`sequence_fields: ["title_eos"]`声明`title_eos`是序列字段，FG会把输入改写成`feature:<sequence_name>__<feature_name>`，从而引用到同一个序列下的中间特征。

**4. 如果需要按token数截断**

如果不要求“保留文本开头的同时保证EOS存在”，可以直接用`tokenizer.json`的`truncation`：

```json
"truncation": {
    "max_length": 128,
    "strategy": "LongestFirst",
    "direction": "Right",
    "stride": 0
}
```

`tokenize_feature`是直接调用tokenizer做Encode的，因此这里的`max_length`限制的是**分词后的token数量**，而不是原始文本的字符数。需要特别区分：

- `direction: Right`：保留前面的token，截掉末尾，因此可能把EOS截掉
- `direction: Left`：保留末尾的token，EOS可以保留，但会丢掉文本开头

`strategy`主要影响文本对的截断方式，单段文本保持默认即可。

**5. 容易混淆的两个参数**

- `text_normalizer`的`max_length`不是文本截断参数，文本超过该长度时它只是跳过normalization并原样输出
- `tokens_as_sequence`时配置的`sequence_length`也不会传给tokenizer做token截断，token数量只能通过tokenizer的`truncation`或分词前的字符截断来控制

`tokenizer.json`中的`padding`见Q20。

______________________________________________________________________

**Q20: tokenize_feature是否应该在tokenizer.json中配置padding**

**一般不建议。** `padding`确实会生效，但补齐出来的pad token在下游和真实token没有区别，TorchEasyRec也不需要定长的输入。

`strategy`配成`{"Fixed": N}`时每条文本都会补齐到N个token；配成`"BatchLongest"`则不起作用，因为FG是逐条调用Encode的，一个“batch”里只有一条文本。

```json
"padding": {
    "strategy": { "Fixed": 128 },
    "direction": "Right",
    "pad_to_multiple_of": null,
    "pad_id": 248044,
    "pad_type_id": 0,
    "pad_token": "<|endoftext|>"
}
```

不建议配置的原因：

- 默认的`tokenize_feature`会把补齐的pad token一起pooling，短文本的向量会被pad的embedding淹没
- `tokens_as_sequence: true`时每条样本的序列长度都变成N，sequence_encoder拿到的长度也全是N，无法区分真实token和padding
- TorchEasyRec在需要稠密序列时会自己按batch内的最大长度padding，并保留每条样本真实的长度用于mask，在tokenizer里补齐反而会丢掉这个信息

如果确实需要定长输出，注意TorchEasyRec生成的FG配置中`output_type`固定为`word_id`，因此只有`pad_id`生效：`pad_token`不会和`pad_id`做一致性校验，配错了不会报错；`pad_id`也不会校验是否在词表范围内，超出词表大小时训练会在embedding查表时越界。

**Q21: 训练报 ValueError: Expected more than 1 value per channel when training**

**报错信息：**

```
  File ".../torch/nn/functional.py", line ..., in batch_norm
    _verify_batch_size(input.size())
  File ".../torch/nn/functional.py", line ..., in _verify_batch_size
    raise ValueError(
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 1024])
```

**原因：** 样本表的行数恰好使得某一轮训练的最后一个batch只有1行。BatchNorm在训练模式下需要在batch内计算统计量，1行样本无法计算方差；batch内负采样、listwise loss等也要求batch内至少有2行样本。TorchEasyRec默认不丢弃任何样本，因此最后一个不足batch_size的batch会原样送入模型。

**解决方法：** 在`data_config`中设置`min_batch_size: 2`，训练时行数小于2的最后一个batch会被丢弃，每个`proc`每轮最多丢弃1行样本；也可以设置`drop_remainder: true`丢弃所有不足batch_size的batch。两个参数都只在训练时生效，评估和预测不受影响。
