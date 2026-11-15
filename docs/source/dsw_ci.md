# CI测试分组

本地和CI统一使用`run.py`。默认容量为2，按scope筛选后为每个组启动一个独立解释器；不传scope时发现全量测试。A10 Nightly继续使用原A10 runner，容量2，不过滤scope。

```bash
PYTHONPATH=. python tzrec/tests/run.py --list_tests
PYTHONPATH=. python tzrec/tests/run.py --hardware cpu --capacity 32 --session tmp/cpu-report
PYTHONPATH=. python tzrec/tests/run.py --hardware h20 --scope gpu --capacity 8 --session tmp/gpu-report
```

`--list_tests`只打印测试ID，不创建报告或启动执行器。`--session`需要空目录，省略时在`tmp/`生成独立目录。

## 分组规则

- CPU和GPU共用`ci_resources`。类和方法均不标记时，单条用例的资源声明默认为1，分组边界默认为模块；只有类或用例可以独立执行时才声明`split="class"`或`split="case"`。不能跨模块或类的共享fixture拆分。
- CPU：64核、256GiB内存、64GiB共享内存上限，容量32，不绑核。与GPU共用调度循环和FIFO队列，默认组占1，双slot组占2，退出后归还；可以同时运行三十二个普通组，不预分批。逻辑单位不是rank数，不设置`WORLD_SIZE`或`TEST_NPROC_PER_NODE`。
- H20/PPU：容量分别为8和16，1个单位对应一张可见卡。`@mark_ci_scope("gpu")`选择用例。普通用例默认1，无需装饰器；只有双卡需求才标`slots=2`，仅按case拆可写`@ci_resources(split="case")`。所有组按队列顺序领取1或2张空闲卡，双卡组不会被后续单卡组插队。组内资源取最大值，没有全池独占或串行基线。调度进程发现用例时可见容器内全部设备；只有测试组启动后才占用分配的卡。父进程设备数量与容量不符直接报错。设备映射和任意卡对通信在机型或镜像变更时验收。
- H20加速器容器116核，PPU为108核，均为1528GiB内存；CPU和加速器同时运行仍为宿主预留4核、16GiB。上述内存包含各容器使用的共享内存。

每组的stdout和stderr独立保存。所有组结束后，终端和`summary.txt`按发现顺序集中输出原生unittest失败堆栈、实际运行数、跳过数和总耗时。一个组测试失败不妨碍其他组继续；进程崩溃、超时或缺少结果会附上组名和日志，作业返回非零。没有逐例结果JSON或恢复日志。

需要隔离的类用`@ci_resources(split="case", timeout=600)`标记，每个参数化case由调度器直接启动，不嵌套运行器。timeout只约束对应组，不改变其他集成测试的总作业期限。实例、代理和常驻runner仍由TorchEasyRec-addons维护。
