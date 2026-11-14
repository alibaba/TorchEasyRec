# CI测试分组

每个CPU、H20或PPU作业在一个容器中运行，`dispatch.py`发现并计划用例，直接调用`run.py --group-file`执行。CPU和加速器是同一节点上的两个独立作业。A10 Nightly仍运行原串行入口。

## 运行

在已准备好的匹配环境中，例如：

```bash
PYTHONPATH=. python tzrec/tests/dispatch.py --hardware cpu --capacity 8 --session tmp/cpu-report
PYTHONPATH=. python tzrec/tests/dispatch.py --hardware h20 --scope gpu --capacity 8 --session tmp/gpu-report
```

`--session`必须是新建或空目录。`--pattern`可缩小本地验收范围；正式CPU CI不传scope或pattern，仍发现全部测试。串行对照使用`python tzrec/tests/run.py --report-dir tmp/serial-report`。

## 分组规则

- CPU和GPU共用`ci_resources`。类和方法均不标记时，单条用例的资源声明默认为1，分组边界默认为模块；只有类或用例可以独立执行时才声明`split="class"`或`split="case"`。不能跨模块或类的共享fixture拆分。
- CPU：64核、256GiB内存、64GiB共享内存上限，容量8，不绑核。与GPU共用调度循环和FIFO队列，默认组占1，双slot组占2，退出后归还；可以同时运行八个普通组，不预分批。逻辑单位不是rank数，不设置`WORLD_SIZE`或`TEST_NPROC_PER_NODE`。
- H20/PPU：容量分别为8和16，1个单位对应一张可见卡。`@mark_ci_scope("gpu")`选择用例。普通用例默认1，无需装饰器；只有双卡需求才标`slots=2`，仅按case拆可写`@ci_resources(split="case")`。所有组按队列顺序领取1或2张空闲卡，双卡组不会被后续单卡组插队。组内资源取最大值，没有全池独占或串行基线。发现测试时仍用原lane设备视图（H20全卡、PPU前两卡），不会因此预占执行资源。父进程设备数量与容量不符直接报错。设备映射和任意卡对通信在机型或镜像变更时验收。
- H20加速器容器116核，PPU为108核，均为1528GiB内存；CPU和加速器同时运行仍为宿主预留4核、16GiB。上述内存包含各容器使用的共享内存。

调度端以`plan.json`记录完整计划，`groups/`保存交给各子进程的用例ID及静态skip决策；`results/`与`logs/`分别保存结果和输出。每组退出码和用例终态必须一致，缺失、重复、异常退出不允许报绿。`summary.json`记录总耗时、计划/上报数量及结果统计。比较串并行时应在相同镜像、资源、数据及凭据下对齐逐例ID和结果；不能只比较测试数。取消时清理本作业启动的进程组，workflow兜底清理本作业容器。

实例、代理、常驻runner和安全排空仍由TorchEasyRec-addons维护。三个workflow直接启动带节点/槽位标签的容器，从工作区路径上传报告。
