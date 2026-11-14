# CI测试分组

每个CPU、H20或PPU作业在一个容器中运行，`dispatch.py`发现并计划用例，直接调用`run.py --group-file`执行。CPU和加速器是同一节点上的两个独立作业。A10 Nightly仍运行原串行入口。

## 运行

在已准备好的匹配环境中，例如：

```bash
PYTHONPATH=. python tzrec/tests/dispatch.py --hardware cpu --workers 4 --session tmp/cpu-report
PYTHONPATH=. python tzrec/tests/dispatch.py --hardware h20 --scope gpu --device-groups '0,1;2,3' --session tmp/gpu-report
```

`--session`必须是新建或空目录。`--pattern`可缩小本地验收范围；正式CPU CI不传scope或pattern，仍发现全部测试。串行对照使用`python tzrec/tests/run.py --report-dir tmp/serial-report`。

## 分组规则

- CPU和GPU共用`ci_resources`。不标记时以模块为边界；只有类或用例可以独立执行时才声明`split="class"`或`split="case"`。不能跨模块或类的共享fixture拆分。
- CPU：所有边界按发现顺序轮流分给四组，每组一个独立checkout和一个`run.py`进程；原本需要独立子进程的用例仍在所在组内按原方式执行。CPU忽略`devices`；任务少时不开空组。数学库限制槽内线程数，整个作业仍为16核、80GiB，分布式测试的rank数不变。
- GPU/PPU：`@mark_ci_scope("gpu")`选择用例。未指定`devices`时保留原设备视图（H20全卡、PPU第一对卡），显式指定`devices=1`或`2`的独立组才进入双卡队列。只设置`split`不会改变设备视图。重复或越界的卡对会被拒绝。作业核对可见卡数，PCI映射和双rank通信留在设备验收。

调度端以`plan.json`记录完整计划，`groups/`保存交给各子进程的用例ID及静态skip决策；`results/`与`logs/`分别保存结果和输出。每组退出码和用例终态必须一致，缺失、重复、异常退出不允许报绿。`summary.json`记录总耗时、计划/上报数量及结果统计。比较串并行时应在相同镜像、资源、数据及凭据下对齐逐例ID和结果；不能只比较测试数。取消时清理本作业启动的进程组，workflow兜底清理本作业容器。

实例、代理、常驻runner和安全排空仍由TorchEasyRec-addons维护。三个workflow直接启动带节点/槽位标签的容器，从工作区路径上传报告。
