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

- CPU：未审核的用例先按原串行/子进程路由执行；明确可在独立checkout内并行的类或方法用`@cpu_parallel`声明。已声明边界轮流分给最多四个组，每组一个`run.py`进程。任务较少时不开空组。槽位限制常用数学库的线程数，作业总限额仍为16核、80GiB，分布式测试的rank数不变。
- GPU/PPU：`@mark_ci_scope("gpu")`决定是否在该lane执行，`@ci_resources(devices=2, split="case")`等声明决定并行边界。未声明的组使用原设备视图（H20全卡、PPU第一对卡）；其他组在空闲卡对上执行。`CUDA_VISIBLE_DEVICES`相对于父进程的可见集合，重复或越界的卡对会被拒绝。
- `split`默认为`module`，只有无跨组fixture依赖才声明`class`或`case`。CPU并行需单独声明，不从GPU标记推断。ODPS、Kafka等共享服务测试留在基线。作业只检查可见卡数；驱动或镜像变化时另做PCI映射及双rank通信验收。

调度端以`plan.json`记录完整计划，`groups/`保存交给各子进程的用例ID及静态skip决策；`results/`与`logs/`分别保存结果和输出。每组退出码和用例终态必须一致，缺失、重复、异常退出不允许报绿。`summary.json`记录总耗时、计划/上报数量及结果统计。比较串并行时应在相同镜像、资源、数据及凭据下对齐逐例ID和结果；不能只比较测试数。取消时清理本作业启动的进程组，workflow兜底清理本作业容器。

实例、代理、常驻runner和安全排空仍由TorchEasyRec-addons维护。三个workflow直接启动带节点/槽位标签的容器，从工作区路径上传报告。
