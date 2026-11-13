# DSW 测试分发

GPU PR 在 H20 上运行完整 `gpu` scope，PPU PR 在 PPU 上运行同一 scope。CPU PR 保留 CPU 镜像的全量串行测试。Nightly 留在原 A10 `tzrec-runner`，保持原工作流和全量串行入口，用于 A10 兼容性回归。

实例启停、常驻 runner和代理由 TorchEasyRec-addons 的 `ci/torcheasyrec/` 管理。本仓库的各lane YAML负责启动单个作业容器；容器内local分发负责测试清单、资源需求、分组和结果。

## 本地运行

先使用与仓库 requirements 匹配的已有环境，准备 proto 和测试数据。从仓库根目录运行：

```bash
PYTHONPATH=. python tzrec/tests/run.py --scope gpu
PYTHONPATH=. python tzrec/tests/dispatch.py local --scope gpu --device-groups '0,1;2,3'
```

第二条命令使用两个双卡 worker。编号相对于启动进程的可见设备；已设置 `CUDA_VISIBLE_DEVICES` 时不会扩大设备范围。拒绝重叠、越界或只有一张卡的分组。每个 worker 使用独立进程、工作目录和可写编译缓存，不改变 `TEST_NPROC_PER_NODE`。

默认会话目录在 `tmp/dispatch-<session-id>/`，可使用 `--session` 指定一个新目录。保留 `logs/`、`results/`、`manifest.json` 和 `summary.json` 便于定位问题；同一个会话不能重复执行。`--timeout` 是会话的秒数上限，默认21600。`--pattern` 和 `--test-dir` 只用于明确缩小本地验证范围，CI 全集不传它们。

## 声明可并行测试

未经声明的测试按模块在原设备视图中串行执行。明确测试可以在双卡环境独立执行后，在类或方法上使用：

```python
from tzrec.utils.test_util import ci_resources, mark_ci_scope


@mark_ci_scope("gpu")
@ci_resources(devices=2, split="case")
class ExampleTest(unittest.TestCase): ...
```

`devices` 支持1或2，表示最少需要的可见设备数；两种需求均由双卡 worker 执行。`split` 默认为 `module`，可声明为 `class` 或 `case`。存在模块 fixture 时禁止拆到类/方法，存在类 fixture 时禁止拆到方法。一次测试中的训练、checkpoint恢复、导出等步骤保持在同一组，不拆分其分布式 ranks。`ci_resources` 不改变 `mark_ci_scope` 的筛选规则。

同一模块中未声明需求的用例，会使尚未拆开的模块组继续使用原设备视图。新增用例需要同时考虑 lane 覆盖和资源需求；GPU-only 用例必须有 `gpu` 标记，单独的 `h20` 标记不再足以覆盖 PR。

## CI 执行与结果

同一容器中先在原设备视图中运行 `baseline`，完成发现及独占组，再启动已经验证过的双卡worker进程。协调器通过作业私有 Unix socket 给空闲 worker 分配任务。可以使用 `--durations` 提供受信串行/历史执行记录中的 `{test_id: seconds}` 映射，优先分发耗时较长的组；时长不影响覆盖范围。

每个计划ID都必须恰有一份终态；静态skip决策变化、重复或缺失回执、worker异常退出都不能产生绿色结果。保留subTest、fixture错误、expected failure及真实skip。父进程通过清单和digest核对结果，不解析日志里的“OK”来决定是否通过。

串行入口可选择写入结构化结果：

```bash
PYTHONPATH=. python tzrec/tests/run.py --report-dir tmp/ci-serial
```

它保留原有发现和子进程分流，新增每例结果日志和 `serial-result.json`。不传 `--report-dir` 时保持原入口行为，Nightly 不传该参数。

退出码：0通过，1测试失败，2不完整或协议/执行错误，130取消。正式比较必须对照同一提交、镜像、数据及环境下的串行与并行ID、skip原因、fixture错误和端到端耗时；不能通过减少rank或忽略失败来提速。

## 上线顺序

1. 部署addons运行时，注册仓库级常驻runner并验证代理和标签排空。
1. 各lane YAML使用`tzci-dsw`及对应业务标签，直接启动带归属标签的单个作业容器，保留`ci-test`检查名。
1. GPU/PPU调用现有CI脚本的`--parallel`模式。Nightly保持原工作流和串行入口。

不再使用`dsw_ci.yml`、维护workflow或legacy回退。DSW DockerBoard拒绝GitHub自动注入的Docker socket挂载，因此三条lane均沿用PPU原有的显式`docker run`方式。
