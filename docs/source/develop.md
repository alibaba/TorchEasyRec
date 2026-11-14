# Develop

## 代码风格

我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为首选代码风格。使用 [ruff](https://github.com/astral-sh/ruff/) 工具进行 美化纠错 和格式化。使用 [pyrefly](https://pyrefly.org/) 工具进行类型静态检查。

我们在每次提交时都会自动使用 [pre-commit hook](https://pre-commit.com/) , 来检查和格式化 `ruff`、`trailing whitespaces`、修复 `end-of-files`问题，对 `requirements.txt` 进行排序。

ruff 的样式配置可以在[.ruff.toml](../../.ruff.toml) 中找到。

pre-commit hook 的配置存储在 [.pre-commit-config](../../.pre-commit-config.yaml) 中。

pyrefly 的配置存储在 [pyrefly.toml](../../pyrefly.toml) 中。

在克隆git仓库后，您需要安装初始化pre-commit:

```bash
pip install -r requirements.txt
pre-commit install
```

在此之后，每次提交检查代码 linters 和格式化程序将被强制执行。类型静态检查需要手动执行。

```bash
pyrefly check
```

如果您只想格式化和整理代码，则可以运行

```bash
pre-commit run -a
pyrefly check
```

## 测试

### 单元测试

```bash
bash scripts/ci_test.sh
```

- 运行单个测试用例

```bash
python -m tzrec.modules.fm_test FactorizationMachineTest.test_fm_0
```

### 并行CI测试

三条lane各自在workflow中启动一个作业容器，镜像tag、设备、CPU/内存和共享内存参数直接写在`unittest*_ci.yml`。CPU仍串行运行；GPU/PPU复用现有脚本的并行模式：

```bash
bash scripts/ci/ci_test.sh --parallel --scope gpu --device-groups '0,1;2,3;4,5;6,7'
bash scripts/ci/ci_test_ppu.sh --parallel --scope gpu --hardware ppu --device-groups '0,1;2,3;4,5;6,7;8,9;10,11;12,13;14,15'
```

脚本准备一次依赖、proto和数据后，使用`dispatch local`在同一容器中启动worker子进程。每组仅看到指定的两张卡，并验证PCI身份和双rank通信；各组使用独立checkout；Triton和Inductor缓存按worker槽位分目录，复用runner工作区的`tzrec-ci-cache`。基线保留H20的全卡视图及PPU的原双卡视图。取消时清理进程组，workflow的always步骤兜底清理本job容器。CPU和GPU作业仍是两个独立容器。

宿主的两个常驻runner均以root运行，各自使用独立注册目录和`_work`；容器使用镜像默认用户。GPU、CPU、PPU镜像分别使用`1.5`、`1.5-cpu`、`1.5-ppu`，每个job重新拉取；实际digest可在拉取日志中核对。工作目录沿用`/__w/TorchEasyRec/TorchEasyRec/run_<run_id>`，报告落在checkout旁的`report_<run_id>_<attempt>`。工作流在结束时清理本次checkout；报告上传成功后清理本地report，上传失败则留存排查。通用回执和失败标记独立保留。

未传`--parallel`的调用继续使用原串行入口，A10 Nightly不变。资源限额作用于整个job，各worker共享该额度；不再创建coordinator或worker容器。报告通过工作区挂载上传，临时checkout和编译缓存不上传。

addons只管理实例、代理和runner，按照`tzci.node`、`tzci.role`容器标签检查排空。测试阶段可通过`TZREC_CI_RECEIPT`输出通用作业回执；准备阶段尚未开始测试，没有回执时不自动重跑。生产切换前仍需部署提供runner上下文的addons版本。

## 文档

我们支持 [MarkDown](https://guides.github.com/features/mastering-markdown/) 格式和 [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) 格式的文档。

如果文档包含公式或表格，我们建议您使用 reStructuredText 格式或使用
[md-to-rst](https://cloudconvert.com/md-to-rst) 将现有的 Markdown 文件转换为 reStructuredText 。

**构建文档**

```bash
bash scripts/build_docs.sh
```

## 构建安装包

**构建pip包**

```bash
# nightly
bash scripts/build_wheel.sh nightly
# release
bash scripts/build_wheel.sh release
```

构建完成后脚本会输出 wheel 的绝对路径（`wheel path: ...`）。

```bash
# WHEEL_PATH 为脚本输出的 wheel 绝对路径，[cu130] 可替换为 cpu、cu126、ppu 等
pip install "tzrec[cu130] @ file://${WHEEL_PATH}"
```

## 构建镜像

```bash
bash scripts/build_docker.sh
```

**构建PPU镜像**

```bash
bash scripts/build_docker_ppu.sh
```
