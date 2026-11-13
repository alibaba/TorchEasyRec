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

复用现有入口，在已配置的CI runner上运行：

```bash
bash scripts/ci/ci_test_cpu.sh --container
bash scripts/ci/ci_test.sh --container
bash scripts/ci/ci_test_ppu.sh --container
```

`--container`由主仓库启动器管理测试镜像、容器、资源预算和设备。每个job重新探测设备；H20最多4组双卡worker，PPU最多8组，实际并发受CPU、内存和容器配额约束。探针容器也属于本job，失败或取消时一起清理。

基线容器调用同一个`ci_test*.sh --native`，一次完成依赖安装、proto生成、数据下载和基线测试。CPU输出完整串行报告；GPU/PPU把可分卡的测试交给dispatch和worker。worker共享只读依赖环境，使用独立checkout和卡对。未指定`--container`的原有调用行为保留，A10 Nightly继续使用原入口。

测试镜像摘要、各硬件预算及超时配置在`tzrec/tests/ci_config.json`。探针会核对PCI设备和双rank通信，不复用上次开机的卡号。CI启动失败时查看probe日志；测试失败时查看GitHub上传的`report/`，包含组结果和有界日志。可在验收时追加`--pattern utils_test.py`运行测试子集。

TorchEasyRec-addons负责实例扩缩容、代理及常驻runner，不启动worker或解析测试结果。runner传入节点身份、槽位及作业存储目录；本仓库为所有探针和测试容器附加节点、槽位、job标签，供addons安全排空检查。生产升级需先部署提供runner上下文的addons版本，再切换测试workflow。

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
