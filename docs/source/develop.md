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

三条lane各自在workflow中启动一个作业容器，镜像tag、设备、CPU/内存和共享内存参数直接写在`unittest*_ci.yml`。CPU在64核、256GiB的限额内动态分配32个逻辑slot；H20/PPU按需分配一张或两张卡：

```bash
bash scripts/ci/ci_test_cpu.sh --capacity 32
bash scripts/ci/ci_test.sh --hardware h20 --scope gpu --capacity 8
bash scripts/ci/ci_test_ppu.sh --scope gpu --hardware ppu --capacity 16
```

脚本准备一次依赖、proto和数据后统一调用`tzrec/tests/run.py`。调度器给每个组启动一个`run_group.py`进程，使用标准unittest执行。CPU、H20、PPU共用同一套按需分配和归还slot的队列。CPU和GPU都用`@ci_resources(split="class")`或`split="case"`声明更细的边界，容量用`capacity`表示，资源需求用`slots`表示，二者不改变测试的rank数。各组共用一次checkout，每组及其子进程有独立的临时目录。调度进程直接发现测试，使用容器可见的全部加速卡；执行时所有组统一按FIFO申请1或2个slot，执行前检查可见设备数。同一节点上的加速器测试组共用workflow设置的Triton和Inductor缓存目录，后续作业可以复用；新节点从空缓存开始，无需同步。

宿主的两个常驻runner均以root运行，各自使用独立注册目录和`_work`；容器使用镜像默认用户。GPU、CPU、PPU镜像分别使用`1.5`、`1.5-cpu`、`1.5-ppu`，每个job重新拉取；实际digest可在拉取日志中核对。工作目录沿用`/__w/TorchEasyRec/TorchEasyRec/run_<run_id>`。失败组日志落在checkout的`tmp/ci-failures/`；上传失败时保留checkout以便排查。

本地默认容量为2。A10 Nightly保留原runner和全量覆盖，按容量2执行，不传scope。资源限额作用于整个job，各worker共享该额度；不再创建coordinator或worker容器。DSW作业成功时不上传artifact；失败时只上传失败组日志。

addons只管理实例、代理和runner，按照`tzci.node`、`tzci.role`容器标签检查排空。生产切换前仍需部署提供runner上下文的addons版本。

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
