# Develop

## 代码风格

我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为首选代码风格。使用 [ruff](https://github.com/astral-sh/ruff/) 工具进行 美化纠错 和格式化。使用 [pyre](https://pyre-check.org/) 工具进行类型静态检查。

我们在每次提交时都会自动使用 [pre-commit hook](https://pre-commit.com/) , 来检查和格式化 `ruff`、`trailing whitespaces`、修复 `end-of-files`问题，对 `requirements.txt` 进行排序。

ruff 的样式配置可以在[.ruff.toml](../../.ruff.toml) 中找到。

pre-commit hook 的配置存储在 [.pre-commit-config](../../.pre-commit-config.yaml) 中。

pyre 的配置存储在 [.pyre_configuration](../../.pyre_configuration) 中。

在克隆git仓库后，您需要安装初始化pre-commit:

```bash
pip install -r requirements.txt
pre-commit install
```

在此之后，每次提交检查代码 linters 和格式化程序将被强制执行。类型静态检查需要手动执行。

```bash
python scripts/pyre_check.py
```

如果您只想格式化和整理代码，则可以运行

```bash
pre-commit run -a
python scripts/pyre_check.py
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

## 构建镜像

```bash
bash scripts/build_docker.sh
```
