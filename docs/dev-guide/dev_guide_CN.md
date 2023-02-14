# LinkeFL 开发指南 - 必读

## 1. 准备本地开发环境

### 1.1 以开发模式安装 LinkeFL
在本地命令行终端依次执行下列命令：
```shell
git@github.com:Linke-Data/LinkeFL.git
cd LinkeFL
bash source_install.sh
```

其中第三步在执行 `source_install.sh` 脚本时，会依次安装 `requirements.txt` 中的第三方依赖，其中包含 `torch`, `xgboost` 等体积较大的包，请耐心等待。所有第三方包安装完成后，会运行 `setup.py` 并调用 [Cython](https://github.com/cython/cython) 来编译 `LinkeFL/linkefl` 目录下所有 Python 源文件，当终端中打印出如下提示信息时：
```
Installing collected packages: linkefl
  Running setup.py develop for linkefl
```
说明 Cython 编译程序已经成功启动，并在后台静默运行。由于以开发模式安装 Python 包时只能分配单线程给 Cython 执行编译任务，所以此编译过程花费时间较长，**请耐心等待**。LinkeFL 只需以开发模式安装一次，后续修改代码后无需再重新安装。最终当终端显示如下信息时，说明 LinkeFL 安装成功。
```
Successfully installed linkefl
```

**(Optional)**: 在进行一个新 Python 项目开发时，一般建议新建一个虚拟环境，而不是使用系统全局 Python，以避免不必要的依赖冲突。这里推荐使用 Python3 自带的 `venv` 模块创建虚拟环境。执行
``` shell
python3 -m venv linkefl_env
source linkefl_env/bin/activate
```
上述命令会在当前目录下创建一个 `linkefl_env` 虚拟环境目录，并激活该环境。

### 1.2 安装 pre-commit
Git 有一种 [Hook 机制](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)，其主要用途是在 Git 中发生某些事件之前（如 commit / merge / push 事件等）触发某个动作，例如在 commit 之前检查代码风格，如果风格不达标则终止本次 commit。为了实现这些事件动作，用户需要在项目 `.git/hooks/` 目录下手动编写 bash 脚本，门槛较高而且比较繁琐。

[pre-commit](https://pre-commit.com/) 是一个管理 Git Hook 的框架，其支持使用简单的 YAML 配置文件来实现各种事件动作，这为用户节省了大量编写 bash 脚本的时间，因此在现代 Python 项目中被广泛使用。在 LinkeFL 项目中，我们也采用 pre-commit 来管理各种 Git Hook，主要用于实现在用户提交 commit 之前对 Python 代码格式化，以及在 push 之前校验本地 Cython 编译结果。

为了使用 pre-commit，首先需要安装对应的 pip 依赖：
``` shell
pip3 install pre-commit
pip3 install commit-msg-hook
```
接着在 LinkeFL 根目录下，依次执行下列命令，将 pre-commit 安装到 LinkeFL 项目的 `.git/hooks/` 中：

``` shell
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```
pre-commit 安装成功后，后续第一次 commit 时会在 LinkeFL 项目下初始化 pre-commit 的执行环境，此过程需要从 GitHub 下载第三方 repo，速度较慢，请耐心等待，同时为了能顺利完成下载，保险起见，**建议开启 VPN**。pre-commit 的初始化过程只会执行一次。


## 2. 开发约定

### 2.1 风格指南

**Python 编码风格**
首先建议阅读 Python 官方提供的 [PEP8](https://peps.python.org/pep-0008/) 代码风格指南，该指南是一份最小指南，提供了编写 Python 代码需要遵循的最基本原则。其它代码规范化工具，如 [Pylint](https://pylint.readthedocs.io/), [yapf](https://github.com/google/yapf), [black](https://black.readthedocs.io/en/stable/) 等，均是在 PEP8 的基础上添加了自定义的代码风格要求。

在 LinkeFL 项目中，我们采用的是 `black` 推荐的编码风格，请开发者认真阅读 [black code style](https://black.readthedocs.io/en/stable/the_black_code_style/index.html). 同时，我们还结合 pre-commit 来实现代码风格的自动化纠正，即每次在 commit 之前，会调用 `black` 纠正代码风格并对 Python 源代码做原地修改。

此外，在 LinkeFL 中我们还结合 pre-commit 使用 [flake8](https://github.com/PyCQA/flake8) 来校验 Python 代码的合规性。flake8 和 black 的主要区别在于：flake8 是一种 `code linter`，会在代码的语义层面对其合法性进行初步校验，并且在校验不通过时提示出错信息，但不会帮用户自动修改代码中的错误，而是需要用户手动修复错误后再重新校验；而 black 只是一个风格纠正的工具，不负责语义层面的代码校验。

在 LinkeFL 中，我们设置 flake8 默认最大行字符数为 `88`，请参照[这篇文章](https://tanjuntao.github.io/2021/01/11/PyCharm%E9%85%8D%E7%BD%AE%E4%BC%98%E5%8C%96/)在 PyCharm 中设置最大行字符数提示。另外，为了在某些情况下禁止 flake8 做代码校验，可通过在行尾添加 `# noqa: <error code> ` 注释来静默 flake8。例如，下面的代码在正常情况下会被 flake8 认为超过了 `88` 最大字符限制，从而报 `E501` 错误。但是 Python 文件路径、URL 等字符串是不建议换行的，所以我们通过在行尾添加 `# noqa: E501` 注释来告诉 flake8 不要校验当前行。

``` python
trainset_path = "/Users/tanjuntao/LinkeFL/linkefl/vfl/data/tabular/give-me-some-credit-active-train.csv"  # noqa: E501
```



**Python 注释风格**
Python docstring（文档字符串）的编写有不同的风格，常用的有 Numpy style 和 Google style，在 LinkeFL 项目中，我们采用的是 Numpy style 文档字符串。为了编写良好的文档字符串，请认真阅读 Numpy 官方提供的[文档字符串风格指南](https://numpydoc.readthedocs.io/en/latest/format.html)。

为了在 PyCharm 中能自动生成 Numpy style 的文档字符串模板，进行如下设置：

```
Preferences => Tools => Python Integrated Tools => Docstrings
```

将 Docstring format 修改为 NumPy，然后点击 Apply，最后点击 OK 即可。

**Git commit message 风格**
在 LinkeFL 中，我们还结合 pre-commit 使用 [commit-msg-hook](https://github.com/dimaka-wix/commit-msg-hook) 来自动校验 commit message 的格式，并且在 commit message 不符合要求时终止 commit。请认真阅读 [commit rules](https://github.com/dimaka-wix/commit-msg-hook#commit-rules) 以便能顺利提交 commit。


### 2.2 分支策略
由于 GitHub 上无法对 master 分支做 push 保护（收费功能），所有 LinkeFL 项目的开发者都有权限向 master 分支 push 代码，所以开发者在往 master 分支提交代码时，请务必要小心慎重。

但为了不引入过多的约束，LinkeFL 中分支管理追求极简策略，主要有如下几点：

* 当需要紧急修复 bug 时，可直接往 master 分支提交
* 当代码变动比较小时，例如 minor change / minor feature 等，在确信不会影响其他模块功能的情况下，可直接往 master 分支提交
* 只有当新功能比较复杂，牵连多个模块，并且有可能破坏其他模块正常功能的情况下，需要开一个新分支。开发者在新分支上完成功能开发和测试后，再提交 Pull Request。新分支的命名建议使用功能名称的缩写如 `tree-plot` 或开发者姓名如 `dev-tanjuntao`


## 3. (Optional) 项目管理

### 3.1 Wheel 包构建
LinkeFL 项目中借助 [setuptools](https://setuptools.pypa.io/) 和 Cython 工具来打包 Python 源码。具体来说，我们使用传统的 `setup.py` 文件和 pip 包管理工具，来调用 setuptools 完成代码打包。尽管 setuptools 官方文档中推荐开发者用 [`pyproject.toml`](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) 配置文件来管理所有的包依赖以及打包配置项，但 LinkeFL 项目在启动时，出于打包方法成熟度以及生态丰富度的考量，还是选择使用 [`setup.py`](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html)。后续开发过程中，如果有开发者能将现有打包代码以及配置项等迁移到 `pyproject.toml` 中，LinkeFL 将采纳 setuptools 的建议，放弃 `setup.py`。另外，Cython 的作用主要是为了源代码保护，因为 setuptools 打包后得到的 Wheel 包，没有对源代码做任何保护，所以我们借助 Cython 将 `.py` 文件编译成 `.so` 二进制文件，再将所有的 `.so` 文件组装到 Wheel 包中，实现源代码的保护。

如果需要在本地从源码构建 Wheel 包（主要是为了 Wheel 包的分发），只需要在 LinkeFL 根目录下执行如下命令即可
``` shell
bash build.sh
```
最终在 `LinkeFL/dist/` 目录下会生成与当前操作系统、当前 Python 版本号对应的 Wheel 包文件，该 Wheel 包可放心分发到其它同等配置的环境中。

此外，为了能在多操作系统、多版本 Python 环境中均构建出 Wheel 包，LinkeFL 借助了 [GitHub Action](https://docs.github.com/en/actions/quickstart)。Github Action 对应的文件存放在 `LinkeFL/.github/workflows/build_wheels.yml`，如果需要修改 Action 的流程，请先认证阅读 Action 的官方文档。

### 3.2 文档构建
LinkeFL 使用 [Sphinx](https://www.sphinx-doc.org/en/master/) 和 [sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/en/latest/) 来构建项目文档以及 API 文档，具体构建过程请参照[文档构建手册](../README.md)。Sphinx 和 autoapi 的所有配置项均保存在 `LinkeFL/docs/source/conf.py` 文件下，修改配置项前，请认真阅读这两个工具官方文档中的配置参数说明。

### 3.3 pre-commit 配置
pre-commit 的所有配置项存放在 LinkeFL 根目录 `.pre-commit-config.yaml` 文件中，如果需要修改、新增、删除 Hook，请先认真阅读 [pre-commit 官方文档](https://pre-commit.com/)。

如果需要临时禁止某个 Hook 的执行，可以使用 `SKIP` 环境变量，例如
``` shell
SKIP=flake8 git commit -m "foo"
```
上述命令会在用户提交 commit 时，跳过 `.pre-commit-config.yaml` 文件中 id 为 `flake8` 的 Hook 的执行。
