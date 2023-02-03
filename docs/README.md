# 如何本地构建文档和编写文档

## 1. 构建文档

### 1.1 安装依赖

LinkeFL 使用广泛流行的自动化文档构建工具 [`Sphinx`](https://www.sphinx-doc.org/) 来构建用户使用说明文档和 API 文档，为了实现在用户本地机器上构建文档，需要首先安装相关 Python 第三方包。

切换到 `LinkeFL/docs` 目录下，执行

``` shell
pip3 install -r doc_requirements.txt
```

用于安装 Sphinx 相关的构建工具以及 Sphinx 相关的文档主题。

### 1.2 本地构建

同样在 `LinkeFL/docs` 目录下，执行

``` shell
make html
```

该命令将会执行文档构建程序，最终会在当前目录下生成 `build` 目录。

接着进入 `LinkeFL/docs/build/html` 目录，找到 `index.html` 文件，该文件是整个项目文档的入口页面，使用系统中任意浏览器打开，即可浏览到当前项目所有构建完成的文档。

## 2. 编写文档

Python docstring（文档字符串）的编写有不同的风格，常用的有 Numpy style 和 Google style，在 LinkeFL 项目中，我们采用的是 Numpy style 文档字符串。

### 2.1 风格标准

为了编写良好的文档字符串，请认真阅读 Numpy 官方提供的[风格指南](https://numpydoc.readthedocs.io/en/latest/format.html)。

### 2.2 PyCharm 设置

为了在 PyCharm 中能自动生成 Numpy style 的文档字符串模板，进行如下设置：

```
Preferences => Tools => Python Integrated Tools => Docstrings
```

将 Docstring format 修改为 NumPy，然后点击 Apply，最后点击 OK。
