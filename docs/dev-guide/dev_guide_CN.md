# LinkeFL 项目开发指南 - 必读

## 1. 本地开发环境准备

### 1.1 以开发模式安装 LinkeFL


第一次安装的时候需要利用 Cython 编译每个 Python 文件，所以会比较慢，耐心等待

### 1.2 安装 pre-commit



``` shell
pip3 install pre-commit
pip3 install commit-msg-hook
```



第一次在本地运行的时候，会下载所有 repos 里面的 hook，会比较慢，后面就好了

``` shell
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```




*最后一次修改时间：





Update pre-commit config

- Add pre-commit buildin hooks for file formatting before git commit
- Add commit-msg-hook to format commit message before git commit-msg
- Add local hook to check Cython built wheels before git push
- Use `os.cpu_count()` for parallel Cython compilation
