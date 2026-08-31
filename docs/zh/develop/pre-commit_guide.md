# Pre-commit 配置指导书

## 一、概述

pre-commit 是一个 Git Hooks 框架，在 `git commit` 时自动运行代码检查和格式化，提前拦截规范问题，避免远程 CI 门禁失败。

| Hook | 功能 |
|------|------|
| **pre-commit-hooks** | 行尾空格、文件末尾换行、YAML/JSON 合法性、大文件、合并冲突标记、私钥检测 |
| **clang-format** | C/C++/asc 代码格式化，遵循 `.clang-format` |
| **ruff-check / ruff-format** | Python 静态检查（自动修复）+ 格式化 |
| **codespell** | 拼写检查（CANN/ascend 等术语已加白名单） |
| **OAT Check** | 开源合规检查（许可证头、二进制/归档文件拦截） |

## 二、环境要求

- **Git**: 2.0+，**Python**: 3.9+，**pre-commit**: 4.0+

> clang-format / ruff / codespell 由 pre-commit 首次运行时自动下载（版本见 `.pre-commit-config.yaml`）；OAT 首次运行时自动 `pip install oat-py`。均无需手动安装。

## 三、安装配置

```bash
# 1. 安装 pre-commit
pip3 install pre-commit

# 2. 安装 Git Hooks
cd /path/to/ops-transformer
pre-commit install        # 取消：pre-commit uninstall

# 3. 配置 git pc 别名（对指定范围提交运行检查，每个环境执行一次）
git config --global alias.pc '!f() { pre-commit run --files $(git diff --name-only "$@"); }; f'
git pc HEAD~x        # 检查最近x笔提交
```

## 四、日常使用

```bash
# 提交时自动检查
git add . && git commit -m "msg"

# 手动运行
pre-commit run                        # 暂存区
pre-commit run --all-files            # 所有文件
pre-commit run clang-format           # 单个hook

# 检查指定文件
pre-commit run --files src/foo.py src/bar.cpp

# 检查指定目录（pre-commit 不支持目录参数，需用 find 展开成文件列表）
find examples -type f | xargs pre-commit run --files

# 跳过（紧急情况）
git commit --no-verify -m "msg"
```

## 五、失败排查

搜索输出中的 `Failed` 定位失败项，查看该 hook 下方输出处理，或粘贴给 AI 获取修复建议。

| 报错                  | 原因               | 处理方式                                          |
|-----------------------|--------------------|---------------------------------------------------|
| clang-format Failed   | 代码存在规范问题 | rebase 最新代码后再做 pre-commit；重新 `git add` 后再次 commit |
| OAT Compliance Failed | 缺版权声明等       | 搜索 `OAT Scan Result Summary` 定位文件；版权声明无法自动订正，需参照仓内文件头手动补齐 |

## 六、补查历史提交（--no-verify 跳过后）

对指定范围的提交变更文件补跑检查，等价于 CI `codecheck_precommit` 门禁：

```bash
git pc HEAD~x        # 最近x笔提交
git pc <commit>^     # 某一笔提交
```

> 部分钩子会自动修复文件，修复后需重新 `git add` 并追加提交。

## 七、检查项说明

1. **pre-commit-hooks** (v4.6.0)：trailing-whitespace、end-of-file-fixer、check-yaml/json、check-added-large-files、check-merge-conflict、detect-private-key
2. **clang-format** (v18.1.8)：遵循项目 `.clang-format`（Google 风格，4 空格缩进，不限列宽不自动拆行，枚举逐行，构造函数初始化列表逐行换行，函数定义大括号换行，指针右对齐 `int *ptr`）
3. **ruff** (v0.14.14)：ruff-check（`--fix` 自动修复）+ ruff-format
4. **codespell** (v2.4.1)：拼写检查，CANN、ascend、EnQue 等术语已加白名单
5. **OAT**：基于 oat-py，检查许可证头（YAML/CSV 已豁免）、禁止二进制和归档文件

## 八、常见问题

**Q1: 首次提交 OAT 检查很慢？**

首次运行需 `pip install oat-py`，属正常现象，后续很快。

**Q2: 手动全量格式化 C/C++？**

```bash
bash scripts/format_cpp.sh            # 整个仓库
bash scripts/format_cpp.sh examples   # 指定目录
```

自动排除 `build/`、`build_out/`、`third_party/`、`.git/`。

**Q3: 如何全局生效，让新 clone 的仓库自动继承？**

```bash
mkdir -p ~/.git-templates
pre-commit init-templatedir ~/.git-templates
git config --global init.templatedir ~/.git-templates

# 已有仓库需重新拷贝模板
cd /path/to/ops-transformer && git init
```

> hook 只是启动器，实际检查仍读取当前仓库的 `.pre-commit-config.yaml`；未配置该文件的仓库会自动跳过，互不影响。

取消：`git config --global --unset init.templatedir && rm -rf ~/.git-templates`

**Q4: 何时需要安装本地 clang-format？**

仅手动批量格式化（`bash scripts/format_cpp.sh`）时需要，建议 18.x；日常 `git commit` 触发的 pre-commit 不依赖此工具。

```bash
sudo apt install clang-format
```

## 相关文档

- [pre-commit官方文档](https://pre-commit.com/)
- [clang-format配置](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT工具](https://gitcode.com/openharmony-sig/tools_oat)
- [代码仓集成pre-commit指导](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
