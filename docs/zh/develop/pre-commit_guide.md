# Pre-commit 配置指导书

## 概述

pre-commit是一个Git Hooks框架，用于在`git commit`时自动运行代码检查和格式化工具。本项目已配置以下检查：

| Hook | 功能 | 说明 |
|------|------|------|
| **pre-commit-hooks** | 基础规范检查 | 行尾空格、文件末尾换行、YAML/JSON 合法性、大文件、合并冲突标记、私钥检测 |
| **clang-format** | C/C++ 代码格式化 | 自动格式化 C/C++/asc 代码，遵循 `.clang-format`（不限列宽，不自动拆行） |
| **ruff-check / ruff-format** | Python 检查与格式化 | ruff 静态检查（自动修复）+ 代码格式化 |
| **codespell** | 拼写检查 | 检测常见拼写错误（CANN/ascend 等术语已加入白名单） |
| **OAT Check** | 开源合规检查 | 检测许可证头、禁止二进制/归档文件提交（基于 oat-py，Python 实现） |

## 环境要求

- **Git**: 2.0+
- **Python**: 3.9+ (pre-commit 4.0+ 依赖；OAT 检查需 3.7+，已被 3.9+ 覆盖)
- **pre-commit**: 4.0+ (本项目 `minimum_pre_commit_version` 要求)

> **工具获取方式说明**：本项目的检查工具分两类，获取方式不同：
> - **pre-commit 自动拉取**（clang-format / ruff / codespell）：首次运行 `pre-commit run` 时，pre-commit 按 `.pre-commit-config.yaml` 中 `rev` 锁定的版本（clang-format v18.1.8、ruff v0.14.14、codespell v2.4.1）自动下载到隔离环境，与系统装的工具互不影响。无需手动安装。
> - **脚本自动安装**（OAT / oat-py）：OAT 为 local hook，首次运行 `scripts/oat_check.sh` 时自动通过 `pip install oat-py>=1.0.1` 安装（带文件锁防并发冲突）。需系统有 Python 3.7+。
>
> 唯一需要手动安装的是 **clang-format**，仅当本地手动运行 `scripts/format_cpp.sh` 时用到（建议装 18.x 以与 pre-commit 拉取的 v18.1.8 保持一致）。

## 安装步骤

注：若需要使用pre-commit提供的代码检查功能，需要按照如下步骤进行安装配置；若无需pre-commit提供的代码检查功能，可不安装。

### 1. 安装pre-commit

```bash
# 方式一: 使用pip
pip3 install pre-commit

# 方式二: 使用系统包管理器(Ubuntu/Debian)
sudo apt install pre-commit
```

### 2. （可选）安装本地 clang-format

仅当需要脱离 pre-commit 手动批量格式化（`bash scripts/format_cpp.sh`）时安装。日常 `git commit` 触发的 pre-commit 不依赖此工具。

```bash
# Ubuntu/Debian
sudo apt install clang-format

# macOS
brew install clang-format
```

### 3. 项目路径下安装Git Hooks

```bash
cd /path/to/ops-transformer
pre-commit install
```

安装成功后会显示：

```bash
pre-commit installed at .git/hooks/pre-commit
```

## 使用方法

### 自动检查（推荐）

每次执行`git commit`时，pre-commit会自动运行检查：

```bash
git add .
git commit -m "your commit message"
```

输出示例：

```text
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml.................................................................Passed
check for added large files..............................................Passed
check for merge conflicts................................................Passed
detect private key.......................................................Passed
check json................................................................Passed
clang-format.............................................................Passed
ruff-check...............................................................Passed
ruff-format..............................................................Passed
codespell.................................................................Passed
OAT Compliance Check.....................................................Passed
```

### 手动运行检查

```bash
# 运行所有检查
pre-commit run

# 运行特定类型检查
pre-commit run clang-format
pre-commit run oat-check

# 检查所有文件（不限于暂存区）
pre-commit run --all-files

# 检查指定目录（pre-commit 不支持目录参数，需用 find 展开成文件列表）
# 以 examples 目录为例：
find examples -type f | xargs pre-commit run --files
```

### 跳过检查（紧急情况）

```bash
git commit --no-verify -m "emergency fix"
```

> **注意**: 仅在紧急情况下使用，正常开发流程应保证检查通过。

## 检查项说明

### 1. pre-commit-hooks 基础规范

由 `pre-commit/pre-commit-hooks` (v4.6.0) 提供，包含：

- `trailing-whitespace`：清理行尾空格
- `end-of-file-fixer`：确保文件末尾有换行
- `check-yaml` / `check-json`：校验配置文件语法
- `check-added-large-files`：阻止大文件提交
- `check-merge-conflict`：检测遗留的合并冲突标记
- `detect-private-key`：检测私钥误提交

### 2. clang-format

自动格式化 C/C++/asc 代码，遵循项目 `.clang-format` 配置（基于 Google 风格）：

- **缩进**: 4 空格
- **列宽**: 不限制（`ColumnLimit: 0`，既不自动拆行也不合并已有换行，换行由开发者自行控制）
- **枚举**: 短枚举不合并成单行（`AllowShortEnumsOnASingleLine: false`），保持逐行
- **构造函数初始化列表**: 逐行换行（`PackConstructorInitializers: Never`），防止不限宽时合并成单行巨码
- **大括号**: 函数定义换行，控制语句同行
- **指针对齐**: 右对齐(`int *ptr`)
- **续行符**: `DontAlign`（`\` 紧贴每行内容，不右对齐，与宏行长度解耦）

### 3. ruff (ruff-check / ruff-format)

由 `ruff-pre-commit` (v0.14.14) 提供，针对 Python 代码：

- `ruff-check`：静态检查并自动修复（`--fix`）
- `ruff-format`：代码格式化

### 4. codespell

由 `codespell` (v2.4.1) 提供，检测常见拼写错误。CANN、ascend、EnQue 等项目术语已加入白名单。

### 5. OAT Compliance Check

OAT (Open Source Audit Tool) 检查开源合规性，基于 Python 版 `oat-py` 实现（无需 Java/Maven）：

| 检查项 | 说明 |
|--------|------|
| 许可证头检查 | 确保源文件包含CANN License头（YAML 配置文件和 CSV 测试用例文件已豁免，见 OAT.xml `defaultPolicyFilter`） |
| 二进制文件检查 | 禁止提交二进制文件 |
| 归档文件检查 | 禁止提交zip/tar等归档文件 |

OAT检查脚本首次运行时会自动：

1. 检测 Python 3.7+ 环境
2. 通过 `pip install oat-py>=1.0.1` 安装 OAT 工具（带文件锁，防止并发 pip 冲突）

## 常见问题

### Q1: 首次提交时OAT检查很慢

**原因**: 首次运行需要通过 `pip install oat-py>=1.0.1` 安装 OAT 工具。

**解决**: 这是正常现象，后续提交会使用已安装的 oat-py，速度会很快。

### 手动全量格式化 C/C++ 代码

仓库提供 `scripts/format_cpp.sh`，可脱离 pre-commit 对任意目录全量格式化，常用于批量整理历史代码或本地预览格式化效果：

```bash
# 格式化整个仓库
bash scripts/format_cpp.sh

# 格式化指定目录
bash scripts/format_cpp.sh examples
```

脚本会自动排除 `build/`、`build_out/`、`third_party/`、`.git/` 目录，按仓库根目录的 `.clang-format` 配置格式化 `.c/.h/.cpp/.hpp/.cc/.hh/.cxx/.hxx` 文件。

## 相关文档

- [pre-commit官方文档](https://pre-commit.com/)
- [clang-format配置](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT工具](https://gitcode.com/openharmony-sig/tools_oat)
- [代码仓集成pre-commit指导](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
