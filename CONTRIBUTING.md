# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请参见 [cann-community](https://gitcode.com/cann/community) 了解行为准则，完成 CLA 协议签署，并熟悉源码仓的贡献流程。

## 准备工作

开发者在准备本地代码与提交 PR 时，需重点关注以下事项：

1. 提交 PR 时，请按照 PR 模板仔细填写本次 PR 的业务背景、目的、方案等信息。
2. 若您的修改涉及新增特性、新增接口、新增配置参数或修改代码流程等（非简单 bug 修复），请务必先通过 Issue 进行方案讨论，以避免代码被拒绝合入。若不确定修改是否属于"简单 bug 修复"，建议提交 Issue 进行讨论。

## 贡献场景

开发者贡献场景主要包括：

- [贡献新算子](#一贡献新算子)
- [算子 Bug 修复](#二算子-bug-修复)
- [算子优化](#三算子优化)
- [文档纠错](#四文档纠错)
- [帮助解决他人 Issue](#五帮助解决他人-issue)

---

## 一、贡献新算子

算子开发贡献流程如下：

![算子开发贡献流程](./docs/zh/figures/op_dev_and_contribute_flow.png "算子开发贡献流程图")

如果您有全新的算子希望基于 NPU 设计与实现，欢迎在 Issue 中提出您的想法与设计方案。完整的贡献过程如下：

### 1. 创建 Issue 需求

新建 `Requirement|需求建议` 类 Issue，并阐明新增算子的设计方案。Issue 一般需包含以下内容：

- **背景信息**
- **价值/作用**
- **设计方案**

具体操作步骤请参阅：[Issue 操作指南](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md)。

### 2. 需求评审

创建需求类 Issue 后，需申报 SIG 议题，并在指定时间参加 Ops-transformer SIG 例会评审。

- [会议议题&纪要](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-transformer)
- [会议链接](https://meeting.osinfra.cn/cann/)

**紧急需求处理：**

若需求紧急，可申请临时 SIG 会议：

1. 填写议题申请：[议题申请](https://etherpad-cann.meeting.osinfra.cn/p/sig-ops-transformer)
2. 发送邮件给 [maintainer](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-transformer/README.md)，建议邮件主题注明：申请临时 SIG 议题及议题内容

**需求接纳：**

若需求被接纳，[SIG 成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-transformer/README.md)将为您分配合适的算子分类路径（如：`experimental/attention`），请将贡献算子提交至 `experimental` 对应算子分类目录。

### 3. PR 提交

#### 生态最简算子交付件

```text
${op_class}                    # 算子分类
├── ${op_name}                 # 算子名
│   ├── ${op_name}.cpp         # 算子 Kernel 实现文件
│   ├── tests                  # 测试目录
│   │   └── test_${op_name}.py # 算子测试文件
│   ├── CMakeLists.txt         # 算子编译配置文件
│   └── README.md              # 算子 README 文档
```

#### PR 提交须知

**提交前准备：**

- 建议在需求评审通过后，再提交 PR。
- 提交 PR 前需完成开发环境准备，并仔细了解项目特定的开发规范和版权声明要求（如涉及开源代码片段引用，请参考片段引用指导），确保您的贡献符合项目标准，签署 CLA。具体操作步骤请参阅：[PR 操作指南](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md)。

**交付件要求：**

| 类型 | 要求 | 参考资料 |
|------|------|---------|
| 代码交付件 | 需提供算子 Kernel 实现、算子测试文件 | [fast_kernel_launch_example](examples/fast_kernel_launch_example/README.md) |
| 文档交付件 | 算子 README 文档为必选，其余文档可视情况提供 | [文档贡献指南](docs/CONTRIBUTING_DOCS.md) |
| 精度要求 | 新贡献算子需满足精度标准 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |

**合规检查：**

- [ ] 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
- [ ] 代码是否编译通过
- [ ] Markdown 文档语法是否符合规范
- [ ] 使用 git 进行代码提交前，可以参考 [pre-commit 工具使用说明](docs/zh/develop/pre-commit_guide.md)来使您的代码提交更合规高效

**提交规范：**

- **贡献目录**：按 SIG 成员意见提交至指定目录 `experimental/${op_class}`，可参考已有算子文件放置规则。
- **PR 提交**：通过 `git` 命令提交目标分支 PR，检查 PR 标题是否清晰、PR 描述是否规范（指明更改内容和原因、是否关联对应 Issue）、是否签署 CLA。

> **注意**：如果您希望贡献项目标准算子，其交付件比生态算子更丰富，包括 Kernel、Tiling 实现等，贡献指导可参考 [附录](#附录)。

### 4. CI 门禁

通过评论 `compile` 指令触发开源仓门禁，并依据 CI 检测结果进行修改。目前 CI 门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及 codecheck 误报，请提交给 SIG 成员屏蔽）
- UT 测试
- 冒烟测试

门禁通过后，请在关联的 Issue 中 @ Committer。

### 5. Committer 检视

Committer 检视后将反馈检视意见，请根据意见修改，完成后 @ Committer。

### 6. Maintainer 合入

Committer 检视通过后，将标注 `/lgtm` 标签。Maintainer 最终审核，确认无问题后，将标注 `/approve` 标签合入 PR。

---

## 二、算子 Bug 修复

如果您在本项目中发现了某些算子 Bug，希望对其进行修复，欢迎您新建 Issue 进行反馈和跟踪处理。

您可以按照 [提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引新建 `Bug-Report|缺陷反馈` 类 Issue 对 Bug 进行描述。

---

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强或性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

您可以按照 [提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引新建 `Requirement|需求建议` 类 Issue 对优化点进行说明，并提供您的设计方案。

---

## 四、文档纠错

如果您在本项目中发现某些算子文档描述错误，欢迎您新建 Issue 进行反馈和修复，文档规范参考 [文档贡献指南](docs/CONTRIBUTING_DOCS.md)。

您可以按照 [提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务) 指引新建 `Documentation|文档反馈` 类 Issue 指出对应文档的问题。

---

## 五、帮助解决他人 Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在 Issue 中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应 Issue 需要进行代码修改，您可以在 Issue 评论框中输入 `/assign` 或 `/assign @yourself`，将该 Issue 分配给您，跟踪协助解决问题。

---

## 附录

### 项目标准算子交付件

```text
${op_class}                           # 算子分类
├── ${op_name}                        # 算子名
│   ├── op_host                       # 算子定义、Tiling 相关实现
│   │   ├── ${op_name}_def.cpp        # 算子定义文件
│   │   ├── ${op_name}_tiling.cpp     # 算子 Tiling 实现文件
│   │   ├── ${op_name}_tiling_${sub_case}.cpp  # 可选，子场景 Tiling 实现，${sub_case} 表示子场景（如 arch35）
│   │   ├── ${op_name}_tiling_${sub_case}.h    # 可选，子场景 Tiling 实现头文件
│   │   └── CMakeLists.txt
│   ├── op_kernel                     # 算子 Kernel 目录
│   │   ├── ${op_name}.cpp            # Kernel 入口文件，包含主函数和调度逻辑
│   │   ├── ${op_name}.h              # Kernel 实现文件，定义 Kernel 头文件，包含函数说明、结构定义、逻辑实现
│   │   ├── ${op_name}_tiling_data.h  # TilingData 文件，存储 Tiling 策略相关配置信息
│   │   └── ${op_name}_tiling_key.h   # TilingKey 文件，定义 Tiling 策略的 key，标识不同划分方式
│   ├── CMakeLists.txt                # 算子编译配置文件，保留原文件即可
│   ├── README.md                     # 算子说明文档
│   └── tests                         # 算子测试文件
│       └── ut                        # 算子 UT 测试文件
```

完整的算子目录结构（含各可选交付件）参见[项目目录结构](docs/zh/install/dir_structure.md#项目目录)。

> **说明**：op_host 目录下参与编译的 Tiling 实现文件，文件名须包含 `_tiling` 标识（如 `${op_name}_tiling.cpp`、`${op_name}_tiling_${sub_case}.cpp`），否则不会被编译系统识别。针对子场景（如特定架构 arch35）拆分 Tiling 实现时，请遵循此命名规则。

### PR 上库要求

**交付件要求：**

| 类型 | 要求 | 参考资料 |
|------|------|---------|
| 代码交付件 | 需提供 op_host 算子 Tiling 实现、op_kernel 算子 Kernel 实现、算子 UT 测试文件 | [算子开发指南](docs/zh/develop/aicore_develop_guide.md) |
| 文档交付件 | 算子 README 文档为必选，其余文档可视情况提供 | [文档贡献指南](docs/CONTRIBUTING_DOCS.md) |

**合规检查：**

- [ ] 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
- [ ] 是否符合标准算子基础编程规范
- [ ] 代码是否编译通过
- [ ] Markdown 文档语法是否符合规范

**提交规范：**

- **贡献目录**：按 SIG 成员意见提交至指定目录 `experimental/${op_class}`，可参考已有算子文件放置规则。
- **PR 提交**：通过 `git` 命令提交目标分支 PR，检查 PR 标题是否清晰、PR 描述是否规范（指明更改内容和原因、是否关联对应 Issue）、是否签署 CLA。
