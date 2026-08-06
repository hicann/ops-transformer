# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前，请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

## 一、贡献新算子

算子开发贡献流程如下：

![算子开发贡献流程](./docs/zh/figures/算子开发贡献流程.png "算子开发贡献流程图")

如果您有全新的算子希望基于NPU进行设计与实现，欢迎在Issue中提出您的想法与设计方案。完整的贡献过程如下：

### 1. 创建Issue需求

新建`Requirement|需求建议`类Issue，并阐明新增算子的设计方案。Issue一般需包含以下内容：

- **背景信息**  
- **价值/作用**  
- **设计方案**

请在提交的Issue中评论`/assign @yourself`认领该任务。

### 2. 需求评审

Sig组将指派Committer对您提交的Issue进行评审并反馈修改意见。请在完成修改后，于Issue中@对应Committer。

若需求被接纳，[sig成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-transformer/README.md)将为您分配合适的算子分类路径（如：`experimental/attention`），请将贡献的算子提交至`experimental`对应算子分类目录。

### 3. PR提交  

生态最简算子交付件如下：

```text
${op_class}                                          # 算子分类
├── ${op_name}                                       # 算子名
│   ├── ${op_name}.cpp                               # 算子Kernel实现文件
│   └── tests                                        
│   │   ├── test_${op_name}.py                       # 算子测试文件
│   ├── CMakeLists.txt                               # 算子编译配置文件
│   ├── README.md                                    # 算子README文档
```

PR上库要求：

- 代码交付件：需提供算子Kernel实现、算子测试文件，开发过程参考[fast_kernel_launch_example](examples/fast_kernel_launch_example/README.md)。
- 文档交付件：算子README文档为必选，其余文档可视情况提供。文档写作模板和规范参考[文档贡献指南](docs/CONTRIBUTING_DOCS.md)。
- 精度要求：新贡献算子需满足精度标准，具体请参见[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。
- 合规检查：
  - 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》
  - 代码是否编译通过
  - Markdown文档语法是否符合规范
  - 使用git进行代码提交前，可以参考[pre-commit工具使用说明](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/develop/pre-commit_guide.md)来使您的代码提交更合规高效。
- 贡献目录：按sig成员意见提交至指定目录下`experimental/${op_class}`，可参考已有算子文件放置规则。
- PR提交：通过`git`命令提交目标分支PR，检查PR标题是否清晰、PR描述是否规范（指明更改内容和原因、是否关联对应Issue）、是否签署CLA。

如果您希望贡献项目标准算子，其交付件比生态算子更丰富，包括Kernel、Tiling实现等，贡献指导可参考[附录](#附录)。

### 4. CI门禁

通过评论`compile`指令触发开源仓门禁，并依据CI检测结果进行修改，目前CI门禁包含以下检查项：

- 代码编译
- 静态检查（如涉及codecheck误报，请提交给sig成员屏蔽）
- UT测试
- 冒烟测试

门禁通过后，请在关联的Issue中@指派的Committer。

### 5. Committer检视

Committer检视后将反馈检视意见，请根据意见修改，完成后@指派的Committer。

### 6. Maintainer合入

Committer检视通过后，标注`/lgtm`标签。Maintainer将在1天内进行最终审核，确认无问题后，将标注`/approve`标签合入PR。

## 二、算子Bug修复

如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建`Bug-Report|缺陷反馈`类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建`Requirement|需求建议`类Issue对优化点进行说明，并提供您的设计方案，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行跟踪优化。

## 四、文档纠错

如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复，文档规范参考[文档贡献指南](docs/CONTRIBUTING_DOCS.md)。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建`Documentation|文档反馈`类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。

## 五、帮助解决他人Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。

## 附录

项目标准算子交付件如下：

```text
${op_class}                                          # 算子分类
├── ${op_name}                                       # 算子名
│   ├── op_host                                      # 算子定义、Tiling相关实现
│   │   ├── ${op_name}_def.cpp                       # 算子定义文件
│   │   ├── ${op_name}_tiling.cpp                    # 算子Tiling实现文件
│   │   └── CMakeLists.txt
│   ├── op_kernel                                    # 算子Kernel目录
│   │   ├── ${op_name}.cpp                           # Kernel入口文件，包含主函数和调度逻辑
│   │   ├── ${op_name}.h                             # Kernel实现文件，定义Kernel头文件，包含函数说明、结构定义、逻辑实现
│   │   ├── ${op_name}_tiling_data.h                 # TilingData文件，存储Tiling策略相关配置信息
│   │   └── ${op_name}_tiling_key.h                  # TilingKey文件，定义Tiling策略的key，标识不同划分方式
│   ├── CMakeLists.txt                               # 算子编译配置文件，保留原文件即可
│   └── README.md                                    # 算子说明文档
│   └── tests                                        # 算子测试文件
│   │   ├── ut                                       # 算子UT测试文件
```

PR上库要求：

- 代码交付件：需提供op_host算子Tiling实现、op_kernel算子Kernel实现、算子UT测试文件，开发过程请参考[算子开发指南](docs/zh/develop/aicore_develop_guide.md)。
- 文档交付件：算子README文档为必选，其余文档可视情况提供。文档写作模板和规范请参见[文档贡献指南](docs/CONTRIBUTING_DOCS.md)。
- 合规检查：
  - 代码是否符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》、是否符合标准算子基础编程规范
  - 代码是否编译通过
  - Markdown文档语法是否符合规范
- 贡献目录：按sig成员意见提交至指定目录下`experimental/${op_class}`，可参考已有算子文件放置规则。
- PR提交：通过`git`命令提交目标分支PR，检查PR标题是否清晰、PR描述是否规范（指明更改内容和原因、是否关联对应Issue）、是否签署CLA。
