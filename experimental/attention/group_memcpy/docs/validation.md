# 验证记录

## Torch 接口目录迁移复核（2026-11-04）

Torch 接口改为算子本地 `torch_extension/`，使用公共 OpBuilder 发现与加载机制；公共扩展恢复为 GroupMemcpy 引入前版本。本次 Windows 环境已通过 7 项源码契约检查、Python 语法检查与 diff 空白检查。迁移后 C++ 编译及 NPU 运行尚未执行，不能沿用历史设备测试结论作为此次迁移的验收结果。

## PR 提交前复核（2026-10-30）

本次在 Windows 环境重新执行 `python experimental/attention/group_memcpy/tests/host/test_source_contract.py`，7 项全部通过；相对源仓库 master 的 `git diff --check` 通过。这些结果只证明源码、接口注册和文档的静态一致性。

本机未提供 CANN/NPU、CMake 或 C++ 编译器，因此本次没有复跑 CANN 编译、CPU C++ Tiling、aclnn C++ 或 Torch NPU 测试，也未完成二级冒烟、性能评测或原直调二进制回放。上述项目需在适配的 Linux/NPU 环境及社区 CI 中验收。

## 历史 Linux / Ascend 910B3 验证记录（2026-10-28）

已使用 CANN 9.1 和配套 Linux/NPU 环境完成完整算子包及 Torch 扩展编译安装。CPU 源码契约 7 项、CPU Tiling 1 项、aclnn C++ 的 141 组逐字节用例及非法参数检查、Torch NPU pytest 150 项全部通过，无跳过。

以上为此前保留的验证结论；原始 Linux 日志未包含在本次提交中，本次 Windows 复核无法独立确认这些设备测试结果。历史记录涉及 CANN 9.1 API/头文件兼容、TilingParse 注册、连续切片的 ACL storage 描述等修复；独立打包入口使用 CANN 自带 ASC 框架，无需下载 cann-cmake。可复现命令见 [测试说明](../tests/README.md)。未提供原直调输入/输出 bin，因此原二进制回放仍未执行。

下文保留早期 Windows 静态验证记录。

日期：2026-10-27。早期 Windows 静态验证使用了独立工作区和验证入口。

## 当前环境

Windows PowerShell，Python 3.9。未找到 `msopgen`、`bisheng`、`npu-smi`、`cmake`、`g++`、`clang++`、`cl`；当前 Python 未安装 `torch`，没有可用的 CANN/NPU 验证环境。

## 执行结果

已运行 `python tests/host/test_source_contract.py`：7 项检查全部通过。`git diff --check` 通过。它们只提供源码/集成静态一致性证据；CANN 编译、安装、C++ 运行和 NPU pytest 均未执行，不能记为通过。

| 检查 | 状态 |
| --- | --- |
| 源码契约、JSON/OpDef/binary 及 Torch 注册一致性 | 通过：7 项标准库 unittest 检查中的相关项 |
| Python 源码语法与本次文档链接 | 通过：AST 解析及相对链接存在性检查 |
| git diff 空白检查 | 通过；新增文件另做空白/编码检查 |
| 生产 C++ Tiling 策略运行 | 未执行：没有 C++ 编译器/CMake |
| msopgen 工程再生成核对 | 未执行：没有 msopgen |
| CANN Host/Kernel/opapi 编译与安装 | 未执行：没有 CANN Linux 环境 |
| aclnn C++ 的 141 组正常调用及非法参数测试 | 已提供代码，未上板执行 |
| Torch 精确字节、stream/内存生命周期、Meta 等测试 | 已提供代码，未上板执行 |
| 原直调输入/输出二进制一致性 | 未执行：原目录没有提供基线 bin |

## Linux 验收顺序

1. 检查 NPU 型号及配套 CANN、PyTorch/torch_npu。当前只配置 ascend910b，不自动宣称其他芯片支持。
2. 可在独立验证目录中运行 `msopgen gen -i /path/to/ops-transformer/experimental/attention/group_memcpy/group_memcpy.json -c ai_core-ascend910b -lan cpp -out ./generated`，核对生成的接口描述。该命令只生成骨架，不会自动替代仓库中手写的 TensorList aclnn API、Torch 集成和 Kernel。当前未生成该目录。
3. 按 [构建说明](../README.md) 使用仓库标准 `build.sh --pkg --experimental` 流程构建并安装实际生成的包，保存真实编译和安装日志。
4. 执行 CPU C++ Tiling 检查、aclnn C++ 测试以及 Torch pytest；参见 [测试命令](../tests/README.md)。NPU 存在时，缺少接口/注册必须算失败。
5. 从可运行的原直调工程导出真实输入及输出到独立验证目录，用 `opapi_test --replay` 逐字节核对；保留原始输出，不使用容差放行。

当前仍待目标环境验证的重点包括：所用 CANN 版本的动态输出 TensorList launcher 与视图 offset 行为，Tail DMA/padding 与流水同步，torch_npu 队列提交及 allocator stream 记录接口。它们已有对应测试入口，但尚无实际编译或设备执行证据。
