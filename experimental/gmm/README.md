# Group Matmul（Experimental）

提供四个分组矩阵乘法算子及三个 Torch 接口，覆盖独立分组输出、原地累加和本地专家计算。

## 目录和接口

| 目录 | ACLNN | Torch 接口及行为 |
| --- | --- | --- |
| `gmm_k_dim` | `aclnnGmmKDim` | `torch.ops.cann_ops_transformer.gmm(..., c=None)`，沿 K 分组，各组独立输出 |
| `gmm_add` | `aclnnGmmAdd` | `torch.ops.cann_ops_transformer.gmm(..., c=c)`，原地执行 `c[g] += A_g @ B_g` |
| `gmm_local_exp` | `aclnnGmmLocalExp` | `torch.ops.cann_ops_transformer.local_exp_gmm(...)`，计算指定本地专家范围 |
| `gmm_local_exp_with_zero` | `aclnnGmmLocalExpWithZero` | `torch.ops.cann_ops_transformer.local_exp_gmm_with_zero(...)`，同时清零范围外输出 |

每个算子具有 `op_host/` 和 `op_kernel/`。`gmm` 的 Torch 适配位于
`gmm_k_dim/torch_extension/`，另两个接口位于各自的 `torch_extension/`。
公共 Host 工具在 `common/tiling_utils.h`，测试在 `tests/test_group_matmul.py`。

输入支持 FP16/BF16，输出可以保持输入类型或使用 FP32。
`batch_sizes` / `problemList` 是每组长度，**不是累积长度**，支持 int32/int64 NPU 张量，
接口内转换为 int32。调用方须保证长度非负且总和匹配被分组的维度。

`gmm` 中 `a` 为 `[M, TotalK]`（`trans_a=False`）或 `[TotalK, M]`
（`trans_a=True`），`b` 为 `[TotalK, N]`，输出为 `[G, M, N]`。
传入 `c` 时结果使用 `c` 的类型；`type_promotion` 只控制新分配的输出。

本地专家接口要求 `a=[TotalM,K]`、`b=[LocalExperts,N,K]`（`trans_b=True`）
或 `[LocalExperts,K,N]`（`trans_b=False`），`trans_a=False`。
普通 `local_exp_gmm` 未定义非本地专家行的内容；需要完整输出清零语义时使用 `with_zero`。
测试的 NZ 路径使用手工打包：
`w.reshape(E,N,K//16,16).permute(0,2,1,3).contiguous().reshape(E,N,K)`，
传入 `trans_b=True, is_b_nz=True`，验证用例的 K/N 均为 16 的倍数。

## 实现说明

- 每个算子独立组织内核文件；公共 `grouped_matmul_utils.h` 放在各内核目录中，便于单独构建。
- Host 层按 `*_def.cpp`、`*_tiling.cpp`、`*_infershape.cpp` 组织。
- Torch 层使用仓库的 `OpBuilder` / `ACLNN_CMD`，并提供 Meta 注册。
- `gmm` 按 `trans_a` 选择输出的 M 维；本地专家接口要求三维权重和 `trans_a=False`。
- Torch schema 标注 `c` 的原地修改和返回别名，并校验设备、类型、形状与专家范围。
- 独立打包入口调用 `experimental/tools/build_torch_vendor.py`，仅在临时构建副本中处理 vendor 相对导入，公共框架源码保持不变。

## 已验证环境

| 项目 | 值 |
| --- | --- |
| NPU | Ascend 910B2C |
| 驱动 / npu-smi | 25.0.rc1.1 |
| CANN | `/usr/local/Ascend/cann` → `cann-8.5.0` |
| 编译器 | GCC 10.3.1，CMake 3.22.0 |
| Python | 3.11.13 |
| Torch / torch_npu | 2.9.0 / 2.9.0 |

已验证的构建入口为 `tools/standalone/CMakeLists.txt`，使用 CANN 自带的
`ascendc_kernel_cmake`。四个算子的常规仓库 CMake 入口已接入，但未验证整仓构建。
SoC 构建参数支持 `ascend910b` / `ascend910_93`，已验证前者。

## 编译

在仓库根目录执行：

```bash
bash experimental/gmm/tools/build_standalone.sh /workspace/group_matmul_build ascend910b
bash experimental/gmm/tools/build_torch.sh \
  /workspace/group_matmul_python codex_gmm /workspace/group_matmul_wheels
```

独立构建生成四个 ACLNN 接口及对应内核二进制，安装到
`<BUILD_DIR>/stage/packages/vendors/experimental_group_matmul`，并生成
`<BUILD_DIR>/package/custom_opp_*.run`。

Torch 打包选择 `gmm_k_dim,gmm_local_exp,gmm_local_exp_with_zero`，wheel 保存到
第三个参数指定的目录，并用 `pip --target` 安装到第一个参数指定的目录。
第三个参数省略时，wheel 保存到安装目录父目录下的 `wheels/`。
打包过程自动清理临时目录，不修改公共框架源码，也不在源码目录创建 build/dist、egg-info 或 vendor 链接。
C++ 桥接在首次调用时 JIT 编译，默认缓存位于
`<BUILD_DIR>/torch_extensions/{gmm,local_exp_gmm,local_exp_gmm_with_zero}`。

构建工具将四个算子的内核文件收集到构建目录，再创建一个 kernel library，
避免 CANN 8.5 在同一 CMake 目录创建多个 source-copy target 时发生目标冲突。

## 执行

在仓库根目录执行，目录与前面的构建示例对应：

```bash
NPU_DEVICE=npu:11 bash experimental/gmm/tools/run_tests.sh \
  /workspace/group_matmul_build /workspace/group_matmul_python \
  cann_ops_transformer_codex_gmm
```

根据可用设备调整 `NPU_DEVICE`。运行脚本加载 CANN 环境，设置 `ASCEND_CUSTOM_OPP_PATH`、
`LD_LIBRARY_PATH`、`PYTHONPATH` 和 `CANN_OPS_TRANSFORMER_PACKAGE`。
`PYTHONPATH` 应指向 wheel 安装目录，而不是其中的包目录；脚本已自动设置。

Python 调用示例（在上述环境变量已设置的进程中）：

```python
import torch
import torch_npu
import cann_ops_transformer_codex_gmm

torch_npu.npu.set_device("npu:11")
groups = torch.tensor([32, 0, 48, 16], dtype=torch.int32, device="npu:11")
a = torch.randn(64, 96, dtype=torch.bfloat16, device="npu:11")
b = torch.randn(96, 96, dtype=torch.bfloat16, device="npu:11")
y = torch.ops.cann_ops_transformer.gmm(a, b, groups, type_promotion=True)
assert y.shape == (4, 64, 96)
torch.npu.synchronize()
```

## 测试结果

CPU 使用从输入 FP16/BF16 转成的 FP32 张量计算逐组矩阵乘，结果再转换到输出类型后比较。
固定随机种子 `20260924`，随机输入乘以 `0.25`。
所有元素必须满足 `torch.testing.assert_close`，不允许通过坏点比例放宽判定。

| 输出类型 | rtol | atol |
| --- | --- | --- |
| FP16 | 0.001 | 0.002 |
| BF16 | 0.01 | 0.02 |
| FP32 | 0.001 | 0.001 |

| 算子 | 用例数 | 最大绝对误差（相对转换到输出类型的参考） | 结果 |
| --- | --- | --- | --- |
| GmmKDim | 8 | 0.00048828125 | 全部通过 |
| GmmAdd | 8 | 0.0078125 | 全部通过 |
| GmmLocalExp | 12 | 0.00048828125 | 全部通过 |
| GmmLocalExpWithZero | 12 | 0.00048828125 | 全部通过 |

K 分组用例使用 `[32,0,48,16]`，`M=64,N=96`，覆盖 `trans_a=False/True`、
FP16/BF16 输入、同类型/FP32 输出，检查零长度组和 `c` 返回地址。
本地专家用例使用 `[16,32,0,48,16]`、专家范围 `[1,4)`、`K=64,N=96`，
覆盖 ND 的两种转置方式、手工 NZ、两种输入类型和两种输出类型；`with_zero` 还检查范围外行严格为零。
Meta 的形状、类型和别名检查通过。

最终输出：`OVERALL PASS: 40 NPU cases; peak_allocated=28.22 MiB`，进程退出码 0。
该显存数值是 PyTorch 记录的分配峰值，不包含驱动与运行时全部显存占用。
验证仅覆盖上述小规模用例，未进行大规模压力测试或性能测量。
