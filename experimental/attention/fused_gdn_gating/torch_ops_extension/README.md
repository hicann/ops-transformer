# fused_gdn_gating — PyTorch 接入层 (TorchNPU/ torch_ops_extension)

将 `fused_gdn_gating` 算子桥接到 PyTorch，注册为 `torch.ops.custom.npu_fused_gdn_gating`
（同时挂载到 `TorchNPU.npu_fused_gdn_gating`），底层调用 ACLNN 接口 `aclnnFusedGdnGating`。

> 结构与配置对齐参考实现 `quant_block_sparse_attn/torch_ops_extension`：编译式 `NpuExtension`
> （`custom_ops.custom_ops_lib`）+ `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` + `EXEC_NPU_CMD_V1`。
> `csrc/ops_common.h`、`csrc/ops_common.cpp` 为该参考的整文件拷贝（自包含）。

## 目录结构

```
torch_ops_extension/
├── setup.py                    # NpuExtension custom_ops.custom_ops_lib，编译 csrc/*.cpp
├── build_and_install.sh        # 构建 wheel 并 pip 安装
├── README.md
└── custom_ops/
    ├── __init__.py             # 导入 custom_ops_lib(.so) 并挂载到 TorchNPU
    ├── csrc/
    │   ├── ops_def_registration.cpp     # TORCH_LIBRARY(custom): npu_fused_gdn_gating schema
    │   ├── ops_common.h / ops_common.cpp # EXEC_NPU_CMD_V1 / ConvertType 基础设施（拷贝）
    │   └── npu_fused_gdn_gating.cpp     # NPU/Meta 前向实现 + TORCH_LIBRARY_IMPL
    └── converter/
        └── __init__.py          # GE 转换器占位（暂无实现）
```

## 前置条件

- Linux；Python 3.8+；GCC 9.4.0+
- PyTorch >= 2.6.0 与匹配版本的 `TorchNPU`
- Ascend CANN Toolkit（`ASCEND_HOME_PATH` 已设置）
- 已部署 `aclnnFusedGdnGating` 算子包，运行时可经
  `ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH` 检索到 `libcust_opapi.so`

## 构建与安装

```sh
# 方式一：构建 wheel 并安装（推荐）
bash build_and_install.sh

# 方式二：就地编译（.so 生成在 custom_ops/custom_ops_lib*.so）
python3 setup.py build_ext --inplace
```

## 用法

### eager / 单算子

```python
import torch, torch_npu
import custom_ops   # 注册 torch.ops.custom.npu_fused_gdn_gating 并挂载到 torch_npu

g, beta_output = torch.ops.custom.npu_fused_gdn_gating(
    A_log,      # [num_heads], FP32/BF16/FP16
    a,          # [batch, num_heads], BF16/FP16
    b,          # [batch, num_heads], BF16/FP16
    dt_bias,    # [num_heads], FP32/BF16/FP16
    beta=1.0,
    threshold=20.0,
)
# 也可：torch_npu.npu_fused_gdn_gating(...)
# g:           [1, batch, num_heads], FP32
# beta_output: [1, batch, num_heads], 同 b dtype
```

### 与 fused_gdn_gating pytest 集成

测试通过环境变量 `FUSED_GDN_GATING_CUSTOM_OPS_PATH`（或默认相对路径
`<fused_gdn_gating>/torch_ops_extension`）检索本扩展，既支持 glob
`custom_ops_lib*.so`（`torch.ops.load_library`），也支持 exec `custom_ops/__init__.py`：

```sh
export FUSED_GDN_GATING_CUSTOM_OPS_PATH=<repo>/experimental/attention/fused_gdn_gating/torch_ops_extension
cd <repo>/experimental/attention/fused_gdn_gating/tests/pytest
bash test_run.sh single
```

注册成功后 `has_npu_fused_gdn_gating_op()` 返回 True。

## 接口与 IR

- 入参/属性/输出与算子原型 `op_host/fused_gdn_gating_def.cpp` 一一对应。
- `EXEC_NPU_CMD_V1` 实参按 IR 声明顺序传入（输入→属性→输出）；Python schema 为便于调用将必选张量前置，
  C++ 实现内部已按 IR 顺序重排。
- `A_log` / `dt_bias` 为 1-D `[num_heads]`，dtype 须一致（FP32/BF16/FP16）。
- `a` / `b` 为 2-D `[batch, num_heads]`，dtype 须一致（BF16 或 FP16）。
- 输出 `g` 为 FP32 `[1, batch, num_heads]`（门控结果 `-exp(A_log) * softplus(a+dt_bias)`，其中 `softplus(x) = log(1+exp(beta*x))/beta`，在 `beta*x > threshold` 时取线性分支 `x`）。
- 输出 `beta_output` 与 `b` 同 dtype `[1, batch, num_heads]`（`sigmoid(b)` 门控结果）。
- `beta` / `threshold` 为 float 标量，默认 `1.0` / `20.0`，分别控制 softplus 缩放与阈值分支切换。
