# CANN Ops Transformer

`cann_ops_transformer` is a high-performance operator extension library designed for Ascend NPU. It leverages Just-In-Time(JIT) compilation to bridge PyTorch functional interfaces with ACLNN library.

## Build & Installation

### Prerequisites

- OS: Linux
- Python: 3.8+
- Compiler: GCC 9.4.0+
- Frameworks:
  - PyTorch>=2.6.0
  - torch_npu (matching your PyTorch version)
- Toolkit: Ascend CANN Toolkit

### 构建 Wheel 包

```sh
# 构建整包（包含所有非 experimental 算子）
bash build.sh --torch_extension

# 构建单算子包（仅包含指定算子）
bash build.sh --torch_extension --ops=flash_attn --vendor_name=custom

# 构建多算子包
bash build.sh --torch_extension --ops=flash_attn,apply_rotary_pos_emb --vendor_name=custom

# 构建实验性算子包（仅包含 experimental 目录下的算子）
bash build.sh --torch_extension --experimental
```

构建完成后，wheel 包会自动复制到 `build_out/` 目录。

**参数说明：**

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `--torch_extension` | 是 | 仅构建 torch_extension wheel 包，不执行 cmake 编译 |
| `--ops=op1,op2,...` | 否 | 指定编译的算子名（逗号分隔），不指定则编译所有常规算子 |
| `--vendor_name=name` | 否 | 指定子包名后缀，用于子包命名和隔离。不指定 `--ops` 时此参数无效，默认为 `custom` |
| `--experimental` | 否 | 仅编译 experimental 目录下的算子（与常规算子互斥），不指定则跳过 experimental 目录 |

**包命名规则：**

| 场景 | 条件 | 包名 | 安装目录 |
| --- | --- | --- | --- |
| 整包 | 不指定 `--ops` | `cann_ops_transformer` | `cann_ops_transformer/` |
| 单算子/多算子包 | 指定 `--ops`，`--vendor_name` 可选 | `cann_ops_transformer_<vendor>` | `cann_ops_transformer_<vendor>/` |

> **命名逻辑：** 不指定 `--ops` 时构建整包，包名固定为 `cann_ops_transformer`；指定 `--ops` 时构建子包，包名为 `cann_ops_transformer_` 拼接 `--vendor_name` 的值（未指定则默认 `custom`）。整包与子包安装目录物理隔离，可共存。

### 安装

```sh
# 安装整包
python3 -m pip install build_out/cann_ops_transformer-*.whl --force-reinstall --no-deps

# 安装单算子包
python3 -m pip install build_out/cann_ops_transformer_custom-*.whl --force-reinstall --no-deps
```

### 整包与子包共存机制

整包和单算子包可以同时安装，互不冲突：

- **整包**安装到 `cann_ops_transformer/` 目录，包含所有常规算子。
- **单算子包**安装到 `cann_ops_transformer_<vendor>/` 目录，与整包物理隔离。
- 单算子包通过 **entry point** 机制注册算子，优先级高于整包。用户调用 `cann_ops_transformer.<op>` 时，若子包已安装则使用子包的算子实现。
- 卸载单算子包后，整包的同名算子自动接管。

```sh
# 安装整包
pip install cann_ops_transformer-*.whl

# 安装单算子包（覆盖整包中的同名算子）
pip install cann_ops_transformer_custom-*.whl

# 卸载单算子包（整包算子自动恢复）
pip uninstall cann_ops_transformer_custom
```

## Quick Start

Using `cann_ops_transformer` is seamless. You can invoke NPU-accelerated operators directly through the library's opset.

```python
import torch
import torch_npu
import cann_ops_transformer

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
# This triggers JIT compilation on the first call
npu_result = cann_ops_transformer.ops.abs(x)

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_result = torch.ops.aten.abs(cpu_x)

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

> For the full operator development specification — directory layout, naming, per-layer implementation (C++ / Python / torchair graph mode), docstring and DeviceGuard requirements — see [torch\_extension 开发规范](cann_ops_transformer/docs/torch_extension_guidelines.md).

To implement a new operator (e.g. `abs`), you need to provide two components: a C++ kernel wrapper and a Python JIT builder, placed in `<category>/<op>/torch_extension/`.

### Directory Structure

```
ops-transformer/
├── activation/abs/                      # 算子所属 category
│   ├── op_host/                         # 原有代码（不动）
│   ├── op_kernel/                       # 原有代码（不动）
│   ├── tests/                           # 原有测试（不动）
│   └── torch_extension/                 # 新增：torch_extension 文件
│       ├── __init__.py                  # 导出 abs 和 convert_abs
│       ├── abs.py                       # Python 前端
│       ├── graph_convert_abs.py         # torchair 图模式 Converter（可选）
│       └── csrc/
│           └── abs.cpp                  # C++ 后端
```

### 1. C++ Backend(`<category>/<op>/torch_extension/csrc/<OP_NAME>.cpp`)

This file bridges PyTorch tensors to the ACLNN C-API.

```cpp
#include <torch/extension.h>
#include "aclnn_common.h"

/**
 * @brief ACLNN Wrapper for aclnnAbs
 * @param x Input Tensor (on NPU)
 * @return Result Tensor
 */
at::Tensor npu_abs(const at::Tensor &x)
{
    // 1. Manually allocate output tensor (standard PyTorch practice)
    at::Tensor y = at::empty_like(x);

    // 2. Launch ACLNN kernel using the helper macro
    ACLNN_CMD(aclnnAbs, x, y);

    return y;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_abs", &npu_abs, "abs");
}
```

### 2. Python Frontend(`<category>/<op>/torch_extension/<OP_NAME>.py`)

This file manages the JIT compilation logic and registers the operator into the PyTorch Dispatcher.

```python
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

class AbsOpBuilder(OpBuilder):
    def __init__(self):
        super(AbsOpBuilder, self).__init__("abs", category="activation")

    def sources(self):
        """Path to C++ source code."""
        return ['csrc/activation/abs.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "abs(Tensor x) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(get_as_library(), self.name, "Meta")
        def abs_meta(x):
            return torch.empty_like(x)

# Instantiate the builder
abs_op_builder = AbsOpBuilder()
abs_op_builder._ensure_initialized()

@impl(get_as_library(), abs_op_builder.name, "PrivateUse1")
def abs(x):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    op_module = abs_op_builder.load()  # Compiles/loads the .so file
    return op_module.npu_abs(x)
```

### 3. Operator init (`<category>/<op>/torch_extension/__init__.py`)

```python
__all__ = ["abs", "convert_abs"]

from .abs import abs
from .graph_convert_abs import convert_abs
```

### Technical Notes

| Component | Responsibility |
| --- | --- |
| **OpBuilder** | Handles JIT compilation of C++ source using `ninja`. |
| **Meta Dispatch** | Allows PyTorch to know the output shape/type without running NPU code. |
| **PrivateUse1** | The specific backend key PyTorch uses to route NPU-specific operations. |
