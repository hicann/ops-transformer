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

### Installation Steps

1. Install Dependencies:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

2. Build the Wheel:

    ```sh
    cd <repo_root>
    # 整包编译（含所有算子）
    bash build.sh --torch_extension
    ```

    编译产物位于 `build_out/` 目录下：
    - 整包：`cann_ops_transformer-1.0.0-*.whl`

3. Install Package:

    ```sh
    python3 -m pip install build_out/*.whl --force-reinstall --no-deps
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
