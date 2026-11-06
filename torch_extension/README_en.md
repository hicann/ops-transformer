# CANN Ops Transformer

English | [简体中文](./README.md)

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

### Building the Wheel Package

```sh
# Build the full package (including all non-experimental operators)
bash build.sh --torch_extension

# Build a single-operator package (containing only the specified operator)
bash build.sh --torch_extension --ops=flash_attn --vendor_name=custom

# Build a multi-operator package
bash build.sh --torch_extension --ops=flash_attn,apply_rotary_pos_emb --vendor_name=custom

# Build an experimental operator package (containing only operators under the experimental directory)
bash build.sh --torch_extension --experimental
```

After the build completes, the wheel packages are automatically copied to the `build_out/` directory.

**Parameter Description:**

| Parameter | Required | Description |
| --- | --- | --- |
| `--torch_extension` | Yes | Builds only the torch_extension wheel package and does not perform cmake compilation |
| `--ops=op1,op2,...` | No | Specifies the operator names to compile (comma-separated). If not specified, all regular operators are compiled |
| `--vendor_name=name` | No | Specifies the subpackage name suffix, used for subpackage naming and isolation. This parameter is invalid when `--ops` is not specified. The default is `custom` |
| `--experimental` | No | Compiles only operators under the experimental directory (mutually exclusive with regular operators). If not specified, the experimental directory is skipped |

**Package Naming Rules:**

| Scenario | Condition | Package Name | Installation Directory |
| --- | --- | --- | --- |
| Full package | `--ops` not specified | `cann_ops_transformer` | `cann_ops_transformer/` |
| Single-operator/multi-operator package | `--ops` specified; `--vendor_name` is optional | `cann_ops_transformer_<vendor>` | `cann_ops_transformer_<vendor>/` |

> **Naming logic:** When `--ops` is not specified, the full package is built, and the package name is fixed to `cann_ops_transformer`. When `--ops` is specified, a subpackage is built, and the package name is `cann_ops_transformer_` concatenated with the value of `--vendor_name` (the default is `custom` if not specified). The full package and subpackages are installed in physically isolated directories and can coexist.

### Installation

```sh
# Install the full package
python3 -m pip install build_out/cann_ops_transformer-*.whl --force-reinstall --no-deps

# Install a single-operator package
python3 -m pip install build_out/cann_ops_transformer_custom-*.whl --force-reinstall --no-deps
```

### Coexistence Mechanism of the Full Package and Subpackages

The full package and single-operator packages can be installed at the same time without conflicts:

- The **full package** is installed in the `cann_ops_transformer/` directory and contains all regular operators.
- A **single-operator package** is installed in the `cann_ops_transformer_<vendor>/` directory, physically isolated from the full package.
- Single-operator packages register operators through the **entry point** mechanism and take priority over the full package. When a user calls `cann_ops_transformer.<op>`, the operator implementation of the subpackage is used if the subpackage is installed.
- After a single-operator package is uninstalled, the operator with the same name in the full package takes over automatically.

```sh
# Install the full package
pip install cann_ops_transformer-*.whl

# Install a single-operator package (overriding the operator with the same name in the full package)
pip install cann_ops_transformer_custom-*.whl

# Uninstall a single-operator package (the operator in the full package is restored automatically)
pip uninstall cann_ops_transformer_custom
```

## Quick Start

Using `cann_ops_transformer` is seamless. You can invoke NPU-accelerated operators directly through the library's opset.

```python
import torch
import torch_npu
import cann_ops_transformer

# Initialize data on NPU
# Layout: batch, sequence, heads, head_dim.
q, k, v = [torch.randn(1, 16, 1, 64, dtype=torch.float16).npu() for _ in range(3)]
scale = 64 ** -0.5

# Generate the task metadata required by the operator.
metadata = cann_ops_transformer.ops.flash_attn_metadata(
    1, 1, 64, batch_size=1, max_seqlen_q=16, max_seqlen_kv=16,
    mask_mode=0, win_left=-1, win_right=-1,
    layout_q="BSND", layout_kv="BSND", layout_out="BSND"
)

# Call the custom NPU operator
# The first call compiles and loads the C++ wrapper.
output, _ = cann_ops_transformer.ops.flash_attn(
    q, k, v, metadata=metadata, softmax_scale=scale
)

# Verify against CPU ATen implementation
q_cpu, k_cpu, v_cpu = [x.cpu().float().transpose(1, 2) for x in (q, k, v)]
expected = torch.nn.functional.scaled_dot_product_attention(
    q_cpu, k_cpu, v_cpu, dropout_p=0.0, is_causal=False, scale=scale
)
assert torch.allclose(output.cpu().float(), expected.transpose(1, 2), rtol=1e-3, atol=1e-3)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

> For the full operator development specification — directory layout, naming, per-layer implementation (C++ / Python / torchair graph mode), docstring and DeviceGuard requirements — see [torch\_extension Development Guidelines](cann_ops_transformer/docs/torch_extension_guidelines_en.md).

To implement a new operator (for example, `abs`), you need to provide two components: a C++ kernel wrapper and a Python JIT builder, placed in `<category>/<op>/torch_extension/`.

### Directory Structure

```
ops-transformer/
├── activation/abs/                      # Category that the operator belongs to
│   ├── op_host/                         # Existing code (unchanged)
│   ├── op_kernel/                       # Existing code (unchanged)
│   ├── tests/                           # Existing tests (unchanged)
│   └── torch_extension/                 # Newly added: torch_extension files
│       ├── __init__.py                  # Exports abs and convert_abs
│       ├── abs.py                       # Python frontend
│       ├── graph_convert_abs.py         # torchair graph mode converter (optional)
│       └── csrc/
│           └── abs.cpp                  # C++ backend
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
