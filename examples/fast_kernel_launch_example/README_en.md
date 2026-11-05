# Fast Kernel Launch

English | [简体中文](./README.md)

## Overview

This document demonstrates how to develop custom NPU operators using Ascend C and the [PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html) capabilities.

**Key advantages:**

- **Single deliverable:** One file completes both operator development and PyTorch framework adaptation.
- **Efficient invocation:** Kernel functions are launched with the `<<<>>>` syntax, making the flow simple and efficient.

## Prerequisites

- For basic environment setup, first refer to [Environment Deployment](../../docs/zh/install/quick_install.md)
- gcc 9.4.0+
- python 3.8+
- torch>=2.6.0
- The matching version of [TorchNPU](https://atomgit.com/Ascend/pytorch/releases)

## Installation Steps

1. Enter the `examples/fast_kernel_launch_example` directory.

2. Install Dependencies | Install Dependencies:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

3. Build Wheel Package | Build the Wheel:

    ```sh
    # NPU_SOC_VERSION sets the compilation model. Use "ascend910b" (default) for Atlas A2 products, "ascend910_93" for Atlas A3 products, and "ascend950" for Ascend 950PR&950DT products
    export NPU_SOC_VERSION=ascend910b
    # -n: non-isolated build (uses existing environment)
    python3 -m build --wheel -n
    ```

    After the build completes, the artifacts are stored in the `dist` folder in the current directory, and the artifact name is `ascend_ops-1.0.0-${python_version}-abi3-${arch}.whl`.
    `${python_version}` indicates the python version in the current environment (python3.8.3 corresponds to cp38), and `${arch}` indicates the CPU architecture.

4. Install Wheel Package | Install Package:

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

5. (Optional) It is recommended to run the following command to clear the compilation cache before rebuilding

   ```sh
    python setup.py clean
    ```

## Quick Start

After installation, you can use NPU operators in the same way as ordinary PyTorch operations. Taking the invocation of the add operator as an example:

```python
import torch
import torch_npu
import ascend_ops  # the built Python package

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.ascend_ops.add(x, y)  # PyTorch Custom Operator Dispatch mechanism: torch.ops.<library_name>.<operator_name>

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

To implement a new operator (such as `add`), you only need to provide a C++ implementation.

1. First, create a folder named after the operator `add` in the csrc directory. In this folder, create a subfolder named after the soc you currently want to develop for, for example `ascend910b`.

2. Create a `CMakeLists.txt` file in the soc directory:

    ```
    add_sources("--npu-arch=dav-2201")
    ```

    Here `dav-2201` is the compilation parameter corresponding to the ascend910b chip. For how to obtain it, see the [NpuArch description and usage guide](https://atomgit.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md).

3. Create an `add.cpp` file in the soc directory (using the operator name as the file name is recommended). This file contains all modules required to develop an AI Core operator:
    - Operator schema registration
    - Operator Meta Function implementation & registration
    - Operator kernel implementation (Ascend C)
    - Operator NPU invocation implementation & registration

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace ascend_ops {  // The current project is one namespace
    namespace Add {         // It is recommended that each operator have its own namespace to prevent global variable pollution

    /**
     * Register the operator schema with the PyTorch framework
     * so that the framework knows that this operator exists
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * Implement the Meta function of the operator, that is, InferShape+InferDtype
     * Infer the output shape and required space of the operator from the inputs,
     * without actually computing the operator
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * Register the Meta function of the operator with the framework
     * so that the framework can call this Meta function and know how much space is required
     * before actually executing the operator computation.
     * This later supports graph acceleration such as torch.compile/AutoGrad/AclGraph
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * NPU operator kernel implementation, using the AscendC API,
     * written for the current soc
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * Implement the operator invocation interface
     * In this interface, the invocation of the NPU kernel needs to be completed:
     * 1. Compute the number/Shape/Dtype of the output Tensor (you can call the Meta function or implement it directly)
     * 2. Compute Tiling: determine how to partition the computation based on the Shape
     * 3. Invoke the NPU kernel
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        // OptionalDeviceGuard ensures that subsequent operations execute in the correct device context
        // It records the current device state and automatically restores it after the scoped code finishes
        const c10::OptionalDeviceGuard guard(x.device());
        auto z = add_meta(x, y);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        int64_t totalLength, numBlocks, blockLength, tileSize;
        totalLength = x.numel();
        std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
        auto x_ptr = (GM_ADDR)x.data_ptr();
        auto y_ptr = (GM_ADDR)y.data_ptr();
        auto z_ptr = (GM_ADDR)z.data_ptr();
        auto acl_call = [=]() -> int {
            AT_DISPATCH_SWITCH(
                x.scalar_type(), "add_npu",
                // Invoke different NPU kernels depending on the data type
                AT_DISPATCH_CASE(torch::kFloat32, [&] {
                    using scalar_t = float;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kFloat16, [&] {
                    using scalar_t = half;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kInt32, [&] {
                    using scalar_t = int32_t;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
            );
            return 0;
        };
        // The RunOpApi/RunOpApiV2 interfaces must be used for the invocation to ensure that the timing is consistent with the aclnn interfaces invoked by TorchNPU.
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * Register the invocation function of the operator with the framework, with Device set to PrivateUse1
     * so that the framework knows to dispatch to this operator implementation
     * when the inputs are all on the NPU Device
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace ascend_ops

    ```

4. Rebuild and install the wheel package by referring to the [Installation Steps](#installation-steps) section.
5. To test the operator API with pytest, refer to the implementation of [test_add.py](tests/add/test_add.py).
