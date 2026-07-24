# Fast Kernel Launch

## 简介

本文档演示如何使用Ascend C和[PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html)能力开发自定义NPU算子。

**核心优势:**

- **单交付件：** 一个文件完成算子开发和PyTorch框架适配。
- **高效调用：** 使用`<<<>>>`语法启动核函数，流程简单高效。

## 环境部署 | Prerequisites

- 请先参考[环境部署](../../docs/zh/install/quick_install.md)完成基础环境搭建
- gcc 9.4.0+
- python 3.8+
- torch>=2.6.0
- 对应版本的[TorchNPU](https://gitcode.com/Ascend/pytorch/releases)

## 安装步骤 | Installation Steps

1. 进入`examples/fast_kernel_launch_example`目录。

2. 安装依赖 | Install Dependencies:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

3. 构建Wheel包 | Build the Wheel:

    ```sh
    # NPU_SOC_VERSION设置编译款型, Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"
    export NPU_SOC_VERSION=ascend910b
    # -n: non-isolated build (uses existing environment)
    python3 -m build --wheel -n
    ```

    构建完成后，产物在当前目录的`dist`文件夹下，产物名`ascend_ops-1.0.0-${python_version}-abi3-${arch}.whl`，
    `${python_version}`表示当前环境中的python版本(python3.8.3为cp38)，`${arch}`表示CPU架构。

4. 安装Wheel包 | Install Package:

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

5.（可选）再次构建前建议先执行以下命令清理编译缓存

   ```sh
    python setup.py clean
    ```

## 快速开始 | Quick Start

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以add算子调用为例。

```python
import torch
import torch_npu
import ascend_ops  # 构建出的python包

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.ascend_ops.add(x, y)  # PyTorch Custom Operator Dispatch机制: torch.ops.<library_name>.<operator_name>

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## 开发指南：新增一个算子 | Developer Guide: Adding a New Operator

为了实现一个新算子(如`add`)，您只需要提供一个C++实现即可。

1. 首先您需要在csrc目录下使用算子名`add`建立一个文件夹，在此文件夹内使用你当前想要开发的soc名建立一个子文件夹`ascend910b`。

2. 在soc目录下新建一个`CMakeLists.txt`

    ```
    add_sources("--npu-arch=dav-2201")
    ```

    这里`dav-2201`为ascend910b芯片对应的编译参数，获取方法参考[NpuArch说明和使用指导](https://gitcode.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md)。

3. 在soc目录下新建一个`add.cpp`(建议使用算子名为文件名)。这个文件包含了开发一个AI Core算子所需要的全部模块。
    - 算子Schema注册
    - 算子Meta Function实现 & 注册
    - 算子Kernel实现(Ascend C)
    - 算子NPU调用实现 & 注册

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace ascend_ops {  // 当前项目为一个命名空间
    namespace Add {         // 建议每个算子自己有一个独立的namespace，防止全局变量污染

    /**
     * 将算子schema注册给PyTorch框架
     * 框架知道有这样一个算子
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * 实现算子的Meta函数，即InferShape+InferDtype
     * 根据输入推导出这个算子的输出是什么样子，需要多少空间，不需要实际计算这个算子
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * 将算子的Meta函数注册给框架
     * 框架可以调用这个Meta函数，在真正执行这个算子计算前知道需要多大空间
     * 后续可以支持torch.compile/AutoGrad/AclGraph等图加速
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * NPU算子Kernel实现，使用AscendC API，面向当前的soc编写
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * 实现算子调用接口
     * 在这个接口中，需要完成NPU Kernel的调用
     * 1. 计算出输出的Tensor的个数/Shape/Dtype(可以调用Meta函数实现，也可以直接实现)
     * 2. 计算Tiling：根据Shape得到如何分块计算
     * 3. 调用NPU Kernel
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        // OptionalDeviceGuard确保后续操作在正确的设备上下文执行
        // 它会记录当前设备状态，执行完作用域代码后自动恢复
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
                // 根据不同的数据类型，调用不同的NPU Kernel
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
        // 需要使用RunOpApi/RunOpApiV2接口调用，保证时序与TorchNPU调用aclnn接口一致。
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * 将算子的调用函数注册给框架，Device为PrivateUse1
     * 框架知道当输入均在NPU Device上时，Dispatch到这个算子实现
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace ascend_ops

    ```

4. 参考[安装步骤](#安装步骤--installation-steps)章节重新构建Wheel包并安装。
5. 基于pytest测试算子API，请参考[test_add.py](tests/add/test_add.py)的实现。
