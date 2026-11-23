# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""从设备地址构造 NPU 张量视图（非拥有型，供 workspace 权重视图使用）。

仅 prefetch_weight 的 workspace 权重/scale 视图路径使用：调用方需保证
地址来自 workspace（对称布局约定）且生命周期覆盖视图使用期。

实现要点：必须走 at_npu::native::from_blob（见内嵌 _CPP_SOURCE），普通
at::from_blob 造出的张量底层不是 NPUStorageImpl，torch 侧进 NPU 算子会报
“the tensor's storage is not an NPUStorageImpl”。

C++ 源内嵌于本文件（_CPP_SOURCE 为唯一真源），运行时落到用户缓存目录
~/.cache/moonep_npu_blob 再现场编译——打包目录只登记 .py（CI 侧 native
构建器不识别目录内 .cpp 条目），源码树与 wheel 均不依赖任何外部 .cpp 文件。
"""

import os

_CPP_SOURCE = r"""
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

// 必须用 torch_npu 的 from_blob：通用的 at::from_blob 只把 device 标成
// PrivateUse1，底层仍是普通 StorageImpl，进 NPU 算子会报
// “the tensor's storage is not an NPUStorageImpl”。at_npu::native::from_blob
// 会构造真正的 NPUStorageImpl（含 NPUStorageDesc），可直接参与 NPU 算子。
#include <torch_npu/csrc/aten/common/from_blob.h>

torch::Tensor npu_from_blob(int64_t addr, std::vector<int64_t> sizes, std::string dtype)
{
    at::TensorOptions options;
    if (dtype == "bf16") {
        options = at::TensorOptions().dtype(at::kBFloat16);
    } else if (dtype == "fp16") {
        options = at::TensorOptions().dtype(at::kHalf);
    } else if (dtype == "fp32") {
        options = at::TensorOptions().dtype(at::kFloat);
    } else if (dtype == "uint8") {
        options = at::TensorOptions().dtype(at::kByte);
    } else if (dtype == "e4m3fn") {
        options = at::TensorOptions().dtype(at::kFloat8_e4m3fn);
    } else if (dtype == "e5m2") {
        options = at::TensorOptions().dtype(at::kFloat8_e5m2);
    } else if (dtype == "e8m0fnu") {
        options = at::TensorOptions().dtype(at::kFloat8_e8m0fnu);
    } else if (dtype == "int32") {
        options = at::TensorOptions().dtype(at::kInt);
    } else {
        TORCH_CHECK(false, "npu_from_blob: unsupported dtype ", dtype);
    }
    options = options.device(c10::DeviceType::PrivateUse1);
    // 非拥有视图：不传 deleter，地址生命周期由对称缓冲区（HCCL）保证。
    return at_npu::native::from_blob(
        reinterpret_cast<void*>(addr), at::IntArrayRef(sizes), options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_from_blob", &npu_from_blob, "NPU tensor view over an existing device address (non-owning)");
}
"""

_EXT = None


def _get_ext():
    global _EXT
    if _EXT is None:
        import torch_npu  # noqa: F401  # 注册 PrivateUse1 后端并保证 torch_npu 库已加载
        from torch.utils.cpp_extension import load

        npu_root = os.path.dirname(torch_npu.__file__)
        npu_lib = os.path.join(npu_root, "lib")
        src_dir = os.path.join(os.path.expanduser("~"), ".cache", "moonep_npu_blob")
        os.makedirs(src_dir, exist_ok=True)
        src_path = os.path.join(src_dir, "npu_blob.cpp")
        if not os.path.exists(src_path) or open(src_path).read() != _CPP_SOURCE:
            with open(src_path, "w") as f:
                f.write(_CPP_SOURCE)
        _EXT = load(
            name="moonep_npu_blob",
            sources=[src_path],
            extra_include_paths=[os.path.join(npu_root, "include")],
            # at_npu::native::from_blob 在 libtorch_npu.so 中，必须显式链接；
            # rpath 保证 dlopen 时能找到（进程里 torch_npu 已加载时也不会冲突）。
            extra_ldflags=[f"-L{npu_lib}", "-ltorch_npu", f"-Wl,-rpath,{npu_lib}"],
            extra_cflags=["-O3", "-w"],
            verbose=False,
        )
    return _EXT


def npu_from_blob(addr, sizes, dtype):
    """addr(int) + sizes(list[int]) + dtype("bf16"/"fp16"/"fp32"/"uint8"/
    "e4m3fn"/"e5m2"/"e8m0fnu"/"int32") -> NPU Tensor（mxfp8 数据面：
    e4m3fn/e5m2 权重 + e8m0fnu scale——OCP MX 标准编码）。"""
    return _get_ext().npu_from_blob(int(addr), list(sizes), str(dtype))
