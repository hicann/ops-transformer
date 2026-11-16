/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_COMMON_OP_KERNEL_TRANSFORMER_AICPU_REGISTER_H
#define OPS_TRANSFORMER_COMMON_OP_KERNEL_TRANSFORMER_AICPU_REGISTER_H

#include "cpu_kernel.h"

namespace aicpu {
__attribute__((weak)) bool RegistCpuKernelV2(const std::string& type, const KERNEL_CREATOR_FUN& fun);
} // namespace aicpu

#if defined(OPS_TRANSFORMER_AICPU_HOST_KERNEL)
#define OPS_TRANSFORMER_REGISTER_CPU_KERNELV2(type, clazz) \
    static std::shared_ptr<aicpu::CpuKernel> Creator_##type##_Kernel() \
    { \
        return aicpu::MakeShared<clazz>(); \
    } \
    static bool g_##type##_Kernel_Creator __attribute__((unused)) = \
        ((&::aicpu::RegistCpuKernelV2) != nullptr) ? ::aicpu::RegistCpuKernelV2((type), Creator_##type##_Kernel) : \
                                                     ::aicpu::RegistCpuKernel((type), Creator_##type##_Kernel)
#else
#define OPS_TRANSFORMER_REGISTER_CPU_KERNELV2(type, clazz) REGISTER_CPU_KERNEL(type, clazz)
#endif

#endif // OPS_TRANSFORMER_COMMON_OP_KERNEL_TRANSFORMER_AICPU_REGISTER_H
