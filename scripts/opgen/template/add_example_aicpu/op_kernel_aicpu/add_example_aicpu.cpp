/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_example_aicpu.h"

#include <cmath>
#include <string>
#include "log.h"
#include "aicpu/cust_cpu_utils.h"

namespace {
const char* const kAddExample = "AddExampleAicpu";
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kSuccess = 0;
const uint32_t kParamInvalid = 1;
} // namespace

namespace aicpu {
uint32_t AddExampleCpuKernel::Compute(CpuKernelContext& ctx)
{
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    Tensor* output = ctx.Output(0);

    if (input0 == nullptr || input1 == nullptr || output == nullptr) {
        KERNEL_LOG_ERROR("Invalid argument");
        return kParamInvalid;
    }

    if (input0->GetDataSize() == 0 || input1->GetDataSize() == 0) {
        return kSuccess;
    }

    // 继续补充算子计算逻辑
}

REGISTER_CPU_KERNEL(kAddExample, AddExampleCpuKernel);

} // namespace aicpu
