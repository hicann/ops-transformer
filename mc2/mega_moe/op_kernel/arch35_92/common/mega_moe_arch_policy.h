/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH_POLICY_H
#define MEGA_MOE_ARCH_POLICY_H

#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1)
#include "c_api/asc_simd.h"
#endif

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_CUBE__) || \
    defined(__DAV_310R6_VEC__) || defined(__DAV_920R1_CUBE__) || defined(__DAV_920R1_VEC__) || defined(__NPU_ARCH__)
#ifndef MEGA_MOE_IS_KERNEL_COMPILE
#define MEGA_MOE_IS_KERNEL_COMPILE
#endif
#endif

namespace MegaMoeImpl {

using namespace AscendC;

// 只描述 CV 配比和逻辑 block/sub-block 映射。Wave、Workspace 和 handoff 属于执行策略，不能塞进这里。
struct MegaMoeArch35HardwarePolicy {
    static constexpr uint32_t VECTOR_CORE_COUNT_PER_CUBE = 2U;

    __aicore__ static uint32_t GetAicBlockIdx()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1)
        return static_cast<uint32_t>(block_idx);
#else
        return AscendC::GetBlockIdx() / AscendC::GetTaskRation();
#endif
    }

    __aicore__ static uint32_t GetAivBlockIdx()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1)
        if ASCEND_IS_AIV {
            return static_cast<uint32_t>(block_idx * asc_get_sub_block_num() + asc_get_sub_block_id());
        }
        return static_cast<uint32_t>(block_idx);
#else
        return AscendC::GetBlockIdx();
#endif
    }

    __aicore__ static uint32_t GetBlockNum()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1) && \
    !defined(__SUPER_KERNEL_STATIC_BLOCK_NUM__) && !defined(__SUPER_KERNEL_DYNAMIC_BLOCK_NUM__)
        return static_cast<uint32_t>(block_num);
#else
        return static_cast<uint32_t>(AscendC::GetBlockNum());
#endif
    }

    __aicore__ static uint32_t GetAivBlockNum()
    {
        return GetBlockNum() * VECTOR_CORE_COUNT_PER_CUBE;
    }

    __aicore__ static uint32_t GetSubBlockIdx()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1)
        return static_cast<uint32_t>(asc_get_sub_block_id());
#else
        return static_cast<uint32_t>(AscendC::GetSubBlockIdx());
#endif
    }
};

// Arch92 是 1C1V，唯一 AIV 的组内编号为 0。
struct MegaMoeArch92HardwarePolicy {
    static constexpr uint32_t VECTOR_CORE_COUNT_PER_CUBE = 1U;

    __aicore__ static uint32_t GetAicBlockIdx()
    {
        return AscendC::GetBlockIdx();
    }

    __aicore__ static uint32_t GetAivBlockIdx()
    {
        return AscendC::GetBlockIdx();
    }

    __aicore__ static uint32_t GetBlockNum()
    {
        return static_cast<uint32_t>(AscendC::GetBlockNum());
    }

    __aicore__ static uint32_t GetAivBlockNum()
    {
        return GetBlockNum();
    }

    __aicore__ static uint32_t GetSubBlockIdx()
    {
        return 0U;
    }
};

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#define MEGA_MOE_ARCH92_DEVICE 1
using MegaMoeHardwarePolicy = MegaMoeArch92HardwarePolicy;
#else
using MegaMoeHardwarePolicy = MegaMoeArch35HardwarePolicy;
#endif

// 单 AIV 执行原先按子核号分配的分支；双 AIV 按真实子核号选择。
template <uint32_t SubBlockIdx>
__aicore__ inline bool IsSubBlockIdx()
{
    if constexpr (MegaMoeHardwarePolicy::VECTOR_CORE_COUNT_PER_CUBE == 1U) {
        return true;
    } else {
        return MegaMoeHardwarePolicy::GetSubBlockIdx() == SubBlockIdx;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH_POLICY_H
