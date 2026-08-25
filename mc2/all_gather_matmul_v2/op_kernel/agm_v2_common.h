/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file agm_v2_common.h
 * \brief
 */
#ifndef AGM_V2_COMMON_H
#define AGM_V2_COMMON_H

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2;
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1;
#define DTYPE_X1 half
#define DTYPE_Y half
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

namespace AscendC {
using A_DTYPE = DTYPE_X1;
using B_DTYPE = DTYPE_X1;
using C_DTYPE = DTYPE_Y;
using BIAS_DTYPE = DTYPE_Y;

constexpr uint32_t EVENT_ID_3 = 3;
constexpr uint32_t EVENT_ID_4 = 4;
constexpr uint32_t MOVE_LEFT_SIZE = 8;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t MAX_HANDLE = 63;
constexpr uint32_t MAX_HANDLE_WITH_SCALE1 = MAX_HANDLE + 1;        // 包含x1Scale通信
constexpr uint32_t SCALE1_HANDLE_IDX = MAX_HANDLE_WITH_SCALE1 - 1; // x1Scale独立handle位
constexpr uint32_t SCALE1_MULTIPLE = 4;
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;  // 只计算不通信
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算
constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr uint64_t PF4_DATA_NUM_PER_BYTE = 2;

#ifndef HCCL_COMM_MODE_CCU
#define HCCL_COMM_MODE_CCU 0
#endif
#ifndef HCCL_COMM_MODE_AICPU
#define HCCL_COMM_MODE_AICPU 1
#endif

template <class T>
struct BiasType {
    using type = float;
};
template <>
struct BiasType<half> {
    using type = half;
};
template <>
struct BiasType<bfloat16_t> {
    using type = bfloat16_t;
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

struct MC2GmAddrs {
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR biasGM;
    GM_ADDR cGM;
    GM_ADDR gatherOut;
    GM_ADDR workspaceGM;
};

struct QuantGmAddrs {
    GM_ADDR scale1GM;
    GM_ADDR scale2GM;
    GM_ADDR scaleGM;
};

enum Mc2CoreType {
    ON_CUBE_AND_VECTOR = 0,
    ON_VECTOR,
    ON_CUBE,
};

template <Mc2CoreType coreType>
__aicore__ inline void Mc2SyncAll()
{
    if constexpr (coreType == Mc2CoreType::ON_CUBE_AND_VECTOR) {
        SyncAll<false>();
    } else if constexpr (coreType == Mc2CoreType::ON_VECTOR) {
        SyncAll();
    } else {
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);
    }
}

} // namespace AscendC
#endif // AGM_V2_COMMON_H
