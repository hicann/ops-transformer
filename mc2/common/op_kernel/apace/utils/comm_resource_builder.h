/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_resource_builder.h
 * \brief Minimal HCCL communication resource definitions for apace ubmem kernels.
 */

#ifndef APACE_COMM_RESOURCE_BUILDER_H
#define APACE_COMM_RESOURCE_BUILDER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;
constexpr uint32_t EP_RANK_OFFSET_STEP = 1024;
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 1024UL * 1024UL;

struct HcclCombinOpParam {
    uint64_t workSpace;                         // client-server communication address
    uint64_t workSpaceSize;                     // client-server communication space size
    uint32_t rankId;                            // current card rankId
    uint32_t rankDim;                           // total card count
    uint64_t winSize;                           // not used by ccu
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM];  // MTE data area (windowsIn[rankId] = local card address)
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM]; // MTE status area

    // for ccu
    uint64_t xnAddr;  // Xn register start address
    uint64_t ckeAddr; // CKE register start address
    uint64_t msAddr;  // MS address, reserved
    uint64_t msSize;  // writable MS count, reserved
};

namespace Apace {
using HcclOpParam = HcclCombinOpParam;

__aicore__ inline uint32_t GetRankId(__gm__ HcclOpParam *winContext)
{
    return winContext->rankId;
}

__aicore__ inline uint32_t GetRankDim(__gm__ HcclOpParam *winContext)
{
    return winContext->rankDim;
}

__aicore__ inline GM_ADDR GetBaseWindAddrByRankId(__gm__ HcclOpParam *winContext, const int32_t rankId)
{
    return (GM_ADDR)(winContext->windowsIn[rankId] + A5_MTE_STATE_WIN_SIZE + rankId * EP_RANK_OFFSET_STEP);
}

__aicore__ inline GM_ADDR GetBaseWindStateAddrByRankId(__gm__ HcclOpParam *winContext, const int32_t rankId)
{
    return (GM_ADDR)(winContext->windowsIn[rankId] + rankId * EP_RANK_OFFSET_STEP);
}
} // namespace Apace

#endif // APACE_COMM_RESOURCE_BUILDER_H
