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
 * \file moe_gating_top_k_arch35_common.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_ARCH35_COMMON_H
#define MOE_GATING_TOP_K_ARCH35_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "basic_api/kernel_operator_utils_intf.h"

namespace MoeGatingTopK {
using namespace AscendC;
using MicroAPI::RegTensor;
constexpr int32_t MIN_FP32 = 0xFF800000;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr uint32_t VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 x, T2 y)
{
    if (y != 0 && x != 0) {
        const T1 quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }

    return x;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace MoeGatingTopK

namespace MoeGatingTopK {

template <typename T>
__aicore__ inline void SmallKAlignEVFNoNorm(__ubuf__ float *inputAddr, __ubuf__ uint32_t *mrgSortAddr,
                                            __ubuf__ T *outputAddr, __ubuf__ uint32_t *expertIdxAddr, uint32_t k,
                                            float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;

        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(k);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx, mrgSortAddr);
        MicroAPI::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
        MicroAPI::Muls(vregGathered, vregGathered, routedScalingFactor, preg0);
        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregGathered, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T>
__aicore__ inline void SmallKAlignEVFWithNorm(__ubuf__ float *inputAddr, __ubuf__ uint32_t *mrgSortAddr,
                                              __ubuf__ T *outputAddr, __ubuf__ uint32_t *expertIdxAddr, uint32_t k,
                                              float eps, float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        RegTensor<float> vregSum;
        RegTensor<float> vregSumBcast;

        MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(k);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx, mrgSortAddr);
        MicroAPI::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
        Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregGathered, preg0);
        MicroAPI::Adds(vregSum, vregSum, eps, preg0);
        MicroAPI::Duplicate(vregSumBcast, vregSum, preg0);
        MicroAPI::Div(vregSumBcast, vregGathered, vregSumBcast, preg0);
        MicroAPI::Muls(vregSumBcast, vregSumBcast, routedScalingFactor, preg0);
        ops::StoreOneTensorForDtypeT<T>(outputAddr, vregSumBcast, preg0, 0);
        MicroAPI::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T>
__aicore__ inline void LargeKAlignEVFNoNorm(__ubuf__ float *inputAddr, __ubuf__ uint32_t *mrgSortAddr,
                                            __ubuf__ T *outputAddr, __ubuf__ uint32_t *expertIdxAddr, uint32_t k,
                                            float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<uint32_t>(k);
            MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                               mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
            MicroAPI::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
            MicroAPI::Muls(vregGathered, vregGathered, routedScalingFactor, preg0);
            ops::StoreOneTensorForDtypeT<T>(outputAddr, vregGathered, preg0, i * VL_FLOAT_SIZE);
            MicroAPI::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregExpertIdx, preg0);
        }
    }
}

template <typename T>
__aicore__ inline void LargeKAlignEVFWithNorm(__ubuf__ float *inputAddr, __ubuf__ uint32_t *mrgSortAddr,
                                              __ubuf__ T *outputAddr, __ubuf__ uint32_t *expertIdxAddr, uint32_t k,
                                              float eps, float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        RegTensor<float> vregSumBcast;
        RegTensor<float> vregOutput;
        RegTensor<float> vregSum;

        MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
        MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<float>();
        MicroAPI::Duplicate(vregSum, static_cast<float>(0), preg0);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = MicroAPI::UpdateMask<uint32_t>(k);
            MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                               mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
            MicroAPI::Duplicate(vregGathered, static_cast<float>(0), preg1);
            MicroAPI::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
            MicroAPI::Add(vregSum, vregSum, vregGathered, preg1);
        }

        Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregSum, preg1);
        MicroAPI::Adds(vregSum, vregSum, eps, preg1);
        MicroAPI::Duplicate(vregSumBcast, vregSum, preg1);
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg1 = MicroAPI::UpdateMask<uint32_t>(k);
            MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                               mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
            MicroAPI::Gather(vregGathered, inputAddr, vregExpertIdx, preg1);
            MicroAPI::Div(vregOutput, vregGathered, vregSumBcast, preg1);
            MicroAPI::Muls(vregOutput, vregOutput, routedScalingFactor, preg1);
            ops::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg1, i * VL_FLOAT_SIZE);
            MicroAPI::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregExpertIdx, preg1);
        }
    }
}

} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_ARCH35_COMMON_H
