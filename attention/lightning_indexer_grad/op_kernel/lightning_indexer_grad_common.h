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
 * \file lightning_indexer_grad_common.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_GRAD_COMMON_H
#define LIGHTNING_INDEXER_GRAD_COMMON_H

namespace LIGCommon {

enum class LIG_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

struct RunInfo {
    uint64_t bIdx;
    uint64_t n2Idx;
    uint64_t s1Idx;
    uint64_t actualSeqQ;
    uint64_t actualSeqK;
    uint64_t prefixSumS1;
    uint64_t prefixSumS2;
    uint64_t loopTimes;
    uint64_t taskId;
    uint64_t realTopk;
    bool isRemainderCore;
};

struct ConstInfo {
    uint32_t batch;
    uint32_t seqlenQ;
    uint32_t seqlenK;
    uint32_t headNumQ;
    uint32_t headNumK;
    uint32_t groupNum;
    uint32_t headDim;
    uint32_t topK;
    uint32_t usedCoreNum;
    int64_t dkSize;
    int64_t dkCoreSize;
    int64_t dkWorkSpaceOffset;
    int64_t dkCoreWorkspaceOffset;
    int64_t keyGatherWorkspaceOffset;
    int64_t reluInWorkspaceOffset;
    int64_t reluGradWorkspaceOffset;
    int64_t scatterAddWorkspaceOffset;
    uint64_t sparseMode;
    bool deterministic;
    int64_t determinLooptimes;
    int64_t determinLen;
    int64_t determinBeginPos;
    uint32_t splitCores;
};

template <typename TYPE, LIG_LAYOUT LAYOUT_T = LIG_LAYOUT::BSND, typename... Args>
struct LIGType {
    using dataType = TYPE;
    static constexpr LIG_LAYOUT layout = LAYOUT_T;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    // Use division-based form to avoid unsigned integer wraparound in (num + rnd - 1).
    if (rnd == 0) {
        return 0;
    }
    T q = num / rnd;
    return (num % rnd == 0) ? (q * rnd) : ((q + 1) * rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    // Use division-based form to avoid unsigned integer wraparound in (num + rnd - 1).
    return (rnd == 0) ? 0 : (num / rnd + (num % rnd != 0 ? 1 : 0));
}

template <typename T>
__aicore__ inline T RoundUp(T val, T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

} // namespace LIGCommon

#endif // LIGHTNING_INDEXER_GRAD_COMMON_H
