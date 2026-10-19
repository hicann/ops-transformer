/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_sort_lib_one_core.h
 * \brief 单核 radix sort 类 (SortRadixOneCore)。
 *        小 N 时直接调用 AscendC::Sort(RADIX_SORT)，0 SyncAll、0 GM workspace。
 *
 * \internal  Do not include directly — use sort_lib.h instead.
 */

#ifndef ENGRAM_SORT_LIB_ONE_CORE_H
#define ENGRAM_SORT_LIB_ONE_CORE_H

#include "kernel_operator.h"
#include "sort_util.h"
#include "sort_lib_params.h"

namespace SortLib::detail {

template <typename ValT, typename IdxT, bool isDescend>
class SortRadixOneCore {
private:
    static constexpr AscendC::SortConfig sortConfigOne{AscendC::SortType::RADIX_SORT, isDescend};

    // === GM input / output ===
    AscendC::GlobalTensor<ValT> inputXGm_;
    AscendC::GlobalTensor<ValT> outValueGm_;
    AscendC::GlobalTensor<uint32_t> outIdxGm_;

    // === UB queues ===
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outValueQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outIdxQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpUb_;

    // === runtime state ===
    uint32_t numTileData_ = 0;
    uint32_t tmpUbSize_ = 0;
    uint32_t halfIndex_ = 0;

public:
    __aicore__ inline SortRadixOneCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const SortLib::SortParams& params, AscendC::TPipe* pipe);
    __aicore__ inline void Process();
};

template <typename ValT, typename IdxT, bool isDescend>
__aicore__ inline void SortRadixOneCore<ValT, IdxT, isDescend>::Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex,
                                                                     GM_ADDR workspace,
                                                                     const SortLib::SortParams& params,
                                                                     AscendC::TPipe* pipe)
{
    // 单核路径不使用 GM workspace，忽略 workspace 参数。
    pipe_ = pipe;
    numTileData_ = params.numTileData;
    tmpUbSize_ = params.tmpUbSize;
    uint32_t idxUbSize32 = RoundUpAlign(static_cast<uint32_t>(numTileData_ * sizeof(uint32_t))); // ub 32对齐
    halfIndex_ = idxUbSize32 / static_cast<uint32_t>(sizeof(uint32_t));
    // sort api只支持uint32，但是调用方可能需要这个排序伴随下表为int64，需扩展最后uint32->int64，需要更大空间
    uint32_t idxBufferBytes = (sizeof(IdxT) == sizeof(int64_t)) ? (2U * idxUbSize32) : idxUbSize32;

    inputXGm_.SetGlobalBuffer((__gm__ ValT*)x);
    outValueGm_.SetGlobalBuffer((__gm__ ValT*)value);
    outIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndex);

    pipe_->InitBuffer(inQueueX_, 1, RoundUpAlign(static_cast<uint32_t>(numTileData_ * sizeof(ValT))));
    pipe_->InitBuffer(outValueQueue_, 1, RoundUpAlign(static_cast<uint32_t>(numTileData_ * sizeof(ValT))));
    pipe_->InitBuffer(outIdxQueue_, 1, idxBufferBytes); // 存排序idx用
    pipe_->InitBuffer(tmpUb_, tmpUbSize_);
}

template <typename ValT, typename IdxT, bool isDescend>
__aicore__ inline void SortRadixOneCore<ValT, IdxT, isDescend>::Process()
{
    AscendC::LocalTensor<ValT> xLocal = inQueueX_.template AllocTensor<ValT>();
    CopyGmToUb(inputXGm_, xLocal, 0, numTileData_);
    inQueueX_.EnQue(xLocal);
    xLocal = inQueueX_.template DeQue<ValT>();

    AscendC::LocalTensor<ValT> sortedValueUb = outValueQueue_.template AllocTensor<ValT>();
    AscendC::LocalTensor<uint32_t> sortIdxUb = outIdxQueue_.template AllocTensor<uint32_t>();
    AscendC::LocalTensor<uint8_t> tmpUb = tmpUb_.template Get<uint8_t>();

    if constexpr (sizeof(IdxT) == sizeof(int64_t)) {
        AscendC::LocalTensor<uint32_t> sortIdxUbHalf = sortIdxUb[halfIndex_];
        AscendC::Sort<ValT, false, sortConfigOne>(sortedValueUb, sortIdxUbHalf, xLocal, tmpUb, numTileData_);
        inQueueX_.FreeTensor(xLocal);
        AscendC::LocalTensor<int64_t> sortIdxUbInt64 = sortIdxUb.template ReinterpretCast<int64_t>();
        AscendC::LocalTensor<int32_t> sortIdxUbInt32 = sortIdxUbHalf.template ReinterpretCast<int32_t>();
        AscendC::Cast(sortIdxUbInt64, sortIdxUbInt32, AscendC::RoundMode::CAST_NONE, numTileData_);
    } else {
        AscendC::Sort<ValT, false, sortConfigOne>(sortedValueUb, sortIdxUb, xLocal, tmpUb, numTileData_);
        inQueueX_.FreeTensor(xLocal);
    }

    outValueQueue_.EnQue(sortedValueUb);
    sortedValueUb = outValueQueue_.template DeQue<ValT>();
    outIdxQueue_.EnQue(sortIdxUb);
    sortIdxUb = outIdxQueue_.template DeQue<uint32_t>();

    CopyUbToGm(outValueGm_, sortedValueUb, 0, numTileData_);
    if constexpr (sizeof(IdxT) == sizeof(int64_t)) {
        AscendC::GlobalTensor<int64_t> outIdxGmInt64 = outIdxGm_.template ReinterpretCast<int64_t>();
        AscendC::LocalTensor<int64_t> sortIdxUbInt64 = sortIdxUb.template ReinterpretCast<int64_t>();
        CopyUbToGm(outIdxGmInt64, sortIdxUbInt64, 0, numTileData_);
    } else {
        CopyUbToGm(outIdxGm_, sortIdxUb, 0, numTileData_);
    }

    outValueQueue_.FreeTensor(sortedValueUb);
    outIdxQueue_.FreeTensor(sortIdxUb);
}

} // namespace SortLib::detail

#endif // ENGRAM_SORT_LIB_ONE_CORE_H
