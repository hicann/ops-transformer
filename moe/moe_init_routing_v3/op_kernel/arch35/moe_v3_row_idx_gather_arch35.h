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
 * \file moe_v3_row_idx_gather_arch35.h
 * \brief
 */
#ifndef MOE_V3_ROW_IDX_GATHER_H_REGBASE_ARCH35
#define MOE_V3_ROW_IDX_GATHER_H_REGBASE_ARCH35

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class RowIdxGather {
public:
    __aicore__ inline RowIdxGather(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> sortedExpertIndicesGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t lastCoreElements_;
    int64_t curElements_;

    int64_t rowIdxType_ = 0;
    int64_t useGatherCopy_ = 0;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeSimt(int64_t elements, int64_t indexBase,
                                                                             __gm__ int32_t *sortedExpertIndicesGmAddr,
                                                                             __gm__ int32_t *expandedRowIdxGmAddr)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(elements);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t outIndices = sortedExpertIndicesGmAddr[index];
        expandedRowIdxGmAddr[outIndices] = indexBase + index;
    }
}

__aicore__ inline void RowIdxGather::Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                          const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = tilingData->expertTokensCountTilingDataOp.needCoreNum;
    perCoreElements_ = tilingData->expertTokensCountTilingDataOp.perCoreElements;
    rowIdxType_ = tilingData->rowIdxType;
    useGatherCopy_ = tilingData->useGatherCopy;

    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                            Align(tilingData->actualExpertNum, sizeof(int32_t)),
                                        1);
    int64_t expertTotalCount = expertTotalCountGm_.GetValue(0);

    perCoreElements_ = Ceil(expertTotalCount, needCoreNum_);
    needCoreNum_ = Ceil(expertTotalCount, perCoreElements_);
    lastCoreElements_ = expertTotalCount - (needCoreNum_ - 1) * perCoreElements_;
    curElements_ = (blockIdx_ == needCoreNum_ - 1 ? lastCoreElements_ : perCoreElements_);
    int64_t totalLength = Align(tilingData->n * tilingData->k, sizeof(int32_t));
    if (useGatherCopy_ && rowIdxType_) {
        sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + totalLength, totalLength);
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreElements_,
                                          curElements_);
    } else {
        sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + totalLength + blockIdx_ * perCoreElements_,
                                               curElements_);
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, totalLength);
    }
}

__aicore__ inline void RowIdxGather::Process()
{
    if (blockIdx_ < needCoreNum_) {
        __gm__ int32_t *expandedRowIdxGmAddr = (__gm__ int32_t *)expandedRowIdxGm_.GetPhyAddr();
        __gm__ int32_t *sortedExpertIndicesGmAddr = (__gm__ int32_t *)sortedExpertIndicesGm_.GetPhyAddr();
        if (useGatherCopy_ && rowIdxType_) {
            asc_vf_call<ComputeSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, curElements_, blockIdx_ * perCoreElements_,
                                     expandedRowIdxGmAddr, sortedExpertIndicesGmAddr);
        } else {
            asc_vf_call<ComputeSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, curElements_, blockIdx_ * perCoreElements_,
                                     sortedExpertIndicesGmAddr, expandedRowIdxGmAddr);
        }
    }
    if (useGatherCopy_) {
        SyncAll();
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_ROW_IDX_GATHER_H_REGBASE_ARCH35
