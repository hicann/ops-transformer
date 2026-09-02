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
 * \file moe_v3_topk_weight_out.h
 * \brief TopkWeight rearrangement for non-FullLoad path (arch35/regbase)
 */
#ifndef MOE_V3_TOPK_WEIGHT_OUT_H
#define MOE_V3_TOPK_WEIGHT_OUT_H

#include "moe_v3_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t TOPK_WEIGHT_SIMT_THREAD_NUM = 1024;

__simt_vf__ __aicore__ LAUNCH_BOUND(TOPK_WEIGHT_SIMT_THREAD_NUM) inline void TopkWeightGatherSimt(
    int64_t elements, int64_t indexBase, int64_t totalLength, __gm__ int32_t *dstToSrcRow, __gm__ float *topkWeight,
    __gm__ volatile float *expandedTopkWeight)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < elements;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t dstIndex = indexBase + index;
        int32_t srcIndex = dstToSrcRow[dstIndex];
        if (srcIndex >= 0 && srcIndex < totalLength) {
            expandedTopkWeight[dstIndex] = topkWeight[srcIndex];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(TOPK_WEIGHT_SIMT_THREAD_NUM) inline void TopkWeightScatterSimt(
    int64_t elements, int64_t indexBase, int64_t outputRows, __gm__ int32_t *srcToDstRow, __gm__ float *topkWeight,
    __gm__ volatile float *expandedTopkWeight)
{
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < elements;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t srcIndex = indexBase + index;
        int32_t dstIndex = srcToDstRow[srcIndex];
        if (dstIndex >= 0 && dstIndex < outputRows) {
            expandedTopkWeight[dstIndex] = topkWeight[srcIndex];
        }
    }
}

class MoeV3TopkWeightOut {
public:
    __aicore__ inline MoeV3TopkWeightOut() {}
    __aicore__ inline void Init(GM_ADDR topkWeight, GM_ADDR expandedRowIdx, GM_ADDR expandedTopkWeight,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<float> topkWeightGm_;
    GlobalTensor<float> expandedTopkWeightGm_;
    GlobalTensor<int32_t> dstToSrcRowGm_;
    GlobalTensor<int32_t> srcToDstRowGm_;

    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t outputRows_;
    int64_t dropPadMode_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t coreElements_;
    int64_t coreNum_;
};

__aicore__ inline void MoeV3TopkWeightOut::Init(GM_ADDR topkWeight, GM_ADDR expandedRowIdx, GM_ADDR expandedTopkWeight,
                                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    coreNum_ = tilingData->coreNum;
    totalLength_ = static_cast<int64_t>(tilingData->n) * tilingData->k;
    dropPadMode_ = tilingData->dropPadMode;
    outputRows_ = (dropPadMode_ == DROP_PAD_MODE) ?
                      static_cast<int64_t>(tilingData->expertNum) * tilingData->expertCapacity :
                      tilingData->activeNum;

    topkWeightGm_.SetGlobalBuffer((__gm__ float *)topkWeight, totalLength_);
    expandedTopkWeightGm_.SetGlobalBuffer((__gm__ float *)expandedTopkWeight, outputRows_);

    int64_t splitBase = totalLength_;
    if (dropPadMode_ == DROP_PAD_MODE) {
        srcToDstRowGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, totalLength_);
    } else {
        GlobalTensor<int32_t> expertTotalCountGm;
        expertTotalCountGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(totalLength_, sizeof(int32_t)) * 2 +
                                               Align(tilingData->actualExpertNum, sizeof(int32_t)),
                                           1);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm);
        splitBase = expertTotalCountGm.GetValue(0);

        if (tilingData->rowIdxType == SCATTER) {
            dstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, totalLength_);
        } else {
            dstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(totalLength_, sizeof(int32_t)),
                                           Align(totalLength_, sizeof(int32_t)));
        }
    }

    perCoreElements_ = Ceil(splitBase, coreNum_);
    needCoreNum_ = Ceil(splitBase, perCoreElements_);
    coreElements_ =
        (blockIdx_ == needCoreNum_ - 1) ? splitBase - (needCoreNum_ - 1) * perCoreElements_ : perCoreElements_;
}

__aicore__ inline void MoeV3TopkWeightOut::Process()
{
    int64_t startIdx = blockIdx_ * perCoreElements_;

    if (dropPadMode_ == DROP_PAD_MODE) {
        int64_t perCoreZeroRows = Ceil(outputRows_, coreNum_);
        int64_t startRow = blockIdx_ * perCoreZeroRows;
        int64_t endRow = Min(startRow + perCoreZeroRows, outputRows_);
        if (startRow < endRow) {
            GlobalTensor<float> zeroGm;
            zeroGm.SetGlobalBuffer((__gm__ float *)expandedTopkWeightGm_.GetPhyAddr() + startRow, endRow - startRow);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            InitGlobalMemory(zeroGm, endRow - startRow, 0.0f);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
#ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
#endif

        if (blockIdx_ < needCoreNum_) {
            uint32_t threadNum = static_cast<uint32_t>(Min(coreElements_, TOPK_WEIGHT_SIMT_THREAD_NUM));
            asc_vf_call<TopkWeightScatterSimt>(dim3{threadNum, 1, 1}, coreElements_, startIdx, outputRows_,
                                               (__gm__ int32_t *)srcToDstRowGm_.GetPhyAddr(),
                                               (__gm__ float *)topkWeightGm_.GetPhyAddr(),
                                               (__gm__ volatile float *)expandedTopkWeightGm_.GetPhyAddr());
        }
        return;
    }

    if (blockIdx_ < needCoreNum_) {
        uint32_t threadNum = static_cast<uint32_t>(Min(coreElements_, TOPK_WEIGHT_SIMT_THREAD_NUM));
        asc_vf_call<TopkWeightGatherSimt>(
            dim3{threadNum, 1, 1}, coreElements_, startIdx, totalLength_, (__gm__ int32_t *)dstToSrcRowGm_.GetPhyAddr(),
            (__gm__ float *)topkWeightGm_.GetPhyAddr(), (__gm__ volatile float *)expandedTopkWeightGm_.GetPhyAddr());
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_TOPK_WEIGHT_OUT_H
