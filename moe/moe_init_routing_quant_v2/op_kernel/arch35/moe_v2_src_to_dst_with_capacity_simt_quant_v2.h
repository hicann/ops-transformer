/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_src_to_dst_with_capacity_simt_quant_v2.h
 * \brief
 */
#ifndef MOE_V2_SRC_TO_DST_WITH_CAPACITY_SIMT_QUANT_V2_H
#define MOE_V2_SRC_TO_DST_WITH_CAPACITY_SIMT_QUANT_V2_H

#include "moe_init_routing_quant_v2_arch35_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;

template <typename T, typename TilingData>
class MoeV2SrcToDstWithCapacitySimt {
public:
    __aicore__ inline MoeV2SrcToDstWithCapacitySimt(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                                const TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncAll();

private:
    __gm__ int32_t *expertFirstIndexGm_;
    __gm__ int32_t *sortedExpertIdGm_;
    __gm__ int32_t *sortedExpertIdIndexGm_;
    __gm__ int32_t *expandedRowIdxGm_;
    __gm__ int32_t *expandedRowIdxIndexGm_;
    __gm__ int32_t *oriExpertIdGm_;

    const InnerMoeV2GatherOutComputeTilingData *srcToDstTilingData_;

    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t coreRows_;
    int64_t expertCapacity_;
    int64_t expertNum_;

    int32_t threadNum_ = 0;
    int32_t startIndex_;
};

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacitySimt<T, TilingData>::SyncAll()
{
    if (coreNum_ == 1) {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacitySimt<T, TilingData>::Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                                          GM_ADDR workspace,
                                                                          const TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    coreNum_ = tilingData->coreNum;
    totalLength_ = tilingData->n * tilingData->k;
    srcToDstTilingData_ = &(tilingData->srcToDstCapacityComputeParamsOp);
    expertCapacity_ = tilingData->expertCapacity;
    expertNum_ = tilingData->expertNum;
    startIndex_ = blockIdx_ * srcToDstTilingData_->perCoreRows;
    if (blockIdx_ == srcToDstTilingData_->needCoreNum - 1) {
        coreRows_ = srcToDstTilingData_->lastCoreRows;
    } else {
        coreRows_ = srcToDstTilingData_->perCoreRows;
    }
    threadNum_ = coreRows_ > THREAD_NUM ? THREAD_NUM : coreRows_;
    int64_t length = MoeInitRoutingQuantV2::Align(totalLength_, sizeof(int32_t));

    sortedExpertIdGm_ = (__gm__ int32_t *)workspace;
    sortedExpertIdIndexGm_ = (__gm__ int32_t *)workspace + length;
    expertFirstIndexGm_ = (__gm__ int32_t *)workspace + length * 2;
    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;
    expandedRowIdxIndexGm_ = (__gm__ int32_t *)workspace + length * 2 + expertNum_;
    oriExpertIdGm_ = (__gm__ int32_t *)workspace + length * 3 + expertNum_;
}

template <typename T, typename TilingData>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeCapacitySimt(
    int64_t coreRows, int64_t startIndex, int64_t expertCapacity, __gm__ int32_t *sortedExpertIdGm,
    __gm__ int32_t *sortedExpertIdIndexGm, __gm__ volatile int32_t *expertFirstIndexGm,
    __gm__ int32_t *expandedRowIdxGm, __gm__ int32_t *expandedRowIdxIndexGm, __gm__ int32_t *oriExpertIdGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int32_t curIndex = index + startIndex;
        int32_t expertId = sortedExpertIdGm[curIndex];
        int32_t expertIdFirstIndex = expertFirstIndexGm[expertId];
        int32_t curExpertIndex = curIndex - expertIdFirstIndex;
        if (curExpertIndex + 1 <= static_cast<int32_t>(expertCapacity)) {
            int32_t dstIndex = sortedExpertIdIndexGm[curIndex];
            int32_t expandedRowIdxIndex = expertId * expertCapacity + curExpertIndex;
            expandedRowIdxGm[dstIndex] = expandedRowIdxIndex;
            expandedRowIdxIndexGm[curIndex] = expandedRowIdxIndex; // 记录原始expandedRowIdxIndex
            oriExpertIdGm[dstIndex] = expertId;                    // 记录原始专家号
        } else {
            int32_t expandedRowIdxIndex = (expertId + 1) * expertCapacity - 1;
            expandedRowIdxIndexGm[curIndex] = expandedRowIdxIndex;
        }
    }
}

template <typename T, typename TilingData>
__aicore__ inline void MoeV2SrcToDstWithCapacitySimt<T, TilingData>::Process()
{
    if (blockIdx_ < srcToDstTilingData_->needCoreNum) {
        asc_vf_call<ComputeCapacitySimt<T, TilingData>>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1},
                                                        this->coreRows_, this->startIndex_, this->expertCapacity_,
                                                        sortedExpertIdGm_, sortedExpertIdIndexGm_, expertFirstIndexGm_,
                                                        expandedRowIdxGm_, expandedRowIdxIndexGm_, oriExpertIdGm_);
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_SRC_TO_DST_WITH_CAPACITY_SIMT_QUANT_V2_H
