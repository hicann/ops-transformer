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
 * \file moe_src_to_dst_op_simt.h
 * \brief
 */
#ifndef MOE_V2_SRC_TO_DST_SIMT_ARCH35_H
#define MOE_V2_SRC_TO_DST_SIMT_ARCH35_H

#include "moe_init_routing_v2_arch35_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

class MoeV2SrcToDstOpSimt {
public:
    __aicore__ inline MoeV2SrcToDstOpSimt(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow, const TilingData *tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutSync();

private:
    __aicore__ inline void SyncAll();

private:
    __gm__ int32_t *expandDstToSrcRowGm_;
    __gm__ int32_t *expandedRowIdxGm_;
    const MoeV2GatherOutComputeTilingData *srcToDstTilingData_;

    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t perCoreRows_;
    int64_t coreRows_;
    int64_t startIndex_;
    int64_t threadNum_;
};

__aicore__ inline void MoeV2SrcToDstOpSimt::SyncAll()
{
    if (coreNum_ == 1) {
        return;
    }
    AscendC::SyncAll();
}

template <typename TilingData>
__aicore__ inline void MoeV2SrcToDstOpSimt::Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow,
                                                 const TilingData *tilingData)
{
    this->blockIdx_ = GetBlockIdx();
    this->coreNum_ = tilingData->coreNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData_ = &(tilingData->srcToDstComputeParamsOp);
    this->perCoreRows_ = this->srcToDstTilingData_->perCoreRows;
    if (this->blockIdx_ == this->srcToDstTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->srcToDstTilingData_->lastCoreRows;
    } else {
        this->coreRows_ = this->srcToDstTilingData_->perCoreRows;
    }
    startIndex_ = this->blockIdx_ * this->perCoreRows_;
    this->threadNum_ = THREAD_NUM < this->coreRows_ ? THREAD_NUM : this->coreRows_;

    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;
    expandDstToSrcRowGm_ = (__gm__ int32_t *)expandDstToSrcRow + Align(this->totalLength_, sizeof(int32_t));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(int64_t coreRows, int64_t startIndex,
                                                                        __gm__ int32_t *expandDstToSrcRowGm,
                                                                        __gm__ int32_t *expandedRowIdxGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t srcIndex = index + startIndex;
        int64_t dstIndex = expandDstToSrcRowGm[srcIndex];
        expandedRowIdxGm[dstIndex] = srcIndex;
    }
}

__aicore__ inline void MoeV2SrcToDstOpSimt::Process()
{
    this->ProcessWithoutSync();
    this->SyncAll();
}

__aicore__ inline void MoeV2SrcToDstOpSimt::ProcessWithoutSync()
{
    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        asc_vf_call<ComputeSimt>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1}, this->coreRows_,
                                 this->startIndex_, expandDstToSrcRowGm_, expandedRowIdxGm_);
    }
}
} // namespace MoeInitRoutingV2
#endif // MOE_SRC_TO_DST_SIMT_H
