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
 * \file moe_src_to_dst_simt_op.h
 * \brief
 */
#ifndef MOE_SRC_TO_DST_SIMT_H
#define MOE_SRC_TO_DST_SIMT_H

#include "moe_init_routing_arch35_common.h"
#include "simt_api/asc_simt.h"

#define THREAD_NUM 2048

namespace MoeInitRouting {
using namespace AscendC;

class MoeSrcToDstSimtOp {
public:
    __aicore__ inline MoeSrcToDstSimtOp(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow,
                                const MoeInitRoutingTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __gm__ int32_t *expandDstToSrcRowGm_;
    __gm__ int32_t *expandedRowIdxGm_;

    const GatherOutComputeTilingData *srcToDstTilingData_;

    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t perCoreRows_;
    int64_t coreRows_;
    int32_t threadNum_;
};

__aicore__ inline void MoeSrcToDstSimtOp::Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow,
                                               const MoeInitRoutingTilingData *tilingData)
{
    int64_t blockNum_ = GetBlockNum();
    this->blockIdx_ = GetBlockIdx();

    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData_ = &(tilingData->srcToDstComputeParamsOp);
    this->perCoreRows_ = this->srcToDstTilingData_->perCoreRows;
    if (this->blockIdx_ == this->srcToDstTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->srcToDstTilingData_->lastCoreRows;
    } else {
        this->coreRows_ = this->srcToDstTilingData_->perCoreRows;
    }
    this->threadNum_ = THREAD_NUM < this->coreRows_ ? THREAD_NUM : this->coreRows_;

    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;
    expandDstToSrcRowGm_ = (__gm__ int32_t *)expandDstToSrcRow;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(int64_t coreRows, int64_t startIndex,
                                                                        __gm__ int32_t *expandDstToSrcRowGm,
                                                                        __gm__ int32_t *expandedRowIdxGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int32_t srcIndex = index + startIndex;
        int32_t dstIndex = expandDstToSrcRowGm[srcIndex];
        expandedRowIdxGm[dstIndex] = srcIndex;
    }
}

__aicore__ inline void MoeSrcToDstSimtOp::Process()
{
    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        int32_t startIndex = this->blockIdx_ * this->perCoreRows_;
        asc_vf_call<ComputeSimt>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1}, this->coreRows_, startIndex,
                                 expandDstToSrcRowGm_, expandedRowIdxGm_);
    }
}
} // namespace MoeInitRouting
#endif // MOE_SRC_TO_DST_SIMT_H
