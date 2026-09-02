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
 * \file moe_v2_expert_token_out_simt_quant_v2.h
 * \brief
 */
#ifndef MOE_V2_QUANT_EXPERT_TOKEN_OUT_SIMT_H
#define MOE_V2_QUANT_EXPERT_TOKEN_OUT_SIMT_H

#include "moe_init_routing_quant_v2_arch35_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;

class MoeV2ExpertTokenOutSimt {
public:
    __aicore__ inline MoeV2ExpertTokenOutSimt(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                GM_ADDR expandedRowIdx, GM_ADDR workspace, const TilingData *tilingData, TPipe *tPipe);
    template <bool CALCHIST = true>
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncAll();
    __aicore__ inline void InitLocal();

private:
    TPipe *pipe;
    GlobalTensor<int32_t> expandedRowIdxGmGT_;
    GlobalTensor<int32_t> expandedExpertIdxGmGT_;
    __gm__ int32_t *expandedRowIdxGm_;
    __gm__ int32_t *expandedExpertIdxGm_;
    __gm__ int32_t *expertTokensBeforeCapacityGm_;
    __gm__ int32_t *expertTokensCountOrCumsumGm_;
    GlobalTensor<int32_t> expertTokensCountOrCumsumGmOut;
    __gm__ int32_t *expertFirstIndexGm_;

    const InnerMoeV2GatherOutComputeTilingData *srcToDstTilingData;

    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t expertNum_;
    int64_t dropPadMode_ = 0;
    int64_t expertTokensCountOrCumsumFlag_ = 0;
    int64_t expertTokensBeforeCapacityFlag_ = 0;
    int32_t startIndex_;
    int64_t needCoreNum_ = 0;
    int32_t threadNum_;
    bool expertCount_ = false;
    bool expertCumsum_ = false;
};

__aicore__ inline void MoeV2ExpertTokenOutSimt::InitLocal()
{
    // expandedRowIdx initialized to -1, which is used in the src_to_dst_with_capacity step.
    // use this step SyncAll to synchronize every core data
    if (this->dropPadMode_ == 0) {
        return;
    }
    InitOutput(expandedRowIdxGmGT_[this->startIndex_], this->coreRows_, static_cast<int32_t>(-1));
}

__aicore__ inline void MoeV2ExpertTokenOutSimt::SyncAll()
{
    if (this->coreNum_ == 1) {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename TilingData>
__aicore__ inline void MoeV2ExpertTokenOutSimt::Init(GM_ADDR expertTokensCountOrCumsum,
                                                     GM_ADDR expertTokensBeforeCapacity, GM_ADDR expandedRowIdx,
                                                     GM_ADDR workspace, const TilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx_ = GetBlockIdx();

    this->coreNum_ = tilingData->coreNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);
    this->expertNum_ = tilingData->expertNum;
    this->dropPadMode_ = tilingData->dropPadMode;
    this->expertTokensCountOrCumsumFlag_ = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag_ = tilingData->expertTokensBeforeCapacityFlag;
    this->needCoreNum_ = this->srcToDstTilingData->needCoreNum;

    this->startIndex_ = this->blockIdx_ * this->srcToDstTilingData->perCoreRows;

    if (this->blockIdx_ == this->needCoreNum_ - 1) {
        this->coreRows_ = this->srcToDstTilingData->lastCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData->lastCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData->lastCoreLastLoopRows;
    } else {
        this->coreRows_ = this->srcToDstTilingData->perCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData->perCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData->perCoreLastLoopRows;
    }
    this->threadNum_ = this->coreRows_ > THREAD_NUM ? THREAD_NUM : this->coreRows_;

    expandedRowIdxGmGT_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, Align(this->totalLength_, sizeof(int32_t)));
    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;

    if (this->dropPadMode_ == DROPLESS_MODE && this->expertTokensCountOrCumsumFlag_ > EXERPT_TOKENS_NONE) {
        if (this->expertTokensCountOrCumsumFlag_ == EXERPT_TOKENS_COUNT) {
            expertTokensBeforeCapacityGm_ = (__gm__ int32_t *)expertTokensCountOrCumsum;
            this->expertCount_ = true;
        } else {
            expertTokensCountOrCumsumGm_ = (__gm__ int32_t *)expertTokensCountOrCumsum;
            expertTokensCountOrCumsumGmOut.SetGlobalBuffer((__gm__ int32_t *)expertTokensCountOrCumsum,
                                                           this->expertNum_);
            this->expertCumsum_ = true;
        }
    }

    if (this->dropPadMode_ == DROP_PAD_MODE && this->expertTokensBeforeCapacityFlag_ == EXERPT_TOKENS_BEFORE_CAPACITY) {
        expertTokensBeforeCapacityGm_ = (__gm__ int32_t *)expertTokensBeforeCapacity;
        this->expertCount_ = true;
    }

    expertFirstIndexGm_ = (__gm__ int32_t *)workspace + Align(this->totalLength_, sizeof(int32_t)) * 2;
    expandedExpertIdxGm_ = (__gm__ int32_t *)workspace;
    expandedExpertIdxGmGT_.SetGlobalBuffer((__gm__ int32_t *)workspace, Align(this->totalLength_, sizeof(int32_t)));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ExpertFirstIndexComputeSimt(
    int64_t coreRows, int64_t startIndex, int64_t totalLength, __gm__ int32_t *expertFirstIndexGm,
    __gm__ int32_t *expandedExpertIdxGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t curIndex = index + startIndex;
        int32_t curExpertId = expandedExpertIdxGm[curIndex];
        if (curIndex == 0) {
            expertFirstIndexGm[curExpertId] = 0;
        }
        if (curIndex == totalLength - 1) {
            return;
        }
        int64_t nextIndex = curIndex + 1;
        int32_t nextExpertId = expandedExpertIdxGm[nextIndex];
        if (curExpertId != nextExpertId) {
            expertFirstIndexGm[nextExpertId] = nextIndex;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void TokensComputeSimt(
    int64_t coreRows, int64_t startIndex, int64_t totalLength, __gm__ int32_t *expertFirstIndexGm,
    __gm__ int32_t *expandedExpertIdxGm, __gm__ int32_t *expertTokensBeforeCapacityGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t curIndex = index + startIndex;
        int64_t nextIndex = curIndex + 1;
        int32_t curExpertId = expandedExpertIdxGm[curIndex];
        if (curIndex == totalLength - 1) { // 最后一个数
            expertTokensBeforeCapacityGm[curExpertId] = nextIndex - expertFirstIndexGm[curExpertId];
            return;
        }
        int32_t nextExpertId = expandedExpertIdxGm[nextIndex];
        if (curExpertId != nextExpertId) {
            expertTokensBeforeCapacityGm[curExpertId] = nextIndex - expertFirstIndexGm[curExpertId];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void CumsumComputeSimt(
    int64_t coreRows, int64_t startIndex, int64_t totalLength, __gm__ int32_t *expandedExpertIdxGm,
    __gm__ int32_t *expertTokensCountOrCumsumGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t curIndex = startIndex + index;
        int32_t curExpertId = expandedExpertIdxGm[curIndex];
        if (curIndex == totalLength - 1) { // 最后一个数
            return;
        }
        int32_t nextExpertId = expandedExpertIdxGm[curIndex + 1];
        if (curExpertId != nextExpertId) {
            int32_t count = nextExpertId - curExpertId;
            int32_t num = threadIdx.y;
            for (; num < count; num += blockDim.y) {
                expertTokensCountOrCumsumGm[curExpertId + num] = curIndex + 1;
            }
        }
    }
}

template <bool CALCHIST>
__aicore__ inline void MoeV2ExpertTokenOutSimt::Process()
{
    if (this->blockIdx_ < this->needCoreNum_) {
        InitLocal();
    }

    if (this->blockIdx_ < this->needCoreNum_) {
        asc_vf_call<ExpertFirstIndexComputeSimt>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1}, this->coreRows_,
                                                 this->startIndex_, this->totalLength_, expertFirstIndexGm_,
                                                 expandedExpertIdxGm_);
    }

    this->SyncAll();

    if constexpr (!CALCHIST) {
        return;
    }

    if (this->expertCount_ && this->blockIdx_ < this->needCoreNum_) {
        asc_vf_call<TokensComputeSimt>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1}, this->coreRows_,
                                       this->startIndex_, this->totalLength_, expertFirstIndexGm_, expandedExpertIdxGm_,
                                       expertTokensBeforeCapacityGm_);
    }

    if (this->expertCumsum_ && this->blockIdx_ < this->needCoreNum_) {
        int32_t threadDimY =
            THREAD_NUM / this->threadNum_ > MAX_THREAD_DIM_Y_NUM ? MAX_THREAD_DIM_Y_NUM : THREAD_NUM / this->threadNum_;
        asc_vf_call<CumsumComputeSimt>(
            dim3{static_cast<uint32_t>(this->threadNum_), static_cast<uint32_t>(threadDimY), 1}, this->coreRows_,
            this->startIndex_, this->totalLength_, expandedExpertIdxGm_, expertTokensCountOrCumsumGm_);
    }

    if (this->expertCumsum_ && this->blockIdx_ == this->needCoreNum_ - 1) {
        int32_t expertId = expandedExpertIdxGmGT_.GetValue(this->totalLength_ - 1);
        int32_t cunsumNum = static_cast<int32_t>(this->totalLength_);
        int32_t count = expertNum_ - expertId;
        if (count > 0) {
            InitOutput(expertTokensCountOrCumsumGmOut[expertId], static_cast<uint32_t>(count), cunsumNum);
        }
    }
}
} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_QUANT_EXPERT_TOKEN_OUT_SIMT_H
