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
 * \file moe_token_permute_with_routing_map_grad_kernel.h
 * \brief
 */

#ifndef MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_KERNEL_H
#define MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_KERNEL_H

#include "kernel_operator.h"
#include "moe_token_unpermute_routing_map_grad.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class KernelMoeTokenPermuteWithRoutingMap {
public:
    __aicore__ inline KernelMoeTokenPermuteWithRoutingMap() {}

    __aicore__ inline void Init(GM_ADDR permuted_probs, GM_ADDR sorted_indices, GM_ADDR probs_grad_out,
                                const MoeTokenpermuteWithRoutingMapDropPadTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitData(const MoeTokenpermuteWithRoutingMapDropPadTilingData &tilingData);

private:
    static constexpr uint64_t BLOCK_SIZE = 32;
    static constexpr uint64_t BUFFER_NUM = 1;
    int64_t blockIdx_;
    int64_t numBlocks_;
    int64_t tokenNum_;
    int64_t expertNum_;
    int64_t capacity_;
    int64_t singleCoreLen_;
    int64_t lastCoreLen_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ProbGradOutQue;
    GlobalTensor<T1> permutedProbsGM;
    GlobalTensor<T2> sortedIndicesGM;
    GlobalTensor<T3> probsGradOutGM;
};

template <typename T1, typename T2, typename T3>
__aicore__ inline void KernelMoeTokenPermuteWithRoutingMap<T1, T2, T3>::InitData(
    const MoeTokenpermuteWithRoutingMapDropPadTilingData &tilingData)
{
    blockIdx_ = AscendC::GetBlockIdx();
    numBlocks_ = AscendC::GetBlockNum();
    tokenNum_ = tilingData.tokenNum;
    expertNum_ = tilingData.expertNum;
    capacity_ = tilingData.capacity;
    singleCoreLen_ = tilingData.singleCoreLen;
    lastCoreLen_ = tilingData.lastCoreLen;
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    ASSERT(capacity_ != 0 && "capacity can not be zero!");
    ASSERT(tokenNum_ != 0 && "tokenNum can not be zero!");
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void KernelMoeTokenPermuteWithRoutingMap<T1, T2, T3>::Init(
    GM_ADDR permuted_probs, GM_ADDR sorted_indices, GM_ADDR probs_grad_out,
    const MoeTokenpermuteWithRoutingMapDropPadTilingData *__restrict tilingData)
{
    InitData(*tilingData);
    permutedProbsGM.SetGlobalBuffer((__gm__ T1 *)permuted_probs);
    sortedIndicesGM.SetGlobalBuffer((__gm__ T2 *)sorted_indices);
    probsGradOutGM.SetGlobalBuffer((__gm__ T3 *)probs_grad_out);

    pipe_.InitBuffer(ProbGradOutQue, BUFFER_NUM, BLOCK_SIZE);
    return;
};

template <typename T1, typename T2, typename T3>
__aicore__ inline void KernelMoeTokenPermuteWithRoutingMap<T1, T2, T3>::Process()
{
    LocalTensor<T3> ProbGradOutTensor = ProbGradOutQue.template AllocTensor<T3>();
    int64_t startIdx = blockIdx_ * singleCoreLen_;
    int64_t processNum = singleCoreLen_;
    DataCopyExtParams copyParams{1, sizeof(T3), 0, 0, 0};
    if (blockIdx_ + 1 == numBlocks_) {
        processNum = lastCoreLen_;
    }
    int64_t endIdx = startIdx + processNum;
    event_t eventIDSToMTE3 = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
    T1 permuteProbsOuputTmp;
    int64_t xIndex, yIndex, outOffset;
    for (int64_t i = startIdx; i < endIdx; i++) {
        permuteProbsOuputTmp = permutedProbsGM.GetValue(i);
        xIndex = sortedIndicesGM.GetValue(i);
        yIndex = i / capacity_;
        outOffset = xIndex * expertNum_ + yIndex;
        ProbGradOutTensor.SetValue(0, permuteProbsOuputTmp);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        DataCopyPad(probsGradOutGM[outOffset], ProbGradOutTensor, copyParams);
    }
    return;
}
#endif // MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_KERNEL_H
