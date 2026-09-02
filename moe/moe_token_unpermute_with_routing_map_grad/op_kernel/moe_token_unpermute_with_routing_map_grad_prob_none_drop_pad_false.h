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
 * \file moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_false.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NONE_DROP_PAD_FALSE_H
#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NONE_DROP_PAD_FALSE_H

#include "moe_token_unpermute_with_routing_map_grad_base.h"

namespace MoeTokenUnpermuteWithRoutingMapGrad {
using namespace AscendC;

template <typename PermutedTokenT, typename IdxT, typename ProbsT = PermutedTokenT>
class MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse
    : protected MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT> {
public:
    __aicore__ inline MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse(){};
    __aicore__ inline void Init(GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId,
                                GM_ADDR routing_map, GM_ADDR permuted_tokens, GM_ADDR probs,
                                GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                const MoeTokenUnpermuteWithRoutingMapGradTilingData &tiling_data);
    __aicore__ inline void Process();

protected:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> padFalseInOutque;
    LocalTensor<PermutedTokenT> inOutLocal;

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
};

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse<PermutedTokenT, IdxT, ProbsT>::Init(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
    const MoeTokenUnpermuteWithRoutingMapGradTilingData &tiling_data)
{
    MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::Init(
        unpermuted_tokens_grad, outIndex, permuteTokenId, routing_map, permuted_tokens, probs, permuted_tokens_grad,
        probs_grad, tiling_data);
    this->pipe.InitBuffer(padFalseInOutque, DOUBLE_BUFFER,
                          (this->inputReserveNum * this->hiddenSizeAlign) * this->inputTypeSize);
}

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse<PermutedTokenT, IdxT, ProbsT>::Process()
{
    int64_t outNumCurrentCore = this->coreIndex < this->formerCoreNum ? this->rowIdMapEachCore : this->rowIdMapTailCore;
    int64_t inputBlockNum = BLOCK_SIZE_32 / this->inputTypeSize;
    for (int64_t indicesLoopTime = 0; indicesLoopTime < outNumCurrentCore;
         indicesLoopTime++) { // 外循环是根据x的数量循环
        int64_t rowIdMapLoopOffset = this->rowIdMapStartOffset + indicesLoopTime;
        int64_t tokenId = this->SafeDiv(rowIdMapLoopOffset, this->topK);
        int64_t permuteTokenId = this->sortedTwiceIndexGm.GetValue(rowIdMapLoopOffset);
        if (permuteTokenId < 0) {
            continue;
        }
        SToMTE2Sync();
        for (int64_t hiddenLoop = 0; hiddenLoop < this->hiddenSizeLoopTimes; hiddenLoop++) {
            uint32_t hiddenLoopNum =
                hiddenLoop == this->hiddenSizeLoopTimes - 1 ? this->hiddenSizeTail : this->hiddenSizeAlign;
            uint32_t hiddenLoopBlockLen = hiddenLoopNum * this->inputTypeSize;
            int64_t hiddenLoopOffset = hiddenLoop * this->hiddenSizeAlign;
            inOutLocal = padFalseInOutque.AllocTensor<PermutedTokenT>();
            copyParams.blockLen = hiddenLoopBlockLen;
            int64_t unpermutedTokensGradOffset = tokenId * this->hiddenSize + hiddenLoopOffset;
            DataCopyPad(inOutLocal, this->unpermutedTokensGradGm[unpermutedTokensGradOffset], copyParams,
                        this->inputPadParams);
            padFalseInOutque.EnQue<QuePosition::VECIN, QuePosition::VECOUT, PermutedTokenT>(inOutLocal);
            inOutLocal = padFalseInOutque.DeQue<QuePosition::VECIN, QuePosition::VECOUT, PermutedTokenT>();
            int64_t permutedTokensGradOffset = permuteTokenId * this->hiddenSize + hiddenLoopOffset;
            DataCopyPad(this->permutedTokensGradGm[permutedTokensGradOffset], inOutLocal, copyParams);
            padFalseInOutque.FreeTensor(inOutLocal);
        }
    }
}

} // namespace MoeTokenUnpermuteWithRoutingMapGrad
#endif // MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NONE_DROP_PAD_FALSE_H
