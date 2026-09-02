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
 * \file moe_token_unpermute_grad_prob_none.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_GRAD_PROB_NONE_H
#define MOE_TOKEN_UNPERMUTE_GRAD_PROB_NONE_H

#include "moe_token_unpermute_grad_base.h"

namespace MoeTokenUnpermuteGrad {
using namespace AscendC;

template <typename OriT, typename IdxT>
class MoeTokenUnpermuteGradProbNone : protected MoeTokenUnpermuteGradBase<OriT, IdxT> {
public:
    __aicore__ inline MoeTokenUnpermuteGradProbNone(){};
    __aicore__ inline void Init(GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices,
                                GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                const MoeTokenUnpermuteGradTilingData &tilingData);
    __aicore__ inline void Process();

protected:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutque; // unpermuted_tokens_grad (x, h)
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueRowIdMap;       // row_id_map (indicesNumPerLoop,)
    LocalTensor<OriT> inOutLocal;
    LocalTensor<IdxT> inQueueRowIdMapLocal;

    DataCopyExtParams copyParams{0, 0, 0, 0, 0};
};

template <typename OriT, typename IdxT>
__aicore__ inline void MoeTokenUnpermuteGradProbNone<OriT, IdxT>::Init(
    GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices, GM_ADDR probs,
    GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, const MoeTokenUnpermuteGradTilingData &tilingData)
{
    MoeTokenUnpermuteGradBase<OriT, IdxT>::Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs,
                                                permuted_tokens_grad, probs_grad, tilingData);
    this->pipe.InitBuffer(inQueueRowIdMap, DOUBLE_BUFFER, this->indicesReserveNum * this->rowIdMapTypeSize);
    this->pipe.InitBuffer(inOutque, DOUBLE_BUFFER,
                          (this->inputReserveNum * this->hiddenSizeAlign) * this->inputTypeSize);
}

template <typename OriT, typename IdxT>
__aicore__ inline void MoeTokenUnpermuteGradProbNone<OriT, IdxT>::Process()
{
    int64_t indicesEachLoop = this->indicesReserveNum;
    int64_t indicesLoopTimes = 0;
    int64_t indicesLastLoop = 0;
    int64_t indicesCoreNum = 0;
    if (this->coreIndex < this->formerCoreNum) {
        indicesLoopTimes = (this->rowIdMapEachCore + indicesEachLoop - 1) / indicesEachLoop;
        indicesLastLoop = this->rowIdMapEachCore - indicesEachLoop * (indicesLoopTimes - 1);
        indicesCoreNum = this->rowIdMapEachCore;
    } else {
        indicesLoopTimes = (this->rowIdMapTailCore + indicesEachLoop - 1) / indicesEachLoop;
        indicesLastLoop = this->rowIdMapTailCore - indicesEachLoop * (indicesLoopTimes - 1);
        indicesCoreNum = this->rowIdMapTailCore;
    }
    int64_t inputBlockNum = BLOCK_SIZE_32 / this->inputTypeSize;

    for (int64_t indicesLoopTime = 0; indicesLoopTime < indicesLoopTimes;
         indicesLoopTime++) { // 外循环是根据x的数量循环
        uint16_t indicesLoopNum = indicesLoopTime == indicesLoopTimes - 1 ? indicesLastLoop : indicesEachLoop;
        int64_t rowIdMapLoopOffset = this->rowIdMapStartOffset + indicesLoopTime * indicesEachLoop;
        // copy in rowidmap, copyParams值为{1, indicesLoopNum * this->rowIdMapTypeSize, 0, 0, 0}
        inQueueRowIdMapLocal = inQueueRowIdMap.template AllocTensor<IdxT>();
        copyParams.blockCount = 1;
        copyParams.blockLen = indicesLoopNum * this->rowIdMapTypeSize;
        copyParams.srcStride = 0;
        DataCopyPad(inQueueRowIdMapLocal, this->sortedIndicesGm[rowIdMapLoopOffset], copyParams,
                    this->rowIdMapPadParams);
        inQueueRowIdMap.EnQue(inQueueRowIdMapLocal);
        inQueueRowIdMapLocal = inQueueRowIdMap.template DeQue<IdxT>();

        for (int64_t hiddenLoop = 0; hiddenLoop < this->hiddenSizeLoopTimes; hiddenLoop++) {
            uint32_t hiddenLoopNum =
                hiddenLoop == this->hiddenSizeLoopTimes - 1 ? this->hiddenSizeTail : this->hiddenSizeAlign;
            uint32_t hiddenLoopBlockLen = hiddenLoopNum * this->inputTypeSize;
            int64_t hiddenLoopOffset = hiddenLoop * this->hiddenSizeAlign;
            // copy in unpermuted_tokens_grad,一次搬(x,h)
            // copyParams值为{indicesLoopNum, hiddenLoopNum * this->inputTypeSize, (this->hiddenSize - hiddenLoopNum) *
            // this->inputTypeSize, 0, 0}
            inOutLocal = inOutque.AllocTensor<OriT>();
            copyParams.blockCount = indicesLoopNum;
            copyParams.blockLen = hiddenLoopBlockLen;
            copyParams.srcStride = static_cast<uint32_t>(this->hiddenSize - hiddenLoopNum) * this->inputTypeSize;
            int64_t unpermutedOutputDOffset = rowIdMapLoopOffset * this->hiddenSize + hiddenLoopOffset;
            DataCopyPad(inOutLocal, this->unpermutedOutputDGm[unpermutedOutputDOffset], copyParams,
                        this->inputPadParams);
            inOutque.EnQue<QuePosition::VECIN, QuePosition::VECOUT, OriT>(inOutLocal);
            inOutLocal = inOutque.DeQue<QuePosition::VECIN, QuePosition::VECOUT, OriT>();

            // copy out permuted_tokens_grad,循环x搬出
            // copyParams值为{1, hiddenLoopNum * this->inputTypeSize, 0, 0, 0}
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            int64_t hiddenSizeUbAlign =
                (hiddenLoopNum + inputBlockNum - 1) / inputBlockNum * inputBlockNum; // pad搬运自动补齐32B
            for (int64_t index = 0; index < indicesLoopNum; index++) {
                int64_t offset = inQueueRowIdMapLocal.GetValue(index);
                int64_t permutedTokensGradOffset = offset * this->hiddenSize + hiddenLoopOffset;
                int64_t inOutOffset = index * hiddenSizeUbAlign;
                if (offset < this->numOutTokens) {
                    DataCopyPad(this->permutedTokensGradGm[permutedTokensGradOffset], inOutLocal[inOutOffset],
                                copyParams);
                }
            }
            inOutque.FreeTensor(inOutLocal);
        }
        inQueueRowIdMap.FreeTensor(inQueueRowIdMapLocal);
    }
}

} // namespace MoeTokenUnpermuteGrad
#endif // MOE_TOKEN_UNPERMUTE_GRAD_PROB_NONE_H
