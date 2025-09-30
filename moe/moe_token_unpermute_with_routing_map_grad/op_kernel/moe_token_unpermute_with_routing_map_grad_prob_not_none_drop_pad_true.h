/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_true.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NOT_NONE_DROP_PAD_TRUE_H
#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NOT_NONE_DROP_PAD_TRUE_H
#include "moe_token_unpermute_with_routing_map_grad_base.h"

namespace MoeTokenUnpermuteWithRoutingMapGrad {
using namespace AscendC;

template <typename OriT, typename IdxT>
class MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadTrue
    : protected MoeTokenUnpermuteWithRoutingMapGradBase<OriT, IdxT>
{
public:
    __aicore__ inline MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadTrue(){};
    __aicore__ inline void Init(
        GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
        GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
        const MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling_data);
    __aicore__ inline void Process();

protected:
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> permutedTokensTQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> unpermutedGradTQue;
    TBuf<TPosition::VECCALC> probGradReduceSumTBuf;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> permutedTokensGradTQue;
    TBuf<TPosition::VECCALC> probGradOutTBuf;

    LocalTensor<float> probGradReduceSumLocal;
    LocalTensor<OriT> permutedTokensGradLocal;
    LocalTensor<OriT> probGradOutLocal;

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
};

template <typename OriT, typename IdxT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadTrue<OriT, IdxT>::Init(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
    const MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling_data)
{
    MoeTokenUnpermuteWithRoutingMapGradBase<OriT, IdxT>::Init(
        unpermuted_tokens_grad, outIndex, permuteTokenId, routing_map, permuted_tokens, probs, permuted_tokens_grad,
        probs_grad, tiling_data);
    // 申请2块空间手动double buffer
    this->pipe.InitBuffer(unpermutedGradTQue, DOUBLE_BUFFER, this->hiddenSizeAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(permutedTokensTQue, DOUBLE_BUFFER, this->hiddenSizeAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(probGradReduceSumTBuf, this->hiddenSizeLoopTimesAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(permutedTokensGradTQue, DOUBLE_BUFFER, this->hiddenSizeAlign * this->inputTypeSize);
    this->pipe.InitBuffer(probGradOutTBuf, BLOCK_SIZE_32);

    InitOutput<OriT>(
        this->probGradGm[this->unpermutedOutputDStartOffset * this->numExpert], this->tokensNum * this->numExpert,
        OriT(0));
    SyncAll();
}

template <typename OriT, typename IdxT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadTrue<OriT, IdxT>::Process()
{
    int64_t outNumCurrentCore = this->coreIndex < this->formerCoreNum ? this->rowIdMapEachCore : this->rowIdMapTailCore;
    probGradOutLocal = probGradOutTBuf.Get<OriT>();
    probGradReduceSumLocal = probGradReduceSumTBuf.Get<float>();
    for (int64_t indicesLoopTime = 0; indicesLoopTime < outNumCurrentCore;
         indicesLoopTime++) { // 根据experts_num* capacity分核，每个核分到outNumCurrentCore个
        int64_t rowIdMapLoopOffset = this->rowIdMapStartOffset + indicesLoopTime;
        int64_t tokenId = this->sortedTwiceIndicesGm.GetValue(rowIdMapLoopOffset);
        int64_t permuteTokenId = this->sortedTwiceIndexGm.GetValue(rowIdMapLoopOffset);
        int64_t probOffset = tokenId * this->numExpert + permuteTokenId / this->capacity;
        OriT probTemp = this->probGm.GetValue(probOffset);
        float prob = 0.0;
        if constexpr (IsSameType<OriT, bfloat16_t>::value) {
            prob = AscendC::ToFloat(probTemp);
        } else {
            prob = static_cast<float>(probTemp);
        }
        Duplicate(probGradReduceSumLocal, float(0), this->hiddenSizeLoopTimesAlign);
        SToMTE2Sync();
        for (int64_t hiddenLoop = 0; hiddenLoop < this->hiddenSizeLoopTimes; hiddenLoop++) {
            uint32_t hiddenLoopNum =
                hiddenLoop == this->hiddenSizeLoopTimes - 1 ? this->hiddenSizeTail : this->hiddenSizeAlign;
            uint32_t hiddenLoopBlockLen = hiddenLoopNum * this->inputTypeSize;
            int64_t hiddenLoopOffset = hiddenLoop * this->hiddenSizeAlign;
            copyParams.blockCount = 1;
            copyParams.blockLen = hiddenLoopBlockLen;
            copyParams.srcStride = static_cast<uint32_t>(this->hiddenSize - hiddenLoopNum) * this->inputTypeSize;
            // 搬入unpermutedTokensGrad 和 permutedToken对应行，如果输入是fp16/bf16需要做原地cast到fp32
            int64_t permutedTokensOffset = permuteTokenId * this->hiddenSize + hiddenLoopOffset;
            int64_t unpermutedTokensGradOffset = tokenId * this->hiddenSize + hiddenLoopOffset;
            LocalTensor<float> unpermutedGradLocalFp32 = unpermutedGradTQue.template AllocTensor<float>();
            LocalTensor<float> permutedTokensLocalFp32 = permutedTokensTQue.template AllocTensor<float>();
            if constexpr (IsSameType<OriT, float>::value) {
                DataCopyPad(
                    unpermutedGradLocalFp32, this->unpermutedTokensGradGm[unpermutedTokensGradOffset], copyParams,
                    this->inputPadParams);
                DataCopyPad(
                    permutedTokensLocalFp32, this->permutedTokensGm[permutedTokensOffset], copyParams,
                    this->inputPadParams);
                MTE2ToVSync();
            } else {
                LocalTensor<OriT> unpermutedGradLocal = unpermutedGradLocalFp32.ReinterpretCast<OriT>();
                DataCopyPad(
                    unpermutedGradLocal[this->hiddenSizeAlign],
                    this->unpermutedTokensGradGm[unpermutedTokensGradOffset], copyParams, this->inputPadParams);
                MTE2ToVSync();
                Cast(
                    unpermutedGradLocalFp32, unpermutedGradLocal[this->hiddenSizeAlign], RoundMode::CAST_NONE,
                    hiddenLoopNum);
                LocalTensor<OriT> permutedTokensLocal = permutedTokensLocalFp32.template ReinterpretCast<OriT>();
                DataCopyPad(
                    permutedTokensLocal[this->hiddenSizeAlign], this->permutedTokensGm[permutedTokensOffset],
                    copyParams, this->inputPadParams);
                MTE2ToVSync();
                Cast(
                    permutedTokensLocalFp32, permutedTokensLocal[this->hiddenSizeAlign], RoundMode::CAST_NONE,
                    hiddenLoopNum);
            }

            // prob梯度等于unpermutedTokensGrad、permutedTokens对应行点乘，再做ReduceSum（每块切分h做一次ReduceSum，最后再一起累加）
            Mul(permutedTokensLocalFp32, permutedTokensLocalFp32, unpermutedGradLocalFp32, hiddenLoopNum);
            permutedTokensTQue.template EnQue(permutedTokensLocalFp32);
            permutedTokensLocalFp32 = permutedTokensTQue.template DeQue<float>();
            this->ReduceSumFunc(probGradReduceSumLocal[hiddenLoop], permutedTokensLocalFp32, hiddenLoopNum);
            VToSSync();
            Muls(permutedTokensLocalFp32, unpermutedGradLocalFp32, prob, hiddenLoopNum);
            permutedTokensTQue.template EnQue(permutedTokensLocalFp32);
            permutedTokensLocalFp32 = permutedTokensTQue.template DeQue<float>();
            // 搬出permutedToken梯度
            permutedTokensGradLocal = permutedTokensGradTQue.template AllocTensor<OriT>();
            if constexpr (IsSameType<OriT, float>::value) {
                DataCopyPad(this->permutedTokensGradGm[permutedTokensOffset], permutedTokensLocalFp32, copyParams);
            } else {
                Cast(permutedTokensGradLocal, permutedTokensLocalFp32, RoundMode::CAST_RINT, hiddenLoopNum);
                VToMTE3Sync();
                DataCopyPad(this->permutedTokensGradGm[permutedTokensOffset], permutedTokensGradLocal, copyParams);
            }
            unpermutedGradTQue.FreeTensor(unpermutedGradLocalFp32);
            permutedTokensTQue.FreeTensor(permutedTokensLocalFp32);
            permutedTokensGradTQue.FreeTensor(permutedTokensGradLocal);
        }
        // 对tmpBufferProbGradReduceSum做ReduceSum，计算当前行对应的prob梯度
        this->ReduceSumFunc(probGradReduceSumLocal, probGradReduceSumLocal, this->hiddenSizeLoopTimes);
        VToSSync();
        float probGradTemp = probGradReduceSumLocal.GetValue(0);
        OriT probGrad;
        if constexpr (IsSameType<OriT, bfloat16_t>::value) {
            probGrad = AscendC::ToBfloat16(probGradTemp);
        } else {
            probGrad = static_cast<OriT>(probGradTemp);
        }
        probGradOutLocal.SetValue(0, probGrad);
        SToMTE3Sync();
        copyParams.blockLen = this->inputTypeSize;
        DataCopyPad(this->probGradGm[probOffset], probGradOutLocal, copyParams);
    }
}

} // namespace MoeTokenUnpermuteWithRoutingMapGrad
#endif // MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NOT_NONE_DROP_PAD_TRUE_H