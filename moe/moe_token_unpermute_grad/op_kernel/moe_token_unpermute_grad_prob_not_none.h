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
 * \file moe_token_unpermute_grad_prob_not_none.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_GRAD_PROB_NOTNONE_H
#define MOE_TOKEN_UNPERMUTE_GRAD_PROB_NOTNONE_H
#include "moe_token_unpermute_grad_base.h"

namespace MoeTokenUnpermuteGrad {
using namespace AscendC;

template <typename OriT, typename IdxT, typename ProbT>
class MoeTokenUnpermuteGradProbNotNone : protected MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT> {
public:
    __aicore__ inline MoeTokenUnpermuteGradProbNotNone(){};
    __aicore__ inline void Init(GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices,
                                GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                const MoeTokenUnpermuteGradTilingData &tilingData);
    __aicore__ inline void Process();

protected:
    TBuf<TPosition::VECCALC> inQueueProb;       // probs (indicesNumPerLoop,) * double_buffer
    TBuf<TPosition::VECCALC> inQueueUnpermuted; // unpermuted_tokens_grad (1, hiddenSizeAlign) * double_buffer
    TBuf<TPosition::VECCALC>
        inQueuePermutedTokens; // permuted_tokens (permutedTokenNumPerLoop, hiddenSizeAlign) * double_buffer
    TBuf<TPosition::VECCALC> outQueuePermutedTokensGrad;
    TBuf<TPosition::VECCALC> outQueueProbGrad;

    TBuf<TPosition::VECCALC> tmpBufferPermutedTokensCast;
    TBuf<TPosition::VECCALC> tmpBufferReserve;
    TBuf<TPosition::VECCALC> tmpBufferUnpermutedCast;
    TBuf<TPosition::VECCALC> tmpBufferProbCast;
    TBuf<TPosition::VECCALC> tmpBufferProbGradCast;
    TBuf<TPosition::VECCALC> tmpBufferPermutedTokensGrad;
    TBuf<TPosition::VECCALC> tmpBufferProbGradReduceSum;

    LocalTensor<OriT> inQueueUnpermutedLocal;
    LocalTensor<OriT> inQueuePermutedTokensLocal;
    LocalTensor<ProbT> inQueueProbLocal;
    LocalTensor<IdxT> inQueueRowIdMapLocal;
    LocalTensor<float> tmpBufferPermutedTokensFp32;
    LocalTensor<float> tmpBufferUnpermutedFp32;
    LocalTensor<float> tmpBufferProbFp32;
    LocalTensor<float> tmpBufferProbGradFp32;
    LocalTensor<float> tmpBufferPermutedTokensGradFp32;
    LocalTensor<float> tmpBufferProbGradReduceSumFp32;
    LocalTensor<OriT> permutedTokensGradLocal;
    LocalTensor<ProbT> probGradLocal;

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    int64_t pingPongFlagIndices = 0;
    int64_t pingPongFlagProbs = 0;
    int64_t pingPongFlagUnpermute = 0;
    int64_t pingPongFlagPermuteToken = 0;
    int64_t pingPongFlagPermuteTokenGrad = 0;
    int64_t pingPongFlagProbsGrad = 0;
    event_t eventIdIndicesVMte2 = EVENT_ID0;      // eventid 0 1用于indices的v和mte2之间同步
    event_t eventIdProbsVMte2 = EVENT_ID2;        // eventid 2 3用于probs的v和mte2之间同步
    event_t eventIdUnpermuteVMte2 = EVENT_ID4;    // eventid 4 5用于unpermuteOutputD的v和mte2之间同步
    event_t eventIdPermuteTokenVMte2 = EVENT_ID6; // eventid 6 7用于permute_token的v和mte2之间同步
    event_t eventIdVMte3 = EVENT_ID0;             // eventid 0 1用于v和mte3之间同步

    int64_t indicesArray[INDICES_PROBS_MAX_RESERVE_NUM]; // 存储indicesNumPerLoop个indices值, 最大512,
                                                         // 为避免GetValue操作设置
    float probsArray[INDICES_PROBS_MAX_RESERVE_NUM]; // 存储indicesNumPerLoop个probs值, 最大512, 为避免GetValue操作设置
};

template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline void MoeTokenUnpermuteGradProbNotNone<OriT, IdxT, ProbT>::Init(
    GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices, GM_ADDR probs,
    GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, const MoeTokenUnpermuteGradTilingData &tilingData)
{
    MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>::Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs,
                                                       permuted_tokens_grad, probs_grad, tilingData);

    // 申请2块空间手动double buffer
    this->pipe.InitBuffer(inQueueUnpermuted, DOUBLE_BUFFER * this->hiddenSizeAlign * this->inputTypeSize);
    this->pipe.InitBuffer(inQueuePermutedTokens,
                          DOUBLE_BUFFER * (this->inputReserveNum * this->hiddenSizeAlign) * this->inputTypeSize);
    this->pipe.InitBuffer(inQueueProb, DOUBLE_BUFFER * this->indicesReserveNumAlign * this->probTypeSize);
    this->pipe.InitBuffer(
        outQueueProbGrad,
        DOUBLE_BUFFER * this->indicesReserveNumAlign *
            this->probTypeSize); // probGrad每indicesNumPerLoop搬出，预留indicesNumPerLoopAlign个元素的空间
    this->pipe.InitBuffer(outQueuePermutedTokensGrad, DOUBLE_BUFFER * this->hiddenSizeAlign * this->inputTypeSize);

    this->pipe.InitBuffer(tmpBufferPermutedTokensCast, this->hiddenSizeAlign * sizeof(float));
    this->pipe.InitBuffer(tmpBufferReserve, BLOCK_SIZE_256);
    this->pipe.InitBuffer(tmpBufferUnpermutedCast, this->hiddenSizeAlign * sizeof(float));
    this->pipe.InitBuffer(tmpBufferProbGradReduceSum, this->indicesReserveNumAlign * sizeof(float));

    this->pipe.InitBuffer(tmpBufferProbCast, this->indicesReserveNumAlign * sizeof(float));
    this->pipe.InitBuffer(tmpBufferProbGradCast, this->indicesReserveNumAlign * sizeof(float));
    this->pipe.InitBuffer(tmpBufferPermutedTokensGrad, this->hiddenSizeAlign * sizeof(float));
}

template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline void MoeTokenUnpermuteGradProbNotNone<OriT, IdxT, ProbT>::Process()
{
    int64_t indicesEachLoop = this->indicesReserveNum; // topK整数倍
    int64_t inputEachLoop = this->inputReserveNum;
    int64_t indicesLoopTimes = 0;
    int64_t indicesLastLoop = 0;
    int64_t indicesNumEachCore = 0;
    int64_t topKInIndicesLastLoopTimes = 0;
    int64_t topKInIndicesEachLoopTimes = 0;
    int64_t permuteInTopKLoopTimes = 0;
    if (this->coreIndex < this->formerCoreNum) {
        indicesNumEachCore = this->rowIdMapEachCore;
    } else {
        indicesNumEachCore = this->rowIdMapTailCore;
    }
    indicesLoopTimes = (indicesNumEachCore + indicesEachLoop - 1) / indicesEachLoop;
    indicesLastLoop = indicesNumEachCore - indicesEachLoop * (indicesLoopTimes - 1); // topK整数倍
    topKInIndicesLastLoopTimes = indicesLastLoop / this->topK;
    topKInIndicesEachLoopTimes = indicesEachLoop / this->topK;
    permuteInTopKLoopTimes = this->topK / this->inputReserveNum; // topK是permutedTokenNumPerLoop的整数倍

    tmpBufferProbFp32 = tmpBufferProbCast.Get<float>();
    tmpBufferProbGradReduceSumFp32 =
        tmpBufferProbGradReduceSum
            .Get<float>(); // indicesNumPerLoop个fp32的空间, hiddensize在循环的时候，每次算indicesNumPerLoop个并累加
    tmpBufferUnpermutedFp32 = tmpBufferUnpermutedCast.Get<float>();
    tmpBufferPermutedTokensGradFp32 = tmpBufferPermutedTokensGrad.Get<float>();
    tmpBufferPermutedTokensFp32 = tmpBufferPermutedTokensCast.Get<float>();
    tmpBufferProbGradFp32 = tmpBufferProbGradCast.Get<float>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (int64_t indicesLoop = 0; indicesLoop < indicesLoopTimes;
         indicesLoop++) { // (token * topK) / indicesNumPerLoop 循环
        int64_t indicesNumLoop = indicesLoop == indicesLoopTimes - 1 ? indicesLastLoop : indicesEachLoop;
        // copy in rowidmap, (indicesNumPerLoop)
        int64_t indicesLoopOffset = indicesLoop * indicesEachLoop;
        int64_t loopOffset = this->rowIdMapStartOffset + indicesLoopOffset;
        for (int64_t i = 0; i < indicesNumLoop; i++) { // indices gm数据存储到数组上，每次搬运indicesNumLoop个
            indicesArray[i] = this->sortedIndicesGm.GetValue(loopOffset + i);
        }

        // copy in prob, (indicesNumPerLoop)
        // eventIdProbsVMte2用于控制probs的2块空间的同步，取值2和3
        // pingPongFlagProbs用于控制probs的2块空间的切换
        eventIdProbsVMte2 = pingPongFlagProbs ? EVENT_ID2 : EVENT_ID3;
        WaitFlag<HardEvent::V_MTE2>(eventIdProbsVMte2);
        inQueueProbLocal = inQueueProb.GetWithOffset<ProbT>(
            this->indicesReserveNumAlign, this->indicesReserveNumAlign * this->probTypeSize * pingPongFlagProbs);
        copyParams.blockLen = (uint32_t)indicesNumLoop * this->probTypeSize;
        DataCopyPad(inQueueProbLocal, this->probGm[loopOffset], copyParams, this->probPadParams);
        SetFlag<HardEvent::MTE2_V>(eventIdProbsVMte2);
        WaitFlag<HardEvent::MTE2_V>(eventIdProbsVMte2);
        if constexpr (IsSameType<ProbT, float>::value) { // fp32类型输入做Copy
            Copy(tmpBufferProbFp32, inQueueProbLocal, 64, this->indicesReserveNumAlignFp32RepeatTimes, {1, 1, 8, 8});
            Copy(tmpBufferProbFp32[this->indicesReserveNumAlignFp32TailOffset],
                 inQueueProbLocal[this->indicesReserveNumAlignFp32TailOffset], this->indicesReserveNumAlignFp32TailMask,
                 1, {1, 1, 8, 8});
        } else { // fp16和bf16类型输入做cast
            Cast(tmpBufferProbFp32, inQueueProbLocal, RoundMode::CAST_NONE, indicesNumLoop);
        }
        SetFlag<HardEvent::V_MTE2>(eventIdProbsVMte2);
        pingPongFlagProbs = 1 - pingPongFlagProbs; // probs的空间切换
        for (int64_t i = 0; i < indicesNumLoop; i++) { // probs ub空间数据存储到数组上，每次搬运indicesNumLoop个
            probsArray[i] = tmpBufferProbFp32.GetValue(i);
        }

        Duplicate(
            tmpBufferProbGradReduceSumFp32, float(0),
            this->indicesReserveNumAlign); // hiddensize循环时，将结果累加到tmpBufferProbGradReduceSumFp32上，最后积攒indicesNumPerLoop个搬出
        for (int64_t hiddensizeLoop = 0; hiddensizeLoop < this->hiddenSizeLoopTimes;
             hiddensizeLoop++) { // hiddensize循环
            int64_t hiddensizeLoopNum =
                hiddensizeLoop == this->hiddenSizeLoopTimes - 1 ? this->hiddenSizeTail : this->hiddenSizeAlign;
            int64_t hiddensizeLoopOffset = hiddensizeLoop * this->hiddenSizeAlign;
            int64_t topKInIndicesLoopTimes = indicesNumLoop / this->topK;
            for (int64_t topKInIndicesLoop = 0; topKInIndicesLoop < topKInIndicesLoopTimes;
                 topKInIndicesLoop++) { // indicesNumPerLoop / topK 循环
                // copy in unpermutedOutputD, (1, hiddenSize)
                int64_t unpermutedIndex = indicesLoopOffset / this->topK + topKInIndicesLoop;
                int64_t unpermutedOutputDloopOffset =
                    (this->unpermutedOutputDStartOffset + unpermutedIndex) * this->hiddenSize + hiddensizeLoopOffset;
                // eventIdUnpermuteVMte2用于控制unpermutedOutputD的2块空间的同步，取值4和5
                // pingPongFlagUnpermute用于控制unpermutedOutputD的2块空间的切换
                eventIdUnpermuteVMte2 = pingPongFlagUnpermute ? EVENT_ID4 : EVENT_ID5;
                WaitFlag<HardEvent::V_MTE2>(eventIdUnpermuteVMte2);
                inQueueUnpermutedLocal = inQueueUnpermuted.GetWithOffset<OriT>(
                    this->hiddenSizeAlign, this->hiddenSizeAlign * this->inputTypeSize * pingPongFlagUnpermute);
                copyParams.blockLen = (uint32_t)hiddensizeLoopNum * this->inputTypeSize;
                DataCopyPad(inQueueUnpermutedLocal, this->unpermutedOutputDGm[unpermutedOutputDloopOffset], copyParams,
                            this->inputPadParams);
                SetFlag<HardEvent::MTE2_V>(eventIdUnpermuteVMte2);
                WaitFlag<HardEvent::MTE2_V>(eventIdUnpermuteVMte2);
                if constexpr (IsSameType<OriT, float>::value) { // fp32类型输入做Copy
                    Copy(tmpBufferUnpermutedFp32, inQueueUnpermutedLocal, 64, this->hiddensizeAlignFp32RepeatTimes,
                         {1, 1, 8, 8});
                    Copy(tmpBufferUnpermutedFp32[this->hiddensizeAlignFp32TailOffset],
                         inQueueUnpermutedLocal[this->hiddensizeAlignFp32TailOffset], this->hiddensizeAlignFp32TailMask,
                         1, {1, 1, 8, 8});
                } else { // fp16和bf16类型输入做cast
                    Cast(tmpBufferUnpermutedFp32, inQueueUnpermutedLocal, RoundMode::CAST_NONE, hiddensizeLoopNum);
                }
                SetFlag<HardEvent::V_MTE2>(eventIdUnpermuteVMte2);
                pingPongFlagUnpermute = 1 - pingPongFlagUnpermute; // unpermutedOutputD的空间切换

                for (int64_t permuteInTopKLoop = 0; permuteInTopKLoop < permuteInTopKLoopTimes;
                     permuteInTopKLoop++) { // this->topK / permutedTokenNumPerLoop 循环
                    // copy in permuted_tokens, (permutedTokenNumPerLoop, hiddenSize).
                    // indicesNumPerLoop是topK倍数，topK是permutedTokenNumPerLoop倍数,
                    // 可以保证搬运permuted_tokens时需要的indices已搬入
                    // eventIdPermuteTokenVMte2用于控制permuted_tokens的2块空间的同步，取值6和7
                    // pingPongFlagPermuteToken用于控制permuted_tokens的2块空间的切换
                    eventIdPermuteTokenVMte2 = pingPongFlagPermuteToken ? EVENT_ID6 : EVENT_ID7;
                    WaitFlag<HardEvent::V_MTE2>(eventIdPermuteTokenVMte2);
                    int64_t inputLoopStartOffset =
                        topKInIndicesLoop * this->topK + permuteInTopKLoop * this->inputReserveNum;
                    for (int64_t inputLoop = 0; inputLoop < this->inputReserveNum;
                         inputLoop++) { // 循环搬入permuted_tokens (permutedTokenNumPerLoop, hiddenSize)
                        int64_t inputLoopOffset = indicesArray[inputLoopStartOffset + inputLoop];
                        inQueuePermutedTokensLocal = inQueuePermutedTokens.GetWithOffset<OriT>(
                            (this->inputReserveNum * this->hiddenSizeAlign),
                            (this->inputReserveNum * this->hiddenSizeAlign) * this->inputTypeSize *
                                pingPongFlagPermuteToken);
                        if (inputLoopOffset < this->numOutTokens) {
                            copyParams.blockLen = (uint32_t)hiddensizeLoopNum * this->inputTypeSize;
                            DataCopyPad(
                                inQueuePermutedTokensLocal[inputLoop * this->hiddenSizeAlign],
                                this->permutedTokensGm[inputLoopOffset * this->hiddenSize + hiddensizeLoopOffset],
                                copyParams, this->inputPadParams);
                        } else { // 处理截断情况，截断搬0
                            Duplicate(inQueuePermutedTokensLocal[inputLoop * this->hiddenSizeAlign], (OriT)0,
                                      this->hiddenSizeAlign);
                        }
                    }
                    SetFlag<HardEvent::MTE2_V>(eventIdPermuteTokenVMte2);

                    WaitFlag<HardEvent::MTE2_V>(eventIdPermuteTokenVMte2); // 等待permuted_tokens的搬运完成
                    for (int64_t permuteLoop = 0; permuteLoop < this->inputReserveNum;
                         permuteLoop++) { // permutedTokenNumPerLoop 循环, 进行计算
                        // calculate permuted_tokens_grad
                        int64_t indicesIndex = inputLoopStartOffset + permuteLoop;
                        int64_t inputGradOffset = indicesArray[indicesIndex];
                        if (inputGradOffset < this->numOutTokens) {
                            Muls(tmpBufferPermutedTokensGradFp32, tmpBufferUnpermutedFp32, probsArray[indicesIndex],
                                 hiddensizeLoopNum); // tmpBuffer2Fp32需要被反复使用，需将计算值保存在tmpBuffer5Fp32
                        } else {                     // 处理截断情况，截断搬0
                            Duplicate(tmpBufferPermutedTokensGradFp32, float(0), this->hiddenSizeAlign);
                        }
                        // copy out permuted_tokens_grad
                        // eventIdVMte3用于控制permuted_tokens_grads的2块空间的同步，取值0和1
                        // pingPongFlagPermuteTokenGrad用于控制permuted_tokens_grad的2块空间的切换
                        eventIdVMte3 = pingPongFlagPermuteTokenGrad ? EVENT_ID0 : EVENT_ID1;
                        WaitFlag<HardEvent::MTE3_V>(eventIdVMte3);
                        permutedTokensGradLocal = outQueuePermutedTokensGrad.GetWithOffset<OriT>(
                            this->hiddenSizeAlign,
                            this->hiddenSizeAlign * this->inputTypeSize * pingPongFlagPermuteTokenGrad);
                        if constexpr (IsSameType<OriT, float>::value) { // fp32类型输入做Copy
                            Copy(permutedTokensGradLocal, tmpBufferPermutedTokensGradFp32, 64,
                                 this->hiddensizeAlignFp32RepeatTimes, {1, 1, 8, 8});
                            Copy(permutedTokensGradLocal[this->hiddensizeAlignFp32TailOffset],
                                 tmpBufferPermutedTokensGradFp32[this->hiddensizeAlignFp32TailOffset],
                                 this->hiddensizeAlignFp32TailMask, 1, {1, 1, 8, 8});
                        } else { // fp16和bf16类型输入做cast
                            Cast(permutedTokensGradLocal, tmpBufferPermutedTokensGradFp32, RoundMode::CAST_RINT,
                                 hiddensizeLoopNum);
                        }
                        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
                        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
                        copyParams.blockLen = (uint32_t)hiddensizeLoopNum * this->inputTypeSize;
                        if (inputGradOffset < this->numOutTokens) {
                            DataCopyPad(
                                this->permutedTokensGradGm[inputGradOffset * this->hiddenSize + hiddensizeLoopOffset],
                                permutedTokensGradLocal, copyParams);
                        }
                        SetFlag<HardEvent::MTE3_V>(eventIdVMte3);
                        pingPongFlagPermuteTokenGrad =
                            1 - pingPongFlagPermuteTokenGrad; // permuted_tokens_grad的空间切换

                        // permuted_tokens的cast操作
                        if constexpr (IsSameType<OriT, float>::value) {
                            Copy(tmpBufferPermutedTokensFp32,
                                 inQueuePermutedTokensLocal[permuteLoop * this->hiddenSizeAlign], 64,
                                 this->hiddensizeAlignFp32RepeatTimes, {1, 1, 8, 8});
                            Copy(tmpBufferPermutedTokensFp32[this->hiddensizeAlignFp32TailOffset],
                                 inQueuePermutedTokensLocal[permuteLoop * this->hiddenSizeAlign +
                                                            this->hiddensizeAlignFp32TailOffset],
                                 this->hiddensizeAlignFp32TailMask, 1, {1, 1, 8, 8});
                        } else {
                            Cast(tmpBufferPermutedTokensFp32,
                                 inQueuePermutedTokensLocal[permuteLoop * this->hiddenSizeAlign], RoundMode::CAST_NONE,
                                 hiddensizeLoopNum);
                        }
                        // calculate prob_grad
                        Mul(tmpBufferPermutedTokensFp32, tmpBufferPermutedTokensFp32, tmpBufferUnpermutedFp32,
                            hiddensizeLoopNum);
                        this->ReduceSumFunc(tmpBufferProbGradFp32[indicesIndex], tmpBufferPermutedTokensFp32,
                                            hiddensizeLoopNum);
                    }
                    SetFlag<HardEvent::V_MTE2>(eventIdPermuteTokenVMte2);    // 触发permuted_tokens的下一次搬运
                    pingPongFlagPermuteToken = 1 - pingPongFlagPermuteToken; // permuted_tokens的空间切换
                }
            }
            Add(tmpBufferProbGradReduceSumFp32, tmpBufferProbGradReduceSumFp32, tmpBufferProbGradFp32,
                indicesNumLoop); // hiddensize被切分，每次循环的结果累加到tmpBufferProbGradReduceSumFp32上
        }

        // copy out prob_grad, 每indicesNumPerLoop个往外搬
        // eventIdVMte3用于控制prob_grad的2块空间的同步，取值0和1
        // pingPongFlagProbsGrad用于控制prob_grad的2块空间的切换
        eventIdVMte3 = pingPongFlagProbsGrad ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::MTE3_V>(eventIdVMte3);
        probGradLocal = outQueueProbGrad.GetWithOffset<ProbT>(
            this->indicesReserveNumAlign, this->indicesReserveNumAlign * this->probTypeSize * pingPongFlagProbsGrad);
        if constexpr (IsSameType<ProbT, float>::value) {
            Copy(probGradLocal, tmpBufferProbGradReduceSumFp32, 64, this->indicesReserveNumAlignFp32RepeatTimes,
                 {1, 1, 8, 8});
            Copy(probGradLocal[this->indicesReserveNumAlignFp32TailOffset],
                 tmpBufferProbGradReduceSumFp32[this->indicesReserveNumAlignFp32TailOffset],
                 this->indicesReserveNumAlignFp32TailMask, 1, {1, 1, 8, 8});
        } else {
            Cast(probGradLocal, tmpBufferProbGradReduceSumFp32, RoundMode::CAST_RINT, indicesNumLoop);
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVMte3);
        copyParams.blockLen = (uint32_t)indicesNumLoop * this->probTypeSize;
        DataCopyPad(this->probGradGm[loopOffset], probGradLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(eventIdVMte3);
        pingPongFlagProbsGrad = 1 - pingPongFlagProbsGrad; // prob_grad的空间切换
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

} // namespace MoeTokenUnpermuteGrad
#endif // MOE_TOKEN_UNPERMUTE_GRAD_PROB_NOTNONE_H
