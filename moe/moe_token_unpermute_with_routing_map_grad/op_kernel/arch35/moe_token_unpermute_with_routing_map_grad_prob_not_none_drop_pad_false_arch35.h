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
 * \file moe_token_unpermute_with_routing_map_grad_prob_not_none_drop_pad_false_arch35.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NOT_NONE_DROP_PAD_FALSE_ARCH35_H
#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NOT_NONE_DROP_PAD_FALSE_ARCH35_H
#include "moe_token_unpermute_with_routing_map_grad_base_arch35.h"
namespace MoeTokenUnpermuteWithRoutingMapGrad {
using namespace AscendC;

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
class MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadFalse
    : protected MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT> {
public:
    __aicore__ inline MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadFalse(){};
    __aicore__ inline void Init(GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId,
                                GM_ADDR routing_map, GM_ADDR permuted_tokens, GM_ADDR probs,
                                GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                const MoeTokenUnpermuteWithRoutingMapGradTilingData &tiling_data);
    __aicore__ inline void Process();

protected:
    TBuf<TPosition::VECCALC> inQueueUnpermuted;
    TBuf<TPosition::VECCALC> inQueuePermutedTokens;
    TBuf<TPosition::VECCALC> outQueuePermutedTokensGrad;
    TBuf<TPosition::VECCALC> outQueueProbGrad;

    TBuf<TPosition::VECCALC> tmpBufferPermutedTokensCast;
    TBuf<TPosition::VECCALC> tmpBufferUnpermutedCast;
    TBuf<TPosition::VECCALC> tmpBufferProbGradCast;
    TBuf<TPosition::VECCALC> tmpBufferPermutedTokensGrad;
    TBuf<TPosition::VECCALC> tmpBufferProbGradReduceSum;

    TBuf<TPosition::VECCALC> routingMapTBuf;
    TBuf<TPosition::VECCALC> probGradTBuf;

    LocalTensor<PermutedTokenT> inQueueUnpermutedLocal;
    LocalTensor<PermutedTokenT> inQueuePermutedTokensLocal;
    LocalTensor<float> tmpBufferPermutedTokensFp32;
    LocalTensor<float> tmpBufferUnpermutedFp32;
    LocalTensor<float> tmpBufferProbGradFp32;
    LocalTensor<float> tmpBufferPermutedTokensGradFp32;
    LocalTensor<float> tmpBufferProbGradReduceSumFp32;
    LocalTensor<PermutedTokenT> permutedTokensGradLocal;
    LocalTensor<ProbsT> probGradLocal;
    LocalTensor<int8_t> tmpBufferRoutingMap;
    LocalTensor<ProbsT> probGradLineLocal;

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    DataCopyPadExtParams<int8_t> routingMapPadParams{false, 0, 0, 0};
    int64_t pingPongFlagUnpermute = 0;
    int64_t pingPongFlagPermuteToken = 0;
    int64_t pingPongFlagPermuteTokenGrad = 0;
    int64_t pingPongFlagProbsGrad = 0;
    event_t eventIdVMte3 = EVENT_ID0;             // eventid 0 1用于v和mte3之间同步
    event_t eventIdIndicesVMte2 = EVENT_ID0;      // eventid 0 1用于indices的v和mte2之间同步
    event_t eventIdProbsVMte2 = EVENT_ID2;        // eventid 2 3用于probs的v和mte2之间同步
    event_t eventIdUnpermuteVMte2 = EVENT_ID4;    // eventid 4 5用于unpermuteOutputD的v和mte2之间同步
    event_t eventIdPermuteTokenVMte2 = EVENT_ID6; // eventid 6 7用于permute_token的v和mte2之间同步
    int64_t
        indicesArray[INDICES_PROBS_MAX_RESERVE_NUM]; // 存储indicesNumPerLoop个indices值, 最大512,为避免GetValue操作设置
    float probsArray[INDICES_PROBS_MAX_RESERVE_NUM]; // 存储indicesNumPerLoop个probs值, 最大512, 为避免GetValue操作设置
    int64_t probsAddrArray[INDICES_PROBS_MAX_RESERVE_NUM];
};

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadFalse<PermutedTokenT, IdxT, ProbsT>::Init(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
    const MoeTokenUnpermuteWithRoutingMapGradTilingData &tiling_data)
{
    MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::Init(
        unpermuted_tokens_grad, outIndex, permuteTokenId, routing_map, permuted_tokens, probs, permuted_tokens_grad,
        probs_grad, tiling_data);
    int64_t probGradSize = 0;
    if (this->coreIndex < this->formerCoreNum) {
        probGradSize = this->tokenNumEachCore * this->numExpert;
    } else {
        probGradSize = this->tokenNumTailCore * this->numExpert;
    }
    GlobalTensor<ProbsT> probGradInitGm;
    probGradInitGm.SetGlobalBuffer((__gm__ ProbsT *)probs_grad + this->unpermutedOutputDStartOffset * this->numExpert);
    Fill(probGradInitGm, probGradSize, static_cast<ProbsT>(0));
    SyncAll();
    // 申请2块空间手动double buffer
    this->pipe.InitBuffer(inQueueUnpermuted, DOUBLE_BUFFER * this->hiddenSizeAlign * this->inputTypeSize);
    this->pipe.InitBuffer(inQueuePermutedTokens, DOUBLE_BUFFER * this->hiddenSizeAlign * this->inputTypeSize);
    this->pipe.InitBuffer(outQueueProbGrad, this->indicesReserveNumAlign * this->probTypeSize);
    this->pipe.InitBuffer(outQueuePermutedTokensGrad, DOUBLE_BUFFER * this->hiddenSizeAlign * this->inputTypeSize);

    this->pipe.InitBuffer(tmpBufferPermutedTokensCast, this->hiddenSizeAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(tmpBufferUnpermutedCast, this->hiddenSizeAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(tmpBufferProbGradReduceSum, this->indicesReserveNumAlign * SIZE_FLOAT);

    this->pipe.InitBuffer(tmpBufferProbGradCast, this->indicesReserveNumAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(tmpBufferPermutedTokensGrad, this->hiddenSizeAlign * SIZE_FLOAT);
    this->pipe.InitBuffer(routingMapTBuf, this->numExpertAlign);
    this->pipe.InitBuffer(probGradTBuf, this->numExpertAlign * this->probTypeSize);
}

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void
MoeTokenUnpermuteWithRoutingMapGradProbNotNoneDropPadFalse<PermutedTokenT, IdxT, ProbsT>::Process()
{
    int64_t indicesLoopTimes = 0;
    int64_t indicesLastLoop = 0;
    int64_t indicesNumEachCore = 0;
    int64_t topKInIndicesLastLoopTimes = 0;
    int64_t topKInIndicesEachLoopTimes = 0;
    if (this->coreIndex < this->formerCoreNum) {
        indicesNumEachCore = this->rowIdMapEachCore;
    } else {
        indicesNumEachCore = this->rowIdMapTailCore;
    }
    indicesLoopTimes = this->CeilDiv(indicesNumEachCore, this->indicesReserveNum);
    indicesLastLoop = indicesNumEachCore - this->indicesReserveNum * (indicesLoopTimes - 1);
    topKInIndicesLastLoopTimes = this->CeilDiv(indicesLastLoop, this->topK);
    topKInIndicesEachLoopTimes = this->CeilDiv(this->indicesReserveNum, this->topK);

    tmpBufferProbGradReduceSumFp32 = tmpBufferProbGradReduceSum.Get<float>();
    tmpBufferUnpermutedFp32 = tmpBufferUnpermutedCast.Get<float>();
    tmpBufferPermutedTokensGradFp32 = tmpBufferPermutedTokensGrad.Get<float>();
    tmpBufferPermutedTokensFp32 = tmpBufferPermutedTokensCast.Get<float>();
    tmpBufferProbGradFp32 = tmpBufferProbGradCast.Get<float>();
    tmpBufferRoutingMap = routingMapTBuf.Get<int8_t>();
    probGradLineLocal = probGradTBuf.Get<ProbsT>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (int64_t indicesLoop = 0; indicesLoop < indicesLoopTimes;
         indicesLoop++) { // (token * topK) / indicesNumPerLoop 循环
        int64_t indicesNumLoop = indicesLoop == indicesLoopTimes - 1 ? indicesLastLoop : this->indicesReserveNum;
        int64_t topKInIndicesLoopTimes = this->SafeDiv(indicesNumLoop, this->topK);
        int64_t indicesLoopOffset = indicesLoop * this->indicesReserveNum;
        int64_t loopOffset = this->rowIdMapStartOffset + indicesLoopOffset;
        int64_t tokenLoopBaseOffset = this->SafeDiv(indicesLoopOffset, this->topK);
        for (int64_t i = 0; i < indicesNumLoop; i++) { // indices gm数据存储到数组上，每次搬运indicesNumLoop个
            indicesArray[i] = this->sortedTwiceIndexGm.GetValue(loopOffset + i);
        }
        int64_t topKInIndices =
            indicesLoop == indicesLoopTimes - 1 ? topKInIndicesLastLoopTimes : topKInIndicesEachLoopTimes;
        int64_t probIndex = 0;
        for (int64_t topKLoop = 0; topKLoop < topKInIndices; topKLoop++) {
            int64_t tokenIndex = this->SafeDiv(loopOffset, this->topK) + topKLoop;
            int64_t routingMapOffset = tokenIndex * this->numExpert;
            copyParams.blockLen = (uint32_t)this->numExpert;
            DataCopyPad(tmpBufferRoutingMap, this->routingMapGm[routingMapOffset], copyParams, routingMapPadParams);
            MTE3ToSSync();
            int64_t entriesForToken = 0;
            for (int64_t expertIndex = 0; expertIndex < this->numExpert; expertIndex++) {
                int8_t isChoose = tmpBufferRoutingMap.GetValue(expertIndex);
                int64_t probOffset = routingMapOffset + expertIndex;
                if (isChoose) {
                    ProbsT probTemp = this->probGm.GetValue(probOffset);
                    float prob = 0.0;
                    if constexpr (IsSameType<ProbsT, bfloat16_t>::value) {
                        prob = AscendC::ToFloat(probTemp);
                    } else if constexpr (IsSameType<ProbsT, half>::value) {
                        prob = static_cast<float>(probTemp);
                    } else {
                        prob = probTemp;
                    }
                    probsArray[probIndex] = prob;
                    probsAddrArray[probIndex] = probOffset;
                    probIndex++;
                    entriesForToken++;
                }
            }
            for (int64_t pad = entriesForToken; pad < this->topK; pad++) {
                probsArray[probIndex] = 0.0f;
                probsAddrArray[probIndex] = -1;
                probIndex++;
            }
        }
        Duplicate(
            tmpBufferProbGradReduceSumFp32, float(0),
            this->indicesReserveNumAlign); // hiddensize循环时，将结果累加到tmpBufferProbGradReduceSumFp32上，最后积攒indicesNumPerLoop个搬出
        for (int64_t hiddensizeLoop = 0; hiddensizeLoop < this->hiddenSizeLoopTimes;
             hiddensizeLoop++) { // hiddensize循环
            int64_t hiddensizeLoopNum =
                hiddensizeLoop == this->hiddenSizeLoopTimes - 1 ? this->hiddenSizeTail : this->hiddenSizeAlign;
            int64_t hiddensizeLoopOffset = hiddensizeLoop * this->hiddenSizeAlign;
            for (int64_t topKInIndicesLoop = 0; topKInIndicesLoop < topKInIndicesLoopTimes;
                 topKInIndicesLoop++) { // indicesNumPerLoop / topK 循环
                // copy in unpermutedOutputD, (1, hiddenSize)
                int64_t unpermutedIndex = tokenLoopBaseOffset + topKInIndicesLoop;
                int64_t unpermutedOutputDloopOffset =
                    (this->unpermutedOutputDStartOffset + unpermutedIndex) * this->hiddenSize + hiddensizeLoopOffset;
                // eventIdUnpermuteVMte2用于控制unpermutedOutputD的2块空间的同步，取值4和5
                // pingPongFlagUnpermute用于控制unpermutedOutputD的2块空间的切换
                eventIdUnpermuteVMte2 = pingPongFlagUnpermute ? EVENT_ID4 : EVENT_ID5;
                WaitFlag<HardEvent::V_MTE2>(eventIdUnpermuteVMte2);
                inQueueUnpermutedLocal = inQueueUnpermuted.GetWithOffset<PermutedTokenT>(
                    this->hiddenSizeAlign, this->hiddenSizeAlign * this->inputTypeSize * pingPongFlagUnpermute);
                copyParams.blockLen = (uint32_t)hiddensizeLoopNum * this->inputTypeSize;
                DataCopyPad(inQueueUnpermutedLocal, this->unpermutedTokensGradGm[unpermutedOutputDloopOffset],
                            copyParams, this->inputPadParams);
                SetFlag<HardEvent::MTE2_V>(eventIdUnpermuteVMte2);
                WaitFlag<HardEvent::MTE2_V>(eventIdUnpermuteVMte2);
                if constexpr (IsSameType<PermutedTokenT, float>::value) { // fp32类型输入做Copy
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

                for (int64_t permuteInTopKLoop = 0; permuteInTopKLoop < this->topK; permuteInTopKLoop++) {
                    int64_t indicesOffset = topKInIndicesLoop * this->topK + permuteInTopKLoop;
                    int64_t permutedTokenOffset = indicesArray[indicesOffset];
                    if (permutedTokenOffset < 0) {
                        continue;
                    }
                    Muls(tmpBufferPermutedTokensGradFp32, tmpBufferUnpermutedFp32, probsArray[indicesOffset],
                         hiddensizeLoopNum);
                    // copy out permuted_tokens_grad
                    // eventIdVMte3用于控制permuted_tokens_grads的2块空间的同步，取值0和1
                    // pingPongFlagPermuteTokenGrad用于控制permuted_tokens_grad的2块空间的切换
                    eventIdVMte3 = pingPongFlagPermuteTokenGrad ? EVENT_ID0 : EVENT_ID1;
                    WaitFlag<HardEvent::MTE3_V>(eventIdVMte3);
                    permutedTokensGradLocal = outQueuePermutedTokensGrad.GetWithOffset<PermutedTokenT>(
                        this->hiddenSizeAlign,
                        this->hiddenSizeAlign * this->inputTypeSize * pingPongFlagPermuteTokenGrad);
                    if constexpr (IsSameType<PermutedTokenT, float>::value) { // fp32类型输入做Copy
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
                    DataCopyPad(
                        this->permutedTokensGradGm[permutedTokenOffset * this->hiddenSize + hiddensizeLoopOffset],
                        permutedTokensGradLocal, copyParams);
                    SetFlag<HardEvent::MTE3_V>(eventIdVMte3);
                    pingPongFlagPermuteTokenGrad = 1 - pingPongFlagPermuteTokenGrad; // permuted_tokens_grad的空间切换
                    // eventIdPermuteTokenVMte2用于控制permuted_tokens的2块空间的同步，取值6和7
                    // pingPongFlagPermuteToken用于控制permuted_tokens的2块空间的切换
                    eventIdPermuteTokenVMte2 = pingPongFlagPermuteToken ? EVENT_ID6 : EVENT_ID7;
                    WaitFlag<HardEvent::V_MTE2>(eventIdPermuteTokenVMte2);
                    inQueuePermutedTokensLocal = inQueuePermutedTokens.GetWithOffset<PermutedTokenT>(
                        this->hiddenSizeAlign, this->hiddenSizeAlign * this->inputTypeSize * pingPongFlagPermuteToken);
                    copyParams.blockLen = (uint32_t)hiddensizeLoopNum * this->inputTypeSize;
                    DataCopyPad(inQueuePermutedTokensLocal,
                                this->permutedTokensGm[permutedTokenOffset * this->hiddenSize + hiddensizeLoopOffset],
                                copyParams, this->inputPadParams);
                    SetFlag<HardEvent::MTE2_V>(eventIdPermuteTokenVMte2);
                    WaitFlag<HardEvent::MTE2_V>(eventIdPermuteTokenVMte2); // 等待permuted_tokens的搬运完成
                    // permuted_tokens的cast操作
                    if constexpr (IsSameType<PermutedTokenT, float>::value) {
                        Copy(tmpBufferPermutedTokensFp32, inQueuePermutedTokensLocal, 64,
                             this->hiddensizeAlignFp32RepeatTimes, {1, 1, 8, 8});
                        Copy(tmpBufferPermutedTokensFp32[this->hiddensizeAlignFp32TailOffset],
                             inQueuePermutedTokensLocal[this->hiddensizeAlignFp32TailOffset],
                             this->hiddensizeAlignFp32TailMask, 1, {1, 1, 8, 8});
                    } else {
                        Cast(tmpBufferPermutedTokensFp32, inQueuePermutedTokensLocal, RoundMode::CAST_NONE,
                             hiddensizeLoopNum);
                    }
                    SetFlag<HardEvent::V_MTE2>(eventIdPermuteTokenVMte2);    // 触发permuted_tokens的下一次搬运
                    pingPongFlagPermuteToken = 1 - pingPongFlagPermuteToken; // permuted_tokens的空间切换

                    // calculate prob_grad
                    Mul(tmpBufferPermutedTokensFp32, tmpBufferPermutedTokensFp32, tmpBufferUnpermutedFp32,
                        hiddensizeLoopNum);
                    this->ReduceSumFunc(tmpBufferProbGradFp32[indicesOffset], tmpBufferPermutedTokensFp32,
                                        hiddensizeLoopNum);
                }
            }
            Add(tmpBufferProbGradReduceSumFp32, tmpBufferProbGradReduceSumFp32, tmpBufferProbGradFp32,
                indicesNumLoop); // hiddensize被切分，每次循环的结果累加到tmpBufferProbGradReduceSumFp32上
        }
        // copy out prob_grad, 每indicesNumPerLoop个往外搬
        // pingPongFlagProbsGrad用于控制prob_grad的2块空间的切换
        probGradLocal = outQueueProbGrad.Get<ProbsT>();
        if constexpr (IsSameType<ProbsT, float>::value) {
            Copy(probGradLocal, tmpBufferProbGradReduceSumFp32, 64, this->indicesReserveNumAlignFp32RepeatTimes,
                 {1, 1, 8, 8});
            Copy(probGradLocal[this->indicesReserveNumAlignFp32TailOffset],
                 tmpBufferProbGradReduceSumFp32[this->indicesReserveNumAlignFp32TailOffset],
                 this->indicesReserveNumAlignFp32TailMask, 1, {1, 1, 8, 8});
        } else {
            Cast(probGradLocal, tmpBufferProbGradReduceSumFp32, RoundMode::CAST_RINT, indicesNumLoop);
        }
        VToSSync();
        for (int64_t topKInIndicesLoop = 0; topKInIndicesLoop < topKInIndicesLoopTimes; topKInIndicesLoop++) {
            Duplicate(probGradLineLocal, ProbsT(0), this->numExpert);
            int64_t probGradBaseOffset = topKInIndicesLoop * this->topK;
            for (int64_t probGradIndex = 0; probGradIndex < this->topK; probGradIndex++) {
                int64_t probGradOffset = probGradBaseOffset + probGradIndex;
                if (probsAddrArray[probGradOffset] < 0) {
                    continue;
                }
                ProbsT probGrad = probGradLocal.GetValue(probGradOffset);
                int64_t addr = probsAddrArray[probGradOffset] % this->numExpert;
                probGradLineLocal.SetValue(addr, probGrad);
            }
            copyParams.blockLen = (uint32_t)this->numExpert * this->probTypeSize;
            SToMTE3Sync();
            DataCopyPad(
                this->probGradGm[(this->unpermutedOutputDStartOffset + tokenLoopBaseOffset + topKInIndicesLoop) *
                                 this->numExpert],
                probGradLineLocal, copyParams);
            MTE3ToSSync();
        }
        pingPongFlagProbsGrad = 1 - pingPongFlagProbsGrad; // prob_grad的空间切换
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

} // namespace MoeTokenUnpermuteWithRoutingMapGrad
#endif // MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NOT_NONE_DROP_PAD_FALSE_ARCH35_H
