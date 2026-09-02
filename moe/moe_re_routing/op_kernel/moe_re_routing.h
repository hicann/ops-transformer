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
 * \file moe_re_routing.h
 * \brief
 */

#ifndef MOE_RE_ROUTING_H
#define MOE_RE_ROUTING_H

#include "kernel_operator.h"
using namespace AscendC;

template <typename T1, typename T2, typename TScale>
class KernelMoeReRouting {
public:
    __aicore__ inline KernelMoeReRouting() {}
    __aicore__ inline KernelMoeReRouting(TPipe *pipe, const MoeReRoutingTilingData *tiling)
        : pipe_(pipe),
          tl_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR tokens, GM_ADDR expertTokensNumPerRank, GM_ADDR perTokenScales,
                                GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx,
                                GM_ADDR expertTokenNum);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(int64_t rows, int64_t offset);
    __aicore__ inline void CopyOut(int64_t rows, int64_t offset);
    __aicore__ inline void ComputeAndCopyOutIndex(int64_t rows, int64_t offset, int64_t head);

    TPipe *pipe_ = nullptr;
    const MoeReRoutingTilingData *tl_;
    constexpr static int64_t blockSize = 32;
    GlobalTensor<T1> srcGm;
    GlobalTensor<T1> dstGm;
    GlobalTensor<TScale> srcScaleGm;
    GlobalTensor<TScale> dstScaleGm;
    GlobalTensor<T2> srcExpertTokenGm;
    GlobalTensor<T2> dstExpertTokenGm;
    GlobalTensor<int32_t> indexGm;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind;
    TQue<QuePosition::VECOUT, 1> idxOutQue;
    int64_t srcOffset = 0;
    int64_t tokenNums = 0;
    int64_t tokenTotalNumsSrc = 0;
    int64_t tokenTotalNumsDst = 0;
    int64_t ubLoop = 1;
    int64_t ubTail = 0;
    int64_t rankLoopNum = 1;
    int64_t rankTailNum = 0;
};

template <typename T1, typename T2, typename TScale>
__aicore__ inline void KernelMoeReRouting<T1, T2, TScale>::Init(GM_ADDR tokens, GM_ADDR expertTokensNumPerRank,
                                                                GM_ADDR perTokenScales, GM_ADDR permuteTokens,
                                                                GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx,
                                                                GM_ADDR expertTokenNum)
{
    srcGm.SetGlobalBuffer((__gm__ T1 *)tokens);
    dstGm.SetGlobalBuffer((__gm__ T1 *)permuteTokens);
    srcExpertTokenGm.SetGlobalBuffer((__gm__ T2 *)expertTokensNumPerRank);
    dstExpertTokenGm.SetGlobalBuffer((__gm__ T2 *)expertTokenNum);
    indexGm.SetGlobalBuffer((__gm__ int32_t *)permuteTokenIdx);
    if (tl_->hasScale == 1) {
        srcScaleGm.SetGlobalBuffer((__gm__ TScale *)perTokenScales);
        dstScaleGm.SetGlobalBuffer((__gm__ TScale *)permutePerTokenScales);
        this->pipe_->InitBuffer(queBind, 2,
                                AlignUp(tl_->ubFactor * tl_->tokensSize * sizeof(T1), blockSize) +
                                    AlignUp(tl_->ubFactor * sizeof(TScale), blockSize));
    } else {
        this->pipe_->InitBuffer(queBind, 2, AlignUp(tl_->ubFactor * tl_->tokensSize * sizeof(T1), blockSize));
    }
    this->pipe_->InitBuffer(idxOutQue, 2, tl_->ubFactor * sizeof(int32_t));
}

template <typename T1, typename T2, typename TScale>
__aicore__ inline void KernelMoeReRouting<T1, T2, TScale>::Process()
{
    int64_t blockId = GetBlockIdx();
    rankLoopNum = tl_->rankNum / tl_->coreNum;
    rankTailNum = tl_->rankNum % tl_->coreNum;
    if (blockId < rankTailNum) {
        ++rankLoopNum;
    }
    for (int64_t rankLoopId = 0; rankLoopId < rankLoopNum; rankLoopId++) {
        int64_t dieIdRingtNow = rankLoopId * tl_->coreNum + blockId;
        tokenTotalNumsSrc = 0;
        tokenTotalNumsDst = 0;
        if (dieIdRingtNow < tl_->rankNum) {
            // 模拟ReduceSum
            for (int64_t dieId = 0; dieId < dieIdRingtNow; dieId++) {
                for (int64_t expertId = 0; expertId < tl_->expertNumPerRank; expertId++) {
                    tokenTotalNumsSrc += srcExpertTokenGm(dieId * tl_->expertNumPerRank + expertId);
                }
            }

            for (int64_t expertId = 0; expertId < tl_->expertNumPerRank; expertId++) {
                tokenNums = srcExpertTokenGm(expertId + dieIdRingtNow * tl_->expertNumPerRank);
                // 累加当前专家在前面die上的token数
                for (int64_t dieId = 0; dieId < dieIdRingtNow; dieId++) {
                    tokenTotalNumsDst += srcExpertTokenGm(dieId * tl_->expertNumPerRank + expertId);
                }
                ubLoop = tokenNums / tl_->ubFactor;
                ubTail = tokenNums % tl_->ubFactor;
                for (int64_t i = 0; i < ubLoop; i++) {
                    CopyIn(tl_->ubFactor, (tokenTotalNumsSrc + i * tl_->ubFactor));
                    ComputeAndCopyOutIndex(tl_->ubFactor, (tokenTotalNumsDst + i * tl_->ubFactor),
                                           (tokenTotalNumsSrc + i * tl_->ubFactor));
                    CopyOut(tl_->ubFactor, (tokenTotalNumsDst + i * tl_->ubFactor));
                }
                if (ubTail != 0) {
                    CopyIn(ubTail, (tokenTotalNumsSrc + ubLoop * tl_->ubFactor));
                    ComputeAndCopyOutIndex(ubTail, (tokenTotalNumsDst + ubLoop * tl_->ubFactor),
                                           (tokenTotalNumsSrc + ubLoop * tl_->ubFactor));
                    CopyOut(ubTail, (tokenTotalNumsDst + ubLoop * tl_->ubFactor));
                }
                tokenTotalNumsSrc += tokenNums;
                // 累加当前专家在当前die以及后续所有die上的token数
                for (int64_t dieId = dieIdRingtNow; dieId < tl_->rankNum; dieId++) {
                    tokenTotalNumsDst += srcExpertTokenGm(dieId * tl_->expertNumPerRank + expertId);
                }
            }

            if (dieIdRingtNow == 0) {
                for (int64_t expertId = 0; expertId < tl_->expertNumPerRank; expertId++) {
                    dstExpertTokenGm(expertId) = 0;
                }
                for (int64_t dieId = 0; dieId < tl_->rankNum; dieId++) {
                    for (int64_t expertId = 0; expertId < tl_->expertNumPerRank; expertId++) {
                        dstExpertTokenGm(expertId) += srcExpertTokenGm(expertId + dieId * tl_->expertNumPerRank);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, typename TScale>
__aicore__ inline void KernelMoeReRouting<T1, T2, TScale>::CopyIn(int64_t rows, int64_t offset)
{
    LocalTensor<T1> srcLocal = queBind.AllocTensor<T1>();
    DataCopyExtParams copyParams(1, rows * tl_->tokensSize * sizeof(T1), 0, 0, 0);
    DataCopyPadExtParams<T1> padParams(false, 0, 0, 0);
    DataCopyPadExtParams<TScale> padParamsScale(false, 0, 0, 0);
    DataCopyPad(srcLocal, srcGm[offset * tl_->tokensSize], copyParams, padParams);
    if (tl_->hasScale == 1) {
        LocalTensor<T1> tmpLocal = srcLocal[AlignUp(rows * tl_->tokensSize * sizeof(T1), blockSize) / sizeof(T1)];
        LocalTensor<TScale> scaleLocal = tmpLocal.template ReinterpretCast<TScale>();
        copyParams.blockLen = rows * sizeof(TScale);
        DataCopyPad(scaleLocal, srcScaleGm[offset], copyParams, padParamsScale);
    }
    queBind.EnQue(srcLocal);
}

template <typename T1, typename T2, typename TScale>
__aicore__ inline void KernelMoeReRouting<T1, T2, TScale>::CopyOut(int64_t rows, int64_t offset)
{
    LocalTensor<T1> dstLocal = queBind.DeQue<T1>();
    DataCopyExtParams copyParams(1, rows * tl_->tokensSize * sizeof(T1), 0, 0, 0);
    DataCopyPad(dstGm[offset * tl_->tokensSize], dstLocal, copyParams);
    if (tl_->hasScale == 1) {
        LocalTensor<T1> tmpLocal = dstLocal[AlignUp(rows * tl_->tokensSize * sizeof(T1), blockSize) / sizeof(T1)];
        LocalTensor<TScale> scaleLocal = tmpLocal.template ReinterpretCast<TScale>();
        copyParams.blockLen = rows * sizeof(TScale);
        DataCopyPad(dstScaleGm[offset], scaleLocal, copyParams);
    }
    queBind.FreeTensor(dstLocal);
}

template <typename T1, typename T2, typename TScale>
__aicore__ inline void KernelMoeReRouting<T1, T2, TScale>::ComputeAndCopyOutIndex(int64_t rows, int64_t offset,
                                                                                  int64_t head)
{
    LocalTensor indexLocal = idxOutQue.AllocTensor<int32_t>();
    CreateVecIndex(indexLocal, static_cast<int32_t>(head), rows);
    idxOutQue.EnQue(indexLocal);
    indexLocal = idxOutQue.DeQue<int32_t>();
    DataCopyExtParams copyParams(1, rows * sizeof(int32_t), 0, 0, 0);
    DataCopyPad(indexGm[offset], indexLocal, copyParams);
    idxOutQue.FreeTensor(indexLocal);
}

#endif // MOE_RE_ROUTING_H
