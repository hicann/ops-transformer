/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_stage2.h
 * \brief stage2：state 递推全程 highType，matmul 为 lowType(A/B) → highType(C)。
 *        state / v_new 作 matmul 输入前先 cast 回 lowType（写入 bf16State_ / bf16VNew_）；
 *        attn_inter 以 highType 写入 attnInter_（out_ 仍 lowType，由 stage3 汇总后写出）。
 *        hasG 由 tilingKey 编译期分发，不再使用运行时 gOptional 开关。
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

template <typename lowType, typename highType>
using StageTwoMT = matmul::MatmulImpl<MatmulType<TPosition::GM, CubeFormat::ND, lowType, true>,
                                      MatmulType<TPosition::GM, CubeFormat::ND, lowType, true>,
                                      MatmulType<TPosition::GM, CubeFormat::ND, highType>>;

template <typename lowType, typename highType>
struct StageTwoParams {
    GlobalTensor<lowType> qPrime;      // (Nv, Sp, Dk)
    GlobalTensor<highType> vInner;     // (Nv, Sp, Dv)  highType v_new（CalVPrime atomic add 目标）
    GlobalTensor<highType> gCum;       // (Nv, Sp)
    GlobalTensor<lowType> kCumdecay;   // (Nv, Sp, Dk)
    GlobalTensor<highType> curState;   // (Nv, Dv, Dk)  highType
    GlobalTensor<highType> finalState; // (Nv, Dv, Dk)  highType（CalStateNew atomic add 目标）
    GlobalTensor<lowType> kg;          // (Nv, Sp, Dk)
    GlobalTensor<highType> attnInter;  // (Sp, Nv, Dv)  highType（CalAttnInter 覆盖写目标）
    GlobalTensor<lowType> bf16State; // (Nv, Dv, Dk)  state 的 lowType 影子（CalVPrime/CalAttnInter 的 B 输入）
    GlobalTensor<lowType> bf16VNew;  // (Nv, Sp, Dv)  v_new 的 lowType 影子（CalStateNew 的 A 输入）
    GM_ADDR ws;
    StageTwoMT<lowType, highType> *mm1;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
};

template <typename lowType, typename highType, bool hasG>
class Stage2 {
public:
    __aicore__ inline void Init(StageTwoParams<lowType, highType> *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(highType)) * (BLOCK_SIZE / sizeof(highType));
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(highType)) * (BLOCK_SIZE / sizeof(highType));
        curChunkSize_ = chunkSize_;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        uint64_t inElem = Std::max((uint64_t)Dv_ * curDk_, (uint64_t)chunkSize_ * paddedDv_);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inElem * sizeof(highType));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, Dv_ * curDk_ * sizeof(highType));
        uint64_t lowCastSize = Std::max((uint64_t)Dv_ * curDk_, (uint64_t)chunkSize_ * paddedDv_) * sizeof(lowType);
        pipe_->InitBuffer(bf16CastQueue_, BUFFER_NUM_ONE, lowCastSize);
        pipe_->InitBuffer(tmpBuff_, (BLOCK_FLOAT_NUM + NUM_ONE) * sizeof(highType));
        uint32_t buffOffset = 0;
        lastGCum_ = tmpBuff_.GetWithOffset<highType>(static_cast<uint32_t>(NUM_ONE), buffOffset);
    }

    __aicore__ inline void Process()
    {
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= AIC_AIV_1_1;
        }
        int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
        int64_t nvStart = coreId * nvPerCore;
        int64_t nvEnd = nvStart + nvPerCore;
        nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
            curChunkSize_ = chunkSize_;
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                auto curState = (cId == 0) ? sTP_->curState[nvId * Dv_ * Dk_] : sTP_->finalState[nvId * Dv_ * Dk_];
                auto finalState = sTP_->finalState[nvId * Dv_ * Dk_];
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0) {
                        CopyIn(curState, Dv_, Dk_);
                        CalGCumExp(sTP_->gCum[nvId * Sp_ + length]);
                        CopyOutLowState(sTP_->bf16State[nvId * Dv_ * Dk_]);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);
                    CrossCoreWaitFlag(0x2);
                    if (GetSubBlockIdx() == 0) {
                        CopyOutState(finalState);
                        CastVNewToLow(sTP_->vInner[nvId * Sp_ * Dv_ + length * Dv_],
                                      sTP_->bf16VNew[nvId * Sp_ * Dv_ + length * Dv_]);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
                    CrossCoreWaitFlag(0x4);
                }
                if ASCEND_IS_AIC {
                    uint64_t mmOffset0 = nvId * Sp_ * Dk_ + length * Dk_;
                    uint64_t mmOffset1 = nvId * Sp_ * Dv_ + length * Dv_;
                    uint64_t attnInterOffset = nvId * Dv_ + length * Nv_ * Dv_;
                    CrossCoreWaitFlag(0x6);
                    CalVPrime(sTP_->kCumdecay[mmOffset0], sTP_->bf16State[nvId * Dv_ * Dk_], sTP_->vInner[mmOffset1]);
                    CalAttnInter(sTP_->qPrime[mmOffset0], sTP_->bf16State[nvId * Dv_ * Dk_],
                                 sTP_->attnInter[attnInterOffset]);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
                    CrossCoreWaitFlag(0x5);
                    CalStateNew(sTP_->bf16VNew[mmOffset1], sTP_->kg[mmOffset0], finalState);
                    SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
                }
            }
        }
    }

    __aicore__ inline void CalGCumExp(GlobalTensor<highType> gCum)
    {
        if constexpr (hasG) {
            DataCacheCleanAndInvalid<highType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                gCum[curChunkSize_ - 1]);
            highType tmpVal = gCum.GetValue(curChunkSize_ - 1);
            lastGCum_.SetValue(0, tmpVal);
            SetFlag<HardEvent::S_V>(S_V_EVENT);
            WaitFlag<HardEvent::S_V>(S_V_EVENT);
            Exp<highType, 0, true>(lastGCum_, lastGCum_, 1);
        } else {
            lastGCum_.SetValue(0, static_cast<highType>(1.0f));
        }
        highType tmpVal = lastGCum_.GetValue(0);
        auto stateIn = inQueue_.DeQue<highType>();
        auto stateOut = outQueue_.AllocTensor<highType>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        Muls(stateOut, stateIn, tmpVal, Dv_ * curDk_);
        auto lowOut = bf16CastQueue_.AllocTensor<lowType>();
        Cast(lowOut, stateIn, RoundMode::CAST_RINT, Dv_ * curDk_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(stateOut);
        bf16CastQueue_.EnQue(lowOut);
        inQueue_.FreeTensor(stateIn);
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<lowType> qPrime, GlobalTensor<lowType> state,
                                        GlobalTensor<highType> attnInter)
    {
        RunMatmul(qPrime, false, state, true, attnInter, curChunkSize_, Dv_, Dk_, 0, Nv_ * Dv_);
    }

    __aicore__ inline void CalVPrime(GlobalTensor<lowType> kCumdecay, GlobalTensor<lowType> state,
                                     GlobalTensor<highType> vPrime)
    {
        RunMatmul(kCumdecay, false, state, true, vPrime, curChunkSize_, Dv_, Dk_, 1);
    }

    __aicore__ inline void CalStateNew(GlobalTensor<lowType> vInner, GlobalTensor<lowType> kg,
                                       GlobalTensor<highType> state)
    {
        RunMatmul(vInner, true, kg, false, state, Dv_, Dk_, curChunkSize_, 1);
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        DataCopyExtParams inParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0), 0, 0};
        int padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void CopyOutState(GlobalTensor<highType> stateNew)
    {
        CopyOut<highType>(stateNew, Dv_, Dk_, false);
    }

    __aicore__ inline void CopyOutLowState(GlobalTensor<lowType> dst)
    {
        auto outLocal = bf16CastQueue_.DeQue<lowType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(Dv_);
        copyParams.blockLen = static_cast<uint32_t>(Dk_ * sizeof(lowType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>(0);
        DataCopyPad(dst, outLocal, copyParams);
        bf16CastQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void CastVNewToLow(GlobalTensor<highType> vNew, GlobalTensor<lowType> lowVNew)
    {
        CopyIn<highType>(vNew, curChunkSize_, Dv_);
        auto vIn = inQueue_.DeQue<highType>();
        auto lowOut = bf16CastQueue_.AllocTensor<lowType>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        Cast(lowOut, vIn, RoundMode::CAST_RINT, curChunkSize_ * paddedDv_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        bf16CastQueue_.EnQue(lowOut);
        inQueue_.FreeTensor(vIn);
        auto outLocal = bf16CastQueue_.DeQue<lowType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curChunkSize_);
        copyParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(lowType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>(0);
        DataCopyPad(lowVNew, outLocal, copyParams);
        bf16CastQueue_.FreeTensor(outLocal);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>(0);
        if (setAtomic) {
            SetAtomicAdd<outType>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    __aicore__ inline void RunMatmul(GlobalTensor<lowType> a, bool transA, GlobalTensor<lowType> b, bool transB,
                                     GlobalTensor<highType> c, int64_t m, int64_t n, int64_t k, int32_t atomic = 0,
                                     int64_t orgN = 0)
    {
        if (orgN > 0) {
            sTP_->mm1->SetOrgShape(m, n, k, k, orgN);
        } else {
            sTP_->mm1->SetOrgShape(m, n, k);
        }
        sTP_->mm1->SetSingleShape(m, n, k);
        sTP_->mm1->SetTensorA(a, transA);
        sTP_->mm1->SetTensorB(b, transB);
        sTP_->mm1->IterateAll(c, atomic);
        sTP_->mm1->End();
    }

    StageTwoParams<lowType, highType> *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> bf16CastQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<highType> lastGCum_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int32_t paddedDv_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_H
