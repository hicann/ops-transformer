/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_update_without_lse_regbase.h
 * \brief
 */
#ifndef ATTENTION_UPDATE_WITHOUT_LSE_REGBASE_H_
#define ATTENTION_UPDATE_WITHOUT_LSE_REGBASE_H_

#include "attention_update_base_regbase.h"

namespace AttentionUpdateOpt {

template <typename goType>
class AttentionUpdateWithoutLse {
public:
    __aicore__ inline AttentionUpdateWithoutLse(TPipe *pipe, const AttentionUpdateTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax, GM_ADDR workSpace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInLse(uint64_t lseGmOffset, uint64_t curBlockNum, uint64_t curBlockNumAlign);
    __aicore__ inline void CopyInGo(uint64_t goOffset, uint64_t curBlockNum, uint64_t bshInLoopNum, uint64_t goAlign);
    __aicore__ inline void CopyOut(uint64_t outOffset, uint64_t curBlockNum, uint64_t bshInLoopNum);

    __aicore__ inline void ComputeLseMVF(uint32_t curBlockNum);
    template <bool hasTailRoll = false>
    __aicore__ inline void ComputeOutputVF(uint32_t lseUbOffset, uint32_t bshNum, uint16_t bshInLoopNum,
                                           uint32_t dAlign, uint32_t goAlign, uint16_t unrollLoops);

    TPipe *pipe_ = nullptr;
    const AttentionUpdateTilingData *tilingData_;

    /* Tensor List */
    ListTensorDesc lseList_;
    ListTensorDesc goList_;

    /* global memory address */
    GlobalTensor<float> lseGm_;
    GlobalTensor<float> outLseMGm_;
    GlobalTensor<goType> goGm_;
    GlobalTensor<goType> outputGm_;

    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> lseInQue_;
    TQue<QuePosition::VECIN, 1> goInQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<> ubTmpBuf_;

    uint32_t blockIdx_;
    uint64_t lseBlockNum_;
    uint64_t goBlockNum_;
    uint64_t bshInLoop_;
    uint64_t perCorePerLoopCount_;
    uint64_t dAlign_;
};

template <typename goType>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::Init(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax,
                                                               GM_ADDR workSpace)
{
    lseList_ = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void *>(lse));
    goList_ = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void *>(go));

    outLseMGm_.SetGlobalBuffer((__gm__ float *)outLseMax);
    outputGm_.SetGlobalBuffer((__gm__ goType *)out);

    blockIdx_ = AscendC::GetBlockIdx();
    lseBlockNum_ = UB_BLOCK_SIZE / sizeof(float);
    goBlockNum_ = UB_BLOCK_SIZE / sizeof(goType);
    uint64_t bshPerCoreCount =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->lastCoreCount : tilingData_->perCoreCount;
    perCorePerLoopCount_ = AscendC::Std::min(tilingData_->perCorePerLoopCount, bshPerCoreCount);
    bshInLoop_ = AscendC::Std::min(tilingData_->bshInLoop, perCorePerLoopCount_);
    uint64_t perCorePerLoopCountAlign = Ops::Base::CeilAlign(perCorePerLoopCount_, lseBlockNum_);
    dAlign_ = Ops::Base::CeilAlign(tilingData_->d, goBlockNum_);

    pipe_->InitBuffer(lseInQue_, BUFFER_NUM, tilingData_->sp * perCorePerLoopCountAlign * sizeof(float));
    pipe_->InitBuffer(goInQue_, BUFFER_NUM, tilingData_->sp * bshInLoop_ * dAlign_ * sizeof(goType));
    pipe_->InitBuffer(outQue_, BUFFER_NUM, bshInLoop_ * dAlign_ * sizeof(goType));
    pipe_->InitBuffer(ubTmpBuf_, tilingData_->sp * perCorePerLoopCountAlign * sizeof(float));
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    bool hasTailRoll = tilingData_->sp % UNROLL_NUM;
    uint16_t unrollLoops = tilingData_->sp / UNROLL_NUM;

    uint64_t bshUbCount = perCorePerLoopCount_;
    uint64_t bshInUbLoops =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->lastCoreLoops : tilingData_->perCoreLoops;
    for (int64_t bshInUbLoopNum = 0; bshInUbLoopNum < bshInUbLoops; bshInUbLoopNum++) {
        if (bshInUbLoopNum == bshInUbLoops - 1) {
            // 核内循环，尾块数据量
            bshUbCount = (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->lastCoreLastLoopCount :
                                                                       tilingData_->perCoreLastLoopCount;
        }

        uint64_t lseGmOffset = blockIdx_ * tilingData_->perCoreCount + bshInUbLoopNum * perCorePerLoopCount_;
        uint64_t bshUbAlignNum = Ops::Base::CeilAlign(bshUbCount, lseBlockNum_);
        CopyInLse(lseGmOffset, bshUbCount, bshUbAlignNum);
        ComputeLseMVF(bshUbAlignNum);

        int64_t bshInLoopNum = bshInLoop_;
        int64_t bshInLoops = Ops::Base::CeilDiv(bshUbCount, bshInLoop_);
        for (uint64_t bshInLoopCount = 0; bshInLoopCount < bshInLoops; bshInLoopCount++) {
            uint64_t goOffset = (blockIdx_ * tilingData_->perCoreCount + bshInUbLoopNum * perCorePerLoopCount_ +
                                 bshInLoopCount * bshInLoop_) *
                                tilingData_->d;
            uint64_t lseUbOffset = bshInLoopCount * bshInLoop_;
            if (bshInLoopCount == bshInLoops - 1) {
                bshInLoopNum = bshUbCount - (bshInLoops - 1) * bshInLoop_;
            }

            uint64_t goAlign = bshInLoopNum * dAlign_;
            CopyInGo(goOffset, tilingData_->d, bshInLoopNum, goAlign);
            if (hasTailRoll) {
                ComputeOutputVF<true>(lseUbOffset, bshUbAlignNum, bshInLoopNum, dAlign_, goAlign, unrollLoops);
            } else {
                ComputeOutputVF<false>(lseUbOffset, bshUbAlignNum, bshInLoopNum, dAlign_, goAlign, unrollLoops);
            }
            CopyOut(goOffset, tilingData_->d, bshInLoopNum);
        }
    }
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::CopyInLse(uint64_t lseGmOffset, uint64_t curBlockNum,
                                                                    uint64_t curBlockNumAlign)
{
    LocalTensor<float> lseUbTensor = lseInQue_.template AllocTensor<float>();
    uint32_t blockLen = curBlockNum * sizeof(float);

    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(blockLen), static_cast<uint32_t>(0),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<float> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<float>(0)};

    for (uint32_t i = 0; i < tilingData_->sp; i++) {
        lseGm_.SetGlobalBuffer((__gm__ float *)lseList_.GetDataPtr<float>(i));
        DataCopyPad(lseUbTensor[i * curBlockNumAlign], lseGm_[lseGmOffset], params, padParamIdx);
    }
    lseInQue_.template EnQue<float>(lseUbTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::ComputeLseMVF(uint32_t curBlockNum)
{
    LocalTensor<float> lseUbTensor = lseInQue_.template DeQue<float>();
    LocalTensor<float> expUbTensor = ubTmpBuf_.Get<float>(tilingData_->sp * curBlockNum);

    __ubuf__ float *lseUbAddr = (__ubuf__ float *)lseUbTensor.GetPhyAddr();
    __ubuf__ float *expUbAddr = (__ubuf__ float *)expUbTensor.GetPhyAddr();

    uint32_t blockStride = static_cast<uint32_t>(curBlockNum);
    uint16_t spSize = static_cast<uint16_t>(tilingData_->sp);
    uint32_t VL = VREG_SIZE / sizeof(float);
    uint16_t vfLoop = Ops::Base::CeilDiv(blockStride, VL);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregLse; // 处理INF后的lse
        AscendC::MicroAPI::RegTensor<float> vregMax; // reduce max
        AscendC::MicroAPI::RegTensor<float> vregSubRes;
        AscendC::MicroAPI::RegTensor<float> vregExpRes;
        AscendC::MicroAPI::RegTensor<float> vregSum; // 累加 reduce sum
        AscendC::MicroAPI::RegTensor<float> vregLogRes;
        AscendC::MicroAPI::RegTensor<float> vregLseM; // 可选输出 lse_max
        AscendC::MicroAPI::RegTensor<float> vregSubLseMRes;
        AscendC::MicroAPI::RegTensor<float> vregLseExpFinal;
        AscendC::MicroAPI::RegTensor<float> vregNegInf;      // 待替换的-INF
        AscendC::MicroAPI::RegTensor<float> vregOriginalMax; // 搬入的初始Max
        AscendC::MicroAPI::RegTensor<float> vregOriginalLse; // 搬入的初始lse

        AscendC::MicroAPI::MaskReg preg1;
        AscendC::MicroAPI::MaskReg pregAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregCompare;
        uint32_t sreg = curBlockNum;
        static constexpr AscendC::MicroAPI::ExpSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                                    AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
        MicroAPI::Duplicate(vregNegInf, NEG_INF);
        for (uint16_t i = 0; i < vfLoop; i++) {
            preg1 = AscendC::MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(sreg);
            AscendC::MicroAPI::LoadAlign<float>(vregOriginalMax, lseUbAddr + i * VL);
            MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vregOriginalMax, POS_INF, pregAll);
            MicroAPI::Select<float>(vregMax, vregNegInf, vregOriginalMax, pregCompare);
            for (uint16_t j = 1; j < spSize; j++) {
                AscendC::MicroAPI::LoadAlign<float>(vregOriginalLse, lseUbAddr + i * VL + j * blockStride);
                MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vregOriginalLse, POS_INF, pregAll);
                MicroAPI::Select<float>(vregLse, vregNegInf, vregOriginalLse, pregCompare);
                AscendC::MicroAPI::Max<float>(vregMax, vregMax, vregLse, preg1);
            }

            AscendC::MicroAPI::Duplicate(vregSum, static_cast<float>(0), preg1);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::LoadAlign<float>(vregOriginalLse, lseUbAddr + i * VL + j * blockStride);
                MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vregOriginalLse, POS_INF, pregAll);
                MicroAPI::Select<float>(vregLse, vregNegInf, vregOriginalLse, pregCompare);
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregSubRes, vregLse, vregMax,
                                                                                         preg1);
                AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregExpRes, vregSubRes, preg1);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregExpRes, vregSum,
                                                                                         preg1);
            }
            AscendC::MicroAPI::Log<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregLogRes, vregSum, preg1);
            AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregLseM, vregLogRes, vregMax,
                                                                                     preg1);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::LoadAlign<float>(vregOriginalLse, lseUbAddr + i * VL + j * blockStride);
                MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vregOriginalLse, POS_INF, pregAll);
                MicroAPI::Select<float>(vregLse, vregNegInf, vregOriginalLse, pregCompare);
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregSubLseMRes, vregLse,
                                                                                         vregLseM, preg1);
                AscendC::MicroAPI::Exp<float, &mode>(vregLseExpFinal, vregSubLseMRes, preg1);
                AscendC::MicroAPI::StoreAlign<float>(expUbAddr + i * VL + j * blockStride, vregLseExpFinal, preg1);
            }
        }
    }
    lseInQue_.template FreeTensor<float>(lseUbTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::CopyInGo(uint64_t goOffset, uint64_t curBlockNum,
                                                                   uint64_t bshInLoopNum, uint64_t goAlign)
{
    LocalTensor<goType> goUbTensor = goInQue_.template AllocTensor<goType>();
    uint32_t blockLen = curBlockNum * sizeof(goType);
    DataCopyExtParams params = {static_cast<uint16_t>(bshInLoopNum), static_cast<uint32_t>(blockLen),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<goType> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                static_cast<goType>(0)};
    for (uint32_t i = 0; i < tilingData_->sp; i++) {
        goGm_.SetGlobalBuffer((__gm__ goType *)goList_.GetDataPtr<goType>(i));
        DataCopyPad(goUbTensor[i * goAlign], goGm_[goOffset], params, padParamIdx);
    }
    goInQue_.template EnQue<goType>(goUbTensor);
}

template <typename goType>
template <bool hasTailRoll>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::ComputeOutputVF(uint32_t lseUbOffset, uint32_t bshNum,
                                                                          uint16_t bshInLoopNum, uint32_t dAlign,
                                                                          uint32_t goAlign, uint16_t unrollLoops)
{
    LocalTensor<goType> sumUbTensor = outQue_.template AllocTensor<goType>();
    LocalTensor<float> expUbTensor = ubTmpBuf_.Get<float>(tilingData_->sp * bshNum);
    LocalTensor<goType> goUbTensor = goInQue_.template DeQue<goType>();

    __ubuf__ goType *sumUbAddr = (__ubuf__ goType *)sumUbTensor.GetPhyAddr();
    __ubuf__ float *expUbAddr = (__ubuf__ float *)expUbTensor.GetPhyAddr();
    __ubuf__ goType *goUbAddr = (__ubuf__ goType *)goUbTensor.GetPhyAddr();

    uint32_t dRealNum = static_cast<uint32_t>(tilingData_->d);
    uint16_t spSize = static_cast<uint16_t>(tilingData_->sp);

    uint64_t VL = VREG_SIZE / sizeof(float);
    uint16_t vfLoop = Ops::Base::CeilDiv(tilingData_->d, VL);
    uint16_t unrollTimes = UNROLL_NUM;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregLse0, vregLse1; // 搬入的lse
        AscendC::MicroAPI::RegTensor<float> vregGo0, vregGo1;   // 搬入的go
        AscendC::MicroAPI::RegTensor<float> vregSumRes;         // 求和的结果

        AscendC::MicroAPI::MaskReg preg1;
        for (uint16_t i = 0; i < bshInLoopNum; i++) {
            uint32_t sreg = dRealNum;
            for (uint16_t j = 0; j < vfLoop; j++) {
                preg1 = AscendC::MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(sreg);
                AscendC::MicroAPI::Duplicate(vregSumRes, 0, preg1);
                for (uint16_t k = 0; k < unrollLoops; k++) {
                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        vregLse0, expUbAddr + (k * unrollTimes * bshNum + lseUbOffset + i));
                    ops::LoadOneTensorForDtypeT<goType>(goUbAddr, vregGo0, preg1,
                                                        k * unrollTimes * goAlign + i * dAlign + j * VL);
                    AscendC::MicroAPI::MulAddDst(vregSumRes, vregLse0, vregGo0, preg1);

                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        vregLse1, expUbAddr + ((k * unrollTimes + 1) * bshNum + lseUbOffset + i));
                    ops::LoadOneTensorForDtypeT<goType>(goUbAddr, vregGo1, preg1,
                                                        (k * unrollTimes + 1) * goAlign + i * dAlign + j * VL);
                    AscendC::MicroAPI::MulAddDst(vregSumRes, vregLse1, vregGo1, preg1);
                }
                if constexpr (hasTailRoll) {
                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        vregLse0, expUbAddr + ((spSize - 1) * bshNum + lseUbOffset + i));
                    ops::LoadOneTensorForDtypeT<goType>(goUbAddr, vregGo0, preg1,
                                                        (spSize - 1) * goAlign + i * dAlign + j * VL);
                    AscendC::MicroAPI::MulAddDst(vregSumRes, vregLse0, vregGo0, preg1);
                }
                ops::StoreOneTensorForDtypeT<goType>(sumUbAddr, vregSumRes, preg1, i * dAlign + j * VL);
            }
        }
    }
    goInQue_.template FreeTensor<goType>(goUbTensor);
    outQue_.template EnQue<goType>(sumUbTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithoutLse<goType>::CopyOut(uint64_t outOffset, uint64_t curBlockNum,
                                                                  uint64_t bshInLoopNum)
{
    LocalTensor<goType> outTensor = outQue_.template DeQue<goType>();
    uint32_t blockLen = curBlockNum * sizeof(goType);
    DataCopyExtParams params = {static_cast<uint16_t>(bshInLoopNum), static_cast<uint32_t>(blockLen),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[outOffset], outTensor, params);
    outQue_.template FreeTensor<goType>(outTensor);
}
} // namespace AttentionUpdateOpt
#endif // ATTENTION_UPDATE_WITHOUT_LSE_REGBASE_H_
