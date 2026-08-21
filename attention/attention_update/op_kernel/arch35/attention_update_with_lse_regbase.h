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
 * \file attention_update_with_lse_regbase.h
 * \brief
 */
#ifndef ATTENTION_UPDATE_WITH_LSE_REGBASE_H_
#define ATTENTION_UPDATE_WITH_LSE_REGBASE_H_

#include "attention_update_base_regbase.h"

namespace AttentionUpdateOpt {

template <typename goType>
class AttentionUpdateWithLse {
public:
    __aicore__ inline AttentionUpdateWithLse(TPipe *pipe, const AttentionUpdateTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax, GM_ADDR workSpace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyLseToUb(uint64_t lseGmOffset, uint64_t curBlockNum, uint64_t curBlockNumAlign);
    __aicore__ inline void CopyLseMaxToGm(uint64_t lseGmOffset, uint64_t curBlockNum);
    __aicore__ inline void CopyGoToUb(uint64_t goOffset, uint64_t curBlockNum, uint64_t bshInLoopNum,
                                      uint64_t goAlignNum);
    __aicore__ inline void CopyOutToGm(uint64_t outOffset, uint64_t curBlockNum, uint64_t bshInLoopNum);

    __aicore__ inline void ComputeMaxVF(uint32_t curBlockNum);
    template <bool hasTailRoll = false>
    __aicore__ inline void ComputeOutVF(uint32_t lseUbOffset, uint32_t bshNum, uint16_t bshInLoopNum,
                                        uint32_t dAlignNum, uint32_t goAlignNum, uint16_t unrollLoops);

    TPipe *pipe_ = nullptr;
    const AttentionUpdateTilingData *tilingData_;

    uint32_t blockIdx_;
    uint64_t bshInLoop_;
    uint64_t perCorePerLoopCount_;
    uint64_t lseBlockNum_;
    uint64_t goBlockNum_;
    uint64_t dAlignNum_;

    /* Tensor List */
    ListTensorDesc lseList_;
    ListTensorDesc goList_;

    /* global memory address */
    GlobalTensor<float> lseGm_;
    GlobalTensor<float> outLseMaxGm_;
    GlobalTensor<goType> goGm_;
    GlobalTensor<goType> outGm_;

    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> lseQueue_;
    TQue<QuePosition::VECIN, 1> goQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECOUT, 1> outLseMaxQueue_;

    TBuf<> ubTmpBuf_;
};

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::Init(GM_ADDR lse, GM_ADDR go, GM_ADDR out, GM_ADDR outLseMax,
                                                            GM_ADDR workSpace)
{
    blockIdx_ = AscendC::GetBlockIdx();
    lseBlockNum_ = UB_BLOCK_SIZE / sizeof(float);
    goBlockNum_ = UB_BLOCK_SIZE / sizeof(goType);
    uint64_t bshPerCoreCount =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->lastCoreCount : tilingData_->perCoreCount;
    perCorePerLoopCount_ = AscendC::Std::min(tilingData_->perCorePerLoopCount, bshPerCoreCount);
    bshInLoop_ = AscendC::Std::min(tilingData_->bshInLoop, perCorePerLoopCount_);
    uint64_t perCorePerLoopCountAlign = Ops::Base::CeilAlign(perCorePerLoopCount_, lseBlockNum_);
    dAlignNum_ = Ops::Base::CeilAlign(tilingData_->d, goBlockNum_);

    lseList_ = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void *>(lse));
    goList_ = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void *>(go));

    outLseMaxGm_.SetGlobalBuffer((__gm__ float *)outLseMax);
    outGm_.SetGlobalBuffer((__gm__ goType *)out);

    pipe_->InitBuffer(lseQueue_, BUFFER_NUM, tilingData_->sp * perCorePerLoopCountAlign * sizeof(float));
    pipe_->InitBuffer(goQueue_, BUFFER_NUM, tilingData_->sp * bshInLoop_ * dAlignNum_ * sizeof(goType));
    pipe_->InitBuffer(outLseMaxQueue_, BUFFER_NUM, perCorePerLoopCountAlign * sizeof(float));
    pipe_->InitBuffer(outQueue_, BUFFER_NUM, bshInLoop_ * dAlignNum_ * sizeof(goType));
    pipe_->InitBuffer(ubTmpBuf_, tilingData_->sp * perCorePerLoopCountAlign * sizeof(float));
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    bool hasTailRoll = tilingData_->sp % UNROLL_NUM;
    uint16_t unrollLoops = tilingData_->sp / UNROLL_NUM;

    // 核内循环，整块数据量
    uint64_t perLoopCount = perCorePerLoopCount_;
    // 核内循环总数
    uint64_t bshUbLoops =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->lastCoreLoops : tilingData_->perCoreLoops;
    for (int64_t bshUbLoopNum = 0; bshUbLoopNum < bshUbLoops; bshUbLoopNum++) {
        if (bshUbLoopNum == bshUbLoops - 1) {
            // 核内循环，尾块数据量
            perLoopCount = (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->lastCoreLastLoopCount :
                                                                         tilingData_->perCoreLastLoopCount;
        }

        uint64_t lseGmOffset = blockIdx_ * tilingData_->perCoreCount + bshUbLoopNum * perCorePerLoopCount_;
        uint64_t bshAlignNum = Ops::Base::CeilAlign(perLoopCount, lseBlockNum_);
        CopyLseToUb(lseGmOffset, perLoopCount, bshAlignNum);
        ComputeMaxVF(bshAlignNum);
        CopyLseMaxToGm(lseGmOffset, perLoopCount);

        int64_t bshInLoopNum = bshInLoop_;                                 // 核内循环，bshInLoop整块数据量
        int64_t bshInLoops = Ops::Base::CeilDiv(perLoopCount, bshInLoop_); // 核内循环，bshInLoop循环总数
        for (uint64_t bshInLoopCount = 0; bshInLoopCount < bshInLoops; bshInLoopCount++) {
            uint64_t goOffset = (blockIdx_ * tilingData_->perCoreCount + bshUbLoopNum * perCorePerLoopCount_ +
                                 bshInLoopCount * bshInLoop_) *
                                tilingData_->d;
            uint64_t lseUbOffset = bshInLoopCount * bshInLoop_;
            if (bshInLoopCount == bshInLoops - 1) {
                bshInLoopNum = perLoopCount - (bshInLoops - 1) * bshInLoop_; // 核内循环，bshInLoop尾块数据量
            }

            uint64_t goAlignNum = bshInLoopNum * dAlignNum_;
            CopyGoToUb(goOffset, tilingData_->d, bshInLoopNum, goAlignNum);
            if (hasTailRoll) {
                ComputeOutVF<true>(lseUbOffset, bshAlignNum, bshInLoopNum, dAlignNum_, goAlignNum, unrollLoops);
            } else {
                ComputeOutVF<false>(lseUbOffset, bshAlignNum, bshInLoopNum, dAlignNum_, goAlignNum, unrollLoops);
            }
            CopyOutToGm(goOffset, tilingData_->d, bshInLoopNum);
        }
    }
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::CopyLseToUb(uint64_t lseGmOffset, uint64_t curBlockNum,
                                                                   uint64_t curBlockNumAlign)
{
    LocalTensor<float> lseTensor = lseQueue_.template AllocTensor<float>();
    uint32_t blockLen = curBlockNum * sizeof(float);

    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(blockLen), static_cast<uint32_t>(0),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<float> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<float>(0)};

    for (uint32_t i = 0; i < tilingData_->sp; i++) {
        lseGm_.SetGlobalBuffer((__gm__ float *)lseList_.GetDataPtr<float>(i));
        DataCopyPad(lseTensor[i * curBlockNumAlign], lseGm_[lseGmOffset], params, padParamIdx);
    }
    lseQueue_.template EnQue<float>(lseTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::ComputeMaxVF(uint32_t curBlockNum)
{
    LocalTensor<float> lseTensor = lseQueue_.template DeQue<float>();
    LocalTensor<float> expTensor = ubTmpBuf_.Get<float>(tilingData_->sp * curBlockNum);
    LocalTensor<float> maxTensor = outLseMaxQueue_.template AllocTensor<float>();

    __ubuf__ float *lseAddr = (__ubuf__ float *)lseTensor.GetPhyAddr();
    __ubuf__ float *expAddr = (__ubuf__ float *)expTensor.GetPhyAddr();
    __ubuf__ float *maxAddr = (__ubuf__ float *)maxTensor.GetPhyAddr();

    uint32_t blockStride = static_cast<uint32_t>(curBlockNum);
    uint16_t spSize = static_cast<uint16_t>(tilingData_->sp);

    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = VREG_SIZE / dtypeSize;
    uint16_t vfLoop = Ops::Base::CeilDiv(blockStride, VL);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vreg1; // 处理INF后的lse
        AscendC::MicroAPI::RegTensor<float> vreg2; // reduce max
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5; // 累加 reduce sum
        AscendC::MicroAPI::RegTensor<float> vreg6;
        AscendC::MicroAPI::RegTensor<float> vreg7; // 可选输出 lse_max
        AscendC::MicroAPI::RegTensor<float> vreg8;
        AscendC::MicroAPI::RegTensor<float> vreg9;
        AscendC::MicroAPI::RegTensor<float> vreg10; // 待替换的-INF
        AscendC::MicroAPI::RegTensor<float> vreg11; // 搬入的初始Max
        AscendC::MicroAPI::RegTensor<float> vreg12; // 搬入的初始lse

        AscendC::MicroAPI::MaskReg preg1;
        AscendC::MicroAPI::MaskReg pregAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregCompare;
        uint32_t sreg = curBlockNum;
        static constexpr AscendC::MicroAPI::ExpSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                                    AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
        MicroAPI::Duplicate(vreg10, NEG_INF);
        for (uint16_t i = 0; i < vfLoop; i++) {
            preg1 = AscendC::MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(sreg);
            AscendC::MicroAPI::LoadAlign<float>(vreg11, lseAddr + i * VL);
            MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vreg11, POS_INF, pregAll);
            MicroAPI::Select<float>(vreg2, vreg10, vreg11, pregCompare);
            for (uint16_t j = 1; j < spSize; j++) {
                AscendC::MicroAPI::LoadAlign<float>(vreg12, lseAddr + i * VL + j * blockStride);
                MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vreg12, POS_INF, pregAll);
                MicroAPI::Select<float>(vreg1, vreg10, vreg12, pregCompare);
                AscendC::MicroAPI::Max<float>(vreg2, vreg2, vreg1, preg1);
            }

            AscendC::MicroAPI::Duplicate(vreg5, static_cast<float>(0), preg1);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::LoadAlign<float>(vreg12, lseAddr + i * VL + j * blockStride);
                MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vreg12, POS_INF, pregAll);
                MicroAPI::Select<float>(vreg1, vreg10, vreg12, pregCompare);
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg1, vreg2, preg1);
                AscendC::MicroAPI::Exp<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg3, preg1);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg4, vreg5, preg1);
            }
            AscendC::MicroAPI::Log<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg5, preg1);
            AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg7, vreg6, vreg2, preg1);
            AscendC::MicroAPI::StoreAlign<float>(maxAddr + i * VL, vreg7, preg1);
            for (uint16_t j = 0; j < spSize; j++) {
                AscendC::MicroAPI::LoadAlign<float>(vreg12, lseAddr + i * VL + j * blockStride);
                MicroAPI::Compares<float, CMPMODE::EQ>(pregCompare, vreg12, POS_INF, pregAll);
                MicroAPI::Select<float>(vreg1, vreg10, vreg12, pregCompare);
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg8, vreg1, vreg7, preg1);
                AscendC::MicroAPI::Exp<float, &mode>(vreg9, vreg8, preg1);
                AscendC::MicroAPI::StoreAlign<float>(expAddr + i * VL + j * blockStride, vreg9, preg1);
            }
        }
    }
    lseQueue_.template FreeTensor<float>(lseTensor);
    outLseMaxQueue_.template EnQue<float>(maxTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::CopyLseMaxToGm(uint64_t lseGmOffset, uint64_t curBlockNum)
{
    LocalTensor<float> outLseMaxTensor = outLseMaxQueue_.template DeQue<float>();
    uint32_t blockLen = curBlockNum * sizeof(float);

    DataCopyExtParams params = {static_cast<uint16_t>(1), static_cast<uint32_t>(blockLen), static_cast<uint32_t>(0),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outLseMaxGm_[lseGmOffset], outLseMaxTensor, params);
    outLseMaxQueue_.template FreeTensor<float>(outLseMaxTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::CopyGoToUb(uint64_t goOffset, uint64_t curBlockNum,
                                                                  uint64_t bshInLoopNum, uint64_t goAlignNum)
{
    LocalTensor<goType> goTensor = goQueue_.template AllocTensor<goType>();
    uint32_t blockLen = curBlockNum * sizeof(goType);
    DataCopyExtParams params = {static_cast<uint16_t>(bshInLoopNum), static_cast<uint32_t>(blockLen),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<goType> padParamIdx = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                static_cast<goType>(0)};
    for (uint32_t i = 0; i < tilingData_->sp; i++) {
        goGm_.SetGlobalBuffer((__gm__ goType *)goList_.GetDataPtr<goType>(i));
        DataCopyPad(goTensor[i * goAlignNum], goGm_[goOffset], params, padParamIdx);
    }
    goQueue_.template EnQue<goType>(goTensor);
}

template <typename goType>
template <bool hasTailRoll>
__aicore__ inline void AttentionUpdateWithLse<goType>::ComputeOutVF(uint32_t lseUbOffset, uint32_t bshNum,
                                                                    uint16_t bshInLoopNum, uint32_t dAlignNum,
                                                                    uint32_t goAlignNum, uint16_t unrollLoops)
{
    LocalTensor<goType> sumTensor = outQueue_.template AllocTensor<goType>();
    LocalTensor<float> expTensor = ubTmpBuf_.Get<float>(tilingData_->sp * bshNum);
    LocalTensor<goType> goTensor = goQueue_.template DeQue<goType>();

    __ubuf__ goType *sumAddr = (__ubuf__ goType *)sumTensor.GetPhyAddr();
    __ubuf__ float *expAddr = (__ubuf__ float *)expTensor.GetPhyAddr();
    __ubuf__ goType *goAddr = (__ubuf__ goType *)goTensor.GetPhyAddr();

    uint32_t dNum = static_cast<uint32_t>(tilingData_->d);
    uint16_t spSize = static_cast<uint16_t>(tilingData_->sp);

    uint32_t dtypeSize = sizeof(float);
    uint64_t VL = VREG_SIZE / dtypeSize;
    uint16_t vfLoop = Ops::Base::CeilDiv(tilingData_->d, VL);
    uint16_t unrollTimes = UNROLL_NUM;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vreg0, vreg1; // 搬入的lse
        AscendC::MicroAPI::RegTensor<float> vreg2, vreg3; // 搬入的go
        AscendC::MicroAPI::RegTensor<float> vreg4;        // 求和的结果

        AscendC::MicroAPI::MaskReg preg1;
        for (uint16_t i = 0; i < bshInLoopNum; i++) {
            uint32_t sreg = dNum;
            for (uint16_t j = 0; j < vfLoop; j++) {
                preg1 = AscendC::MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(sreg);
                AscendC::MicroAPI::Duplicate(vreg4, 0, preg1);
                for (uint16_t k = 0; k < unrollLoops; k++) {
                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        vreg0, expAddr + (k * unrollTimes * bshNum + lseUbOffset + i));
                    ops::LoadOneTensorForDtypeT<goType>(goAddr, vreg2, preg1,
                                                        k * unrollTimes * goAlignNum + i * dAlignNum + j * VL);
                    AscendC::MicroAPI::MulAddDst(vreg4, vreg0, vreg2, preg1);

                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        vreg1, expAddr + ((k * unrollTimes + 1) * bshNum + lseUbOffset + i));
                    ops::LoadOneTensorForDtypeT<goType>(goAddr, vreg3, preg1,
                                                        (k * unrollTimes + 1) * goAlignNum + i * dAlignNum + j * VL);
                    AscendC::MicroAPI::MulAddDst(vreg4, vreg1, vreg3, preg1);
                }
                if constexpr (hasTailRoll) {
                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        vreg0, expAddr + ((spSize - 1) * bshNum + lseUbOffset + i));
                    ops::LoadOneTensorForDtypeT<goType>(goAddr, vreg2, preg1,
                                                        (spSize - 1) * goAlignNum + i * dAlignNum + j * VL);
                    AscendC::MicroAPI::MulAddDst(vreg4, vreg0, vreg2, preg1);
                }
                ops::StoreOneTensorForDtypeT<goType>(sumAddr, vreg4, preg1, i * dAlignNum + j * VL);
            }
        }
    }
    goQueue_.template FreeTensor<goType>(goTensor);
    outQueue_.template EnQue<goType>(sumTensor);
}

template <typename goType>
__aicore__ inline void AttentionUpdateWithLse<goType>::CopyOutToGm(uint64_t outOffset, uint64_t curBlockNum,
                                                                   uint64_t bshInLoopNum)
{
    LocalTensor<goType> outTensor = outQueue_.template DeQue<goType>();
    uint32_t blockLen = curBlockNum * sizeof(goType);
    DataCopyExtParams params = {static_cast<uint16_t>(bshInLoopNum), static_cast<uint32_t>(blockLen),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outGm_[outOffset], outTensor, params);
    outQueue_.template FreeTensor<goType>(outTensor);
}
} // namespace AttentionUpdateOpt
#endif // ATTENTION_UPDATE_WITH_LSE_REGBASE_H_
