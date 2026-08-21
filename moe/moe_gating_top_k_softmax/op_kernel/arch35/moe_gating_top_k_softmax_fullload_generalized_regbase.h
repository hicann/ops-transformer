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
 * \file moe_gating_top_k_softmax_fullload_generalized_regbase.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_FULLLOAD_GENERALIZED_REGBASE_H
#define MOE_GATING_TOP_K_SOFTMAX_FULLLOAD_GENERALIZED_REGBASE_H
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"

namespace MoeGatingTopKSoftmax {
using namespace AscendC;

constexpr int32_t FLOAT32_NEG_INF = 0xFF800000; // -inf -2139095040
constexpr int64_t REPEAT_BLOCKS = 8;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t REPEAT_BYTES = 256;
constexpr int64_t B8_BLOCK_COUNT = 32;
constexpr int64_t B32_BLOCK_COUNT = 8;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t KEY_VALUE_FACTOR = 2;
constexpr uint8_t VALUE_PATTERN = 1;
constexpr uint8_t INDEX_PATTERN = 2;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t CONSTANT_THREE = 3;
constexpr int64_t CONSTANT_TWO = 2;
constexpr int64_t B32_VF_COUNT = Ops::Base::GetVRegSize() / sizeof(int32_t);

static constexpr AscendC::MicroAPI::CastTrait castTrait = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

template <typename T, bool hasFinished = false, bool needPadNegInf = false>
class MoeGatingTopKSoftmaxFullloadGenerlized {
public:
    __aicore__ inline MoeGatingTopKSoftmaxFullloadGenerlized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                const MoeGatingTopKSoftmaxRegbaseTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitIndices();
    __aicore__ inline void CopyInX(int64_t loop, int64_t rowCount);
    __aicore__ inline void ComputeSoftmax(int64_t rowCount);
    __aicore__ inline void ComputeTopK(int64_t row);
    __aicore__ inline void HandleOneRepeatSortNum(int64_t rowCount, const LocalTensor<float> &softmaxTensor,
                                                  const LocalTensor<float> &sortedTensor,
                                                  LocalTensor<uint32_t> &expertIdxTensor);
    __aicore__ inline void ComputeRowIdx(int64_t loop, int64_t rowCount);
    __aicore__ inline void CopyOutRowIdx(int64_t loop, int64_t rowCount);
    __aicore__ inline void CopyYExpertIdxOut(int64_t loop, int64_t rowCount);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> finishedInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> rowIdxOutQueue_;

    TBuf<TPosition::VECCALC> expertIdxBuf_;
    TBuf<TPosition::VECCALC> rowIdxBaseBuf_;
    TBuf<TPosition::VECCALC> softmaxBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<bool> finishedGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> rowIdxGm_;

    const MoeGatingTopKSoftmaxRegbaseTilingData *tilingData_ = nullptr;

    int64_t blockIdx_ = 0;
    int64_t perCoreRowCount_ = 0;
    int64_t curCoreRowCount_ = 0;

    int64_t curCoreLoopCount_ = 0;
    int64_t perLoopRowCount_ = 0;
    int64_t lastLoopRowCount_ = 0;

    int64_t curentRowBaseIndex_ = 0;

    int64_t expertCount_ = 0;
    int64_t k_ = 0;
    int64_t expertCountAlign_ = 0;
    int64_t kAlign_ = 0;
    int64_t rows_ = 0;
    int64_t padNegInfCount_ = 0;
    int64_t sortRepeatTimes_ = 0;
};

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::Init(
    GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
    const MoeGatingTopKSoftmaxRegbaseTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
        curCoreLoopCount_ = tilingData_->lastCoreLoopCount;
        perLoopRowCount_ = tilingData_->lastCorePerLoopRowCount;
        lastLoopRowCount_ = tilingData_->lastCoreLastLoopRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
        curCoreLoopCount_ = tilingData_->perCoreLoopCount;
        perLoopRowCount_ = tilingData_->perCorePerLoopRowCount;
        lastLoopRowCount_ = tilingData_->perCoreLastLoopRowCount;
    }
    curentRowBaseIndex_ = blockIdx_ * perCoreRowCount_;
    expertCount_ = tilingData_->expertCount;
    k_ = tilingData_->k;
    kAlign_ = tilingData_->kAlign;
    expertCountAlign_ = tilingData_->expertCountAlign;
    rows_ = tilingData_->rows;
    padNegInfCount_ = tilingData_->padNegInfCount;
    sortRepeatTimes_ = tilingData_->sortRepeatTimes;

    // init input gm buf
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    finishedGm_.SetGlobalBuffer((__gm__ bool *)finished + perCoreRowCount_ * blockIdx_, expertCount_);

    // init output gm buf
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    rowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)rowIdx + perCoreRowCount_ * k_ * blockIdx_, k_);

    // init copy in que
    pipe_->InitBuffer(xInQueue_, BUFFER_NUM, expertCountAlign_ * sizeof(T) * perLoopRowCount_);
    pipe_->InitBuffer(finishedInQueue_, BUFFER_NUM, Ops::Base::GetUbBlockSize() * perLoopRowCount_);

    // init copy out que
    pipe_->InitBuffer(yOutQueue_, BUFFER_NUM, kAlign_ * sizeof(float) * perLoopRowCount_);
    pipe_->InitBuffer(expertIdxOutQueue_, BUFFER_NUM, kAlign_ * sizeof(int32_t) * perLoopRowCount_);
    pipe_->InitBuffer(rowIdxOutQueue_, BUFFER_NUM, kAlign_ * sizeof(int32_t) * perLoopRowCount_);

    // init buf que
    pipe_->InitBuffer(expertIdxBuf_, expertCountAlign_ * sizeof(float) * perLoopRowCount_);
    pipe_->InitBuffer(rowIdxBaseBuf_, expertCountAlign_ * sizeof(int32_t) * perLoopRowCount_);
    pipe_->InitBuffer(softmaxBuf_, expertCountAlign_ * sizeof(float) * perLoopRowCount_);
    pipe_->InitBuffer(calcTmpBuf_, (expertCountAlign_ * perLoopRowCount_ * KEY_VALUE_FACTOR +
                                    expertCountAlign_ * CONSTANT_TWO * KEY_VALUE_FACTOR) *
                                       sizeof(float));
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::InitIndices()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> rowIdxBaseTensor = rowIdxBaseBuf_.Get<int32_t>();
    ArithProgression(rowIdxBaseTensor, 0, static_cast<int32_t>(rows_), k_);
    uint16_t rowLoops = static_cast<uint32_t>(perLoopRowCount_);
    __VEC_SCOPE__
    {
        uint32_t repeatCount = B32_VF_COUNT;
        uint16_t expertCountloops = (expertCount_ + repeatCount - 1) / repeatCount;
        uint32_t precessExpert = expertCount_;
        AscendC::MicroAPI::RegTensor<int32_t> vreg0;
        __ubuf__ int32_t *expertIdxTensorAddr = (__ubuf__ int32_t *)expertIdxTensor.GetPhyAddr();
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < expertCountloops; i++) {
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessExpert);
            uint16_t expertCountloopsOffset = i * repeatCount;
            AscendC::MicroAPI::Arange(vreg0, expertCountloopsOffset);
            for (uint16_t j = 0; j < rowLoops; j++) {
                AscendC::MicroAPI::StoreAlign(expertIdxTensorAddr + (j * expertCountAlign_) + expertCountloopsOffset,
                                              vreg0, mask);
            }
        }
    }

    __VEC_SCOPE__
    {
        uint32_t repeatCount = B32_VF_COUNT;
        uint16_t kLoops = (k_ + repeatCount - 1) / repeatCount;
        uint32_t precessK = k_;
        AscendC::MicroAPI::RegTensor<int32_t> vreg0, vreg1;
        __ubuf__ int32_t *rowIdxBaseTensorAddr = (__ubuf__ int32_t *)rowIdxBaseTensor.GetPhyAddr();
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < kLoops; i++) {
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessK);
            uint16_t kLoopsOffset = i * repeatCount;
            AscendC::MicroAPI::LoadAlign(vreg0, rowIdxBaseTensorAddr + kLoopsOffset);
            for (uint16_t j = 1; j < rowLoops; j++) {
                AscendC::MicroAPI::Adds(vreg1, vreg0, j, mask);
                AscendC::MicroAPI::StoreAlign(rowIdxBaseTensorAddr + (j * expertCountAlign_) + kLoopsOffset, vreg1,
                                              mask);
            }
        }
    }
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::CopyInX(int64_t loop,
                                                                                                      int64_t rowCount)
{
    LocalTensor<T> xTensor = xInQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = rowCount;
    dataCopyParams.blockLen = expertCount_ * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (expertCountAlign_ - expertCount_) * sizeof(T) / BLOCK_BYTES;
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xTensor, xGm_[loop * perLoopRowCount_ * expertCount_], dataCopyParams, dataCopyPadParams);
    if constexpr (hasFinished) {
        LocalTensor<bool> finishedTensor = finishedInQueue_.AllocTensor<bool>();
        DataCopyPadExtParams finishedDataCopyPadParams{false, 0, 0, false};
        dataCopyParams.blockCount = rowCount;
        dataCopyParams.blockLen = sizeof(bool);
        dataCopyParams.dstStride = 0;
        DataCopyPad(finishedTensor, finishedGm_[loop * perLoopRowCount_], dataCopyParams, finishedDataCopyPadParams);
        finishedInQueue_.EnQue<bool>(finishedTensor);
    }
    xInQueue_.EnQue<T>(xTensor);
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::ComputeSoftmax(
    int64_t rowCount)
{
    LocalTensor<T> xTensor = xInQueue_.DeQue<T>();
    LocalTensor<float> softmaxTensor = softmaxBuf_.Get<float>();
    LocalTensor<float> xTensorFp32 = softmaxBuf_.Get<float>();
    LocalTensor<float> reduceValueTensor = calcTmpBuf_.Get<float>();
    LocalTensor<float> tmpTensor =
        calcTmpBuf_.Get<float>()[(rowCount + B32_BLOCK_COUNT - 1) / B32_BLOCK_COUNT * B32_BLOCK_COUNT];

    Duplicate(softmaxTensor, 0.0f, rowCount * expertCountAlign_);
    uint16_t rowLoops = static_cast<uint32_t>(rowCount);
    __VEC_SCOPE__
    {
        uint32_t repeatCount = B32_VF_COUNT;
        uint16_t expertCountLoops = (expertCount_ + repeatCount - 1) / repeatCount;

        __ubuf__ T *xTensorAddr = (__ubuf__ T *)xTensor.GetPhyAddr();
        __ubuf__ float *xTensorFp32Addr = (__ubuf__ float *)xTensorFp32.GetPhyAddr();

        AscendC::MicroAPI::RegTensor<float> reduceVreg, reduceMidRreg, dupVreg, vreg0;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < rowLoops; i++) {
            uint32_t remain = expertCount_;
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(remain);
            uint16_t rowLoopsOffset = i * expertCountAlign_;
            uint32_t offset = rowLoopsOffset;
            ops::LoadOneTensorForDtypeT<T>(xTensorAddr, reduceMidRreg, mask, offset);
            for (uint16_t j = 1; j < expertCountLoops; j++) {
                mask = AscendC::MicroAPI::UpdateMask<int32_t>(remain);
                offset = rowLoopsOffset + j * repeatCount;
                ops::LoadOneTensorForDtypeT<T>(xTensorAddr, vreg0, mask, offset);
                AscendC::MicroAPI::Max<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reduceMidRreg, reduceMidRreg,
                                                                                         vreg0, mask);
            }
            remain = expertCount_;
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(remain);
            Reg::Reduce<Reg::ReduceType::MAX>(reduceVreg, reduceMidRreg, mask);
            AscendC::MicroAPI::Duplicate(dupVreg, reduceVreg, mask);
            for (uint16_t j = 0; j < expertCountLoops; j++) {
                offset = rowLoopsOffset + j * repeatCount;
                ops::LoadOneTensorForDtypeT<T>(xTensorAddr, vreg0, mask, offset);
                AscendC::MicroAPI::Sub(vreg0, vreg0, dupVreg, mask);
                AscendC::MicroAPI::Exp(vreg0, vreg0, mask);
                AscendC::MicroAPI::StoreAlign(xTensorFp32Addr + offset, vreg0, mask);
                mask = AscendC::MicroAPI::UpdateMask<int32_t>(remain);
            }
        }
    }

    if constexpr (needPadNegInf) {
        int duplicateIndex = expertCount_ - padNegInfCount_;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << padNegInfCount_;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[CONSTANT_TWO] = {mask0, 0};
        Duplicate(softmaxTensor[duplicateIndex], 0.0f, mask, rowCount, 1,
                  expertCountAlign_ * sizeof(float) / BLOCK_BYTES);
    }

    uint32_t shape[] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(expertCountAlign_)};
    ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(reduceValueTensor, softmaxTensor,
                                                          tmpTensor.template ReinterpretCast<uint8_t>(), shape, true);
    Brcb(tmpTensor, reduceValueTensor, (rowCount + B32_BLOCK_COUNT - 1) / B32_BLOCK_COUNT, {1, REPEAT_BLOCKS});

    __VEC_SCOPE__
    {
        uint32_t repeatCount = B32_VF_COUNT;
        uint16_t expertCountLoops = (expertCount_ + repeatCount - 1) / repeatCount;
        __ubuf__ float *softmaxTensorAddr = (__ubuf__ float *)softmaxTensor.GetPhyAddr();
        __ubuf__ float *sumTensorAddr = (__ubuf__ float *)tmpTensor.GetPhyAddr();
        AscendC::MicroAPI::RegTensor<float> sumVreg, vreg0;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < rowLoops; i++) {
            uint32_t precessExpert = expertCount_;
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessExpert);
            AscendC::MicroAPI::LoadAlign(sumVreg, sumTensorAddr + i * B32_BLOCK_COUNT);
            AscendC::MicroAPI::Duplicate(sumVreg, sumVreg, mask);

            uint16_t rowLoopsOffset = i * expertCountAlign_;
            for (uint16_t j = 0; j < expertCountLoops; j++) {
                uint16_t expertCountLoopsOffset = j * repeatCount;
                AscendC::MicroAPI::LoadAlign(vreg0, softmaxTensorAddr + rowLoopsOffset + expertCountLoopsOffset);
                AscendC::MicroAPI::Div(vreg0, vreg0, sumVreg, mask);
                AscendC::MicroAPI::StoreAlign(softmaxTensorAddr + rowLoopsOffset + expertCountLoopsOffset, vreg0, mask);
                mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessExpert);
            }
        }
    }
    xInQueue_.FreeTensor<T>(xTensor);
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::HandleOneRepeatSortNum(
    int64_t rowCount, const LocalTensor<float> &softmaxTensor, const LocalTensor<float> &sortedTensor,
    LocalTensor<uint32_t> &expertIdxTensor)
{
    // 如果只选top1，无需排序，用max求最大值作为top1
    if (k_ == 1 && expertCount_ <= B32_BLOCK_COUNT) {
        __VEC_SCOPE__
        {
            __ubuf__ float *softmaxTensorAddr = (__ubuf__ float *)softmaxTensor.GetPhyAddr();
            __ubuf__ float *sortedTensorAddr = (__ubuf__ float *)sortedTensor.GetPhyAddr();
            AscendC::MicroAPI::RegTensor<float> valueAndIndexReg;
            AscendC::MicroAPI::MaskReg maskForValueAndIndex =
                AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::VL2>();
            AscendC::MicroAPI::MaskReg maskForExpertCount;
            uint32_t uint32ExpertCount_ = static_cast<uint32_t>(expertCount_);
            int64_t kvExpertCountAlign_ = expertCountAlign_ * KEY_VALUE_FACTOR;

            for (uint16_t i = 0; i < static_cast<uint16_t>(rowCount); i++) {
                AscendC::MicroAPI::LoadAlign(valueAndIndexReg, softmaxTensorAddr + i * expertCountAlign_);
                uint32_t expertCountForMask = uint32ExpertCount_;
                maskForExpertCount = AscendC::MicroAPI::UpdateMask<uint32_t>(expertCountForMask);
                Reg::Reduce<Reg::ReduceType::MAX>(valueAndIndexReg, valueAndIndexReg, maskForExpertCount);
                StoreAlign(sortedTensorAddr + kvExpertCountAlign_ * i, valueAndIndexReg, maskForValueAndIndex);
            }
        }
    } else {
        Sort32(sortedTensor, softmaxTensor, expertIdxTensor, rowCount);
    }
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::ComputeTopK(
    int64_t rowCount)
{
    LocalTensor<float> softmaxTensor = softmaxBuf_.Get<float>();
    LocalTensor<float> sortedTensor = calcTmpBuf_.Get<float>();
    LocalTensor<float> sortTmpTensor = calcTmpBuf_.Get<float>()[rowCount * expertCountAlign_ * KEY_VALUE_FACTOR];
    LocalTensor<uint32_t> expertIdxTensor = expertIdxBuf_.Get<uint32_t>();
    LocalTensor<bool> finishedTensor;
    LocalTensor<T> yOutTensor = yOutQueue_.AllocTensor<T>();
    LocalTensor<int32_t> expertIdxOutTensor = expertIdxOutQueue_.AllocTensor<int32_t>();

    if (expertCountAlign_ == ONE_REPEAT_SORT_NUM) {
        HandleOneRepeatSortNum(rowCount, softmaxTensor, sortedTensor, expertIdxTensor);
    } else {
        for (int i = 0; i < rowCount; i++) {
            int64_t rowCountOffset = expertCountAlign_ * i;
            Sort<float, true>(sortedTensor[rowCountOffset * KEY_VALUE_FACTOR], softmaxTensor[rowCountOffset],
                              expertIdxTensor[rowCountOffset], sortTmpTensor, sortRepeatTimes_);
        }
    }
    if constexpr (hasFinished) {
        finishedTensor = finishedInQueue_.DeQue<bool>();
    }
    uint16_t rowLoops = static_cast<uint32_t>(rowCount);
    __VEC_SCOPE__
    {
        uint32_t repeatCount = B32_VF_COUNT;
        uint16_t loopK = (k_ + repeatCount - 1) / repeatCount;
        uint16_t loopEnd = loopK - 1;
        uint16_t lastLoopKCount = k_ % repeatCount == 0 ? repeatCount : k_ % repeatCount;
        AscendC::MicroAPI::RegTensor<float> valueVreg;
        AscendC::MicroAPI::RegTensor<int32_t> indexVreg;
        AscendC::MicroAPI::RegTensor<int8_t> finishedB8Vreg;
        AscendC::MicroAPI::RegTensor<int32_t> finishedB32Vreg;
        AscendC::MicroAPI::UnalignRegForStore u0, u1;
        AscendC::MicroAPI::MaskReg mask, finishedMask;
        finishedMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        __ubuf__ int32_t *sortedTensorAddr = (__ubuf__ int32_t *)sortedTensor.GetPhyAddr();
        __ubuf__ T *yOutTensorAddr = (__ubuf__ T *)yOutTensor.GetPhyAddr();
        __ubuf__ int32_t *expertIdxOutTensorAddr = (__ubuf__ int32_t *)expertIdxOutTensor.GetPhyAddr();
        __ubuf__ int8_t *finishedTensorAddr = (__ubuf__ int8_t *)finishedTensor.GetPhyAddr();
        for (uint16_t i = 0; i < rowLoops; i++) {
            uint32_t precessK = k_;
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessK);
            if constexpr (hasFinished) {
                AscendC::MicroAPI::LoadAlign(finishedB8Vreg, finishedTensorAddr + i * B8_BLOCK_COUNT);
                AscendC::MicroAPI::Cast<int32_t, int8_t, castTrait>(finishedB32Vreg, finishedB8Vreg, finishedMask);
                AscendC::MicroAPI::Duplicate(finishedB32Vreg, finishedB32Vreg, mask);
                AscendC::MicroAPI::Muls(finishedB32Vreg, finishedB32Vreg, static_cast<int32_t>(expertCount_), mask);
            }
            uint16_t rowLoopsOffset = i * expertCountAlign_ * KEY_VALUE_FACTOR;
            for (uint16_t j = 0; j < loopEnd; j++) {
                AscendC::MicroAPI::LoadAlign<int32_t, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (AscendC::MicroAPI::RegTensor<int32_t> &)valueVreg, indexVreg,
                    sortedTensorAddr + rowLoopsOffset + j * repeatCount * KEY_VALUE_FACTOR);
                if constexpr (!IsSameType<T, float>::value) {
                    ops::StoreUnAlignOneTensor<T>(yOutTensorAddr, valueVreg, u0, mask, repeatCount);
                } else {
                    AscendC::MicroAPI::StoreUnAlign(yOutTensorAddr, valueVreg, u0, repeatCount);
                }
                AscendC::MicroAPI::StoreUnAlignPost(yOutTensorAddr, u0, 0);
                if constexpr (hasFinished) {
                    AscendC::MicroAPI::Max(indexVreg, indexVreg, finishedB32Vreg, mask);
                }
                AscendC::MicroAPI::StoreUnAlign(expertIdxOutTensorAddr, indexVreg, u1, repeatCount);
                AscendC::MicroAPI::StoreUnAlignPost(expertIdxOutTensorAddr, u1, 0);
                mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessK);
            }
            AscendC::MicroAPI::LoadAlign<int32_t, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (AscendC::MicroAPI::RegTensor<int32_t> &)valueVreg, indexVreg,
                sortedTensorAddr + rowLoopsOffset + (loopEnd)*repeatCount * KEY_VALUE_FACTOR);
            if constexpr (!IsSameType<T, float>::value) {
                ops::StoreUnAlignOneTensor<T>(yOutTensorAddr, valueVreg, u0, mask, lastLoopKCount);
            } else {
                AscendC::MicroAPI::StoreUnAlign(yOutTensorAddr, valueVreg, u0, lastLoopKCount);
            }
            AscendC::MicroAPI::StoreUnAlignPost(yOutTensorAddr, u0, 0);
            if constexpr (hasFinished) {
                AscendC::MicroAPI::Max(indexVreg, indexVreg, finishedB32Vreg, mask);
            }
            AscendC::MicroAPI::StoreUnAlign(expertIdxOutTensorAddr, indexVreg, u1, lastLoopKCount);
            AscendC::MicroAPI::StoreUnAlignPost(expertIdxOutTensorAddr, u1, 0);
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessK);
        }
    }
    if constexpr (hasFinished) {
        finishedInQueue_.FreeTensor(finishedTensor);
    }
    yOutQueue_.EnQue<T>(yOutTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxOutTensor);
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::ComputeRowIdx(
    int64_t loop, int64_t rowCount)
{
    LocalTensor<int32_t> rowIdxOutTensor = rowIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> rowIdxBaseTensor = rowIdxBaseBuf_.Get<int32_t>();
    int32_t indexBase = curentRowBaseIndex_ + loop * perLoopRowCount_;
    int32_t kAlign = kAlign_;
    uint16_t rowLoops = static_cast<uint32_t>(rowCount);
    __VEC_SCOPE__
    {
        uint32_t repeatCount = B32_VF_COUNT;
        uint16_t loopK = (k_ + repeatCount - 1) / repeatCount;
        uint32_t precessK = k_;
        AscendC::MicroAPI::RegTensor<int32_t> vreg0, vreg1;
        __ubuf__ int32_t *rowIdxBaseTensorAddr = (__ubuf__ int32_t *)rowIdxBaseTensor.GetPhyAddr();
        __ubuf__ int32_t *rowIdxOutTensorAddr = (__ubuf__ int32_t *)rowIdxOutTensor.GetPhyAddr();
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < loopK; i++) {
            mask = AscendC::MicroAPI::UpdateMask<int32_t>(precessK);
            uint16_t loopKOffset = i * repeatCount;
            AscendC::MicroAPI::LoadAlign(vreg0, rowIdxBaseTensorAddr + loopKOffset);
            for (uint16_t j = 0; j < rowLoops; j++) {
                AscendC::MicroAPI::Adds(vreg1, vreg0, indexBase + j, mask);
                AscendC::MicroAPI::StoreAlign(rowIdxOutTensorAddr + (j * kAlign) + loopKOffset, vreg1, mask);
            }
        }
    }
    rowIdxOutQueue_.EnQue<int32_t>(rowIdxOutTensor);
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::CopyOutRowIdx(
    int64_t loop, int64_t rowCount)
{
    LocalTensor<int32_t> rowIdxOutTensor = rowIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(rowCount), static_cast<uint32_t>(k_ * sizeof(int32_t)),
                                     static_cast<uint32_t>((kAlign_ - k_) * sizeof(int32_t) / BLOCK_BYTES), 0, 0};
    DataCopyPad(rowIdxGm_[loop * perLoopRowCount_ * k_], rowIdxOutTensor, dataCopyParams);
    rowIdxOutQueue_.FreeTensor(rowIdxOutTensor);
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::CopyYExpertIdxOut(
    int64_t loop, int64_t rowCount)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOutTensor = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyXParams{1, static_cast<uint32_t>(rowCount * k_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams dataCopyIdxParams{1, static_cast<uint32_t>(rowCount * k_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(yGm_[loop * perLoopRowCount_ * k_], yOutTensor, dataCopyXParams);
    DataCopyPad(expertIdxGm_[loop * perLoopRowCount_ * k_], expertIdxOutTensor, dataCopyIdxParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOutTensor);
}

template <typename T, bool hasFinished, bool needPadNegInf>
__aicore__ inline void MoeGatingTopKSoftmaxFullloadGenerlized<T, hasFinished, needPadNegInf>::Process()
{
    InitIndices();
    for (int loop = 0; loop < curCoreLoopCount_; loop++) {
        int64_t rowCount = loop == (curCoreLoopCount_ - 1) ? lastLoopRowCount_ : perLoopRowCount_;
        ComputeRowIdx(loop, rowCount);
        CopyOutRowIdx(loop, rowCount);
        CopyInX(loop, rowCount);
        ComputeSoftmax(rowCount);
        ComputeTopK(rowCount);
        CopyYExpertIdxOut(loop, rowCount);
    }
}
} // namespace MoeGatingTopKSoftmax
#endif // MOE_GATING_TOP_K_SOFTMAX_FULLLOAD_GENERALIZED_REGBASE_H
