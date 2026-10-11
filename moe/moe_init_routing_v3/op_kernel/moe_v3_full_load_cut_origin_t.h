/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_full_load_cut_origin_t.h
 * \brief FullLoad counting sort template for decode (small T) path.
 *        Replaces O(N log N) merge-sort with O(N) counting sort when
 *        actual expert ratio is low (e.g. 16/256 = 6.25%).
 */
#ifndef MOE_V3_FULL_LOAD_CUT_ORIGIN_T_H
#define MOE_V3_FULL_LOAD_CUT_ORIGIN_T_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3FullLoadCutOriginT {
public:
    __aicore__ inline MoeV3FullLoadCutOriginT(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokens, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadExpertIdxAndScale();
    __aicore__ inline void LoadXBackground();
    __aicore__ inline void FilterAndCount();
    __aicore__ inline void FilterAndCountVector();
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();
    __aicore__ inline void WaitXLoad();
    __aicore__ inline void GatherAndWrite();

private:
    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;
    static constexpr int64_t VECTOR_FILTER_THRESHOLD = 64;

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> buf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int64_t> expertTokensGm_;
    GlobalTensor<float> expandedScaleGm_;
    GlobalTensor<int32_t> workspaceGm_;

    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t cols_;
    int64_t totalLength_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t rowIdxType_;
    int64_t isInputScale_;
    int64_t filterNeedCoreNum_;
    int64_t coreNum_;
    int64_t expertTokensNumFlag_;
    int64_t activeNum_;

    int64_t coreTokenStart_;
    int64_t coreTokenEnd_;
    int64_t coreTokenNum_;
    int64_t coreFlatStart_;
    int64_t coreEntries_;
    int64_t expertCountStride_;
    int64_t colsAligned_;
    int64_t scaleSlotSize_;
    int64_t maxFilteredCount_;
    int64_t filteredCount_;

    // Vector filter derived constants
    int64_t entriesAligned_;
    int64_t maskBytes_;

    // UB region offsets (byte offsets from buf_ start)
    int64_t xLocalOffset_;
    int64_t expertIdxLocalOffset_;
    int64_t scaleLocalOffset_;
    int64_t expertCountLocalOffset_;
    int64_t allCoreExpertCountLocalOffset_;
    int64_t expertTokensLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t filteredPairsLocalOffset_;

    // Vector filter additional UB offsets
    int64_t expertIdxFp32LocalOffset_;
    int64_t compareMask0Offset_;
    int64_t compareMask1Offset_;
    int64_t gatherMaskOffset_;
    int64_t gatheredExpertLocalOffset_;
    int64_t flatIdxBufferLocalOffset_;
    int64_t gatheredIdxLocalOffset_;

    int64_t totalBufSize_;
};

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX,
                                                        GM_ADDR expandedRowIdx, GM_ADDR expertTokens,
                                                        GM_ADDR expandedScale, GM_ADDR workspace,
                                                        const MoeInitRoutingV3TilingData* tiling, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    n_ = tiling->n;
    k_ = tiling->k;
    cols_ = tiling->cols;
    totalLength_ = n_ * k_;
    expertStart_ = tiling->expertStart;
    expertEnd_ = tiling->expertEnd;
    actualExpertNum_ = tiling->actualExpertNum;
    rowIdxType_ = tiling->rowIdxType;
    isInputScale_ = tiling->isInputScale;
    filterNeedCoreNum_ = tiling->gatherOutComputeParamsOp.needCoreNum;
    coreNum_ = tiling->coreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    activeNum_ = tiling->activeNum;

    // Per-core token distribution
    int64_t filterPerCoreTokens = tiling->gatherOutComputeParamsOp.perCoreIndicesElements;
    coreTokenStart_ = blockIdx_ * filterPerCoreTokens;
    coreTokenEnd_ = Min(coreTokenStart_ + filterPerCoreTokens, n_);
    if (blockIdx_ == filterNeedCoreNum_ - 1) {
        coreTokenEnd_ = n_;
    }
    coreTokenNum_ = coreTokenEnd_ - coreTokenStart_;
    coreFlatStart_ = coreTokenStart_ * k_;
    coreEntries_ = coreTokenNum_ * k_;

    expertCountStride_ = OpsBaseCeilAlign(actualExpertNum_, static_cast<int64_t>(8));
    colsAligned_ =
        OpsBaseCeilAlign(cols_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES) / static_cast<int64_t>(sizeof(T));
    scaleSlotSize_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(float)); // 8
    maxFilteredCount_ = coreEntries_;
    filteredCount_ = 0;

    // Vector filter decision
    entriesAligned_ = Ceil(coreEntries_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    maskBytes_ = OpsBaseCeilAlign(Ceil(entriesAligned_, static_cast<int64_t>(8)), BLOCK_BYTES);

    // Setup GlobalTensors
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
    expandedXGm_.SetGlobalBuffer((__gm__ T*)expandedX);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokens);
    }
    if (isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float*)expandedScale);
    }

    // Modification 1: Use InitGlobalMemory for GATHER mode pre-fill -1
    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens * k_ * blockIdx_];
            InitGlobalMemory(expandedRowIdxGmTmp, coreEntries_, -1);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        SyncAll();
    }

    // Compute UB layout offsets
    int64_t offset = 0;

    // xLocal: coreTokenNum_ * colsAligned_ elements of T
    xLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(coreTokenNum_ * colsAligned_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES);

    // expertIdxLocal: coreEntries_ elements of int32
    expertIdxLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(coreEntries_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // scaleLocal: coreTokenNum_ * scaleSlotSize_ elements of float (32B-aligned per token)
    scaleLocalOffset_ = offset;
    if (isInputScale_) {
        offset += OpsBaseCeilAlign(coreTokenNum_ * scaleSlotSize_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES);
    }

    // expertCountLocal: expertCountStride_ elements of int32
    expertCountLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // allCoreExpertCountLocal: filterNeedCoreNum_ * expertCountStride_ elements of int32
    allCoreExpertCountLocalOffset_ = offset;
    offset +=
        OpsBaseCeilAlign(filterNeedCoreNum_ * expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // expertTokensLocal: actualExpertNum_ elements of int64
    expertTokensLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(actualExpertNum_ * static_cast<int64_t>(sizeof(int64_t)), BLOCK_BYTES);

    // prefixSumLocal: expertCountStride_ elements of int32
    prefixSumLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(expertCountStride_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // filteredPairsLocal: maxFilteredCount_ * 2 elements of int32
    filteredPairsLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(maxFilteredCount_ * 2 * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // Vector filter additional buffers
    // expertIdxFp32Local: entriesAligned_ elements of float (independent buffer for CompareScalar)
    expertIdxFp32LocalOffset_ = offset;
    offset += OpsBaseCeilAlign(entriesAligned_ * static_cast<int64_t>(sizeof(float)), BLOCK_BYTES);

    // compareMask0, compareMask1, gatherMask
    compareMask0Offset_ = offset;
    offset += maskBytes_;
    compareMask1Offset_ = offset;
    offset += maskBytes_;
    gatherMaskOffset_ = offset;
    offset += maskBytes_;

    // gatheredExpertLocal: entriesAligned_ elements of int32
    gatheredExpertLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(entriesAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // flatIdxBufferLocal: entriesAligned_ elements of int32
    flatIdxBufferLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(entriesAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    // gatheredIdxLocal: entriesAligned_ elements of int32
    gatheredIdxLocalOffset_ = offset;
    offset += OpsBaseCeilAlign(entriesAligned_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);

    totalBufSize_ = offset;
    pipe_->InitBuffer(buf_, totalBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // ======= Phase A: Load + Filter + Count (all cores parallel) =======
    LoadExpertIdxAndScale();
    LoadXBackground();
    FilterAndCount();
    WriteExpertCountToWorkspace();

    SyncAll();

    // ======= Phase B: Global reduce + offset computation (all cores parallel) =======
    ComputeGlobalOffset();
    // Modification 5: Only write expert tokens when flag is set
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }

    // ======= Phase C: Wait x load + data gather (all cores parallel) =======
    WaitXLoad();
    GatherAndWrite();
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::LoadExpertIdxAndScale()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);

    if (isInputScale_) {
        LocalTensor<float> scaleLocal = buf_.Get<float>()[scaleLocalOffset_ / sizeof(float)];
        DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(sizeof(float)), 0,
                                          0, 0};
        DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
        DataCopyPad(scaleLocal, scaleGm_[coreTokenStart_], scaleCopyParams, scalePadParams);
    }

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::LoadXBackground()
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];

    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    DataCopyExtParams copyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(cols_ * sizeof(T)), 0,
                                 static_cast<uint32_t>((colsAligned_ - cols_) * sizeof(T)), 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[coreTokenStart_ * cols_], copyParams, padParams);
    // Do NOT wait here -- x loads in background, waited in WaitXLoad()
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    // Modification 2: Use Duplicate for expertCountLocal initialization
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Modification 3: Branch based on coreEntries_ size
    FilterAndCountVector();
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::FilterAndCountVector()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    // Cast int32 -> fp32 into separate buffer (only needed for CompareScalar)
    LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32LocalOffset_ / sizeof(float)];
    Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, entriesAligned_);
    PipeBarrier<PIPE_V>();

    // CompareScalar + And for range [expertStart_, expertEnd_)
    LocalTensor<uint8_t> compareMask0 = buf_.Get<uint8_t>()[compareMask0Offset_];
    LocalTensor<uint8_t> compareMask1 = buf_.Get<uint8_t>()[compareMask1Offset_];
    LocalTensor<uint8_t> gatherMaskLocal = buf_.Get<uint8_t>()[gatherMaskOffset_];

    CompareScalar(compareMask0, expertIdxFp32Local, static_cast<float>(expertStart_), CMPMODE::GE, entriesAligned_);
    PipeBarrier<PIPE_V>();
    CompareScalar(compareMask1, expertIdxFp32Local, static_cast<float>(expertEnd_), CMPMODE::LT, entriesAligned_);
    PipeBarrier<PIPE_V>();
    And(gatherMaskLocal.ReinterpretCast<uint16_t>(), compareMask0.ReinterpretCast<uint16_t>(),
        compareMask1.ReinterpretCast<uint16_t>(),
        Ceil(entriesAligned_, MASK_STRIDE) * MASK_STRIDE / DST_REP_STRIDE / 2);
    PipeBarrier<PIPE_V>();

    // GatherMask int32 directly on preserved expertIdxLocal
    LocalTensor<int32_t> gatheredExpertLocal = buf_.Get<int32_t>()[gatheredExpertLocalOffset_ / sizeof(int32_t)];
    uint64_t rsvdCnt = 0;
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = DST_REP_STRIDE;
    gatherMaskParams.src1RepeatStride = DST_REP_STRIDE;
    GatherMask(gatheredExpertLocal, expertIdxLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(coreEntries_), gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    int64_t filteredInBatch = static_cast<int64_t>(rsvdCnt);

    // ArithProgression (int32) + GatherMask (int32) for flat indices
    LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferLocalOffset_ / sizeof(int32_t)];
    ArithProgression<int32_t>(flatIdxBufferLocal, static_cast<int32_t>(0), 1, static_cast<int32_t>(coreEntries_));
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxLocalOffset_ / sizeof(int32_t)];
    uint64_t idxRsvdCnt = 0;
    GatherMask(gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(coreEntries_), gatherMaskParams, idxRsvdCnt);
    PipeBarrier<PIPE_V>();

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Scalar: build pairs + accumulate expert count
    filteredCount_ = 0;
    for (int64_t j = 0; j < filteredInBatch; j++) {
        int32_t expertId = gatheredExpertLocal.GetValue(j);
        int32_t localFlatIdx = gatheredIdxLocal.GetValue(j);
        int32_t expertOffset = expertId - static_cast<int32_t>(expertStart_);

        int32_t curCount = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, curCount + 1);

        filteredPairsLocal.SetValue(filteredCount_ * 2, localFlatIdx);
        filteredPairsLocal.SetValue(filteredCount_ * 2 + 1, expertOffset);
        filteredCount_++;
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(workspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::ComputeGlobalOffset()
{
    // Invalidate cache to see workspace writes from all cores
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm_);

    LocalTensor<int32_t> allCoreExpertCountLocal =
        buf_.Get<int32_t>()[allCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    // Load all cores' expert counts from workspace
    int64_t totalWsElements = filterNeedCoreNum_ * expertCountStride_;
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalWsElements * sizeof(int32_t)), 0,
                                 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(allCoreExpertCountLocal, workspaceGm_[0], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    // Reuse prefixSumLocal as totalCountLocal first: vectorized cross-core sum
    LocalTensor<int32_t> totalCountLocal = prefixSumLocal;
    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < filterNeedCoreNum_; c++) {
        Add(totalCountLocal, totalCountLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Read totalCount values into scalar array before reusing prefixSumLocal
    int32_t totalForExpertArr[128]; // actualExpertNum_ <= 128
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        totalForExpertArr[e] = totalCountLocal.GetValue(e);
    }

    // Prefix sum for this core: sum of counts from cores [0, blockIdx_)
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < blockIdx_; c++) {
        Add(prefixSumLocal, prefixSumLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
        PipeBarrier<PIPE_V>();
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // Scalar computation: per-expert global start offset for this core
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    int64_t cumulativeSum = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalForExpertArr[e];
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
            expertTokensLocal.SetValue(e, static_cast<int64_t>(totalForExpert));
        }

        // expertCountLocal[e] = global start position for this core's entries of expert e
        expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        cumulativeSum += totalForExpert;
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::WriteExpertTokens()
{
    if (blockIdx_ == 0) {
        LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
    }
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::WaitXLoad()
{
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
}

template <typename T>
__aicore__ inline void MoeV3FullLoadCutOriginT<T>::GatherAndWrite()
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
    LocalTensor<float> scaleLocal = buf_.Get<float>()[scaleLocalOffset_ / sizeof(float)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    // Modification 4: Use prefixSumLocal area as DMA scratch buffer for rowIdx writes
    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    for (int64_t fIdx = 0; fIdx < filteredCount_; fIdx++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);

        // Modification 4: Directly use and increment expertCountLocal
        int32_t newPos = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, newPos + 1);

        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
        int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);

        // Write rowIdx
        if (rowIdxType_ == GATHER) {
            idxBuf.SetValue(0, newPos);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
        } else {
            idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
        }

        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        if (newPos < activeNum_) {
            // Copy x row: UB -> GM
            DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * cols_], xLocal[tokenRow * colsAligned_],
                        xCopyParams);
            // Copy scale
            if (isInputScale_) {
                DataCopyPad(expandedScaleGm_[newPos], scaleLocal[tokenRow * scaleSlotSize_], scaleCopyParams);
            }
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_FULL_LOAD_CUT_ORIGIN_T_H
