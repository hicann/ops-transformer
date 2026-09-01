/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_counting_sort_full_load_unquantized.h
 * \brief A5 CountingSort FullLoad
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortFullLoadUnquantized {
public:
    __aicore__ inline MoeV3CountingSortFullLoadUnquantized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    // --- 公共逻辑（原模板基类） ---
    __aicore__ inline void InitCommon(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                      GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
    __aicore__ inline void PrefillAndSync();
    __aicore__ inline void ZeroOutExpandedDropPad(GlobalTensor<T> &expandedXGm);
    __aicore__ inline void ComputeCommonUbLayout();
    __aicore__ inline void LoadExpertIdx();
    __aicore__ inline void LoadXBackground(GlobalTensor<T> &xGm);
    __aicore__ inline void FilterAndCount();
    __aicore__ inline void FilterAndCountVector();
    __aicore__ inline void WriteExpertCountToWorkspace();
    __aicore__ inline void ComputeGlobalOffset();
    __aicore__ inline void WriteExpertTokens();
    __aicore__ inline void WaitXLoadCommon();
    // 聚合搬出（csAggrEnable=1 时；FullLoad 模板仅 dropless，故无 dropPad 分支）
    __aicore__ inline void BucketByExpert();
    __aicore__ inline void GatherAndWriteByExpert(LocalTensor<T> &xLocal, LocalTensor<float> &scaleLocal,
                                                  GlobalTensor<T> &expandedXGmRef,
                                                  GlobalTensor<float> &expandedScaleGmRef);

    // --- 非量化搬运 ---
    __aicore__ inline void LoadScale();
    __aicore__ inline void GatherAndWrite();
    __aicore__ inline void GatherOneRow(int64_t newPos, int64_t tokenRow, int64_t origFlatIdx);

    static constexpr int64_t DST_REP_STRIDE = 8;
    static constexpr int64_t MASK_STRIDE = 64;

    TPipe *pipe_;
    TBuf<TPosition::VECCALC> buf_;

    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<int64_t> expertTokensGm_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> expandedScaleGm_;

    int64_t quantMode_;

    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t cols_;
    int64_t totalLength_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t actualExpertNum_;
    int64_t expertNum_;
    int64_t rowIdxType_;
    int64_t isInputScale_;
    int64_t filterNeedCoreNum_;
    int64_t coreNum_;
    int64_t expertTokensNumFlag_;
    int64_t expertTokensNumType_;
    int64_t activeNum_;
    int64_t dropPadMode_;
    int64_t expertCapacity_;
    int64_t outputRows_;
    int64_t filterPerCoreTokens_;

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
    int64_t entriesAligned_;
    int64_t maskBytes_;
    int64_t expertTokensBufElems_; // expertTokensLocal 缓冲元素数（KEY_VALUE 需 2 倍）
    int64_t expertCountElements_;  // 实际写出 expertTokens 的元素数

    // 公共 UB region offsets (byte offsets from buf_ start)
    int64_t xLocalOffset_;
    int64_t expertIdxLocalOffset_;
    int64_t scaleLocalOffset_;
    int64_t expertCountLocalOffset_;
    int64_t allCoreExpertCountLocalOffset_;
    int64_t expertTokensLocalOffset_;
    int64_t prefixSumLocalOffset_;
    int64_t filteredPairsLocalOffset_;
    int64_t expertIdxFp32LocalOffset_;
    int64_t compareMask0Offset_;
    int64_t compareMask1Offset_;
    int64_t gatherMaskOffset_;
    int64_t gatheredExpertLocalOffset_;
    int64_t flatIdxBufferLocalOffset_;
    int64_t gatheredIdxLocalOffset_;
    int64_t commonBufSize_; // 公共区结尾 offset

    // 聚合搬出参数（从 tiling 读取）
    int64_t csAggrEnable_{0};
    int64_t aggrOutRows_{0};        // k：搬出聚合 UB 容纳行数
    int64_t aggrOutBufBytes_{0};    // 搬出聚合区字节数
    int64_t gatherOutBufOffset_{0}; // 搬出聚合区在 buf_ 中的字节偏移
    // 分桶区（按专家外循环聚合用）
    int64_t bucketBaseOffset_{0};      // 桶数据基址（存 localFlatIdx，按 expert 分段）
    int64_t bucketOffsetTblOffset_{0}; // 偏移表基址（actualExpertNum_ 个 int32，每桶起始偏移）
    int64_t bucketCountTblOffset_{0};  // 每桶元素数（actualExpertNum_ 个 int32）
};

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale,
                                                                     GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                                     GM_ADDR expertTokensCountOrCumsum,
                                                                     GM_ADDR expandedScale, GM_ADDR workspace,
                                                                     const MoeInitRoutingV3Arch35TilingData *tiling,
                                                                     TPipe *pipe)
{
    InitCommon(expertIdx, expandedRowIdx, expertTokensCountOrCumsum, workspace, tiling, pipe);

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    if (isInputScale_) {
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale);
    }

    // GATHER 预填 rowIdx=-1；dropPad 还需整块清零 expandedX（空槽零填充）
    PrefillAndSync();
    if (rowIdxType_ == GATHER && dropPadMode_ == DROP_PAD_MODE) {
        ZeroOutExpandedDropPad(expandedXGm_);
        if (isInputScale_) {
            // scale 透传空槽清零（golden 语义：空槽为 0）
            int64_t perCoreRows = Ceil(outputRows_, filterNeedCoreNum_);
            int64_t rowStart = blockIdx_ * perCoreRows;
            int64_t rowEnd = Min(rowStart + perCoreRows, outputRows_);
            if (rowStart < rowEnd) {
                GlobalTensor<float> scaleZeroSeg = expandedScaleGm_[rowStart];
                InitGlobalMemory(scaleZeroSeg, rowEnd - rowStart, static_cast<float>(0));
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }
        }
    }
    if (rowIdxType_ == GATHER) {
        SyncAll();
    }

    // 非量化无量化临时区，UB 尺寸即公共区尺寸
    pipe_->InitBuffer(buf_, commonBufSize_);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadScale()
{
    if (!isInputScale_) {
        return;
    }
    LocalTensor<float> scaleLocal = buf_.template Get<float>()[scaleLocalOffset_ / sizeof(float)];
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(sizeof(float)), 0, 0,
                                      0};
    DataCopyPadExtParams<float> scalePadParams{false, 0, 0, 0};
    DataCopyPad(scaleLocal, scaleGm_[coreTokenStart_], scaleCopyParams, scalePadParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherOneRow(int64_t newPos, int64_t tokenRow,
                                                                             int64_t origFlatIdx)
{
    LocalTensor<T> xLocal = buf_.template Get<T>()[xLocalOffset_ / sizeof(T)];
    LocalTensor<float> scaleLocal =
        isInputScale_ ? buf_.template Get<float>()[scaleLocalOffset_ / sizeof(float)] : LocalTensor<float>();

    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<int32_t> idxBuf = buf_.template Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    if (rowIdxType_ == GATHER) {
        idxBuf.SetValue(0, static_cast<int32_t>(newPos));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
    } else {
        idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
    }

    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    if (newPos < outputRows_) {
        DataCopyPad(expandedXGm_[static_cast<int64_t>(newPos) * cols_], xLocal[tokenRow * colsAligned_], xCopyParams);
        if (isInputScale_) {
            DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(expandedScaleGm_[newPos], scaleLocal[tokenRow * scaleSlotSize_], scaleCopyParams);
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherAndWrite()
{
    if (csAggrEnable_) {
        // 聚合搬出路径：按专家外循环 + k 行切批
        LocalTensor<T> xLocal = buf_.template Get<T>()[xLocalOffset_ / sizeof(T)];
        LocalTensor<float> scaleLocal =
            isInputScale_ ? buf_.template Get<float>()[scaleLocalOffset_ / sizeof(float)] : LocalTensor<float>();
        BucketByExpert();
        GatherAndWriteByExpert(xLocal, scaleLocal, expandedXGm_, expandedScaleGm_);
        return;
    }

    LocalTensor<int32_t> expertCountLocal = buf_.template Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.template Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    for (int64_t fIdx = 0; fIdx < filteredCount_; fIdx++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(fIdx * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(fIdx * 2 + 1);

        int32_t slot = expertCountLocal.GetValue(expertOffset);
        expertCountLocal.SetValue(expertOffset, slot + 1);

        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
        int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);

        int64_t newPos = static_cast<int64_t>(slot);
        GatherOneRow(newPos, tokenRow, origFlatIdx);
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::Process()
{
    if (blockIdx_ >= filterNeedCoreNum_) {
        SyncAll();
        return;
    }

    // Phase A: Load + Filter + Count（核间并行）
    LoadExpertIdx();
    LoadScale();
    LoadXBackground(xGm_);
    FilterAndCount();
    WriteExpertCountToWorkspace();

    SyncAll();

    // Phase B: 核间归约 + 偏移计算
    ComputeGlobalOffset();
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        WriteExpertTokens();
    }

    // Phase C: 等待 x 加载 + 数据搬运
    WaitXLoadCommon();
    GatherAndWrite();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::InitCommon(
    GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
    const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe)
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
    expertNum_ = tiling->expertNum;
    rowIdxType_ = tiling->rowIdxType;
    isInputScale_ = tiling->isInputScale;
    filterNeedCoreNum_ = tiling->countingSortParamsOp.filterNeedCoreNum;
    coreNum_ = tiling->coreNum;
    expertTokensNumFlag_ = tiling->expertTokensNumFlag;
    expertTokensNumType_ = tiling->expertTokensNumType;
    activeNum_ = tiling->activeNum;
    dropPadMode_ = tiling->dropPadMode;
    quantMode_ = tiling->quantMode;
    outputRows_ = activeNum_;
    csAggrEnable_ = tiling->countingSortParamsOp.csAggrEnable;
    aggrOutRows_ = tiling->countingSortParamsOp.csAggrOutRows;
    aggrOutBufBytes_ = tiling->countingSortParamsOp.csAggrOutBufBytes;

    filterPerCoreTokens_ = tiling->countingSortParamsOp.filterPerCoreTokens;
    coreTokenStart_ = blockIdx_ * filterPerCoreTokens_;
    coreTokenEnd_ = Min(coreTokenStart_ + filterPerCoreTokens_, n_);
    if (blockIdx_ == filterNeedCoreNum_ - 1) {
        coreTokenEnd_ = n_;
    }
    coreTokenNum_ = coreTokenEnd_ - coreTokenStart_;
    coreFlatStart_ = coreTokenStart_ * k_;
    coreEntries_ = coreTokenNum_ * k_;

    expertCountStride_ = AlignElem(actualExpertNum_, COUNTING_SORT_ONE_BLOCK_ELEMENT);
    colsAligned_ = Align(cols_, static_cast<int64_t>(sizeof(T)));
    scaleSlotSize_ = BLOCK_BYTES / static_cast<int64_t>(sizeof(float)); // 8
    maxFilteredCount_ = coreEntries_;
    filteredCount_ = 0;

    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        expertTokensBufElems_ = Min(actualExpertNum_ + 1, expertNum_) * EXPERT_ID_VALUE_NUM;
    } else {
        expertTokensBufElems_ = actualExpertNum_;
    }
    expertCountElements_ = expertTokensBufElems_;

    entriesAligned_ = Ceil(coreEntries_, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
    maskBytes_ = AlignBytes(Ceil(entriesAligned_, static_cast<int64_t>(8)), static_cast<int64_t>(1));

    // 公共 GM
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx);
    workspaceGm_.SetGlobalBuffer((__gm__ int32_t *)workspace);
    if (expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCountOrCumsum);
    }

    ComputeCommonUbLayout();
}

// GATHER 模式预填 rowIdx=-1 + dropPad 清零后 SyncAll。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::PrefillAndSync()
{
    if (rowIdxType_ == GATHER) {
        if (blockIdx_ < filterNeedCoreNum_) {
            GlobalTensor<int32_t> expandedRowIdxGmTmp = expandedRowIdxGm_[filterPerCoreTokens_ * k_ * blockIdx_];
            InitGlobalMemory(expandedRowIdxGmTmp, coreEntries_, static_cast<int32_t>(-1));
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }
}

// dropPad 空槽零填充：跨核均分 outputRows_，每核清零自己负责的 expandedX 行段。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::ZeroOutExpandedDropPad(GlobalTensor<T> &expandedXGm)
{
    int64_t perCoreRows = Ceil(outputRows_, filterNeedCoreNum_);
    int64_t rowStart = blockIdx_ * perCoreRows;
    int64_t rowEnd = Min(rowStart + perCoreRows, outputRows_);
    if (rowStart >= rowEnd) {
        return;
    }
    int64_t elemCnt = (rowEnd - rowStart) * cols_;
    GlobalTensor<T> zeroSeg = expandedXGm[rowStart * cols_];
    InitGlobalMemory(zeroSeg, elemCnt, static_cast<T>(0));
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::ComputeCommonUbLayout()
{
    int64_t offset = 0;
    // 聚合搬出：UB 最前独立预留 aggrOutBufBytes_ 作为 gatherOutBuf（不挤占 xLocal）
    if (csAggrEnable_ == 1 && aggrOutBufBytes_ > 0) {
        gatherOutBufOffset_ = offset;
        offset += AlignBytes(aggrOutBufBytes_, static_cast<int64_t>(sizeof(T)));
    }
    xLocalOffset_ = offset;
    int64_t xLocalBytes = AlignBytes(coreTokenNum_ * colsAligned_, static_cast<int64_t>(sizeof(T)));
    offset += xLocalBytes;

    expertIdxLocalOffset_ = offset;
    offset += AlignBytes(coreEntries_, static_cast<int64_t>(sizeof(int32_t)));

    scaleLocalOffset_ = offset;
    // per-token scale 全载缓冲仅非量化透传路径需要；动态量化的 (LE,H) smooth 走单行缓冲（子类）
    if (isInputScale_) {
        offset += AlignBytes(coreTokenNum_ * scaleSlotSize_, static_cast<int64_t>(sizeof(float)));
    }

    expertCountLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    allCoreExpertCountLocalOffset_ = offset;
    offset += AlignBytes(filterNeedCoreNum_ * expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    expertTokensLocalOffset_ = offset;
    offset += AlignBytes(expertTokensBufElems_, static_cast<int64_t>(sizeof(int64_t)));

    prefixSumLocalOffset_ = offset;
    offset += AlignBytes(expertCountStride_, static_cast<int64_t>(sizeof(int32_t)));

    filteredPairsLocalOffset_ = offset;
    offset += AlignBytes(maxFilteredCount_ * 2, static_cast<int64_t>(sizeof(int32_t)));

    expertIdxFp32LocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(float)));

    compareMask0Offset_ = offset;
    offset += maskBytes_;
    compareMask1Offset_ = offset;
    offset += maskBytes_;
    gatherMaskOffset_ = offset;
    offset += maskBytes_;

    gatheredExpertLocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(int32_t)));

    flatIdxBufferLocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(int32_t)));

    gatheredIdxLocalOffset_ = offset;
    offset += AlignBytes(entriesAligned_, static_cast<int64_t>(sizeof(int32_t)));

    // 公共区结尾
    // 聚合搬出分桶区（csAggrEnable=1 时）：bucketBase + offsetTbl + countTbl
    if (csAggrEnable_ == 1) {
        bucketBaseOffset_ = offset;
        offset += AlignBytes(maxFilteredCount_, static_cast<int64_t>(sizeof(int32_t)));
        bucketOffsetTblOffset_ = offset;
        offset += AlignBytes(actualExpertNum_, static_cast<int64_t>(sizeof(int32_t)));
        bucketCountTblOffset_ = offset;
        offset += AlignBytes(actualExpertNum_, static_cast<int64_t>(sizeof(int32_t)));
    }
    commonBufSize_ = offset;
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadExpertIdx()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(coreEntries_ * sizeof(int32_t)), 0, 0,
                                 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdxLocal, expertIdxGm_[coreFlatStart_], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::LoadXBackground(GlobalTensor<T> &xGm)
{
    LocalTensor<T> xLocal = buf_.Get<T>()[xLocalOffset_ / sizeof(T)];
    DataCopyExtParams copyParams{static_cast<uint16_t>(coreTokenNum_), static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0,
                                 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[coreTokenStart_ * cols_], copyParams, padParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    FilterAndCountVector();
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::FilterAndCountVector()
{
    LocalTensor<int32_t> expertIdxLocal = buf_.Get<int32_t>()[expertIdxLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];

    // Cast int32 -> fp32 用于 CompareScalar
    LocalTensor<float> expertIdxFp32Local = buf_.Get<float>()[expertIdxFp32LocalOffset_ / sizeof(float)];
    Cast(expertIdxFp32Local, expertIdxLocal, RoundMode::CAST_ROUND, entriesAligned_);
    PipeBarrier<PIPE_V>();

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

    LocalTensor<int32_t> flatIdxBufferLocal = buf_.Get<int32_t>()[flatIdxBufferLocalOffset_ / sizeof(int32_t)];
    ArithProgression<int32_t>(flatIdxBufferLocal, static_cast<int32_t>(0), 1, static_cast<int32_t>(coreEntries_));
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> gatheredIdxLocal = buf_.Get<int32_t>()[gatheredIdxLocalOffset_ / sizeof(int32_t)];
    uint64_t idxRsvdCnt = 0;
    GatherMask(gatheredIdxLocal, flatIdxBufferLocal, gatherMaskLocal.ReinterpretCast<uint32_t>(), true,
               static_cast<uint32_t>(coreEntries_), gatherMaskParams, idxRsvdCnt);

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 标量：构建 pairs + 累加 expert count
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
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WriteExpertCountToWorkspace()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCountStride_ * sizeof(int32_t)),
                                 0, 0, 0};
    DataCopyPad(workspaceGm_[blockIdx_ * expertCountStride_], expertCountLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::ComputeGlobalOffset()
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm_);

    LocalTensor<int32_t> allCoreExpertCountLocal =
        buf_.Get<int32_t>()[allCoreExpertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> prefixSumLocal = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];

    int64_t totalWsElements = filterNeedCoreNum_ * expertCountStride_;
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(totalWsElements * sizeof(int32_t)), 0,
                                 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(allCoreExpertCountLocal, workspaceGm_[0], copyParams, padParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    // 复用 prefixSumLocal 作为 totalCountLocal：向量化核间求和
    LocalTensor<int32_t> totalCountLocal = prefixSumLocal;
    Duplicate(totalCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    for (int64_t c = 0; c < filterNeedCoreNum_; c++) {
        PipeBarrier<PIPE_V>();
        Add(totalCountLocal, totalCountLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    int32_t totalForExpertArr[COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM]; // actualExpertNum_ <=
                                                                    // COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        totalForExpertArr[e] = totalCountLocal.GetValue(e);
    }

    // 本核 prefix sum：核 [0, blockIdx_) 的计数累加
    Duplicate(prefixSumLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    for (int64_t c = 0; c < blockIdx_; c++) {
        PipeBarrier<PIPE_V>();
        Add(prefixSumLocal, prefixSumLocal, allCoreExpertCountLocal[c * expertCountStride_], expertCountStride_);
    }

    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 标量：本核在专家 e 全局起始位置
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    // KEY_VALUE 模式：先整块清零（含尾部哨兵 0,0），再按非零专家紧凑写入，保证末尾始终有 (0,0) 分隔。
    if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
        Duplicate(expertTokensLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertTokensBufElems_ * EXPERT_ID_VALUE_NUM));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    int64_t cumulativeSum = 0;
    int64_t keyValueOffset = 0; // KEY_VALUE 模式下的紧凑写出下标（跳过 count==0 的专家）
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        int32_t totalForExpert = totalForExpertArr[e];
        int32_t prefixForExpert = prefixSumLocal.GetValue(e);

        // expertTokens 由 core 0 统一写出，支持 COUNT / CUMSUM / KEY_VALUE 三种模式
        if (blockIdx_ == 0 && expertTokensNumFlag_ != EXERPT_TOKENS_NONE) {
            if (expertTokensNumType_ == EXERPT_TOKENS_KEY_VALUE) {
                // 仅对 count!=0 的专家写 [expertId(全局), count] 键值对
                if (totalForExpert != 0) {
                    expertTokensLocal.SetValue(keyValueOffset * EXPERT_ID_VALUE_NUM,
                                               static_cast<int64_t>(e + expertStart_));
                    expertTokensLocal.SetValue(keyValueOffset * EXPERT_ID_VALUE_NUM + 1,
                                               static_cast<int64_t>(totalForExpert));
                    keyValueOffset++;
                }
            } else if (expertTokensNumType_ == EXERPT_TOKENS_COUNT) {
                expertTokensLocal.SetValue(e, static_cast<int64_t>(totalForExpert));
            } else {
                // CUMSUM：写累计和
                expertTokensLocal.SetValue(e, cumulativeSum + static_cast<int64_t>(totalForExpert));
            }
        }

        // dropPad：seed 为该专家跨核 per-expert 前缀（本核在该专家的起始 slot）；
        // dropless：seed 为全局跨专家 cumulative + per-expert 前缀。
        if (dropPadMode_ == DROP_PAD_MODE) {
            expertCountLocal.SetValue(e, prefixForExpert);
        } else {
            expertCountLocal.SetValue(e, static_cast<int32_t>(cumulativeSum) + prefixForExpert);
        }
        cumulativeSum += totalForExpert;
    }
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WriteExpertTokens()
{
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<int64_t> expertTokensLocal = buf_.Get<int64_t>()[expertTokensLocalOffset_ / sizeof(int64_t)];
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    // COUNT/CUMSUM: expertCountElements_==actualExpertNum_；KEY_VALUE: 非零对*2+哨兵
    uint32_t expertCount = static_cast<uint32_t>(expertCountElements_);
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertCount * sizeof(int64_t)), 0, 0,
                                 0};
    DataCopyPad(expertTokensGm_, expertTokensLocal, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::WaitXLoadCommon()
{
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
}

// 分桶：扫描 filteredPairsLocal，按 expertOffset 分桶，每桶存 localFlatIdx。
// 注意：Phase B ComputeGlobalOffset 已把 expertCountLocal[eo] 改写为本核在专家 e 的全局起始 newPos
// （dropless 下 = cumulativeSum + prefixForExpert），不再是每专家 count。若直接派生 bucketCountTbl
// 会用 seed 当 count，前缀和溢出 maxFilteredCount_，bucketBase 写越界。故重新扫 filteredPairsLocal
// 统计每专家 count；expertCountLocal 保持 seed 语义供 GatherAndWriteByExpert 用。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::BucketByExpert()
{
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketBase = buf_.Get<int32_t>()[bucketBaseOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketOffsetTbl = buf_.Get<int32_t>()[bucketOffsetTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketCountTbl = buf_.Get<int32_t>()[bucketCountTblOffset_ / sizeof(int32_t)];

    // 第一遍：扫描 filteredPairsLocal 统计每专家 count
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        bucketCountTbl.SetValue(e, static_cast<int32_t>(0));
    }
    for (int64_t i = 0; i < filteredCount_; i++) {
        int32_t expertOffset = filteredPairsLocal.GetValue(i * 2 + 1);
        int32_t cnt = bucketCountTbl.GetValue(expertOffset);
        bucketCountTbl.SetValue(expertOffset, cnt + 1);
    }

    // 第二遍：前缀和得 bucketOffsetTbl（sum=filteredCount_，保证 bucketBase 不越界）
    int32_t prefix = 0;
    for (int64_t e = 0; e < actualExpertNum_; e++) {
        bucketOffsetTbl.SetValue(e, prefix);
        prefix += bucketCountTbl.GetValue(e);
    }

    // 第三遍：按 expertOffset 扫填桶；cursorTbl 独立保留（bucketOffsetTbl 须原样供
    // GatherAndWriteByExpert 读起始偏移，不能当游标递增）。向量化清零替代标量初始化。
    LocalTensor<int32_t> cursorTbl = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    Duplicate(cursorTbl, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (int64_t i = 0; i < filteredCount_; i++) {
        int32_t localFlatIdx = filteredPairsLocal.GetValue(i * 2);
        int32_t expertOffset = filteredPairsLocal.GetValue(i * 2 + 1);
        int32_t baseOff = bucketOffsetTbl.GetValue(expertOffset);
        int32_t cur = cursorTbl.GetValue(expertOffset);
        bucketBase.SetValue(baseOff + cur, localFlatIdx);
        cursorTbl.SetValue(expertOffset, cur + 1);
    }
}

// 聚合搬出主循环：外层 expertOffset，内层按 k 切批，每批一次 DataCopyPad 写连续 GM 段。
// expandedX 行聚合；rowIdx/scale 在同一批循环内逐行穿插（合并原 WriteBypassForRowIdxScale）。
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherAndWriteByExpert(
    LocalTensor<T> &xLocal, LocalTensor<float> &scaleLocal, GlobalTensor<T> &expandedXGmRef,
    GlobalTensor<float> &expandedScaleGmRef)
{
    LocalTensor<int32_t> bucketBase = buf_.Get<int32_t>()[bucketBaseOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketOffsetTbl = buf_.Get<int32_t>()[bucketOffsetTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketCountTbl = buf_.Get<int32_t>()[bucketCountTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    LocalTensor<T> gatherOutBuf = buf_.Get<T>()[gatherOutBufOffset_ / sizeof(T)];

    DataCopyExtParams idxCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyExtParams scaleCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};

    for (int64_t eo = 0; eo < actualExpertNum_; eo++) {
        int64_t count_e = static_cast<int64_t>(bucketCountTbl.GetValue(eo));
        if (count_e == 0) {
            continue;
        }
        int64_t expertSeed = static_cast<int64_t>(expertCountLocal.GetValue(eo));
        int64_t baseOff = static_cast<int64_t>(bucketOffsetTbl.GetValue(eo));
        int64_t slotStart = 0;
        int64_t remaining = count_e;

        while (remaining > 0) {
            int64_t batchRows = Min(aggrOutRows_, remaining);
            int64_t batchStartNewPos = expertSeed + slotStart;
            int64_t origBaseOff = baseOff + slotStart;

            // 单次循环完成 UB→UB + rowIdx + scale（合并原三次循环）
            for (int64_t r = 0; r < batchRows; r++) {
                int32_t localFlatIdx = bucketBase.GetValue(origBaseOff + r);
                int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
                int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);
                int64_t newPos = batchStartNewPos + r;

                // 1) UB→UB 拷贝到 gatherOutBuf
                //    cols_*sizeof(T) 非 32B 对齐时，DataCopy 会按 32B 向下取整丢失尾部元素，改用 VF Copy（内部 mask
                //    处理尾部）
                if ((cols_ * static_cast<int64_t>(sizeof(T))) % BLOCK_BYTES == 0) {
                    DataCopy(gatherOutBuf[r * colsAligned_], xLocal[tokenRow * colsAligned_],
                             static_cast<int32_t>(cols_));
                } else {
                    Copy(gatherOutBuf[r * colsAligned_], xLocal[tokenRow * colsAligned_], static_cast<uint32_t>(cols_));
                }

                // 2) rowIdx 写（GATHER: [origFlatIdx]=newPos；SCATTER: [newPos]=origFlatIdx）
                if (newPos < outputRows_) {
                    if (rowIdxType_ == GATHER) {
                        idxBuf.SetValue(0, static_cast<int32_t>(newPos));
                        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                        DataCopyPad(expandedRowIdxGm_[origFlatIdx], idxBuf, idxCopyParams);
                    } else {
                        idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
                        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                        DataCopyPad(expandedRowIdxGm_[newPos], idxBuf, idxCopyParams);
                    }
                    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

                    // 3) scale 逐行写（isInputScale_ 时）
                    if (isInputScale_) {
                        DataCopyPad(expandedScaleGmRef[newPos], scaleLocal[tokenRow * scaleSlotSize_], scaleCopyParams);
                    }
                }
            }

            // 一次 DataCopyPad 写连续 GM 段（3 参版：GM, Local, extParams）
            if (batchStartNewPos < outputRows_) {
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams xCopyParams{static_cast<uint16_t>(batchRows),
                                              static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
                DataCopyPad(expandedXGmRef[batchStartNewPos * cols_], gatherOutBuf, xCopyParams);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }

            slotStart += batchRows;
            remaining -= batchRows;
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
