/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Standalone Ascend950 (arch35) kernel implementation for MoeFusedTopk.
 *
 * This header intentionally owns the whole kernel class instead of including
 * the Ascend910C implementation.  Keeping the duplicated code here allows
 * the two SoC paths to evolve independently.
 */
#ifndef ASCENDC_MOE_FUSED_TOPK_ARCH35_H_
#define ASCENDC_MOE_FUSED_TOPK_ARCH35_H_

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_fused_topk_arch35_tiling_def.h"
#include "adv_api/reduce/sum.h"

using namespace AscendC;

constexpr static uint32_t BASE_COUNT = 256;
constexpr static uint32_t REPEAT_BYTES = 256;
constexpr static uint32_t BLOCK_BYTES = 32;
constexpr static uint32_t SORT_UNIT = 32;
constexpr static uint32_t ADD_COUNT_THIRTY_TWO = 32;
constexpr static uint32_t ADD_COUNT_SIXTY_FOUR = 64;
constexpr static uint32_t ADD_COUNT_ONE_TWENTY_EIGHT = 128;
constexpr static uint32_t BUFFER_NUM = 1;
constexpr static uint32_t BUFFER_NUM_ONE = 1;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t BROADCAST_DIM = 2;
constexpr static uint32_t BROADCAST_AXIS = 1;
constexpr static uint32_t SORTED_COEF = 2;
constexpr static int32_t FLOAT_BYTES = 4;
constexpr static uint8_t REPEAT_STRIDE_EIGHT = 8;

template <typename inputT, typename calT, uint32_t enableExpertMapping>
class MoeFusedTopkArch35 {
public:
    __aicore__ inline MoeFusedTopkArch35() = default;
    __aicore__ inline void InitTilingData(MoeFusedTopkArch35TilingData *tilingData, GM_ADDR x, GM_ADDR addNum,
                                          GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y, GM_ADDR indices,
                                          GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInAddNum();
    __aicore__ inline void CopyInX(const int32_t loop);
    __aicore__ inline void ActivateAndAdd();
    __aicore__ inline void GroupTopkImpl();
    __aicore__ inline void GroupReduceSumInternelImpl();
    __aicore__ inline void GatherSigmoidImpl();
    __aicore__ inline void NormImpl();
    __aicore__ inline void CopyFromWorkspace();
    __aicore__ inline void CopyToWorkspace();
    __aicore__ inline void CopyOut(const int32_t loop);
    __aicore__ inline void CopyInMappingNum();

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    }

private:
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECIN> xInBuf_;
    TBuf<TPosition::VECCALC> addNumInBuf_;
    TBuf<TPosition::VECOUT> yOutBuf_;
    TBuf<TPosition::VECOUT> indicesOutBuf_;
    TBuf<TPosition::VECIN> assistBuf_;
    TBuf<TPosition::VECCALC> sigmoidBuf_;
    TBuf<TPosition::VECIN> sigmoidAddBuf_;
    TBuf<TPosition::VECIN> sortedBuf_;
    TBuf<TPosition::VECIN> topkValueBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    TBuf<TPosition::VECIN> mappingNumBuf_;

    uint32_t secondDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_ = 1;
    uint32_t n_ = 1;
    uint32_t k_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 1;
    float scale_ = 1.0F;
    uint32_t groupEles_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t tableDim_ = 0;
    int64_t outBatchStride_ = 0;
    int64_t batchOffset_ = 0;
    uint32_t loopBatch_ = 0;

    uint32_t groupElesAlignBlockCountFp32_ = 0;
    uint32_t groupElesAlignSortCount_ = 0;
    uint32_t secondAlignBlockCountFp32_ = 0;
    uint32_t secondAlignBlockCountFp16_ = 0;
    int64_t wsOffset_ = 0;
    uint32_t sortRepeatTimes_ = 1;
    uint32_t wholeSortNum_ = 1;
    uint32_t topKSortRepeatTimes_ = 1;

    uint32_t topnPad_ = 0;
    uint32_t topkPad_ = 0;
    uint32_t topkMaxValue_ = 0;
    uint32_t topkMinValue_ = 0;
    float floatNegativeInf_ = -3.4e38F;
    bool allScoresNan_ = false;

    GlobalTensor<inputT> mGmX_;
    GlobalTensor<inputT> mGmAddNum_;
    GlobalTensor<float> mGmY_;
    GlobalTensor<int32_t> mGmIndices_;
    GlobalTensor<float> mGmWorkspace_;
    GlobalTensor<int32_t> mappingNumGm_;
    GlobalTensor<int32_t> mappingTableGm_;
    TopkTiling topkTilingData_;
};

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::InitTilingData(
    MoeFusedTopkArch35TilingData *tilingData, GM_ADDR x, GM_ADDR addNum, GM_ADDR mappingNum, GM_ADDR mappingTable,
    GM_ADDR y, GM_ADDR indices, GM_ADDR workspace)
{
    secondDimSize_ = tilingData->secondDimSize;
    groupNum_ = tilingData->groupNum;
    groupTopk_ = tilingData->groupTopk;
    n_ = tilingData->topN;
    k_ = tilingData->topK;
    activateType_ = tilingData->activateType;
    isNorm_ = tilingData->isNorm;
    scale_ = tilingData->scale;
    groupEles_ = tilingData->groupEles;
    expertNum_ = tilingData->expertNum;
    tableDim_ = tilingData->tableDim;
    topkMaxValue_ = tilingData->topkMaxValue;
    topkMinValue_ = tilingData->topkMinValue;
    topkTilingData_ = tilingData->topkTilingData;

    uint32_t batchPerCore = tilingData->batchPerCore;
    uint32_t tailBatch = tilingData->tailBatch;
    uint32_t blockIdx = GetBlockIdx();
    uint64_t workspacePerCore = tilingData->workspacePerCore / sizeof(float);
    uint32_t perBlockCountFp32 = BLOCK_BYTES / sizeof(float);
    uint32_t perBlockCountFp16 = BLOCK_BYTES / sizeof(half);
    if (blockIdx < tailBatch) {
        loopBatch_ = batchPerCore + 1;
        batchOffset_ = blockIdx * loopBatch_;
    } else {
        loopBatch_ = batchPerCore;
        batchOffset_ = blockIdx * batchPerCore + tailBatch;
    }
    topkPad_ = CeilAlign(k_, BLOCK_BYTES / sizeof(calT));
    topnPad_ = CeilAlign(n_, BLOCK_BYTES / sizeof(calT));

    uint32_t tmpNegativeInf = NEGATIVE_MIN_VAULE_FP32;
    floatNegativeInf_ = *((float *)&tmpNegativeInf);

    outBatchStride_ = k_ * batchOffset_;
    mGmX_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(x));
    mGmAddNum_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT *>(addNum));
    mGmY_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + outBatchStride_);
    mGmIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices) + outBatchStride_);
    mGmWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    mappingNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mappingNum));
    mappingTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mappingTable));

    groupElesAlignBlockCountFp32_ = CeilAlign(groupEles_, perBlockCountFp32);
    groupElesAlignSortCount_ = CeilAlign(groupEles_, SORT_UNIT);
    secondAlignBlockCountFp32_ = CeilAlign(secondDimSize_, perBlockCountFp32);
    secondAlignBlockCountFp16_ = CeilAlign(secondDimSize_, perBlockCountFp16);
    wsOffset_ = blockIdx * workspacePerCore;
    sortRepeatTimes_ = CeilDiv(secondDimSize_, SORT_UNIT);
    topKSortRepeatTimes_ = CeilDiv(k_, SORT_UNIT);
    wholeSortNum_ = sortRepeatTimes_ * SORT_UNIT;
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::CopyInAddNum()
{
    LocalTensor<float> addNumLocal = addNumInBuf_.Get<float>();
    LocalTensor<int32_t> assistLocal = assistBuf_.Get<int32_t>();
    uint32_t secondDimSizeInputBytes = secondDimSize_ * sizeof(inputT);

    ArithProgression(assistLocal, 0, 1, wholeSortNum_);
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(addNumLocal, mGmAddNum_, {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        LocalTensor<inputT> addNumLocalInputT = addNumLocal.template ReinterpretCast<inputT>();
        // Cast permits this small-to-large in-place expansion when the source
        // begins at AlignUp(dst + count * sizeof(inputT), 32). Keep the input
        // in the upper half so its bytes cannot be overwritten by earlier
        // float results.
        DataCopyPad(addNumLocalInputT[secondAlignBlockCountFp16_], mGmAddNum_, {1, secondDimSizeInputBytes, 0, 0, 0},
                    {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(addNumLocal, addNumLocalInputT[secondAlignBlockCountFp16_], RoundMode::CAST_NONE, secondDimSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::CopyInX(const int32_t loop)
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    int64_t xOffset = loop * secondDimSize_ + batchOffset_ * static_cast<int64_t>(secondDimSize_);
    uint32_t secondDimSizeInputBytes = secondDimSize_ * sizeof(inputT);
    if constexpr (IsSameType<inputT, float>::value) {
        DataCopyPad(xLocal, mGmX_[xOffset], {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        LocalTensor<inputT> xLocalInputT = xLocal.template ReinterpretCast<inputT>();
        // See CopyInAddNum: this is the Cast-supported small-to-large
        // overlapping layout, with the source placed after the float output.
        DataCopyPad(xLocalInputT[secondAlignBlockCountFp16_], mGmX_[xOffset], {1, secondDimSizeInputBytes, 0, 0, 0},
                    {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(xLocal, xLocalInputT[secondAlignBlockCountFp16_], RoundMode::CAST_NONE, secondDimSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::ActivateAndAdd()
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    LocalTensor<float> addNumLocal = addNumInBuf_.Get<float>();
    LocalTensor<float> expTensor = tempBuf_.Get<float>();
    LocalTensor<float> sigmoidTensor = sigmoidBuf_.Get<float>();
    LocalTensor<float> sigmoidAddTensor = sigmoidAddBuf_.Get<float>();

    Muls(xLocal, xLocal, -1.0F, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(expTensor, xLocal, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(expTensor, expTensor, 1.0F, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(sigmoidAddTensor, 1.0F, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    static constexpr DivConfig divConfig = {DivAlgo::PRECISION_0ULP_FTZ_TRUE};
    Div<float, divConfig>(sigmoidTensor, sigmoidAddTensor, expTensor, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Add(sigmoidAddTensor, sigmoidTensor, addNumLocal, secondDimSize_);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::CopyToWorkspace()
{
    LocalTensor<float> sigmoidAddTensor = sigmoidAddBuf_.Get<float>();
    DataCopyPad(mGmWorkspace_[wsOffset_], sigmoidAddTensor,
                {1, static_cast<uint32_t>(secondDimSize_ * sizeof(float)), 0, 0, 0});
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::CopyFromWorkspace()
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    DataCopyExtParams xWorkspaceGroupCopyParams{1, static_cast<uint32_t>(groupEles_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> xWorkspaceGroupPadParams{
        true, 0, static_cast<uint8_t>(groupElesAlignBlockCountFp32_ - groupEles_), floatNegativeInf_};
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Duplicate<float>(xLocal, floatNegativeInf_, groupElesAlignSortCount_ * groupNum_);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    for (size_t i = 0; i < groupNum_; ++i) {
        DataCopyPad(xLocal[groupElesAlignSortCount_ * i], mGmWorkspace_[wsOffset_ + groupEles_ * i],
                    xWorkspaceGroupCopyParams, xWorkspaceGroupPadParams);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // AICore sort does not define the ordering of NaN values.  The CPU
    // reference orders them after valid routing scores, so normalize only
    // the ranking copy to the existing minimum finite sentinel.  The sigmoid
    // tensor itself remains untouched and is still used for the final output.
    LocalTensor<uint8_t> validScoreMask = sortedBuf_.Get<uint8_t>();
    // xLocal is laid out as groupNum_ rows with each row padded to a sort
    // unit. Normalize the complete padded layout; using secondDimSize_ here
    // only covers the first few rows when groupEles_ is not sort-aligned.
    uint32_t groupedScoreCount = groupElesAlignSortCount_ * groupNum_;
    Compare(validScoreMask, xLocal, xLocal, CMPMODE::EQ, groupedScoreCount);
    Select(xLocal, validScoreMask, xLocal, floatNegativeInf_, SELMODE::VSEL_TENSOR_SCALAR_MODE, groupedScoreCount);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::InitBuffer(TPipe *inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(xInBuf_, sizeof(float) * groupElesAlignSortCount_ * groupNum_);
    pipe_->InitBuffer(addNumInBuf_, sizeof(float) * secondAlignBlockCountFp16_);
    pipe_->InitBuffer(yOutBuf_, sizeof(float) * topkPad_);
    pipe_->InitBuffer(indicesOutBuf_, sizeof(int32_t) * topKSortRepeatTimes_ * SORT_UNIT);
    pipe_->InitBuffer(sigmoidBuf_, sizeof(float) * secondAlignBlockCountFp32_);
    pipe_->InitBuffer(sigmoidAddBuf_, sizeof(float) * secondAlignBlockCountFp32_);
    // The temporary buffer is shared by TopK, Sigmoid, BroadCast, Sort and Sum on arch35.
    pipe_->InitBuffer(tempBuf_, topkMaxValue_);
    pipe_->InitBuffer(sortedBuf_, sizeof(int64_t) * groupNum_ * groupElesAlignSortCount_);
    pipe_->InitBuffer(topkValueBuf_, sizeof(float) * sortRepeatTimes_ * SORT_UNIT);
    pipe_->InitBuffer(assistBuf_, sizeof(uint32_t) * wholeSortNum_);
    if constexpr (enableExpertMapping) {
        pipe_->InitBuffer(mappingNumBuf_, sizeof(int32_t) * expertNum_);
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::GroupReduceSumInternelImpl()
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> topkGroupValue = topkValueBuf_.Get<float>();
    LocalTensor<uint8_t> tempTensor = tempBuf_.Get<uint8_t>();
    // TOPK_NORMAL writes groupNum_ rows of topnPad_ values. Place the index
    // output after the complete value region; wholeSortNum_ only describes
    // one input row and can overlap the value output when groupNum_ > 1.
    LocalTensor<int32_t> indicesLocal = sortedTensor[groupNum_ * topnPad_].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> sourceIndicesLocal;
    LocalTensor<bool> finishLocal;
    AscendC::TopKInfo topkInfo = {static_cast<int32_t>(groupNum_),
                                  CeilAlign(static_cast<int32_t>(groupEles_), SORT_UNIT),
                                  static_cast<int32_t>(groupEles_)};
    AscendC::TopK<calT, false, false, false, TopKMode::TOPK_NORMAL>(
        sortedTensor, indicesLocal, xLocal, sourceIndicesLocal, finishLocal, tempTensor, n_, topkTilingData_, topkInfo);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate<float>(topkGroupValue, floatNegativeInf_, CeilAlign(groupNum_, SORT_UNIT));
    AscendC::PipeBarrier<PIPE_V>();
    // 3510's basic ReduceSum cannot safely write every unaligned group scalar.
    AscendC::Sum<calT>(topkGroupValue, sortedTensor, tempTensor, {groupNum_, topnPad_, n_});
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::GroupTopkImpl()
{
    // Sort the group scores and retain the hardware ordering for equal and
    // non-finite scores. The ATK reference uses the same AscendC TopK/Sort
    // tie behavior for these non-finite anomaly inputs (notably -inf).
    LocalTensor<uint32_t> assistLocal = assistBuf_.Get<uint32_t>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> topkGroupValue = topkValueBuf_.Get<float>();
    LocalTensor<float> sigmoidAddTensor = sigmoidAddBuf_.Get<float>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();
    LocalTensor<uint8_t> selectedGroupMask = assistBuf_.Get<uint8_t>();

    // selectedGroupMask reuses assistBuf_ later in this function. Rebuild
    // the sort indices for every batch; the all-NaN Gather path returns
    // early and therefore does not restore this shared buffer.
    ArithProgression(assistLocal.template ReinterpretCast<int32_t>(), 0, 1, wholeSortNum_);
    AscendC::PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedTensor, topkGroupValue, assistLocal, tempTensor, CeilDiv(groupNum_, SORT_UNIT));
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> dstOffset = sortedTensor.template ReinterpretCast<int32_t>();
    Duplicate(topkGroupValue, float(0), groupNum_);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // Stable CPU sorting orders an all-NaN score row by its original expert
    // index.  Detect this rare IEEE-special case here; the final gather uses
    // the same deterministic order below.  The normal path performs just one
    // scalar read and retains the vector-only sorting fast path.
    allScoresNan_ = false;
    LocalTensor<uint32_t> scoreBits = sigmoidAddTensor.template ReinterpretCast<uint32_t>();
    uint32_t firstScoreBits = scoreBits.GetValue(0);
    if ((firstScoreBits & 0x7F800000U) == 0x7F800000U && (firstScoreBits & 0x007FFFFFU) != 0U) {
        allScoresNan_ = true;
        for (size_t scoreIndex = 1; scoreIndex < secondDimSize_; ++scoreIndex) {
            uint32_t score = scoreBits.GetValue(scoreIndex);
            if ((score & 0x7F800000U) != 0x7F800000U || (score & 0x007FFFFFU) == 0U) {
                allScoresNan_ = false;
                break;
            }
        }
    }

    for (size_t i = 0; i < groupTopk_; i++) {
        int32_t selectedGroupIndex = dstOffset.GetValue(SORTED_COEF * i + 1);
        topkGroupValue.SetValue(selectedGroupIndex, float(1));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint32_t dstShape[BROADCAST_DIM] = {(uint32_t)groupNum_, (uint32_t)groupEles_};
    uint32_t srcShape[BROADCAST_DIM] = {(uint32_t)groupNum_, 1};
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    BroadCast<float, BROADCAST_DIM, BROADCAST_AXIS>(sortedTensor, topkGroupValue, dstShape, srcShape, sharedTmpBuffer);
    AscendC::PipeBarrier<PIPE_V>();
    // Use IEEE -inf for the aligned tail. Valid -inf scores are clamped to
    // the finite sentinel below, so padded slots can never enter expert TopK.
    uint32_t negativeInfinityBits = 0xFF800000;
    float floatNegativeInfinity = *((float *)&negativeInfinityBits);
    Duplicate(topkGroupValue, floatNegativeInfinity, wholeSortNum_);
    // The reference semantics use +0 for a valid but deselected expert;
    // only the aligned tail must remain -inf.
    Duplicate(topkGroupValue, float(0), secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(selectedGroupMask, sortedTensor, float(0), CMPMODE::GT, secondDimSize_);
    // Clamp only the candidate scores. The destination remains -inf for
    // deselected experts and for the aligned tail, while a selected -inf is
    // represented by the finite sentinel and therefore stays ahead of tail
    // slots during Sort.
    Maxs(sigmoidAddTensor, sigmoidAddTensor, floatNegativeInf_, secondDimSize_);
    Select(topkGroupValue, selectedGroupMask, sigmoidAddTensor, topkGroupValue, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();

    // NumPy's stable descending argsort places NaN scores after finite
    // candidates.  The hardware Sort has no such ordering guarantee, so
    // normalize NaNs in the final ranking buffer as well.  This buffer is
    // separate from sigmoidTensor, therefore output values keep their IEEE
    // behavior while indices follow the reference ordering.
    Compare(selectedGroupMask, topkGroupValue, topkGroupValue, CMPMODE::EQ, secondDimSize_);
    Select(topkGroupValue, selectedGroupMask, topkGroupValue, floatNegativeInf_, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::GatherSigmoidImpl()
{
    LocalTensor<int32_t> indicesLocal = indicesOutBuf_.Get<int32_t>();
    LocalTensor<float> sigmoidTensor = sigmoidBuf_.Get<float>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> groupTopkValue = topkValueBuf_.Get<float>();
    LocalTensor<float> yLocal = yOutBuf_.Get<float>();
    LocalTensor<uint32_t> assistLocal = assistBuf_.Get<uint32_t>();
    LocalTensor<int32_t> sortIndexInit = assistBuf_.Get<int32_t>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();

    if (allScoresNan_) {
        // For an all-NaN row, NumPy stable argsort first emits the masked
        // (zero-score) groups and then the selected NaN groups, both in their
        // original order.  Build those raw expert indices directly and use
        // the usual Gather/CopyOut flow so expert mapping and normalization
        // remain unchanged.
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        uint32_t deselectedExpertNum = (groupNum_ - groupTopk_) * groupEles_;
        for (size_t kIndex = 0; kIndex < k_; ++kIndex) {
            uint32_t expertIndex = 0;
            if (kIndex < deselectedExpertNum) {
                expertIndex = groupTopk_ * groupEles_ + kIndex;
            } else {
                expertIndex = kIndex - deselectedExpertNum;
            }
            indicesLocal.SetValue(kIndex, static_cast<int32_t>(expertIndex));
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Muls(sortedTensor.template ReinterpretCast<int32_t>(), indicesLocal, FLOAT_BYTES, k_);
        AscendC::PipeBarrier<PIPE_V>();
        Gather(yLocal, sigmoidTensor, sortedTensor.template ReinterpretCast<uint32_t>(), 0, k_);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    // Sort/Extract stay entirely on vector hardware; no O(topK * expertNum)
    // scalar tie scan is allowed on the performance path.
    ArithProgression(sortIndexInit, 0, 1, wholeSortNum_);
    AscendC::PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedTensor, groupTopkValue, assistLocal, tempTensor, sortRepeatTimes_);
    AscendC::PipeBarrier<PIPE_V>();
    Extract(tempTensor, indicesLocal.template ReinterpretCast<uint32_t>(), sortedTensor, topKSortRepeatTimes_);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(sortedTensor.template ReinterpretCast<int32_t>(), indicesLocal, FLOAT_BYTES, k_);
    AscendC::PipeBarrier<PIPE_V>();
    Gather(yLocal, sigmoidTensor, sortedTensor.template ReinterpretCast<uint32_t>(), 0, k_);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::NormImpl()
{
    LocalTensor<float> yLocal = yOutBuf_.Get<float>();
    // Sum requires dst, src and sharedTmpBuffer to be non-overlapping. The
    // group TopK values are dead after GatherSigmoidImpl, so reuse that buffer
    // for the single reduction result and retain tempBuf_ as scratch only.
    LocalTensor<float> reduceResult = topkValueBuf_.Get<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    AscendC::Sum<calT>(reduceResult, yLocal, sharedTmpBuffer, {1, topkPad_, k_});
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float reduceSumValue = 1 / reduceResult.GetValue(0);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(yLocal, yLocal, reduceSumValue, k_);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(yLocal, yLocal, scale_, k_);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::CopyInMappingNum()
{
    LocalTensor<int32_t> mappingNumLocal = mappingNumBuf_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertNum_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(mappingNumLocal, mappingNumGm_, copyParams, padParams);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::CopyOut(const int32_t loop)
{
    int64_t offset = loop * k_;
    LocalTensor<float> yLocal = yOutBuf_.Get<float>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(k_ * sizeof(float)), 0, 0, 0};
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(mGmY_[offset], yLocal, copyParams);

    LocalTensor<int32_t> indicesLocal = indicesOutBuf_.Get<int32_t>();
    if constexpr (enableExpertMapping) {
        LocalTensor<int32_t> mappingNumLocal = mappingNumBuf_.Get<int32_t>();
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (size_t kIndex = 0; kIndex < k_; ++kIndex) {
            uint32_t expertId = indicesLocal.GetValue(kIndex);
            uint32_t expertMappingNum = mappingNumLocal.GetValue(expertId);
            uint32_t redundantOffset = expertMappingNum == 0 ? 0 : (batchOffset_ + loop) % expertMappingNum;
            uint32_t tableOffset = expertId * tableDim_ + redundantOffset;
            indicesLocal.SetValue(kIndex, mappingTableGm_[tableOffset].GetValue(0));
        }
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    }
    DataCopyPad(mGmIndices_[offset], indicesLocal, copyParams);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopkArch35<inputT, calT, enableExpertMapping>::Process()
{
    CopyInAddNum();
    if constexpr (enableExpertMapping) {
        CopyInMappingNum();
    }
    for (size_t loop = 0; loop < loopBatch_; ++loop) {
        CopyInX(loop);
        ActivateAndAdd();
        CopyToWorkspace();
        CopyFromWorkspace();
        GroupReduceSumInternelImpl();
        GroupTopkImpl();
        GatherSigmoidImpl();
        if (isNorm_ == 1) {
            NormImpl();
        }
        CopyOut(loop);
    }
}

#endif // ASCENDC_MOE_FUSED_TOPK_ARCH35_H_
