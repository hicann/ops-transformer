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
 * \file stem_indexer_service_vector.h
 * \brief
 */
#ifndef stem_indexer_SERVICE_VECTOR_H
#define stem_indexer_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "stem_indexer_common.h"
#include "../arch35/vf/stem_indexer_vector1.h"
#include "../arch35/vf/stem_indexer_topk.h"
#include "../arch35/vf/stem_indexer_topk_utils.h"

namespace SIKernel {
using namespace SICommon;
template <typename SIT>
class SIVector {
public:
    // =================================类型定义区=================================
    using QK_T = float32_t;
    using SCORE_T = typename SIT::scoreType;

    __aicore__ inline SIVector(){};
    __aicore__ inline void ProcessVec1(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
    __aicore__ inline void ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx);
    __aicore__ inline void InitBuffers(TPipe *pipe, bool needCleanOutput);
    __aicore__ inline void InitParams(const struct SICommon::ConstInfo &constInfo);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> vbiasGm, GlobalTensor<int32_t> sparseIndicesGm,
                                              GlobalTensor<int32_t> sparseSeqLenGm);
    __aicore__ inline void InitSparseIndicesToNegOne(uint32_t gS1Idx, uint32_t actMBaseSize, uint32_t actS1Size,
                                                     int64_t indiceOutOffset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<float> vBiasGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;

    static constexpr uint32_t TOPK_MTE3_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MAX_TOPK_COUNT = 256;
    static constexpr uint32_t TOPK_ALIGN_SIZE = 256;
    static constexpr uint32_t TOPK_ROW_BATCH = 4;
    static constexpr uint32_t INDICE_LEN_ROW_STRIDE = 8U;
    static constexpr uint32_t OUTPUT_INIT_ALIGN_ELEMS = 512U / sizeof(int32_t);
    static constexpr int32_t K_SMALL_SEQ_MAX = 56;
    static constexpr int32_t K_MEDIUM_SEQ_MAX = 160;

private:
    __aicore__ inline void CleanOutput();
    __aicore__ inline void InitOutputSingleCore(GlobalTensor<int32_t> &outputGm, uint64_t totalSize, int32_t value);
    __aicore__ inline uint32_t CalcDynamicTopkCount(uint32_t s1Idx, int64_t kbOffset, int32_t numPromptK);
    __aicore__ inline void WriteDirectIndices(uint32_t outputLen, int64_t baseLenOffset, int64_t baseOutOffset,
                                              uint32_t curAivGSize, int64_t curAivGS1ProcNum, uint32_t actS1Size,
                                              uint32_t s1Idx, LocalTensor<uint32_t> &indicesOutLocal);
    __aicore__ inline void CopyOutPackedResults(const SICommon::RunInfo &info, int64_t curAivGS1Idx,
                                                int64_t curAivGS1ProcNum);
    __aicore__ inline void RunTopKBatchRows(const SICommon::RunInfo &info, bool isFirstS2InnerLoop,
                                            bool isLastS2InnerLoop, const SICommon::RowIdx4 &rowIdx4,
                                            const SICommon::TopkNum4 &topkNum4,
                                            const SICommon::S2ValidLen4 &s2ValidLen4, uint32_t batchRowNum,
                                            int64_t curAivGS1Idx);
    // ================================Local Buffer区====================================
    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1Local_;
    // tmp buff for vbias
    TBuf<TPosition::VECCALC> vBiasBuf_;
    LocalTensor<float> vBiasLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> globalIndexBuf_;
    LocalTensor<uint32_t> globalIndexLocal_;

    TBuf<TPosition::VECCALC> histogramBuf_;
    LocalTensor<uint32_t> histogramLocal_;

    TBuf<TPosition::VECCALC> indiceLenBuf_;
    LocalTensor<uint32_t> indiceLenLocal_;

    int32_t blockId_ = -1;
    bool needCleanOutput_ = false;
    // para for vector
    int32_t s2BaseSize_ = 0;
    uint32_t mrgRowStride_ = 0;
    int32_t mBaseSize_ = 0;
    float alpha_ = 1.0f;
    float kBlockNumRateMedium_ = 0.2f;
    uint32_t kBlockNumBiasMedium_ = 30U;
    float kBlockNumRateLarge_ = 0.1f;
    uint32_t kBlockNumBiasLarge_ = 30U;
    uint32_t initialBlocks_ = 4U;
    uint32_t windowSize_ = 4U;
    struct SICommon::ConstInfo constInfo_;
    SIKernel::StemIndexerTopk<SCORE_T> topkOp_;
};

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitBuffers(TPipe *pipe, bool needCleanOutput)
{
    needCleanOutput_ = needCleanOutput;
    // 三缓冲，每槽32KB；TopK histogram已拆为独立UB。
    pipe->InitBuffer(resMm1Buf_, SICommon::MM1_RES_BUFFER_NUM * SICommon::MM1_RES_SLOT_BYTES);
    resMm1Local_ = resMm1Buf_.Get<QK_T>(); // qk

    // vBias与MM1结果使用相同的三缓冲槽号。
    pipe->InitBuffer(vBiasBuf_, SICommon::VBIAS_BUFFER_NUM * s2BaseSize_ * sizeof(float));
    vBiasLocal_ = vBiasBuf_.Get<float>(); // vBias

    // Topk
    // uint16和uint32申请相同的物理空间；uint16通过更大的元素stride保留每行输出264个int32_t的空间。
    pipe->InitBuffer(mrgValueBuf_, ((constInfo_.mBaseSize + 1U) >> 1U) * (MAX_TOPK_COUNT + SICommon::TRUNK_LEN_256) *
                                       sizeof(uint32_t));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();

    // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    pipe->InitBuffer(globalIndexBuf_, ((constInfo_.mBaseSize + 1U) >> 1U) * MAX_TOPK_COUNT * sizeof(uint32_t));
    globalIndexLocal_ = globalIndexBuf_.Get<uint32_t>();

    // TopK batch4 histogram独立申请，不再复用resMm1结果槽。
    pipe->InitBuffer(histogramBuf_, SIKernel::StemIndexerTopk<SCORE_T>::HISTOGRAM_SIZE_U32 * sizeof(uint32_t));
    histogramLocal_ = histogramBuf_.Get<uint32_t>();

    // indice_len
    // 每行长度独占一个32B槽位，满足UB地址对齐要求并支持多行连续搬出。
    pipe->InitBuffer(indiceLenBuf_, ((constInfo_.mBaseSize + 1U) >> 1U) * INDICE_LEN_ROW_STRIDE * sizeof(uint32_t));
    indiceLenLocal_ = indiceLenBuf_.Get<uint32_t>();

    // 三个流水槽初始均归Cube使用。
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 0U);
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 1U);
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT + 2U);
    if (needCleanOutput_) {
        CleanOutput();
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CleanOutput()
{
    uint64_t totalIndiceSize =
        constInfo_.batchSize * constInfo_.qHeadNum * constInfo_.qSeqSize * constInfo_.sparseCount;
    InitOutputSingleCore(sparseIndicesGm, totalIndiceSize, static_cast<int32_t>(SICommon::INVALID_IDX));
    SyncAll();
    uint64_t totalLenSize = constInfo_.batchSize * constInfo_.qHeadNum * constInfo_.qSeqSize;
    InitOutputSingleCore(sparseSeqLenGm, totalLenSize, static_cast<int32_t>(0));
    SyncAll();
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitOutputSingleCore(GlobalTensor<int32_t> &outputGm, uint64_t totalSize,
                                                           int32_t value)
{
    uint64_t aivCoreNum = static_cast<uint64_t>(GetBlockNum()) * GetTaskRation();
    uint64_t coreIdx = static_cast<uint64_t>(blockId_);
    uint64_t singleCoreSize =
        SICommon::Align((totalSize + aivCoreNum - 1U) / aivCoreNum, static_cast<uint64_t>(OUTPUT_INIT_ALIGN_ELEMS));
    int64_t offset = static_cast<int64_t>(coreIdx) * static_cast<int64_t>(singleCoreSize);
    if (offset < static_cast<int64_t>(totalSize)) {
        uint64_t initSize = Min(singleCoreSize, totalSize - static_cast<uint64_t>(offset));
        InitOutput<int32_t>(outputGm[offset], initSize, value);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitParams(const struct SICommon::ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
    s2BaseSize_ = constInfo.s2BaseSize; // 256
    mBaseSize_ = constInfo.mBaseSize;
    alpha_ = constInfo.alpha;
    kBlockNumRateMedium_ = constInfo.kBlockNumRateMedium;
    kBlockNumBiasMedium_ = constInfo.kBlockNumBiasMedium;
    kBlockNumRateLarge_ = constInfo.kBlockNumRateLarge;
    kBlockNumBiasLarge_ = constInfo.kBlockNumBiasLarge;
    initialBlocks_ = constInfo.initialBlocks;
    windowSize_ = constInfo.windowSize;
    blockId_ = GetBlockIdx();
    mrgRowStride_ = (MAX_TOPK_COUNT + SICommon::TRUNK_LEN_256) * sizeof(uint32_t) / sizeof(SCORE_T);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitVecInputTensor(GlobalTensor<float> vBiasGm,
                                                         GlobalTensor<int32_t> sparseIndicesGm,
                                                         GlobalTensor<int32_t> sparseSeqLenGm)
{
    this->vBiasGm = vBiasGm;
    this->sparseIndicesGm = sparseIndicesGm;
    this->sparseSeqLenGm = sparseSeqLenGm;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 2);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 2);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitSparseIndicesToNegOne(uint32_t gS1Idx, uint32_t actMBaseSize,
                                                                uint32_t actS1Size, int64_t indiceOutOffset)
{
    if (actS1Size == 0U) {
        return;
    }
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = static_cast<int64_t>(gS1Idx) * static_cast<int64_t>(mBaseSize_);
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = actMBaseSize;
    // 当前AIV核处理的GS1起始索引（双核分工）
    uint32_t aivParity = static_cast<uint32_t>(blockId_) & 1U;
    int64_t curAivGS1Idx =
        curGS1Idx + static_cast<int64_t>(aivParity) * ((static_cast<int64_t>(curGS1ProcNum) + 1LL) >> 1U);
    // 当前AIV核需要处理的GS1行数量
    int64_t curAivGS1ProcNum = (aivParity == 0U) ? ((static_cast<uint64_t>(curGS1ProcNum) + 1U) >> 1U) :
                                                   (static_cast<uint64_t>(curGS1ProcNum) >> 1U);
    if (curAivGS1ProcNum == 0) {
        return;
    }
    uint32_t curGlobalGIdx = curAivGS1Idx / actS1Size;
    uint32_t curGlobalS1Idx = curAivGS1Idx % actS1Size;
    int64_t lastAivGS1Idx = curAivGS1Idx + curAivGS1ProcNum - 1;
    uint32_t lastGlobalGIdx = lastAivGS1Idx / actS1Size;
    uint32_t lastGlobalS1Idx = lastAivGS1Idx % actS1Size;
    int64_t startPhysicalRow = static_cast<int64_t>(curGlobalGIdx) * static_cast<int64_t>(constInfo_.qSeqSize) +
                               static_cast<int64_t>(curGlobalS1Idx);
    int64_t endPhysicalRow = static_cast<int64_t>(lastGlobalGIdx) * static_cast<int64_t>(constInfo_.qSeqSize) +
                             static_cast<int64_t>(lastGlobalS1Idx);
    uint64_t clearRowNum = static_cast<uint64_t>(endPhysicalRow - startPhysicalRow + 1LL);
    int32_t neg = -1;
    int64_t outSplit1Offset = indiceOutOffset + startPhysicalRow * static_cast<int64_t>(constInfo_.kSeqSize);
    GlobalTensor<int32_t> indiceSplitOut = sparseIndicesGm[outSplit1Offset];
    AscendC::InitGlobalMemory(indiceSplitOut, clearRowNum * constInfo_.kSeqSize, neg);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessVec1(const SICommon::RunInfo &info)
{
    auto bufferIdx = info.loop % SICommon::MM1_RES_BUFFER_NUM;
    int64_t gS1BasePerVecSize_ = (constInfo_.mBaseSize + 1U) >> 1U; // 当前每个vec核上分到的mbase/2行
    int64_t curS2Idx = static_cast<int64_t>(info.s2Idx) * static_cast<int64_t>(s2BaseSize_);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + bufferIdx);
    // vBiasGm --> vBiasLocal_ 搬运vBias
    int64_t vBiasGmOffset = info.tensorVBiasOffset + curS2Idx;
    DataCopyPadExtParams<float> padVBiasParams{false, 0, 0, 0};
    DataCopyExtParams vBiasDataCopyExtParams;
    vBiasDataCopyExtParams.blockCount = 1;
    vBiasDataCopyExtParams.blockLen = info.actualSingleProcessSInnerSize * sizeof(float);
    vBiasDataCopyExtParams.srcStride = 0;
    vBiasDataCopyExtParams.dstStride = 0;
    const int64_t vBiasLocalOffset = static_cast<int64_t>(bufferIdx) * static_cast<int64_t>(s2BaseSize_);
    DataCopyPad(vBiasLocal_[vBiasLocalOffset], vBiasGm[vBiasGmOffset], vBiasDataCopyExtParams, padVBiasParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + bufferIdx);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + bufferIdx);

    if (info.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    // CV同步：V核等待C核完成mm1并将结果搬运到UB。
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_CV_EVENT + bufferIdx);

    const int64_t resMm1Offset =
        static_cast<int64_t>(bufferIdx) * static_cast<int64_t>(SICommon::MM1_RES_SLOT_BYTES / sizeof(QK_T));
    auto qkBase = resMm1Local_[resMm1Offset];
    auto outBase = mrgValueLocal_[static_cast<int64_t>(MAX_TOPK_COUNT)];
    auto vBiasBase = vBiasLocal_[vBiasLocalOffset];

    vector1::MulRSquareAndAddVBiasVF(outBase, qkBase, vBiasBase, constInfo_.rSquare, gS1BasePerVecSize_, s2BaseSize_,
                                     mrgRowStride_);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + bufferIdx);
}

template <typename SIT>
__aicore__ inline uint32_t SIVector<SIT>::CalcDynamicTopkCount(uint32_t s1Idx, int64_t kbOffset, int32_t numPromptK)
{
    // 计算 kStart（分段函数）
    int32_t kStart;
    if (numPromptK < K_SMALL_SEQ_MAX) {
        kStart = numPromptK;
    } else if (numPromptK < K_MEDIUM_SEQ_MAX) {
        kStart = static_cast<int32_t>(numPromptK * kBlockNumRateMedium_ + kBlockNumBiasMedium_);
    } else {
        kStart = static_cast<int32_t>(numPromptK * kBlockNumRateLarge_ + kBlockNumBiasLarge_);
    }

    // 计算 s1Pos 和 decayLen
    int64_t s1Pos = static_cast<int64_t>(s1Idx) + kbOffset;
    int32_t decayLen = numPromptK - kStart;

    // 边界条件：s1Pos < kStart 或 decayLen <= 1 时，直接返回 kStart
    if (s1Pos < kStart || decayLen <= 1) {
        return static_cast<uint32_t>(kStart);
    }

    // 公式定义 kEnd = alpha * kStart，插值完成后再向下取整。
    float kStartFloat = static_cast<float>(kStart);
    float kEnd = kStartFloat * alpha_;

    // 计算插值系数 t
    float t = static_cast<float>(s1Pos - kStart) / static_cast<float>(decayLen - 1);
    float interpolatedTopkCount = kStartFloat + t * (kEnd - kStartFloat);

    // 正数转有符号整数等价于 floor；外推到负数时先钳制，避免无符号转换。
    int32_t dynamicTopkCount = interpolatedTopkCount < 1.0f ? 1 : static_cast<int32_t>(interpolatedTopkCount);
    dynamicTopkCount = Min(dynamicTopkCount, kStart);

    return static_cast<uint32_t>(dynamicTopkCount);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::WriteDirectIndices(uint32_t outputLen, int64_t baseLenOffset,
                                                         int64_t baseOutOffset, uint32_t curAivGSize,
                                                         int64_t curAivGS1ProcNum, uint32_t actS1Size, uint32_t s1Idx,
                                                         LocalTensor<uint32_t> &indicesOutLocal)
{
    outputLen = Min(outputLen, MAX_TOPK_COUNT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    Duplicate(indiceLenLocal_, outputLen, 1);
    CreateVecIndex(indicesOutLocal.ReinterpretCast<int32_t>(), 0, outputLen);
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams indiceLenCp;
    indiceLenCp.blockCount = 1;
    indiceLenCp.blockLen = static_cast<uint16_t>(sizeof(uint32_t));
    indiceLenCp.srcStride = 0;
    indiceLenCp.dstStride = 0;
    AscendC::DataCopyParams copyOut;
    copyOut.blockCount = 1;
    copyOut.blockLen = static_cast<uint16_t>(outputLen * sizeof(int32_t));
    copyOut.srcStride = 0;
    copyOut.dstStride = 0;
    for (uint32_t gIdx = 0; gIdx < curAivGSize; gIdx++) {
        uint32_t mInnerIdx = gIdx * actS1Size + s1Idx;
        if (mInnerIdx >= curAivGS1ProcNum) {
            break;
        }
        DataCopyPad(
            sparseSeqLenGm[baseLenOffset + static_cast<int64_t>(gIdx) * static_cast<int64_t>(constInfo_.qSeqSize)],
            indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCp);
        DataCopyPad(
            sparseIndicesGm[baseOutOffset + static_cast<int64_t>(gIdx) * static_cast<int64_t>(constInfo_.qSeqSize) *
                                                static_cast<int64_t>(constInfo_.kSeqSize)],
            indicesOutLocal.ReinterpretCast<int32_t>(), copyOut);
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CopyOutPackedResults(const SICommon::RunInfo &info, int64_t curAivGS1Idx,
                                                           int64_t curAivGS1ProcNum)
{
    const uint32_t packedOutputLen = MAX_TOPK_COUNT + initialBlocks_ + windowSize_;
    const uint32_t copyLen = Min(packedOutputLen, static_cast<uint32_t>(constInfo_.kSeqSize));
    const uint32_t copyBytes = copyLen * sizeof(int32_t);
    const uint32_t srcRowBytes = mrgRowStride_ * sizeof(SCORE_T);
    const uint32_t srcBlockBytes = (copyBytes + 31U) & ~31U;

    AscendC::DataCopyParams packedCopyParams;
    packedCopyParams.blockLen = static_cast<uint16_t>(copyBytes);
    // UB侧stride以32B data block为单位，GM侧stride以字节为单位。
    packedCopyParams.srcStride = static_cast<uint16_t>((srcRowBytes - srcBlockBytes) >> 5U);
    packedCopyParams.dstStride =
        static_cast<uint32_t>((static_cast<uint32_t>(constInfo_.kSeqSize) - copyLen) * sizeof(int32_t));

    AscendC::DataCopyParams packedLenCopyParams;
    packedLenCopyParams.blockLen = static_cast<uint16_t>(sizeof(uint32_t));
    // 每个长度槽位正好占32B，当前块补齐后紧接下一行，因此srcStride为0。
    packedLenCopyParams.srcStride = 0U;
    packedLenCopyParams.dstStride = 0U;
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    uint32_t localRowStart = 0U;
    const uint32_t rowNum = static_cast<uint32_t>(curAivGS1ProcNum);
    while (localRowStart < rowNum) {
        int64_t globalGS1Idx = curAivGS1Idx + static_cast<int64_t>(localRowStart);
        uint32_t globalGIdx = static_cast<uint32_t>(globalGS1Idx / info.actS1Size);
        uint32_t globalS1Idx = static_cast<uint32_t>(globalGS1Idx % info.actS1Size);
        uint32_t segmentRowNum = Min(rowNum - localRowStart, info.actS1Size - globalS1Idx);
        int64_t baseOutOffset =
            info.indiceOutOffset + (static_cast<int64_t>(globalGIdx) * static_cast<int64_t>(constInfo_.qSeqSize) +
                                    static_cast<int64_t>(globalS1Idx)) *
                                       static_cast<int64_t>(constInfo_.kSeqSize);
        int64_t baseLenOffset = info.indiceLenOffset +
                                static_cast<int64_t>(globalGIdx) * static_cast<int64_t>(constInfo_.qSeqSize) +
                                static_cast<int64_t>(globalS1Idx);
        const int64_t mrgValueOffset = static_cast<int64_t>(localRowStart) * static_cast<int64_t>(mrgRowStride_);
        const int64_t indiceLenLocalOffset =
            static_cast<int64_t>(localRowStart) * static_cast<int64_t>(INDICE_LEN_ROW_STRIDE);

        packedCopyParams.blockCount = static_cast<uint16_t>(segmentRowNum);
        DataCopyPad(sparseIndicesGm[baseOutOffset], mrgValueLocal_[mrgValueOffset].template ReinterpretCast<int32_t>(),
                    packedCopyParams);
        packedLenCopyParams.blockCount = static_cast<uint16_t>(segmentRowNum);
        DataCopyPad(sparseSeqLenGm[baseLenOffset], indiceLenLocal_[indiceLenLocalOffset].ReinterpretCast<int32_t>(),
                    packedLenCopyParams);
        localRowStart += segmentRowNum;
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::RunTopKBatchRows(const SICommon::RunInfo &info, bool isFirstS2InnerLoop,
                                                       bool isLastS2InnerLoop, const SICommon::RowIdx4 &rowIdx4,
                                                       const SICommon::TopkNum4 &topkNum4,
                                                       const SICommon::S2ValidLen4 &s2ValidLen4, uint32_t batchRowNum,
                                                       int64_t curAivGS1Idx)
{
    // padding（各行按自身 s2BlockValidLen 补齐到 TRUNK_LEN_256，非零行结果一致）
    if (isFirstS2InnerLoop) {
        TopkUtils::PadMrgFourRowsFirstS2RangeVF<SCORE_T>(mrgValueLocal_, rowIdx4, topkNum4, s2ValidLen4, mrgRowStride_);
    } else {
        TopkUtils::PadMrgFourRowsNotFirstS2RangeVF<SCORE_T>(mrgValueLocal_, rowIdx4, topkNum4, s2ValidLen4,
                                                            mrgRowStride_);
    }

    uint32_t inputOffset = isFirstS2InnerLoop ? TOPK_ALIGN_SIZE : 0U;
    uint32_t validLen = isFirstS2InnerLoop ? SICommon::TRUNK_LEN_256 : (TOPK_ALIGN_SIZE + SICommon::TRUNK_LEN_256);

    const int64_t reuseMm1ResOffset = static_cast<int64_t>(info.loop % SICommon::MM1_RES_BUFFER_NUM) *
                                      static_cast<int64_t>(SICommon::MM1_RES_SLOT_BYTES / sizeof(QK_T));
    LocalTensor<uint32_t> reuseMm1ResLocal = resMm1Local_[reuseMm1ResOffset].ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();

    topkOp_.Batch4Rows(mrgValueLocal_, reuseMm1ResLocal, histogramLocal_, globalIndexLocal_, rowIdx4, topkNum4,
                       batchRowNum, mrgRowStride_, inputOffset, validLen, info.s2Idx, info.s2LoopEnd + 1);

    if (!isLastS2InnerLoop) {
        return;
    }
    PipeBarrier<PIPE_V>();
    const uint32_t packedOutputLen = MAX_TOPK_COUNT + initialBlocks_ + windowSize_;
    for (uint32_t rowIdx = 0; rowIdx < batchRowNum; rowIdx++) {
        uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, rowIdx);
        uint32_t globalS1Idx = static_cast<uint32_t>(curAivGS1Idx + mInnerIdx) % info.actS1Size;
        uint32_t curS1RealS2Len = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            int32_t curS1RealS2LenTmp = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                                        static_cast<int32_t>(globalS1Idx) + 1;
            curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
        }

        uint32_t topkSelectNum = Min(SICommon::GetLane(topkNum4, rowIdx), MAX_TOPK_COUNT);
        int64_t mrgValueOffset = static_cast<int64_t>(mInnerIdx) * static_cast<int64_t>(mrgRowStride_);
        int64_t globalIndexOffset = static_cast<int64_t>(mInnerIdx) * static_cast<int64_t>(MAX_TOPK_COUNT);
        LocalTensor<uint32_t> packedRow = mrgValueLocal_[mrgValueOffset].template ReinterpretCast<uint32_t>();
        TopkUtils::PackSparseIndicesU32VF(packedRow, globalIndexLocal_[globalIndexOffset], packedOutputLen,
                                          initialBlocks_, topkSelectNum, windowSize_, curS1RealS2Len - windowSize_);

        const int64_t indiceLenLocalOffset =
            static_cast<int64_t>(mInnerIdx) * static_cast<int64_t>(INDICE_LEN_ROW_STRIDE);
        Duplicate(indiceLenLocal_[indiceLenLocalOffset], initialBlocks_ + topkSelectNum + windowSize_, 1);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop,
                                                  bool isLastS2InnerLoop)
{
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = static_cast<int64_t>(info.gS1Idx) * static_cast<int64_t>(mBaseSize_);
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = info.actMBaseSize;
    // 当前AIV核处理的GS1起始索引（双核分工）
    uint32_t aivParity = static_cast<uint32_t>(blockId_) & 1U;
    int64_t curAivGS1Idx =
        curGS1Idx + static_cast<int64_t>(aivParity) * ((static_cast<int64_t>(curGS1ProcNum) + 1LL) >> 1U);
    // 当前AIV核需要处理的GS1行数量
    int64_t curAivGS1ProcNum = (aivParity == 0U) ? ((static_cast<uint64_t>(curGS1ProcNum) + 1U) >> 1U) :
                                                   (static_cast<uint64_t>(curGS1ProcNum) >> 1U);

    // S2 基本块信息
    uint32_t s2BlockStart = info.s2Idx * s2BaseSize_;

    // 计算动态 topkCount 所需的块级常量（使用原始 actS2Size）
    int64_t kbOffset = static_cast<int64_t>(info.actS2Size) - static_cast<int64_t>(info.actS1Size);
    int32_t numPromptK = static_cast<int32_t>(info.promptLen);

    uint32_t curAivGSize = CeilDiv(curAivGS1ProcNum, info.actS1Size);
    // 当前核处理的 S1 轴大小
    // 当mBaseSize小于actS1Size时，curAivS1Size为curAivGS1ProcNum
    // 当mBaseSize大于等于actS1Size时， curAivS1Size为info.actS1Size
    uint32_t curAivS1Size = Min(info.actS1Size, static_cast<uint32_t>(curAivGS1ProcNum));
    if (isFirstS2InnerLoop && initialBlocks_ > 0) {
        TopkUtils::PadMrgRowsInitBlockVF(mrgValueLocal_, mrgRowStride_, static_cast<uint32_t>(curAivGS1ProcNum),
                                         TOPK_ALIGN_SIZE, initialBlocks_);
    }

    SICommon::RowIdx4 rowIdx4 = {0U, 0U, 0U, 0U};
    SICommon::TopkNum4 topkNum4 = {0U, 0U, 0U, 0U};
    SICommon::S2ValidLen4 s2ValidLen4 = {0U, 0U, 0U, 0U};
    uint32_t batchRowNum = 0;

    for (uint32_t s1Idx = 0; s1Idx < curAivS1Size; s1Idx++) {
        uint32_t nowCurAivGS1Idx = curAivGS1Idx + s1Idx;
        uint32_t globalS1Idx = nowCurAivGS1Idx % info.actS1Size;
        uint32_t curS1RealS2Len = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            int32_t curS1RealS2LenTmp = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                                        static_cast<int32_t>(globalS1Idx) + 1;
            curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
        }

        uint32_t dynamicTopkCount = CalcDynamicTopkCount(globalS1Idx, kbOffset, numPromptK);
        uint32_t topkSelectNum = Min(dynamicTopkCount, MAX_TOPK_COUNT);
        uint32_t totalOutput = initialBlocks_ + topkSelectNum + windowSize_;
        if ((constInfo_.attenMaskFlag && (curS1RealS2Len <= initialBlocks_ + windowSize_)) ||
            (curS1RealS2Len <= totalOutput)) {
            if (isLastS2InnerLoop) {
                const uint32_t packedOutputLen = MAX_TOPK_COUNT + initialBlocks_ + windowSize_;
                uint32_t directOutputLen = Min(curS1RealS2Len, packedOutputLen);
                for (uint32_t gIdx = 0; gIdx < curAivGSize; gIdx++) {
                    uint32_t mInnerIdx = gIdx * info.actS1Size + s1Idx;
                    if (mInnerIdx >= curAivGS1ProcNum) {
                        break;
                    }
                    int64_t mrgValueOffset = static_cast<int64_t>(mInnerIdx) * static_cast<int64_t>(mrgRowStride_);
                    int64_t globalIndexOffset = static_cast<int64_t>(mInnerIdx) * static_cast<int64_t>(MAX_TOPK_COUNT);
                    LocalTensor<uint32_t> packedRow =
                        mrgValueLocal_[mrgValueOffset].template ReinterpretCast<uint32_t>();
                    TopkUtils::PackSparseIndicesU32VF(packedRow, globalIndexLocal_[globalIndexOffset], packedOutputLen,
                                                      0U, 0U, directOutputLen, 0U);
                    const int64_t indiceLenLocalOffset =
                        static_cast<int64_t>(mInnerIdx) * static_cast<int64_t>(INDICE_LEN_ROW_STRIDE);
                    Duplicate(indiceLenLocal_[indiceLenLocalOffset], directOutputLen, 1);
                }
            }
            continue;
        }

        uint32_t s2ValidEnd = Max(curS1RealS2Len - windowSize_, 0);
        uint32_t s2BlockEnd = Min(s2BlockStart + s2BaseSize_, s2ValidEnd);
        uint32_t s2BlockValidLen = (s2BlockEnd > s2BlockStart) ? (s2BlockEnd - s2BlockStart) : 0;

        for (uint32_t gIdx = 0; gIdx < curAivGSize; gIdx++) {
            uint32_t mInnerIdx = gIdx * info.actS1Size + s1Idx;
            if (mInnerIdx >= curAivGS1ProcNum) {
                break;
            }

            SICommon::SetLane(rowIdx4, batchRowNum, mInnerIdx);
            SICommon::SetLane(topkNum4, batchRowNum, topkSelectNum);
            SICommon::SetLane(s2ValidLen4, batchRowNum, s2BlockValidLen);
            batchRowNum++;
            if (batchRowNum == TOPK_ROW_BATCH) {
                RunTopKBatchRows(info, isFirstS2InnerLoop, isLastS2InnerLoop, rowIdx4, topkNum4, s2ValidLen4,
                                 batchRowNum, curAivGS1Idx);
                batchRowNum = 0;
            }
        }
    }
    if (batchRowNum > 0U) {
        uint32_t tailLane = batchRowNum - 1U;
        uint32_t tailRowIdx = SICommon::GetLane(rowIdx4, tailLane);
        uint32_t tailTopkNum = SICommon::GetLane(topkNum4, tailLane);
        uint32_t tailS2ValidLen = SICommon::GetLane(s2ValidLen4, tailLane);
        for (uint32_t lane = batchRowNum; lane < TOPK_ROW_BATCH; lane++) {
            SICommon::SetLane(rowIdx4, lane, tailRowIdx);
            SICommon::SetLane(topkNum4, lane, tailTopkNum);
            SICommon::SetLane(s2ValidLen4, lane, tailS2ValidLen);
        }
        RunTopKBatchRows(info, isFirstS2InnerLoop, isLastS2InnerLoop, rowIdx4, topkNum4, s2ValidLen4, batchRowNum,
                         curAivGS1Idx);
    }
    // TopK已不再访问当前resMm1Local_槽，先归还给Cube，使下一轮MM1可与最终MTE3搬出并行。
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_V>(SICommon::CROSS_VC_EVENT +
                                                      (info.loop % SICommon::MM1_RES_BUFFER_NUM));
    if (isLastS2InnerLoop) {
        CopyOutPackedResults(info, curAivGS1Idx, curAivGS1ProcNum);
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx)
{
    if ASCEND_IS_AIV {
        uint32_t actMBaseSize = tempLoopInfo.actMBaseSize;
        uint32_t actS1Size = tempLoopInfo.actS1Size;
        uint32_t actS2Size = tempLoopInfo.actS2Size;
        int64_t qFlatBase = static_cast<int64_t>(tempLoopInfo.bIdx) * static_cast<int64_t>(constInfo_.qSeqSize) *
                                static_cast<int64_t>(constInfo_.qHeadNum) +
                            static_cast<int64_t>(tempLoopInfo.n2Idx) * static_cast<int64_t>(constInfo_.gSize) *
                                static_cast<int64_t>(constInfo_.qSeqSize);
        int64_t indiceLenOffset = qFlatBase;
        int64_t indiceOutOffset = qFlatBase * static_cast<int64_t>(constInfo_.kSeqSize);

        if (!needCleanOutput_) {
            InitSparseIndicesToNegOne(gS1Idx, actMBaseSize, actS1Size, indiceOutOffset);
        }

        // DirectOutput不参与MM1槽同步，复用空闲的TopK索引区，
        // 避免混合DirectOutput/MM1场景下与后续Cube Fixpipe写resMm1Local_[0]产生冲突。
        LocalTensor<uint32_t> indicesOutLocal = globalIndexLocal_;

        // 当前处理的gS1行起始索引（全局位置）
        int64_t curGS1Idx = static_cast<int64_t>(gS1Idx) * static_cast<int64_t>(mBaseSize_);
        // 当前gS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
        int64_t curGS1ProcNum = actMBaseSize;
        // 当前AIV核处理的gS1起始索引（双核分工）
        uint32_t aivParity = static_cast<uint32_t>(blockId_) & 1U;
        int64_t curAivGS1Idx =
            curGS1Idx + static_cast<int64_t>(aivParity) * ((static_cast<int64_t>(curGS1ProcNum) + 1LL) >> 1U);
        // 当前AIV核需要处理的gS1行数量
        int64_t curAivGS1ProcNum = (aivParity == 0U) ? ((static_cast<uint64_t>(curGS1ProcNum) + 1U) >> 1U) :
                                                       (static_cast<uint64_t>(curGS1ProcNum) >> 1U);

        uint32_t curS1RealS2Len = actS2Size;
        uint32_t curAivGSize = CeilDiv(curAivGS1ProcNum, actS1Size);
        // 当前核处理的 S1 轴大小
        // 当mBaseSize小于actS1Size时，curAivS1Size为curAivGS1ProcNum
        // 当mBaseSize大于等于actS1Size时， curAivS1Size为info.actS1Size
        uint32_t curAivS1Size = Min(actS1Size, static_cast<uint32_t>(curAivGS1ProcNum));
        for (uint32_t s1Idx = 0; s1Idx < curAivS1Size; s1Idx++) {
            uint32_t nowCurAivGS1Idx = curAivGS1Idx + s1Idx;
            uint32_t globalGIdx = nowCurAivGS1Idx / actS1Size;
            uint32_t globalS1Idx = nowCurAivGS1Idx % actS1Size;
            int64_t baseLenOffset = indiceLenOffset +
                                    static_cast<int64_t>(globalGIdx) * static_cast<int64_t>(constInfo_.qSeqSize) +
                                    static_cast<int64_t>(globalS1Idx);
            int64_t baseOutOffset =
                indiceOutOffset + (static_cast<int64_t>(globalGIdx) * static_cast<int64_t>(constInfo_.qSeqSize) +
                                   static_cast<int64_t>(globalS1Idx)) *
                                      static_cast<int64_t>(constInfo_.kSeqSize);
            if (constInfo_.attenMaskFlag) {
                int32_t curS1RealS2LenTmp = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size) +
                                            static_cast<int32_t>(globalS1Idx) + 1;
                curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
            }
            WriteDirectIndices(curS1RealS2Len, baseLenOffset, baseOutOffset, curAivGSize, curAivGS1ProcNum, actS1Size,
                               s1Idx, indicesOutLocal);
        }
    }
}
} // namespace SIKernel
#endif
