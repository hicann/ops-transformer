/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_kernel_fast_arch22.h
 * \brief
 * Arch22 cube-fill-vector kernel using the AICPU fast metadata plan.
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_FAST_ARCH22_H
#define LIGHTNING_INDEXER_V2_KERNEL_FAST_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_fast_arch22.h"
#include "lightning_indexer_v2_service_cube_fast_arch22.h"
#include "lightning_indexer_v2_service_vector_fast_arch22.h"
#include "../lightning_indexer_v2_metadata.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace LIServiceVec;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

struct FastTempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U; // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;  // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
};

template <typename LIT>
class LightningIndexerV2KernelFast {
public:
    __aicore__ inline LightningIndexerV2KernelFast() {}
    __aicore__ inline void Init(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* w, __gm__ uint8_t* cuSeqlensQ,
                                __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK,
                                __gm__ uint8_t* cmpResidualK, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* outputIdxOffset, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
                                const LIV2TilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process()
    {
        ProcessCubeFillVector();
    }

    // ================================ 类型定义区 ================================
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    using OUT_V_T = float;
    using W_T = float;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_V2_LAYOUT K_LAYOUT_T = LIT::keyLayout;

    using MM1_OUT_T = float;

    LightningIndexerV2ServiceCubeFast<LIT> matmulService;
    LightningIndexerV2ServiceVectorFast<LIT> vectorService;

    // ================================ 常量区 ================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SPARSE_COUNT_8K = 8192;

protected:
    TPipe* pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    // ================================ Global Buffer区 ================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> usedSeqLengthsKGm;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    GlobalTensor<float> vec1ResGm;    // 存放TopK计算中间结果

    GlobalTensor<int32_t> cmpResidualKGm;
    bool isCmpResidualKValid = false;

    // ================================ 类成员变量 ================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    LIV2FastCommon::ConstInfo constInfo{};
    FastTempLoopInfo tempLoopInfo{};
    LIV2Common::SplitCoreInfo splitCoreInfo{};

    // ================================ Init functions ================================
    __aicore__ inline void InitTilingData(const LIV2TilingData* __restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensK,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK);
    // ================================ Split Core ================================
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================ Process functions ================================
    __aicore__ inline void InitCubeFillVector(__gm__ uint8_t* metadata, __gm__ uint8_t* workspace,
                                              __gm__ uint8_t* weights);
    __aicore__ inline void ProcessCubeFillVector();
    // ================================ Params Calc ================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void SetGS1LoopParams(uint32_t actS1Size, uint32_t actS2Size, uint32_t actS2SizeOrig);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t>& actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, uint32_t usedLenKDims,
                                                  bool isAccumSeq, GlobalTensor<uint32_t>& actualSeqLengthsGm,
                                                  GlobalTensor<uint32_t>& usedSeqLengthsKGm, uint32_t defaultSeqLen,
                                                  uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size, uint32_t& actS2Size,
                                               uint32_t& actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t mm1ResLoop, uint32_t s2LoopIdx,
                                       LIV2FastCommon::RunInfo& runInfo);
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::InitTilingData(const LIV2TilingData* __restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->maskMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.cmpRatio = tilingData->cmpRatio;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->topk;
    constInfo.preTokens = tilingData->preTokens;
    constInfo.nextTokens = tilingData->nextTokens;
    constInfo.returnValue = tilingData->returnValue;
    constInfo.batchInvariant = tilingData->batchInvariant != 0;

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == LI_V2_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_V2_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount <= BASE_TOPK) ? false : true;

    constInfo.s1BaseSize = constInfo.gSize == 32 ? 6U : 8U;
    constInfo.realSparseCount = constInfo.sparseCount;
    // The arch22 K input is AutoContiguous. Use the packed page stride of the
    // kernel input, rather than the stride of the original caller tensor.
    constInfo.kCacheStride = static_cast<uint64_t>(constInfo.kCacheBlockSize) * constInfo.kHeadNum * HEAD_DIM;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = LIV2Common::Align(constInfo.mBaseSize, LIV2FastCommon::ConstInfo::BLOCK_CUBE);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ,
                                                                           __gm__ uint8_t* actualSeqLengths,
                                                                           __gm__ uint8_t* sequsedQ,
                                                                           __gm__ uint8_t* sequsedK)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize + 1;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t*)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize + 1;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t*)actualSeqLengths, constInfo.actualLenDims);
    }
    if (sequsedK == nullptr) {
        constInfo.usedLenKDims = 0;
    } else {
        constInfo.usedLenKDims = constInfo.batchSize;
        usedSeqLengthsKGm.SetGlobalBuffer((__gm__ uint32_t*)sequsedK, constInfo.usedLenKDims);
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2KernelFast<LIT>::GetActualSeqLen(
    uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq, GlobalTensor<uint32_t>& actualSeqLengthsGm,
    uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq) {
        return actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx + 1);
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2KernelFast<LIT>::GetActualSeqLenKey(
    uint32_t bIdx, uint32_t actualLenDims, uint32_t usedLenKDims, bool isAccumSeq,
    GlobalTensor<uint32_t>& actualSeqLengthsGm, GlobalTensor<uint32_t>& usedSeqLengthsKGm, uint32_t defaultSeqLen,
    uint32_t cmpRatio)
{
    int32_t residualK = isCmpResidualKValid ? cmpResidualKGm.GetValue(bIdx) : 0;
    if (actualLenDims == 0 && usedLenKDims == 0) {
        // BSND
        uint32_t result = defaultSeqLen * cmpRatio + residualK;
        return result;
    }
    if (isAccumSeq) {
        // TND
        return (actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + residualK;
    }
    // PA_BBND
    return usedSeqLengthsKGm.GetValue(bIdx) * cmpRatio + residualK;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size,
                                                                              uint32_t& actS2Size,
                                                                              uint32_t& actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig = GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.usedLenKDims, constInfo.isAccumSeqS2,
                                       actualSeqLengthsGm, usedSeqLengthsKGm, constInfo.kSeqSize, constInfo.cmpRatio);
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2KernelFast<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx,
                                                                                      uint32_t actS1Size,
                                                                                      uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    // 压缩前的validS2LenBase
    int32_t validS2Len =
        (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) /
        static_cast<int32_t>(constInfo.cmpRatio);

    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights, __gm__ uint8_t* cuSeqlensQ,
    __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK, __gm__ uint8_t* cmpResidualK,
    __gm__ uint8_t* blockTable, __gm__ uint8_t* outputIdxOffset, __gm__ uint8_t* metadata,
    __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
    const LIV2TilingData* __restrict tiling, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // 两个 AIV 对应一个 AIC 核组，通过 AIV 核号计算核组编号
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // AIC 核号直接作为核组编号
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK);

    if (cmpResidualK != nullptr) {
        cmpResidualKGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualK);
        isCmpResidualKValid = true;
    }
    pipe = tPipe;
    InitCubeFillVector(metadata, workspace, weights);
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
    if ASCEND_IS_AIV {
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        vectorService.InitParams(constInfo, tiling);
        vectorService.InitDiagVecGlobalTensor(mm1ResGm, indiceOutGm);
        vectorService.InitCfvTopkState(vec1ResGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
        keyGm.SetGlobalBuffer((__gm__ K_T*)key);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
        matmulService.InitDiagGlobalTensor(weightsGm, mm1ResGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    SetGS1LoopParams(tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::SetGS1LoopParams(uint32_t actS1Size, uint32_t actS2Size,
                                                                           uint32_t actS2SizeOrig)
{
    tempLoopInfo.actS1Size = actS1Size;
    tempLoopInfo.actS2Size = actS2Size;
    tempLoopInfo.actS2SizeOrig = actS2SizeOrig;
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (tempLoopInfo.bN2Idx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == LI_V2_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::CalcRunInfo(uint32_t loop, uint32_t mm1ResLoop,
                                                                      uint32_t s2LoopIdx,
                                                                      LIV2FastCommon::RunInfo& runInfo)
{
    runInfo.loop = loop;
    runInfo.mm1ResLoop = mm1ResLoop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        LIV2Common::Align(runInfo.actualSingleProcessSInnerSize, LIV2FastCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx);
            if constexpr (K_LAYOUT_T == LI_V2_LAYOUT::TND) {
                actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx);
            } else {
                actualSeqKPrefixSum = 0; // PA 的地址由 block table 决定，不读取可选 cuSeqlensK。
            }
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D (Q数据在GM中按原始mBaseSize排列，不使用对齐后的大小)
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset =
        keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::InitCubeFillVector(__gm__ uint8_t* metadata,
                                                                             __gm__ uint8_t* workspace,
                                                                             __gm__ uint8_t* weights)
{
    GlobalTensor<uint32_t> metadataGm;
    metadataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
    usedCoreNum = metadataGm.GetValue(optiling::LI_V2_FAST_CORE_NUM_INDEX);
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }
    const uint32_t base = optiling::LI_V2_FAST_CORE_INFO_INDEX + aiCoreIdx * optiling::LI_V2_FAST_CORE_INFO_SIZE;
    splitCoreInfo.bN2Start = metadataGm.GetValue(base + optiling::LI_V2_FAST_BN2_START_INDEX);
    splitCoreInfo.gS1Start = metadataGm.GetValue(base + optiling::LI_V2_FAST_M_START_INDEX);
    splitCoreInfo.s2Start = 0;
    splitCoreInfo.bN2End = metadataGm.GetValue(base + optiling::LI_V2_FAST_BN2_END_INDEX);
    splitCoreInfo.gS1End = metadataGm.GetValue(base + optiling::LI_V2_FAST_M_END_INDEX);

    constexpr uint64_t SCORE_STORAGE_COLS = 16U;
    constexpr uint64_t CANDIDATE_GROUP_NUM = 3U;
    constexpr uint64_t MAX_QUERY_ROWS_PER_AIV = 4U;
    constexpr uint64_t CANDIDATE_FIELD_NUM = 2U;

    // 每个候选按 value/index 两个32位字段预留空间
    constexpr uint64_t SCORE_BYTES =
        LIV2FastCommon::ConstInfo::SCORE_RING_DEPTH * S2_BASE_SIZE * SCORE_STORAGE_COLS * sizeof(float);
    constexpr uint64_t STATE_BYTES =
        CANDIDATE_GROUP_NUM * MAX_QUERY_ROWS_PER_AIV * BASE_TOPK * CANDIDATE_FIELD_NUM * sizeof(float);
    mm1ResGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + aiCoreIdx * SCORE_BYTES));
    vec1ResGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + GetBlockNum() * SCORE_BYTES +
                                                              (aiCoreIdx * 2 + (tmpBlockIdx & 1U)) * STATE_BYTES));
    weightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2KernelFast<LIT>::ProcessCubeFillVector()
{
    if (aiCoreIdx >= usedCoreNum) {
        return;
    }

    bool cubeEventsInitialized = false;
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        for (uint32_t slot = 0; slot < LIV2FastCommon::ConstInfo::SCORE_RING_DEPTH; ++slot) {
            CrossCoreSetFlag<LIV2FastCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(slot);
        }
    }
    LIV2FastCommon::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            if ASCEND_IS_AIV {
                CalcRunInfo(gloop, gloop, splitCoreInfo.s2Start, runInfo);
                // 长序列路径一次消费本 S1 tile 的全部可见 S2，成功后跳过逐块流程。
                if (vectorService.ProcessSequenceTopk(runInfo)) {
                    gloop += tempLoopInfo.s2LoopEnd + 1;
                    splitCoreInfo.s2Start = 0;
                    continue;
                }
            }
            for (uint32_t s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                CalcRunInfo(gloop, gloop, s2LoopIdx, runInfo);
                // ring slot 循环复用，跨核 flag 防止 AIC 写入覆盖 AIV 尚未消费的数据。
                uint32_t scoreEvent = runInfo.loop % LIV2FastCommon::ConstInfo::SCORE_RING_DEPTH;
                if ASCEND_IS_AIC {
                    if (!cubeEventsInitialized) {
                        matmulService.AllocEventID();
                        cubeEventsInitialized = true;
                    }
                    CrossCoreWaitFlag(scoreEvent);
                    matmulService.ComputeCubeStages(runInfo);
                    CrossCoreSetFlag<LIV2FastCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(scoreEvent);
                } else {
                    CrossCoreWaitFlag(scoreEvent);
                    vectorService.ProcessTopkOnly(runInfo);
                    CrossCoreSetFlag<LIV2FastCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(scoreEvent);
                }
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        splitCoreInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        if (cubeEventsInitialized) {
            matmulService.FreeEventID();
        }
        for (uint32_t slot = 0; slot < LIV2FastCommon::ConstInfo::SCORE_RING_DEPTH; ++slot) {
            CrossCoreWaitFlag(slot);
        }
    }
}
} // namespace LIV2Kernel
#endif // LIGHTNING_INDEXER_V2_KERNEL_FAST_ARCH22_H
