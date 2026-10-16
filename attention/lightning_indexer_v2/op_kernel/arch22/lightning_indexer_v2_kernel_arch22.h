/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_kernel_arch22.h
 * \brief
 * LIV2主Kernel类：初始化Tiling/Buffer、分核(SplitCore)、主循环(ProcessMain)、基本块处理(ProcessBaseBlock)、LD归约(ProcessDecode)
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_ARCH22_H
#define LIGHTNING_INDEXER_V2_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch22.h"
#include "lightning_indexer_v2_service_vector_arch22.h"
#include "lightning_indexer_v2_service_cube_arch22.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace LIServiceVec;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
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

struct LogicalTaskCursor {
    uint32_t bN2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t actS1Size = 0U;
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U;
    bool actualSeqLenReady = false;
};

struct LogicalTaskInfo {
    uint32_t bN2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t actS1Size = 0U;
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U;
    uint32_t s2BlockNum = 0U;
};

struct TaskWaveInfo {
    bool assigned = false;
    bool hasMerge = false;
    bool isMergeLeader = false;
    uint32_t partitionNum = 1U;
    uint32_t bN2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t actS1Size = 0U;
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U;
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;
};

template <typename LIT>
class LightningIndexerV2Kernel {
public:
    __aicore__ inline LightningIndexerV2Kernel() {}
    __aicore__ inline void Init(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* w, __gm__ uint8_t* cuSeqlensQ,
                                __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK,
                                __gm__ uint8_t* cmpResidualK, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* outputIdxOffset, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
                                const LIV2TilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();

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

    LightningIndexerV2ServiceCube<LIT> matmulService;
    LightningIndexerV2ServiceVector<LIT> vectorService;

    // ================================ 常量区 ================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SPARSE_COUNT_8K = 8192;
    static constexpr uint32_t S2_BLOCKS_PER_PARTITION = 4;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // workspace double buffer
    static constexpr uint32_t WS_DOUBLE = 2;

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
    GlobalTensor<OUT_V_T> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> usedSeqLengthsKGm;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息

    GlobalTensor<int32_t> cmpResidualKGm;
    bool isCmpResidualKValid = false;

    // ================================ 类成员变量 ================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t batchInvariantMm1ResLoop = 0U; // 跨logical task和wave保持mm1ResGm双缓冲轮转

    LIV2Common::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    LIV2Common::SplitCoreInfo splitCoreInfo{};

    // ================================ Init functions ================================
    __aicore__ inline void InitTilingData(const LIV2TilingData* __restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensK,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK);
    // ================================ Split Core ================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t& coreNum, LIV2Common::SplitCoreInfo& info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline bool GetNextLogicalTask(LogicalTaskCursor& cursor, LogicalTaskInfo& taskInfo);
    __aicore__ inline uint32_t GetS2PartitionNum(uint32_t actS1Size, uint32_t s2BlockNum);
    __aicore__ inline bool BuildNextTaskWave(LogicalTaskCursor& cursor, TaskWaveInfo& waveInfo);
    // ================================ Process functions ================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBatchInvariant();
    __aicore__ inline void ProcessTaskPartition(const TaskWaveInfo& waveInfo);
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint32_t mm1ResLoop, uint64_t s2LoopIdx,
                                            LIV2Common::RunInfo& runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    __aicore__ inline void ProcessZeroLengthOutput();
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
                                       LIV2Common::RunInfo& runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitTilingData(const LIV2TilingData* __restrict tilingData)
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

    constInfo.s1BaseSize = constInfo.isSparseCountOver2K ? SPARSE_COUNT_8K / constInfo.sparseCount * 2 : 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = LIV2Common::Align(constInfo.mBaseSize, ConstInfo::BLOCK_CUBE);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ,
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
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                          bool isAccumSeq,
                                                                          GlobalTensor<uint32_t>& actualSeqLengthsGm,
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
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                                             uint32_t usedLenKDims, bool isAccumSeq,
                                                                             GlobalTensor<uint32_t>& actualSeqLengthsGm,
                                                                             GlobalTensor<uint32_t>& usedSeqLengthsKGm,
                                                                             uint32_t defaultSeqLen, uint32_t cmpRatio)
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
__aicore__ inline void LightningIndexerV2Kernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size,
                                                                          uint32_t& actS2Size, uint32_t& actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig = GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.usedLenKDims, constInfo.isAccumSeqS2,
                                       actualSeqLengthsGm, usedSeqLengthsKGm, constInfo.kSeqSize, constInfo.cmpRatio);
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
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
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;

    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum =
                constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) :
                                                        GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline LightningIndexerV2Kernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t& coreNum,
                                                                LIV2Common::SplitCoreInfo& info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2SizeOrig);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            s2Loop = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : s2BaseNum;
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = constInfo.isSparseCountOver2K ? s2BaseNum - 1 :
                                                                 s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline bool LightningIndexerV2Kernel<LIT>::GetNextLogicalTask(LogicalTaskCursor& cursor,
                                                                         LogicalTaskInfo& taskInfo)
{
    uint32_t totalBN2 = constInfo.batchSize * constInfo.kHeadNum;
    while (cursor.bN2Idx < totalBN2) {
        if (!cursor.actualSeqLenReady) {
            uint32_t bIdx = cursor.bN2Idx / constInfo.kHeadNum;
            GetS1S2ActualSeqLen(bIdx, cursor.actS1Size, cursor.actS2Size, cursor.actS2SizeOrig);
            cursor.actualSeqLenReady = true;
        }
        if (cursor.actS1Size == 0 || cursor.actS2Size == 0) {
            ++cursor.bN2Idx;
            cursor.gS1Idx = 0;
            cursor.actualSeqLenReady = false;
            continue;
        }

        uint32_t s1GBaseNum = CeilDiv(cursor.actS1Size, constInfo.s1BaseSize);
        while (cursor.gS1Idx < s1GBaseNum) {
            uint32_t gS1Idx = cursor.gS1Idx++;
            uint32_t s2BlockNum = constInfo.attenMaskFlag ?
                                      GetS2BaseBlockNumOnMask(gS1Idx, cursor.actS1Size, cursor.actS2SizeOrig) :
                                      CeilDiv(cursor.actS2Size, constInfo.s2BaseSize);
            if (s2BlockNum == 0) {
                continue;
            }
            taskInfo.bN2Idx = cursor.bN2Idx;
            taskInfo.gS1Idx = gS1Idx;
            taskInfo.actS1Size = cursor.actS1Size;
            taskInfo.actS2Size = cursor.actS2Size;
            taskInfo.actS2SizeOrig = cursor.actS2SizeOrig;
            taskInfo.s2BlockNum = s2BlockNum;
            return true;
        }
        ++cursor.bN2Idx;
        cursor.gS1Idx = 0;
        cursor.actualSeqLenReady = false;
    }
    return false;
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetS2PartitionNum(uint32_t actS1Size, uint32_t s2BlockNum)
{
    // 原LD流程的单路候选集固定为BASE_TOPK；两个最小partition不足以摊薄LD和同步开销。
    if (constInfo.isSparseCountOver2K || s2BlockNum <= 2 * S2_BLOCKS_PER_PARTITION) {
        return 1;
    }

    uint32_t s1TaskNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
    uint32_t maxS2PartitionNum = Max(1U, usedCoreNum / s1TaskNum);
    uint32_t s2PartitionNum = Max(1U, s2BlockNum / S2_BLOCKS_PER_PARTITION);
    return Min(maxS2PartitionNum, s2PartitionNum);
}

template <typename LIT>
__aicore__ inline bool LightningIndexerV2Kernel<LIT>::BuildNextTaskWave(LogicalTaskCursor& cursor,
                                                                        TaskWaveInfo& waveInfo)
{
    waveInfo.assigned = false;
    waveInfo.hasMerge = false;
    waveInfo.isMergeLeader = false;
    waveInfo.partitionNum = 1U;
    uint32_t coreStart = 0;

    while (coreStart < usedCoreNum) {
        LogicalTaskCursor nextCursor = cursor;
        LogicalTaskInfo taskInfo;
        if (!GetNextLogicalTask(nextCursor, taskInfo)) {
            break;
        }

        uint32_t partitionNum = GetS2PartitionNum(taskInfo.actS1Size, taskInfo.s2BlockNum);
        if (coreStart + partitionNum > usedCoreNum) {
            break;
        }
        if (partitionNum > 1) {
            waveInfo.hasMerge = true;
        }
        if (aiCoreIdx >= coreStart && aiCoreIdx < coreStart + partitionNum) {
            uint32_t partitionIdx = aiCoreIdx - coreStart;
            uint32_t minBlockNum = taskInfo.s2BlockNum / partitionNum;
            uint32_t extraBlockPartitionNum = taskInfo.s2BlockNum % partitionNum;
            uint32_t partitionBlockNum = minBlockNum + (partitionIdx < extraBlockPartitionNum ? 1 : 0);

            waveInfo.assigned = true;
            waveInfo.isMergeLeader = partitionIdx == 0 && partitionNum > 1;
            waveInfo.partitionNum = partitionNum;
            waveInfo.bN2Idx = taskInfo.bN2Idx;
            waveInfo.gS1Idx = taskInfo.gS1Idx;
            waveInfo.actS1Size = taskInfo.actS1Size;
            waveInfo.actS2Size = taskInfo.actS2Size;
            waveInfo.actS2SizeOrig = taskInfo.actS2SizeOrig;
            waveInfo.s2Start = partitionIdx * minBlockNum + Min(partitionIdx, extraBlockPartitionNum);
            waveInfo.s2End = waveInfo.s2Start + partitionBlockNum - 1;
        }
        coreStart += partitionNum;
        cursor = nextCursor;
    }
    return coreStart > 0;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx,
                                                                          uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_V2_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_V2_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount + // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                                           __gm__ uint8_t* weights, __gm__ uint8_t* cuSeqlensQ,
                                                           __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ,
                                                           __gm__ uint8_t* sequsedK, __gm__ uint8_t* cmpResidualK,
                                                           __gm__ uint8_t* blockTable, __gm__ uint8_t* outputIdxOffset,
                                                           __gm__ uint8_t* metadata, __gm__ uint8_t* sparseIndices,
                                                           __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
                                                           const LIV2TilingData* __restrict tiling, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK);

    if (cmpResidualK != nullptr) {
        cmpResidualKGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualK);
        isCmpResidualKValid = true;
    }
    // 计算分核
    if (!constInfo.batchInvariant) {
        SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);
    }

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S)|vec1ResGm(存LD中间结果)|vec1ParamGm(存LD参数)
    // |Core0_mm1ResDB0-Core0_mm1ResDB1-Core1_mm1ResDB0....Core23_mm1ResDB0-Core23_mm1ResDB1|Core0_vec1Res...
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.mBaseSizeAlign * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T*)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // ld流程需要ws大小: [aicnum, 2, CeilDiv(constInfo.mBaseSize, constInfo.gSize), topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    vec1ResGm.SetGlobalBuffer((__gm__ float*)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    // (aic, 8, 2, 16)
    // (aic, s1_cube, 头尾，16ele)
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t*)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ OUT_V_T*)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_T*)weights);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, weightsGm, indiceOutGm, valueOutGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T*)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    SetGS1LoopParams(tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::SetGS1LoopParams(uint32_t actS1Size, uint32_t actS2Size,
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
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcRunInfo(uint32_t loop, uint32_t mm1ResLoop,
                                                                  uint32_t s2LoopIdx, LIV2Common::RunInfo& runInfo)
{
    runInfo.loop = loop;
    runInfo.mm1ResLoop = mm1ResLoop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
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
        LIV2Common::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, LIV2Common::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx);
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx);
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
__aicore__ inline void LightningIndexerV2Kernel<LIT>::Process()
{
    if (constInfo.batchInvariant) {
        ProcessBatchInvariant();
        return;
    }
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }
    ProcessMain();
    ProcessDecode();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalQuerySize = constInfo.batchSize * constInfo.qSeqSize;
        if (constInfo.batchInvariant && constInfo.outputLayout == LI_V2_LAYOUT::TND && constInfo.actualLenQDims > 0) {
            totalQuerySize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
        }
        uint64_t totalOutputSize = totalQuerySize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            LIV2Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
            if (constInfo.returnValue) {
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

                GlobalTensor<uint32_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t*)valueOutGm.GetPhyAddr());
                GlobalTensor<uint32_t> valueOut = valueOutGmTmp[baseSize];

                AscendC::InitGlobalMemory(valueOut, dealSize, constInfo.NEG_INF_FLOAT);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessZeroLengthOutput()
{
    if ASCEND_IS_AIV {
        uint64_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
            uint32_t actS1Size, actS2Size, actS2SizeOrig;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            (void)actS2SizeOrig;
            if (actS1Size != 0 && actS2Size != 0) {
                continue;
            }

            uint64_t outputRowBase;
            uint64_t outputRowCount;
            if (constInfo.outputLayout == LI_V2_LAYOUT::TND) {
                uint64_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx);
                outputRowBase = tBase * constInfo.kHeadNum;
                outputRowCount = static_cast<uint64_t>(actS1Size) * constInfo.kHeadNum;
            } else {
                outputRowBase = static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum;
                outputRowCount = constInfo.qSeqSize * constInfo.kHeadNum;
            }
            for (uint64_t rowIdx = tmpBlockIdx; rowIdx < outputRowCount; rowIdx += aivCoreNum) {
                vectorService.CleanInvalidOutput((outputRowBase + rowIdx) * constInfo.sparseCount);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessTaskPartition(const TaskWaveInfo& waveInfo)
{
    if (!waveInfo.assigned) {
        return;
    }

    splitCoreInfo.bN2Start = waveInfo.bN2Idx;
    splitCoreInfo.bN2End = waveInfo.bN2Idx;
    splitCoreInfo.gS1Start = waveInfo.gS1Idx;
    splitCoreInfo.gS1End = waveInfo.gS1Idx;
    splitCoreInfo.s2Start = waveInfo.s2Start;
    splitCoreInfo.s2End = waveInfo.s2End;
    splitCoreInfo.isLD = waveInfo.isMergeLeader;

    GetBN2Idx(waveInfo.bN2Idx);
    SetGS1LoopParams(waveInfo.actS1Size, waveInfo.actS2Size, waveInfo.actS2SizeOrig);
    CalcS2LoopParams(waveInfo.bN2Idx, waveInfo.gS1Idx);
    LIV2Common::RunInfo runInfo;
    uint32_t loop = 0;
    for (uint32_t s2LoopIdx = waveInfo.s2Start; s2LoopIdx <= waveInfo.s2End; ++s2LoopIdx) {
        ProcessBaseBlock(loop, batchInvariantMm1ResLoop, s2LoopIdx, runInfo);
        ++loop;
        ++batchInvariantMm1ResLoop;
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessBatchInvariant()
{
    LogicalTaskCursor cursor;
    TaskWaveInfo waveInfo;
    if (!BuildNextTaskWave(cursor, waveInfo)) {
        ProcessInvalid();
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    do {
        ProcessTaskPartition(waveInfo);
        if (!waveInfo.hasMerge) {
            continue;
        }

        if ASCEND_IS_AIV {
            if (waveInfo.isMergeLeader) {
                vectorService.FreeEventID();
            }
            SyncAll();
            if (waveInfo.isMergeLeader) {
                vectorService.InitLDBuffers(pipe);
                ICachePreLoad(LD_PREFETCH_LEN);
                // BatchInvariant wave的partition拓扑固定，直接传给LD，避免从可复用workspace反推链路。
                uint32_t taskS1Start = waveInfo.gS1Idx * constInfo.s1BaseSize;
                uint32_t taskS1ProcNum = Min(constInfo.s1BaseSize, tempLoopInfo.actS1Size - taskS1Start);
                uint64_t taskOutOffset = indiceOutCoreOffset + taskS1Start * constInfo.sparseCount;
                vectorService.ProcessLD(waveInfo.partitionNum, taskS1ProcNum, taskOutOffset);
            }
            SyncAll();
            if (waveInfo.isMergeLeader) {
                pipe->Reset();
                vectorService.InitBuffers(pipe);
                vectorService.AllocEventID();
            }
            CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
            CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
            // Release AIC only after every partial TopK merge and buffer reset is complete.
            CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(SYNC_V1_C1_FLAG);
        } else {
            CrossCoreWaitFlag(constInfo.syncV1C1);
            CrossCoreWaitFlag(constInfo.syncV1C1);
            CrossCoreWaitFlag(SYNC_V1_C1_FLAG);
        }
    } while (BuildNextTaskWave(cursor, waveInfo));

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
    ProcessZeroLengthOutput();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    LIV2Common::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessBaseBlock(uint32_t loop, uint32_t mm1ResLoop,
                                                                       uint64_t s2LoopIdx, LIV2Common::RunInfo& runInfo)
{
    CalcRunInfo(loop, mm1ResLoop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        if (constInfo.batchInvariant && runInfo.isFirstS2InnerLoop) {
            matmulService.ResetLogicalTaskState();
        }
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<LIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (splitCoreInfo.isLD) {
            vectorService.ProcessLD();
        }
    }
}
} // namespace LIV2Kernel
#endif // LIGHTNING_INDEXER_V2_KERNEL_ARCH22_H
