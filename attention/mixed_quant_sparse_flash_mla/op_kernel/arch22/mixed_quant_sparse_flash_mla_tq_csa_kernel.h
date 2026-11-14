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
 * \file mixed_quant_sparse_flash_mla_tq_csa_kernel.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_TQ_CSA_KERNEL_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_TQ_CSA_KERNEL_H

#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "mixed_quant_sparse_flash_mla_tq_common.h"
#include "mixed_quant_sparse_flash_mla_tiling_data_arch22.h"
#include "mixed_quant_sparse_flash_mla_tq_csa_block_cube.h"
#include "mixed_quant_sparse_flash_mla_tq_csa_block_vector.h"
#include "../mixed_quant_sparse_flash_mla_metadata.h"

namespace SASKernel {
using namespace matmul;
using namespace optiling;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bn2IdxInCurCore = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1

    int32_t actS1Size = 0; // TND场景下当前Batch循环处理的S1轴的大小
    int32_t actOriS2Size = 0;
    int32_t actCmpS2Size = 0;

    bool curActSeqLenIsZero = false;

    uint32_t tndCoreStartKVSplitPos = 0;
    bool tndIsS2SplitCore = false;
    uint32_t gS1Idx = 0U;
    uint32_t s1StartIdx = 0;
    uint32_t s1EndIdx = 0;
    uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小
    uint32_t cmpLoopTimes = 0;
    uint32_t oriLoopTimes = 0;

    // sparsemode = 4
    int32_t oriMaskRight = 0;
    int32_t oriMaskLeft = 0;

    // sparsemode = 3
    int32_t cmpMaskRight = 0;

    uint64_t actualSeqQPrefixSum = 0;
    uint64_t actualSeqKVPrefixSum = 0;
    uint64_t actualSeqCmpKVPrefixSum = 0;
};

template <typename SAST>
class MixedQuantSparseFlashMlaTqCsaKernel {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SAST::queryType;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using SINKS_T = float;
    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline MixedQuantSparseFlashMlaTqCsaKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKV,
                                __gm__ uint8_t* cuSeqlensCmpKV, __gm__ uint8_t* seqUsedQ, __gm__ uint8_t* seqUsedOriKV,
                                __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKV,
                                __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* workspace,
                                const optiling::MixedQuantSparseFlashMlaTqTilingData* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t SAS_PRELOAD_TASK_CACHE_SIZE = 3;
    static constexpr uint32_t MERGE_CACHE_GM_BUF_NUM = 3;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_INT8_PLAN_READY = 10;
    static constexpr uint32_t SYNC_INT8_EMPTY_PLAN_READ = 11;
    static constexpr uint32_t SYNC_INT8_FAST_DRAINED = 12;

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const optiling::MixedQuantSparseFlashMlaTqTilingData* __restrict tilingData = nullptr;

    TPipe* pipe = nullptr;
    GlobalTensor<uint32_t> metadataGm;
    uint64_t mSizeVStart = 0ULL;
    uint64_t topKBaseOffset = 0ULL;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorCmpBCoreOffset = 0ULL;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;

    ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    Int8CsaInput<SAST> int8Input;
    RunInfo int8NextInfo{};
    // Existing 3*512 decoded GM rows become ori512 + cmp512 + cmp512.
    // Four query credits protect all live suffix windows of the append stream.
    // Batch/head/query changes cold-append, never reset the ring cursor.
    uint32_t oriCacheHead = 0U;
    uint32_t oriCacheLastBatch = 0U;
    uint32_t oriCacheLastHead = 0U;
    uint32_t oriCacheLastQuery = 0U;
    int64_t oriCacheLastBegin = 0;
    int64_t oriCacheLastEnd = 0;
    bool oriCacheValid = false;
    __aicore__ inline void PlanOriReuse(RunInfo& info, int64_t begin, int64_t end);

    KvQuantSparseFlashMlaCsaBlockCube<SAST> cubeBlock;
    KvQuantSparseFlashMlaCsaBlockVector<SAST> vectorBlock;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> cmpKvGm;
    GlobalTensor<uint8_t> cmpTqGm;
    GlobalTensor<SINKS_T> sinksGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;

    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<KV_T> kvMergeGm_;

    GlobalTensor<UPDATE_T> vec2ResGm;

    GlobalTensor<T> accumOutGm;

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengthsKV);
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengthsKV,
                                            __gm__ uint8_t* actualSeqLengthsCmpKV);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void ProcessInt8Balance();
    __aicore__ inline __attribute__((always_inline)) void ProcessInt8GenericTail(uint32_t startBatch,
                                                                                 int64_t startQuery);

    template <bool INT8_FAST = true>
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint32_t cmpLoop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE]);
    // ================================Offset Calc=====================================
    __aicore__ inline void GetSparseActualSeqLen();
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void CalcParams(uint32_t loop, uint32_t cmpLoop, uint64_t s2Start, uint32_t s2LoopIdx,
                                      RunInfo& info);
    __aicore__ inline int32_t GetActualSeqLenQ(uint32_t bIdx);
    __aicore__ inline int32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t& bIdx, uint32_t& n2Idx);
    // ================================Mm1==============================================
    template <bool INT8_FAST = true>
    __aicore__ inline void ComputeMm1(const RunInfo& info);
    // ================================Mm2==============================================
    template <bool INT8_FAST = true>
    __aicore__ inline void ComputeMm2(const RunInfo& info);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitTilingData()
{
    // singleCoreParams
    // singleCoreTensorSize
    constInfo.mmResUbSize = tilingData->tqBaseParams.mmResUbSize;
    constInfo.bmm2ResUbSize = tilingData->tqBaseParams.bmm2ResUbSize;

    // baseParams
    constInfo.batchSize = tilingData->tqBaseParams.batchSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->tqBaseParams.nNumOfQInOneGroup;
    constInfo.kvSeqSize = tilingData->tqBaseParams.kvSeqSize;
    constInfo.qSeqSize = tilingData->tqBaseParams.qSeqSize;
    constInfo.oriMaxBlockNumPerBatch = tilingData->tqBaseParams.oriMaxBlockNumPerBatch;
    constInfo.cmpMaxBlockNumPerBatch = tilingData->tqCmpParams.cmpMaxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->tqBaseParams.paBlockSize;
    constInfo.paOriBlockSize = tilingData->tqBaseParams.oriBlockSize;
    constInfo.paCmpBlockSize = tilingData->tqBaseParams.cmpBlockSize;
    constInfo.outputLayout = static_cast<SAS_LAYOUT>(tilingData->tqBaseParams.outputLayout);
    constInfo.kvHeadNum = kvHeadNum;
    constInfo.headDim = headDim;
    constInfo.oriMaskMode = tilingData->tqBaseParams.oriMaskMode;
    constInfo.oriKvStride0 = tilingData->tqBaseParams.oriKvStride0;
    constInfo.oriWinLeft = tilingData->tqBaseParams.oriWinLeft;
    constInfo.oriWinRight = tilingData->tqBaseParams.oriWinRight;

    constInfo.mqActualLenDimsQ = tilingData->tqBaseParams.actualLenDimsQ;
    constInfo.mqActualLenDimsKV = tilingData->tqBaseParams.actualLenDimsKV;
    constInfo.returnSoftmaxLse = tilingData->tqBaseParams.returnSoftmaxLse;
    constInfo.kvQuantMode = tilingData->tqBaseParams.kvQuantMode;
    // innerSplitParams
    constInfo.mBaseSize = constInfo.gSize;
    constInfo.s2BaseSize = tilingData->tqBaseParams.s2BaseSize;

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE;
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;

    // cmp
    constInfo.cmpRatio = tilingData->tqCmpParams.cmpRatio;
    constInfo.sparseBlockCount = tilingData->tqCmpParams.sparseBlockCount;
    constInfo.sparseBlockSize = 1; // sparseBlockSize 固定为1
    constInfo.cmpMaskMode = tilingData->tqCmpParams.cmpMaskMode;
    constInfo.cmpKvStride0 = tilingData->tqCmpParams.cmpKvStride0;
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorBlock.InitBuffers(pipe);
    } else {
        cubeBlock.InitBuffers(pipe);
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitActualSeqLen(__gm__ uint8_t* actualSeqLengthsQ,
                                                                                   __gm__ uint8_t* actualSeqLengthsKV)
{
    if (constInfo.mqActualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t*)actualSeqLengthsKV, constInfo.mqActualLenDimsKV);
    }
    if (constInfo.mqActualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t*)actualSeqLengthsQ, constInfo.mqActualLenDimsQ);
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitActualSeqLen(
    __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* actualSeqLengthsCmpKV)
{
    if (constInfo.mqActualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t*)actualSeqLengthsKV, constInfo.mqActualLenDimsKV);
        actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t*)actualSeqLengthsCmpKV, constInfo.mqActualLenDimsKV);
    }
    if (constInfo.mqActualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t*)actualSeqLengthsQ, constInfo.mqActualLenDimsQ);
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx,
                                                                                    uint32_t n2Idx)
{
    if (constInfo.outputLayout == SAS_LAYOUT::TND) {
        if (tempLoopInfo.actS1Size == 0) {
            return;
        }
        uint32_t tBase = actualSeqLengthsQGm.GetValue(bIdx);
        uint32_t s1Count = tempLoopInfo.s1EndIdx - s1Idx + 1;

        uint64_t attenOutOffset = (tBase + s1Idx) * kvHeadNum * constInfo.gSize * headDim + // T轴、s1轴偏移
                                  n2Idx * constInfo.gSize * headDim;                        // N2轴偏移
        uint64_t lseOffset = (tBase + s1Idx) * constInfo.gSize +                            // T轴、s1轴偏移
                             n2Idx * constInfo.qSeqSize * constInfo.gSize;                  // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * constInfo.gSize * headDim, 0);
        if (constInfo.returnSoftmaxLse) {
            matmul::InitOutput<T>(softmaxLseGm[lseOffset], s1Count * constInfo.gSize, 0);
        }
    } else if (constInfo.outputLayout == SAS_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * kvHeadNum * constInfo.gSize * headDim +
                                  s1Idx * kvHeadNum * constInfo.gSize * headDim +                // B轴、S1轴偏移
                                  n2Idx * constInfo.gSize * headDim;                             // N2轴偏移
        uint64_t lseOffset = bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.gSize + // B轴偏移
                             n2Idx * constInfo.qSeqSize * constInfo.gSize +                      // N2轴偏移
                             s1Idx * constInfo.gSize;                                            // S1轴偏移
        uint32_t s1Count = tempLoopInfo.s1EndIdx - s1Idx + 1;
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * constInfo.gSize * headDim, 0);
        if (constInfo.returnSoftmaxLse) {
            matmul::InitOutput<T>(softmaxLseGm[lseOffset], s1Count * constInfo.gSize, 0);
        }
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitOutputSingleCore()
{
    uint32_t coreNum = GetBlockNum();
    if (coreNum != 0) {
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum); // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
        SyncAll();
    }
}

template <typename SAST>
__aicore__ inline int32_t MixedQuantSparseFlashMlaTqCsaKernel<SAST>::GetActualSeqLenQ(uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        int32_t actualSeqQPrefixSum = actualSeqLengthsQGm.GetValue(bIdx);
        int32_t actualSeqQNextSum = actualSeqLengthsQGm.GetValue(bIdx + 1);
        tempLoopInfo.actualSeqQPrefixSum = static_cast<uint64_t>(actualSeqQPrefixSum);
        return actualSeqQNextSum - actualSeqQPrefixSum;
    } else {
        tempLoopInfo.actualSeqQPrefixSum = static_cast<uint64_t>(bIdx * constInfo.qSeqSize);
        if (constInfo.mqActualLenDimsQ == 0) {
            return static_cast<int32_t>(constInfo.qSeqSize);
        } else {
            return actualSeqLengthsQGm.GetValue(bIdx);
        }
    }
}

template <typename SAST>
__aicore__ inline int32_t MixedQuantSparseFlashMlaTqCsaKernel<SAST>::GetActualSeqLenKV(uint32_t bIdx)
{
    if constexpr (PAGE_ATTENTION) {
        tempLoopInfo.actualSeqKVPrefixSum = static_cast<uint64_t>(bIdx * constInfo.kvSeqSize);
        if (constInfo.mqActualLenDimsKV == 0) {
            return static_cast<int32_t>(constInfo.kvSeqSize);
        }
        return actualSeqLengthsKVGm.GetValue(bIdx);
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND) {
        return static_cast<int32_t>(constInfo.kvSeqSize);
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        int32_t actualSeqKVPrefixSum = actualSeqLengthsKVGm.GetValue(bIdx);
        int32_t actualSeqKVNextSum = actualSeqLengthsKVGm.GetValue(bIdx + 1);
        tempLoopInfo.actualSeqCmpKVPrefixSum = actualSeqLengthsCmpKVGm.GetValue(bIdx);
        tempLoopInfo.actualSeqKVPrefixSum = actualSeqKVPrefixSum;
        return actualSeqKVNextSum - actualSeqKVPrefixSum;
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::GetSparseActualSeqLen()
{
    // 行无效通过ori部分判断, ori部分如果有行无效那么ori和cmp都有
    if (static_cast<int32_t>(tempLoopInfo.s1EndIdx) < -(tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size)) {
        tempLoopInfo.actOriS2Size = 0;
        tempLoopInfo.actCmpS2Size = 0;
        return;
    }

    // 对于cmp部分还有top k, tempLoopInfo.actS2Size只针对cmp
    int32_t thresHold = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;
    tempLoopInfo.actCmpS2Size = Min(constInfo.sparseBlockCount * constInfo.sparseBlockSize, thresHold);
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.actCmpS2Size == 0 && tempLoopInfo.actOriS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices,
    __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKV, __gm__ uint8_t* cuSeqlensCmpKV,
    __gm__ uint8_t* seqUsedQ, __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKV,
    __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    const optiling::MixedQuantSparseFlashMlaTqTilingData* __restrict tiling, __gm__ uint8_t* gmTiling, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;
    InitTilingData();
    if constexpr (SAST::int8Kv) {
        int8Input.Init(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, cuSeqlensOriKV,
                       cuSeqlensCmpKV, seqUsedQ, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV, cmpTopkLength,
                       tiling->int8Params);
    }
    if (KV_LAYOUT_T == SAS_LAYOUT::TND && LAYOUT_T == SAS_LAYOUT::TND) {
        InitActualSeqLen(cuSeqlensQ, cuSeqlensOriKV, cuSeqlensCmpKV);
    } else if (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        InitActualSeqLen(seqUsedQ, cuSeqlensOriKV, cuSeqlensCmpKV);
    } else if ((PAGE_ATTENTION || KV_LAYOUT_T == SAS_LAYOUT::BSND) && LAYOUT_T == SAS_LAYOUT::TND) {
        InitActualSeqLen(cuSeqlensQ, seqUsedOriKV);
    } else if ((PAGE_ATTENTION || KV_LAYOUT_T == SAS_LAYOUT::BSND)) {
        InitActualSeqLen(seqUsedQ, seqUsedOriKV);
    }

    metadataGm.SetGlobalBuffer((__gm__ uint32_t*)metadata);
    InitCalcParamsEach();

    pipe = tPipe;
    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
    oriKvGm.SetGlobalBuffer((__gm__ KV_T*)oriKV);
    if (constInfo.kvQuantMode == 3) {
        cmpTqGm.SetGlobalBuffer((__gm__ uint8_t*)cmpKV);
    } else {
        cmpKvGm.SetGlobalBuffer((__gm__ KV_T*)cmpKV);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ SINKS_T*)sinks);
    }

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
    softmaxLseGm.SetGlobalBuffer((__gm__ T*)softmaxLse);

    if ASCEND_IS_AIV {
        if (LAYOUT_T != SAS_LAYOUT::TND) {
            if (constInfo.needInit) {
                InitOutputSingleCore();
            }
        }
    }

    if constexpr (PAGE_ATTENTION) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)oriBlockTable);
        cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t*)cmpBlockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t*)cmpSparseIndices);

    // workspace 内存排布
    // |Q--|mm1ResGm|vec1ResGm|mm2ResGm|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T*)(workspace + offset +
                            aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ Q_T*)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T*)(workspace + offset +
                            aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    vec2ResGm.SetGlobalBuffer(
        (__gm__ T*)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T);

    uint64_t mergeCacheCoreSize =
        static_cast<uint64_t>(constInfo.s2BaseSize) * constInfo.headDim * MERGE_CACHE_GM_BUF_NUM;
    kvMergeGm_.SetGlobalBuffer((__gm__ KV_T*)(workspace + offset + aiCoreIdx * mergeCacheCoreSize * sizeof(KV_T)));
    offset += static_cast<uint64_t>(tilingData->tqBaseParams.usedCoreNum) * mergeCacheCoreSize * sizeof(KV_T);

    if ASCEND_IS_AIV {
        vectorBlock.InitParams(constInfo, tilingData);
        if constexpr (SAST::int8Kv) {
            vectorBlock.InitInt8Input(int8Input);
        }
        vectorBlock.InitVec0GlobalTensor(kvMergeGm_, oriKvGm, cmpKvGm, cmpTqGm, oriBlockTableGm, cmpBlockTableGm);
        vectorBlock.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm, actualSeqLengthsKVGm, topKGm,
                                         sinksGm, softmaxLseGm);
        vectorBlock.InitVec2GlobalTensor(accumOutGm, vec2ResGm, mm2ResGm, attentionOutGm);
    }

    if ASCEND_IS_AIC {
        cubeBlock.InitParams(constInfo);
        cubeBlock.InitMm1GlobalTensor(queryGm, oriKvGm, cmpKvGm, mm1ResGm);
        cubeBlock.InitMm2GlobalTensor(vec1ResGm, mm2ResGm, attentionOutGm);
        cubeBlock.InitPageAttentionInfo(oriKvGm, kvMergeGm_, oriBlockTableGm, cmpBlockTableGm);
    }
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::InitCalcParamsEach()
{
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_BN2_START_INDEX, false));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_M_START_INDEX, false));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_S2_START_INDEX, false));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_BN2_END_INDEX, false));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_M_END_INDEX, false));
    constInfo.s2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_S2_END_INDEX, false));
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::CalcParams(uint32_t loop, uint32_t cmpLoop,
                                                                             uint64_t s2Start, uint32_t s2LoopIdx,
                                                                             RunInfo& info)
{
    if constexpr (SAST::int8Kv) {
        info = int8NextInfo;
        info.loop = loop;
        info.cmpLoop = loop; // Both ori/cmp use the three-slot floating KV buffer.
        return;
    }
    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;
    info.loop = loop;
    info.cmpLoop = cmpLoop;
    info.bIdx = tempLoopInfo.bIdx;
    info.n2IdxReal = tempLoopInfo.n2Idx;

    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s1Idx = tempLoopInfo.gS1Idx / constInfo.gSize;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;
    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;
    info.actS1Size = tempLoopInfo.actS1Size;
    info.actOriS2Size = tempLoopInfo.actOriS2Size;

    // M方向的尾块
    info.actMBaseSize = tempLoopInfo.mBasicSizeTail;

    if ASCEND_IS_AIV {
        info.mSize = info.actMBaseSize;
        info.mSizeV = (info.mSize <= 16) ? info.mSize : ((CeilDiv(info.mSize, 16) + 1) / 2 * 16);
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    info.isFirstSInnerLoop = s2LoopIdx == s2Start;
    if (info.isFirstSInnerLoop) {
        tempLoopInfo.bn2IdxInCurCore++;
    }
    info.isLastS2Loop = (s2LoopIdx == (tempLoopInfo.s2LoopTimes - 1));
    info.bn2IdxInCurCore = tempLoopInfo.bn2IdxInCurCore - 1;

    uint64_t tndBIdxOffsetForQ = tempLoopInfo.actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
    uint64_t tndBIdxOffsetForKV = tempLoopInfo.actualSeqKVPrefixSum * constInfo.kvHeadNum * constInfo.headDim;
    uint64_t tndBIdxOffsetForCmpKV = tempLoopInfo.actualSeqCmpKVPrefixSum * constInfo.kvHeadNum * constInfo.headDim;

    if (info.isFirstSInnerLoop) {
        tensorACoreOffset = tndBIdxOffsetForQ + info.gS1Idx * constInfo.headDim;
        tensorBCoreOffset = tndBIdxOffsetForKV + info.n2Idx * constInfo.headDim; // 当前为PA场景，该变量失效
        tensorCmpBCoreOffset = tndBIdxOffsetForCmpKV + info.n2Idx * constInfo.headDim;
        if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) { // B,S1,N2 K
            topKBaseOffset = (info.bIdx * constInfo.qSeqSize + tempLoopInfo.s1StartIdx) * constInfo.kvHeadNum *
                                 constInfo.sparseBlockCount +
                             info.n2Idx * constInfo.sparseBlockCount;
        } else if (LAYOUT_T == SAS_LAYOUT::TND) { // T N2 K
            topKBaseOffset = (tempLoopInfo.actualSeqQPrefixSum + tempLoopInfo.s1StartIdx) * constInfo.kvHeadNum *
                                 constInfo.sparseBlockCount +
                             info.n2Idx * constInfo.sparseBlockCount;
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.tensorBOffset = tensorBCoreOffset;
    info.tensorCmpBOffset = tensorCmpBCoreOffset;
    info.attenOutOffset = tensorACoreOffset;
    info.topKBaseOffset = topKBaseOffset;

    if (s2LoopIdx < tempLoopInfo.oriLoopTimes) {
        // S2首次循环只能在ori_kv
        info.isOriOnly = true;
        info.relativeS2Idx = 0;
        uint64_t s2Offset = info.s2Idx * constInfo.s2BaseSize;
        if (s2LoopIdx + 1 == tempLoopInfo.oriLoopTimes) {
            info.actualSingleProcessSInnerSize = (tempLoopInfo.oriMaskRight - tempLoopInfo.oriMaskLeft + 1) - s2Offset;
        } else {
            info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
        }
        info.s2StartPoint = tempLoopInfo.oriMaskLeft;
        info.cmpS2IdLimit = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;
    } else {
        info.isOriOnly = false;
        info.relativeS2Idx = info.s2Idx - tempLoopInfo.oriLoopTimes;
        uint64_t s2Offset = info.relativeS2Idx * constInfo.s2BaseSize;
        if (info.relativeS2Idx + 1 == tempLoopInfo.cmpLoopTimes) {
            info.actualSingleProcessSInnerSize = tempLoopInfo.actCmpS2Size - s2Offset;
        } else {
            info.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
        }
        info.s2StartPoint = 0;
        info.cmpS2IdLimit = (tempLoopInfo.cmpMaskRight + tempLoopInfo.s1EndIdx + 1) / constInfo.cmpRatio;
    }

    info.actualSingleProcessSInnerSizeAlign =
        SASAlign(info.actualSingleProcessSInnerSize, KvQuantSparseFlashMlaCsaBlockVector<SAST>::BYTE_BLOCK);
    if (info.isOriOnly) {
        info.v0S2Start = 0;
        info.v0S2DealSize = 0;
    } else {
        info.v0S2Start = info.relativeS2Idx * constInfo.s2BaseSize;
        info.v0S2DealSize = constInfo.kvQuantMode == 3 ? info.actualSingleProcessSInnerSize : constInfo.s2BaseSize;
    }
}

template <typename SAST>
template <bool INT8_FAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::ComputeMm1(const RunInfo& info)
{
    uint32_t nBufferLoopTimes = CeilDiv(info.actMBaseSize, constInfo.nBufferMBaseSize);
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        cubeBlock.template ComputeMm1<INT8_FAST>(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename SAST>
template <bool INT8_FAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::ComputeMm2(const RunInfo& info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        cubeBlock.template ComputeMm2<INT8_FAST>(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::Process()
{
    if constexpr (SAST::int8Kv) {
        if ASCEND_IS_AIV {
            vectorBlock.InitInt8Outputs();
        }
        SyncAll<false>();
    }
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, FA_CORE_ENABLE_INDEX, false));
    if (hasLoad == 0) {
        return;
    }
    if ASCEND_IS_AIV {
        vectorBlock.AllocEventID();
        vectorBlock.InitSoftmaxDefaultBuffer();
    } else {
        cubeBlock.AllocEventID();
    }
    ProcessBalance();
    if ASCEND_IS_AIV {
        vectorBlock.FreeEventID();
    } else {
        cubeBlock.FreeEventID();
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::GetBN2Idx(uint32_t bN2Idx, uint32_t& bIdx,
                                                                            uint32_t& n2Idx)
{
    bIdx = bN2Idx / kvHeadNum;
    n2Idx = bN2Idx % kvHeadNum;
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::PlanOriReuse(RunInfo& info, int64_t begin,
                                                                               int64_t end)
{
    uint32_t reuse = 0U;
    if (oriCacheValid && info.bIdx == oriCacheLastBatch && info.n2IdxReal == oriCacheLastHead &&
        info.s1Idx == oriCacheLastQuery + 1U && begin >= oriCacheLastBegin && begin < oriCacheLastEnd &&
        end >= oriCacheLastEnd && end - begin == 128 && oriCacheLastEnd - oriCacheLastBegin == 128) {
        reuse = static_cast<uint32_t>(oriCacheLastEnd - begin);
    }
    info.oriReuseRows = reuse;
    info.oriCacheWriteRow = oriCacheHead;
    info.oriCacheStartRow = (oriCacheHead + 512U - reuse) & 511U;
    oriCacheHead = (oriCacheHead + info.actualSingleProcessSInnerSize - reuse) & 511U;
    oriCacheLastBatch = info.bIdx;
    oriCacheLastHead = info.n2IdxReal;
    oriCacheLastQuery = info.s1Idx;
    oriCacheLastBegin = begin;
    oriCacheLastEnd = end;
    oriCacheValid = true;
}

template <typename SAST>
__aicore__ inline __attribute__((always_inline)) void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::ProcessInt8GenericTail(
    uint32_t startBatch, int64_t startQuery)
{
    int8NextInfo = {};
    RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE]{};
    uint32_t loop = 0;
    if ASCEND_IS_AIC {
        for (uint32_t i = 0; i < 4; ++i) {
            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        }
    }
    uint32_t endBatch = constInfo.bN2End + (constInfo.gS1End != 0 ? 1U : 0U);
    for (uint32_t b = startBatch; b < endBatch && b < int8Input.params.batch; ++b) {
        int64_t qEnd = Min(int8Input.QLength(b), int8Input.QStorage(b));
        if (b == constInfo.bN2End && constInfo.gS1End != 0) {
            qEnd = Min(qEnd, int64_t(constInfo.gS1End));
        }
        for (int64_t s = b == startBatch ? startQuery : 0; s < qEnd; ++s) {
            int8Input.BeginQuery(b, s);
            if ASCEND_IS_AIV {
                vectorBlock.SetInt8SelectionPrepared(false);
            }
            // Bounded address acceleration is optional. Every unsupported
            // resource/layout case keeps the original scalar selection path.
            const bool canPrepare = !FLASH_DECODE && constInfo.mmResUbSize >= 8U && int8Input.CanPrepareQuery();
            if (canPrepare) {
                uint32_t oriRows = 0U, cmpRows = 0U;
                // V1(L-2) has already consumed MM1's head in iteration L-1.
                // Only AIV0 publishes one32B plan before QK(L) overwrites it.
                const uint64_t planOffset = static_cast<uint64_t>(loop % dbWorkspaceRatio) * constInfo.mmResUbSize;
                if ASCEND_IS_AIV {
                    vectorBlock.PrepareInt8SelectionGeneric(int8Input, planOffset, oriRows, cmpRows);
                    CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(SYNC_INT8_PLAN_READY);
                    // Empty queries do not advance loop or consume a V0 credit.
                    // Wait for AIC to read the reused plan location first.
                    if (oriRows + cmpRows == 0U) {
                        CrossCoreWaitFlag(SYNC_INT8_EMPTY_PLAN_READ);
                    }
                } else {
                    CrossCoreWaitFlag(SYNC_INT8_PLAN_READY);
                    GlobalTensor<uint32_t> plans;
                    plans.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(mm1ResGm.GetPhyAddr(0)),
                                          dbWorkspaceRatio * constInfo.mmResUbSize);
                    auto plan = plans[planOffset];
                    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(plan);
                    oriRows = plan.GetValue(0);
                    cmpRows = plan.GetValue(1);
                    if (oriRows + cmpRows == 0U) {
                        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(SYNC_INT8_EMPTY_PLAN_READ);
                    }
                }
                int8Input.UsePreparedSelection(oriRows, cmpRows);
            }
            RunInfo current{}, next{};
            bool valid = int8Input.NextTile(current, constInfo.s2BaseSize);
            while (valid) {
                // One-tile lookahead preserves the existing online-softmax
                // first/last semantics even when selected indices are invalid.
                bool hasNext = int8Input.NextTile(next, constInfo.s2BaseSize);
                current.isLastS2Loop = !hasNext;
                current.curSInnerLoopTimes = current.s2Idx + (hasNext ? 2U : 1U);
                int8NextInfo = current;
                PreloadPipeline<false>(loop, loop, 0, current.s2Idx, extraInfo);
                ++loop;
                current = next;
                valid = hasNext;
            }
        }
    }
    int8NextInfo = {};
    for (uint32_t flush = 0; flush < PRELOAD_NUM; ++flush) {
        PreloadPipeline<false>(loop, loop, 0, 0, extraInfo);
        ++loop;
    }
    if ASCEND_IS_AIV {
        for (uint32_t i = 0; i < 4; ++i) {
            CrossCoreWaitFlag(3);
        }
    }
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::ProcessInt8Balance()
{
    const auto& p = int8Input.params;
    const bool qualified = !FLASH_DECODE && LAYOUT_T == SAS_LAYOUT::TND && PAGE_ATTENTION && p.heads == 64U &&
                           p.topk == 512U && p.oriBlockSize == 128U &&
                           (p.cmpBlockSize == 64U || p.cmpBlockSize == 128U) && p.oriMask == 4U && p.cmpMask == 3U &&
                           p.winLeft == 127 && p.winRight == 0 && (p.cmpRatio == 1U || p.cmpRatio == 4U) &&
                           constInfo.s2BaseSize == 512U && constInfo.bmm2ResUbSize == 64U * 512U;
    uint32_t resumeBatch = constInfo.bN2Start;
    int64_t resumeQuery = constInfo.gS1Start;
    if (qualified) {
        bool fallback = false;
        RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE]{};
        uint32_t loop = 0;
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < 4; ++i) {
                CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
            }
        }
        uint32_t endBatch = constInfo.bN2End + (constInfo.gS1End != 0 ? 1U : 0U);
        for (uint32_t b = constInfo.bN2Start; b < endBatch && b < int8Input.params.batch && !fallback; ++b) {
            int64_t qEnd = Min(int8Input.QLength(b), int8Input.QStorage(b));
            if (b == constInfo.bN2End && constInfo.gS1End != 0) {
                qEnd = Min(qEnd, int64_t(constInfo.gS1End));
            }
            for (int64_t s = b == constInfo.bN2Start ? constInfo.gS1Start : 0; s < qEnd; ++s) {
                int8Input.BeginQuery(b, s);
                int64_t oriLogicalBegin = 0, oriLogicalEnd = 0;
                int8Input.GetOriLogicalRange(oriLogicalBegin, oriLogicalEnd);
                if (oriLogicalEnd - oriLogicalBegin != 128) {
                    resumeBatch = b;
                    resumeQuery = s;
                    fallback = true;
                    break;
                }
                if ASCEND_IS_AIV {
                    vectorBlock.SetInt8SelectionPrepared(false);
                }
                bool canPrepare = int8Input.CanPrepareQuery();
                if (!canPrepare || loop % 2U != 0U) {
                    resumeBatch = b;
                    resumeQuery = s;
                    fallback = true;
                    break;
                }
                uint32_t oriRows = 0, cmpRows = 0;
                if (canPrepare) {
                    // In this two-tile/query experiment, Vec2 only uses accumulation
                    // slot0. Slot1's first128B hold two 64B plan slots, not QK data.
                    uint64_t planOffset =
                        constInfo.bmm2ResUbSize + static_cast<uint64_t>(loop % dbWorkspaceRatio) * 16U;
                    if ASCEND_IS_AIV {
                        vectorBlock.PrepareInt8Selection(int8Input, planOffset, oriRows, cmpRows);
                        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(SYNC_INT8_PLAN_READY);
                        // Empty queries do not advance the ring; wait until Cube has
                        // read their counts before the next query reuses this slot.
                        if (oriRows + cmpRows == 0U) {
                            CrossCoreWaitFlag(SYNC_INT8_EMPTY_PLAN_READ);
                        }
                    } else {
                        CrossCoreWaitFlag(SYNC_INT8_PLAN_READY);
                        auto plan = vec2ResGm[planOffset];
                        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(plan);
                        oriRows = static_cast<uint32_t>(static_cast<int32_t>(plan.GetValue(0)));
                        cmpRows = static_cast<uint32_t>(static_cast<int32_t>(plan.GetValue(1)));
                        if (oriRows + cmpRows == 0U) {
                            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(SYNC_INT8_EMPTY_PLAN_READ);
                        }
                    }
                    if (oriRows != 128U || cmpRows != 512U) {
                        resumeBatch = b;
                        resumeQuery = s;
                        fallback = true;
                        break;
                    }
                    int8Input.UsePreparedSelection(oriRows, cmpRows);
                }
                if (canPrepare) {
                    // Preserve dynamic counts, zero-side handling and all task fields.
                    // curSInnerLoopTimes intentionally matches the old one-tile
                    // lookahead marker; it is NOT the query's total tile count.
                    const uint32_t tileSize = constInfo.s2BaseSize;
                    const uint32_t oriTileCount = oriRows / tileSize + (oriRows % tileSize != 0U);
                    const uint32_t cmpTileCount = cmpRows / tileSize + (cmpRows % tileSize != 0U);
                    const uint32_t tileCount = oriTileCount + cmpTileCount;
                    for (uint32_t tileIdx = 0U; tileIdx < tileCount; ++tileIdx) {
                        RunInfo current{};
                        int8Input.MakePreparedTile(current, tileIdx, oriTileCount, tileSize);
                        const bool hasNext = tileIdx + 1U < tileCount;
                        current.isLastS2Loop = !hasNext;
                        current.curSInnerLoopTimes = current.s2Idx + (hasNext ? 2U : 1U);
                        if (current.isOriOnly) {
                            PlanOriReuse(current, oriLogicalBegin, oriLogicalEnd);
                        }
                        int8NextInfo = current;
                        PreloadPipeline(loop, loop, 0, current.s2Idx, extraInfo);
                        ++loop;
                    }
                    int8Input.FinishPreparedTiles(tileCount);
                } else {
                    RunInfo current{}, next{};
                    bool valid = int8Input.NextTile(current, constInfo.s2BaseSize);
                    while (valid) {
                        // One-tile lookahead preserves the existing online-softmax
                        // first/last semantics even when selected indices are invalid.
                        bool hasNext = int8Input.NextTile(next, constInfo.s2BaseSize);
                        current.isLastS2Loop = !hasNext;
                        current.curSInnerLoopTimes = current.s2Idx + (hasNext ? 2U : 1U);
                        if (current.isOriOnly) {
                            PlanOriReuse(current, oriLogicalBegin, oriLogicalEnd);
                        }
                        int8NextInfo = current;
                        PreloadPipeline(loop, loop, 0, current.s2Idx, extraInfo);
                        ++loop;
                        current = next;
                        valid = hasNext;
                    }
                }
            }
        }
        int8NextInfo = {};
        for (uint32_t flush = 0; flush < PRELOAD_NUM; ++flush) {
            PreloadPipeline(loop, loop, 0, 0, extraInfo);
            ++loop;
        }
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < 4; ++i) {
                CrossCoreWaitFlag(3);
            }
        }
        if (!fallback) {
            return;
        }
        // The original fast drain above consumed all four credits, even if
        // the first query failed. Join both AIVs before generic seeds again.
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(SYNC_INT8_FAST_DRAINED);
        } else {
            CrossCoreWaitFlag(SYNC_INT8_FAST_DRAINED);
        }
        oriCacheValid = false;
    }
    // One fallback call site keeps generic code from being duplicated at
    // each dynamic eligibility check. Actual failed query is retried here.
    ProcessInt8GenericTail(resumeBatch, resumeQuery);
}

template <typename SAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::ProcessBalance()
{
    if constexpr (SAST::int8Kv) {
        ProcessInt8Balance();
        return;
    }
    RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    uint32_t cmpLoop = 0;
    uint32_t gS1LoopEnd = 0;
    bool globalLoopStart = true;

    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
    }

    // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
        }
    } else if ((constInfo.gS1End != 0) || (constInfo.s2End != 0)) {
        constInfo.bN2End += 1;
    }

    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        tempLoopInfo.actS1Size = GetActualSeqLenQ(tempLoopInfo.bIdx); // 获取actualSeqLength
        bool isS1ZeroAndLastBatch = (tempLoopInfo.actS1Size == 0) && ((constInfo.outputLayout == SAS_LAYOUT::BSND) ||
                                                                      (bN2LoopIdx + 1 == constInfo.bN2End));
        uint32_t gS1SplitNum = CeilDiv(tempLoopInfo.actS1Size * constInfo.gSize, constInfo.mBaseSize);

        if (tempLoopInfo.actS1Size <= 0) {
            if (!isS1ZeroAndLastBatch) {
                continue;
            }
            // Flush the two-stage preload pipeline without forming a fake query
            // tile. In particular, avoid `actS1Size - 1` and 0/0 below.
            tempLoopInfo.s2LoopTimes = 0;
            for (uint32_t flush = 0; flush < PRELOAD_NUM; ++flush) {
                PreloadPipeline(gloop, cmpLoop, 0, flush, extraInfo);
                ++gloop;
            }
            continue;
        }

        // 当处于最后一个BN2时, 且gS1End为0时, 说明当前BN2里的所有数据都在当前核处理
        gS1LoopEnd = (bN2LoopIdx + 1 == constInfo.bN2End && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
        // 当处于最后一个BN2且当前S1为0时，需要进入循环计算preload导致的未完成的部分
        gS1LoopEnd = isS1ZeroAndLastBatch ? gS1LoopEnd + 1 : gS1LoopEnd;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < gS1LoopEnd; gS1LoopIdx++) {
            tempLoopInfo.actOriS2Size = GetActualSeqLenKV(tempLoopInfo.bIdx);
            // 计算需要的数据, 避免重复计算
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;
            tempLoopInfo.s1StartIdx = tempLoopInfo.gS1Idx / constInfo.gSize;
            tempLoopInfo.s1EndIdx =
                Min((tempLoopInfo.s1StartIdx + constInfo.mBaseSize / constInfo.gSize - 1), tempLoopInfo.actS1Size - 1);

            // 此处均为闭区间
            tempLoopInfo.oriMaskRight = tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size +
                                        static_cast<int32_t>(tempLoopInfo.s1EndIdx) + constInfo.oriWinRight;
            tempLoopInfo.oriMaskLeft = Max(tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size +
                                               static_cast<int32_t>(tempLoopInfo.s1StartIdx) - constInfo.oriWinLeft,
                                           0);
            tempLoopInfo.cmpMaskRight = tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size;
            GetSparseActualSeqLen();
            UpdateInnerLoopCond();

            uint32_t oriS2Size = 0;
            uint32_t oriSplitNum = 0;
            uint32_t cmpSplitNum = 0;
            uint32_t cmpS2Size = 0;
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == gS1LoopEnd);
            if (tempLoopInfo.curActSeqLenIsZero) {
                if ASCEND_IS_AIV {
                    InitAllZeroOutput(tempLoopInfo.bIdx, tempLoopInfo.s1StartIdx, tempLoopInfo.n2Idx);
                }
                if (!isEnd) {
                    continue;
                }
            } else {
                oriS2Size = static_cast<uint32_t>(tempLoopInfo.oriMaskRight - tempLoopInfo.oriMaskLeft + 1);
                oriSplitNum = CeilDiv(oriS2Size, constInfo.s2BaseSize);
                cmpS2Size = tempLoopInfo.actCmpS2Size;
                cmpSplitNum = CeilDiv(cmpS2Size, constInfo.s2BaseSize);
            }

            uint32_t s2SplitNum = oriSplitNum + cmpSplitNum;

            tempLoopInfo.oriLoopTimes = oriSplitNum;
            tempLoopInfo.cmpLoopTimes = cmpSplitNum;
            tempLoopInfo.s2LoopTimes = s2SplitNum;

            uint32_t s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : tempLoopInfo.s2LoopTimes;
            tempLoopInfo.s2LoopTimes = s2LoopEnd;
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            tempLoopInfo.tndIsS2SplitCore = ((constInfo.s2Start == 0) && (s2LoopEnd == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? 2 : 0;
            uint32_t curTopKIdx = 0;
            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (s2LoopEnd + extraLoop); s2LoopIdx++) {
                PreloadPipeline(gloop, cmpLoop, constInfo.s2Start, s2LoopIdx, extraInfo);
                ++gloop;
                if (s2LoopIdx >= tempLoopInfo.oriLoopTimes && s2LoopIdx < s2LoopEnd) { // 用于判断v0使用的循环GM的id
                    ++cmpLoop;
                }
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
    }
}

template <typename SAST>
template <bool INT8_FAST>
__aicore__ inline void MixedQuantSparseFlashMlaTqCsaKernel<SAST>::PreloadPipeline(
    uint32_t loop, uint32_t cmpLoop, uint64_t s2Start, uint64_t s2LoopIdx,
    RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo& extraInfo0 = extraInfo[loop % SAS_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo& extraInfo2 = extraInfo[(loop + 2) % SAS_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo& extraInfo1 = extraInfo[(loop + 1) % SAS_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, cmpLoop, s2Start, s2LoopIdx, extraInfo0);
    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            const bool wave256 =
                INT8_FAST && SAST::int8Kv && !extraInfo0.isOriOnly && extraInfo0.actualSingleProcessSInnerSize == 512U;
            if ((!extraInfo0.isOriOnly || SAST::int8Kv) && !wave256) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            ComputeMm1<INT8_FAST>(extraInfo0);
        } else {
            if (extraInfo0.isFirstSInnerLoop) {
                CrossCoreWaitFlag(3);
            }
            vectorBlock.template ProcessVec0L<INT8_FAST>(extraInfo0);
            const bool wave256 =
                INT8_FAST && SAST::int8Kv && !extraInfo0.isOriOnly && extraInfo0.actualSingleProcessSInnerSize == 512U;
            if ((!extraInfo0.isOriOnly || SAST::int8Kv) && !wave256) {
                CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            vectorBlock.ProcessVec1L(extraInfo2);
        }
        if ASCEND_IS_AIC {
            ComputeMm2<INT8_FAST>(extraInfo2);
            if (extraInfo2.isLastS2Loop) {
                CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE2>(3);
            }
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            vectorBlock.ProcessVec2L(extraInfo1);
        }
        extraInfo1.isValid = false;
    }
}

} // namespace SASKernel
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_TQ_CSA_KERNEL_H
