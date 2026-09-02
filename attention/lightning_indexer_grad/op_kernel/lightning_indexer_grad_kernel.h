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
 * \file lightning_indexer_grad_kernel.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_GRAD_KERNEL_H
#define LIGHTNING_INDEXER_GRAD_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_grad_common.h"
#include "lightning_indexer_grad_service_cube.h"
#include "lightning_indexer_grad_service_vector.h"

namespace LigKernel {
using namespace matmul;
using namespace LIGCommon;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

struct CoreSplitInfo {
    uint64_t beginPos;
    uint64_t length;
};
template <typename LIGT>
class LIGKernel {
public:
    __aicore__ inline LIGKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *dy,
                                __gm__ uint8_t *sparse_indices, __gm__ uint8_t *weights,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *dq,
                                __gm__ uint8_t *dk, __gm__ uint8_t *dweights, __gm__ uint8_t *workspace,
                                const LIGTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, GlobalTensor<uint32_t> &actualSeqLengthsGm);
    __aicore__ inline uint32_t GetPrefixSeqLen(uint32_t bIdx, GlobalTensor<uint32_t> &actualSeqLengthsGm);
    __aicore__ inline void InitRunInfo(uint64_t bIndex, const CoreSplitInfo &split,
                                       const CoreSplitInfo &determineSplit);
    __aicore__ inline void UpdateRunInfo(uint64_t bIndex, LIGCommon::RunInfo &runInfo, uint64_t taskId);
    __aicore__ inline CoreSplitInfo DoSplit(uint64_t totalLoopSize, uint64_t coreIdx, uint64_t coreNum);
    __aicore__ inline CoreSplitInfo SplitCore(uint64_t bIndex, uint64_t coreIdx, uint64_t coreNum);
    __aicore__ inline CoreSplitInfo CalculateDetermineLoopTimes(uint64_t bIndex, uint64_t coreIdx, uint64_t coreNum);
    __aicore__ inline void CopyRunInfo(LIGCommon::RunInfo &dstRunInfo, LIGCommon::RunInfo srcRunInfo);
    __aicore__ inline uint64_t CalcRealTopk(LIGCommon::RunInfo &runInfo);
    __aicore__ inline void ProcessVec1(uint64_t taskId);
    __aicore__ inline void ProcessVec2(uint64_t taskId);
    __aicore__ inline void ProcessVec3(uint64_t taskId);
    __aicore__ inline void ProcessCube1(uint64_t taskId);
    __aicore__ inline void ProcessCube2(uint64_t taskId);

    using D_T = typename LIGT::dataType;
    static constexpr LIG_LAYOUT LAYOUT = LIGT::layout;

    LIGMatmul<LIGT> matmulService;
    LIGVector<LIGT> vectorService;

    static constexpr uint64_t SYNC_V1_C1_FLAG = 7;
    static constexpr uint64_t SYNC_C1_V2_FLAG = 8;
    static constexpr uint64_t SYNC_V2_C2_FLAG = 9;
    static constexpr uint64_t SYNC_C2_V3_FLAG = 10;
    static constexpr uint64_t SYNC_C2_V1_FLAG = 11;

protected:
    TPipe *pipe = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCubeCoreNum = 0U;
    uint64_t taskId = 0;

    // ================================Input GlobalTensor=================================
    uint32_t batch = 0;
    uint32_t seqlenQ = 0;
    uint32_t seqlenK = 0;
    uint32_t topK = 0;
    uint32_t headNumQ = 0;
    uint32_t headNumK = 0;
    uint32_t groupNum = 0;
    uint32_t headDim = 0;
    uint32_t maxLoopLen = 0;

    // ================================Input GlobalTensor=================================
    GlobalTensor<D_T> queryGm;
    GlobalTensor<D_T> keyGm;
    GlobalTensor<D_T> dyGm;
    GlobalTensor<D_T> weightsGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGmK;

    // ================================workspace GlobalTensor=============================
    GlobalTensor<float> dkWorkSpaceGm;
    GlobalTensor<float> dkCoreWorkspaceGM;
    GlobalTensor<D_T> keyGatherPingGm;
    GlobalTensor<D_T> keyGatherPongGm;
    GlobalTensor<float> reluInPingGm;
    GlobalTensor<float> reluInPongGm;
    GlobalTensor<D_T> reluGradPingGm;
    GlobalTensor<D_T> reluGradPongGm;
    GlobalTensor<float> scatterAddPingGm;
    GlobalTensor<float> scatterAddPongGm;

    GlobalTensor<D_T> keyGatherGm;
    GlobalTensor<float> reluInGm;
    GlobalTensor<D_T> reluGradGm;
    GlobalTensor<float> scatterAddGm;

    // ================================Output GlobalTensor================================
    GlobalTensor<D_T> dqGm;
    GlobalTensor<D_T> dkGm;
    GlobalTensor<D_T> dweightsGm;

    LIGCommon::ConstInfo constInfo{};

    LIGCommon::RunInfo runInfoStore[4];
    LIGCommon::RunInfo runInfo;

    // ================================Init functions=====================================
    __aicore__ inline void InitTilingData(const LIGTilingData *__restrict tilingData);
};

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::InitTilingData(const LIGTilingData *__restrict tilingData)
{
    constInfo.batch = tilingData->batch;
    constInfo.seqlenQ = tilingData->seqlenQ;
    constInfo.seqlenK = tilingData->seqlenK;
    constInfo.headNumQ = tilingData->headNumQ;
    constInfo.headNumK = tilingData->headNumK;
    constInfo.groupNum = tilingData->groupNum;
    constInfo.headDim = tilingData->headDim;
    constInfo.topK = tilingData->topK;
    constInfo.usedCoreNum = tilingData->usedCoreNum;
    constInfo.dkSize = tilingData->dkSize;
    constInfo.dkCoreSize = tilingData->dkCoreSize;
    constInfo.dkWorkSpaceOffset = tilingData->dkWorkSpaceOffset;
    constInfo.dkCoreWorkspaceOffset = tilingData->dkCoreWorkspaceOffset;
    constInfo.keyGatherWorkspaceOffset = tilingData->keyGatherWorkspaceOffset;
    constInfo.reluInWorkspaceOffset = tilingData->reluInWorkspaceOffset;
    constInfo.reluGradWorkspaceOffset = tilingData->reluGradWorkspaceOffset;
    constInfo.scatterAddWorkspaceOffset = tilingData->scatterAddWorkspaceOffset;
    constInfo.sparseMode = tilingData->sparseMode;
    constInfo.deterministic = tilingData->deterministic;
    constInfo.splitCores = tilingData->usedCoreNum / 2;
    usedCubeCoreNum = tilingData->usedCoreNum / 2;
    return;
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                         __gm__ uint8_t *actualSeqLengthsK)
{
    if (actualSeqLengthsQ != nullptr) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.batch);
    }
    if (actualSeqLengthsK != nullptr) {
        actualSeqLengthsGmK.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsK, constInfo.batch);
    }
}

template <typename LIGT>
__aicore__ inline uint32_t LIGKernel<LIGT>::GetActualSeqLen(uint32_t bIdx, GlobalTensor<uint32_t> &actualSeqLengthsGm)
{
    if (bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename LIGT>
__aicore__ inline uint32_t LIGKernel<LIGT>::GetPrefixSeqLen(uint32_t bIdx, GlobalTensor<uint32_t> &actualSeqLengthsGm)
{
    if (bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return 0;
    }
}

template <typename LIGT>
__aicore__ inline uint64_t LIGKernel<LIGT>::CalcRealTopk(LIGCommon::RunInfo &runInfo)
{
    if (constInfo.sparseMode == 0) {
        return constInfo.topK;
    } else {
        int64_t s2RealSize = (runInfo.actualSeqK - runInfo.actualSeqQ) + runInfo.s1Idx + 1;
        if (s2RealSize <= 0) {
            return 0;
        } else {
            return s2RealSize >= constInfo.topK ? constInfo.topK : s2RealSize;
        }
    }
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::InitRunInfo(uint64_t bIndex, const CoreSplitInfo &split,
                                                    const CoreSplitInfo &determineSplit)
{
    // B -> S1 -> N2 -> G -> D
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        runInfo.n2Idx = split.beginPos % constInfo.headNumK;
        runInfo.s1Idx = (split.beginPos / constInfo.headNumK) % constInfo.seqlenQ;
        runInfo.actualSeqQ = constInfo.seqlenQ;
        runInfo.actualSeqK = constInfo.seqlenK;
        runInfo.prefixSumS1 = 0;
        runInfo.prefixSumS2 = 0;
        runInfo.loopTimes = split.length;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        runInfo.n2Idx = split.beginPos % constInfo.headNumK;
        uint64_t tIdx = split.beginPos / constInfo.headNumK;
        runInfo.actualSeqQ = 0;
        runInfo.actualSeqK = 0;
        runInfo.prefixSumS1 = 0;
        runInfo.prefixSumS2 = 0;
        // linear loop to find correct batch
        for (uint32_t i = 0; i < constInfo.batch; i++) {
            uint32_t currentPrefixSum = actualSeqLengthsGmQ.GetValue(i);
            if (tIdx <= currentPrefixSum) {
                if (tIdx == currentPrefixSum) {
                    runInfo.s1Idx = 0;
                } else {
                    runInfo.s1Idx = tIdx;
                }
                runInfo.actualSeqQ = GetActualSeqLen(bIndex, actualSeqLengthsGmQ);
                runInfo.actualSeqK = GetActualSeqLen(bIndex, actualSeqLengthsGmK);
                runInfo.prefixSumS1 = GetPrefixSeqLen(bIndex, actualSeqLengthsGmQ);
                runInfo.prefixSumS2 = GetPrefixSeqLen(bIndex, actualSeqLengthsGmK);
                break;
            }
        }
        runInfo.loopTimes = split.length;
    }
    runInfo.bIdx = bIndex;
    runInfo.taskId = taskId;
    runInfo.realTopk = CalcRealTopk(runInfo);
    constInfo.determinLen = determineSplit.length;
    constInfo.determinBeginPos = determineSplit.beginPos;
    runInfo.isRemainderCore = split.length == maxLoopLen ? true : false;
    CopyRunInfo(runInfoStore[taskId], runInfo);
    taskId++;
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::UpdateRunInfo(uint64_t bIndex, LIGCommon::RunInfo &runInfo, uint64_t taskId)
{
    uint32_t maxSeqQ = (LIGT::layout == LIG_LAYOUT::BSND) ? constInfo.seqlenQ : runInfo.actualSeqQ;

    runInfo.n2Idx++;
    if (runInfo.n2Idx >= constInfo.headNumK) {
        runInfo.n2Idx = 0;
        runInfo.s1Idx++;
        if (runInfo.s1Idx >= maxSeqQ) {
            runInfo.s1Idx = 0;
        }
    }
    runInfo.bIdx = bIndex;
    if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        if (runInfo.bIdx < constInfo.batch) {
            runInfo.actualSeqQ = GetActualSeqLen(runInfo.bIdx, actualSeqLengthsGmQ);
            runInfo.actualSeqK = GetActualSeqLen(runInfo.bIdx, actualSeqLengthsGmK);
            runInfo.prefixSumS1 = GetPrefixSeqLen(runInfo.bIdx, actualSeqLengthsGmQ);
            runInfo.prefixSumS2 = GetPrefixSeqLen(runInfo.bIdx, actualSeqLengthsGmK);
        }
    }
    runInfo.realTopk = CalcRealTopk(runInfo);
    CopyRunInfo(runInfoStore[taskId % 4], runInfo);
}

template <typename LIGT>
__aicore__ inline CoreSplitInfo LIGKernel<LIGT>::DoSplit(uint64_t totalLoopSize, uint64_t coreIdx, uint64_t coreNum)
{
    uint64_t base = totalLoopSize / coreNum;
    uint64_t remainder = totalLoopSize % coreNum;

    CoreSplitInfo info{0, 0};

    if (coreIdx < remainder) {
        info.length = base + 1;
        info.beginPos = coreIdx * (base + 1);
    } else {
        info.length = base;
        info.beginPos = remainder * (base + 1) + (coreIdx - remainder) * base;
    }

    if (info.beginPos >= totalLoopSize) {
        info.beginPos = totalLoopSize;
        info.length = 0;
    } else if (info.beginPos + info.length > totalLoopSize) {
        info.length = totalLoopSize - info.beginPos;
    }
    return info;
}

template <typename LIGT>
__aicore__ inline CoreSplitInfo LIGKernel<LIGT>::SplitCore(uint64_t bIndex, uint64_t coreIdx, uint64_t coreNum)
{
    uint64_t totalLoops = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        totalLoops = constInfo.seqlenQ * constInfo.headNumK;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        uint64_t actualSeqQ = GetActualSeqLen(bIndex, actualSeqLengthsGmQ);
        totalLoops = actualSeqQ * constInfo.headNumK;
    }
    if (totalLoops / coreNum == 0) {
        constInfo.splitCores = totalLoops % coreNum;
    }
    if (totalLoops % coreNum != 0) {
        maxLoopLen = totalLoops / coreNum + 1;
    }
    return DoSplit(totalLoops, coreIdx, coreNum);
}

template <typename LIGT>
__aicore__ inline CoreSplitInfo LIGKernel<LIGT>::CalculateDetermineLoopTimes(uint64_t bIndex, uint64_t coreIdx,
                                                                             uint64_t coreNum)
{
    uint64_t total = constInfo.seqlenK;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        total = constInfo.seqlenK;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        total = GetActualSeqLen(bIndex, actualSeqLengthsGmK);
    }
    if (constInfo.splitCores < usedCubeCoreNum) {
        return DoSplit(total, coreIdx, constInfo.splitCores * 2);
    }
    return DoSplit(total, coreIdx, coreNum);
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *dy,
                                             __gm__ uint8_t *sparse_indices, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                             __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dweights,
                                             __gm__ uint8_t *workspace, const LIGTilingData *__restrict tiling,
                                             TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    pipe = tPipe;
    InitTilingData(tiling);

    // init input global tensor
    queryGm.SetGlobalBuffer((__gm__ D_T *)query);
    keyGm.SetGlobalBuffer((__gm__ D_T *)key);
    dyGm.SetGlobalBuffer((__gm__ D_T *)dy);
    weightsGm.SetGlobalBuffer((__gm__ D_T *)weights);
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparse_indices);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengthsK);

    // init output global tensor
    dqGm.SetGlobalBuffer((__gm__ D_T *)dq);
    dkGm.SetGlobalBuffer((__gm__ D_T *)dk);
    dweightsGm.SetGlobalBuffer((__gm__ D_T *)dweights);

    // init workspace global tensor
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.dkWorkSpaceOffset / sizeof(float));

    uint64_t keyCoreSize = constInfo.seqlenK * constInfo.headDim * sizeof(float);
    dkCoreWorkspaceGM.SetGlobalBuffer((__gm__ float *)workspace + constInfo.dkCoreWorkspaceOffset / sizeof(float));

    uint64_t keyGatherSize = 2048 * 128 * sizeof(D_T) * 2;
    keyGatherPingGm.SetGlobalBuffer(
        (__gm__ D_T *)(workspace + constInfo.keyGatherWorkspaceOffset + aiCoreIdx * keyGatherSize));
    keyGatherPongGm.SetGlobalBuffer(
        (__gm__ D_T *)(workspace + constInfo.keyGatherWorkspaceOffset + aiCoreIdx * keyGatherSize + keyGatherSize / 2));

    uint64_t reluInSize = 64 * 2048 * sizeof(float) * 2;
    reluInPingGm.SetGlobalBuffer(
        (__gm__ float *)(workspace + constInfo.reluInWorkspaceOffset + aiCoreIdx * reluInSize));
    reluInPongGm.SetGlobalBuffer(
        (__gm__ float *)(workspace + constInfo.reluInWorkspaceOffset + aiCoreIdx * reluInSize + reluInSize / 2));

    uint64_t reluGradSize = 64 * 2048 * sizeof(D_T) * 2;
    reluGradPingGm.SetGlobalBuffer(
        (__gm__ D_T *)(workspace + constInfo.reluGradWorkspaceOffset + aiCoreIdx * reluGradSize));
    reluGradPongGm.SetGlobalBuffer(
        (__gm__ D_T *)(workspace + constInfo.reluGradWorkspaceOffset + aiCoreIdx * reluGradSize + reluGradSize / 2));

    uint64_t scatterAddSize = 2048 * 128 * sizeof(float) * 2;
    scatterAddPingGm.SetGlobalBuffer(
        (__gm__ float *)(workspace + constInfo.scatterAddWorkspaceOffset + aiCoreIdx * scatterAddSize));
    scatterAddPongGm.SetGlobalBuffer((__gm__ float *)(workspace + constInfo.scatterAddWorkspaceOffset +
                                                      aiCoreIdx * scatterAddSize + scatterAddSize / 2));

    if ASCEND_IS_AIV {
        vectorService.Init(pipe);
    }
    if ASCEND_IS_AIC {
        matmulService.Init(pipe);
    }
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::CopyRunInfo(LIGCommon::RunInfo &dstRunInfo, LIGCommon::RunInfo srcRunInfo)
{
    dstRunInfo.bIdx = srcRunInfo.bIdx;
    dstRunInfo.n2Idx = srcRunInfo.n2Idx;
    dstRunInfo.s1Idx = srcRunInfo.s1Idx;
    dstRunInfo.actualSeqQ = srcRunInfo.actualSeqQ;
    dstRunInfo.actualSeqK = srcRunInfo.actualSeqK;
    dstRunInfo.prefixSumS1 = srcRunInfo.prefixSumS1;
    dstRunInfo.prefixSumS2 = srcRunInfo.prefixSumS2;
    dstRunInfo.loopTimes = srcRunInfo.loopTimes;
    dstRunInfo.taskId = srcRunInfo.taskId;
    dstRunInfo.realTopk = srcRunInfo.realTopk;
    dstRunInfo.isRemainderCore = srcRunInfo.isRemainderCore;
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::ProcessVec1(uint64_t taskId)
{
    keyGatherGm = (taskId & 1) ? keyGatherPingGm : keyGatherPongGm;
    if (runInfoStore[taskId].realTopk > 0) {
        vectorService.GatherTopk(sparseIndicesGm, keyGm, keyGatherGm, constInfo, runInfoStore[taskId]);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_C1_FLAG);
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::ProcessVec2(uint64_t taskId)
{
    reluInGm = (taskId & 1) ? reluInPingGm : reluInPongGm;
    reluGradGm = (taskId & 1) ? reluGradPingGm : reluGradPongGm;
    AscendC::WaitEvent(SYNC_C1_V2_FLAG);
    if (runInfoStore[taskId].realTopk > 0) {
        vectorService.ReluGrad(reluInGm, reluGradGm, dyGm, reluGradGm, dweightsGm, constInfo, runInfoStore[taskId]);
    } else {
        vectorService.InitOutputDqAndDweights(dweightsGm, dqGm, constInfo, runInfoStore[taskId]);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V2_C2_FLAG);
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::ProcessVec3(uint64_t taskId)
{
    scatterAddGm = (taskId & 1) ? scatterAddPingGm : scatterAddPongGm;
    AscendC::WaitEvent(SYNC_C2_V3_FLAG);
    if (likely(!constInfo.deterministic)) {
        vectorService.ScatterAdd(sparseIndicesGm, scatterAddGm, dkWorkSpaceGm, constInfo, runInfoStore[taskId]);
    } else {
        SyncAll();
        vectorService.InitOutputDkcoreGm(dkCoreWorkspaceGM, constInfo, runInfoStore[taskId]);
        SyncAll();
        vectorService.ScatterAdd(sparseIndicesGm, scatterAddGm, dkCoreWorkspaceGM, constInfo, runInfoStore[taskId]);
        SyncAll();
        vectorService.DeterministicMerge(dkCoreWorkspaceGM, dkWorkSpaceGm, constInfo, runInfoStore[taskId]);
    }
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::ProcessCube1(uint64_t taskId)
{
    keyGatherGm = (taskId & 1) ? keyGatherPingGm : keyGatherPongGm;
    reluInGm = (taskId & 1) ? reluInPingGm : reluInPongGm;
    reluGradGm = (taskId & 1) ? reluGradPingGm : reluGradPongGm;

    AscendC::WaitEvent(SYNC_V1_C1_FLAG);
    if (runInfoStore[taskId].realTopk > 0) {
        matmulService.AllocEventID();
        matmulService.Cube2(weightsGm, dyGm, reluGradGm, constInfo, runInfoStore[taskId]);
        matmulService.FreeEventID();

        matmulService.AllocEventID();
        matmulService.Cube1(queryGm, keyGatherGm, reluInGm, constInfo, runInfoStore[taskId]);
        matmulService.FreeEventID();
    }
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_V2_FLAG);
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::ProcessCube2(uint64_t taskId)
{
    keyGatherGm = (taskId & 1) ? keyGatherPingGm : keyGatherPongGm;
    reluGradGm = (taskId & 1) ? reluGradPingGm : reluGradPongGm;
    scatterAddGm = (taskId & 1) ? scatterAddPingGm : scatterAddPongGm;

    AscendC::WaitEvent(SYNC_V2_C2_FLAG);
    if (runInfoStore[taskId].realTopk > 0) {
        matmulService.AllocEventID();
        matmulService.Cube3(reluGradGm, keyGatherGm, dqGm, constInfo, runInfoStore[taskId]);
        matmulService.FreeEventID();

        matmulService.AllocEventID();
        matmulService.Cube4(reluGradGm, queryGm, scatterAddGm, constInfo, runInfoStore[taskId]);
        matmulService.FreeEventID();
    }
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V3_FLAG);
}

template <typename LIGT>
__aicore__ inline void LIGKernel<LIGT>::Process()
{
    for (uint32_t bIndex = 0; bIndex < constInfo.batch; bIndex++) {
        taskId = 0;
        CoreSplitInfo split = SplitCore(bIndex, aiCoreIdx, usedCubeCoreNum);
        CoreSplitInfo determineSplit = CalculateDetermineLoopTimes(bIndex, GetBlockIdx(), usedCubeCoreNum * 2);

        InitRunInfo(bIndex, split, determineSplit);
        for (uint32_t i = 0; runInfo.loopTimes > 0 && i < runInfo.loopTimes + 3; i++) {
            // pre1
            if (i == 0) {
                if ASCEND_IS_AIV {
                    ProcessVec1(i % 4);
                }
            }

            // pre2
            if (i == 1) {
                if ASCEND_IS_AIV {
                    if (runInfo.loopTimes > 1) {
                        ProcessVec1(i % 4);
                    }
                }
                if ASCEND_IS_AIC {
                    ProcessCube1((i - 1) % 4);
                }
            }

            // pre3
            if (i == 2) {
                if ASCEND_IS_AIV {
                    ProcessVec2((i - 2) % 4);
                    if (runInfo.loopTimes > 2) {
                        AscendC::WaitEvent(SYNC_C2_V1_FLAG);
                        ProcessVec1(i % 4);
                    }
                }
                if ASCEND_IS_AIC {
                    if (runInfo.loopTimes > 1) {
                        ProcessCube1((i - 1) % 4);
                    }
                    ProcessCube2((i - 2) % 4);
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V1_FLAG);
                }
            }

            // MID
            if (runInfo.loopTimes >= 3 && i >= 3 && i < runInfo.loopTimes) {
                if ASCEND_IS_AIV {
                    ProcessVec2((i - 2) % 4);
                    ProcessVec3((i - 3) % 4);
                    AscendC::WaitEvent(SYNC_C2_V1_FLAG);
                    ProcessVec1(i % 4);
                }
                if ASCEND_IS_AIC {
                    ProcessCube1((i - 1) % 4);
                    ProcessCube2(((i - 2) % 4));
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V1_FLAG);
                }
            }

            // end3
            if (runInfo.loopTimes > 2 && i == runInfo.loopTimes) {
                if ASCEND_IS_AIV {
                    ProcessVec2((i - 2) % 4);
                    ProcessVec3((i - 3) % 4);
                }
                if ASCEND_IS_AIC {
                    ProcessCube1((i - 1) % 4);
                    ProcessCube2(((i - 2) % 4));
                }
            }

            // end2
            if (runInfo.loopTimes > 1 && i == runInfo.loopTimes + 1) {
                if ASCEND_IS_AIV {
                    ProcessVec2((i - 2) % 4);
                    ProcessVec3((i - 3) % 4);
                }
                if ASCEND_IS_AIC {
                    ProcessCube2(((i - 2) % 4));
                }
            }

            // end1
            if (i == runInfo.loopTimes + 2) {
                if ASCEND_IS_AIV {
                    ProcessVec3((i - 3) % 4);
                }
            }
            if (i < runInfo.loopTimes) {
                UpdateRunInfo(bIndex, runInfo, taskId++);
            }
        }
        if ASCEND_IS_AIV {
            if (unlikely(constInfo.deterministic && !runInfo.isRemainderCore)) {
                SyncAll();
                vectorService.InitOutputDkcoreGm(dkCoreWorkspaceGM, constInfo,
                                                 runInfoStore[(runInfo.loopTimes - 1) % 4]);
                SyncAll();
                SyncAll(); // 轮空AIV核保持与非轮空核同步等待，等待scatterAdd结果放入dkCoreWorkspaceGM后累加
                vectorService.DeterministicMerge(dkCoreWorkspaceGM, dkWorkSpaceGm, constInfo,
                                                 runInfoStore[(runInfo.loopTimes - 1) % 4]);
            }
        }

        if ASCEND_IS_AIV {
            vectorService.ReleaseEvents();
            SyncAll();
        }
    }
    return;
}
} // namespace LigKernel
#endif // LIGHTNING_INDEXER_GRAD_KERNEL_H
