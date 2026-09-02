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
 * \file dense_lightning_indexer_grad_kl_loss_base.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_BASE_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "dense_lightning_indexer_grad_kl_loss_common.h"
#include "dense_lightning_indexer_grad_kl_loss_tiling_data.h"
#include "dense_lightning_indexer_grad_kl_loss_vector.h"
#include "dense_lightning_indexer_grad_kl_loss_vector2.h"
#include "dense_lightning_indexer_grad_kl_loss_service_cube.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename DLIT>
class DenseLightningIndexerGradKLLossBase {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename DLIT::inputQT;
    using KV_T = typename DLIT::inputKT;
    using W_T = typename DLIT::inputWT;
    using OUT_T = typename DLIT::outputT;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = KV_T;
    using MM12_OUT_T = T;
    using INFO_INT_64_T = int64_t;

    static constexpr bool hasRope = DLIT::hasRope;
    static constexpr bool deterministic = DLIT::deterministic;
    static constexpr DLILayout LAYOUT_T = DLIT::inputQLayout;
    static constexpr DLILayout KV_LAYOUT_T = DLIT::inputKLayout;

    __aicore__ inline DenseLightningIndexerGradKLLossBase(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *queryIndex,
                                __gm__ uint8_t *keyIndex, __gm__ uint8_t *weight, __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxMaxIndex,
                                __gm__ uint8_t *softmaxSumIndex, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKey,
                                __gm__ uint8_t *dQueryIndex, __gm__ uint8_t *dKeyIndex, __gm__ uint8_t *dWeight,
                                __gm__ uint8_t *loss, __gm__ uint8_t *workspace,
                                const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitConstInfo();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx, int64_t &bEndIdx,
                                               int64_t &s1EndIdx);
    __aicore__ inline int64_t FindBIndex(int64_t bIndex, int64_t curIndex, int64_t &accumulateLen);
    __aicore__ inline int64_t GetEndS1Etx(int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm,
                                          DLILayout layout);
    __aicore__ inline void GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s2Idx, uint32_t s1Size,
                                      uint32_t s2Size, int64_t accumS1Idx, int64_t accumS2Idx, uint32_t curS1Size,
                                      uint32_t curS2StepSize, int32_t sparseMaskStartIdx, int32_t s2EndIdx);
    __aicore__ inline int32_t GetActualSeqLens(int32_t bIdx, int32_t defaultLens,
                                               GlobalTensor<int64_t> &actualSeqLensGm, DLILayout layout,
                                               int64_t &accumLen);
    __aicore__ inline int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK,
                                             DLISparseMode sparseMode);

    TPipe *pipe = nullptr;
    const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tilingData = nullptr;
    DLIGradKLLossConstInfo constInfo;
    DLIGradKLLossRunInfo runInfos[3];

    // cube and vector class
    DLITMatmulService<DLIT> matmulService;
    DLIKLLossVectorService<DLIT> vectorService;
    DLIKLLossVector2Service<DLIT> vector2Service;

    // input GM
    GlobalTensor<Q_T> queryGm, queryIndexGm;
    GlobalTensor<W_T> weightGm;
    GlobalTensor<KV_T> keyGm, keyIndexGm;
    GlobalTensor<Q_ROPE_T> queryRopeGm;
    GlobalTensor<K_ROPE_T> keyRopeGm;
    GlobalTensor<T> softmaxMaxGm, softmaxSumGm, softmaxMaxIndexGm, softmaxSumIndexGm;
    GlobalTensor<int64_t> actualSeqLengthsQueryGm, actualSeqLengthsKeyGm;

    // output GM
    GlobalTensor<OUT_T> dQueryIndexGm, dKeyIndexGm;
    GlobalTensor<W_T> dWeightGm;
    GlobalTensor<T> lossGm;

    // workspace
    GlobalTensor<MM12_OUT_T> bmm1Res, bmm2Res;
    GlobalTensor<T> pSyncGm;
    GlobalTensor<T> sySyncGm;
    GlobalTensor<KV_T> reluGradRes;
    GlobalTensor<T> reluGm;
    GlobalTensor<T> dWeightGmFloat;
    GlobalTensor<T> dKeyIndexGmFloat;
    GlobalTensor<T> dQueryIndexGmFloat;
    GlobalTensor<T> dKeyIndexDeterGmFloat;
    GlobalTensor<T> lossGmDeterFloat;
    GlobalTensor<INFO_INT_64_T> deterCoreInfoGm;
};

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossBase<DLIT>::InitConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx(); // vec:0-47
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % 2;
    } else {
        constInfo.aicIdx = GetBlockIdx(); // cube:0-23
    }
    constInfo.aivNum = GetBlockNum() * AIC_AIV_RATIO;

    auto &baseInfo = tilingData->baseParams;
    constInfo.t1Size = tilingData->multiCoreParams.totalSize;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.n2IndexSize = baseInfo.n2IndexSize;
    constInfo.gSizeQuery = baseInfo.gSizeQuery;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.kSize = baseInfo.kSize;
    constInfo.n1Size = constInfo.n2Size * constInfo.gSizeQuery;
    constInfo.n1IndexSize = constInfo.n2IndexSize * constInfo.gSizeQueryIndex;

    constInfo.dSizeQuery = baseInfo.dSizeQuery;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.dSizeRope = ROPE_D_SIZE;
    constInfo.sparseMode = static_cast<DLISparseMode>(baseInfo.sparseMode);
    constInfo.scaleValue = baseInfo.scaleValue;

    constInfo.s2BaseSize = N_WORKSPACE_SIZE;
    constInfo.aicNum = tilingData->multiCoreParams.coreNum;

    if constexpr (hasRope) {
        constInfo.dSizeActual = constInfo.dSizeQuery + constInfo.dSizeQueryRope;
    } else {
        constInfo.dSizeActual = constInfo.dSizeQuery;
    }
    constInfo.dLoopTimesCube = CeilDiv(constInfo.dSizeActual, CUBE_BASE_BLOCK);

    if constexpr (LAYOUT_T == DLILayout::TND) {
        constInfo.softmaxHeadOffset = tilingData->multiCoreParams.padTotalSize;
    } else if constexpr (LAYOUT_T == DLILayout::BSND) {
        constInfo.softmaxHeadOffset = constInfo.s1Size;
    }

    auto &initOutputParams = tilingData->initOutputParams;
    constInfo.dKeyGmOffset = initOutputParams.singleCoreSize * constInfo.aivIdx;
    if (constInfo.aivIdx == constInfo.aivNum - 1) {
        constInfo.dKeySingleCoreSize = initOutputParams.totalOutputSize - constInfo.dKeyGmOffset;
    } else {
        constInfo.dKeySingleCoreSize = initOutputParams.singleCoreSize;
    }

    if constexpr (LAYOUT_T == DLILayout::TND) {
        // TND场景可能存在用来pad的无效T，需要清零
        if (constInfo.aivIdx >= initOutputParams.t1PadRemainderSize) {
            constInfo.dwPadSize = initOutputParams.t1PadSingleCoreSize * constInfo.n1IndexSize;
            constInfo.dqPadSize = constInfo.dwPadSize * constInfo.dSizeQueryIndex;
            constInfo.dwPadOffset = (constInfo.t1Size + constInfo.aivIdx * initOutputParams.t1PadSingleCoreSize +
                                     initOutputParams.t1PadRemainderSize) *
                                    constInfo.n1IndexSize;
            constInfo.dqPadOffset = constInfo.dwPadOffset * constInfo.dSizeQueryIndex;
        } else {
            constInfo.dwPadSize = (initOutputParams.t1PadSingleCoreSize + 1) * constInfo.n1IndexSize;
            constInfo.dqPadSize = constInfo.dwPadSize * constInfo.dSizeQueryIndex;
            constInfo.dwPadOffset =
                (constInfo.t1Size + constInfo.aivIdx * initOutputParams.t1PadSingleCoreSize + constInfo.aivIdx) *
                constInfo.n1IndexSize;
            constInfo.dqPadOffset = constInfo.dwPadOffset * constInfo.dSizeQueryIndex;
        }

        if (constInfo.aivIdx >= initOutputParams.t2PadRemainderSize) {
            constInfo.dkPadSize =
                initOutputParams.t2PadSingleCoreSize * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
            constInfo.dkPadOffset =
                initOutputParams.totalOutputSize +
                (constInfo.aivIdx * initOutputParams.t2PadSingleCoreSize + initOutputParams.t2PadRemainderSize) *
                    constInfo.dSizeQueryIndex;
        } else {
            constInfo.dkPadSize =
                (initOutputParams.t2PadSingleCoreSize + 1) * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
            constInfo.dkPadOffset = initOutputParams.totalOutputSize +
                                    (constInfo.aivIdx * initOutputParams.t2PadSingleCoreSize + constInfo.aivIdx) *
                                        constInfo.dSizeQueryIndex;
        }
    }

    // 确定性参数
    constInfo.dKeyDeterGmLength = S2_BASE_STEP * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
    constInfo.dKeyDeterGmOffset = constInfo.aicIdx * constInfo.dKeyDeterGmLength;
    constInfo.maxLoopSize = baseInfo.maxLoopSize;
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossBase<DLIT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    int64_t bmm1Offset = constInfo.n1Size * S1_BASE_STEP * S2_BASE_STEP * sizeof(float);      // * 2;
    int64_t bmm2Offset = constInfo.n1IndexSize * S1_BASE_STEP * S2_BASE_STEP * sizeof(float); // * 2;
    int64_t psySyncSize = AIC_AIV_RATIO * S1_VEC_SIZE_8 * S2_BASE_STEP * sizeof(float);
    int64_t dWeightFloatSzie = S1_BASE_STEP * constInfo.n1IndexSize * sizeof(float);
    if constexpr (IsSameType<W_T, float>::value) {
        dWeightFloatSzie = 0;
    }
    int64_t reluGradSize = constInfo.n1IndexSize * S1_BASE_STEP * S2_BASE_STEP * sizeof(float); // * 2;
    int64_t dKeyIndexFloatSzie =
        constInfo.bSize * constInfo.s2Size * constInfo.n2IndexSize * constInfo.dSizeQueryIndex * sizeof(float);
    if constexpr (LAYOUT_T == DLILayout::TND) {
        int64_t t2Size = this->actualSeqLengthsKeyGm.GetValue(constInfo.bSize - 1);
        dKeyIndexFloatSzie = t2Size * constInfo.n2IndexSize * constInfo.dSizeQueryIndex * sizeof(float);
    }
    int64_t dQueryIndexFloatSzie = S1_BASE_STEP * constInfo.n1IndexSize * constInfo.dSizeQueryIndex * sizeof(float);
    int64_t coreTotalOffset = constInfo.aicIdx * (bmm1Offset * 2 + bmm2Offset * 2 + psySyncSize * 2 * DOUBLE_BUFFER +
                                                  reluGradSize * 2 + dWeightFloatSzie + dQueryIndexFloatSzie);

    uint64_t offset = 0;

    // 每个核共用同一块GM
    if constexpr (deterministic) {
        int64_t dKeyIndexDeterGmSize =
            constInfo.aicNum * S2_BASE_STEP * constInfo.n2IndexSize * constInfo.dSizeQueryIndex * sizeof(float);
        int64_t lossDeterGmSize = constInfo.aivNum * optiling::DETER_LOSS_TMP_GM_NUM * sizeof(float);
        int64_t coreInfoDeterGmSize = constInfo.aicNum * optiling::DETER_CORE_INFO_TMP_GM_NUM * sizeof(int64_t);
        dKeyIndexDeterGmFloat.SetGlobalBuffer((__gm__ T *)(workspace + offset));
        offset += dKeyIndexDeterGmSize;
        lossGmDeterFloat.SetGlobalBuffer((__gm__ T *)(workspace + offset));
        offset += lossDeterGmSize;
        deterCoreInfoGm.SetGlobalBuffer((__gm__ INFO_INT_64_T *)(workspace + offset));
        offset += coreInfoDeterGmSize;
    }
    dKeyIndexGmFloat.SetGlobalBuffer((__gm__ T *)(workspace + offset));
    offset += dKeyIndexFloatSzie;

    // workspace 按核分, 每个核内不同workspace相邻
    bmm1Res.SetGlobalBuffer((__gm__ MM12_OUT_T *)(workspace + offset + coreTotalOffset));
    offset += bmm1Offset * 2;

    bmm2Res.SetGlobalBuffer((__gm__ MM12_OUT_T *)(workspace + offset + coreTotalOffset));
    reluGm.SetGlobalBuffer((__gm__ T *)(bmm2Res.GetPhyAddr()));
    offset += bmm2Offset * 2;

    pSyncGm.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += psySyncSize * 2;

    sySyncGm.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += psySyncSize * 2;

    if constexpr (!IsSameType<W_T, float>::value) {
        dWeightGmFloat.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
        offset += dWeightFloatSzie;
    }

    dQueryIndexGmFloat.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += dQueryIndexFloatSzie;

    reluGradRes.SetGlobalBuffer((__gm__ KV_T *)(workspace + offset + coreTotalOffset));
    offset += reluGradSize * 2;

    if ASCEND_IS_AIV {
        uint32_t dWeightVecSize = dWeightFloatSzie / AIC_AIV_RATIO / sizeof(T);
        uint32_t dQueryVecSize = dQueryIndexFloatSzie / AIC_AIV_RATIO / sizeof(T);
        if (constInfo.dKeySingleCoreSize > 0) {
            AscendC::InitOutput(dKeyIndexGmFloat[constInfo.dKeyGmOffset], constInfo.dKeySingleCoreSize,
                                static_cast<T>(0));
        }
        if constexpr (deterministic) {
            AscendC::InitOutput(dKeyIndexDeterGmFloat[constInfo.dKeyDeterGmOffset], constInfo.dKeyDeterGmLength,
                                static_cast<T>(0));
            AscendC::InitOutput(lossGmDeterFloat[constInfo.aivIdx * optiling::DETER_LOSS_TMP_GM_NUM],
                                optiling::DETER_LOSS_TMP_GM_NUM, static_cast<T>(0));
        }
    }
    SyncAll();
}

template <typename DLIT>
__aicore__ inline int64_t DenseLightningIndexerGradKLLossBase<DLIT>::FindBIndex(int64_t bIndex, int64_t curIndex,
                                                                                int64_t &accumulateLen)
{
    for (int index = bIndex; index < constInfo.bSize; index++) {
        int64_t actualLen = this->actualSeqLengthsQueryGm.GetValue(index);
        if (curIndex < actualLen) {
            return index;
        }
        accumulateLen = actualLen;
    }
    return tilingData->multiCoreParams.totalSize >= curIndex ? constInfo.bSize : -1;
}

template <typename DLIT>
__aicore__ inline int64_t DenseLightningIndexerGradKLLossBase<DLIT>::GetEndS1Etx(int32_t bIdx, int32_t defaultLens,
                                                                                 GlobalTensor<int64_t> &actualSeqLensGm,
                                                                                 DLILayout layout)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        return defaultLens;
    }

    if (layout == DLILayout::TND) {
        if (bIdx == 0) {
            return actualSeqLensGm.GetValue(0);
        } else {
            return (actualSeqLensGm.GetValue(bIdx) - actualSeqLensGm.GetValue(bIdx - 1));
        }
    } else {
        assert(false, "do not support current layout!\n");
        return 0;
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossBase<DLIT>::CalcMultiCoreOffset(int64_t &bStartIdx,
                                                                                      int64_t &s1StartIdx,
                                                                                      int64_t &bEndIdx,
                                                                                      int64_t &s1EndIdx)
{
    int64_t actualSum = 0;
    int64_t bS1Index = tilingData->multiCoreParams.bS1Index[constInfo.aicIdx];
    int64_t bS1EndIndex = constInfo.aicIdx + 1 < constInfo.aicNum ?
                              tilingData->multiCoreParams.bS1Index[constInfo.aicIdx + 1] :
                              tilingData->multiCoreParams.totalSize;
    if constexpr (LAYOUT_T == DLILayout::TND) {
        bStartIdx = FindBIndex(0, bS1Index, actualSum);
        s1StartIdx = bS1Index - actualSum;
        bEndIdx = FindBIndex(bStartIdx, bS1EndIndex - 1, actualSum);
        s1EndIdx = bS1EndIndex - actualSum;
    } else {
        bStartIdx = bS1Index / constInfo.s1Size;
        bEndIdx = (bS1EndIndex - 1) / constInfo.s1Size;
        s1StartIdx = bS1Index - bStartIdx * constInfo.s1Size;
        s1EndIdx = bS1EndIndex - bEndIdx * constInfo.s1Size;
    }
}

template <typename DLIT>
__aicore__ inline int32_t DenseLightningIndexerGradKLLossBase<DLIT>::GetActualSeqLens(
    int32_t bIdx, int32_t defaultLens, GlobalTensor<int64_t> &actualSeqLensGm, DLILayout layout, int64_t &accumLen)
{
    if (actualSeqLensGm.GetSize() <= 0) {
        accumLen = bIdx * defaultLens;
        return defaultLens;
    }

    if (layout == DLILayout::TND) {
        if (bIdx == 0) {
            accumLen = 0;
            return actualSeqLensGm.GetValue(0);
        } else {
            accumLen = actualSeqLensGm.GetValue(bIdx - 1);
            return (actualSeqLensGm.GetValue(bIdx) - accumLen);
        }
    } else {
        accumLen = bIdx * defaultLens;
        return defaultLens;
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossBase<DLIT>::GetRunInfo(
    int64_t taskId, int64_t bIdx, int64_t s1Idx, int64_t s2Idx, uint32_t s1Size, uint32_t s2Size, int64_t accumS1Idx,
    int64_t accumS2Idx, uint32_t curS1Size, uint32_t curS2StepSize, int32_t sparseMaskStartIdx, int32_t s2EndIdx)
{
    auto &runInfo = runInfos[taskId % 3];
    if (s2Idx >= s2EndIdx) {
        runInfo.isValid = false;
        return;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;

    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    runInfo.s2Idx = s2Idx;
    runInfo.accumS1Idx = accumS1Idx + s1Idx;
    runInfo.accumS2Idx = accumS2Idx + s2Idx;

    runInfo.curS1Size = curS1Size;
    runInfo.curS1SizeVec = (curS1Size + constInfo.subBlockIdx) / AIC_AIV_RATIO;
    runInfo.curS1SizeAlign16 = CeilDiv(curS1Size, C0_SIZE) * C0_SIZE;
    runInfo.curS2StepSize = curS2StepSize;
    runInfo.curS2StepSizeAlign8 = CeilDiv(curS2StepSize, FLOAT_DATA_BLOCK_NUM) * FLOAT_DATA_BLOCK_NUM;
    runInfo.curS2LoopTimes = CeilDiv(curS2StepSize, constInfo.s2BaseBlk);
    runInfo.sparseMaskStartIdx = sparseMaskStartIdx;

    runInfo.queryTensorOffset = runInfo.accumS1Idx * constInfo.n1Size * constInfo.dSizeQuery;
    runInfo.queryRopeTensorOffset = runInfo.accumS1Idx * constInfo.n1Size * constInfo.dSizeQueryRope;
    runInfo.queryIndexTensorOffset = runInfo.accumS1Idx * constInfo.n1IndexSize * constInfo.dSizeQueryIndex;
    runInfo.keyTensorOffset = runInfo.accumS2Idx * constInfo.n2Size * constInfo.dSizeQuery;
    runInfo.keyRopeTensorOffset = runInfo.accumS2Idx * constInfo.n2Size * constInfo.dSizeQueryRope;
    runInfo.keyIndexTensorOffset = runInfo.accumS2Idx * constInfo.n2IndexSize * constInfo.dSizeQueryIndex;
    runInfo.weightTensorOffset = runInfo.accumS1Idx * constInfo.n1IndexSize +
                                 constInfo.subBlockIdx * curS1Size / AIC_AIV_RATIO * constInfo.n1IndexSize;
    runInfo.softmaxIndexTensorOffset = runInfo.accumS1Idx + constInfo.subBlockIdx * curS1Size / AIC_AIV_RATIO;

    if constexpr (LAYOUT_T == DLILayout::TND) {
        runInfo.softmaxTensorOffset = runInfo.accumS1Idx + constInfo.subBlockIdx * curS1Size / AIC_AIV_RATIO;
    } else if constexpr (LAYOUT_T == DLILayout::BSND) {
        runInfo.softmaxTensorOffset =
            bIdx * constInfo.n1Size * constInfo.s1Size + s1Idx + constInfo.subBlockIdx * curS1Size / AIC_AIV_RATIO;
    }

    runInfo.lastS2 = (s2Idx + S2_BASE_STEP >= s2EndIdx);
    runInfo.isValid = true;
}

template <typename DLIT>
__aicore__ inline int32_t DenseLightningIndexerGradKLLossBase<DLIT>::GetS2SparseLen(int32_t s1Idx,
                                                                                    int32_t actualSeqLensQ,
                                                                                    int32_t actualSeqLensK,
                                                                                    DLISparseMode sparseMode)
{
    if (sparseMode == DLISparseMode::RightDown) {
        return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
    } else {
        return 0;
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossBase<DLIT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *queryIndex, __gm__ uint8_t *keyIndex,
    __gm__ uint8_t *weight, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxMaxIndex,
    __gm__ uint8_t *softmaxSumIndex, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKey, __gm__ uint8_t *dQueryIndex,
    __gm__ uint8_t *dKeyIndex, __gm__ uint8_t *dWeight, __gm__ uint8_t *loss, __gm__ uint8_t *workspace,
    const optiling::DenseLightningIndexerGradKLLossTilingData *__restrict tiling, TPipe *tPipe)
{
    // init tiling data
    pipe = tPipe;
    tilingData = tiling;
    InitConstInfo();

    // init input GlobalTensor
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    queryIndexGm.SetGlobalBuffer((__gm__ Q_T *)queryIndex);
    keyIndexGm.SetGlobalBuffer((__gm__ KV_T *)keyIndex);
    weightGm.SetGlobalBuffer((__gm__ W_T *)weight);
    softmaxMaxGm.SetGlobalBuffer((__gm__ T *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ T *)softmaxSum);
    softmaxMaxIndexGm.SetGlobalBuffer((__gm__ T *)softmaxMaxIndex);
    softmaxSumIndexGm.SetGlobalBuffer((__gm__ T *)softmaxSumIndex);
    if constexpr (DLIT::hasRope) {
        queryRopeGm.SetGlobalBuffer((__gm__ Q_ROPE_T *)queryRope);
        keyRopeGm.SetGlobalBuffer((__gm__ K_ROPE_T *)keyRope);
    }
    if (actualSeqLengthsQuery != nullptr) {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQuery, constInfo.bSize);
    } else {
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsQuery, 0);
    }
    if (actualSeqLengthsKey != nullptr) {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKey, constInfo.bSize);
    } else {
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKey, 0);
    }

    // init output GlobalTensor
    dQueryIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dQueryIndex);
    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKeyIndex);
    dWeightGm.SetGlobalBuffer((__gm__ W_T *)dWeight);
    lossGm.SetGlobalBuffer((__gm__ T *)loss);
    lossGm.SetValue(0, 0.0F);
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        lossGm);
    if (constInfo.dqPadSize > 0) {
        AscendC::InitOutput(dQueryIndexGm[constInfo.dqPadOffset], constInfo.dqPadSize, static_cast<OUT_T>(0));
    }
    if (constInfo.dkPadSize > 0) {
        AscendC::InitOutput(dKeyIndexGm[constInfo.dkPadOffset], constInfo.dkPadSize, static_cast<OUT_T>(0));
    }
    if (constInfo.dwPadSize > 0) {
        AscendC::InitOutput(dWeightGm[constInfo.dwPadOffset], constInfo.dwPadSize, static_cast<W_T>(0));
    }

    // init wordspace
    InitWorkspace(workspace);

    // init cubeService and vectorService
    if ASCEND_IS_AIV {
        // InitVecOP
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitBuffers(pipe);
        vectorService.InitVector1GM(softmaxMaxGm, softmaxSumGm, softmaxMaxIndexGm, softmaxSumIndexGm, bmm1Res, bmm2Res,
                                    weightGm, pSyncGm, sySyncGm, lossGm, dWeightGmFloat, reluGm, reluGradRes, dWeightGm,
                                    dQueryIndexGmFloat, dQueryIndexGm);
        if constexpr (deterministic) {
            vectorService.InitVector1DeterGM(deterCoreInfoGm, dKeyIndexDeterGmFloat, dKeyIndexGmFloat,
                                             lossGmDeterFloat);
        }
    } else if ASCEND_IS_AIC {
        // initCubeOP
        matmulService.InitParams(constInfo);
        matmulService.InitBuffers(pipe);
        matmulService.InitMm1GlobalTensor(queryGm, keyGm, queryRopeGm, keyRopeGm, actualSeqLengthsQueryGm,
                                          actualSeqLengthsKeyGm, bmm1Res, dKeyIndex);
        matmulService.InitMm2GlobalTensor(queryIndexGm, keyIndexGm, bmm2Res);
        matmulService.InitMm5GlobalTensor(reluGradRes, dKeyIndexGmFloat);
        matmulService.InitMm6GlobalTensor(dQueryIndexGmFloat);
        if constexpr (deterministic) {
            matmulService.InitMm5DeterGlobalTensor(dKeyIndexDeterGmFloat);
        }
    }
}

template <typename DLIT>
__aicore__ inline void DenseLightningIndexerGradKLLossBase<DLIT>::Process()
{
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    int64_t bStartIdx, s1StartIdx, bEndIdx, s1EndIdx;
    CalcMultiCoreOffset(bStartIdx, s1StartIdx, bEndIdx, s1EndIdx);

    int64_t taskId = 0;
    uint32_t s2PreloadTail = 0;
    if constexpr (deterministic) {
        if ASCEND_IS_AIV {
            if (constInfo.aivIdx % 2 == 0) {
                vectorService.SaveDeterRunInfoInvalid();
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    for (int64_t bIdx = bStartIdx; bIdx <= bEndIdx; bIdx++) {
        bool lastB = (bIdx == bEndIdx);
        int64_t s1StartIdxThisBatch = 0;
        int64_t s1EndIdxThisBatch = 0;
        if constexpr (LAYOUT_T == DLILayout::TND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            s1EndIdxThisBatch =
                (!lastB) ? GetEndS1Etx(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T) : s1EndIdx;
        } else if constexpr (LAYOUT_T == DLILayout::BSND) {
            s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
            ;
            s1EndIdxThisBatch = (!lastB) ? constInfo.s1Size : s1EndIdx;
        }

        for (int64_t s1Idx = s1StartIdxThisBatch; s1Idx < s1EndIdxThisBatch; s1Idx += S1_BASE_STEP) {
            bool lastS1 = ((s1Idx + S1_BASE_STEP) >= s1EndIdxThisBatch);
            int64_t accumS1Idx, accumS2Idx;
            uint32_t actualSeqLensQ =
                GetActualSeqLens(bIdx, constInfo.s1Size, actualSeqLengthsQueryGm, LAYOUT_T, accumS1Idx);
            uint32_t actualSeqLensK =
                GetActualSeqLens(bIdx, constInfo.s2Size, actualSeqLengthsKeyGm, KV_LAYOUT_T, accumS2Idx);
            int64_t s1EndIdxThisLoop = lastS1 ? s1EndIdxThisBatch : (s1Idx + S1_BASE_STEP);
            uint32_t curS1Size = s1EndIdxThisLoop - s1Idx;
            // idx需要减1
            int32_t s2EndIdx =
                GetS2SparseLen(s1EndIdxThisLoop, actualSeqLensQ, actualSeqLensK, constInfo.sparseMode) - 1;
            // 最后一个BS需要额外循环两次，因为preload方式会产生尾巴
            if (lastB && lastS1) {
                s2PreloadTail = S2_BASE_STEP * 1;
            }
            for (int64_t s2Idx = 0; s2Idx < s2EndIdx + s2PreloadTail; s2Idx += S2_BASE_STEP) {
                bool lastS2 = ((s2Idx + S2_BASE_STEP) >= s2EndIdx);

                // 更新本次循环的参数
                uint32_t s2EndIdxThisLoop = lastS2 ? s2EndIdx : (s2Idx + S2_BASE_STEP);
                uint32_t curS2StepSize = s2EndIdxThisLoop - s2Idx;
                uint32_t sparseMaskStartIdx = s2EndIdx - curS1Size;
                int32_t sparseMaskStartIdxThisLoop = 0;

                if (sparseMaskStartIdx >= (s2Idx + S2_BASE_STEP - 1)) {
                    // 表示当前S2Loop不需要处理S2mask
                    sparseMaskStartIdxThisLoop = S2_BASE_STEP - 1;
                } else if (sparseMaskStartIdx < s2Idx) {
                    // 表示当前s2Loop横切mask的对角线，只处理后一部分S1数据
                    sparseMaskStartIdxThisLoop = sparseMaskStartIdx - s2Idx;
                } else {
                    sparseMaskStartIdxThisLoop = sparseMaskStartIdx % S2_BASE_STEP;
                }

                GetRunInfo(taskId, bIdx, s1Idx, s2Idx, actualSeqLensQ, actualSeqLensK, accumS1Idx, accumS2Idx,
                           curS1Size, curS2StepSize, sparseMaskStartIdxThisLoop, s2EndIdx);

                DLIGradKLLossRunInfo &runInfo0 = runInfos[taskId % 3];          // 当前轮
                DLIGradKLLossRunInfo &runInfoNeg1 = runInfos[(taskId + 2) % 3]; // 上1轮
                DLIGradKLLossRunInfo &runInfoNeg2 = runInfos[(taskId + 1) % 3]; // 上2轮

                if ASCEND_IS_AIC {
                    if (runInfo0.isValid) {
                        matmulService.ComputeMm1(runInfo0); // C1
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_P_FLAG[runInfo0.taskIdMod2]);

                        matmulService.ComputeMm2(runInfo0); // C1
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_SY_FLAG[runInfo0.taskIdMod2]);
                    }
                }

                if ASCEND_IS_AIV {
                    if (runInfoNeg1.isValid) {
                        CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG[runInfoNeg1.taskIdMod2]);
                        CrossCoreWaitFlag(SYNC_C1_TO_V1_SY_FLAG[runInfoNeg1.taskIdMod2]);
                        vectorService.ProcessVector1(runInfoNeg1); // V1
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C2_DW_FLAG[runInfoNeg1.taskIdMod2]);
                    }
                }
                if ASCEND_IS_AIC {
                    if (runInfoNeg1.isValid) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V1_TO_C2_DW_FLAG[runInfoNeg1.taskIdMod2]);
                        if constexpr (deterministic) {
                            matmulService.ComputeMm34Deter(runInfoNeg1); // C2
                        } else {
                            matmulService.ComputeMm34(runInfoNeg1); // C2
                        }

                        if (runInfoNeg1.lastS2) {
                            // send msg for cast
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_DW_FLAG[runInfoNeg1.taskIdMod2]);
                        }
                    }
                }
                if constexpr (deterministic) {
                    if (constInfo.aivIdx % 2 == 0) {
                        if ASCEND_IS_AIV {
                            if (runInfoNeg1.isValid) {
                                vectorService.SaveDeterRunInfo(runInfoNeg1);
                            } else {
                                vectorService.SaveDeterRunInfoInvalid();
                            }
                        }
                    }
                    SyncAll<false>();
                    if ASCEND_IS_AIV {
                        vectorService.DeterAddKIndexGrad(runInfoNeg1);
                    }
                    SyncAll<false>();
                }
                if ASCEND_IS_AIV {
                    if (runInfoNeg1.isValid && runInfoNeg1.lastS2) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C2_TO_V2_DW_FLAG[runInfoNeg1.taskIdMod2]);
                        vectorService.CastOutWeightGrad(runInfoNeg1);
                        vectorService.CastOutQIndexGrad(runInfoNeg1);
                    }
                }
                taskId++;
            }
        }
    }

    if constexpr (deterministic) {
        for (int i = taskId; i < constInfo.maxLoopSize; i++) {
            if (constInfo.aivIdx % 2 == 0) {
                if ASCEND_IS_AIV {
                    vectorService.SaveDeterRunInfoInvalid();
                }
            }
            SyncAll<false>();
            if ASCEND_IS_AIV {
                vectorService.DeterAddKIndexGrad(runInfos[(taskId + 2) % 3]);
            }
            SyncAll<false>();
        }
    }
    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }
    if ASCEND_IS_AIV {
        vector2Service.InitParams(constInfo, tilingData);

        vector2Service.InitVector2GM(dKeyIndexGmFloat, dKeyIndexGm);
        if constexpr (deterministic) {
            vector2Service.InitVector2DeterGM(lossGm, lossGmDeterFloat);
        }
        vector2Service.InitBuffers(pipe);
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        if constexpr (deterministic) {
            vector2Service.DeterSumLoss();
        }
        vector2Service.ProcessVectorDk();
    }
}

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_BASE_H
