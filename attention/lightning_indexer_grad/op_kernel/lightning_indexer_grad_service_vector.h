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
 * \file lightning_indexer_grad_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_grad_common.h"
namespace LigKernel {
using namespace LIGCommon;
using namespace AscendC;

constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t LD_PARAM_NUM = 16;

template <typename LIGT>
class LIGVector {
public:
    using dataType = typename LIGT::dataType;

    __aicore__ inline LIGVector(){};
    __aicore__ inline void Init(TPipe *pipe);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();
    __aicore__ inline void InitOutputDqAndDweights(GlobalTensor<dataType> dweightsGmTensor,
                                                   GlobalTensor<dataType> dqTensor, LIGCommon::ConstInfo constInfo,
                                                   LIGCommon::RunInfo runInfo);
    __aicore__ inline void InitOutputDkcoreGm(GlobalTensor<float> dkCoreWorkspaceGM, LIGCommon::ConstInfo constInfo,
                                              LIGCommon::RunInfo runInfo);
    __aicore__ inline void GatherTopk(GlobalTensor<int32_t> sparseIndicesTensor, GlobalTensor<dataType> keyTensor,
                                      GlobalTensor<dataType> gatherKTensor, LIGCommon::ConstInfo constInfo,
                                      LIGCommon::RunInfo runInfo);
    __aicore__ inline void ScatterAdd(GlobalTensor<int32_t> sparseIndicesTensor, GlobalTensor<float> scatterAddTensor,
                                      GlobalTensor<float> dkWorkSpaceGmTensor, LIGCommon::ConstInfo constInfo,
                                      LIGCommon::RunInfo runInfo);
    __aicore__ inline void ReluGrad(GlobalTensor<float> reluInGmTensor, GlobalTensor<dataType> reluGradInGmTensor,
                                    GlobalTensor<dataType> dyGmTensor, GlobalTensor<dataType> reluGradOutGmTensor,
                                    GlobalTensor<dataType> dweightsGmTensor, LIGCommon::ConstInfo constInfo,
                                    LIGCommon::RunInfo runInfo);
    __aicore__ inline void DeterministicMerge(GlobalTensor<float> dkCoreWorkspaceGM, GlobalTensor<float> dkWorkSpaceGm,
                                              const LIGCommon::ConstInfo &constInfo, const LIGCommon::RunInfo &runInfo);

protected:
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;
    constexpr static int64_t LIMIT_TOPK = 2048;
    constexpr static int64_t LIMIT_GROUPNUM = 128;
    constexpr static uint16_t DATA_BLOCK_SIZE = 32;

    constexpr static int64_t gatherPingUbOffset = 0;
    constexpr static int64_t gatherPingUbSize = 128 * 4 * 8;

    constexpr static int64_t gatherPongUbOffset = gatherPingUbOffset + gatherPingUbSize;
    constexpr static int64_t gatherPongUbSize = 128 * 4 * 8;

    constexpr static int64_t indicesUbOffset = gatherPongUbOffset + gatherPongUbSize;
    constexpr static int64_t indicesUbSize = LIMIT_TOPK * 4;

    constexpr static int64_t reluInPingUbOffset = 0 + indicesUbSize;
    constexpr static int64_t reluInPingUbSize = 4 * LIMIT_TOPK * 4;

    constexpr static int64_t reluInPongUbOffset = reluInPingUbOffset + reluInPingUbSize;
    constexpr static int64_t reluInPongUbSize = 4 * LIMIT_TOPK * 4;

    constexpr static int64_t maskPingUbOffset = reluInPongUbOffset + reluInPongUbSize;
    constexpr static int64_t maskPingUbSize = LIMIT_TOPK / 2;

    constexpr static int64_t maskPongUbOffset = maskPingUbOffset + maskPingUbSize;
    constexpr static int64_t maskPongUbSize = LIMIT_TOPK / 2;

    constexpr static int64_t reluGradPingUbOffset = maskPongUbOffset + maskPongUbSize;
    constexpr static int64_t reluGradPingUbSize = 4 * LIMIT_TOPK * 2;

    constexpr static int64_t reluGradPongUbOffset = reluGradPingUbOffset + reluGradPingUbSize;
    constexpr static int64_t reluGradPongUbSize = 4 * LIMIT_TOPK * 2;

    constexpr static int64_t reluGradOutPingUbOffset = reluGradPongUbOffset + reluGradPongUbSize;
    constexpr static int64_t reluGradOutPingUbSize = 4 * LIMIT_TOPK * 2;

    constexpr static int64_t reluGradOutPongUbOffset = reluGradOutPingUbOffset + reluGradOutPingUbSize;
    constexpr static int64_t reluGradOutPongUbSize = 4 * LIMIT_TOPK * 2;

    constexpr static int64_t dyUbOffset = reluGradOutPongUbOffset + reluGradOutPongUbSize;
    constexpr static int64_t dyUbSize = LIMIT_TOPK * 2;

    constexpr static int64_t dyFloatUbOffset = dyUbOffset + dyUbSize;
    constexpr static int64_t dyFloatUbSize = LIMIT_TOPK * 4;

    constexpr static int64_t zeroFloatUbOffset = dyFloatUbOffset + dyFloatUbSize;
    constexpr static int64_t zeroFloatUbSize = 256;

    constexpr static int64_t reduceFloatUbOffset = zeroFloatUbOffset + zeroFloatUbSize;
    constexpr static int64_t reduceFloatUbSize = LIMIT_GROUPNUM * 4;

    constexpr static int64_t reduceUbOffset = reduceFloatUbOffset + reduceFloatUbSize;
    constexpr static int64_t reduceUbSize = LIMIT_GROUPNUM * 2;

    constexpr static int64_t zeroDataUbOffset = reduceUbOffset + reduceUbSize;
    constexpr static int64_t MAX_UB_SIZE = TOTAL_SIZE / sizeof(float) / 2;

    TPipe *pipe;
    TBuf<> unifiedBuffer;

    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdVToS;
    event_t eventIdMte3ToMte2;
    event_t eventIdVToMte2;

    event_t eventIdMte2ToMTE3;
    event_t eventIdMte3ToMTE2;
    event_t eventIdVToMte2Ping;
    event_t eventIdVToMte2Pong;
    event_t eventIdMte3ToVPing;
    event_t eventIdMte3ToVPong;
    event_t eventIdMte2ToMTE3Ping;
    event_t eventIdMte2ToMTE3Pong;
    event_t eventIdMte3ToMTE2Ping;
    event_t eventIdMte3ToMTE2Pong;
};

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::Init(TPipe *pipeIn)
{
    pipe = pipeIn;
    InitBuffers();
    AllocEvents();
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::AllocEvents()
{
    eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdVToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
    eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());

    eventIdVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    eventIdMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

    eventIdMte2ToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    eventIdMte3ToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdMte2ToMTE3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    eventIdMte2ToMTE3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    eventIdMte3ToMTE2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdMte3ToMTE2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventIdVToS);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);

    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToVPong);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Pong);
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::InitBuffers()
{
    pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
    uint64_t ubSize = gatherPingUbSize + gatherPongUbSize + indicesUbSize + reluInPingUbSize + reluInPongUbSize +
                      maskPingUbSize + maskPongUbSize + reluGradPingUbSize + reluGradOutPingUbSize + dyUbSize +
                      dyFloatUbSize + zeroFloatUbSize + reduceFloatUbSize + reduceUbSize;
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::GatherTopk(GlobalTensor<int32_t> sparseIndicesTensor,
                                                   GlobalTensor<dataType> keyTensor,
                                                   GlobalTensor<dataType> gatherKTensor, LIGCommon::ConstInfo constInfo,
                                                   LIGCommon::RunInfo runInfo)
{
    LocalTensor<int32_t> indiceUb =
        unifiedBuffer.GetWithOffset<int32_t>(indicesUbSize / sizeof(int32_t), indicesUbOffset);
    LocalTensor<dataType> gatherPingUb =
        unifiedBuffer.GetWithOffset<dataType>(gatherPingUbSize / sizeof(dataType), gatherPingUbOffset);
    LocalTensor<dataType> gatherPongUb =
        unifiedBuffer.GetWithOffset<dataType>(gatherPongUbSize / sizeof(dataType), gatherPongUbOffset);

    uint64_t loopBegin = (GetBlockIdx() % 2 == 0) ? 0 : runInfo.realTopk / 2;
    uint64_t loopEnd = (GetBlockIdx() % 2 == 0) ? runInfo.realTopk / 2 : runInfo.realTopk;
    // [B, S1, K]
    uint64_t indicesOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        indicesOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.topK + runInfo.s1Idx * constInfo.topK;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        indicesOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.topK;
    }

    DataCopy(indiceUb, sparseIndicesTensor[indicesOffset], (runInfo.realTopk + 7) / 8 * 8);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    AscendC::SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    AscendC::WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Ping);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Pong);

    uint64_t pingPongOffset = 0;
    uint8_t splitDataCopyLen = 8;
    for (uint64_t i = loopBegin, cnt = 0; i < loopEnd; i++, cnt++) {
        event_t eventIdMte2ToMTE3PingPong =
            ((cnt / splitDataCopyLen) & 1) ? eventIdMte2ToMTE3Ping : eventIdMte2ToMTE3Pong;
        event_t eventIdMte3ToMTE2PingPong =
            ((cnt / splitDataCopyLen) & 1) ? eventIdMte3ToMTE2Ping : eventIdMte3ToMTE2Pong;
        LocalTensor<dataType> &gatherPingPongUb = ((cnt / splitDataCopyLen) & 1) ? gatherPingUb : gatherPongUb;
        int64_t singleIndice = indiceUb.GetValue(i);
        bool isDualPrefetch = (i + 1 < loopEnd) && (indiceUb.GetValue(i + 1) == singleIndice + 1) &&
                              ((cnt % splitDataCopyLen) < splitDataCopyLen - 1);
        uint64_t copyLen = isDualPrefetch ? (2 * constInfo.headDim) : constInfo.headDim;
        // [B, S2, N2, D]
        uint64_t keyOffset = 0;
        if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
            keyOffset = runInfo.bIdx * constInfo.seqlenK * constInfo.headNumK * constInfo.headDim +
                        singleIndice * constInfo.headNumK * constInfo.headDim + runInfo.n2Idx * constInfo.headDim;
        } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
            keyOffset = runInfo.prefixSumS2 * constInfo.headNumK * constInfo.headDim +
                        singleIndice * constInfo.headNumK * constInfo.headDim + runInfo.n2Idx * constInfo.headDim;
        }
        pingPongOffset = cnt % splitDataCopyLen * constInfo.headDim;
        if ((cnt % splitDataCopyLen == 0) || (i == loopBegin)) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2PingPong);
        }
        DataCopy(gatherPingPongUb[pingPongOffset], keyTensor[keyOffset], copyLen);
        if (isDualPrefetch) {
            i++;
            cnt++;
        }
        if (((cnt + 1) % splitDataCopyLen == 0) || (i >= loopEnd - 1)) {
            uint64_t gatherKOffset = (loopBegin + cnt / splitDataCopyLen * splitDataCopyLen) * constInfo.headDim;
            uint64_t splitNum = ((cnt + 1) % splitDataCopyLen == 0) ? splitDataCopyLen : ((cnt + 1) % splitDataCopyLen);
            if (splitNum > 0) {
                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3PingPong);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3PingPong);
                DataCopy(gatherKTensor[gatherKOffset], gatherPingPongUb, constInfo.headDim * splitNum);
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2PingPong);
            }
        }
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Ping);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Pong);
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::ScatterAdd(GlobalTensor<int32_t> sparseIndicesTensor,
                                                   GlobalTensor<float> scatterAddTensor,
                                                   GlobalTensor<float> dkWorkSpaceGmTensor,
                                                   LIGCommon::ConstInfo constInfo, LIGCommon::RunInfo runInfo)
{
    if (runInfo.realTopk <= 0) {
        return;
    }
    LocalTensor<int32_t> indiceUb =
        unifiedBuffer.GetWithOffset<int32_t>(indicesUbSize / sizeof(int32_t), indicesUbOffset);
    LocalTensor<float> gatherPingUb =
        unifiedBuffer.GetWithOffset<float>(gatherPingUbSize / sizeof(float), gatherPingUbOffset);
    LocalTensor<float> gatherPongUb =
        unifiedBuffer.GetWithOffset<float>(gatherPongUbSize / sizeof(float), gatherPongUbOffset);

    int64_t currentCoreIndex = GetBlockIdx();
    uint64_t loopBegin = (currentCoreIndex % 2 == 0) ? 0 : runInfo.realTopk / 2;
    uint64_t loopEnd = (currentCoreIndex % 2 == 0) ? runInfo.realTopk / 2 : runInfo.realTopk;

    // [B, S1, K]
    uint64_t indicesOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        indicesOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.topK + runInfo.s1Idx * constInfo.topK;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        indicesOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.topK;
    }
    DataCopy(indiceUb, sparseIndicesTensor[indicesOffset], (runInfo.realTopk + 7) / 8 * 8);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    AscendC::SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    AscendC::WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Ping);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Pong);

    uint64_t batch = 0; // 批次计数，每处理一个批次（可能含1或2个索引）递增1
    for (uint64_t i = loopBegin; i < loopEnd;) {
        bool canMerge = (i + 1 < loopEnd) && (indiceUb.GetValue(i + 1) == indiceUb.GetValue(i) + 1);
        uint64_t copyLen = canMerge ? (2 * constInfo.headDim) : constInfo.headDim;
        uint64_t step = canMerge ? 2 : 1;

        event_t eventIdMte2ToMTE3 = (batch & 1) ? eventIdMte2ToMTE3Ping : eventIdMte2ToMTE3Pong;
        event_t eventIdMte3ToMTE2 = (batch & 1) ? eventIdMte3ToMTE2Ping : eventIdMte3ToMTE2Pong;
        LocalTensor<float> &gatherPingPongUb = (batch & 1) ? gatherPingUb : gatherPongUb;

        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2);

        uint64_t srcOffset = i * constInfo.headDim;
        DataCopy(gatherPingPongUb, scatterAddTensor[srcOffset], copyLen);

        // 计算目标偏移（基于第一个索引）
        int64_t singleIndice = indiceUb.GetValue(i);
        uint64_t dkeyOffset;
        if (unlikely(constInfo.deterministic)) {
            if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
                dkeyOffset =
                    singleIndice * constInfo.headDim + currentCoreIndex / 2 * constInfo.seqlenK * constInfo.headDim;
            } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
                dkeyOffset =
                    singleIndice * constInfo.headDim + currentCoreIndex / 2 * runInfo.actualSeqK * constInfo.headDim;
            }
        } else {
            if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
                dkeyOffset = runInfo.bIdx * constInfo.seqlenK * constInfo.headNumK * constInfo.headDim +
                             singleIndice * constInfo.headNumK * constInfo.headDim + runInfo.n2Idx * constInfo.headDim;
            } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
                dkeyOffset = runInfo.prefixSumS2 * constInfo.headNumK * constInfo.headDim +
                             singleIndice * constInfo.headNumK * constInfo.headDim + runInfo.n2Idx * constInfo.headDim;
            }
        }

        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3);
        AscendC::SetAtomicAdd<float>();
        DataCopy(dkWorkSpaceGmTensor[dkeyOffset], gatherPingPongUb, copyLen);
        AscendC::SetAtomicNone();
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2);

        i += step;
        batch++;
    }

    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Ping);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2Pong);
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::DeterministicMerge(GlobalTensor<float> dkCoreWorkspaceGM,
                                                           GlobalTensor<float> dkWorkSpaceGm,
                                                           const LIGCommon::ConstInfo &constInfo,
                                                           const LIGCommon::RunInfo &runInfo)
{
    uint64_t determinLen = constInfo.determinLen;
    uint64_t determinBeginPos = constInfo.determinBeginPos;
    if (determinLen <= 0) {
        return;
    }

    uint64_t maxTileRows = MAX_UB_SIZE / constInfo.headDim;
    uint64_t stridePerRow = constInfo.headNumK * constInfo.headDim;
    LocalTensor<float> dkCoreWorkspaceUb = unifiedBuffer.GetWithOffset<float>(MAX_UB_SIZE, gatherPingUbOffset);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2);
    for (int core = 0; core < constInfo.splitCores; core++) {
        uint64_t processed = 0;
        while (processed < determinLen) {
            uint64_t tileRows = maxTileRows < (determinLen - processed) ? maxTileRows : (determinLen - processed);
            uint64_t dkCoreWorkspaceOffset;
            if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
                dkCoreWorkspaceOffset = static_cast<uint64_t>(core) * constInfo.seqlenK * constInfo.headDim +
                                        (determinBeginPos + processed) * constInfo.headDim;
            } else {
                dkCoreWorkspaceOffset =
                    core * runInfo.actualSeqK * constInfo.headDim + (determinBeginPos + processed) * constInfo.headDim;
            }

            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2);
            DataCopy(dkCoreWorkspaceUb, dkCoreWorkspaceGM[dkCoreWorkspaceOffset], tileRows * constInfo.headDim);
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMTE3);

            uint64_t baseDkOffset;
            if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
                baseDkOffset = runInfo.bIdx * constInfo.seqlenK * constInfo.headNumK * constInfo.headDim;
            } else {
                // prefixSumS2 is number of previous tokens; target starts from prefixSumS2
                baseDkOffset = runInfo.prefixSumS2 * constInfo.headNumK * constInfo.headDim;
            }

            baseDkOffset += runInfo.n2Idx * constInfo.headDim;
            uint64_t dkeyOffset = baseDkOffset + (determinBeginPos + processed) * stridePerRow;
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = tileRows;
            dataCopyParams.blockLen = constInfo.headDim * sizeof(float) / 32;
            dataCopyParams.dstStride = (stridePerRow - constInfo.headDim) * sizeof(float) / 32;
            dataCopyParams.srcStride = 0;

            AscendC::SetAtomicAdd<float>();
            DataCopy(dkWorkSpaceGm[dkeyOffset], dkCoreWorkspaceUb, dataCopyParams);
            AscendC::SetAtomicNone();
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2);
            processed += tileRows;
        }
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMTE2);
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::InitOutputDqAndDweights(GlobalTensor<dataType> dweightsGmTensor,
                                                                GlobalTensor<dataType> dqTensor,
                                                                LIGCommon::ConstInfo constInfo,
                                                                LIGCommon::RunInfo runInfo)
{
    uint32_t maxLen = constInfo.groupNum * constInfo.headDim;
    LocalTensor<dataType> zeroDataTensor = unifiedBuffer.GetWithOffset<dataType>(maxLen, zeroDataUbOffset);

    uint64_t blockGroupBegin = (GetBlockIdx() % 2 == 0) ? 0 : constInfo.groupNum / 2;
    uint64_t blockGroupNum = (GetBlockIdx() % 2 == 0) ? constInfo.groupNum / 2 : (constInfo.groupNum + 1) / 2;

    uint64_t dweightsGmOffset = 0;
    uint64_t dqGmOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        dweightsGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ + runInfo.s1Idx * constInfo.headNumQ +
                           runInfo.n2Idx * constInfo.groupNum + blockGroupBegin;
        dqGmOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ * constInfo.headDim +
                     runInfo.s1Idx * constInfo.headNumQ * constInfo.headDim +
                     runInfo.n2Idx * constInfo.groupNum * constInfo.headDim + blockGroupBegin * constInfo.headDim;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        dweightsGmOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ +
                           runInfo.n2Idx * constInfo.groupNum + blockGroupBegin;
        dqGmOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ * constInfo.headDim +
                     runInfo.n2Idx * constInfo.groupNum * constInfo.headDim + blockGroupBegin * constInfo.headDim;
    }

    AscendC::Duplicate(zeroDataTensor, static_cast<dataType>(0.0), maxLen);
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::DataCopy(dweightsGmTensor[dweightsGmOffset], zeroDataTensor, blockGroupNum);
    AscendC::DataCopy(dqTensor[dqGmOffset], zeroDataTensor, blockGroupNum * constInfo.headDim);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::InitOutputDkcoreGm(GlobalTensor<float> dkCoreWorkspaceGM,
                                                           LIGCommon::ConstInfo constInfo, LIGCommon::RunInfo runInfo)
{
    uint64_t dkCorePerSize;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        dkCorePerSize = static_cast<uint64_t>(constInfo.seqlenK) * constInfo.headDim / 2;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        dkCorePerSize = runInfo.actualSeqK * constInfo.headDim / 2;
    }

    LocalTensor<float> zeroDataTensor = unifiedBuffer.GetWithOffset<float>(MAX_UB_SIZE, gatherPingUbOffset);
    AscendC::Duplicate(zeroDataTensor, static_cast<float>(0.0), MAX_UB_SIZE);
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    uint64_t processed = 0;
    while (processed < dkCorePerSize) {
        uint64_t curBlockSize = MAX_UB_SIZE < (dkCorePerSize - processed) ? MAX_UB_SIZE : (dkCorePerSize - processed);
        AscendC::DataCopy(dkCoreWorkspaceGM[GetBlockIdx() * dkCorePerSize + processed], zeroDataTensor, curBlockSize);
        processed += curBlockSize;
    }

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

// reluGrad calc elements num should be devided by two block in groupNum axis
template <typename LIGT>
__aicore__ inline void LIGVector<LIGT>::ReluGrad(GlobalTensor<float> reluInGmTensor,
                                                 GlobalTensor<dataType> reluGradInGmTensor,
                                                 GlobalTensor<dataType> dyGmTensor,
                                                 GlobalTensor<dataType> reluGradOutGmTensor,
                                                 GlobalTensor<dataType> dweightsGmTensor,
                                                 LIGCommon::ConstInfo constInfo, LIGCommon::RunInfo runInfo)
{
    LocalTensor<float> reluInPingTensor =
        unifiedBuffer.GetWithOffset<float>(reluInPingUbSize / sizeof(float), reluInPingUbOffset);
    LocalTensor<float> reluInPongTensor =
        unifiedBuffer.GetWithOffset<float>(reluInPongUbSize / sizeof(float), reluInPongUbOffset);

    LocalTensor<dataType> reluGradPingTensor =
        unifiedBuffer.GetWithOffset<dataType>(reluGradPingUbSize / sizeof(dataType), reluGradPingUbOffset);
    LocalTensor<dataType> reluGradPongTensor =
        unifiedBuffer.GetWithOffset<dataType>(reluGradPongUbSize / sizeof(dataType), reluGradPongUbOffset);

    LocalTensor<dataType> reluGradOutPingTensor =
        unifiedBuffer.GetWithOffset<dataType>(reluGradOutPingUbSize / sizeof(dataType), reluGradOutPingUbOffset);
    LocalTensor<dataType> reluGradOutPongTensor =
        unifiedBuffer.GetWithOffset<dataType>(reluGradOutPongUbSize / sizeof(dataType), reluGradOutPongUbOffset);

    LocalTensor<uint8_t> maskPingTensor =
        unifiedBuffer.GetWithOffset<uint8_t>(maskPingUbSize / sizeof(uint8_t), maskPingUbOffset);
    LocalTensor<uint8_t> maskPongTensor =
        unifiedBuffer.GetWithOffset<uint8_t>(maskPongUbSize / sizeof(uint8_t), maskPongUbOffset);

    LocalTensor<float> zeroFloatTensor =
        unifiedBuffer.GetWithOffset<float>(zeroFloatUbSize / sizeof(float), zeroFloatUbOffset);
    LocalTensor<dataType> dyTensor = unifiedBuffer.GetWithOffset<dataType>(dyUbSize / sizeof(dataType), dyUbOffset);
    LocalTensor<float> dyFloatTensor =
        unifiedBuffer.GetWithOffset<float>(dyFloatUbSize / sizeof(float), dyFloatUbOffset);
    LocalTensor<float> reduceFloatTensor =
        unifiedBuffer.GetWithOffset<float>(reduceFloatUbSize / sizeof(float), reduceFloatUbOffset);
    LocalTensor<dataType> reduceTensor =
        unifiedBuffer.GetWithOffset<dataType>(reduceUbSize / sizeof(dataType), reduceUbOffset);

    LocalTensor<float> reduceSumWorkSpaceTensor =
        unifiedBuffer.GetWithOffset<float>(gatherPingUbSize / sizeof(float), gatherPingUbOffset);

    AscendC::Duplicate(zeroFloatTensor, static_cast<float>(0.0), zeroFloatUbSize / sizeof(float));
    AscendC::PipeBarrier<PIPE_V>();

    uint64_t totalElements = constInfo.groupNum * runInfo.realTopk;
    uint64_t blockGroupBegin = (GetBlockIdx() % 2 == 0) ? 0 : constInfo.groupNum / 2;
    uint64_t blockGroupNum = (GetBlockIdx() % 2 == 0) ? constInfo.groupNum / 2 : (constInfo.groupNum + 1) / 2;
    uint64_t floatMask = 64;
    uint64_t halfMask = 128;
    uint64_t groupNumDivide = 4;
    uint64_t groupNumTail = (blockGroupNum % groupNumDivide != 0) ? (blockGroupNum % groupNumDivide) : groupNumDivide;
    uint64_t loopTimes = (blockGroupNum + groupNumDivide - 1) / groupNumDivide;
    BinaryRepeatParams repeatParamsCompare = {0, 1, 0, 1, 8, 0};
    BinaryRepeatParams repeatParamsSelect = {1, 1, 0, 8, 8, 1};

    // [B, S1, K]
    uint64_t dyOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        dyOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.topK + runInfo.s1Idx * constInfo.topK;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        dyOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.topK;
    }
    uint64_t topkPadLenBf16 = (runInfo.realTopk + 15) / 16 * 16;
    uint64_t topkPadLenFp32 = (runInfo.realTopk + 7) / 8 * 8;
    uint32_t topkStride = (topkPadLenBf16 == topkPadLenFp32) ? 0 : 1;
    uint32_t garbageLen = topkPadLenBf16 - runInfo.realTopk;
    AscendC::DataCopyExtParams dyCopyParams{1, static_cast<uint32_t>(runInfo.realTopk * sizeof(dataType)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<dataType> dyPadParams{true, 0,
                                                        static_cast<uint8_t>(topkPadLenBf16 - runInfo.realTopk), 0};
    AscendC::DataCopyPad(dyTensor, dyGmTensor[dyOffset], dyCopyParams, dyPadParams);
    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Cast(dyFloatTensor, dyTensor, RoundMode::CAST_NONE, topkPadLenBf16);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping);
    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPing);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPong);
    for (uint64_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        uint64_t currentGroupNum = (loopIdx == loopTimes - 1) ? groupNumTail : groupNumDivide;
        uint64_t elementsNum = currentGroupNum * topkPadLenBf16;
        uint64_t groupOffset = (blockGroupBegin + loopIdx * groupNumDivide) * runInfo.realTopk;
        uint64_t elementsNumAlign = (elementsNum + 127) / 128 * 128;
        uint64_t compareRepeatTimes = elementsNumAlign / floatMask;
        uint64_t selectRepeatTimes = elementsNumAlign / halfMask;

        event_t eventIdVToMte2PingPong = (loopIdx & 1) ? eventIdVToMte2Ping : eventIdVToMte2Pong;
        event_t eventIdMte3ToVPingPong = (loopIdx & 1) ? eventIdMte3ToVPing : eventIdMte3ToVPong;
        LocalTensor<float> reluInTensor = (loopIdx & 1) ? reluInPingTensor : reluInPongTensor;
        LocalTensor<dataType> reluGradTensor = (loopIdx & 1) ? reluGradPingTensor : reluGradPongTensor;
        LocalTensor<dataType> reluGradOutTensor = (loopIdx & 1) ? reluGradOutPingTensor : reluGradOutPongTensor;
        LocalTensor<uint8_t> maskTensor = (loopIdx & 1) ? maskPingTensor : maskPongTensor;

        // copyIn reluGrad, reluIn
        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2PingPong);
        AscendC::DataCopyExtParams reluInCopyParams{static_cast<uint16_t>(currentGroupNum),
                                                    static_cast<uint32_t>(runInfo.realTopk * sizeof(float)), 0,
                                                    topkStride, 0};
        AscendC::DataCopyPadExtParams<float> reluInPadParams{
            true, 0, static_cast<uint8_t>(topkPadLenFp32 - runInfo.realTopk), 0};
        AscendC::DataCopyPad(reluInTensor, reluInGmTensor[groupOffset], reluInCopyParams, reluInPadParams);

        AscendC::DataCopyExtParams reluGradCopyParams{static_cast<uint16_t>(currentGroupNum),
                                                      static_cast<uint32_t>(runInfo.realTopk * sizeof(dataType)), 0, 0,
                                                      0};
        AscendC::DataCopyPadExtParams<dataType> reluGradPadParams{
            true, 0, static_cast<uint8_t>(topkPadLenBf16 - runInfo.realTopk), 0};
        AscendC::DataCopyPad(reluGradTensor, reluGradInGmTensor[groupOffset], reluGradCopyParams, reluGradPadParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPingPong);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        // compute reluGrad
        AscendC::Compare(maskTensor, reluInTensor, zeroFloatTensor, CMPMODE::GT, floatMask, compareRepeatTimes,
                         repeatParamsCompare);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(reluGradOutTensor.template ReinterpretCast<half>(), maskTensor,
                        reluGradTensor.template ReinterpretCast<half>(), static_cast<half>(0.0),
                        SELMODE::VSEL_TENSOR_SCALAR_MODE, halfMask, selectRepeatTimes, repeatParamsSelect);
        AscendC::PipeBarrier<PIPE_V>();
        // compute broadcastMul
        for (uint32_t j = 0; j < currentGroupNum; j++) {
            AscendC::Mul(reluInTensor[j * topkPadLenBf16], reluInTensor[j * topkPadLenBf16], dyFloatTensor,
                         topkPadLenFp32);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // compute reduceSum
        for (uint32_t j = 0; j < currentGroupNum; j++) {
            AscendC::ReduceSum<float>(reduceSumWorkSpaceTensor, reluInTensor[j * topkPadLenBf16],
                                      reduceSumWorkSpaceTensor, topkPadLenFp32);
            AscendC::SetFlag<HardEvent::V_S>(eventIdVToS);
            AscendC::WaitFlag<HardEvent::V_S>(eventIdVToS);
            float topKSum = reduceSumWorkSpaceTensor.GetValue(0);
            reduceFloatTensor.SetValue(loopIdx * groupNumDivide + j, static_cast<float>(topKSum));
        }
        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2PingPong);
        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // copyout reluGrad
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopyExtParams reluGradOutCopyParams{static_cast<uint16_t>(currentGroupNum),
                                                         static_cast<uint32_t>(runInfo.realTopk * sizeof(dataType)), 0,
                                                         0, 0};
        AscendC::DataCopyPad(reluGradOutGmTensor[groupOffset], reluGradOutTensor, reluGradOutCopyParams);
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPingPong);
    }
    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping);
    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPing);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPong);

    // copyout dweights out in the tail
    AscendC::PipeBarrier<PIPE_V>();
    Cast(reduceTensor, reduceFloatTensor, RoundMode::CAST_ROUND, blockGroupNum);
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    // [B, S1, N1]
    uint64_t dweightOffset = 0;
    if constexpr (LIGT::layout == LIG_LAYOUT::BSND) {
        dweightOffset = runInfo.bIdx * constInfo.seqlenQ * constInfo.headNumQ + runInfo.s1Idx * constInfo.headNumQ +
                        runInfo.n2Idx * constInfo.groupNum + blockGroupBegin;
    } else if constexpr (LIGT::layout == LIG_LAYOUT::TND) {
        dweightOffset = (runInfo.prefixSumS1 + runInfo.s1Idx) * constInfo.headNumQ +
                        runInfo.n2Idx * constInfo.groupNum + blockGroupBegin;
    }
    AscendC::DataCopyExtParams dWeightsCopyOutParams{static_cast<uint16_t>(1),
                                                     static_cast<uint32_t>(blockGroupNum * sizeof(dataType)), 0, 0, 0};
    AscendC::DataCopyPad(dweightsGmTensor[dweightOffset], reduceTensor, dWeightsCopyOutParams);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}
} // namespace LigKernel
#endif
