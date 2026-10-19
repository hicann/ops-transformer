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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_post_david.h
 * \brief
 */
#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_DAVID_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_DAVID_H_
#include "vector_api/vf_post_reduce_sink_david.h"
#include "kernel_operator.h"

#define FAG_POST_CLASS_TEMPLATE \
    template <typename T1, typename T2, typename OUTDTYPE = T1, const uint8_t SPLIT_AXIS = 0, const bool IS_ROPE = 0, \
              const uint8_t DETER_SPARSE_TYPE = 0>

#define FAG_POST_FUNCTION_TEMPLATE \
    template <typename T1, typename T2, typename OUTDTYPE, const uint8_t SPLIT_AXIS, const bool IS_ROPE, \
              const uint8_t DETER_SPARSE_TYPE>

#define FAG_POST_FUNCTION_PARAMS_TEMPLATE T1, T2, OUTDTYPE, SPLIT_AXIS, IS_ROPE, DETER_SPARSE_TYPE

FAG_POST_CLASS_TEMPLATE
class FlashAttentionScoreGradS1S2BNGS1S2PostDavid {
public:
    __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2PostDavid(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dqRope,
                                __gm__ uint8_t* dkRope, __gm__ uint8_t* dsink, __gm__ uint8_t* workspace,
                                FagTilingType ordTilingData, TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSink();
    __aicore__ inline void ProcessDqkv();

    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t DAVID_POST_BASE = 128 * 224;
    uint32_t ROPE_DIM = 64;
    uint32_t VALUE_DIM = 128;
    uint32_t POST_S_BASE = 128;

    TPipe* pipe;
    FagTilingType tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    GlobalTensor<OUTDTYPE> dqkv[3];
    GlobalTensor<OUTDTYPE> dqkRope[2];
    GlobalTensor<float> dqkvWorkspace[3];
    GlobalTensor<float> dsinkWorkspace;
    GlobalTensor<float> dsinkGm;
    TBuf<> dsinkResBuf;
    uint32_t cBlockIdx;
    uint64_t loop;
    uint64_t inputTotalSize;
    uint64_t qPostTailNum;
    uint64_t qPostBaseNum;
    uint32_t isSink = 0;
    uint64_t sinkPostTailNum = 0;
    uint64_t bSinkSize = 0;
    uint64_t s1SinkOuter = 0;
    uint64_t s2SinkOuter = 0;
};

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PostDavid<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dqRope, __gm__ uint8_t* dkRope,
    __gm__ uint8_t* dsink, __gm__ uint8_t* workspace, FagTilingType ordTilingData, TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE*)dq);
    dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE*)dk);
    dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE*)dv);

    dqkRope[0].SetGlobalBuffer((__gm__ OUTDTYPE*)dqRope);
    dqkRope[1].SetGlobalBuffer((__gm__ OUTDTYPE*)dkRope);

    loop = tilingData->postTilingData.qPostBlockFactor;
    inputTotalSize = tilingData->postTilingData.qPostBlockTotal;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;

    dqkvWorkspace[0].SetGlobalBuffer((__gm__ float*)workspace +
                                     tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[1].SetGlobalBuffer((__gm__ float*)workspace +
                                     tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[2].SetGlobalBuffer((__gm__ float*)workspace +
                                     tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));

    isSink = tilingData->s1s2BNGS1S2BaseParams.sinkOptional;
    if (unlikely(isSink)) {
        dsinkWorkspace.SetGlobalBuffer((__gm__ float*)workspace +
                                       tilingData->postTilingData.dsinkWorkSpaceOffset / sizeof(float));
        dsinkGm.SetGlobalBuffer((__gm__ float*)dsink);
        s1SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s1SinkOuter;
        s2SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s2SinkOuter;
        bSinkSize = tilingData->s1s2BNGS1S2BaseParams.b;
        pipe->InitBuffer(dsinkResBuf, sizeof(float));
    }

    pipe->InitBuffer(inQueuePing, 1, DAVID_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePing, 1, DAVID_POST_BASE * sizeof(OUTDTYPE));
    pipe->InitBuffer(inQueuePong, 1, DAVID_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePong, 1, DAVID_POST_BASE * sizeof(OUTDTYPE));
}

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PostDavid<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::ProcessSink()
{
    event_t eventIDVToSPing = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIDVToSPong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    event_t eventIDMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    inputTotalSize = tilingData->postTilingData.sinkPostBlockTotal;
    loop = tilingData->postTilingData.sinkPostBlockFactor;
    sinkPostTailNum = tilingData->postTilingData.sinkPostTailNum;
    uint64_t sinkReduceAxis = tilingData->postTilingData.sinkReduceAxis;
    uint64_t begin = cBlockIdx * loop;
    uint64_t end = begin + loop;
    if (end > inputTotalSize) {
        end = inputTotalSize;
    }
    GM_ADDR baseAddr = (__gm__ uint8_t*)dsinkWorkspace.GetPhyAddr();
    for (uint64_t nIdx = begin; nIdx < end; nIdx++) {
        SetFlag<HardEvent::MTE3_S>(eventIDMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMte3ToS);
        float dsinkAccu = 0.0;
        uint64_t dsinkBaseOffset = nIdx * bSinkSize * s1SinkOuter * s2SinkOuter;
        for (uint64_t pingIdx = 0; pingIdx < sinkReduceAxis; pingIdx = pingIdx + (DAVID_POST_BASE << 1)) {
            LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
            uint64_t pongIdx = pingIdx + DAVID_POST_BASE;
            uint64_t pingSize = pongIdx < sinkReduceAxis ? DAVID_POST_BASE : sinkPostTailNum;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = pingSize * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<float> copyPadParams;
            copyPadParams.isPad = ((pingSize % 8) == 0) ? false : true;
            copyPadParams.leftPadding = 0;
            copyPadParams.rightPadding = 8 - (pingSize % 8);
            copyPadParams.paddingValue = 0.0;
            DataCopyPad(vecInPing, dsinkWorkspace[dsinkBaseOffset + pingIdx], copyParams, copyPadParams);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();
            LocalTensor<float> vecOutPing = outQueuePing.AllocTensor<float>();
            ReduceSink<float>(vecOutPing, vecInPing, pingSize);
            uint64_t pongSize = 0;
            LocalTensor<float> vecInPong;
            bool neeedPong = pongIdx < sinkReduceAxis;
            if (neeedPong) {
                vecInPong = inQueuePong.AllocTensor<float>();
                pongSize = pongIdx + DAVID_POST_BASE < sinkReduceAxis ? DAVID_POST_BASE : sinkPostTailNum;
                copyParams.blockLen = pongSize * sizeof(float);
                copyPadParams.isPad = ((pongSize % 8) == 0) ? false : true;
                copyPadParams.rightPadding = 8 - (pongSize % 8);
                DataCopyPad(vecInPong, dsinkWorkspace[dsinkBaseOffset + pongIdx], copyParams, copyPadParams);
            }
            SetFlag<HardEvent::V_S>(eventIDVToSPing);
            WaitFlag<HardEvent::V_S>(eventIDVToSPing);
            dsinkAccu += vecOutPing.GetValue(0);
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<float>();
            inQueuePing.FreeTensor(vecInPing);
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<float> vecOutPong;
            if (neeedPong) {
                inQueuePong.EnQue(vecInPong);
                inQueuePong.DeQue<float>();
                vecOutPong = outQueuePong.AllocTensor<float>();
                ReduceSink<float>(vecOutPong, vecInPong, pongSize);
                SetFlag<HardEvent::V_S>(eventIDVToSPong);
                WaitFlag<HardEvent::V_S>(eventIDVToSPong);
                dsinkAccu += vecOutPong.GetValue(0);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<float>();
                inQueuePong.FreeTensor(vecInPong);
                outQueuePong.FreeTensor(vecOutPong);
            }
        }
        dsinkAccu = -dsinkAccu;
        LocalTensor<float> dsinkResTensor = dsinkResBuf.Get<float>();
        dsinkResTensor.SetValue(0, dsinkAccu);
        SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
        DataCopyPad(dsinkGm[nIdx], dsinkResTensor, {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
    }
}

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PostDavid<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::ProcessDqkv()
{
    for (int qkvIdx = 0; qkvIdx < 3; qkvIdx++) {
        if (qkvIdx == 1) {
            loop = tilingData->postTilingData.kPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.kPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.kPostTailNum;
            qPostBaseNum = tilingData->postTilingData.kPostBaseNum;
        } else if (qkvIdx == 2) {
            loop = tilingData->postTilingData.vPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.vPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.vPostTailNum;
            qPostBaseNum = tilingData->postTilingData.vPostBaseNum;
        }
        uint64_t blockCore = loop * qPostBaseNum;
        uint64_t begin = cBlockIdx * blockCore;
        uint64_t end = begin + blockCore;

        if (end > inputTotalSize) {
            end = inputTotalSize;
        }

        uint64_t dqPingGmOffset = cBlockIdx * loop * POST_S_BASE * VALUE_DIM;
        for (uint64_t pingIdx = begin; pingIdx < end; pingIdx = pingIdx + (qPostBaseNum << 1)) {
            LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
            uint64_t pongIdx = pingIdx + qPostBaseNum;
            uint64_t pingSize = pongIdx < inputTotalSize ? qPostBaseNum : qPostTailNum;
            uint64_t dqPongGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE; // 128*128
            uint64_t seqPingSize = pingSize / (ROPE_DIM + VALUE_DIM);
            DataCopy(vecInPing, dqkvWorkspace[qkvIdx][pingIdx], (pingSize + 7) >> 3 << 3, true);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();

            if (qkvIdx < 2) {
                Muls(vecInPing, vecInPing, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, pingSize);
            }
            LocalTensor<OUTDTYPE> vecOutPing = outQueuePing.AllocTensor<OUTDTYPE>();
            Cast(vecOutPing, vecInPing, RoundMode::CAST_ROUND, pingSize);
            uint64_t pongSize;
            uint64_t seqPongSize;
            LocalTensor<float> vecIn_pong;
            bool neeedPong = loop > 1 && pongIdx < end;
            if (neeedPong) {
                vecIn_pong = inQueuePong.AllocTensor<float>();
                pongSize = pongIdx + qPostBaseNum < inputTotalSize ? qPostBaseNum : qPostTailNum;
                seqPongSize = pongSize / (ROPE_DIM + VALUE_DIM);
                DataCopy(vecIn_pong, dqkvWorkspace[qkvIdx][pongIdx], (pongSize + 7) >> 3 << 3, true);
            }
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<OUTDTYPE>();
            inQueuePing.FreeTensor(vecInPing);
            if constexpr (IS_ROPE) {
                if (qkvIdx == 2) {
                    DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, pingSize);
                } else {
                    // 96 * (128 + 64)基本块
                    DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = seqPingSize;
                    // 128 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.blockLen = 4 * sizeof(OUTDTYPE);
                    // 64 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                    DataCopy(dqkv[qkvIdx][dqPingGmOffset], vecOutPing, dataCopyParams);

                    dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                    dataCopyParams.srcStride = 4 * sizeof(OUTDTYPE);
                    DataCopy(dqkRope[qkvIdx][dqPingGmOffset / 2], vecOutPing[128], dataCopyParams);
                }
                dqPingGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE;
            } else {
                DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
            }
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<OUTDTYPE> vecOutPong;
            if (neeedPong) {
                vecOutPong = outQueuePong.AllocTensor<OUTDTYPE>();
                inQueuePong.EnQue(vecIn_pong);
                inQueuePong.DeQue<float>();
                if (qkvIdx < 2) {
                    Muls(vecIn_pong, vecIn_pong, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, pongSize);
                }
                Cast(vecOutPong, vecIn_pong, RoundMode::CAST_ROUND, pongSize);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<OUTDTYPE>();
                inQueuePong.FreeTensor(vecIn_pong);
                if constexpr (IS_ROPE) {
                    if (qkvIdx == 2) {
                        DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, pongSize);
                    } else {
                        // 96 * (128 + 64)基本块
                        DataCopyParams dataCopyParams;
                        dataCopyParams.blockCount = seqPongSize;
                        // 128 * sizeof(OUTDTYPE) / 32B
                        dataCopyParams.blockLen = 4 * sizeof(OUTDTYPE);
                        // 64 * sizeof(OUTDTYPE) / 32B
                        dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                        DataCopy(dqkv[qkvIdx][dqPongGmOffset], vecOutPong, dataCopyParams);

                        dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                        dataCopyParams.srcStride = 4 * sizeof(OUTDTYPE);
                        DataCopy(dqkRope[qkvIdx][dqPongGmOffset / 2], vecOutPong[128], dataCopyParams);
                    }
                    dqPingGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE;
                } else {
                    DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
                }
                outQueuePong.FreeTensor(vecOutPong);
            }
        }
    }
}

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2PostDavid<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if constexpr (SPLIT_AXIS != BN2) {
        ProcessDqkv();
    }
    if (unlikely(isSink)) {
        ProcessSink();
    }
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_H_
