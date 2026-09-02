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
 * \file flash_attention_score_grad_nz_post.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_GRAD_NZ_POST_H_
#define FLASH_ATTENTION_SCORE_GRAD_NZ_POST_H_
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "vector_api/vf_transdata.h"
#include "flash_attention_score_grad_common.h"

TEMPLATES_DEF
class FlashAttentionScoreGradNzPost {
public:
    __aicore__ inline FlashAttentionScoreGradNzPost(){};
    __aicore__ inline void Init(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink,
                                GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR workspace,
                                FagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void Process();
    template <uint8_t MM_IDX = 0>
    __aicore__ inline void DoWork();
    template <uint8_t MM_IDX = 0>
    __aicore__ inline void GetPosition(uint64_t index);
    template <uint8_t MM_IDX = 0>
    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, 1> &inQue, uint64_t srcM, uint64_t srcN, uint64_t srcOffset);
    __aicore__ inline void ProcessSink();

    constexpr static uint64_t MAX_PROCESS_D = 160;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t ROPE_D = 64;

    TPipe *pipe;
    FagTilingType tilingData;

    // GlobalBuffer
    GlobalTensor<OUTDTYPE> dqGm;
    GlobalTensor<OUTDTYPE> dkGm;
    GlobalTensor<OUTDTYPE> dvGm;
    GlobalTensor<OUTDTYPE> dqRopeGm;
    GlobalTensor<OUTDTYPE> dkRopeGm;
    GlobalTensor<float> dqWorkSpace;
    GlobalTensor<float> dkWorkSpace;
    GlobalTensor<float> dvWorkSpace;
    GlobalTensor<float> dsinkWorkSpace;
    GlobalTensor<float> dsinkGm;
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKVlenAddr;

    // UB buffer
    TQue<QuePosition::VECIN, 1> inQueue[2];
    TQue<QuePosition::VECOUT, 1> outQueue[2];
    TBuf<> dsinkResBuf;

    uint64_t b;
    uint64_t n1;
    uint64_t n2;
    uint64_t s1;
    uint64_t s2;
    uint64_t d;
    uint64_t d1;
    uint64_t dAlign16;
    uint64_t d1Align16;

    uint64_t usedCoreNum;
    uint64_t vBlockIdx;
    uint64_t s1Outer;
    uint64_t s2Outer;
    uint64_t bIdx;
    uint64_t nIdx;
    uint64_t s1Idx;
    uint64_t s2Idx;
    uint64_t sIdx;
    uint64_t s1Tail;
    uint64_t s2Tail;
    float scaleValue;
    uint8_t layoutType;
    uint64_t maxBufferNum;
    // sink
    bool isSink;
    uint64_t s1SinkOuter = 0;
    uint64_t s2SinkOuter = 0;

    uint64_t curN = 0;
    uint64_t curS = 0;
    uint64_t curD = 0; // copy in headDim
    uint64_t outD = 0; // copy out headDim
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void FlashAttentionScoreGradNzPost<TEMPLATE_ARGS>::Init(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink, GM_ADDR actual_seq_qlen,
    GM_ADDR actual_seq_kvlen, GM_ADDR workspace, FagTilingType ordTilingData, TPipe *pipeIn)
{
    pipe = pipeIn;
    tilingData = ordTilingData;
    // Init WorkSpace
    dqWorkSpace.SetGlobalBuffer((__gm__ float *)workspace +
                                tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpace.SetGlobalBuffer((__gm__ float *)workspace +
                                tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    dvWorkSpace.SetGlobalBuffer((__gm__ float *)workspace +
                                tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));

    isSink = tilingData->s1s2BNGS1S2BaseParams.sinkOptional;
    if (unlikely(isSink)) {
        dsinkWorkSpace.SetGlobalBuffer((__gm__ float *)workspace +
                                       tilingData->postTilingData.dsinkWorkSpaceOffset / sizeof(float));
        dsinkGm.SetGlobalBuffer((__gm__ float *)dsink);
        s1SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s1SinkOuter;
        s2SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s2SinkOuter;
        pipe->InitBuffer(dsinkResBuf, sizeof(float));
    }

    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    if constexpr (IS_ROPE) {
        dqRopeGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dqRope);
        dkRopeGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dkRope);
    }
    if constexpr (IS_TND) {
        actualSeqQlenAddr = actual_seq_qlen;
        actualSeqKVlenAddr = actual_seq_kvlen;
    }

    // Get tiling info
    b = tilingData->s1s2BNGS1S2BaseParams.b;
    n2 = tilingData->s1s2BNGS1S2BaseParams.n2;
    n1 = n2 * tilingData->s1s2BNGS1S2BaseParams.g;
    s1 = tilingData->s1s2BNGS1S2BaseParams.s1;
    s2 = tilingData->s1s2BNGS1S2BaseParams.s2;
    d = tilingData->s1s2BNGS1S2BaseParams.d;
    d1 = tilingData->s1s2BNGS1S2BaseParams.d1;
    dAlign16 = AlignTo16(d);
    d1Align16 = AlignTo16(d1);

    usedCoreNum = tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter >> 1;
    vBlockIdx = GetBlockIdx();
    s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    s1Tail = s1 - (s1Outer - 1) * CUBE_BASEM;
    s2Tail = s2 - (s2Outer - 1) * CUBE_BASEN;
    scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;
    layoutType = tilingData->s1s2BNGS1S2BaseParams.layout;

    // Init ub buffer
    maxBufferNum = CUBE_BASEM * dAlign16;
    pipe->InitBuffer(inQueue[0], 1, maxBufferNum * sizeof(float));
    pipe->InitBuffer(inQueue[1], 1, maxBufferNum * sizeof(float));
    pipe->InitBuffer(outQueue[0], 1, maxBufferNum * sizeof(OUTDTYPE));
    if (dAlign16 <= MAX_PROCESS_D) {
        pipe->InitBuffer(outQueue[1], 1, maxBufferNum * sizeof(OUTDTYPE));
    } else {
        outQueue[1] = outQueue[0];
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FlashAttentionScoreGradNzPost<TEMPLATE_ARGS>::GetPosition(uint64_t index)
{
    if constexpr (IS_TND) {
        int64_t totalLen = 0;
        for (uint64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
            if constexpr (MM_IDX == DQ_IDX) {
                uint64_t sq = (bDimIdx == 0) ? ((__gm__ int64_t *)actualSeqQlenAddr)[bDimIdx] :
                                               (((__gm__ int64_t *)actualSeqQlenAddr)[bDimIdx] -
                                                ((__gm__ int64_t *)actualSeqQlenAddr)[bDimIdx - 1]);
                s1Outer = Ceil<uint64_t>(sq, CUBE_BASEM);
                totalLen += n1 * s1Outer;
                if (totalLen > index) {
                    bIdx = bDimIdx;
                    uint64_t bTail = index - (totalLen - n1 * s1Outer);
                    nIdx = bTail / s1Outer;
                    sIdx = bTail % s1Outer;
                    curS = sq;
                    break;
                }
            } else {
                uint64_t skv = (bDimIdx == 0) ? ((__gm__ int64_t *)actualSeqKVlenAddr)[bDimIdx] :
                                                (((__gm__ int64_t *)actualSeqKVlenAddr)[bDimIdx] -
                                                 ((__gm__ int64_t *)actualSeqKVlenAddr)[bDimIdx - 1]);
                s2Outer = Ceil<uint64_t>(skv, CUBE_BASEN);
                totalLen += n2 * s2Outer;
                if (totalLen > index) {
                    bIdx = bDimIdx;
                    uint64_t bTail = index - (totalLen - n2 * s2Outer);
                    nIdx = bTail / s2Outer;
                    sIdx = bTail % s2Outer;
                    curS = skv;
                    break;
                }
            }
        }
    } else {
        if constexpr (MM_IDX == DQ_IDX) {
            bIdx = index / (n1 * s1Outer);
            uint64_t bTail = index % (n1 * s1Outer);
            nIdx = bTail / s1Outer;
            sIdx = bTail % s1Outer;
        } else {
            bIdx = index / (n2 * s2Outer);
            uint64_t bTail = index % (n2 * s2Outer);
            nIdx = bTail / s2Outer;
            sIdx = bTail % s2Outer;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FlashAttentionScoreGradNzPost<TEMPLATE_ARGS>::CopyIn(TQue<QuePosition::VECIN, 1> &inQue,
                                                                            uint64_t srcM, uint64_t srcN,
                                                                            uint64_t srcOffset)
{
    LocalTensor<float> tensor = inQue.AllocTensor<float>();
    if constexpr (MM_IDX == DQ_IDX) {
        DataCopy(tensor, dqWorkSpace[srcOffset], srcM * srcN);
    } else if constexpr (MM_IDX == DK_IDX) {
        DataCopy(tensor, dkWorkSpace[srcOffset], srcM * srcN);
    } else {
        DataCopy(tensor, dvWorkSpace[srcOffset], srcM * srcN);
    }
    inQue.EnQue(tensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FlashAttentionScoreGradNzPost<TEMPLATE_ARGS>::Process()
{
    if ASCEND_IS_AIV {
        DoWork<0>();
        DoWork<1>();
        DoWork<2>();
        if (unlikely(isSink)) {
            ProcessSink();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FlashAttentionScoreGradNzPost<TEMPLATE_ARGS>::ProcessSink()
{
    event_t eventIDVToSPing = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIDVToSPong = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    event_t eventIDMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    uint64_t inputTotalSize = n1;
    uint64_t loop = Ceil<uint64_t>(n1, usedCoreNum);
    uint64_t sinkReduceAxis = b * s1SinkOuter * s2SinkOuter;
    uint64_t sinkPostTailNumTmp = sinkReduceAxis % maxBufferNum;
    uint64_t sinkPostTailNum = sinkPostTailNumTmp == 0 ? maxBufferNum : sinkPostTailNumTmp;
    uint64_t beginIdx = vBlockIdx * loop;
    uint64_t endIdx = beginIdx + loop;
    if (endIdx > inputTotalSize) {
        endIdx = inputTotalSize;
    }
    GM_ADDR baseAddr = (GM_ADDR)dsinkWorkSpace.GetPhyAddr();
    for (uint64_t nIdx = beginIdx; nIdx < endIdx; nIdx++) {
        SetFlag<HardEvent::MTE3_S>(eventIDMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMte3ToS);
        float dsinkAccu = 0.0;
        uint64_t dsinkBaseOffset = nIdx * b * s1SinkOuter * s2SinkOuter;
        for (uint64_t pingIdx = 0; pingIdx < sinkReduceAxis; pingIdx = pingIdx + (maxBufferNum << 1)) {
            LocalTensor<float> vecInPing = inQueue[0].AllocTensor<float>();
            uint64_t pongIdx = pingIdx + maxBufferNum;
            uint64_t pingSize = pongIdx < sinkReduceAxis ? maxBufferNum : sinkPostTailNum;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = pingSize * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<float> copyPadParams;
            copyPadParams.isPad = ((pingSize % commondef::NUM_EIGHT) == 0) ? false : true;
            copyPadParams.leftPadding = 0;
            copyPadParams.rightPadding = commondef::NUM_EIGHT - (pingSize % commondef::NUM_EIGHT);
            copyPadParams.paddingValue = 0.0;
            DataCopyPad(vecInPing, dsinkWorkSpace[dsinkBaseOffset + pingIdx], copyParams, copyPadParams);
            inQueue[0].EnQue(vecInPing);
            inQueue[0].DeQue<float>();
            LocalTensor<float> vecOutPing = outQueue[0].AllocTensor<float>();
            ReduceSink<float>(vecOutPing, vecInPing, pingSize);
            uint64_t pongSize = 0;
            LocalTensor<float> vecInPong;
            bool neeedPong = pongIdx < sinkReduceAxis;
            if (neeedPong) {
                vecInPong = inQueue[1].AllocTensor<float>();
                pongSize = pongIdx + maxBufferNum < sinkReduceAxis ? maxBufferNum : sinkPostTailNum;
                copyParams.blockLen = pongSize * sizeof(float);
                copyPadParams.isPad = ((pongSize % commondef::NUM_EIGHT) == 0) ? false : true;
                copyPadParams.rightPadding = commondef::NUM_EIGHT - (pongSize % commondef::NUM_EIGHT);
                DataCopyPad(vecInPong, dsinkWorkSpace[dsinkBaseOffset + pongIdx], copyParams, copyPadParams);
            }
            SetFlag<HardEvent::V_S>(eventIDVToSPing);
            WaitFlag<HardEvent::V_S>(eventIDVToSPing);
            dsinkAccu += vecOutPing.GetValue(0);
            outQueue[0].EnQue(vecOutPing);
            outQueue[0].DeQue<float>();
            inQueue[0].FreeTensor(vecInPing);
            outQueue[0].FreeTensor(vecOutPing);
            LocalTensor<float> vecOutPong;
            if (neeedPong) {
                inQueue[1].EnQue(vecInPong);
                inQueue[1].DeQue<float>();
                vecOutPong = outQueue[1].AllocTensor<float>();
                ReduceSink<float>(vecOutPong, vecInPong, pongSize);
                SetFlag<HardEvent::V_S>(eventIDVToSPong);
                WaitFlag<HardEvent::V_S>(eventIDVToSPong);
                dsinkAccu += vecOutPong.GetValue(0);
                outQueue[1].EnQue(vecOutPong);
                outQueue[1].DeQue<float>();
                inQueue[1].FreeTensor(vecInPong);
                outQueue[1].FreeTensor(vecOutPong);
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

TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FlashAttentionScoreGradNzPost<TEMPLATE_ARGS>::DoWork()
{
    uint64_t blockNum = 0;      // 每个核需要处理的基本块数量
    uint64_t totalBlockNum = 0; // 所有核需要处理的基本块总和
    uint64_t copyLoopTimes = 0;
    uint64_t sOuter = MM_IDX == DQ_IDX ? s1Outer : s2Outer;
    uint64_t sTail = MM_IDX == DQ_IDX ? s1Tail : s2Tail;
    uint64_t coreLoopOffsetNum = 0;
    if constexpr (IS_TND) {
        if constexpr (MM_IDX == DQ_IDX) {
            for (uint64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
                uint64_t sq = (bDimIdx == 0) ? ((__gm__ int64_t *)actualSeqQlenAddr)[bDimIdx] :
                                               (((__gm__ int64_t *)actualSeqQlenAddr)[bDimIdx] -
                                                ((__gm__ int64_t *)actualSeqQlenAddr)[bDimIdx - 1]);
                totalBlockNum += n1 * Ceil<int64_t>(sq, CUBE_BASEM);
            }
            blockNum = Ceil<uint64_t>(totalBlockNum, usedCoreNum);
            curN = n1;
            curD = dAlign16; // dq headDim
            outD = d;
        } else {
            for (uint64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
                uint64_t skv = (bDimIdx == 0) ? ((__gm__ int64_t *)actualSeqKVlenAddr)[bDimIdx] :
                                                (((__gm__ int64_t *)actualSeqKVlenAddr)[bDimIdx] -
                                                 ((__gm__ int64_t *)actualSeqKVlenAddr)[bDimIdx - 1]);
                totalBlockNum += n2 * Ceil<int64_t>(skv, CUBE_BASEN);
            }
            blockNum = Ceil<uint64_t>(totalBlockNum, usedCoreNum);
            curN = n2;
            curD = ((MM_IDX == DK_IDX) ? dAlign16 : d1Align16);
            outD = ((MM_IDX == DK_IDX) ? d : d1);
        }
        usedCoreNum = Ceil<uint64_t>(totalBlockNum, blockNum);
        coreLoopOffsetNum = vBlockIdx * blockNum; // 核间
        if (vBlockIdx == usedCoreNum - 1) {
            blockNum = totalBlockNum - (usedCoreNum - 1) * blockNum;
        }
    } else {
        if constexpr (MM_IDX == DQ_IDX) {
            totalBlockNum = b * n1 * s1Outer;
            blockNum = Ceil<uint64_t>(totalBlockNum, usedCoreNum);
            curN = n1;
            curS = s1;
            curD = dAlign16; // dq headDim
            outD = d;
        } else {
            totalBlockNum = b * n2 * s2Outer;
            blockNum = Ceil<uint64_t>(totalBlockNum, usedCoreNum);
            curN = n2;
            curS = s2;
            curD = ((MM_IDX == DK_IDX) ? dAlign16 : d1Align16);
            outD = ((MM_IDX == DK_IDX) ? d : d1);
        }
        usedCoreNum = Ceil<uint64_t>(totalBlockNum, blockNum);
        coreLoopOffsetNum = vBlockIdx * blockNum; // 核间
        if (vBlockIdx == usedCoreNum - 1) {
            blockNum = totalBlockNum - (usedCoreNum - 1) * blockNum;
        }
    }
    if (vBlockIdx >= usedCoreNum) {
        return;
    }
    uint8_t taskIdMod2 = 0;
    for (uint64_t taskId = 0; taskId < blockNum; taskId++) {
        taskIdMod2 = taskId & 1;
        GetPosition<MM_IDX>(coreLoopOffsetNum + taskId);
        uint64_t srcM = 0;
        if constexpr (IS_TND) {
            sOuter = MM_IDX == DQ_IDX ? s1Outer : s2Outer;
            sTail = curS - (sOuter - 1) * CUBE_BASEM;
        }
        srcM = (sIdx == sOuter - 1) ? sTail : CUBE_BASEM;
        uint64_t srcOffset = 0;
        uint64_t prefixS = 0;
        if constexpr (IS_TND) {
            if (unlikely(bIdx == 0)) {
                srcOffset = nIdx * curS * curD + sIdx * CUBE_BASEM * curD;
            } else {
                prefixS = MM_IDX == DQ_IDX ? ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1] :
                                             ((__gm__ int64_t *)actualSeqKVlenAddr)[bIdx - 1];
                srcOffset = prefixS * curN * curD + nIdx * curS * curD + sIdx * CUBE_BASEM * curD;
            }
        } else {
            srcOffset = bIdx * curN * curS * curD + nIdx * curS * curD + sIdx * CUBE_BASEM * curD;
        }
        CopyIn<MM_IDX>(inQueue[taskIdMod2], srcM, curD, srcOffset);
        LocalTensor<OUTDTYPE> outTensor = outQueue[taskIdMod2].AllocTensor<OUTDTYPE>();
        LocalTensor<float> srcTensor = inQueue[taskIdMod2].DeQue<float>();
        LocalTensor<OUTDTYPE> srcTensorB16 = srcTensor.ReinterpretCast<OUTDTYPE>();
        if constexpr (MM_IDX == DQ_IDX || MM_IDX == DK_IDX) {
            Muls(srcTensor, srcTensor, scaleValue, srcM * curD);
        }
        Cast(srcTensorB16, srcTensor, RoundMode::CAST_ROUND, srcM * curD);
        Transdata<OUTDTYPE>(outTensor, srcTensorB16, srcM, curD); // ub n轴保证block对齐

        // rope场景分两次copy out，d大小分别为128和64
        DataCopyExtParams intriParamsOut;
        intriParamsOut.blockCount = srcM;
        if constexpr (IS_ROPE && !(MM_IDX == DV_IDX)) {
            outD = curD - ROPE_D;
            intriParamsOut.blockLen = outD * sizeof(OUTDTYPE);
            intriParamsOut.srcStride = ROPE_D * C0_SIZE;
        } else {
            intriParamsOut.blockLen = outD * sizeof(OUTDTYPE);
            intriParamsOut.srcStride = 0;
        }
        uint64_t dqkvGmOffset = 0;
        if constexpr (IS_TND) {
            if (unlikely(bIdx == 0)) {
                dqkvGmOffset = sIdx * CUBE_BASEM * curN * outD + nIdx * outD;
            } else {
                dqkvGmOffset = prefixS * curN * outD + sIdx * CUBE_BASEM * curN * outD + nIdx * outD;
            }
            intriParamsOut.dstStride = static_cast<uint32_t>((curN - 1) * outD * sizeof(OUTDTYPE));
        } else {
            if (layoutType == BNGSD) {
                dqkvGmOffset = bIdx * curN * curS * outD + nIdx * curS * outD + sIdx * CUBE_BASEM * outD;
                intriParamsOut.dstStride = 0;
            } else if (layoutType == SBNGD) {
                dqkvGmOffset = sIdx * CUBE_BASEM * b * curN * outD + bIdx * curN * outD + nIdx * outD;
                intriParamsOut.dstStride = static_cast<uint32_t>((b * curN - 1) * outD * sizeof(OUTDTYPE));
            } else if (layoutType == BSNGD) {
                dqkvGmOffset = bIdx * curS * curN * outD + sIdx * CUBE_BASEM * curN * outD + nIdx * outD;
                intriParamsOut.dstStride = static_cast<uint32_t>((curN - 1) * outD * sizeof(OUTDTYPE));
            }
        }
        outQueue[taskIdMod2].EnQue(outTensor);
        outQueue[taskIdMod2].DeQue<OUTDTYPE>();
        if constexpr (MM_IDX == DQ_IDX) {
            DataCopyPad(dqGm[dqkvGmOffset], outTensor, intriParamsOut); // copy dq out
            if constexpr (IS_ROPE) {
                intriParamsOut.blockLen = ROPE_D * sizeof(OUTDTYPE);
                intriParamsOut.srcStride = outD * C0_SIZE;
                DataCopyPad(dqRopeGm[dqkvGmOffset >> 1], outTensor[outD], intriParamsOut);
            }
        } else if constexpr (MM_IDX == DK_IDX) {
            DataCopyPad(dkGm[dqkvGmOffset], outTensor, intriParamsOut); // copy dk out
            if constexpr (IS_ROPE) {
                intriParamsOut.blockLen = ROPE_D * sizeof(OUTDTYPE);
                intriParamsOut.srcStride = outD * C0_SIZE;
                DataCopyPad(dkRopeGm[dqkvGmOffset >> 1], outTensor[outD], intriParamsOut);
            }
        } else {
            DataCopyPad(dvGm[dqkvGmOffset], outTensor, intriParamsOut); // copy dv out
        }
        inQueue[taskIdMod2].FreeTensor(srcTensor);
        outQueue[taskIdMod2].FreeTensor(outTensor);
    }
}

#endif
