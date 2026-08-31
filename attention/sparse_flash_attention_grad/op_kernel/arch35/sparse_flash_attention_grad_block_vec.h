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
 * \file sparse_flash_attention_grad_block_vec.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_BLOCK_VEC_H
#define SPARSE_FLASH_ATTENTION_GRAD_BLOCK_VEC_H

#include "sparse_flash_attention_grad_common.h"
#include "vector_api/cast_softmax_grad_sfag.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax_sfag.h"
#include "vector_api/vf_broadcast_sub_mul_sfag.h"
#include "vector_api/vf_cast_transdata_deconflict_sfag.h"
namespace SfagBaseApi {
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t SYNC_V0_V1_DS_A_MAX_DONE_FLAG = 10;
constexpr uint32_t BIT_MASK_NUM = 8;
constexpr int64_t MAX_N1 = optiling::sfag::MAX_N1;

TEMPLATES_DEF
class FAGBlockVec {
private:
    __aicore__ inline void GetBS1Index(int64_t bS1Index, int64_t &bIdx, int64_t &s1Idx, FagConstInfo &constInfo,
                                       int64_t startBIdx);
    __aicore__ inline void GetRunInfo(int64_t bIdx, int64_t s1Idx, FagRunInfo &runInfo, FagConstInfo &constInfo,
                                      int64_t accumS2Len, int32_t actualSeqLensQ, int32_t actualSeqLensK);
    __aicore__ inline int32_t GetActualSeqLens(int64_t bIdx, GlobalTensor<int32_t> &actualSeqLensGm, int64_t &accumLen,
                                               FagConstInfo constInfo);
    __aicore__ inline LocalTensor<int32_t> LoadTopkToUb(uint64_t gmIndex, uint32_t count);

public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo,
                                             AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR dy, GM_ADDR y, GM_ADDR sparseIndices,
                                            GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR keyRope, GM_ADDR dq,
                                            GM_ADDR dk, GM_ADDR dv, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
                                            GM_ADDR workspace, GM_ADDR sinks, GM_ADDR dSinks);
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitCubeVecSharedParams(FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                                   float qScaleDs);
    __aicore__ inline void GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                    FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm, LocalTensor<CALC_TYPE> &dkInTensor,
                                      LocalTensor<CALC_TYPE> &dvInTensor, FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ScatterAddHead64(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                            const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                            const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                            const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm, FagConstInfo &constInfo,
                                            FagRunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm, FagConstInfo &constInfo,
                                           FagRunInfo &runInfo);
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId);
    __aicore__ inline void FinalizeDSinkAcc(FagConstInfo &constInfo);
    __aicore__ inline void ReduceDSink(LocalTensor<CALC_TYPE> &scratchTensor, LocalTensor<CALC_TYPE> &reduceOutTensor,
                                       FagConstInfo &constInfo);
    template <const bool IS_DQ = false>
    __aicore__ inline void CopyUB2L1(FagConstInfo &constInfo, FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                     LocalTensor<INPUT_TYPE> &srcTensor);

    constexpr static bool IS_D_NO_EQUAL = true;
    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static float FP8_MAX = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ? 57344 : 448;
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = 128;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t DETER_EXCEED_USE_SIZE = 2 * 1024;
    constexpr static uint32_t TOPK_UB_SIZE = SFAG_GATHER_S2_HEAD_N * sizeof(int32_t);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;

    // vector gm addr
    GlobalTensor<INPUT_TYPE> keyGm, keyRopeGm, dyGm;
    GlobalTensor<OUTDTYPE> yGm, pseGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm;
    GM_ADDR pseSlope;
    GlobalTensor<uint8_t> dropMaskWorkspaceGm;
    GlobalTensor<float> dsAmaxWorkSpaceGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> actualSeqLengthsKeyGm;
    GlobalTensor<float> sinksGm, dSinksGm, dSinkWorkSpaceGm;

    // ub buffer
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> topkUbBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TQue<QuePosition::VECOUT, 1> dsAmaxOutQue;
    TQue<QuePosition::VECIN, 1> sinksInQue;
    TQue<QuePosition::VECIN, 1> maxForSinkQue;
    TQue<QuePosition::VECIN, 1> sumForSinkQue;
    TBuf<> sinkTensor;
    TBuf<> dSinkAcc;
    TBuf<> sinkRowSumBuf;

    TPipe *pipe;
    SFagTilingType tilingData;

    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;

    // optional info
    AttenMaskInfo *attenMaskInfoPtr;
    PseInfo *pseInfoPtr;

    DataCopyPadExtParams<INPUT_TYPE> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams intriParamsRope;
    DataCopyExtParams outParamK;
    DataCopyExtParams outParamRope;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData,
                                                                     uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                     uint32_t vSubBlockIdx, FagConstInfo &constInfo,
                                                                     AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
    attenMaskInfoPtr = &attenMaskInfo;
    pseInfoPtr = &pseInfo;

    intriParamsKey.blockLen = constInfo.selectedBlockSize * constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    intriParamsRope.blockLen = constInfo.selectedBlockSize * constInfo.dRopeSize * sizeof(INPUT_TYPE);
    intriParamsRope.dstStride = 0;
    intriParamsRope.blockCount = 2;

    outParamK.blockCount = 2;
    outParamK.blockLen = constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
    outParamK.srcStride = 0;
    outParamK.dstStride = constInfo.dRopeSize * sizeof(INPUT_TYPE);

    outParamRope.blockCount = 2;
    outParamRope.blockLen = constInfo.dRopeSize * sizeof(INPUT_TYPE);
    outParamRope.srcStride = 0;
    outParamRope.dstStride = constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR key, GM_ADDR dy, GM_ADDR y,
                                                                    GM_ADDR sparseIndices, GM_ADDR softmaxMax,
                                                                    GM_ADDR softmaxSum, GM_ADDR keyRope, GM_ADDR dq,
                                                                    GM_ADDR dk, GM_ADDR dv, GM_ADDR actualSeqQlen,
                                                                    GM_ADDR actualSeqKvlen, GM_ADDR workspace,
                                                                    GM_ADDR sinks, GM_ADDR dSinks)
{
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    keyRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)keyRope);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);

    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);

    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    if constexpr (!KV_MERGE) {
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    }
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlen);
    actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqKvlen);

    if constexpr (IS_SINKS) {
        sinksGm.SetGlobalBuffer((__gm__ float *)sinks);
        dSinksGm.SetGlobalBuffer((__gm__ float *)dSinks);
        dSinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dSinkWorkSpaceOffset / sizeof(float));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitUbBuffer()
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dy and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * maxSumQue: for max sum double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);

    pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE + DETER_EXCEED_USE_SIZE);
    pipe->InitBuffer(pOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
    pipe->InitBuffer(topkUbBuf, TOPK_UB_SIZE);

    // dSinkAcc 跨 S1 循环累加，需清零
    if constexpr (IS_SINKS) {
        pipe->InitBuffer(sinksInQue, 1, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(maxForSinkQue, 1, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(sumForSinkQue, 1, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(sinkTensor, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(dSinkAcc, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(sinkRowSumBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
        LocalTensor<CALC_TYPE> dSinkAccTensor = dSinkAcc.Get<CALC_TYPE>();
        Duplicate(dSinkAccTensor, 0.0f, VECTOR_BASEM);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline LocalTensor<int32_t> FAGBlockVec<TEMPLATE_ARGS>::LoadTopkToUb(uint64_t gmIndex, uint32_t count)
{
    constexpr uint32_t INT32_PER_BLK = 8; // 32B
    uint32_t copyCnt = (count + INT32_PER_BLK - 1) / INT32_PER_BLK * INT32_PER_BLK;
    LocalTensor<int32_t> topkUb = topkUbBuf.Get<int32_t>();
    DataCopy(topkUb, topkIndicesGm[gmIndex], copyCnt);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
    return topkUb;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                                            FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    outParamK.blockCount = 2;
    outParamRope.blockCount = 2;
    uint64_t gmOffset = runInfo.t1Index * (constInfo.n2Size * constInfo.selectedBlockCount) +
                        runInfo.n2Index * constInfo.selectedBlockCount + runInfo.blkCntOffset;
    uint32_t mergePingPong = 0;
    AscendC::TEventID mte2WaitMte3EventId;
    AscendC::TEventID mte3WaitMte2EventId;
    AscendC::TEventID mte2WaitMte3Ping = EVENT_ID0;
    AscendC::TEventID mte2WaitMte3Pong = EVENT_ID1;
    AscendC::TEventID mte3WaitMte2Ping = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
    AscendC::TEventID mte3WaitMte2Pong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();

    // ------------- MergeKv --------------
    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = GetSubBlockIdx() == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = GetSubBlockIdx() == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outWsOffset = GetSubBlockIdx() == 0 ? 0 : firstVecEnd * constInfo.selectedBlockSize * constInfo.dTotalSize;
    uint32_t i;

    LocalTensor<INPUT_TYPE> gatherTensorPing = dSOutQue.AllocTensor<INPUT_TYPE>();
    LocalTensor<INPUT_TYPE> gatherTensorPong = pOutQue.AllocTensor<INPUT_TYPE>();

    LocalTensor<INPUT_TYPE> gatherRopeTensorPing = gatherTensorPing[2 * constInfo.commonConstInfo.dSize];
    LocalTensor<INPUT_TYPE> gatherRopeTensorPong = gatherTensorPong[2 * constInfo.commonConstInfo.dSize];

    // N<=64 UnDeter: copy this tile's INT32 indices into leftover UB, then GetValue from UB.
    bool useUbTopk = false;
    LocalTensor<int32_t> topkUb;
    if constexpr (!IS_DETER) {
        if (constInfo.isHeadNLe64 && runInfo.actualSelCntOffset > 0) {
            useUbTopk = true;
            topkUb = LoadTopkToUb(gmOffset, static_cast<uint32_t>(runInfo.actualSelCntOffset));
        }
    }

    for (i = curBlk; i < curBlk + curActualSelCntOffset / 2 * 2; i += 2) {
        int64_t keyOffset1 =
            (useUbTopk ? topkUb.GetValue(i) : topkIndicesGm.GetValue(gmOffset + i)) * constInfo.selectedBlockSize;
        int64_t keyOffset2 = (useUbTopk ? topkUb.GetValue(i + 1) : topkIndicesGm.GetValue(gmOffset + i + 1)) *
                             constInfo.selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - constInfo.selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsRope.blockCount = 2;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        // CopyIn
        intriParamsKey.srcStride =
            s2OrgStride * constInfo.n2Size * constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;

        if (keyOffset2 <= keyOffset1) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor,
                        keyGm[runInfo.keyOffsetWithRopeForMm12 +
                              keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                        intriParamsKey, padParams);
            DataCopyPad(gatherTensor[constInfo.selectedBlockSize * constInfo.commonConstInfo.dSize],
                        keyGm[runInfo.keyOffsetWithRopeForMm12 +
                              keyOffset2 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                        intriParamsKey, padParams);
        } else {
            DataCopyPad(gatherTensor,
                        keyGm[runInfo.keyOffsetWithRopeForMm12 +
                              keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                        intriParamsKey, padParams);
        }

        intriParamsRope.srcStride = s2OrgStride * constInfo.n2Size * constInfo.dRopeSize * sizeof(INPUT_TYPE);
        LocalTensor<INPUT_TYPE> &gatherRopeTensor = mergePingPong ? gatherRopeTensorPing : gatherRopeTensorPong;

        if constexpr (IS_ROPE) {
            if (keyOffset2 <= keyOffset1) {
                intriParamsRope.blockCount = 1;
                DataCopyPad(
                    gatherRopeTensor,
                    keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset1 * constInfo.n2Size * constInfo.dRopeSize],
                    intriParamsRope, padParams);
                DataCopyPad(
                    gatherRopeTensor[constInfo.selectedBlockSize * constInfo.dRopeSize],
                    keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset2 * constInfo.n2Size * constInfo.dRopeSize],
                    intriParamsRope, padParams);
            } else {
                DataCopyPad(
                    gatherRopeTensor,
                    keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset1 * constInfo.n2Size * constInfo.dRopeSize],
                    intriParamsRope, padParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        // CopyOut
        DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        if constexpr (IS_ROPE) {
            DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + constInfo.commonConstInfo.dSize + outWsOffset],
                        gatherRopeTensor, outParamRope);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        outWsOffset += 2 * constInfo.dTotalSize;
        mergePingPong = 1 - mergePingPong;
    }
    if (i < curActualSelCntEnd) {
        int64_t keyOffset1 =
            (useUbTopk ? topkUb.GetValue(i) : topkIndicesGm.GetValue(gmOffset + i)) * constInfo.selectedBlockSize;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);

        // CopyIn
        intriParamsRope.blockCount = 1;
        intriParamsKey.blockCount = 1;
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<INPUT_TYPE> &gatherRopeTensor = mergePingPong ? gatherRopeTensorPing : gatherRopeTensorPong;

        DataCopyPad(
            gatherTensor,
            keyGm[runInfo.keyOffsetWithRopeForMm12 + keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
            intriParamsKey, padParams);
        if constexpr (IS_ROPE) {
            DataCopyPad(
                gatherRopeTensor,
                keyRopeGm[runInfo.commonRunInfo.kRopeOffset + keyOffset1 * constInfo.n2Size * constInfo.dRopeSize],
                intriParamsRope, padParams);
        }

        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);

        outParamK.blockCount = 1;
        outParamRope.blockCount = 1;
        // CopyOut
        DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        if constexpr (IS_ROPE) {
            DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + constInfo.commonConstInfo.dSize + outWsOffset],
                        gatherRopeTensor, outParamRope);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        mergePingPong = 1 - mergePingPong;
    }
    dSOutQue.FreeTensor(gatherTensorPing);
    pOutQue.FreeTensor(gatherTensorPong);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    uint32_t loopNum = Ceil<uint32_t>(runInfo.halfGRealSize, constInfo.sfmgMaxLoopSize);
    uint32_t loopSize = Ceil<uint32_t>(runInfo.halfGRealSize, loopNum);
    uint32_t tailLoopSize = runInfo.halfGRealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    for (int32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
        }
        CopyInSoftmaxGrad<INPUT_TYPE, CALC_TYPE, OUTDTYPE, VECTOR_BASEM, 512, IS_D_NO_EQUAL>(
            constInfo, runInfo, loopIdx, curLoopSize, loopSize, attenMaskOrYInQue, pseOrDyInQue, dyGm, yGm);
        CalculateCastSoftmaxGrad<INPUT_TYPE, CALC_TYPE, OUTDTYPE, VECTOR_BASEM, 512>(
            constInfo, curLoopSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor[loopSize * loopIdx],
            runInfo.quantScaleInfo.deqScaleDyValue);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                              int64_t taskId)
{
    CopyInMaxSum<float, VECTOR_BASEM>(constInfo, runInfo, maxSumQue[taskId & 1], softmaxMaxGm, softmaxSumGm);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::FinalizeDSinkAcc(FagConstInfo &constInfo)
{
    if constexpr (IS_SINKS) {
        // dSinkAcc 由 V3 的 Mul/Sub 写入，UB→GM 的 DataCopy 走 MTE3，须等 V 完成
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);

        LocalTensor<CALC_TYPE> dSinkAccTensor = dSinkAcc.Get<CALC_TYPE>();
        int64_t gSize = constInfo.commonConstInfo.gSize;
        int64_t firstHalfG = (gSize + 1) >> 1;
        int64_t halfGRealSize = (vSubBlockIdx == 0) ? firstHalfG : (gSize - firstHalfG);
        int64_t halfGOffset = firstHalfG * vSubBlockIdx;
        if (halfGRealSize <= 0) {
            return; // gSize=1 时 sub1 halfG=0，无数据可写
        }

        if constexpr (IS_DETER) {
            // 写本核 slot，由循环外 ReduceDSink 单 writer 跨核归约
            int64_t slotOffset = cBlockIdx * MAX_N1 + halfGOffset;
            DataCopyExtParams dSinkCopyParams;
            dSinkCopyParams.blockCount = 1;
            dSinkCopyParams.blockLen = static_cast<uint16_t>(halfGRealSize * sizeof(CALC_TYPE));
            dSinkCopyParams.srcStride = 0;
            dSinkCopyParams.dstStride = 0;
            DataCopyPad(dSinkWorkSpaceGm[slotOffset], dSinkAccTensor, dSinkCopyParams);
        } else {
            // 用 DataCopyPad（字节级 blockLen）而非 DataCopy——halfGOffset/halfGRealSize 在小 N1 下
            // 非 32B 对齐，DataCopy 是对齐搬运接口语义未定义；DataCopyPad 专为非对齐设计，稳妥。
            // V→MTE3 同步已在函数头完成（guard V3 写入的 dSinkAcc），此处无需重复
            DataCopyExtParams dSinkCopyParams;
            dSinkCopyParams.blockCount = 1;
            dSinkCopyParams.blockLen = static_cast<uint16_t>(halfGRealSize * sizeof(CALC_TYPE));
            dSinkCopyParams.srcStride = 0;
            dSinkCopyParams.dstStride = 0;
            SetAtomicAdd<CALC_TYPE>();
            DataCopyPad(dSinksGm[halfGOffset], dSinkAccTensor, dSinkCopyParams);
            SetAtomicNone();
        }
    }
}

// scratch/reduceOut 复用 mm1ResBuf[0]/[1]（循环后空闲，TBuf 无 Position 约束，手动管同步）
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ReduceDSink(LocalTensor<CALC_TYPE> &scratchTensor,
                                                               LocalTensor<CALC_TYPE> &reduceOutTensor,
                                                               FagConstInfo &constInfo)
{
    if constexpr (IS_SINKS && IS_DETER) {
        // 跨核 slot 可见性由 ProcessDeter 中 ReduceDSink 调用前的 SyncALLCores()（SyncAll<false> 硬 barrier）保证。
        // 本处不再设置 CrossCore flag：David 上 flag 11-14 非可靠全核 barrier，单 writer 会读到未写完的 slot。
        int64_t usedCoreNum = tilingData->baseParams.usedCoreNum;
        if (cBlockIdx == usedCoreNum - 1 && vSubBlockIdx == 0) {
            int64_t gSize = constInfo.commonConstInfo.gSize;
            // MTE2 搬 usedCoreNum 个 slot（每核独占 MAX_N1 槽位，各写各段不相交）→ scratch 对应段
            DataCopyPadExtParams<CALC_TYPE> copyPadParams; // 默认 isPad=false：等长搬运，不补位
            for (int64_t i = 0; i < usedCoreNum; i++) {
                DataCopyExtParams copyIn;
                copyIn.blockCount = 1;
                copyIn.blockLen = static_cast<uint16_t>(gSize * sizeof(CALC_TYPE));
                copyIn.srcStride = 0;
                copyIn.dstStride = 0;
                DataCopyPad(scratchTensor[i * MAX_N1], dSinkWorkSpaceGm[i * MAX_N1], copyIn, copyPadParams);
            }
            // MTE2→V 显式同步：regbase 无自动同步，等所有 slot 搬入完成
            event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            // reduceOut = Σ_i scratch[i*MAX_N1]
            Duplicate(reduceOutTensor, 0.0f, gSize);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t i = 0; i < usedCoreNum; i++) {
                Add(reduceOutTensor, reduceOutTensor, scratchTensor[i * MAX_N1], gSize);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // V→MTE3 显式同步：regbase 无自动同步，等归约完成
            event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint16_t>(gSize * sizeof(CALC_TYPE));
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(dSinksGm[0], reduceOutTensor, outParams);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DQ>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                             LocalTensor<INPUT_TYPE> &dstTensor,
                                                             LocalTensor<INPUT_TYPE> &srcTensor)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.firstHalfGRealSize * FRACTAL_NZ_C0_SIZE;
    DataCopyParams dataCopyParams;
    uint32_t copyN = VECTOR_BASEN;
    if (constInfo.isHeadNLe64) {
        copyN =
            IS_FP8_INPUT ? AlignTo32(runInfo.commonRunInfo.s2RealSize) : AlignTo16(runInfo.commonRunInfo.s2RealSize);
    }
    dataCopyParams.blockCount = copyN / FRACTAL_NZ_C0_SIZE;
    dataCopyParams.blockLen = (uint16_t)(runInfo.halfGRealSize * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    dataCopyParams.srcStride =
        (uint16_t)((VECTOR_BASEM + 1 - runInfo.halfGRealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM);
    uint32_t s1RealSizeAlignTo16 = AlignTo16(constInfo.commonConstInfo.gSize);
    dataCopyParams.dstStride = (s1RealSizeAlignTo16 - runInfo.halfGRealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_SINKS) {
        if (runInfo.halfGRealSize == 0) {
            return;
        }
        LocalTensor<CALC_TYPE> maxSumTensor = maxSumQue[runInfo.commonRunInfo.taskIdMod2].DeQue<CALC_TYPE>();
        CopyInSinksAndMaxSum<CALC_TYPE, VECTOR_BASEM>(constInfo, runInfo, sinksGm, softmaxMaxGm, softmaxSumGm,
                                                      sinksInQue, maxForSinkQue, sumForSinkQue);
        // maxSumTensor 由 Reuse 重载持有，下方统一 Free
        CalculatePseMulsSelSimpleSoftMaxReuse<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, maxSumTensor, attenMaskOrYInQue, pseOrDyInQue,
            mm2ResTensor, mm2ResTensor, pseSlope);
        LocalTensor<CALC_TYPE> sinkPkTensor = sinkTensor.Get<CALC_TYPE>();
        // TQue DeQue 自动 MTE2→V 同步，无需额外 PipeBarrier
        CalculateSinkSimpleSoftMax<CALC_TYPE, VECTOR_BASEM>(constInfo, runInfo, sinksInQue, maxForSinkQue,
                                                            sumForSinkQue, sinkPkTensor);
        // SinkSoftMaxVF 是 __simd_vf__ 微指令，StoreAlign(P_sink→sinkTensor) 的写依赖编译器自动 PIPE_V
        // 同步无法感知；det 调度下 V3 标准 Mul 会读到上一 task 的旧 P_sink，强制 V2 写入先于 V3 读取
        AscendC::PipeBarrier<PIPE_V>();
        maxSumQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(maxSumTensor);
    } else {
        CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, maxSumQue[runInfo.commonRunInfo.taskIdMod2],
            attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, pseSlope);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////

    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>();
    // rowSumNeg[h] = -Σ_j P[h,j]·dp[h,j]（P∈mm2、dp∈mm1，fp32），须在 BroadcastSubMul 覆盖 mm1(dp) 前计算
    // 比复用 Di=rowsum(out_bf16·dy) 更精确（golden 已同步为 -(p*dp).sum(-1)）
    // dSinkAcc 用 Add 跨 key-block task 累加（每个 task 的 Σ_j 是部分和）
    if constexpr (IS_SINKS) {
        LocalTensor<CALC_TYPE> sinkRowSumTensor = sinkRowSumBuf.Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> sinkPkTensor = sinkTensor.Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> dSinkAccTensor = dSinkAcc.Get<CALC_TYPE>();
        if (runInfo.halfGRealSize > 0 && runInfo.commonRunInfo.s2RealSize > 0) {
            CalculateSinkNegRowSum<CALC_TYPE, VECTOR_BASEN>(sinkRowSumTensor, mm2ResTensor, mm1ResTensor,
                                                            static_cast<uint16_t>(runInfo.halfGRealSize),
                                                            static_cast<uint16_t>(runInfo.commonRunInfo.s2RealSize));
            // dSinkAcc += P_sink·rowSumNeg；P_sink 就绪由 ProcessVec2 的 PIPE_V barrier 保证
            Mul(sinkPkTensor, sinkPkTensor, sinkRowSumTensor, runInfo.halfGRealSize);
            Add(dSinkAccTensor, dSinkAccTensor, sinkPkTensor, runInfo.halfGRealSize);
        }
    }
    if (runInfo.commonRunInfo.s2RealSize > static_cast<uint32_t>(S2TemplateType::Aligned64)) {
        BroadcastSubMul<CALC_TYPE, static_cast<uint32_t>(S2TemplateType::Aligned128), 0>(
            mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor, runInfo.halfGRealSize,
            runInfo.commonRunInfo.s2RealSize);
    } else {
        BroadcastSubMul<CALC_TYPE, static_cast<uint32_t>(S2TemplateType::Aligned64), 0>(
            mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor, runInfo.halfGRealSize,
            runInfo.commonRunInfo.s2RealSize);
    }

    LocalTensor<uint8_t> selrIndexesTensor;
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<INPUT_TYPE>();

    // copy ds from ub to l1
    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1<true>(constInfo, runInfo, dsL1Tensor, vecOutBuffer);

    dSOutQue.FreeTensor(vecOutBuffer);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<uint8_t> selrIndexesTensor;
    LocalTensor<INPUT_TYPE> vecOutBuffer1 = pOutQue.AllocTensor<INPUT_TYPE>();
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer1, mm2ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer1);
    pOutQue.DeQue<INPUT_TYPE>();

    // copy p from ub to l1
    LocalTensor<INPUT_TYPE> pL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    CopyUB2L1(constInfo, runInfo, pL1Tensor, vecOutBuffer1);

    pOutQue.FreeTensor(vecOutBuffer1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(FagCVSharedParams &sharedParams,
                                                                           int32_t aicIdx, uint8_t subBlockIdx,
                                                                           float qScaleDs)
{
    sharedParams.qScaleDs = qScaleDs;
    /* ssbuf send message */
    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t *>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);    //
#pragma unroll
            for (int i = 0; i < sizeof(FagCVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAddHead64(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                                    const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                                    const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                                                    const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                                                    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    // Dedicated scratch in dSOutQue/pOutQue (idle after Gather/Softmax FreeTensor).
    // Rope HEAD_DIM_ALIGN=576: ping+pong 4-row dk is 18KB and does not fit in the
    // first 16KB of mm1/mm2; borrowing that slot collides with live mm12.
    constexpr uint32_t SCATTER_UB_ROWS = 4;
    constexpr uint32_t SCATTER_DK_BYTES =
        2 * SCATTER_UB_ROWS * HEAD_DIM_ALIGN * static_cast<uint32_t>(sizeof(CALC_TYPE));
    constexpr uint32_t SCATTER_DV_BYTES = 2 * SCATTER_UB_ROWS * 512 * static_cast<uint32_t>(sizeof(CALC_TYPE));
    constexpr uint32_t DS_QUE_BYTES = VECTOR_BASEM * VREG_SIZE + VREG_SIZE + DETER_EXCEED_USE_SIZE;
    constexpr uint32_t P_QUE_BYTES = VECTOR_BASEM * VREG_SIZE + VREG_SIZE;
    static_assert(SCATTER_DK_BYTES <= DS_QUE_BYTES, "ScatterAddHead64 dk ping-pong exceeds dSOutQue");
    static_assert(SCATTER_DV_BYTES <= P_QUE_BYTES, "ScatterAddHead64 dv ping-pong exceeds pOutQue");

    int64_t UB_ROW_SIZE = SCATTER_UB_ROWS;
    int64_t s2RealSize = runInfo.commonRunInfo.s2RealSize;
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = (vSubBlockIdx == 0) ? firstCoreKSize : (s2RealSize - firstCoreKSize);
    if (currentCoreKSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> dkInTensor = dSOutQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> dvInTensor = pOutQue.AllocTensor<CALC_TYPE>();

    uint64_t gmOffset = runInfo.t1Index * (constInfo.n2Size * constInfo.selectedBlockCount) +
                        vSubBlockIdx * firstCoreKSize + runInfo.blkCntOffset;
    bool useUbTopk = true;
    LocalTensor<int32_t> topkUb = LoadTopkToUb(gmOffset, static_cast<uint32_t>(currentCoreKSize));

    SetAtomicAdd<CALC_TYPE>();
    int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
    int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    GlobalTensor<float> dkOutGm = dkWorkSpaceGm[runInfo.keyOffsetWithRope];
    GlobalTensor<float> dvOutGm = dvWorkSpaceGm[runInfo.commonRunInfo.valueOffset];
    int64_t currentMm4SrcOffset =
        runInfo.mm4ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    int64_t currentMm5SrcOffset =
        runInfo.mm5ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * 512;

    // 4-row ping/pong in dedicated dSOutQue (dk) / pOutQue (dv).
    uint32_t dkSlot = static_cast<uint32_t>(UB_ROW_SIZE * HEAD_DIM_ALIGN);
    uint32_t dvSlot = static_cast<uint32_t>(UB_ROW_SIZE * 512);
    uint32_t pingPong = 0;
    event_t mte3ToMte2Ping = EVENT_ID0;
    event_t mte3ToMte2Pong = EVENT_ID1;

    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPong ? mte3ToMte2Pong : mte3ToMte2Ping;
        uint32_t dkOff = pingPong * dkSlot;
        uint32_t dvOff = pingPong * dvSlot;
        WaitFlag<HardEvent::MTE3_MTE2>(backEvent);
        DataCopy(dkInTensor[dkOff], mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * dkSlot], dkSlot);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor[dkOff], dkInTensor[dkOff], (float)constInfo.scaleValue, dkSlot);
        DataCopy(dvInTensor[dvOff], mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * dvSlot], dvSlot);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
            int32_t s2Idx = useUbTopk ? topkUb.GetValue(static_cast<uint32_t>(loop * UB_ROW_SIZE + row)) :
                                        topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
            if (s2Idx >= 0) {
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[dkOff + row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                DataCopy(dvOutGm[s2Idx * constInfo.commonConstInfo.dSizeV],
                         dvInTensor[dvOff + row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(backEvent);
        pingPong = 1 - pingPong;
    }

    event_t backEvent = pingPong ? mte3ToMte2Pong : mte3ToMte2Ping;
    uint32_t dkOff = pingPong * dkSlot;
    uint32_t dvOff = pingPong * dvSlot;
    WaitFlag<HardEvent::MTE3_MTE2>(backEvent);
    DataCopy(dkInTensor[dkOff], mm4ResWorkSpaceGm[currentMm4SrcOffset + (maxLoops - 1) * dkSlot],
             tailRows * HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Muls(dkInTensor[dkOff], dkInTensor[dkOff], (float)constInfo.scaleValue, tailRows * HEAD_DIM_ALIGN);
    DataCopy(dvInTensor[dvOff], mm5ResWorkSpaceGm[currentMm5SrcOffset + (maxLoops - 1) * dvSlot], tailRows * 512);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    for (int64_t row = 0; row < tailRows; row++) {
        int32_t s2Idx = useUbTopk ? topkUb.GetValue(static_cast<uint32_t>((maxLoops - 1) * UB_ROW_SIZE + row)) :
                                    topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
        if (s2Idx >= 0) {
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[dkOff + row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
            DataCopy(dvOutGm[s2Idx * constInfo.commonConstInfo.dSizeV],
                     dvInTensor[dvOff + row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(backEvent);
    SetAtomicNone();
    dSOutQue.FreeTensor(dkInTensor);
    pOutQue.FreeTensor(dvInTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                                              LocalTensor<CALC_TYPE> &dkInTensor,
                                                              LocalTensor<CALC_TYPE> &dvInTensor,
                                                              FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    // N<=64: shrink the per-loop tile to 4 rows so ping(4 rows) + pong(4 rows)
    // fit in the first 16KB of the opposite mm1/mm2 slot and do not reach the
    // [16KB,32KB) region that the next task's mm12 rewrites.
    int64_t UB_ROW_SIZE = 8;
    int64_t s2RealSize = runInfo.commonRunInfo.s2RealSize;
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = (vSubBlockIdx == 0) ? firstCoreKSize : (s2RealSize - firstCoreKSize);
    if (currentCoreKSize == 0) {
        return;
    }

    SetAtomicAdd<CALC_TYPE>();
    int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
    int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    GlobalTensor<float> dkOutGm = dkWorkSpaceGm[runInfo.keyOffsetWithRope];
    GlobalTensor<float> dvOutGm = dvWorkSpaceGm[runInfo.commonRunInfo.valueOffset];
    int64_t currentMm4SrcOffset =
        runInfo.mm4ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    int64_t currentMm5SrcOffset = runInfo.mm5ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize *
                                                             constInfo.commonConstInfo.dSizeV;
    uint64_t gmOffset = runInfo.t1Index * (constInfo.n2Size * constInfo.selectedBlockCount) +
                        vSubBlockIdx * firstCoreKSize + runInfo.blkCntOffset;
    bool useUbTopk = currentCoreKSize > 0;
    LocalTensor<int32_t> topkUb;
    if (useUbTopk) {
        topkUb = LoadTopkToUb(gmOffset, static_cast<uint32_t>(currentCoreKSize));
    }
    // 1 - main loop
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        DataCopy(dkInTensor, mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                 UB_ROW_SIZE * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, UB_ROW_SIZE * HEAD_DIM_ALIGN);
        DataCopy(dvInTensor,
                 mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV],
                 UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        if constexpr (KV_MERGE) {
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                    dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        }
        int32_t s2IdxLocal[8];
        for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
            s2IdxLocal[row] = useUbTopk ? topkUb.GetValue(static_cast<uint32_t>(loop * UB_ROW_SIZE + row)) :
                                          topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
        }
        for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
            int32_t s2Idx = s2IdxLocal[row];
            if (s2Idx >= 0) {
                if constexpr (!KV_MERGE) {
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                }
                DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                if constexpr (!KV_MERGE) {
                    DataCopy(dvOutGm[s2Idx * constInfo.commonConstInfo.dSizeV],
                             dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
                }
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    // 2 - tail loop
    DataCopy(dkInTensor, mm4ResWorkSpaceGm[currentMm4SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN],
             tailRows * HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, tailRows * HEAD_DIM_ALIGN);
    DataCopy(dvInTensor,
             mm5ResWorkSpaceGm[currentMm5SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV],
             tailRows * constInfo.commonConstInfo.dSizeV);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    if constexpr (KV_MERGE) {
        PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < tailRows; row++) {
            Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }
    int32_t s2IdxTail[8];
    for (int64_t row = 0; row < tailRows; row++) {
        s2IdxTail[row] = useUbTopk ? topkUb.GetValue(static_cast<uint32_t>((maxLoops - 1) * UB_ROW_SIZE + row)) :
                                     topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
    }
    for (int64_t row = 0; row < tailRows; row++) {
        int32_t s2Idx = s2IdxTail[row];
        if (s2Idx >= 0) {
            if constexpr (!KV_MERGE) {
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            }
            DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
            if constexpr (!KV_MERGE) {
                DataCopy(dvOutGm[s2Idx * constInfo.commonConstInfo.dSizeV],
                         dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
            }
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    SetAtomicNone();
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm,
                                                                   FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t UB_ROW_SIZE = 8;
    int64_t usedCoreNum = tilingData->baseParams.usedCoreNum;
    int64_t coreNum = GetBlockNum();
    int64_t totalVec = coreNum * 2;
    int64_t bS1Index = -1;
    int64_t bIdx = 0;
    int64_t s1Idx = 0;
    int64_t preBIdx = -1;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int64_t actualSeqLensQ = 0;
    int64_t actualSeqLensK = 0;
    LocalTensor<CALC_TYPE> dkInTensor = dSOutQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> dvInTensor = pOutQue.AllocTensor<CALC_TYPE>();
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    // 全核一致的 token 上界：TND 下物理 T1 之外还要去掉尾 padding。提到循环外避免每个 idx 重复读 GM。
    int64_t tokenLimit = tilingData->baseParams.totalSize;
    if constexpr (IS_TND) {
        int64_t validT1 = this->actualSeqLengthsQueryGm.GetValue(constInfo.bSize - 1);
        tokenLimit = Min(tokenLimit, validT1);
    }

    for (int64_t idx = 0; idx < usedCoreNum; idx++) {
        bS1Index = runInfo.sTaskId * usedCoreNum + idx;
        if (bS1Index >= tokenLimit) {
            break;
        }

        GetBS1Index(bS1Index, bIdx, s1Idx, constInfo, preBIdx);
        if (bIdx != preBIdx) {
            actualSeqLensQ = GetActualSeqLens(bIdx, this->actualSeqLengthsQueryGm, accumS1Len, constInfo);
            actualSeqLensK = GetActualSeqLens(bIdx, this->actualSeqLengthsKeyGm, accumS2Len, constInfo);
            preBIdx = bIdx;
        }
        GetRunInfo(bIdx, s1Idx, runInfo, constInfo, accumS2Len, actualSeqLensQ, actualSeqLensK);

        // 对应s2real的平均和取模值，做均分操作
        int64_t remainder = runInfo.actualSelectedBlockCount % totalVec;
        int64_t avgSize = runInfo.actualSelectedBlockCount / totalVec;
        // 前remainder 个块分配 avgSize + 1， 其余分配avgSize
        int64_t currentCoreKSize = avgSize + (this->vBlockIdx < remainder ? 1 : 0);
        if (currentCoreKSize == 0) {
            // V核同步等待所有V核完成某一个C核上S2的计算
            CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
            continue;
        }
        int64_t s2SrcOffset = this->vBlockIdx < remainder ?
                                  (this->vBlockIdx * (avgSize + 1)) :
                                  (remainder * (avgSize + 1) + (this->vBlockIdx - remainder) * avgSize);

        SetAtomicAdd<CALC_TYPE>();
        int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
        int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;

        int64_t currentMm4SrcOffset =
            runInfo.deterTaskIdMod2 * constInfo.selectedBlockCount * HEAD_DIM_ALIGN * coreNum +
            idx * constInfo.selectedBlockCount * HEAD_DIM_ALIGN +
            s2SrcOffset * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
        int64_t currentMm5SrcOffset =
            runInfo.deterTaskIdMod2 * constInfo.selectedBlockCount * constInfo.commonConstInfo.dSizeV * coreNum +
            idx * constInfo.selectedBlockCount * constInfo.commonConstInfo.dSizeV +
            s2SrcOffset * constInfo.selectedBlockSize * constInfo.commonConstInfo.dSizeV;
        // 1 - main loop
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        uint64_t gmOffset = bS1Index * (constInfo.n2Size * constInfo.selectedBlockCount) + s2SrcOffset;
        for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            DataCopy(dkInTensor, mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                     UB_ROW_SIZE * HEAD_DIM_ALIGN);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, UB_ROW_SIZE * HEAD_DIM_ALIGN);
            DataCopy(dvInTensor,
                     mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV],
                     UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            if constexpr (KV_MERGE) {
                PipeBarrier<PIPE_V>();
                for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                    Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                        dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            }
            int32_t s2IdxLocal[8];
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                s2IdxLocal[row] = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
            }
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                int32_t s2Idx = s2IdxLocal[row];
                if (s2Idx >= 0) {
                    if constexpr (!KV_MERGE) {
                        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    }
                    DataCopy(dkWorkSpaceGm[runInfo.keyOffsetWithRope + s2Idx * HEAD_DIM_ALIGN],
                             dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                    if constexpr (!KV_MERGE) {
                        int64_t dvOffset = runInfo.commonRunInfo.valueOffset + s2Idx * constInfo.commonConstInfo.dSizeV;
                        DataCopy(dvWorkSpaceGm[dvOffset], dvInTensor[row * constInfo.commonConstInfo.dSizeV],
                                 constInfo.commonConstInfo.dSizeV);
                    }
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        // 2 - tail loop
        DataCopy(dkInTensor, mm4ResWorkSpaceGm[currentMm4SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                 tailRows * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, tailRows * HEAD_DIM_ALIGN);
        DataCopy(
            dvInTensor,
            mm5ResWorkSpaceGm[currentMm5SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * constInfo.commonConstInfo.dSizeV],
            tailRows * constInfo.commonConstInfo.dSizeV);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        if constexpr (KV_MERGE) {
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < tailRows; row++) {
                Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                    dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        }
        int32_t s2IdxTail[8];
        for (int64_t row = 0; row < tailRows; row++) {
            s2IdxTail[row] = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
        }
        for (int64_t row = 0; row < tailRows; row++) {
            int32_t s2Idx = s2IdxTail[row];
            if (s2Idx >= 0) {
                if constexpr (!KV_MERGE) {
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                }
                DataCopy(dkWorkSpaceGm[runInfo.keyOffsetWithRope + s2Idx * HEAD_DIM_ALIGN],
                         dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                if constexpr (!KV_MERGE) {
                    DataCopy(
                        dvWorkSpaceGm[runInfo.commonRunInfo.valueOffset + s2Idx * constInfo.commonConstInfo.dSizeV],
                        dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
                }
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        // V核同步等待所有V核完成某一个C核上S2的计算
        CrossCoreSetFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
        CrossCoreWaitFlag<0, PIPE_MTE3>(SCATTER_VEC_SYNC_FLAG);
    }
    SetAtomicNone();
    dSOutQue.FreeTensor(dkInTensor);
    pOutQue.FreeTensor(dvInTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GetBS1Index(int64_t bS1Index, int64_t &bIdx, int64_t &s1Idx,
                                                               FagConstInfo &constInfo, int64_t startBIdx)
{
    if constexpr (IS_TND) {
        // 调用序列内 bS1Index 单调不降，从上一次命中的 batch 续扫，避免每个 token 都做 O(B) 的 GM 标量扫描。
        int64_t start = startBIdx > 0 ? startBIdx : 0;
        int64_t actualSum = start > 0 ? this->actualSeqLengthsQueryGm.GetValue(start - 1) : 0;
        bIdx = constInfo.bSize - 1;
        for (int64_t index = start; index < constInfo.bSize; index++) {
            int64_t actualLen = this->actualSeqLengthsQueryGm.GetValue(index);
            if (bS1Index < actualLen) {
                bIdx = index;
                break;
            }
            actualSum = actualLen;
        }
        s1Idx = bS1Index - actualSum;
    } else {
        bIdx = bS1Index / constInfo.commonConstInfo.s1Size;
        s1Idx = bS1Index - bIdx * constInfo.commonConstInfo.s1Size;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GetRunInfo(int64_t bIdx, int64_t s1Idx, FagRunInfo &runInfo,
                                                              FagConstInfo &constInfo, int64_t accumS2Len,
                                                              int32_t actualSeqLensQ, int32_t actualSeqLensK)
{
    int64_t accumS2Idx;
    if constexpr (IS_TND) {
        runInfo.commonRunInfo.actualS1Size = actualSeqLensQ;
        runInfo.commonRunInfo.actualS2Size = actualSeqLensK;
        accumS2Idx = accumS2Len;
    } else {
        accumS2Idx = bIdx * constInfo.commonConstInfo.s2Size;
    }
    runInfo.keyOffsetWithRope = accumS2Idx * constInfo.n2Size * constInfo.dTotalSize;
    runInfo.commonRunInfo.valueOffset = accumS2Idx * constInfo.n2Size * constInfo.commonConstInfo.dSizeV;
    // 必须与生产侧 FlashAttentionScoreGradKernelBase::GetActualSelCount 得到同一个值：
    // 少一个 Ceil(/selectedBlockSize) 就会让 Scatter 读到生产侧从未写过的 mm4/mm5 区域。
    int64_t maxS2 = static_cast<int64_t>(runInfo.commonRunInfo.actualS2Size);
    if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
        maxS2 = Max(static_cast<int64_t>(runInfo.commonRunInfo.actualS2Size) -
                        static_cast<int64_t>(runInfo.commonRunInfo.actualS1Size) + s1Idx + 1,
                    static_cast<int64_t>(0));
    }
    runInfo.actualSelectedBlockCount = Min(static_cast<int64_t>(constInfo.selectedBlockCount),
                                           CeilDiv(maxS2, static_cast<int64_t>(constInfo.selectedBlockSize)));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t FAGBlockVec<TEMPLATE_ARGS>::GetActualSeqLens(int64_t bIdx,
                                                                       GlobalTensor<int32_t> &actualSeqLensGm,
                                                                       int64_t &accumLen, FagConstInfo constInfo)
{
    if constexpr (IS_TND) {
        if (bIdx == 0) {
            accumLen = 0;
            return actualSeqLensGm.GetValue(0);
        } else {
            accumLen = actualSeqLensGm.GetValue(bIdx - 1);
            return (actualSeqLensGm.GetValue(bIdx) - accumLen);
        }
    } else {
        return 0;
    }
}

TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void InitUbBuffer() {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR dy, GM_ADDR y, GM_ADDR sparseIndices,
                                            GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR keyRope, GM_ADDR dq,
                                            GM_ADDR dk, GM_ADDR dv, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
                                            GM_ADDR workspace, GM_ADDR sinks, GM_ADDR dSinks) {};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo,
                                             AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {};
    __aicore__ inline void GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                    FagRunInfo &runInfo) {};
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm, LocalTensor<CALC_TYPE> &dkInTensor,
                                      LocalTensor<CALC_TYPE> &dvInTensor, FagConstInfo &constInfo,
                                      FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec4(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo) {};
    __aicore__ inline void ScatterAddHead64(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                            const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                            const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                            const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm, FagConstInfo &constInfo,
                                            FagRunInfo &runInfo) {};
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dvWorkSpaceGm, FagConstInfo &constInfo,
                                           FagRunInfo &runInfo) {};
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId) {};
    __aicore__ inline void FinalizeDSinkAcc(FagConstInfo &constInfo) {};
    __aicore__ inline void ReduceDSink(LocalTensor<CALC_TYPE> &scratchTensor, LocalTensor<CALC_TYPE> &reduceOutTensor,
                                       FagConstInfo &constInfo) {};
    __aicore__ inline void InitCubeVecSharedParams(FagCVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                                   float qScaleDs) {};
};

} // namespace SfagBaseApi

#endif
