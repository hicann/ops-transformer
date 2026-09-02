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
 * \file sparse_flash_mla_grad_block_vec.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_BLOCK_VEC_H
#define SPARSE_FLASH_MLA_GRAD_BLOCK_VEC_H

#include "sparse_flash_mla_grad_common.h"
#include "vector_api/cast_softmax_grad_sfmg.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax_sfmg.h"
#include "vector_api/vf_broadcast_sub_mul_sfmg.h"
#include "vector_api/vf_cast_transdata_deconflict_sfmg.h"
#include "vector_api/vf_cal_dsink.h"
#include "vector_api/vf_cal_softmax_sink.h"
namespace SfagBaseApi {
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t SYNC_V0_V1_DS_A_MAX_DONE_FLAG = 10;
constexpr uint32_t BIT_MASK_NUM = 8;
constexpr uint16_t SRCM = 128;

TEMPLATES_DEF
class FAGBlockVec {
public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo,
                                             AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out,
                                            GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices, GM_ADDR softmax_lse,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1,
                                            GM_ADDR dq, GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR actualSeqQlen,
                                            GM_ADDR actualSeqOriKvlen, GM_ADDR actualSeqCmpKvlen,
                                            GM_ADDR usedSeqQlenAddr, GM_ADDR usedOriSeqKvlenAddr,
                                            GM_ADDR usedCmpSeqKvlenAddr, GM_ADDR cmpResidualKAddr,
                                            GM_ADDR oriTopkLengthAddr, GM_ADDR cmpTopkLengthAddr, GM_ADDR workspace,
                                            FagConstInfo &constInfo);
    __aicore__ inline void InitUbBuffer(FagConstInfo &constInfo);
    __aicore__ inline void InitCubeVecSharedParams(FagRunInfo &runInfo, int32_t s2RealSize);
    __aicore__ inline int32_t GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
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
    template <const bool IS_SPARSE = true>
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm, LocalTensor<CALC_TYPE> &dkInTensor,
                                      LocalTensor<CALC_TYPE> &dvInTensor, FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dOriKWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dCmpKWorkSpaceGm, FagConstInfo &constInfo,
                                           FagRunInfo &runInfo);
    __aicore__ inline void CopyLse(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void CopySink(FagConstInfo &constInfo);
    __aicore__ inline void ProcessSoftmaxSink(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                          FagConstInfo &constInfo, FagRunInfo &runInfo);
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
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;

    // vector gm addr
    GlobalTensor<INPUT_TYPE> oriKVGm, cmpKVGm, dyGm;
    GlobalTensor<OUTDTYPE> yGm, pseGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> pseFloatGm, softmaxLseGm, sinkGm, dsinkGm, oriSoftmaxL1Gm, cmpSoftmaxL1Gm;
    GM_ADDR pseSlope;
    GlobalTensor<float> dsinkWorkSpaceGm;

    GlobalTensor<int32_t> oritopkIndicesGm;
    GlobalTensor<int32_t> cmptopkIndicesGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> actualSeqLengthsOriKeyGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKeyGm;

    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqOriKvlenAddr;
    GM_ADDR actualSeqCmpKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedOriSeqKvlenAddr;
    GM_ADDR usedCmpSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;
    GM_ADDR oriTopkLengthAddr;
    GM_ADDR cmpTopkLengthAddr;

    // ub buffer
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECOUT, 1> softmaxL1OutQue;
    TQue<QuePosition::VECIN, 1> LseQue[2];
    TQue<QuePosition::VECIN, 1> sinkQue;
    TQue<QuePosition::VECOUT, 1> dSinkQue;
    TBuf<> softmaxSinkResBuf[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TQue<QuePosition::VECOUT, 1> dsAmaxOutQue;

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
    DataCopyExtParams outParamK;

private:
    __aicore__ inline bool GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx, int64_t &bS1Index,
                                       int64_t &accumS1Len, int64_t &accumS2Len, int64_t &accumS3Len,
                                       int64_t &actualSeqLensQ, int64_t &actualSeqOriLensK, int64_t &actualSeqCmpLensK,
                                       int64_t &actualOriSelectedBlockCount, int64_t &actualCmpSelectedBlockCount,
                                       int64_t &winStart, int64_t taskId, FagConstInfo &constInfo);
    __aicore__ inline void GetRunInfo(int64_t bIdx, int64_t s1Idx, FagRunInfo &runInfo, FagConstInfo &constInfo,
                                      int64_t accumS2Len, int32_t actualSeqLensQ, int32_t actualSeqLensOriK,
                                      int32_t actualSeqLensCmpK);
    __aicore__ inline int32_t GetActualSeqLens(int64_t bIdx, GlobalTensor<int32_t> &actualSeqLensGm, int64_t &accumLen,
                                               FagConstInfo constInfo);
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

    outParamK.blockCount = 2;
    outParamK.blockLen = constInfo.commonConstInfo.dSize * sizeof(INPUT_TYPE);
    outParamK.srcStride = 0;
    outParamK.dstStride = 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out, GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices,
    GM_ADDR softmax_lse, GM_ADDR sink, GM_ADDR dsink, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1, GM_ADDR dq,
    GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR actualSeqQlen, GM_ADDR actualSeqOriKvlen, GM_ADDR actualSeqCmpKvlen,
    GM_ADDR usedSeqQlenAddr, GM_ADDR usedOriSeqKvlenAddr, GM_ADDR usedCmpSeqKvlenAddr, GM_ADDR cmpResidualKAddr,
    GM_ADDR oriTopkLengthAddr, GM_ADDR cmpTopkLengthAddr, GM_ADDR workspace, FagConstInfo &constInfo)
{
    oriKVGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)ori_kv);
    cmpKVGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)cmp_kv);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)d_out);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)out);
    oriSoftmaxL1Gm.SetGlobalBuffer((__gm__ float *)ori_softmax_l1);
    cmpSoftmaxL1Gm.SetGlobalBuffer((__gm__ float *)cmp_softmax_l1);
    actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlen);
    actualSeqLengthsOriKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqOriKvlen);
    actualSeqLengthsCmpKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqCmpKvlen);

    this->actualSeqQlenAddr = actualSeqQlen;
    this->actualSeqOriKvlenAddr = actualSeqOriKvlen;
    this->actualSeqCmpKvlenAddr = actualSeqCmpKvlen;
    this->usedSeqQlenAddr = usedSeqQlenAddr;
    this->usedOriSeqKvlenAddr = usedOriSeqKvlenAddr;
    this->usedCmpSeqKvlenAddr = usedCmpSeqKvlenAddr;
    this->cmpResidualKAddr = cmpResidualKAddr;
    this->oriTopkLengthAddr = oriTopkLengthAddr;
    this->cmpTopkLengthAddr = cmpTopkLengthAddr;

    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmax_lse);
    oritopkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)ori_sparse_indices);
    cmptopkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmp_sparse_indices);
    if (constInfo.isSink) {
        sinkGm.SetGlobalBuffer((__gm__ float *)sink);
        dsinkGm.SetGlobalBuffer((__gm__ float *)dsink);
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->baseParams.dSinkWorkSpaceOffset / sizeof(float)); // todo offset
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitUbBuffer(FagConstInfo &constInfo)
{
    /**
     * UB划分，buffer大小分配
     * attenMaskOrYInQue: for y and attenMask
     * pseOrDyInQue: for dy and pse
     * dSOutQue: for dq dk left ub matrix
     * pOutQue: for dv left ub matrix
     * softmaxGradResBuf: for softmax_grad result
     * dropMaskBuf: for dropMask
     * LseQue: forlse double buffer
     **/
    pipe->InitBuffer(attenMaskOrYInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(pseOrDyInQue, 1, VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE));

    pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
    pipe->InitBuffer(softmaxSinkResBuf[0], 64 * sizeof(float));
    pipe->InitBuffer(softmaxSinkResBuf[1], 64 * sizeof(float));
    pipe->InitBuffer(LseQue[0], 1, 64 * sizeof(float));
    pipe->InitBuffer(LseQue[1], 1, 64 * sizeof(float));
    pipe->InitBuffer(sinkQue, 1, 64 * sizeof(float));
    pipe->InitBuffer(dSinkQue, 1, 64 * sizeof(float));

    pipe->InitBuffer(dSOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE + DETER_EXCEED_USE_SIZE);
    pipe->InitBuffer(pOutQue, 1, VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
    pipe->InitBuffer(softmaxL1OutQue, 1, 128 * sizeof(float));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &dOriKWorkSpaceGm,
                                                                   const GlobalTensor<CALC_TYPE> &dCmpKWorkSpaceGm,
                                                                   FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t UB_ROW_SIZE = 8;
    int64_t usedCoreNum = constInfo.usedCoreNum;
    int64_t coreNum = GetBlockNum();
    int64_t totalVec = coreNum * 2;
    int64_t usedBS1Index = -1;
    int64_t bS1Index = -1;
    int64_t bIdx;
    int64_t s1Idx;
    int64_t preBIdx = -1;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int64_t accumS3Len = 0;
    int64_t actualSeqLensQ = 0;
    int64_t actualSeqLensOriK = 0;
    int64_t actualSeqLensCmpK = 0;
    int64_t actualOriSelectedBlockCount = 0;
    int64_t actualCmpSelectedBlockCount = 0;
    int64_t winStart = 0;
    GlobalTensor<CALC_TYPE> dkWorkSpaceGm;
    GlobalTensor<int32_t> topkIndicesGm;
    int64_t currentSelectedBlockCount;
    if (runInfo.isOriKV) {
        dkWorkSpaceGm = dOriKWorkSpaceGm;
        topkIndicesGm = oritopkIndicesGm;
        currentSelectedBlockCount = constInfo.oriSelectedBlockCount;
    } else {
        dkWorkSpaceGm = dCmpKWorkSpaceGm;
        topkIndicesGm = cmptopkIndicesGm;
        currentSelectedBlockCount = constInfo.cmpSelectedBlockCount;
    }
    LocalTensor<CALC_TYPE> dkInTensor = dSOutQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> dvInTensor = pOutQue.AllocTensor<CALC_TYPE>();
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    SetAtomicAdd<CALC_TYPE>();
    for (int64_t idx = 0; idx < usedCoreNum; idx++) {
        usedBS1Index = runInfo.sTaskId * usedCoreNum + idx;
        if (!GetBS1Index(usedBS1Index, bIdx, s1Idx, bS1Index, accumS1Len, accumS2Len, accumS3Len, actualSeqLensQ,
                         actualSeqLensOriK, actualSeqLensCmpK, actualOriSelectedBlockCount, actualCmpSelectedBlockCount,
                         winStart, runInfo.sTaskId, constInfo)) {
            continue;
        }

        // 对应s2real的平均和取模值，做均分操作
        int64_t actualSelectedBlockCount = runInfo.isOriKV ? actualOriSelectedBlockCount : actualCmpSelectedBlockCount;
        if (runInfo.isOriKV) {
            actualSelectedBlockCount = runInfo.kvSeqOffset + PROCESS_KV_SIZE < actualOriSelectedBlockCount ?
                                           PROCESS_KV_SIZE :
                                           actualOriSelectedBlockCount - runInfo.kvSeqOffset;
        } else {
            actualSelectedBlockCount = runInfo.kvSeqOffset + PROCESS_KV_SIZE < actualCmpSelectedBlockCount ?
                                           PROCESS_KV_SIZE :
                                           actualCmpSelectedBlockCount - runInfo.kvSeqOffset;
        }
        actualSelectedBlockCount = Max(actualSelectedBlockCount, 0);
        int64_t remainder = actualSelectedBlockCount % totalVec;
        int64_t avgSize = actualSelectedBlockCount / totalVec;
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

        int64_t maxLoops = CeilDiv(currentCoreKSize, UB_ROW_SIZE);
        int64_t tailRows = currentCoreKSize - (maxLoops - 1) * UB_ROW_SIZE;
        int64_t currentMm4SrcOffset = runInfo.deterTaskIdMod2 * PROCESS_KV_SIZE * HEAD_DIM_ALIGN * coreNum +
                                      idx * PROCESS_KV_SIZE * HEAD_DIM_ALIGN +
                                      s2SrcOffset * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
        int64_t currentMm5SrcOffset =
            runInfo.deterTaskIdMod2 * PROCESS_KV_SIZE * constInfo.commonConstInfo.dSizeV * coreNum +
            idx * PROCESS_KV_SIZE * constInfo.commonConstInfo.dSizeV +
            s2SrcOffset * constInfo.selectedBlockSize * constInfo.commonConstInfo.dSizeV;
        int64_t keyOffsetWithRope = accumS2Len * constInfo.n2Size * constInfo.dTotalSize;
        if (!runInfo.isOriKV) {
            keyOffsetWithRope = accumS3Len * constInfo.n2Size * constInfo.dTotalSize;
        }

        int64_t kvOffset = runInfo.isSparse ? keyOffsetWithRope :
                                              keyOffsetWithRope + s2SrcOffset * constInfo.dTotalSize +
                                                  runInfo.kvSeqOffset * constInfo.dTotalSize;
        if (runInfo.isOriKV && !runInfo.isSparse) {
            kvOffset += tilingData->baseParams.oriSparseMode == 4 ? winStart * constInfo.dTotalSize : 0;
        }
        GlobalTensor<float> dkOutGm = dkWorkSpaceGm[kvOffset];
        // main loop
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        uint64_t gmOffset =
            bS1Index * (constInfo.n2Size * currentSelectedBlockCount) + s2SrcOffset + runInfo.kvSeqOffset;
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
            if (runInfo.isSparse) {
                for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                    int s2Idx = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
                    if (s2Idx >= 0) {
                        Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                            dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
                        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                        DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                    }
                }
            } else {
                Add(dkInTensor, dkInTensor, dvInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkOutGm[loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
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
        if (runInfo.isSparse) {
            for (int64_t row = 0; row < tailRows; row++) {
                int32_t s2Idx = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
                if (s2Idx >= 0) {
                    Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                        dvInTensor[row * constInfo.commonConstInfo.dSizeV], constInfo.commonConstInfo.dSizeV);
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                }
            }
        } else {
            Add(dkInTensor, dkInTensor, dvInTensor, tailRows * HEAD_DIM_ALIGN);
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopy(dkOutGm[(maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, tailRows * HEAD_DIM_ALIGN);
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
__aicore__ inline bool FAGBlockVec<TEMPLATE_ARGS>::GetBS1Index(
    int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx, int64_t &bS1Index, int64_t &accumS1Len, int64_t &accumS2Len,
    int64_t &accumS3Len, int64_t &actualSeqLensQ, int64_t &actualSeqOriLensK, int64_t &actualSeqCmpLensK,
    int64_t &actualOriSelectedBlockCount, int64_t &actualCmpSelectedBlockCount, int64_t &winStart, int64_t taskId,
    FagConstInfo &constInfo)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;
    int64_t t3Offset = 0;
    int64_t bIndex = constInfo.hasUsedSeqQ ? -1 : 0;
    int64_t s1Index = 0;
    int64_t nextUsedT1Offset = 0;
    int64_t usedT1Offset = 0;
    int64_t curT1 = 0;
    int64_t curT2 = 0;
    int64_t curS1 = constInfo.commonConstInfo.s1Size;
    int64_t curS2 = constInfo.commonConstInfo.s2Size;
    int64_t curS3 = constInfo.s3Size;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;
    int64_t usedS3 = 0;
    int64_t oriWinStart = 0;
    int64_t oriWinEnd = 0;
    int64_t oriTopkLength = 0;
    int64_t cmpTopkLength = 0;

    if (usedT1Index >= constInfo.totalNum) {
        return false;
    }
    if (!constInfo.hasUsedSeqQ) {
        if constexpr (IS_TND) {
            curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            while (usedT1Index >= curT1) {
                curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[++bIndex];
            }
            bIndex = bIndex - 1;
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            s1Index = usedT1Index - t1Offset;
        } else {
            bIndex = usedT1Index / constInfo.commonConstInfo.s1Size;
            s1Index = usedT1Index % constInfo.commonConstInfo.s1Size;
            t1Offset = usedT1Index - s1Index;
        }
        bS1Index = usedT1Index;
    } else {
        while (usedT1Index >= nextUsedT1Offset) {
            usedT1Offset = nextUsedT1Offset;
            nextUsedT1Offset = ((__gm__ int32_t *)usedSeqQlenAddr)[++bIndex] + usedT1Offset;
        }
        s1Index = usedT1Index - usedT1Offset;
        if constexpr (IS_TND) {
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
        } else {
            t1Offset = bIndex * constInfo.commonConstInfo.s1Size;
        }
        bS1Index = t1Offset + s1Index;
    }
    if constexpr (IS_TND) {
        if (unlikely(bIndex == 0)) {
            t2Offset = 0;
            t3Offset = 0;
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex + 1];
            if constexpr (IsOriKVExist) {
                curS2 = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex + 1];
            }
            if constexpr (IsCmpKVExist) {
                curS3 = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex + 1];
            }
        } else {
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex + 1] - ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            if constexpr (IsOriKVExist) {
                curS2 = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex + 1] -
                        ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex];
                t2Offset = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex];
            }
            if constexpr (IsCmpKVExist) {
                curS3 = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex + 1] -
                        ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex];
                t3Offset = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex];
            }
        }
    } else {
        t2Offset = bIndex * curS2;
        t3Offset = bIndex * curS3;
    }

    if (constInfo.hasUsedSeqQ) {
        usedS1 = ((__gm__ int32_t *)usedSeqQlenAddr)[bIndex];
    } else {
        usedS1 = curS1;
    }
    if (constInfo.hasUsedSeqOriKV) {
        usedS2 = ((__gm__ int32_t *)usedOriSeqKvlenAddr)[bIndex];
    } else {
        usedS2 = curS2;
    }
    if (constInfo.hasUsedSeqCmpKV) {
        usedS3 = ((__gm__ int32_t *)usedCmpSeqKvlenAddr)[bIndex];
    } else {
        usedS3 = curS3;
    }

    if (tilingData->baseParams.oriSparseMode == 4) {
        oriWinStart = Max(s1Index + usedS2 - usedS1 - tilingData->baseParams.oriWinLeft, 0);
        oriWinEnd = Min(s1Index + usedS2 - usedS1 + tilingData->baseParams.oriWinRight + 1, usedS2);
        usedS2 = Max(oriWinEnd - oriWinStart, 0);
    } else if (tilingData->baseParams.oriSparseMode == 3) {
        usedS2 = Max(usedS2 - usedS1 + s1Index + 1, 0);
    }

    if (tilingData->baseParams.cmpSparseMode == 3 && constInfo.cmpRatio == 1) {
        usedS3 = Max(usedS3 - usedS1 + s1Index + 1, 0);
    } else if (tilingData->baseParams.cmpSparseMode == 3 && constInfo.cmpRatio != 1) {
        int64_t oriUsedS3 = usedS3 * constInfo.cmpRatio + ((__gm__ int32_t *)cmpResidualKAddr)[bIndex];
        usedS3 = Max((oriUsedS3 - usedS1 + s1Index + 1) / constInfo.cmpRatio, 0);
    }

    if (constInfo.hasOriTopK) {
        oriTopkLength = ((__gm__ int32_t *)oriTopkLengthAddr)[bS1Index];
    } else if (IsOriKVSparse) {
        oriTopkLength = constInfo.oriSelectedBlockCount;
    } else {
        oriTopkLength = curS2;
    }

    if (constInfo.hasCmpTopK) {
        cmpTopkLength = ((__gm__ int32_t *)cmpTopkLengthAddr)[bS1Index];
    } else if (IsCmpKVSparse) {
        cmpTopkLength = constInfo.cmpSelectedBlockCount;
    } else {
        cmpTopkLength = curS3;
    }

    actualOriSelectedBlockCount = Min(Min(oriTopkLength, curS2), usedS2);
    actualCmpSelectedBlockCount = Min(Min(cmpTopkLength, curS3), usedS3);

    bIdx = bIndex;
    s1Idx = s1Index;
    actualSeqLensQ = usedS1;
    actualSeqOriLensK = usedS2;
    actualSeqCmpLensK = usedS3;
    accumS1Len = t1Offset;
    accumS2Len = t2Offset;
    accumS3Len = t3Offset;
    winStart = oriWinStart;
    return true;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::GetRunInfo(int64_t bIdx, int64_t s1Idx, FagRunInfo &runInfo,
                                                              FagConstInfo &constInfo, int64_t accumS2Len,
                                                              int32_t actualSeqLensQ, int32_t actualSeqLensOriK,
                                                              int32_t actualSeqLensCmpK)
{}

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

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t FAGBlockVec<TEMPLATE_ARGS>::GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (!runInfo.isSparse) {
        return 0;
    }

    int32_t s2RealSize = 0;
    bool multiCopy = true;
    outParamK.blockCount = 2;
    uint64_t indicesGmOffset = runInfo.sparseIndicesOffset;
    uint32_t mergePingPong = 0;
    AscendC::TEventID mte2WaitMte3EventId;
    AscendC::TEventID mte3WaitMte2EventId;
    AscendC::TEventID mte2WaitMte3Ping = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    AscendC::TEventID mte2WaitMte3Pong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    AscendC::TEventID mte3WaitMte2Ping = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
    AscendC::TEventID mte3WaitMte2Pong = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Ping);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

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

    GlobalTensor<int32_t> curTopkIndicesGm = runInfo.isOriKV ? oritopkIndicesGm : cmptopkIndicesGm;
    GlobalTensor<INPUT_TYPE> keyGm = runInfo.isOriKV ? oriKVGm : cmpKVGm;
    for (i = curBlk; i < curBlk + curActualSelCntOffset / 2 * 2; i += 2) {
        int32_t indices1 = curTopkIndicesGm.GetValue(indicesGmOffset + i);
        int32_t indices2 = curTopkIndicesGm.GetValue(indicesGmOffset + i + 1);

        if (unlikely(indices1 < 0)) {
            multiCopy = false;
            break;
        } else if (unlikely(indices2 < 0)) {
            s2RealSize += 1;
            multiCopy = false;
        } else {
            s2RealSize += 2;
        }

        int64_t keyOffset1 = indices1 * constInfo.selectedBlockSize;
        int64_t keyOffset2 = indices2 * constInfo.selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - constInfo.selectedBlockSize;
        intriParamsKey.blockCount = 2;

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
                        keyGm[runInfo.commonRunInfo.keyOffset +
                              keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                        intriParamsKey, padParams);
            if (multiCopy) {
                DataCopyPad(gatherTensor[constInfo.selectedBlockSize * constInfo.commonConstInfo.dSize],
                            keyGm[runInfo.commonRunInfo.keyOffset +
                                  keyOffset2 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                            intriParamsKey, padParams);
            }
        } else {
            DataCopyPad(gatherTensor,
                        keyGm[runInfo.commonRunInfo.keyOffset +
                              keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                        intriParamsKey, padParams);
        }

        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        // CopyOut
        DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        outWsOffset += 2 * constInfo.dTotalSize;
        mergePingPong = 1 - mergePingPong;
    }
    if (i < curActualSelCntEnd && multiCopy) {
        int64_t indices1 = curTopkIndicesGm.GetValue(indicesGmOffset + i);
        int64_t keyOffset1 = indices1 * constInfo.selectedBlockSize;
        if (indices1 >= 0) {
            s2RealSize += 1;
        }

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Ping : mte2WaitMte3Pong;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Ping : mte3WaitMte2Pong;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        // CopyIn
        intriParamsKey.blockCount = 1;
        LocalTensor<INPUT_TYPE> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        if (indices1 >= 0) {
            DataCopyPad(gatherTensor,
                        keyGm[runInfo.commonRunInfo.keyOffset +
                              keyOffset1 * constInfo.n2Size * constInfo.commonConstInfo.dSize],
                        intriParamsKey, padParams);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
            outParamK.blockCount = 1;
            // CopyOut
            DataCopyPad(selectedKWorkSpaceGm[runInfo.kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        mergePingPong = 1 - mergePingPong;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Ping);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    dSOutQue.FreeTensor(gatherTensorPing);
    pOutQue.FreeTensor(gatherTensorPong);

    return s2RealSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0 || runInfo.commonRunInfo.s2RealSize == 0) {
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
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyLse(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    CopyInLse<float, VECTOR_BASEM>(constInfo, runInfo, LseQue[runInfo.sTaskIdMod2], softmaxLseGm);
    LocalTensor<float> lseTensor = LseQue[runInfo.sTaskIdMod2].AllocTensor<float>();
    LseQue[runInfo.sTaskIdMod2].FreeTensor(lseTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopySink(FagConstInfo &constInfo)
{
    if (constInfo.halfGRealSize == 0) {
        return;
    }
    CopyInSink<float, VECTOR_BASEM>(constInfo, sinkQue, sinkGm);
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DQ>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                             LocalTensor<INPUT_TYPE> &dstTensor,
                                                             LocalTensor<INPUT_TYPE> &srcTensor)
{
    if (runInfo.halfGRealSize == 0 || runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    uint32_t scmOffset = vSubBlockIdx == 0 ? 0 : runInfo.firstHalfGRealSize * FRACTAL_NZ_C0_SIZE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;
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
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    LocalTensor<CALC_TYPE> softmaxL1ResTensor = softmaxL1OutQue.AllocTensor<CALC_TYPE>();
    float pScalar = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.commonConstInfo.gSize));
    if (runInfo.isSparse) {
        CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN, true>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, LseQue[runInfo.sTaskIdMod2], attenMaskOrYInQue,
            pseOrDyInQue, mm2ResTensor, mm2ResTensor, softmaxL1ResTensor, pseSlope, pScalar);
    } else {
        CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, false, false, VECTOR_BASEM, VECTOR_BASEN, false>(
            constInfo, runInfo, *pseInfoPtr, *attenMaskInfoPtr, LseQue[runInfo.sTaskIdMod2], attenMaskOrYInQue,
            pseOrDyInQue, mm2ResTensor, mm2ResTensor, softmaxL1ResTensor, pseSlope, pScalar);
    }

    if (runInfo.halfGRealSize != 0) {
        softmaxL1OutQue.EnQue(softmaxL1ResTensor);
        softmaxL1OutQue.DeQue<CALC_TYPE>();

        DataCopyExtParams dataCopyGmParams;
        dataCopyGmParams.blockCount = 1;
        dataCopyGmParams.blockLen = runInfo.commonRunInfo.s2RealSize * sizeof(CALC_TYPE);
        dataCopyGmParams.srcStride = 0;
        dataCopyGmParams.dstStride = 0;
        SetAtomicAdd<CALC_TYPE>();
        if (runInfo.isOriKV && runInfo.isSparse) {
            DataCopyPad(oriSoftmaxL1Gm[runInfo.t1Index * runInfo.selectedBlockCount + runInfo.blkCntOffset],
                        softmaxL1ResTensor, dataCopyGmParams);
        } else if (runInfo.isSparse) {
            DataCopyPad(cmpSoftmaxL1Gm[runInfo.t1Index * runInfo.selectedBlockCount + runInfo.blkCntOffset],
                        softmaxL1ResTensor, dataCopyGmParams);
        }
        SetAtomicNone();
    }
    softmaxL1OutQue.FreeTensor(softmaxL1ResTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3(Buffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                                               LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                               LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////

    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>();
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
    if (runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
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
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessSoftmaxSink(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF: softmaxSink
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> lseTensor = LseQue[runInfo.sTaskIdMod2].AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> sinkTensor = sinkQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> softmaxSinkTensor = softmaxSinkResBuf[runInfo.sTaskIdMod2].Get<CALC_TYPE>();
    LseQue[runInfo.sTaskIdMod2].EnQue(lseTensor);
    LseQue[runInfo.sTaskIdMod2].DeQue<CALC_TYPE>();
    sinkQue.EnQue(sinkTensor);
    sinkQue.DeQue<CALC_TYPE>();
    CalculateSoftMaxSink<CALC_TYPE>(softmaxSinkTensor, lseTensor, sinkTensor, runInfo.halfGRealSize);
    LseQue[runInfo.sTaskIdMod2].FreeTensor(lseTensor);
    sinkQue.FreeTensor(sinkTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                  LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                  FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VFSink: Sink
    ///////////////////////////////////////////////////////////////
    if (runInfo.halfGRealSize == 0 || runInfo.commonRunInfo.s2RealSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> dsinkOutTensor = dSinkQue.AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> softmaxSinkTensor = softmaxSinkResBuf[runInfo.sTaskIdMod2].Get<CALC_TYPE>();

    CalculateDsink<CALC_TYPE>(dsinkOutTensor, mm2ResTensor, mm1ResTensor, softmaxSinkTensor, runInfo.actualSelCntOffset,
                              runInfo.halfGRealSize);

    dSinkQue.EnQue(dsinkOutTensor);
    dSinkQue.DeQue<CALC_TYPE>();
    SetAtomicAdd<CALC_TYPE>();
    uint64_t dsinkWorkSpaceOffset =
        cBlockIdx * constInfo.commonConstInfo.gSize + vSubBlockIdx * runInfo.firstHalfGRealSize;
    DataCopyPad(dsinkWorkSpaceGm[dsinkWorkSpaceOffset], dsinkOutTensor,
                {1, static_cast<uint32_t>(runInfo.halfGRealSize * sizeof(CALC_TYPE)), 0, 0, 0});
    SetAtomicNone();
    dSinkQue.FreeTensor(dsinkOutTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitCubeVecSharedParams(FagRunInfo &runInfo, int32_t s2RealSize) {}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_SPARSE>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                                              const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm,
                                                              LocalTensor<CALC_TYPE> &dkInTensor,
                                                              LocalTensor<CALC_TYPE> &dvInTensor,
                                                              FagConstInfo &constInfo, FagRunInfo &runInfo)
{
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

    int64_t kvOffset = IS_SPARSE ? runInfo.commonRunInfo.keyOffset :
                                   runInfo.kSelectedWsAddr + vSubBlockIdx * firstCoreKSize * HEAD_DIM_ALIGN;
    GlobalTensor<float> dkOutGm = dkWorkSpaceGm[kvOffset];
    int64_t currentMm4SrcOffset =
        runInfo.mm4ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    int64_t currentMm5SrcOffset =
        runInfo.mm5ResWsAddr + vSubBlockIdx * firstCoreKSize * constInfo.selectedBlockSize * HEAD_DIM_ALIGN;
    // 1 - main loop
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    uint64_t gmOffset = runInfo.sparseIndicesOffset + vSubBlockIdx * firstCoreKSize;
    GlobalTensor<int32_t> topkIndicesGm = runInfo.isOriKV ? oritopkIndicesGm : cmptopkIndicesGm;
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        DataCopy(dkInTensor, mm4ResWorkSpaceGm[currentMm4SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                 UB_ROW_SIZE * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, UB_ROW_SIZE * HEAD_DIM_ALIGN);
        DataCopy(dvInTensor, mm5ResWorkSpaceGm[currentMm5SrcOffset + loop * UB_ROW_SIZE * HEAD_DIM_ALIGN],
                 UB_ROW_SIZE * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        if constexpr (IS_SPARSE) {
            for (int64_t row = 0; row < UB_ROW_SIZE; row++) {
                int32_t s2Idx = topkIndicesGm[gmOffset + loop * UB_ROW_SIZE].GetValue(row);
                if (s2Idx >= 0) {
                    Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                        dvInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                }
            }
        } else {
            Add(dkInTensor, dkInTensor, dvInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            DataCopy(dkOutGm[loop * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, UB_ROW_SIZE * HEAD_DIM_ALIGN);
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
    DataCopy(dvInTensor, mm5ResWorkSpaceGm[currentMm5SrcOffset + (maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN],
             tailRows * HEAD_DIM_ALIGN);
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    if constexpr (IS_SPARSE) {
        for (int64_t row = 0; row < tailRows; row++) {
            int32_t s2Idx = topkIndicesGm[gmOffset + (maxLoops - 1) * UB_ROW_SIZE].GetValue(row);
            if (s2Idx >= 0) {
                Add(dkInTensor[row * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN],
                    dvInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN], dkInTensor[row * HEAD_DIM_ALIGN], HEAD_DIM_ALIGN);
            }
        }
    } else {
        Add(dkInTensor, dkInTensor, dvInTensor, tailRows * HEAD_DIM_ALIGN);
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopy(dkOutGm[(maxLoops - 1) * UB_ROW_SIZE * HEAD_DIM_ALIGN], dkInTensor, tailRows * HEAD_DIM_ALIGN);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    SetAtomicNone();
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void InitUbBuffer(FagConstInfo &constInfo) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out,
                                            GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices, GM_ADDR softmax_lse,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1,
                                            GM_ADDR dq, GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR actualSeqQlen,
                                            GM_ADDR actualSeqOriKvlen, GM_ADDR actualSeqCmpKvlen,
                                            GM_ADDR usedSeqQlenAddr, GM_ADDR usedOriSeqKvlenAddr,
                                            GM_ADDR usedCmpSeqKvlenAddr, GM_ADDR cmpResidualKAddr,
                                            GM_ADDR oriTopkLengthAddr, GM_ADDR cmpTopkLengthAddr, GM_ADDR workspace,
                                            FagConstInfo &constInfo) {};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, SFagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo &constInfo,
                                             AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {};
    __aicore__ inline int32_t GatherKV(const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo)
    {
        return 0;
    };
    template <const bool IS_SPARSE = true>
    __aicore__ inline void ScatterAdd(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                      const GlobalTensor<CALC_TYPE> &dkWorkSpaceGm, LocalTensor<CALC_TYPE> &dkInTensor,
                                      LocalTensor<CALC_TYPE> &dvInTensor, FagConstInfo &constInfo,
                                      FagRunInfo &runInfo) {};
    __aicore__ inline void ScatterAddDeter(const GlobalTensor<CALC_TYPE> &mm4ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &mm5ResWorkSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dkWorOriKSpaceGm,
                                           const GlobalTensor<CALC_TYPE> &dkWorCmpKSpaceGm, FagConstInfo &constInfo,
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
    __aicore__ inline void CopyLse(FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void CopySink(FagConstInfo &constInfo) {};
    __aicore__ inline void ProcessSoftmaxSink(FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                          FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void InitCubeVecSharedParams(FagRunInfo &runInfo, int32_t s2RealSize) {};
};

} // namespace SfagBaseApi

#endif
