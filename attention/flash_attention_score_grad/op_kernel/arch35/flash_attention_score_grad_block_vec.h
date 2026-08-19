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
 * \file flash_attention_score_grad_block_vec.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_H

#include "flash_attention_score_grad_common.h"
#include "cube_api/mutex_buffer.h"
#include "vector_api/cast_softmax_grad.h"
#include "vector_api/dropout.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax.h"
#include "vector_api/vf_broadcast_sub_mul.h"
#include "vector_api/vf_cast_transdata_deconflict.h"
#include "vector_api/vf_cal_sink.h"
#include "vector_api/vf_ds_abs_reduce_max.h"
namespace FagBaseApi {
constexpr uint32_t BIT_MASK_NUM = 8;
constexpr uint32_t BIT32_ALIGN_NUM = 8;

TEMPLATES_DEF
class FAGBlockVec {
private:
    template <uint8_t MM_IDX>
    __aicore__ inline void DqkvMulsAndCastFromUB(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                 LocalTensor<CALC_TYPE> &inputTensor,
                                                 TQue<QuePosition::VECOUT, 1> &outQue);
    template <uint8_t MM_IDX>
    __aicore__ inline void DqkvMulsAndCastFromGM(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                 GlobalTensor<CALC_TYPE> &inputTensor,
                                                 TQue<QuePosition::VECIN, 1> &inQue,
                                                 TQue<QuePosition::VECOUT, 1> &outQue);
    template <uint8_t RES_IDX>
    __aicore__ inline uint16_t FindAvailableRows(FagConstInfo &constInfo, uint64_t writeStartAddr,
                                                 uint64_t writeLastAddr, uint16_t blockCount, uint32_t blockLen,
                                                 uint32_t dstStride);

public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, AttenMaskInfo &attenMaskInfo,
                                             PseInfo &pseInfo, DropMaskInfo &dropInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace);
    __aicore__ inline void SetOldDeterFp32Param(FagConstInfo &constInfo);
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    __aicore__ inline void CopyDpToTmpBuf(LocalTensor<CALC_TYPE> &mm1ResTensor);
    __aicore__ inline void ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                          FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline void ProcessVec3(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter = false);
    __aicore__ inline void ProcessVec4(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);

    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCast(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);

    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId);
    template <const bool IS_DQ = false>
    __aicore__ inline void CopyUB2L1(FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                     LocalTensor<INPUT_TYPE> &srcTensor);
    __aicore__ inline void CopyUB2L1Deter(FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                          LocalTensor<INPUT_TYPE> &srcTensor);
    __aicore__ inline void CopyUBToL1Vec3(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                          LocalTensor<INPUT_TYPE> &vecOutBuffer, FagRunInfo &runInfo,
                                          bool isDqNeedDeter);
    template <const bool IS_DK>
    __aicore__ inline void ProcessPostDeter(FagConstInfo &constInfo, GlobalTensor<float> dkvWorkSpaceTensor,
                                            GlobalTensor<INPUT_TYPE> &dkvGmTensor, int64_t specialHalfS2RealSize,
                                            int64_t specialFirstHalfS2RealSize, uint64_t dAlign16, uint64_t dvAlign16,
                                            int64_t specialDkGmOffset, int64_t specialDvGmOffset);
    __aicore__ inline void DeterCompute(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                        FagConstInfo &constInfo, bool dqIsNeedDeter[2], bool dkDvIsNeedDeter[2],
                                        bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                        int16_t *deterPpFlag);
    __aicore__ inline void DeterComputeDq(FagConstInfo &constInfo, bool isLastSort, int64_t computeBlockIdx,
                                          int64_t remainLoopNum, int16_t *deterPpFlag);
    __aicore__ inline void DeterComputeDkv(LocalTensor<CALC_TYPE> &dkDeterBuf, LocalTensor<CALC_TYPE> &dvDeterBuf,
                                           FagConstInfo &constInfo, bool isLastSort, int64_t computeBlockIdx,
                                           int64_t remainLoopNum, int16_t *deterPpFlag);
    __aicore__ inline void DeterComputeDqkv(LocalTensor<CALC_TYPE> &dkDeterBuf, LocalTensor<CALC_TYPE> &dvDeterBuf,
                                            FagConstInfo &constInfo, bool isLastSort, int64_t computeBlockIdx,
                                            int64_t remainLoopNum, int16_t *deterPpFlag);
    __aicore__ inline void WriteDataToDkv(LocalTensor<CALC_TYPE> &dkDeterBuf, LocalTensor<CALC_TYPE> &dvDeterBuf,
                                          FagConstInfo &constInfo, uint32_t dkSrcOfs, uint32_t dvSrcOfs);
    __aicore__ inline void WriteOffsetToGM(FagConstInfo &constInfo, int64_t queryGmOffset, int64_t keyGmOffset,
                                           int64_t valueGmOffset, int64_t computeBlockIdx, int64_t remainLoopNum);

    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 12 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D768 = 96 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D768 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;
    constexpr static uint32_t INT64_BLOCK_NUM = 32 / sizeof(int64_t);
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static int64_t OUTINDEX = -1;
    constexpr static uint32_t DETER_DKV_UB_SIZE = VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE);
    constexpr static uint32_t OFFSET_NUM = 3;
    constexpr static uint32_t INT64_BYTES = 4;

    // vector gm addr
    GlobalTensor<INPUT_TYPE> valueGm;
    GlobalTensor<OUTDTYPE> yGm, pseGm, dyGm;
    GlobalTensor<uint8_t> dropMaskGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm, sfmgWorkSpaceGm;
    GM_ADDR pseSlope;
    GlobalTensor<uint8_t> dropMaskWorkspaceGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<OUTDTYPE> dqRopeGm, dkRopeGm;
    GlobalTensor<float> sinkGm, dsinkGm, dsinkWorkSpaceGm;

    // ub buffer
    TQue<QuePosition::VECIN, 1> attenMaskOrYInQue;
    TQue<QuePosition::VECIN, 1> pseOrDyInQue;
    TQue<QuePosition::VECOUT, 1> dSOutQue;
    TQue<QuePosition::VECOUT, 1> pOutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue[2];
    TBuf<> softmaxGradResBuf;
    TBuf<> dropMaskBuf;
    TBuf<> dropmaskIndexVecBuf;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TQue<QuePosition::VECOUT, 1> dsinkResOutQue[2];

    TPipe *pipe;
    FagTilingType tilingData;

    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;
    event_t eventIDSToV = static_cast<event_t>(0);

    // optional info
    AttenMaskInfo attenMaskInfoPtr;
    PseInfo pseInfoPtr;
    DropMaskInfo dropInfoPtr;

    // old deter info
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool, std::nullptr_t>::type isFirstDeter;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[36], std::nullptr_t>::type dqOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[36], std::nullptr_t>::type dkOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[36], std::nullptr_t>::type dvOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), GlobalTensor<float>, std::nullptr_t>::type deterGm;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), GlobalTensor<int64_t>, std::nullptr_t>::type
        deterOffsetGm;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), GlobalTensor<float>, std::nullptr_t>::type dqWorkSpaceGm;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), GlobalTensor<float>, std::nullptr_t>::type dkWorkSpaceGm;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), GlobalTensor<float>, std::nullptr_t>::type dvWorkSpaceGm;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT, uint64_t, std::nullptr_t>::type
        dqGmBaseAddr;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT, uint64_t, std::nullptr_t>::type
        dkGmBaseAddr;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT, uint64_t, std::nullptr_t>::type
        dvGmBaseAddr;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT, uint64_t, std::nullptr_t>::type
        dqGmLimitOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT, uint64_t, std::nullptr_t>::type
        dkGmLimitOffset;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT, uint64_t, std::nullptr_t>::type
        dvGmLimitOffset;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::SetOldDeterFp32Param(FagConstInfo &constInfo)
{
    dqGmBaseAddr = (uint64_t)(dqWorkSpaceGm.GetPhyAddr());
    dkGmBaseAddr = (uint64_t)(dkWorkSpaceGm.GetPhyAddr());
    dvGmBaseAddr = (uint64_t)(dvWorkSpaceGm.GetPhyAddr());
    uint64_t qSize = 0;
    uint64_t kSize = 0;
    uint64_t vSize = 0;
    if constexpr (IS_TND) {
        qSize = ((__gm__ int64_t *)constInfo.seqS1_addr)[constInfo.bSize - 1] * constInfo.commonConstInfo.n2GD;
        kSize = ((__gm__ int64_t *)constInfo.seqS2_addr)[constInfo.bSize - 1] * constInfo.commonConstInfo.n2D;
        vSize = ((__gm__ int64_t *)constInfo.seqS2_addr)[constInfo.bSize - 1] * constInfo.commonConstInfo.n2Dv;
    } else {
        qSize = constInfo.bSize * constInfo.commonConstInfo.n2GS1D;
        kSize = constInfo.bSize * constInfo.commonConstInfo.n2S2D;
        vSize = constInfo.bSize * constInfo.commonConstInfo.n2S2Dv;
    }
    uint64_t dqSizeByte = qSize * sizeof(CALC_TYPE);
    uint64_t dkSizeByte = kSize * sizeof(CALC_TYPE);
    uint64_t dvSizeByte = vSize * sizeof(CALC_TYPE);
    dqGmLimitOffset = dqGmBaseAddr + dqSizeByte;
    dkGmLimitOffset = dkGmBaseAddr + dkSizeByte;
    dvGmLimitOffset = dvGmBaseAddr + dvSizeByte;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::SetVecBlockParams(TPipe *pipe, FagTilingType tilingData,
                                                                     uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                     uint32_t vSubBlockIdx,
                                                                     AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo,
                                                                     DropMaskInfo &dropInfo)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
    attenMaskInfoPtr = attenMaskInfo;
    pseInfoPtr = pseInfo;
    dropInfoPtr = dropInfo;
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        isFirstDeter = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR softmaxMax,
    GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace)
{
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);
    pseGm.SetGlobalBuffer((__gm__ OUTDTYPE *)pseShift);
    pseFloatGm.SetGlobalBuffer((__gm__ float *)pseShift);
    dropMaskGm.SetGlobalBuffer((GM_ADDR)dropMask);
    attenMaskU8Gm.SetGlobalBuffer((GM_ADDR)attenMask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    pseSlope = pseShift;
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    if constexpr (IS_ROPE) {
        dqRopeGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dqRope);
        dkRopeGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dkRope);
    }
    if constexpr (IS_DROP) {
        if (tilingData->preTilingData.dropoutIsDivisibleBy8 == 0) {
            dropMaskWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace +
                                                tilingData->postTilingData.dropMaskGmOffset);
        }
    }
    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.enablePreSfmg)) {
        sfmgWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        tilingData->postTilingData.sfmgWorkSpaceOffset / sizeof(float));
    }
    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) {
        sinkGm.SetGlobalBuffer((__gm__ float *)sink);
        dsinkGm.SetGlobalBuffer((__gm__ float *)dsink);
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                         tilingData->postTilingData.dsinkWorkSpaceOffset / sizeof(float));
    }
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        deterGm.SetGlobalBuffer((__gm__ float *)workspace +
                                tilingData->baseDeterParam.deterGmOffset / sizeof(CALC_TYPE));
        deterOffsetGm.SetGlobalBuffer((__gm__ int64_t *)workspace +
                                      tilingData->baseDeterParam.deterWorkSpaceOffset / sizeof(int64_t));
        if constexpr (!IS_FP32_INPUT) {
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        } else {
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)dq);
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)dk);
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)dv);
        }
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
    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.enablePreSfmg)) {
        pipe->InitBuffer(maxSumQue[0], 1,
                         VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO + VECTOR_BASEM * sizeof(float));
        pipe->InitBuffer(maxSumQue[1], 1,
                         VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO + VECTOR_BASEM * sizeof(float));
    } else {
        pipe->InitBuffer(softmaxGradResBuf, VECTOR_BASEM * sizeof(CALC_TYPE));
        pipe->InitBuffer(maxSumQue[0], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
        pipe->InitBuffer(maxSumQue[1], 1, VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * NUM_TWO);
    }
    if constexpr (IS_DROP) {
        pipe->InitBuffer(dropMaskBuf, VECTOR_BASEM * VECTOR_BASEN * sizeof(uint8_t) / BIT_MASK_NUM); // 1k
        pipe->InitBuffer(dropmaskIndexVecBuf,
                         VECTOR_BASEM * VECTOR_BASEN / (NUM_TWO * BIT_MASK_NUM) * sizeof(int32_t)); // 2k
    }
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        pipe->InitBuffer(deterInOutQue, 1, DETER_DQ_UB_SIZE);
        pipe->InitBuffer(deterOffsetBuf, DETER_OFFSET_UB_SIZE); // 3k
    }
    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) {
        pipe->InitBuffer(dsinkResOutQue[0], 1, BIT32_ALIGN_NUM * sizeof(float));
        pipe->InitBuffer(dsinkResOutQue[1], 1, BIT32_ALIGN_NUM * sizeof(float));
    }

    if constexpr (!IS_FP32_INPUT) {
        pipe->InitBuffer(dSOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
        pipe->InitBuffer(pOutQue, 1, (VECTOR_BASEM + 1) * VECTOR_BASEN * sizeof(OUTDTYPE));
    } else {
        // input type fp32, exceed ub size so need to reuse dSOutQue
        pipe->InitBuffer(dSOutQue, 1,
                         VECTOR_BASEM * VECTOR_BASEN * sizeof(INPUT_TYPE) + VECTOR_BASEN * sizeof(INPUT_TYPE));
        pOutQue = dSOutQue;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF1: Cast + SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    LocalTensor<CALC_TYPE> softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    if constexpr (HEAD_DIM_ALIGN <= VECTOR_BASEN) {
        CopyInSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN, IS_D_NO_EQUAL>(
            constInfo, runInfo, 0, runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.halfS1RealSize,
            attenMaskOrYInQue, pseOrDyInQue, dyGm, yGm);
        CalculateCastSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN>(
            constInfo, runInfo.commonRunInfo.halfS1RealSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor);
    } else {
        uint32_t loopNum = Ceil<uint32_t>(runInfo.commonRunInfo.halfS1RealSize, constInfo.sfmgMaxLoopSize);
        uint32_t loopSize = Ceil<uint32_t>(runInfo.commonRunInfo.halfS1RealSize, loopNum);
        uint32_t tailLoopSize = runInfo.commonRunInfo.halfS1RealSize - (loopNum - 1) * loopSize;
        uint32_t curLoopSize = loopSize;
        for (int32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            if (loopIdx == loopNum - 1) {
                curLoopSize = tailLoopSize;
            }
            CopyInSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN, IS_D_NO_EQUAL>(
                constInfo, runInfo, loopIdx, curLoopSize, loopSize, attenMaskOrYInQue, pseOrDyInQue, dyGm, yGm);
            CalculateCastSoftmaxGrad<OUTDTYPE, CALC_TYPE, VECTOR_BASEM, HEAD_DIM_ALIGN>(
                constInfo, curLoopSize, attenMaskOrYInQue, pseOrDyInQue, softmaxGradResTensor[loopSize * loopIdx]);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                              int64_t taskId)
{
    if (unlikely(constInfo.enablePreSfmg)) {
        if constexpr (DETER_SPARSE_TYPE == NO_DETER || IS_DETER_NEW(DETER_SPARSE_TYPE)) {
            CopyInMaxSumWithSfmg<float, VECTOR_BASEM, IS_ROPE>(constInfo, runInfo, maxSumQue[taskId & 1], softmaxMaxGm,
                                                               softmaxSumGm, sfmgWorkSpaceGm);
        }
    } else {
        CopyInMaxSum<float, VECTOR_BASEM, IS_ROPE>(constInfo, runInfo, maxSumQue[taskId & 1], softmaxMaxGm,
                                                   softmaxSumGm);
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DQ>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagRunInfo &runInfo, LocalTensor<INPUT_TYPE> &dstTensor,
                                                             LocalTensor<INPUT_TYPE> &srcTensor)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    uint32_t scmOffset =
        vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
    dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE /
                                         INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    dataCopyParams.srcStride = (uint16_t)((VECTOR_BASEM + 1 - runInfo.commonRunInfo.halfS1RealSize) *
                                          FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
    uint32_t s1RealSizeAlignTo16 = AlignTo16(runInfo.commonRunInfo.s1RealSize);
    dataCopyParams.dstStride = (s1RealSizeAlignTo16 - runInfo.commonRunInfo.halfS1RealSize) *
                               FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1Deter(FagRunInfo &runInfo,
                                                                  LocalTensor<INPUT_TYPE> &dstTensor,
                                                                  LocalTensor<INPUT_TYPE> &srcTensor)
{
    uint32_t scmOffset =
        (vSubBlockIdx == 0 ? 0 : runInfo.commonRunInfo.firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE);
    DataCopyParams dataCopyParams;
    if (runInfo.commonRunInfo.halfS1RealSize != 0) {
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
        dataCopyParams.blockLen = (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE /
                                             INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.srcStride = (uint16_t)((VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize + 1) *
                                              FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.dstStride = (uint16_t)(CUBE_BASEM - runInfo.commonRunInfo.halfS1RealSize) *
                                   FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
        DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
    }
    if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
        // copy 补零的数据
        scmOffset = (vSubBlockIdx == 0 ?
                         runInfo.commonRunInfo.s1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE :
                         (VECTOR_BASEM + runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE);
        dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE;
        dataCopyParams.blockLen = (uint16_t)((VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) *
                                             FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.srcStride = (uint16_t)((runInfo.commonRunInfo.halfS1RealSize + 1) *
                                              FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE);
        dataCopyParams.dstStride = (uint16_t)(VECTOR_BASEM + runInfo.commonRunInfo.halfS1RealSize) *
                                   FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE / INPUT_BLOCK_NUM_FOR_INPUT_DTYPE;
        DataCopy(dstTensor[scmOffset],
                 srcTensor[runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE], dataCopyParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                               FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF2: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    CopyInAttenMask<IS_ATTEN_MASK, VECTOR_BASEM, VECTOR_BASEN>(constInfo, runInfo, attenMaskInfoPtr, attenMaskOrYInQue,
                                                               pseOrDyInQue, attenMaskU8Gm);
    CopyInPse<OUTDTYPE, CALC_TYPE, IS_PSE>(constInfo, runInfo, pseInfoPtr, pseOrDyInQue, pseGm);
    CalculatePseMulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, false, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD(DETER_SPARSE_TYPE),
                                     VECTOR_BASEM, VECTOR_BASEN>(
        constInfo, runInfo, pseInfoPtr, attenMaskInfoPtr, maxSumQue[runInfo.commonRunInfo.taskIdMod2],
        attenMaskOrYInQue, pseOrDyInQue, mm2ResTensor, mm2ResTensor, pseSlope);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyDpToTmpBuf(LocalTensor<CALC_TYPE> &mm1ResTensor)
{
    LocalTensor<CALC_TYPE> DpDropTensor = attenMaskOrYInQue.AllocTensor<CALC_TYPE>();
    DataCopy(DpDropTensor, mm1ResTensor, VECTOR_BASEM * VECTOR_BASEN);
    attenMaskOrYInQue.EnQue(DpDropTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                  LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                  FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    LocalTensor<CALC_TYPE> dsinkOutTensor = dsinkResOutQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<CALC_TYPE>();
    LocalTensor<CALC_TYPE> sumTensor = maxSumQue[runInfo.commonRunInfo.taskIdMod2].AllocTensor<CALC_TYPE>();
    float sinkScale = sinkGm.GetValue(runInfo.sinkN1Idx);
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    maxSumQue[runInfo.commonRunInfo.taskIdMod2].EnQue(sumTensor);
    maxSumQue[runInfo.commonRunInfo.taskIdMod2].DeQue<CALC_TYPE>();
    LocalTensor<CALC_TYPE> maxTensor = sumTensor[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)];

    if constexpr (IS_DROP) {
        LocalTensor<CALC_TYPE> DpDropTensor = attenMaskOrYInQue.DeQue<CALC_TYPE>();
        CalculateSink<CALC_TYPE, static_cast<uint16_t>(VECTOR_BASEN)>(
            dsinkOutTensor, mm2ResTensor, DpDropTensor, maxTensor, sumTensor, sinkScale,
            static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
            static_cast<uint16_t>(runInfo.commonRunInfo.s2RealSize));
        attenMaskOrYInQue.FreeTensor(DpDropTensor);
    } else {
        CalculateSink<CALC_TYPE, static_cast<uint16_t>(VECTOR_BASEN)>(
            dsinkOutTensor, mm2ResTensor, mm1ResTensor, maxTensor, sumTensor, sinkScale,
            static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
            static_cast<uint16_t>(runInfo.commonRunInfo.s2RealSize));
    }
    maxSumQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(sumTensor);
    dsinkResOutQue[runInfo.commonRunInfo.taskIdMod2].EnQue(dsinkOutTensor);
    dsinkResOutQue[runInfo.commonRunInfo.taskIdMod2].DeQue<CALC_TYPE>();
    DataCopyPad(dsinkWorkSpaceGm[runInfo.dsinkWorkSpaceOffset], dsinkOutTensor,
                {1, static_cast<uint32_t>(sizeof(CALC_TYPE)), 0, 0, 0});
    dsinkResOutQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(dsinkOutTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUBToL1Vec3(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<INPUT_TYPE> &vecOutBuffer,
    FagRunInfo &runInfo, bool isDqNeedDeter)
{
    dSOutQue.EnQue(vecOutBuffer);
    dSOutQue.DeQue<INPUT_TYPE>();

    // copy ds from ub to l1
    LocalTensor<INPUT_TYPE> dsL1Tensor = dstBuffer.GetTensor<INPUT_TYPE>();
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (isDqNeedDeter) {
            CopyUB2L1Deter(runInfo, dsL1Tensor, vecOutBuffer);
            dsL1Tensor.SetAddrWithOffset(dsL1Tensor, CUBE_BASEM * CUBE_BASEN);
            CopyUB2L1(runInfo, dsL1Tensor, vecOutBuffer);
        } else {
            CopyUB2L1<true>(runInfo, dsL1Tensor, vecOutBuffer);
        }
    } else {
        CopyUB2L1<true>(runInfo, dsL1Tensor, vecOutBuffer);
    }

    dSOutQue.FreeTensor(vecOutBuffer);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec3(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm1ResTensor,
    LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter)
{
    ///////////////////////////////////////////////////////////////
    // VF3: sub + mul
    // VF4: dq dk cast + nd2nz
    ///////////////////////////////////////////////////////////////
    if (dropInfoPtr.dropMaskOuter) {
        if (dropInfoPtr.boolMode) {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskWorkspaceGm, runInfo.commonRunInfo,
                                     constInfo.commonConstInfo, dropInfoPtr);
        } else {
            CopyInDropOuter<IS_DROP>(dropMaskBuf, attenMaskOrYInQue, dropMaskGm, runInfo.commonRunInfo,
                                     constInfo.commonConstInfo, dropInfoPtr);
        }
    } else {
        GenDropMask<IS_DROP>(dropMaskBuf, dropmaskIndexVecBuf, runInfo.commonRunInfo, constInfo.commonConstInfo,
                             dropInfoPtr);
    }
    CalculateDropout<CALC_TYPE, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfoPtr, mm1ResTensor, mm1ResTensor,
                                                       dropMaskBuf);
    if (unlikely(constInfo.isSink && IS_DROP)) {
        CopyDpToTmpBuf(mm1ResTensor);
    }

    LocalTensor<CALC_TYPE> softmaxGradResTensor;
    if (unlikely(constInfo.enablePreSfmg)) {
        softmaxGradResTensor = maxSumQue[runInfo.commonRunInfo.taskIdMod2]
                                   .template AllocTensor<CALC_TYPE>()[VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE /
                                                                      sizeof(CALC_TYPE) * NUM_TWO];
    } else {
        softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>();
    }

    LocalTensor<INPUT_TYPE> vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>();
    if (runInfo.commonRunInfo.s2RealSize > static_cast<uint32_t>(S2TemplateType::Aligned64)) {
        BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned128), 0, false>(
            mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize,
            runInfo.commonRunInfo.s2RealSize);
    } else {
        if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            if (constInfo.deterConstInfo.noNeedDeter) {
                BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64), 0, false>(
                    mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor,
                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
            } else { // 64~128的脏数据需要清零，避免后面的mm有脏数据参与计算
                BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64),
                                IS_DETER_OLD(DETER_SPARSE_TYPE), false>(
                    mm1ResTensor, mm1ResTensor, softmaxGradResTensor, mm2ResTensor,
                    runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
            }
        } else {
            BroadcastSubMul<CALC_TYPE, static_cast<uint16_t>(S2TemplateType::Aligned64),
                            IS_DETER_OLD(DETER_SPARSE_TYPE), false>(mm1ResTensor, mm1ResTensor, softmaxGradResTensor,
                                                                    mm2ResTensor, runInfo.commonRunInfo.halfS1RealSize,
                                                                    runInfo.commonRunInfo.s2RealSize);
        }
    }

    if (unlikely(constInfo.enablePreSfmg)) {
        maxSumQue[runInfo.commonRunInfo.taskIdMod2].FreeTensor(softmaxGradResTensor);
    }

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) { // 确定性计算尾块脏数据补零
        if (!constInfo.deterConstInfo.noNeedDeter) {
            if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
                Duplicate<CALC_TYPE>(mm1ResTensor[runInfo.commonRunInfo.halfS1RealSize * VECTOR_BASEN], 0,
                                     (VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) * VECTOR_BASEN);
            }
        }
    }

    // input type fp32, no post, mov muls here
    if constexpr (IS_FP32_INPUT) {
        Muls(mm1ResTensor, mm1ResTensor, constInfo.scaleValue, VECTOR_BASEM * VECTOR_BASEN);
    }

    LocalTensor<uint8_t> selrIndexesTensor;
    CastTransdataDeconflict<INPUT_TYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer, mm1ResTensor, selrIndexesTensor,
                                                                 VECTOR_BASEM);
    CopyUBToL1Vec3(dstBuffer, vecOutBuffer, runInfo, isDqNeedDeter);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec4(
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer, LocalTensor<CALC_TYPE> &mm2ResTensor,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    ///////////////////////////////////////////////////////////////
    // VF5: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<uint8_t> selrIndexesTensor;
    LocalTensor<OUTDTYPE> vecOutBuffer1 = pOutQue.AllocTensor<OUTDTYPE>();
    CalculateDropout<CALC_TYPE, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfoPtr, mm2ResTensor, mm2ResTensor,
                                                       dropMaskBuf);
    CastTransdataDeconflict<OUTDTYPE, CALC_TYPE, VECTOR_BASEN>(vecOutBuffer1, mm2ResTensor, selrIndexesTensor,
                                                               VECTOR_BASEM);
    pOutQue.EnQue(vecOutBuffer1);
    pOutQue.DeQue<OUTDTYPE>();

    // copy p from ub to l1
    LocalTensor<OUTDTYPE> pL1Tensor = dstBuffer.GetTensor<OUTDTYPE>();
    CopyUB2L1<false>(runInfo, pL1Tensor, vecOutBuffer1);

    pOutQue.FreeTensor(vecOutBuffer1);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessMulsAndCast(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if constexpr (IS_WRITE_UB) {
        DqkvMulsAndCastFromUB<MM_IDX>(constInfo, runInfo, inputTensor, dSOutQue);
    } else {
        DqkvMulsAndCastFromGM<MM_IDX>(constInfo, runInfo, inputTensor, attenMaskOrYInQue, dSOutQue);
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DqkvMulsAndCastFromUB(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                         LocalTensor<CALC_TYPE> &inputTensor,
                                                                         TQue<QuePosition::VECOUT, 1> &outQue)
{
    if ((MM_IDX == DQ_IDX && runInfo.commonRunInfo.halfS1RealSize == 0) ||
        (MM_IDX != DQ_IDX && runInfo.halfS2RealSize == 0)) {
        return;
    }
    uint64_t dSize = (MM_IDX == DV_IDX) ? constInfo.commonConstInfo.dSizeV : constInfo.commonConstInfo.dSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.blockCount = (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.halfS1RealSize : runInfo.halfS2RealSize;
    intriParamsOut.blockLen = dSize * sizeof(OUTDTYPE);
    intriParamsOut.srcStride = 0;
    GlobalTensor<OUTDTYPE> dqkvGmTensor = MM_IDX == DQ_IDX ? dqGm : (MM_IDX == DK_IDX ? dkGm : dvGm);
    uint32_t dataSize = intriParamsOut.blockCount * AlignTo16(dSize);
    uint64_t dqkvGmOffset =
        (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.queryOffset :
                             (MM_IDX == DK_IDX ? runInfo.commonRunInfo.keyOffset : runInfo.commonRunInfo.valueOffset);

    uint64_t halfSRealSize =
        (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.firstHalfS1RealSize : runInfo.firstHalfS2RealSize;
    if constexpr (IS_TND) {
        intriParamsOut.dstStride =
            static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
        dqkvGmOffset += vSubBlockIdx * dSize * constInfo.commonConstInfo.n2G * halfSRealSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParamsOut.dstStride = 0;
            dqkvGmOffset += vSubBlockIdx * halfSRealSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParamsOut.dstStride =
                static_cast<uint32_t>((constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
            dqkvGmOffset += vSubBlockIdx * halfSRealSize * constInfo.commonConstInfo.n2G * constInfo.bSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParamsOut.dstStride =
                static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE));
            dqkvGmOffset += vSubBlockIdx * halfSRealSize * constInfo.commonConstInfo.n2G * dSize;
        }
    }

    if constexpr (MM_IDX != DV_IDX) {
        Muls(inputTensor, inputTensor, constInfo.scaleValue, dataSize);
    }

    LocalTensor<OUTDTYPE> dqkvCastTensor = outQue.template AllocTensor<OUTDTYPE>();
    Cast(dqkvCastTensor, inputTensor, RoundMode::CAST_ROUND, dataSize);
    outQue.EnQue(dqkvCastTensor);
    outQue.template DeQue<OUTDTYPE>();

    DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
    outQue.FreeTensor(dqkvCastTensor);
}

TEMPLATES_DEF_NO_DEFAULT
template <uint8_t MM_IDX>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DqkvMulsAndCastFromGM(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                         GlobalTensor<CALC_TYPE> &inputTensor,
                                                                         TQue<QuePosition::VECIN, 1> &inQue,
                                                                         TQue<QuePosition::VECOUT, 1> &outQue)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    uint32_t dSize;
    if constexpr (MM_IDX == DV_IDX && IS_D_NO_EQUAL) {
        dSize = constInfo.commonConstInfo.dSizeV;
    } else {
        dSize = constInfo.commonConstInfo.dSize;
    }
    uint32_t curDAlign = AlignTo16(dSize);
    uint64_t halfSRealSize = 0;
    uint64_t firsthalfSRealSize = 0;
    if constexpr (SPLIT_AXIS == BN2) {
        halfSRealSize = (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.halfS1RealSize : runInfo.halfS2RealSize;
        firsthalfSRealSize =
            (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.firstHalfS1RealSize : runInfo.firstHalfS2RealSize;
    } else {
        halfSRealSize = runInfo.halfS2RealSize;
        firsthalfSRealSize = runInfo.firstHalfS2RealSize;
    }

    uint32_t maxLoopSize = VECTOR_BASEM * VECTOR_BASEN / curDAlign;
    uint32_t loopNum = Ceil<uint32_t>(halfSRealSize, maxLoopSize);
    if (loopNum == 0) {
        return;
    }

    uint32_t loopSize = Ceil<uint32_t>(halfSRealSize, loopNum);
    uint32_t tailLoopSize = halfSRealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.srcStride = 0;
    uint64_t dqkvGmOffset =
        (MM_IDX == DQ_IDX) ? runInfo.commonRunInfo.queryOffset :
                             (MM_IDX == DK_IDX ? runInfo.commonRunInfo.keyOffset : runInfo.commonRunInfo.valueOffset);
    uint64_t ropeGmOffset;
    GlobalTensor<OUTDTYPE> dqkRopeGmTensor;
    int64_t ropeDstStride;
    if constexpr (IS_ROPE) {
        dSize = constInfo.commonConstInfo.dSizeV;
        dqkvGmOffset = (MM_IDX == DQ_IDX) ?
                           runInfo.queryOffsetWithRopeForMm12 :
                           (MM_IDX == DK_IDX ? runInfo.keyOffsetWithRopeForMm12 : runInfo.commonRunInfo.valueOffset);
        ropeGmOffset = MM_IDX == DQ_IDX ? runInfo.commonRunInfo.qRopeOffset : runInfo.commonRunInfo.kRopeOffset;
        dqkRopeGmTensor = MM_IDX == DQ_IDX ? dqRopeGm : dkRopeGm;
        if constexpr (IS_TND) {
            ropeDstStride = (constInfo.commonConstInfo.n2G - 1) * constInfo.dRopeSize * sizeof(OUTDTYPE);
            ropeGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.n2GDr;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                ropeDstStride = 0;
                ropeGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.dRopeSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                ropeDstStride =
                    (constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * constInfo.dRopeSize * sizeof(OUTDTYPE);
                ropeGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.bN2GDr;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                ropeDstStride = (constInfo.commonConstInfo.n2G - 1) * constInfo.dRopeSize * sizeof(OUTDTYPE);
                ropeGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.n2GDr;
            }
        }
    }

    if constexpr (IS_TND) {
        intriParamsOut.dstStride = (constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE);
        dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * dSize * constInfo.commonConstInfo.n2G;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            intriParamsOut.dstStride = 0;
            dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            intriParamsOut.dstStride = (constInfo.bSize * constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE);
            dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.n2G * constInfo.bSize * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            intriParamsOut.dstStride = (constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(OUTDTYPE);
            dqkvGmOffset += vSubBlockIdx * firsthalfSRealSize * constInfo.commonConstInfo.n2G * dSize;
        }
    }

    GlobalTensor<OUTDTYPE> dqkvGmTensor = MM_IDX == DQ_IDX ? dqGm : (MM_IDX == DK_IDX ? dkGm : dvGm);
    int64_t dkvWorkSpaceOffet =
        this->cBlockIdx * this->CUBE_BASEN * this->HEAD_DIM_ALIGN + vSubBlockIdx * firsthalfSRealSize * curDAlign;
    if constexpr (IS_BN2_MULTIBLK && MM_IDX == DQ_IDX) {
        dkvWorkSpaceOffet = this->cBlockIdx * AlignTo128(constInfo.commonConstInfo.s1Size) * HEAD_DIM_ALIGN +
                            runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN +
                            vSubBlockIdx * firsthalfSRealSize * curDAlign;
    }
    uint32_t data_size = curLoopSize * curDAlign;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
            data_size = curLoopSize * curDAlign;
        }

        LocalTensor<CALC_TYPE> dqkvTensor = inQue.AllocTensor<CALC_TYPE>();
        DataCopy(dqkvTensor, inputTensor[dkvWorkSpaceOffet + loopIdx * loopSize * curDAlign], data_size);

        inQue.EnQue(dqkvTensor);
        inQue.DeQue();
        if constexpr (MM_IDX != DV_IDX) {
            Muls(dqkvTensor, dqkvTensor, constInfo.scaleValue, data_size);
        }
        LocalTensor<OUTDTYPE> dqkvCastTensor = outQue.template AllocTensor<OUTDTYPE>();
        Cast(dqkvCastTensor, dqkvTensor, RoundMode::CAST_ROUND, data_size);
        inQue.FreeTensor(dqkvTensor);
        outQue.EnQue(dqkvCastTensor);
        outQue.template DeQue<OUTDTYPE>();

        if constexpr (IS_ROPE) {
            intriParamsOut.blockCount = curLoopSize;
            if (MM_IDX != DV_IDX) {
                uint16_t dBlockLen = NUM_FOUR * sizeof(OUTDTYPE);
                uint16_t dRopeBlockLen = NUM_TWO * sizeof(OUTDTYPE);
                intriParamsOut.blockLen = constInfo.commonConstInfo.dSizeV * sizeof(OUTDTYPE);
                intriParamsOut.srcStride = dRopeBlockLen;
                DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
                DataCopyExtParams ropeParamsOut;
                ropeParamsOut.blockCount = curLoopSize;
                ropeParamsOut.blockLen = constInfo.dRopeSize * sizeof(OUTDTYPE);
                ropeParamsOut.srcStride = dBlockLen;
                ropeParamsOut.dstStride = ropeDstStride;
                DataCopyPad(dqkRopeGmTensor[ropeGmOffset], dqkvCastTensor[dSize], ropeParamsOut);
            } else {
                // 128 * sizeof(OUTDTYPE) / 32B
                intriParamsOut.blockLen = constInfo.commonConstInfo.dSizeV * sizeof(OUTDTYPE);
                intriParamsOut.srcStride = 0;
                DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
            }
        } else {
            intriParamsOut.blockCount = curLoopSize;
            intriParamsOut.blockLen = dSize * sizeof(OUTDTYPE);
            DataCopyPad(dqkvGmTensor[dqkvGmOffset], dqkvCastTensor, intriParamsOut);
        }
        outQue.FreeTensor(dqkvCastTensor);

        if constexpr (IS_TND) {
            dqkvGmOffset += loopSize * dSize * constInfo.commonConstInfo.n2G;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dqkvGmOffset += loopSize * dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dqkvGmOffset += loopSize * constInfo.commonConstInfo.n2G * constInfo.bSize * dSize;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dqkvGmOffset += loopSize * constInfo.commonConstInfo.n2G * dSize;
            }
        }

        if constexpr (IS_ROPE) {
            if constexpr (IS_TND) {
                ropeGmOffset += loopSize * constInfo.commonConstInfo.n2GDr; // use g axis because of bn2 limit: nq = nkv
            } else {
                if (constInfo.commonConstInfo.layoutType == BNGSD) {
                    ropeGmOffset += loopSize * constInfo.dRopeSize;
                } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                    ropeGmOffset += loopSize * constInfo.commonConstInfo.bN2GDr;
                } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                    ropeGmOffset += loopSize * constInfo.commonConstInfo.n2GDr;
                }
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <const bool IS_DK>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessPostDeter(
    FagConstInfo &constInfo, GlobalTensor<float> dkvWorkSpaceTensor, GlobalTensor<INPUT_TYPE> &dkvGmTensor,
    int64_t specialHalfS2RealSize, int64_t specialFirstHalfS2RealSize, uint64_t dAlign16, uint64_t dvAlign16,
    int64_t specialDkGmOffset, int64_t specialDvGmOffset)
{
    TQue outQue = IS_DK ? dSOutQue : pOutQue;
    if (specialHalfS2RealSize == 0) {
        return;
    }
    uint32_t dSize = constInfo.commonConstInfo.dSize;
    uint32_t curDAlign = dAlign16;
    int64_t dkvGmOffset = IS_DK ? specialDkGmOffset : specialDvGmOffset;

    if constexpr (!IS_DK && IS_D_NO_EQUAL) {
        dSize = constInfo.commonConstInfo.dSizeV;
        curDAlign = dvAlign16;
    }

    uint32_t maxLoopSize = this->VECTOR_BASEM * this->VECTOR_BASEN / curDAlign;
    uint32_t loopNum = Ceil<uint32_t>(specialHalfS2RealSize, maxLoopSize);
    if (loopNum == 0) {
        return;
    }

    uint32_t loopSize = Ceil<uint32_t>(specialHalfS2RealSize, loopNum);
    uint32_t tailLoopSize = specialHalfS2RealSize - (loopNum - 1) * loopSize;
    uint32_t curLoopSize = loopSize;
    DataCopyExtParams intriParamsOut;
    intriParamsOut.srcStride = 0;
    intriParamsOut.dstStride = static_cast<uint32_t>((constInfo.commonConstInfo.n2G - 1) * dSize * sizeof(INPUT_TYPE));
    dkvGmOffset += this->vSubBlockIdx * specialFirstHalfS2RealSize * dSize * constInfo.commonConstInfo.n2G;

    uint32_t data_size = curLoopSize * curDAlign;
    for (uint32_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        if (loopIdx == loopNum - 1) {
            curLoopSize = tailLoopSize;
            data_size = curLoopSize * curDAlign;
        }

        LocalTensor<CALC_TYPE> dkvTensor = attenMaskOrYInQue.AllocTensor<CALC_TYPE>();
        DataCopy(dkvTensor,
                 dkvWorkSpaceTensor[this->vSubBlockIdx * specialFirstHalfS2RealSize * curDAlign +
                                    loopIdx * loopSize * curDAlign],
                 data_size);

        attenMaskOrYInQue.EnQue(dkvTensor);
        attenMaskOrYInQue.DeQue();
        if constexpr (IS_DK) {
            Muls(dkvTensor, dkvTensor, constInfo.scaleValue, data_size);
        }
        LocalTensor<INPUT_TYPE> dkvCastTensor = outQue.template AllocTensor<INPUT_TYPE>();
        Cast(dkvCastTensor, dkvTensor, RoundMode::CAST_ROUND, data_size);
        attenMaskOrYInQue.FreeTensor(dkvTensor);
        outQue.EnQue(dkvCastTensor);
        outQue.template DeQue<INPUT_TYPE>();

        intriParamsOut.blockCount = curLoopSize;
        intriParamsOut.blockLen = dSize * sizeof(INPUT_TYPE);

        DataCopyPad(dkvGmTensor[dkvGmOffset], dkvCastTensor, intriParamsOut);
        outQue.FreeTensor(dkvCastTensor);
        dkvGmOffset += loopSize * dSize * constInfo.commonConstInfo.n2G;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterCompute(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                FagConstInfo &constInfo, bool dqIsNeedDeter[2],
                                                                bool dkDvIsNeedDeter[2], bool isLastSort,
                                                                int64_t computeBlockIdx, int64_t remainLoopNum,
                                                                int16_t *deterPpFlag)
{
    int32_t pingpangIdx = computeBlockIdx & 1;
    if (likely(constInfo.deterConstInfo.noNeedDeter)) {
        if (remainLoopNum > 1) {
            SyncAll<true>();
        }
    } else {
        if (dqIsNeedDeter[pingpangIdx] && !dkDvIsNeedDeter[pingpangIdx]) {
            // 该轮次只有dq需要做deter
            DeterComputeDq(constInfo, isLastSort, computeBlockIdx, remainLoopNum, deterPpFlag);
            isFirstDeter = false;
        } else if (!dqIsNeedDeter[pingpangIdx] && dkDvIsNeedDeter[pingpangIdx]) {
            // 该轮次只有dk dv需要做deter
            DeterComputeDkv(mm1ResTensor, mm2ResTensor, constInfo, isLastSort, computeBlockIdx, remainLoopNum,
                            deterPpFlag);
            isFirstDeter = false;
        } else if (dqIsNeedDeter[pingpangIdx] && dkDvIsNeedDeter[pingpangIdx]) {
            // 该轮次dq dk dv都需要做deter
            DeterComputeDqkv(mm1ResTensor, mm2ResTensor, constInfo, isLastSort, computeBlockIdx, remainLoopNum,
                             deterPpFlag);
            isFirstDeter = false;
        } else {
            // 该轮次不需要做确定性计算，只需要加上全核同步
            if (remainLoopNum > 1) {
                SyncAll();
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterComputeDq(FagConstInfo &constInfo, bool isLastSort,
                                                                  int64_t computeBlockIdx, int64_t remainLoopNum,
                                                                  int16_t *deterPpFlag)
{
    // 卡指定流水的全核同步
    SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
    SyncAll<true, syncAllConfigMte2ToMte2>();

    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    DataCopy(deterOffsetTensor,
             deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * OFFSET_NUM],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * OFFSET_NUM), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    int16_t bufId = 0;
    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    if (unlikely(isLastSort)) {
        bufId = *deterPpFlag;
    } else {
        bufId = 1 - *deterPpFlag;
    }
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * NUM_TWO) * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;

    // dq deter
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dqEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DQ_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;
    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(CALC_TYPE));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqDstStride);

    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(DETER_DQ_UB_SIZE / (constInfo.deterConstInfo.dqEachVectorSize * sizeof(CALC_TYPE)));
    uint8_t loopTimes =
        static_cast<uint8_t>((constInfo.deterConstInfo.usedCubeCoreNum + eachLoopBlockCount - 1) / eachLoopBlockCount);
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;
    AscendC::SetAtomicAdd<CALC_TYPE>();
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<CALC_TYPE> dqDeterBuf = deterInOutQue.AllocTensor<CALC_TYPE>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1) ?
                                        eachLoopBlockCount :
                                        constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dqDeterBuf, deterGm[dqSrcOfs], dataCopyParams);
        }
        dqSrcOfs += eachLoopBlockCount * BASE_DQ_SIZE;
        deterInOutQue.EnQue(dqDeterBuf);
        deterInOutQue.DeQue<CALC_TYPE>();

        eachLoopStart = loopIdx * eachLoopBlockCount;
        if (loopIdx == loopTimes - 1) {
            eachLoopEnd = constInfo.deterConstInfo.usedCubeCoreNum;
        } else {
            eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        }
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            dqOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM);
        }
        // dq 每个V核需要处理所有C核的dq结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dqOffset[cIx] == OUTINDEX) {
                continue;
            }
            dqOffset[cIx] += constInfo.deterConstInfo.deterVecCoreS1Offset;
            if constexpr (IS_FP32_INPUT) {
                dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
                uint64_t totalCopySizeBytes = dataCopyPadParams.blockCount <= 0 ?
                                                  0 :
                                                  ((dataCopyPadParams.blockCount * dataCopyPadParams.blockLen) +
                                                   ((dataCopyPadParams.blockCount - 1) * dataCopyPadParams.dstStride));
                uint64_t writeStartAddr = dqGmBaseAddr + dqOffset[cIx] * sizeof(CALC_TYPE);
                uint64_t writeLastAddr = writeStartAddr + totalCopySizeBytes;
                uint16_t avaliableRows =
                    FindAvailableRows<DQ_IDX>(constInfo, writeStartAddr, writeLastAddr, dataCopyPadParams.blockCount,
                                              dataCopyPadParams.blockLen, dataCopyPadParams.dstStride);
                if (avaliableRows == 0) {
                    continue;
                }
                dataCopyPadParams.blockCount = avaliableRows;
            }
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dqWorkSpaceGm[dqOffset[cIx]],
                                     dqDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dqEachVectorSize],
                                     dataCopyPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        deterInOutQue.FreeTensor(dqDeterBuf);
    }
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }

    if (remainLoopNum > NUM_TWO) { // 最后两轮不需要卡
        SyncAll<true, syncAllConfigMte3ToMte3>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterComputeDkv(LocalTensor<CALC_TYPE> &dkDeterBuf,
                                                                   LocalTensor<CALC_TYPE> &dvDeterBuf,
                                                                   FagConstInfo &constInfo, bool isLastSort,
                                                                   int64_t computeBlockIdx, int64_t remainLoopNum,
                                                                   int16_t *deterPpFlag)
{
    SyncAll<>();
    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    DataCopy(deterOffsetTensor,
             deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * OFFSET_NUM],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * OFFSET_NUM), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    for (uint16_t cIx = 0; cIx < constInfo.deterConstInfo.usedCubeCoreNum; cIx++) {
        dkOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM);
        dvOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM * NUM_TWO);
    }
    int16_t bufId = 0;
    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    if (isLastSort) {
        bufId = *deterPpFlag;
    } else {
        bufId = 1 - *deterPpFlag;
    }
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * NUM_TWO) * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dkSrcOfs = dqSrcOfs + BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dvSrcOfs = dkSrcOfs + BASE_DKV_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;
    dkSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;
    dvSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;

    // dk dv deter
    AscendC::SetAtomicAdd<CALC_TYPE>();

    WriteDataToDkv(dkDeterBuf, dvDeterBuf, constInfo, dkSrcOfs, dvSrcOfs);
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    if (remainLoopNum > NUM_TWO) { // 最后两轮不需要卡
        SyncAll(); // 由于复用了mm1mm2的在ub中的buf，所以为了防止确定性计算还没有做完，后面的mm已经做完的情况，踩数据，所以只能卡scalar
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <uint8_t RES_IDX>
__aicore__ inline uint16_t FAGBlockVec<TEMPLATE_ARGS>::FindAvailableRows(FagConstInfo &constInfo,
                                                                         uint64_t writeStartAddr,
                                                                         uint64_t writeLastAddr, uint16_t blockCount,
                                                                         uint32_t blockLen, uint32_t dstStride)
{
    uint64_t gmLimitOffset =
        RES_IDX == DQ_IDX ? dqGmLimitOffset : (RES_IDX == DK_IDX ? dkGmLimitOffset : dvGmLimitOffset);
    if (writeLastAddr > gmLimitOffset) {
        uint64_t remainBytes = writeStartAddr > gmLimitOffset ? 0 : gmLimitOffset - writeStartAddr;
        if (remainBytes == 0 || blockCount == 0) {
            return 0;
        }
        uint64_t avaliableRows = 0;
        uint64_t rowSize = blockLen + dstStride;
        if (remainBytes >= blockLen) {
            avaliableRows = 1 + (remainBytes - blockLen) / rowSize;
        }
        if (avaliableRows >= blockCount) {
            avaliableRows = blockCount;
        }
        return static_cast<uint16_t>(avaliableRows);
    } else {
        return blockCount;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterComputeDqkv(LocalTensor<CALC_TYPE> &dkDeterBuf,
                                                                    LocalTensor<CALC_TYPE> &dvDeterBuf,
                                                                    FagConstInfo &constInfo, bool isLastSort,
                                                                    int64_t computeBlockIdx, int64_t remainLoopNum,
                                                                    int16_t *deterPpFlag)
{
    SyncAll<>();
    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    DataCopy(deterOffsetTensor,
             deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * OFFSET_NUM],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * OFFSET_NUM), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
    for (uint16_t cIx = 0; cIx < constInfo.deterConstInfo.usedCubeCoreNum; cIx++) {
        dqOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM);
        dkOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM);
        dvOffset[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM * NUM_TWO);
    }
    int16_t bufId = 0;
    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    if (isLastSort) {
        bufId = *deterPpFlag;
    } else {
        bufId = 1 - *deterPpFlag;
    }
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * 2) * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dkSrcOfs = dqSrcOfs + BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dvSrcOfs = dkSrcOfs + BASE_DKV_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;
    dkSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;
    dvSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;

    // dq deter
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dqEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DQ_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;
    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(CALC_TYPE));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqDstStride);

    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(DETER_DQ_UB_SIZE / (constInfo.deterConstInfo.dqEachVectorSize * sizeof(CALC_TYPE)));
    uint8_t loopTimes =
        static_cast<uint8_t>((constInfo.deterConstInfo.usedCubeCoreNum + eachLoopBlockCount - 1) / eachLoopBlockCount);
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;
    AscendC::SetAtomicAdd<CALC_TYPE>();
    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<CALC_TYPE> dqDeterBuf = deterInOutQue.AllocTensor<CALC_TYPE>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1) ?
                                        eachLoopBlockCount :
                                        constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dqDeterBuf, deterGm[dqSrcOfs], dataCopyParams);
        }
        dqSrcOfs += eachLoopBlockCount * BASE_DQ_SIZE;
        deterInOutQue.EnQue(dqDeterBuf);
        deterInOutQue.DeQue<CALC_TYPE>();

        eachLoopStart = loopIdx * eachLoopBlockCount;
        if (loopIdx == loopTimes - 1) {
            eachLoopEnd = constInfo.deterConstInfo.usedCubeCoreNum;
        } else {
            eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        }
        // dq 每个V核需要处理所有C核的dq结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dqOffset[cIx] == OUTINDEX) {
                continue;
            }
            dqOffset[cIx] += constInfo.deterConstInfo.deterVecCoreS1Offset;
            if constexpr (IS_FP32_INPUT) {
                dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
                uint64_t totalCopySizeBytes = dataCopyPadParams.blockCount <= 0 ?
                                                  0 :
                                                  ((dataCopyPadParams.blockCount * dataCopyPadParams.blockLen) +
                                                   ((dataCopyPadParams.blockCount - 1) * dataCopyPadParams.dstStride));
                uint64_t writeStartAddr = dqGmBaseAddr + dqOffset[cIx] * sizeof(CALC_TYPE);
                uint64_t writeLastAddr = writeStartAddr + totalCopySizeBytes;
                uint16_t avaliableRows =
                    FindAvailableRows<DQ_IDX>(constInfo, writeStartAddr, writeLastAddr, dataCopyPadParams.blockCount,
                                              dataCopyPadParams.blockLen, dataCopyPadParams.dstStride);
                if (avaliableRows == 0) {
                    continue;
                }
                dataCopyPadParams.blockCount = avaliableRows;
            }
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dqWorkSpaceGm[dqOffset[cIx]],
                                     dqDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dqEachVectorSize],
                                     dataCopyPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        deterInOutQue.FreeTensor(dqDeterBuf);
    }

    // dk dv deter
    WriteDataToDkv(dkDeterBuf, dvDeterBuf, constInfo, dkSrcOfs, dvSrcOfs);
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    if (remainLoopNum > NUM_TWO) { // 最后两轮不需要卡
        SyncAll(); // 由于复用了mm1mm2的在ub中的buf，所以为了防止确定性计算还没有做完，后面的mm已经做完的情况，踩数据，所以只能卡scalar
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::WriteDataToDkv(LocalTensor<CALC_TYPE> &dkDeterBuf,
                                                                  LocalTensor<CALC_TYPE> &dvDeterBuf,
                                                                  FagConstInfo &constInfo, uint32_t dkSrcOfs,
                                                                  uint32_t dvSrcOfs)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dkvEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DKV_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;

    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(CALC_TYPE));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDkDstStride);

    DataCopyExtParams dataCopyDvPadParams;
    dataCopyDvPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyDvPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(CALC_TYPE));
    dataCopyDvPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvSrcStride);
    dataCopyDvPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvDstStride);

    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(DETER_DKV_UB_SIZE / (constInfo.deterConstInfo.dkvEachVectorSize * sizeof(CALC_TYPE)));
    uint8_t loopTimes =
        static_cast<uint8_t>((constInfo.deterConstInfo.usedCubeCoreNum + eachLoopBlockCount - 1) / eachLoopBlockCount);
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;

    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1) ?
                                        eachLoopBlockCount :
                                        constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dkDeterBuf, deterGm[dkSrcOfs], dataCopyParams);
            DataCopy(dvDeterBuf, deterGm[dvSrcOfs], dataCopyParams);
        }
        dkSrcOfs += eachLoopBlockCount * BASE_DKV_SIZE;
        dvSrcOfs += eachLoopBlockCount * BASE_DKV_SIZE;

        SetFlag<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);

        eachLoopStart = loopIdx * eachLoopBlockCount;
        if (loopIdx == loopTimes - 1) {
            eachLoopEnd = constInfo.deterConstInfo.usedCubeCoreNum;
        } else {
            eachLoopEnd = Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        }
        // dq 每个V核需要处理所有C核的dk,dv结果
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dkOffset[cIx] == OUTINDEX && dvOffset[cIx] == OUTINDEX) {
                continue;
            }
            dkOffset[cIx] += constInfo.deterConstInfo.deterDkVecCoreS2Offset;
            dvOffset[cIx] += constInfo.deterConstInfo.deterDvVecCoreS2Offset;
            if constexpr (IS_FP32_INPUT) {
                dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
                dataCopyDvPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
                uint64_t totalCopySizeBytesDk =
                    dataCopyPadParams.blockCount <= 0 ?
                        0 :
                        ((dataCopyPadParams.blockCount * dataCopyPadParams.blockLen) +
                         ((dataCopyPadParams.blockCount - 1) * dataCopyPadParams.dstStride));
                uint64_t totalCopySizeBytesDv =
                    dataCopyDvPadParams.blockCount <= 0 ?
                        0 :
                        ((dataCopyDvPadParams.blockCount * dataCopyDvPadParams.blockLen) +
                         ((dataCopyDvPadParams.blockCount - 1) * dataCopyDvPadParams.dstStride));
                uint64_t writeStartAddrDk = dkGmBaseAddr + dkOffset[cIx] * sizeof(CALC_TYPE);
                uint64_t writeStartAddrDv = dvGmBaseAddr + dvOffset[cIx] * sizeof(CALC_TYPE);
                uint64_t writeLastAddrDk = writeStartAddrDk + totalCopySizeBytesDk;
                uint64_t writeLastAddrDv = writeStartAddrDv + totalCopySizeBytesDv;
                uint16_t avaliableRowsDk = FindAvailableRows<DK_IDX>(
                    constInfo, writeStartAddrDk, writeLastAddrDk, dataCopyPadParams.blockCount,
                    dataCopyPadParams.blockLen, dataCopyPadParams.dstStride);
                uint16_t avaliableRowsDv = FindAvailableRows<DV_IDX>(
                    constInfo, writeStartAddrDv, writeLastAddrDv, dataCopyDvPadParams.blockCount,
                    dataCopyDvPadParams.blockLen, dataCopyDvPadParams.dstStride);
                if (avaliableRowsDk == 0 && avaliableRowsDv == 0) {
                    continue;
                }
                dataCopyPadParams.blockCount = avaliableRowsDk;
                dataCopyDvPadParams.blockCount = avaliableRowsDv;
                if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum && dataCopyPadParams.blockCount > 0) {
                    AscendC::DataCopyPad(dkWorkSpaceGm[dkOffset[cIx]],
                                         dkDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                         dataCopyPadParams);
                }
                if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum && dataCopyDvPadParams.blockCount > 0) {
                    AscendC::DataCopyPad(dvWorkSpaceGm[dvOffset[cIx]],
                                         dvDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                         dataCopyDvPadParams);
                }
            } else {
                if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                    AscendC::DataCopyPad(dkWorkSpaceGm[dkOffset[cIx]],
                                         dkDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                         dataCopyPadParams);
                    AscendC::DataCopyPad(dvWorkSpaceGm[dvOffset[cIx]],
                                         dvDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                         dataCopyDvPadParams);
                }
            }
            PipeBarrier<PIPE_MTE3>();
        }
        if (loopIdx < loopTimes - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::WriteOffsetToGM(FagConstInfo &constInfo, int64_t queryGmOffset,
                                                                   int64_t keyGmOffset, int64_t valueGmOffset,
                                                                   int64_t computeBlockIdx, int64_t remainLoopNum)
{
    if (constInfo.deterConstInfo.noNeedDeter) {
        return;
    }

    if (vSubBlockIdx == 1) {
        return;
    }
    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (computeBlockIdx > 0) {
        WaitFlag<HardEvent::MTE3_S>(constInfo.deterConstInfo.eventIDMte3ToScalar);
    }
    deterOffsetTensor.SetValue(0, queryGmOffset);
    deterOffsetTensor.SetValue(INT64_BYTES, keyGmOffset);
    deterOffsetTensor.SetValue(INT64_BYTES * NUM_TWO, valueGmOffset);

    SetFlag<HardEvent::S_MTE3>(constInfo.deterConstInfo.eventIDScalarToMte3);
    WaitFlag<HardEvent::S_MTE3>(constInfo.deterConstInfo.eventIDScalarToMte3);
    DataCopy(deterOffsetGm[(computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum + cBlockIdx) * INT64_BYTES *
                           OFFSET_NUM],
             deterOffsetTensor, {1, OFFSET_NUM, 0, 0});
    if (remainLoopNum > 1) {
        SetFlag<HardEvent::MTE3_S>(constInfo.deterConstInfo.eventIDMte3ToScalar);
    }
}

TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void InitUbBuffer() {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope,
                                            GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace) {};
    __aicore__ inline void SetVecBlockParams(TPipe *pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, AttenMaskInfo &attenMaskInfo,
                                             PseInfo &pseInfo, DropMaskInfo &dropInfo) {};
    __aicore__ inline void SetOldDeterFp32Param(FagConstInfo &constInfo) {};
    __aicore__ inline void ProcessVec1(FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec2(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo) {};
    __aicore__ inline void CopyDpToTmpBuf(LocalTensor<CALC_TYPE> &mm1ResTensor) {};
    __aicore__ inline void ProcessVecSink(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                          FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void ProcessVec3(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter = false) {};
    __aicore__ inline void ProcessVec4(MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dstBuffer,
                                       LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo) {};
    template <typename T, bool IS_WRITE_UB, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCast(typename DqkvResPos<T, IS_WRITE_UB>::PosType inputTensor,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo) {};
    __aicore__ inline void CopyMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo, int64_t taskId) {};
    template <const bool IS_DK>
    __aicore__ inline void ProcessPostDeter(FagConstInfo &constInfo, GlobalTensor<float> dkvWorkSpaceTensor,
                                            GlobalTensor<INPUT_TYPE> &dkvGmTensor, int64_t specialHalfS2RealSize,
                                            int64_t specialFirstHalfS2RealSize, uint64_t dAlign16, uint64_t dvAlign16,
                                            int64_t specialDkGmOffset, int64_t specialDvGmOffset) {};
    __aicore__ inline void DeterCompute(LocalTensor<CALC_TYPE> &mm1ResTensor, LocalTensor<CALC_TYPE> &mm2ResTensor,
                                        FagConstInfo &constInfo, bool dqIsNeedDeter[2], bool dkDvIsNeedDeter[2],
                                        bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                        int16_t *deterPpFlag) {};
    __aicore__ inline void DeterComputeDq(FagConstInfo &constInfo, bool isLastSort, int64_t computeBlockIdx,
                                          int64_t remainLoopNum, int16_t *deterPpFlag) {};
    __aicore__ inline void DeterComputeDkv(LocalTensor<CALC_TYPE> &dkDeterBuf, LocalTensor<CALC_TYPE> &dvDeterBuf,
                                           FagConstInfo &constInfo, bool isLastSort, int64_t computeBlockIdx,
                                           int64_t remainLoopNum, int16_t *deterPpFlag) {};
    __aicore__ inline void DeterComputeDqkv(LocalTensor<CALC_TYPE> &dkDeterBuf, LocalTensor<CALC_TYPE> &dvDeterBuf,
                                            FagConstInfo &constInfo, bool isLastSort, int64_t computeBlockIdx,
                                            int64_t remainLoopNum, int16_t *deterPpFlag) {};
    __aicore__ inline void WriteOffsetToGM(FagConstInfo &constInfo, int64_t queryGmOffset, int64_t keyGmOffset,
                                           int64_t valueGmOffset, int64_t computeBlockIdx, int64_t remainLoopNum) {};
};

} // namespace FagBaseApi

#endif
