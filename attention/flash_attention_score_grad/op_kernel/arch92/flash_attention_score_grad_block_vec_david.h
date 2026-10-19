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
 * \file flash_attention_score_grad_block_vec_david.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_DAVID_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_DAVID_H

#include "flash_attention_score_grad_common_david.h"
#include "vector_api/cast_softmax_grad_david.h"
#include "vector_api/dropout_david.h"
#include "vector_api/pse_atten_mask_muls_simple_softmax_david.h"
#include "vector_api/vf_ds_abs_reduce_max_david.h"
#include "vector_api/vf_broadcast_sub_mul_david.h"
#include "vector_api/vf_cast_transdata_deconflict_david.h"
#include "vector_api/vf_cal_sink_david.h"
#include "flash_attention_score_grad_tiling_data_david.h"
namespace FagBaseApi {
constexpr uint32_t BIT_MASK_NUM = 8;

template <typename T>
constexpr bool is_e4m3_v = std::is_same_v<T, float8_e4m3_t>;

template <typename T>
constexpr bool is_e5m2_v = std::is_same_v<T, float8_e5m2_t>;

template <typename T>
constexpr bool is_f8_v = is_e4m3_v<T> || is_e5m2_v<T>;

template <typename T>
constexpr bool is_f16_v = std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>;

template <typename T>
constexpr bool is_f32_v = std::is_same_v<T, float>;
using namespace optiling::fag;

TEMPLATES_DEF
class FAGBlockVec {
private:
public:
    __aicore__ inline FAGBlockVec(){};
    __aicore__ inline void SetVecBlockParams(TPipe* pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, AttenMaskInfo& attenMaskInfo,
                                             PseInfo& pseInfo, DropMaskInfo& dropInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR sink, GM_ADDR dsink,
                                            GM_ADDR workspace);
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void InitAsync();
    __aicore__ inline void FreeAsync(FagConstInfo& constInfo);
    __aicore__ inline void PostCompute(FagRunInfo& runInfo, FagConstInfo& constInfo, GlobalTensor<float>& dqWorkSpaceGm,
                                       GlobalTensor<float>& dkWorkSpaceGm);
    __aicore__ inline void ProcessVec1(FagRunInfo& runInfo, FagConstInfo& constInfo);
    __aicore__ inline void ProcessVec2(FagRunInfo& runInfo, FagRunInfo& nextRunInfo, FagConstInfo& constInfo,
                                       bool isInterBlockLast, LocalTensor<T2>& mm1ResTensor,
                                       LocalTensor<T2>& mm2ResTensor, int8_t innerLoopNum, int8_t innerLoopIdx,
                                       int8_t baseUbBufIdx);
    __aicore__ inline void AsyncAndMte2CopyBeforeReCompute(FagRunInfo& runInfo, FagRunInfo& nextRunInfo,
                                                           FagConstInfo& constInfo, bool isInnerBlockLast,
                                                           bool isInterBlockLast);
    __aicore__ inline void ReCompute(LocalTensor<T2>& mm1ResQueInTensor, LocalTensor<T2>& mm2ResQueInTensor,
                                     FagRunInfo& runInfo, FagConstInfo& constInfo, bool isInnerBlockFirst,
                                     bool isInnerBlockLast, bool isInterBlockLast);
    __aicore__ inline void AsyncAfterReCompute(bool isInnerBlockLast, bool isInterBlockLast);
    __aicore__ inline void CopyDpToTmpBuf(LocalTensor<T2>& mm1ResQueInTensor);
    __aicore__ inline void ProcessVecSink(LocalTensor<T2>& mm2ResQueInTensor, FagRunInfo& runInfo,
                                          FagConstInfo& constInfo, bool isInnerBlockLast);
    template <typename T, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCast(GlobalTensor<T>& workSpaceGm, FagConstInfo& constInfo,
                                              FagRunInfo& runInfo);
    template <typename CopyType>
    __aicore__ inline void CopyUB2L1(FagRunInfo& runInfo, LocalTensor<CopyType>& dstTensor,
                                     LocalTensor<CopyType>& srcTensor);

    template <typename CopyType>
    __aicore__ inline void CopyUB2L1Deter(FagRunInfo& runInfo, LocalTensor<CopyType>& dstTensor,
                                          LocalTensor<CopyType>& srcTensor, uint32_t scmOffset);

    __aicore__ inline void WriteOffsetToGM(FagConstInfo& constInfo, int64_t queryGmOffset, int64_t keyGmOffset,
                                           int64_t valueGmOffset, int64_t computeBlockIdx, int64_t remainLoopNum);
    __aicore__ inline void DeterCompute(LocalTensor<T2>& mm1ResTensor, LocalTensor<T2>& mm2ResTensor,
                                        FagConstInfo& constInfo, bool dqIsNeedDeter, bool dkDvIsNeedDeter,
                                        bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                        int16_t deterPpFlag);
    __aicore__ inline void DeterComputeDq(FagConstInfo& constInfo, bool isLastSort, int64_t computeBlockIdx,
                                          int64_t remainLoopNum, int16_t deterPpFlag);
    __aicore__ inline void DeterComputeDkv(LocalTensor<T2>& dkBuf, LocalTensor<T2>& dvBuf, FagConstInfo& constInfo,
                                           bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                           int16_t deterPpFlag);
    __aicore__ inline void DeterComputeDqkv(LocalTensor<T2>& dkBuf, LocalTensor<T2>& dvBuf, FagConstInfo& constInfo,
                                            bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                            int16_t deterPpFlag);
    __aicore__ inline void WriteDataToDkv(LocalTensor<T2>& dkBuf, LocalTensor<T2>& dvBuf, FagConstInfo& constInfo,
                                          uint32_t dkSrcOfs, uint32_t dvSrcOfs);

    constexpr static bool IS_FP8_INPUT = is_f8_v<T1>;
    constexpr static bool IS_FP32_INPUT = is_f32_v<T1>;
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 6;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = 64;
    constexpr static uint32_t VECTOR_BASEN = 128;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(T1);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;
    constexpr static uint32_t UB_BUF_NUM = IS_FP32_INPUT ? 2 : 3;
    constexpr static uint32_t POST_UB_SIZE = 32 * 1024;
    constexpr static uint32_t DS_P_L1_BUF_SIZE = CUBE_BASEM * CUBE_BASEN;
    constexpr static uint32_t OFFSET_NUM = 3;
    constexpr static uint32_t INT64_BYTES = 4;
    constexpr static uint32_t INT64_BLOCK_NUM = 4;
    constexpr static uint32_t NUM_TWO = 2;
    constexpr static int64_t OUTINDEX = -1;
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static uint32_t FLOAT_BLOCK_SIZE = 32 / sizeof(float);
    constexpr static uint32_t DETER_DKV_UB_SIZE = VECTOR_BASEM * VECTOR_BASEN * sizeof(T2);
    constexpr static uint32_t BIT32_ALIGN_NUM = 8;

    using DsType = std::conditional_t<is_e4m3_v<T1>, bfloat16_t, T1>;
    // vector gm addr
    // input
    GlobalTensor<T1> valueGm, dxGm, yGm, pseGm, dqGm, dkGm, dvGm;
    GlobalTensor<uint8_t> dropMaskGm, dropMaskWorkspaceGm, attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, pseFloatGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm;
    GlobalTensor<float> sinkGm, dsinkGm, dsinkWorkSpaceGm;
    __gm__ uint8_t* pseSlope;
    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, sfmgWorkspaceGm;
    GlobalTensor<std::conditional_t<SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK, T1, float>> dvWorkSpaceGm;

    GlobalTensor<float> deterGm;
    GlobalTensor<int64_t> deterOffsetGm;

    // old deter info
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), bool, std::nullptr_t>::type isFirstDeter;
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[MAX_CORE_NUM], std::nullptr_t>::type
        dqOffsetArr{};
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[MAX_CORE_NUM], std::nullptr_t>::type
        dkOffsetArr{};
    typename std::conditional<IS_DETER_OLD(DETER_SPARSE_TYPE), int64_t[MAX_CORE_NUM], std::nullptr_t>::type
        dvOffsetArr{};
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> deterInOutQue;
    TBuf<> deterOffsetBuf;
    TBuf<> vselrIndexesBuf;
    TBuf<> dsAmaxBuf;
    TBuf<> postBuf[2];

    // CV
    TBuf<TPosition::A1> tBufL1Ds;
    LocalTensor<DsType> L1DsTensor;

    TBuf<TPosition::A1> tBufL1P;
    LocalTensor<T1> L1PTensor;

    // ub reuse
    TBuf<> maxSumSfmgBuf[UB_BUF_NUM];
    TBuf<> dropmaskIndexVecBuf;
    TBuf<> dSOutBuf[UB_BUF_NUM];
    TBuf<> dropMaskBuf;
    TBuf<> pOutBuf[UB_BUF_NUM];
    TBuf<> maxSumInBuf[2];
    TQue<QuePosition::VECOUT, 1> dsinkResOutQue[2];
    TBuf<> dsinkResBuf[2];

    TPipe* pipe;
    FagTilingType tilingData;

    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;

    // 同步事件
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t QKDXV_RES_READY[3] = {0, 1, 2};
    static constexpr uint64_t QKDXV_RES_RELEASE[3] = {3, 4, 5};
    static constexpr uint64_t DS_READY = 8;
    static constexpr uint64_t POUT_READY = 9;
    static constexpr uint64_t DS_RELEASE = 10;
    static constexpr uint64_t POUT_RELEASE = 11;
    static constexpr uint64_t ALL_DONE = 12;

    // optional info
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    DropMaskInfo dropInfo;

    TEventID maxSumMte2ToMte3[UB_BUF_NUM];
    TEventID mte2ToV[UB_BUF_NUM];
    uint32_t mte2ToVEventId = 0;
    TEventID postMte2ToMte3[2];

    uint32_t ubBufIdx = 0;
    uint32_t toBeFreeBufIdx = 0;
    uint32_t postPingpong = 0;
    int64_t deterGmOffset = 0;
    float dsinkAccu = 0.0;
    uint32_t sinkPingpong = 0;
    uint32_t dsinkPingpong = 0;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::SetVecBlockParams(TPipe* pipe, FagTilingType tilingData,
                                                                     uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                     uint32_t vSubBlockIdx,
                                                                     AttenMaskInfo& attenMaskInfo, PseInfo& pseInfo,
                                                                     DropMaskInfo& dropInfo)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
    this->attenMaskInfo = attenMaskInfo;
    this->pseInfo = pseInfo;
    this->dropInfo = dropInfo;
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        isFirstDeter = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR softmaxMax,
    GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR sink, GM_ADDR dsink, GM_ADDR workspace)
{
    valueGm.SetGlobalBuffer((__gm__ T1*)value);
    dxGm.SetGlobalBuffer((__gm__ T1*)dy);
    yGm.SetGlobalBuffer((__gm__ T1*)y);
    pseGm.SetGlobalBuffer((__gm__ T1*)pseShift);
    pseFloatGm.SetGlobalBuffer((__gm__ float*)pseShift);
    dropMaskGm.SetGlobalBuffer((GM_ADDR)dropMask);
    attenMaskU8Gm.SetGlobalBuffer((GM_ADDR)attenMask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmaxSum);
    deqScaleQGm.SetGlobalBuffer((__gm__ float*)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float*)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float*)deqScaleV);
    pseSlope = pseShift;
    dqGm.SetGlobalBuffer((__gm__ T1*)dq);
    dkGm.SetGlobalBuffer((__gm__ T1*)dk);
    dvGm.SetGlobalBuffer((__gm__ T1*)dv);
    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace +
                                    tilingData->postTilingData.sfmgWorkSpaceOffset / sizeof(T2));
    if constexpr (IS_DROP) {
        if (tilingData->preTilingData.dropoutIsDivisibleBy8 == 0) {
            dropMaskWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace +
                                                tilingData->postTilingData.dropMaskGmOffset);
        }
    }

    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) {
        sinkGm.SetGlobalBuffer((__gm__ float*)sink);
        dsinkGm.SetGlobalBuffer((__gm__ float*)dsink);
        dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                         tilingData->postTilingData.dsinkWorkSpaceOffset / sizeof(float));
    }

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        deterGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->postTilingData.deterGmOffset / sizeof(float));
        deterOffsetGm.SetGlobalBuffer((__gm__ int64_t*)workspace +
                                      tilingData->postTilingData.deterWorkSpaceOffset / sizeof(int64_t));
        if constexpr (!IS_FP32_INPUT) {
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                          tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                          tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                          tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        } else {
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)dq);
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)dk);
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)dv);
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

    if constexpr (IS_DROP) {
        pipe->InitBuffer(dropMaskBuf, VECTOR_BASEM * VECTOR_BASEN * sizeof(uint8_t) / BIT_MASK_NUM); // 1k
        pipe->InitBuffer(dropmaskIndexVecBuf, VECTOR_BASEM * VECTOR_BASEN / 16 * sizeof(int32_t));   // 2k
    }

    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) {
        for (int i = 0; i < NUM_TWO; ++i) {
            pipe->InitBuffer(dsinkResOutQue[i], 1, BIT32_ALIGN_NUM * sizeof(float));
            pipe->InitBuffer(dsinkResBuf[i], sizeof(float));
        }
    }

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        pipe->InitBuffer(deterInOutQue, 1, DETER_DQ_UB_SIZE);
        pipe->InitBuffer(deterOffsetBuf, DETER_OFFSET_UB_SIZE); // 3k
    }
    if constexpr (IS_FP8_INPUT) {
        pipe->InitBuffer(vselrIndexesBuf, 128 * sizeof(uint8_t));
        LocalTensor<uint8_t> selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
        for (int i = 0; i < 128; i++) {
            selrIndexesTensor.SetValue(i, i * 2);
        }
        pipe->InitBuffer(dsAmaxBuf, VREG_SIZE / 2);
        pipe->InitBuffer(maxSumInBuf[0], 3072); // max sum grad 共3k
        pipe->InitBuffer(maxSumInBuf[1], 3072);
    }
    if constexpr (!IS_FP32_INPUT) {
        // Use +1 padding to reduce bank conflict
        for (int32_t i = 0; i < UB_BUF_NUM; i++) {
            if (IS_FP8_INPUT) {
                pipe->InitBuffer(dSOutBuf[i], (VECTOR_BASEM + 1) * VREG_SIZE * 2);
            } else {
                pipe->InitBuffer(dSOutBuf[i], VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
            }
            pipe->InitBuffer(maxSumSfmgBuf[i],
                             VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2 + VECTOR_BASEM * sizeof(uint32_t));
            pipe->InitBuffer(pOutBuf[i], VECTOR_BASEM * VREG_SIZE + VREG_SIZE);
        }
    } else {
        for (int32_t i = 0; i < UB_BUF_NUM; i++) {
            pipe->InitBuffer(dSOutBuf[i], VECTOR_BASEM * VECTOR_BASEN * sizeof(T1) + VECTOR_BASEN * sizeof(T1));
            pipe->InitBuffer(pOutBuf[i], VECTOR_BASEM * VECTOR_BASEN * sizeof(T1) + VECTOR_BASEN * sizeof(T1));
            pipe->InitBuffer(maxSumSfmgBuf[i],
                             VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2 + VECTOR_BASEM * sizeof(uint32_t));
        }
    }
    // split BN2 post compute or sink
    if (SPLIT_AXIS == BN2 || tilingData->s1s2BNGS1S2BaseParams.sinkOptional) {
        pipe->InitBuffer(postBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2)); // 32k
    }
    if (SPLIT_AXIS == BN2) {
        pipe->InitBuffer(postBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2)); // 32k
    }
    // L1 tensor
    pipe->InitBuffer(tBufL1Ds, DS_P_L1_BUF_SIZE * sizeof(DsType));
    L1DsTensor = tBufL1Ds.template Get<DsType>();
    pipe->InitBuffer(tBufL1P, DS_P_L1_BUF_SIZE * sizeof(T1));
    L1PTensor = tBufL1P.template Get<T1>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::InitAsync()
{
    // 先发vec的同步
    for (int32_t i = 0; i < UB_BUF_NUM; i++) {
        maxSumMte2ToMte3[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(maxSumMte2ToMte3[i]);

        // Mte2 到 V 的同步
        mte2ToV[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    }

    postMte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    postMte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[0]);
    SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[1]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::FreeAsync(FagConstInfo& constInfo)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(this->DS_RELEASE);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(this->POUT_RELEASE);
    if constexpr (!IS_FP32_INPUT) {
        toBeFreeBufIdx = (ubBufIdx + UB_BUF_NUM - 1) % UB_BUF_NUM; // 匹配消除
        WaitFlag<HardEvent::MTE3_MTE2>(maxSumMte2ToMte3[toBeFreeBufIdx]);
    }
    for (int32_t i = 0; i < UB_BUF_NUM; i++) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(maxSumMte2ToMte3[i]);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2ToV[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2ToV[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(postMte2ToMte3[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(postMte2ToMte3[1]);
    // deter 相关参数
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE3>(constInfo.deterConstInfo.eventIDScalarToMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_S>(constInfo.deterConstInfo.eventIDMte3ToScalar);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(constInfo.deterConstInfo.eventIDMte2ToMte3);
    }
    CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(ALL_DONE); // 等DV搬完
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec2(FagRunInfo& runInfo, FagRunInfo& nextRunInfo,
                                                               FagConstInfo& constInfo, bool isInterBlockLast,
                                                               LocalTensor<T2>& mm1ResTensor,
                                                               LocalTensor<T2>& mm2ResTensor, int8_t innerLoopNum,
                                                               int8_t innerLoopIdx, int8_t baseUbBufIdx)
{
    bool isInnerBlockFirst = innerLoopIdx == 0;
    bool isInnerBlockLast = (innerLoopIdx == innerLoopNum - 1);
    ubBufIdx = baseUbBufIdx; // TODO 临时措施
    uint32_t mSize =
        (isInnerBlockLast) ? (runInfo.commonRunInfo.s1RealSize - innerLoopIdx * VECTOR_BASEM) : VECTOR_BASEM;

    runInfo.commonRunInfo.halfS1RealSize = mSize;
    runInfo.commonRunInfo.vecS1Offset = innerLoopIdx * VECTOR_BASEM;
    AsyncAndMte2CopyBeforeReCompute(runInfo, nextRunInfo, constInfo, isInnerBlockLast, isInterBlockLast);
    ReCompute(mm1ResTensor, mm2ResTensor, runInfo, constInfo, isInnerBlockFirst, isInnerBlockLast,
              isInterBlockLast); // VF2 3
    AsyncAfterReCompute(isInnerBlockLast, isInterBlockLast);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::PostCompute(FagRunInfo& runInfo, FagConstInfo& constInfo,
                                                               GlobalTensor<float>& dqWorkSpaceGm,
                                                               GlobalTensor<float>& dkWorkSpaceGm)
{
    uint32_t dAlign = AlignTo32(constInfo.commonConstInfo.dSize * sizeof(T1)) / sizeof(T1); // 对齐到一个block
    uint32_t sSize = POST_UB_SIZE / sizeof(T2) / dAlign;
    uint64_t dqGmOffset = runInfo.dqGmOffset;
    uint64_t dkGmOffset = runInfo.dkGmOffset;
    uint64_t dstOffset;
    DataCopyExtParams inParams;
    inParams.blockLen = constInfo.commonConstInfo.dSize * sizeof(T2);
    inParams.dstStride = (dAlign - constInfo.commonConstInfo.dSize) * sizeof(T2) / 32;
    inParams.srcStride = 0;
    DataCopyExtParams outParams;
    outParams.blockLen = constInfo.commonConstInfo.dSize * sizeof(T1);
    outParams.srcStride = 0;
    // Dq
    uint32_t dqLoopTimes = (runInfo.commonRunInfo.s1RealSize + sSize - 1) / sSize;
    uint32_t s1Size = (runInfo.commonRunInfo.s1RealSize + dqLoopTimes - 1) / dqLoopTimes;
    for (uint32_t i = 0; i < dqLoopTimes; i++) {
        uint32_t realS = (i == dqLoopTimes - 1) ? (runInfo.commonRunInfo.s1RealSize - i * s1Size) : s1Size;
        uint32_t data_size = realS * dAlign;
        LocalTensor<T2> dqResTensor = postBuf[postPingpong].Get<T2>();
        uint32_t srcOffset =
            runInfo.taskPingPong * CUBE_BASEM * HEAD_DIM_ALIGN + i * s1Size * constInfo.commonConstInfo.dSize;
        WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
        inParams.blockCount = static_cast<uint16_t>(realS);
        DataCopyPad(dqResTensor, dqWorkSpaceGm[srcOffset], inParams, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        Muls(dqResTensor, dqResTensor, constInfo.scaleValue, data_size);
        LocalTensor<T1> dqCastTensor = postBuf[postPingpong].Get<T1>();
        Cast(dqCastTensor, dqResTensor, RoundMode::CAST_ROUND, data_size);
        SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        outParams.blockCount = realS;
        if constexpr (IS_TND) {
            dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.n2GD;
            outParams.dstStride = (constInfo.commonConstInfo.n2GD - constInfo.commonConstInfo.dSize) * sizeof(T1);
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.dSize;
                outParams.dstStride = (constInfo.commonConstInfo.dSize - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.n2GD;
                outParams.dstStride = (constInfo.commonConstInfo.n2GD - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.bN2GD;
                outParams.dstStride = (constInfo.commonConstInfo.bN2GD - constInfo.commonConstInfo.dSize) * sizeof(T1);
            }
        }
        DataCopyPad(dqGm[dstOffset], dqCastTensor, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
        postPingpong = 1 - postPingpong;
    }

    // Dk
    uint32_t dkLoopTimes = (runInfo.commonRunInfo.s2RealSize + sSize - 1) / sSize;
    uint32_t s2Size = (runInfo.commonRunInfo.s2RealSize + dkLoopTimes - 1) / dkLoopTimes;
    for (uint32_t i = 0; i < dkLoopTimes; i++) {
        uint32_t realS = (i == dkLoopTimes - 1) ? (runInfo.commonRunInfo.s2RealSize - i * s2Size) : s2Size;
        uint32_t data_size = realS * dAlign;
        LocalTensor<T2> dkResTensor = postBuf[postPingpong].Get<T2>();
        uint32_t srcOffset =
            runInfo.taskPingPong * CUBE_BASEN * HEAD_DIM_ALIGN + i * s2Size * constInfo.commonConstInfo.dSize;
        WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
        inParams.blockCount = static_cast<uint16_t>(realS);
        DataCopyPad(dkResTensor, dkWorkSpaceGm[srcOffset], inParams, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        Muls(dkResTensor, dkResTensor, constInfo.scaleValue, data_size);
        LocalTensor<T1> dkCastTensor = postBuf[postPingpong].Get<T1>();
        Cast(dkCastTensor, dkResTensor, RoundMode::CAST_ROUND, data_size);
        SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        outParams.blockCount = realS;
        if constexpr (IS_TND) {
            dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.n2D;
            outParams.dstStride = (constInfo.commonConstInfo.n2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.dSize;
                outParams.dstStride = (constInfo.commonConstInfo.dSize - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.n2D;
                outParams.dstStride = (constInfo.commonConstInfo.n2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.bN2D;
                outParams.dstStride = (constInfo.commonConstInfo.bN2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
            }
        }
        DataCopyPad(dkGm[dstOffset], dkCastTensor, outParams);
        SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
        postPingpong = 1 - postPingpong;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVec1(FagRunInfo& runInfo, FagConstInfo& constInfo)
{
    runInfo.commonRunInfo.halfS1RealSize =
        runInfo.commonRunInfo.s1RealSize < VECTOR_BASEM ? runInfo.commonRunInfo.s1RealSize : VECTOR_BASEM;
    runInfo.commonRunInfo.vecS1Offset = 0;
    WaitFlag<HardEvent::MTE3_MTE2>(maxSumMte2ToMte3[ubBufIdx]);
    CopyInMaxSumWithTbuf<T2, VECTOR_BASEM>(constInfo, runInfo, maxSumSfmgBuf[ubBufIdx], softmaxMaxGm, softmaxSumGm,
                                           sfmgWorkspaceGm);
    CopyInAttenMask<IS_ATTEN_MASK, VECTOR_BASEM, VECTOR_BASEN>(constInfo, runInfo, attenMaskInfo, dSOutBuf[ubBufIdx],
                                                               attenMaskU8Gm);
    if (unlikely(dropInfo.dropMaskOuter)) {
        GlobalTensor<uint8_t> dropMaskTensor = dropInfo.boolMode ? dropMaskWorkspaceGm : dropMaskGm;
        CopyInDropOuter<IS_DROP>(dSOutBuf[ubBufIdx], dSOutBuf[ubBufIdx], dropMaskTensor, runInfo.commonRunInfo,
                                 constInfo.commonConstInfo, dropInfo);
    }
    CopyInPseReuseBuf<T1, T2, IS_PSE>(constInfo, runInfo, pseInfo, pOutBuf[ubBufIdx], pseGm);
    SetFlag<HardEvent::MTE2_V>(mte2ToV[ubBufIdx]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::AsyncAndMte2CopyBeforeReCompute(
    FagRunInfo& runInfo, FagRunInfo& nextRunInfo, FagConstInfo& constInfo, bool isInnerBlockLast, bool isInterBlockLast)
{
    // Copy for next buffer
    auto nextUbBufIdx = ubBufIdx + 1 == UB_BUF_NUM ? 0 : ubBufIdx + 1;
    WaitFlag<HardEvent::MTE3_MTE2>(maxSumMte2ToMte3[nextUbBufIdx]); // Wait ubBufIdx + 1
    FagRunInfo actualRunInfo;
    if (!isInnerBlockLast) {
        actualRunInfo = runInfo;
        actualRunInfo.commonRunInfo.vecS1Offset += VECTOR_BASEM;
    } else if (!isInterBlockLast) {
        actualRunInfo = nextRunInfo;
        actualRunInfo.commonRunInfo.vecS1Offset = 0;
    }

    if (!isInnerBlockLast || !isInterBlockLast) {
        actualRunInfo.commonRunInfo.halfS1RealSize =
            (actualRunInfo.commonRunInfo.s1RealSize - actualRunInfo.commonRunInfo.vecS1Offset < VECTOR_BASEM) ?
                actualRunInfo.commonRunInfo.s1RealSize - actualRunInfo.commonRunInfo.vecS1Offset :
                VECTOR_BASEM;
        CopyInMaxSumWithTbuf<T2, VECTOR_BASEM>(constInfo, actualRunInfo, maxSumSfmgBuf[nextUbBufIdx], softmaxMaxGm,
                                               softmaxSumGm, sfmgWorkspaceGm);
        CopyInAttenMask<IS_ATTEN_MASK, VECTOR_BASEM, VECTOR_BASEN>(constInfo, actualRunInfo, attenMaskInfo,
                                                                   dSOutBuf[nextUbBufIdx], attenMaskU8Gm);
        if (unlikely(dropInfo.dropMaskOuter)) {
            GlobalTensor<uint8_t> dropMaskTensor = dropInfo.boolMode ? dropMaskWorkspaceGm : dropMaskGm;
            CopyInDropOuter<IS_DROP>(dSOutBuf[nextUbBufIdx], dSOutBuf[nextUbBufIdx], dropMaskTensor,
                                     actualRunInfo.commonRunInfo, constInfo.commonConstInfo, dropInfo);
        }
        CopyInPseReuseBuf<T1, T2, IS_PSE>(constInfo, actualRunInfo, pseInfo, pOutBuf[nextUbBufIdx], pseGm);
    }

    if (!isInnerBlockLast) {
        actualRunInfo.commonRunInfo.halfS1RealSize = VECTOR_BASEM; // 若当前不是最后一块，切分大小一定是整的
        actualRunInfo.commonRunInfo.vecS1Offset -= VECTOR_BASEM;
    }

    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(QKDXV_RES_READY[ubBufIdx]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyDpToTmpBuf(LocalTensor<T2>& mm1ResQueInTensor)
{
    LocalTensor<T2> DpDropTensor = postBuf[0].Get<T2>();
    DataCopy(DpDropTensor, mm1ResQueInTensor, VECTOR_BASEM * VECTOR_BASEN);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessVecSink(LocalTensor<T2>& mm2ResQueInTensor,
                                                                  FagRunInfo& runInfo, FagConstInfo& constInfo,
                                                                  bool isInnerBlockLast)
{
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    uint32_t maxSumTensorOffset = 0;
    uint32_t maxOffset = VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(T2);
    LocalTensor<T2> dsinkOutTensor = dsinkResOutQue[dsinkPingpong].AllocTensor<T2>();
    LocalTensor<T2> maxSumTensor = maxSumSfmgBuf[ubBufIdx].template Get<T2>();
    float sinkScale = sinkGm.GetValue(runInfo.sinkN1Idx);
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    LocalTensor<T2> sumTensor = maxSumTensor[maxSumTensorOffset];             // sum
    LocalTensor<T2> maxTensor = maxSumTensor[maxSumTensorOffset + maxOffset]; // max
    LocalTensor<T2> DpDropTensor = postBuf[0].Get<T2>();

    CalculateSink<T2, static_cast<uint16_t>(VECTOR_BASEN)>(dsinkOutTensor, mm2ResQueInTensor, DpDropTensor, maxTensor,
                                                           sumTensor, sinkScale,
                                                           static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                                                           static_cast<uint16_t>(runInfo.commonRunInfo.s2RealSize));
    dsinkResOutQue[dsinkPingpong].EnQue(dsinkOutTensor);
    dsinkResOutQue[dsinkPingpong].DeQue<T2>();
    SetFlag<HardEvent::V_S>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    WaitFlag<HardEvent::V_S>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    dsinkAccu += dsinkOutTensor.GetValue(0);
    if (isInnerBlockLast) {
        LocalTensor<float> dsinkResTensor = dsinkResBuf[sinkPingpong].Get<float>();
        dsinkResTensor.SetValue(0, dsinkAccu);
        SetFlag<HardEvent::S_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        WaitFlag<HardEvent::S_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        DataCopyPad(dsinkWorkSpaceGm[runInfo.dsinkWorkSpaceOffset], dsinkResTensor,
                    {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
        dsinkAccu = 0.0;
        sinkPingpong = 1 - sinkPingpong;
    }
    dsinkResOutQue[dsinkPingpong].FreeTensor(dsinkOutTensor);
    dsinkPingpong = 1 - dsinkPingpong;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ReCompute(LocalTensor<T2>& mm1ResQueInTensor,
                                                             LocalTensor<T2>& mm2ResQueInTensor, FagRunInfo& runInfo,
                                                             FagConstInfo& constInfo, bool isInnerBlockFirst,
                                                             bool isInnerBlockLast, bool isInterBlockLast)
{
    ///////////////////////////////////////////////////////////////
    // VF: pse + attenMask + muls + simpleSoftmax copyIn+calculate
    ///////////////////////////////////////////////////////////////
    if (!dropInfo.dropMaskOuter) {
        GenDropMask<IS_DROP>(dropMaskBuf, dropmaskIndexVecBuf, runInfo.commonRunInfo, constInfo.commonConstInfo,
                             dropInfo);
    }
    CalculateDropout<T2, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfo, mm1ResQueInTensor, mm1ResQueInTensor,
                                                dropMaskBuf);

    if (unlikely(constInfo.isSink)) {
        CopyDpToTmpBuf(mm1ResQueInTensor); // dp
    }
    if (!isInnerBlockLast || !isInterBlockLast)
        SetFlag<HardEvent::MTE2_V>(mte2ToV[(ubBufIdx + 1) % UB_BUF_NUM]);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV[ubBufIdx]);
    float deqScaleQK = IS_FP8_INPUT ? runInfo.quantScaleInfo.deqScaleQValue * runInfo.quantScaleInfo.deqScaleKValue : 1;
    float deqScaleDyV =
        IS_FP8_INPUT ? runInfo.quantScaleInfo.deqScaleDyValue * runInfo.quantScaleInfo.deqScaleVValue : 1;
    CalculatePseMulsSelSimpleSoftMaxWithTbuf<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DETER_OLD(DETER_SPARSE_TYPE),
                                             VECTOR_BASEM, VECTOR_BASEN>(
        constInfo, runInfo, pseInfo, attenMaskInfo, maxSumSfmgBuf[ubBufIdx], dSOutBuf[ubBufIdx], mm2ResQueInTensor,
        mm2ResQueInTensor, mm1ResQueInTensor, mm1ResQueInTensor, pseSlope, pOutBuf[ubBufIdx], deqScaleQK, deqScaleDyV);

    if (unlikely(constInfo.isSink)) {
        ProcessVecSink(mm2ResQueInTensor, runInfo, constInfo, isInnerBlockLast);
    }

    // p dropout
    CalculateDropout<T2, IS_DROP, VECTOR_BASEN>(constInfo, runInfo, dropInfo, mm2ResQueInTensor, mm2ResQueInTensor,
                                                dropMaskBuf);

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) { // 确定性计算尾块脏数据补零
        if (!constInfo.deterConstInfo.noNeedDeter) {
            if (runInfo.commonRunInfo.halfS1RealSize != VECTOR_BASEM) {
                Duplicate<T2>(mm1ResQueInTensor[runInfo.commonRunInfo.halfS1RealSize * VECTOR_BASEN], 0,
                              (VECTOR_BASEM - runInfo.commonRunInfo.halfS1RealSize) * VECTOR_BASEN);
            }
        }
    }

    // input type fp32, no post, mov muls here
    if constexpr (IS_FP32_INPUT) {
        /// TODO: 更高效的乘法实现：MulsWithStaticLength
        Muls(mm1ResQueInTensor, mm1ResQueInTensor, constInfo.scaleValue, VECTOR_BASEM * VECTOR_BASEN);
    }

    ///////////////////////////////////////////////////////////////
    // VF: cast + nd2nz
    ///////////////////////////////////////////////////////////////
    LocalTensor<uint8_t> selrIndexesTensor;
    if constexpr (IS_FP8_INPUT) {
        selrIndexesTensor = vselrIndexesBuf.Get<uint8_t>();
    }
    LocalTensor<DsType> vecDsOutBuffer = dSOutBuf[ubBufIdx].template Get<DsType>();
    CastTransdataDeconflict<DsType, T2, VECTOR_BASEN>(vecDsOutBuffer, mm1ResQueInTensor, selrIndexesTensor,
                                                      VECTOR_BASEM);
    if (isInnerBlockFirst) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(DS_RELEASE);
    }
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (!constInfo.deterConstInfo.noNeedDeter) {
            CopyUB2L1Deter(runInfo, L1DsTensor, vecDsOutBuffer,
                           runInfo.commonRunInfo.vecS1Offset); // needdeter流程将dsout结果无效行数据重置为0
            uint32_t vecOffestNum = (runInfo.commonRunInfo.s1RealSize + VECTOR_BASEM - 1) / VECTOR_BASEM;
            uint32_t innerLoopNum = (CUBE_BASEM + this->VECTOR_BASEM - 1) / this->VECTOR_BASEM;
            if (vecOffestNum != innerLoopNum && isInnerBlockLast) // 无效基本块数据置零
            {
                pipe_barrier(PIPE_ALL);
                Duplicate<T1>(vecDsOutBuffer, 0, (VECTOR_BASEM + 1) * VECTOR_BASEN);
                for (uint32_t i = vecOffestNum; i < innerLoopNum; i++) {
                    uint32_t scmOffset = i * VECTOR_BASEM; // s1方向按VECTOR_BASEM大小偏移
                    CopyUB2L1Deter(runInfo, L1DsTensor, vecDsOutBuffer, scmOffset);
                }
            }
        } else {
            CopyUB2L1(runInfo, L1DsTensor, vecDsOutBuffer);
        }
    } else {
        CopyUB2L1(runInfo, L1DsTensor, vecDsOutBuffer);
    }
    if (isInnerBlockLast) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(DS_READY);
    }
    LocalTensor<T1> vecDpOutBuffer = pOutBuf[ubBufIdx].template Get<T1>();
    CastTransdataDeconflict<T1, T2, VECTOR_BASEN>(vecDpOutBuffer, mm2ResQueInTensor, selrIndexesTensor, VECTOR_BASEM);
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(QKDXV_RES_RELEASE[ubBufIdx]);
    if (isInnerBlockFirst) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(POUT_RELEASE);
    }
    CopyUB2L1(runInfo, L1PTensor, vecDpOutBuffer);
    if (isInnerBlockLast) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(POUT_READY);
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename CopyType>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1(FagRunInfo& runInfo, LocalTensor<CopyType>& dstTensor,
                                                             LocalTensor<CopyType>& srcTensor)
{
    constexpr uint32_t FRACTAL_NZ_C0_SIZE_CT = 32 / sizeof(CopyType);
    constexpr uint32_t INPUT_BLOCK_NUM_CT = 32 / sizeof(CopyType);
    /// NOTE: 对于FP32,FP4类型，下方的下标计算需要额外考虑
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    uint32_t scmOffset = runInfo.commonRunInfo.vecS1Offset * FRACTAL_NZ_C0_SIZE_CT;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (runInfo.commonRunInfo.s2RealSize + FRACTAL_NZ_C0_SIZE_CT - 1) / FRACTAL_NZ_C0_SIZE_CT;
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        dataCopyParams.blockCount = CUBE_BASEN / FRACTAL_NZ_C0_SIZE_CT;
    }
    dataCopyParams.blockLen =
        (uint16_t)(runInfo.commonRunInfo.halfS1RealSize * FRACTAL_NZ_C0_SIZE_CT / INPUT_BLOCK_NUM_CT);
    dataCopyParams.srcStride = (uint16_t)((VECTOR_BASEM + 1 - runInfo.commonRunInfo.halfS1RealSize) *
                                          FRACTAL_NZ_C0_SIZE_CT / INPUT_BLOCK_NUM_CT);
    dataCopyParams.dstStride =
        (CUBE_BASEM - runInfo.commonRunInfo.halfS1RealSize) * FRACTAL_NZ_C0_SIZE_CT / INPUT_BLOCK_NUM_CT;
    event_t v2mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(v2mte3);
    WaitFlag<HardEvent::V_MTE3>(v2mte3);
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename CopyType>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::CopyUB2L1Deter(FagRunInfo& runInfo, LocalTensor<CopyType>& dstTensor,
                                                                  LocalTensor<CopyType>& srcTensor, uint32_t scmOffset)
{
    constexpr uint32_t FRACTAL_NZ_C0_SIZE_CT = 32 / sizeof(CopyType);
    constexpr uint32_t INPUT_BLOCK_NUM_CT = 32 / sizeof(CopyType);
    /// NOTE: 对于FP32,FP4类型，下方的下标计算需要额外考虑
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    scmOffset = scmOffset * FRACTAL_NZ_C0_SIZE_CT;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = CUBE_BASEN / FRACTAL_NZ_C0_SIZE_CT;
    dataCopyParams.blockLen = (uint16_t)(VECTOR_BASEM * FRACTAL_NZ_C0_SIZE_CT / INPUT_BLOCK_NUM_CT);
    dataCopyParams.srcStride =
        (uint16_t)((VECTOR_BASEM + 1 - VECTOR_BASEM) * FRACTAL_NZ_C0_SIZE_CT / INPUT_BLOCK_NUM_CT);
    dataCopyParams.dstStride = (CUBE_BASEM - VECTOR_BASEM) * FRACTAL_NZ_C0_SIZE_CT / INPUT_BLOCK_NUM_CT;
    event_t v2mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    SetFlag<HardEvent::V_MTE3>(v2mte3);
    WaitFlag<HardEvent::V_MTE3>(v2mte3);
    DataCopy(dstTensor[scmOffset], srcTensor, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::AsyncAfterReCompute(bool isInnerBlockLast, bool isInterBlockLast)
{
    if (!isInnerBlockLast || !isInterBlockLast) {
        SetFlag<HardEvent::MTE3_MTE2>(maxSumMte2ToMte3[ubBufIdx]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::WriteOffsetToGM(FagConstInfo& constInfo, int64_t queryGmOffset,
                                                                   int64_t keyGmOffset, int64_t valueGmOffset,
                                                                   int64_t computeBlockIdx, int64_t remainLoopNum)
{
    // 将本核当前块的 q/k/v GM offset 写入 deterOffsetGm，供 DeterCompute 阶段读取
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

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterCompute(LocalTensor<T2>& mm1ResTensor,
                                                                LocalTensor<T2>& mm2ResTensor, FagConstInfo& constInfo,
                                                                bool dqIsNeedDeter, bool dkDvIsNeedDeter,
                                                                bool isLastSort, int64_t computeBlockIdx,
                                                                int64_t remainLoopNum, int16_t deterPpFlag)
{
    if (likely(constInfo.deterConstInfo.noNeedDeter)) {
        if (remainLoopNum > 1) {
            SyncAll();
        }
    } else {
        if (dqIsNeedDeter && !dkDvIsNeedDeter) {
            // 该轮次只有 dq 需要做 deter
            DeterComputeDq(constInfo, isLastSort, computeBlockIdx, remainLoopNum, deterPpFlag);
            isFirstDeter = false;
        } else if (!dqIsNeedDeter && dkDvIsNeedDeter) {
            // 该轮次只有 dk dv 需要做 deter
            DeterComputeDkv(mm1ResTensor, mm2ResTensor, constInfo, isLastSort, computeBlockIdx, remainLoopNum,
                            deterPpFlag);
            isFirstDeter = false;
        } else if (dqIsNeedDeter && dkDvIsNeedDeter) {
            // 该轮次 dq dk dv 都需要做 deter
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
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterComputeDq(FagConstInfo& constInfo, bool isLastSort,
                                                                  int64_t computeBlockIdx, int64_t remainLoopNum,
                                                                  int16_t deterPpFlag)
{
    SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
    SyncAll<true, syncAllConfigMte2ToMte2>();

    LocalTensor<int64_t> deterOffsetTensor = deterOffsetBuf.Get<int64_t>();
    if (!isFirstDeter) {
        WaitFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2); // sclar已做完，可以开始gm搬运
    }
    DataCopy(deterOffsetTensor,
             deterOffsetGm[computeBlockIdx * constInfo.deterConstInfo.usedCubeCoreNum * INT64_BLOCK_NUM * OFFSET_NUM],
             {1, static_cast<uint16_t>(constInfo.deterConstInfo.usedCubeCoreNum * OFFSET_NUM), 0, 0});
    SetFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar); // 搬运完，可以进行scaler

    // 非last基本块处理的都是上一轮mm的结果， 最后一次循环需要补充处理本轮的mm结果
    int16_t bufId = isLastSort ? deterPpFlag : (1 - deterPpFlag);
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * 2) * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dqEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DQ_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;

    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS1Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqDstStride);

    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(DETER_DQ_UB_SIZE / (constInfo.deterConstInfo.dqEachVectorSize * sizeof(T2)));
    uint8_t loopTimes =
        static_cast<uint8_t>((constInfo.deterConstInfo.usedCubeCoreNum + eachLoopBlockCount - 1) / eachLoopBlockCount);
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;

    AscendC::SetAtomicAdd<T2>();
    WaitFlag<HardEvent::MTE2_S>(constInfo.deterConstInfo.eventIDMte2ToScalar);

    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<T2> dqDeterBuf = deterInOutQue.AllocTensor<T2>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1) ?
                                        eachLoopBlockCount :
                                        constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dqDeterBuf, deterGm[dqSrcOfs], dataCopyParams);
        }
        dqSrcOfs += eachLoopBlockCount * BASE_DQ_SIZE;
        deterInOutQue.EnQue(dqDeterBuf);
        deterInOutQue.DeQue<T2>();

        eachLoopStart = loopIdx * eachLoopBlockCount;
        eachLoopEnd = (loopIdx == loopTimes - 1) ?
                          constInfo.deterConstInfo.usedCubeCoreNum :
                          Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            dqOffsetArr[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM);
        }
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dqOffsetArr[cIx] == OUTINDEX) {
                continue;
            }
            dqOffsetArr[cIx] += constInfo.deterConstInfo.deterVecCoreS1Offset;
            // todo fp32
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dqWorkSpaceGm[dqOffsetArr[cIx]],
                                     dqDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dqEachVectorSize],
                                     dataCopyPadParams);
            }
            PipeBarrier<PIPE_MTE3>(); // 保证上个核的基本块被搬出去再搬下一个
        }
        deterInOutQue.FreeTensor(dqDeterBuf);
    }
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(
            constInfo.deterConstInfo.eventIDScalarToMte2); // 上面的scaler操作已做完，通知deteroffset可以继续搬运？
    }
    if (remainLoopNum > NUM_TWO) {
        SyncAll<true, syncAllConfigMte3ToMte3>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterComputeDkv(LocalTensor<T2>& dkDeterBuf,
                                                                   LocalTensor<T2>& dvDeterBuf, FagConstInfo& constInfo,
                                                                   bool isLastSort, int64_t computeBlockIdx,
                                                                   int64_t remainLoopNum, int16_t deterPpFlag)
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
        dkOffsetArr[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM);
        dvOffsetArr[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM * NUM_TWO);
    }

    int16_t bufId = isLastSort ? deterPpFlag : (1 - deterPpFlag);
    uint32_t dqSrcOfs = bufId * (BASE_DQ_SIZE + BASE_DKV_SIZE * 2) * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dkSrcOfs = dqSrcOfs + BASE_DQ_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    uint32_t dvSrcOfs = dkSrcOfs + BASE_DKV_SIZE * constInfo.deterConstInfo.usedCubeCoreNum;
    dqSrcOfs += vBlockIdx * constInfo.deterConstInfo.dqEachVectorSize;
    dkSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;
    dvSrcOfs += vBlockIdx * constInfo.deterConstInfo.dkvEachVectorSize;

    AscendC::SetAtomicAdd<T2>();
    WriteDataToDkv(dkDeterBuf, dvDeterBuf, constInfo, dkSrcOfs, dvSrcOfs);
    AscendC::SetAtomicNone();

    if (remainLoopNum > 0) {
        SetFlag<HardEvent::S_MTE2>(constInfo.deterConstInfo.eventIDScalarToMte2);
    }
    if (remainLoopNum > NUM_TWO) {
        SyncAll();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::DeterComputeDqkv(LocalTensor<T2>& dkDeterBuf,
                                                                    LocalTensor<T2>& dvDeterBuf,
                                                                    FagConstInfo& constInfo, bool isLastSort,
                                                                    int64_t computeBlockIdx, int64_t remainLoopNum,
                                                                    int16_t deterPpFlag)
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
        dqOffsetArr[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM);
        dkOffsetArr[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM);
        dvOffsetArr[cIx] = deterOffsetTensor.GetValue(cIx * INT64_BLOCK_NUM * OFFSET_NUM + INT64_BLOCK_NUM * NUM_TWO);
    }

    int16_t bufId = isLastSort ? deterPpFlag : (1 - deterPpFlag);
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
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqDstStride);

    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(DETER_DQ_UB_SIZE / (constInfo.deterConstInfo.dqEachVectorSize * sizeof(T2)));
    uint8_t loopTimes =
        static_cast<uint8_t>((constInfo.deterConstInfo.usedCubeCoreNum + eachLoopBlockCount - 1) / eachLoopBlockCount);
    uint8_t eachLoopStart = 0;
    uint8_t eachLoopEnd = 0;

    AscendC::SetAtomicAdd<T2>();
    for (uint8_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
        LocalTensor<T2> dqDeterBuf = deterInOutQue.AllocTensor<T2>();
        dataCopyParams.blockCount = (loopIdx < loopTimes - 1) ?
                                        eachLoopBlockCount :
                                        constInfo.deterConstInfo.usedCubeCoreNum - loopIdx * eachLoopBlockCount;
        if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
            DataCopy(dqDeterBuf, deterGm[dqSrcOfs], dataCopyParams);
        }
        dqSrcOfs += eachLoopBlockCount * BASE_DQ_SIZE;
        deterInOutQue.EnQue(dqDeterBuf);
        deterInOutQue.DeQue<T2>();
        eachLoopStart = loopIdx * eachLoopBlockCount;
        eachLoopEnd = (loopIdx == loopTimes - 1) ?
                          constInfo.deterConstInfo.usedCubeCoreNum :
                          Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);

        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dqOffsetArr[cIx] == OUTINDEX) {
                continue;
            }
            dqOffsetArr[cIx] += constInfo.deterConstInfo.deterVecCoreS1Offset;
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dqWorkSpaceGm[dqOffsetArr[cIx]],
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
    if (remainLoopNum > NUM_TWO) {
        SyncAll();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::WriteDataToDkv(LocalTensor<T2>& dkDeterBuf,
                                                                  LocalTensor<T2>& dvDeterBuf, FagConstInfo& constInfo,
                                                                  uint32_t dkSrcOfs, uint32_t dvSrcOfs)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.deterConstInfo.dkvEachVectorSize / FLOAT_BLOCK_SIZE;
    dataCopyParams.srcStride = BASE_DKV_SIZE / FLOAT_BLOCK_SIZE - dataCopyParams.blockLen;
    dataCopyParams.dstStride = 0;

    DataCopyExtParams dataCopyPadParams;
    dataCopyPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSize * sizeof(T2));
    dataCopyPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDqkSrcStride);
    dataCopyPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDkDstStride);

    DataCopyExtParams dataCopyDvPadParams;
    dataCopyDvPadParams.blockCount = static_cast<uint16_t>(constInfo.deterConstInfo.eachVecCoreS2Offset);
    dataCopyDvPadParams.blockLen = static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(T2));
    dataCopyDvPadParams.srcStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvSrcStride);
    dataCopyDvPadParams.dstStride = static_cast<uint32_t>(constInfo.deterConstInfo.deterDvDstStride);

    uint8_t eachLoopBlockCount =
        static_cast<uint8_t>(DETER_DKV_UB_SIZE / (constInfo.deterConstInfo.dkvEachVectorSize * sizeof(T2)));
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
        eachLoopEnd = (loopIdx == loopTimes - 1) ?
                          constInfo.deterConstInfo.usedCubeCoreNum :
                          Min((loopIdx + 1) * eachLoopBlockCount, constInfo.deterConstInfo.usedCubeCoreNum);
        for (uint16_t cIx = eachLoopStart; cIx < eachLoopEnd; cIx++) {
            if (dkOffsetArr[cIx] == OUTINDEX && dvOffsetArr[cIx] == OUTINDEX) {
                continue;
            }
            dkOffsetArr[cIx] += constInfo.deterConstInfo.deterDkVecCoreS2Offset;
            dvOffsetArr[cIx] += constInfo.deterConstInfo.deterDvVecCoreS2Offset;
            if (vBlockIdx < constInfo.deterConstInfo.usedVectorCoreNum) {
                AscendC::DataCopyPad(dkWorkSpaceGm[dkOffsetArr[cIx]],
                                     dkDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                     dataCopyPadParams);
                AscendC::DataCopyPad(dvWorkSpaceGm[dvOffsetArr[cIx]],
                                     dvDeterBuf[(cIx - eachLoopStart) * constInfo.deterConstInfo.dkvEachVectorSize],
                                     dataCopyDvPadParams);
            }
            PipeBarrier<PIPE_MTE3>();
        }
        if (loopIdx < loopTimes - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(constInfo.deterConstInfo.eventIDMte3ToMte2);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, uint8_t MM_IDX>
__aicore__ inline void FAGBlockVec<TEMPLATE_ARGS>::ProcessMulsAndCast(GlobalTensor<T>& workSpaceGm,
                                                                      FagConstInfo& constInfo, FagRunInfo& runInfo)
{
    uint32_t dAlign = AlignTo32(constInfo.commonConstInfo.dSize * sizeof(T1)) / sizeof(T1); // 对齐到一个block
    uint32_t sSize = POST_UB_SIZE / sizeof(T2) / dAlign;
    uint64_t dqGmOffset = runInfo.dqGmOffset;
    uint64_t dkGmOffset = runInfo.dkGmOffset;
    // uint64_t dvGmOffset = runInfo.dvGmOffset;
    uint64_t dstOffset;
    DataCopyExtParams inParams;
    inParams.blockLen = constInfo.commonConstInfo.dSize * sizeof(T2);
    inParams.dstStride = (dAlign - constInfo.commonConstInfo.dSize) * sizeof(T2) / 32;
    inParams.srcStride = 0;
    DataCopyExtParams outParams;
    outParams.blockLen = constInfo.commonConstInfo.dSize * sizeof(T1);
    outParams.srcStride = 0;

    if constexpr (MM_IDX == DQ_IDX) {
        uint32_t dqLoopTimes = (runInfo.commonRunInfo.s1RealSize + sSize - 1) / sSize;
        uint32_t s1Size = (runInfo.commonRunInfo.s1RealSize + dqLoopTimes - 1) / dqLoopTimes;
        for (uint32_t i = 0; i < dqLoopTimes; i++) {
            uint32_t realS = (i == dqLoopTimes - 1) ? (runInfo.commonRunInfo.s1RealSize - i * s1Size) : s1Size;
            uint32_t data_size = realS * dAlign;
            LocalTensor<T2> dqResTensor = postBuf[postPingpong].Get<T2>();
            uint32_t srcOffset = cBlockIdx * constInfo.commonConstInfo.s1Size * HEAD_DIM_ALIGN +
                                 runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN +
                                 i * s1Size * constInfo.commonConstInfo.dSize;
            WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
            inParams.blockCount = static_cast<uint16_t>(realS);
            DataCopyPad(dqResTensor, workSpaceGm[srcOffset], inParams, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            Muls(dqResTensor, dqResTensor, constInfo.scaleValue, data_size);
            LocalTensor<T1> dqCastTensor = postBuf[postPingpong].Get<T1>();
            Cast(dqCastTensor, dqResTensor, RoundMode::CAST_ROUND, data_size);
            SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            outParams.blockCount = realS;
            if constexpr (IS_TND) {
                dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.n2GD;
                outParams.dstStride = (constInfo.commonConstInfo.n2GD - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else {
                if (constInfo.commonConstInfo.layoutType == BNGSD) {
                    dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.dSize;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.dSize - constInfo.commonConstInfo.dSize) * sizeof(T1);
                } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                    dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.n2GD;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.n2GD - constInfo.commonConstInfo.dSize) * sizeof(T1);
                } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                    dstOffset = dqGmOffset + i * s1Size * constInfo.commonConstInfo.bN2GD;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.bN2GD - constInfo.commonConstInfo.dSize) * sizeof(T1);
                }
            }
            DataCopyPad(dqGm[dstOffset], dqCastTensor, outParams);
            SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
            postPingpong = 1 - postPingpong;
        }
    } else if constexpr (MM_IDX == DK_IDX) {
        uint32_t dkLoopTimes = (runInfo.commonRunInfo.s2RealSize + sSize - 1) / sSize;
        uint32_t s2Size = (runInfo.commonRunInfo.s2RealSize + dkLoopTimes - 1) / dkLoopTimes;
        for (uint32_t i = 0; i < dkLoopTimes; i++) {
            uint32_t realS = (i == dkLoopTimes - 1) ? (runInfo.commonRunInfo.s2RealSize - i * s2Size) : s2Size;
            uint32_t data_size = realS * dAlign;
            LocalTensor<T2> dkResTensor = postBuf[postPingpong].Get<T2>();
            uint32_t srcOffset = cBlockIdx * constInfo.commonConstInfo.s2Size * HEAD_DIM_ALIGN +
                                 runInfo.s2oIdx * CUBE_BASEN * HEAD_DIM_ALIGN +
                                 i * s2Size * constInfo.commonConstInfo.dSize;
            WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
            inParams.blockCount = static_cast<uint16_t>(realS);
            DataCopyPad(dkResTensor, workSpaceGm[srcOffset], inParams, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            Muls(dkResTensor, dkResTensor, constInfo.scaleValue, data_size);
            LocalTensor<T1> dkCastTensor = postBuf[postPingpong].Get<T1>();
            Cast(dkCastTensor, dkResTensor, RoundMode::CAST_ROUND, data_size);
            SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            outParams.blockCount = realS;
            if constexpr (IS_TND) {
                dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.n2D;
                outParams.dstStride = (constInfo.commonConstInfo.n2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else {
                if (constInfo.commonConstInfo.layoutType == BNGSD) {
                    dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.dSize;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.dSize - constInfo.commonConstInfo.dSize) * sizeof(T1);
                } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                    dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.n2D;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.n2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
                } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                    dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.bN2D;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.bN2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
                }
            }
            DataCopyPad(dkGm[dstOffset], dkCastTensor, outParams);
            SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
            postPingpong = 1 - postPingpong;
        }
    } else {
        uint32_t dvLoopTimes = (runInfo.commonRunInfo.s2RealSize + sSize - 1) / sSize;
        uint32_t s2Size = (runInfo.commonRunInfo.s2RealSize + dvLoopTimes - 1) / dvLoopTimes;
        for (uint32_t i = 0; i < dvLoopTimes; i++) {
            uint32_t realS = (i == dvLoopTimes - 1) ? (runInfo.commonRunInfo.s2RealSize - i * s2Size) : s2Size;
            uint32_t data_size = realS * dAlign;
            LocalTensor<T2> dvResTensor = postBuf[postPingpong].Get<T2>();
            uint32_t srcOffset = cBlockIdx * constInfo.commonConstInfo.s2Size * HEAD_DIM_ALIGN +
                                 runInfo.s2oIdx * CUBE_BASEN * HEAD_DIM_ALIGN +
                                 i * s2Size * constInfo.commonConstInfo.dSize;
            WaitFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
            inParams.blockCount = static_cast<uint16_t>(realS);
            DataCopyPad(dvResTensor, workSpaceGm[srcOffset], inParams, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            LocalTensor<T1> dvCastTensor = postBuf[postPingpong].Get<T1>();
            Cast(dvCastTensor, dvResTensor, RoundMode::CAST_ROUND, data_size);
            SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            outParams.blockCount = realS;
            if constexpr (IS_TND) {
                dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.n2D;
                outParams.dstStride = (constInfo.commonConstInfo.n2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
            } else {
                if (constInfo.commonConstInfo.layoutType == BNGSD) {
                    dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.dSize;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.dSize - constInfo.commonConstInfo.dSize) * sizeof(T1);
                } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                    dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.n2D;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.n2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
                } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                    dstOffset = dkGmOffset + i * s2Size * constInfo.commonConstInfo.bN2D;
                    outParams.dstStride =
                        (constInfo.commonConstInfo.bN2D - constInfo.commonConstInfo.dSize) * sizeof(T1);
                }
            }
            DataCopyPad(dvGm[dstOffset], dvCastTensor, outParams);
            SetFlag<HardEvent::MTE3_MTE2>(postMte2ToMte3[postPingpong]);
            postPingpong = 1 - postPingpong;
        }
    }
}

TEMPLATES_DEF
class FAGBlockVecDummy {
public:
    __aicore__ inline void SetVecBlockParams(TPipe* pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, AttenMaskInfo& attenMaskInfo,
                                             PseInfo& pseInfo, DropMaskInfo& dropInfo) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR pseShift, GM_ADDR dropMask,
                                            GM_ADDR attenMask, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR sink, GM_ADDR dsink,
                                            GM_ADDR workspace) {};
    __aicore__ inline void InitUbBuffer() {};
    __aicore__ inline void InitAsync() {};
    __aicore__ inline void FreeAsync(FagConstInfo& constInfo) {};
    __aicore__ inline void PostCompute(FagRunInfo& runInfo, FagConstInfo& constInfo, GlobalTensor<float>& dqWorkSpaceGm,
                                       GlobalTensor<float>& dkWorkSpaceGm) {};
    __aicore__ inline void ProcessVec1(FagRunInfo& runInfo, FagConstInfo& constInfo) {};
    __aicore__ inline void ProcessVec2(FagRunInfo& runInfo, FagRunInfo& nextRunInfo, FagConstInfo& constInfo,
                                       bool isInterBlockLast, LocalTensor<T2>& mm1ResTensor,
                                       LocalTensor<T2>& mm2ResTensor, int8_t innerLoopNum, int8_t innerLoopIdx,
                                       int8_t baseUbBufIdx) {};
    __aicore__ inline void AsyncAndMte2CopyBeforeReCompute(FagRunInfo& runInfo, FagRunInfo& nextRunInfo,
                                                           FagConstInfo& constInfo, bool isInnerBlockLast,
                                                           bool isInterBlockLast) {};
    __aicore__ inline void ReCompute(LocalTensor<T2>& mm1ResQueInTensor, LocalTensor<T2>& mm2ResQueInTensor,
                                     FagRunInfo& runInfo, FagConstInfo& constInfo, bool isInnerBlockFirst,
                                     bool isInnerBlockLast, bool isInterBlockLast) {};
    __aicore__ inline void AsyncAfterReCompute(bool isInnerBlockLast, bool isInterBlockLast) {};
    __aicore__ inline void WriteOffsetToGM(FagConstInfo& constInfo, int64_t queryGmOffset, int64_t keyGmOffset,
                                           int64_t valueGmOffset, int64_t computeBlockIdx, int64_t remainLoopNum) {};
    __aicore__ inline void DeterCompute(LocalTensor<T2>& mm1ResTensor, LocalTensor<T2>& mm2ResTensor,
                                        FagConstInfo& constInfo, bool dqIsNeedDeter, bool dkDvIsNeedDeter,
                                        bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                        int16_t deterPpFlag) {};
    __aicore__ inline void DeterComputeDq(FagConstInfo& constInfo, bool isLastSort, int64_t computeBlockIdx,
                                          int64_t remainLoopNum, int16_t deterPpFlag) {};
    __aicore__ inline void DeterComputeDkv(LocalTensor<T2>& dkBuf, LocalTensor<T2>& dvBuf, FagConstInfo& constInfo,
                                           bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                           int16_t deterPpFlag) {};
    __aicore__ inline void DeterComputeDqkv(LocalTensor<T2>& dkBuf, LocalTensor<T2>& dvBuf, FagConstInfo& constInfo,
                                            bool isLastSort, int64_t computeBlockIdx, int64_t remainLoopNum,
                                            int16_t deterPpFlag) {};
    __aicore__ inline void WriteDataToDkv(LocalTensor<T2>& dkBuf, LocalTensor<T2>& dvBuf, FagConstInfo& constInfo,
                                          uint32_t dkSrcOfs, uint32_t dvSrcOfs) {};
    __aicore__ inline void CopyDpToTmpBuf(LocalTensor<T2>& mm1ResQueInTensor) {};
    __aicore__ inline void ProcessVecSink(LocalTensor<T2>& mm2ResQueInTensor, FagRunInfo& runInfo,
                                          FagConstInfo& constInfo, bool isInnerBlockLast) {};
    template <typename T, uint8_t MM_IDX>
    __aicore__ inline void ProcessMulsAndCast(GlobalTensor<T>& workSpaceGm, FagConstInfo& constInfo,
                                              FagRunInfo& runInfo) {};
    template <typename CopyType>
    __aicore__ inline void CopyUB2L1(FagRunInfo& runInfo, LocalTensor<CopyType>& dstTensor,
                                     LocalTensor<CopyType>& srcTensor) {};

    template <typename CopyType>
    __aicore__ inline void CopyUB2L1Deter(FagRunInfo& runInfo, LocalTensor<CopyType>& dstTensor,
                                          LocalTensor<CopyType>& srcTensor, uint32_t scmOffset) {};
};

} // namespace FagBaseApi

#endif
