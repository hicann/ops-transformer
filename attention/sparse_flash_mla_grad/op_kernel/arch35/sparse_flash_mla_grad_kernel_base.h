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
 * \file flash_attention_score_grad_kernel_base.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_KERNEL_BASE_H
#define SPARSE_FLASH_MLA_GRAD_KERNEL_BASE_H

#include "sparse_flash_mla_grad_common.h"

namespace SfagBaseApi {

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class SparseFlashMlaGradKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out,
                                GM_ADDR ori_sparse_indices, GM_ADDR cmp_sparse_indices, GM_ADDR cu_seqlens_q,
                                GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv, GM_ADDR seqused_q,
                                GM_ADDR seqused_ori_kv, GM_ADDR seqused_cmp_kv, GM_ADDR cmp_residual_kv,
                                GM_ADDR ori_topk_length, GM_ADDR cmp_topk_length, GM_ADDR softmax_lse, GM_ADDR sinks,
                                GM_ADDR metadata, GM_ADDR dq, GM_ADDR dori_kv, GM_ADDR dcmp_kv, GM_ADDR dsinks,
                                GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1, GM_ADDR workspace,
                                SFagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void InitCVCommonBuffer();
    __aicore__ inline void InitCVCommonGlobalBuffer(GM_ADDR workspace);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetOptionalInfo();
    template <const bool IS_ORI_KV = true>
    __aicore__ inline void SetRunInfo(FagRunInfo &runInfo, FagRunInfo &preRunInfo, int64_t taskId, int64_t sTaskId,
                                      int64_t deterTaskId, int64_t kvOffset);
    template <const bool IS_ORI_KV = true>
    __aicore__ inline void SetDeterRunInfo(FagRunInfo &runInfo, int64_t sTaskId, int64_t deterTaskId,
                                           int64_t kvSeqOffset);
    __aicore__ inline void Process();
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetQueryOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetQueryRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDxOffset(FagRunInfo &runInfo);
    template <const bool IS_ORI_KV = true>
    __aicore__ inline int64_t GetKeyOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetValueOffset(FagRunInfo &runInfo);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);
    __aicore__ inline void GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int64_t &actSelBlkCount);
    __aicore__ inline void GetTndSeqLen(const int64_t t1Idx, int64_t &bIdx);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline ChildClass *GetDerived()
    {
        return static_cast<ChildClass *>(this);
    }

    constexpr static bool IS_D_NO_EQUAL = false;
    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static float FP8_MAX = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ? 57344 : 448;
    constexpr static uint32_t BITS_EACH_UINT64 = 64;
    constexpr static uint32_t MAX_BITS_IN_TILING = 32 * BITS_EACH_UINT64;
    constexpr static uint32_t INT64_BLOCK_NUM = 32 / sizeof(int64_t);
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = 128;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t CUBE_BASEK = 128;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_FP8 = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static int64_t OUTINDEX = -1;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;
    constexpr static uint32_t DETER_DKV_UB_SIZE = VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE);

    constexpr static bool IS_DQ_RES_EXCEED_UB = HEAD_DIM_ALIGN > VECTOR_BASEN;
    constexpr static bool IS_DKV_RES_EXCEED_UB =
        VECTOR_BASEN / CV_CORE_RATIO * HEAD_DIM_ALIGN > VECTOR_BASEM *VECTOR_BASEN;
    constexpr static bool IS_DQ_WRITE_UB = !IS_DQ_RES_EXCEED_UB;
    constexpr static bool IS_DK_WRITE_UB = !IS_DKV_RES_EXCEED_UB;
    constexpr static bool IS_DV_WRITE_UB = !IS_DKV_RES_EXCEED_UB;

protected:
    TPipe *pipe;

    // output global mmemory
    GlobalTensor<float> dqWorkSpaceGm, dOriKVWorkSpaceGm, dCmpKVWorkSpaceGm, mm4ResWorkSpaceGm, mm5ResWorkSpaceGm;
    GlobalTensor<INPUT_TYPE> selectedKWorkSpaceGm;
    // CV核间共享Buffer
    TBuf<> mm1ResBuf[2];
    TBuf<> mm2ResBuf[2];
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buf;

    GM_ADDR prefixNAddr;
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqOriKvlenAddr;
    GM_ADDR actualSeqCmpKvlenAddr;
    GM_ADDR oriTopkLengthAddr;
    GM_ADDR cmpTopkLengthAddr;

    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedSeqOriKvlenAddr;
    GM_ADDR usedSeqCmpKvlenAddr;
    GM_ADDR cmpResidualKVAddr;

    GM_ADDR metadataGm;

    uint32_t coreNum = 0;
    uint32_t vBlockIdx = 0;
    uint32_t cBlockIdx = 0;
    uint32_t vSubBlockIdx = 0;
    int64_t lastS2oCvDimIdx = -1; // 上一次的s2方向基本块idx
    int64_t lastBdimIdx = -1;     // 上一次的b方向基本块idx
    int64_t lastN2dimIdx = -1;    // 上一次的n2方向基本块idx
    uint8_t kvPingPong = 1;
    bool isLastLoop = false;
    int64_t s2CvBegin = 0;
    int64_t s2CvEnd = 0;
    int64_t actualCalcS1Token = 0; // 转换后实际计算使用的S1Token
    int64_t actualCalcS2Token = 0;
    int64_t curS1 = 0;
    int64_t curS2 = 0;
    int64_t curS3 = 0;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;
    int64_t usedS3 = 0;
    int64_t oriTopkLength = 0;
    int64_t cmpTopkLength = 0;

    int64_t usedT1Offset = 0;
    int64_t nextUsedT1Offset = 0;
    int64_t bIndex = 0;
    int64_t s1Index = 0;
    int64_t formerCoreProcessS1Num;
    SFagTilingType tilingData;
    FagConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;

    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    int64_t blkCntOffset = 0;
    int64_t oriKvOffset = 0;
    int64_t cmpKvOffset = 0;
    int64_t actualSelectedBlockCount = 0;
    int64_t actualOriSelectedBlockCount = 0;
    int64_t actualCmpSelectedBlockCount = 0;
    int64_t maxOriSeqKV = 0;
    int64_t maxCmpSeqKV = 0;

    int64_t usedT1Index = 0; // 用于负载均衡
    int64_t t1Index = 0;
    int64_t t2Index = 0;
    int64_t t3Index = 0;
    int64_t n2Index = 0;
    int64_t lastT1Index = -1;
    int64_t usedCoreNum = 0;
    int64_t processBS1ByCore = 0;
    int64_t oriWinStart = 0;
    int64_t oriWinEnd = 0;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::AllocEventID()
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);

        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);

        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[1]);
    }
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::FreeEventID()
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);

        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[0]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[1]);
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[0]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[1]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[0]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[1]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Init(
    GM_ADDR query, GM_ADDR ori_kv, GM_ADDR cmp_kv, GM_ADDR d_out, GM_ADDR out, GM_ADDR ori_sparse_indices,
    GM_ADDR cmp_sparse_indices, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_ori_kv, GM_ADDR cu_seqlens_cmp_kv,
    GM_ADDR seqused_q, GM_ADDR seqused_ori_kv, GM_ADDR seqused_cmp_kv, GM_ADDR cmp_residual_kv, GM_ADDR ori_topk_length,
    GM_ADDR cmp_topk_length, GM_ADDR softmax_lse, GM_ADDR sinks, GM_ADDR metadata, GM_ADDR dq, GM_ADDR dori_kv,
    GM_ADDR dcmp_kv, GM_ADDR dsinks, GM_ADDR ori_softmax_l1, GM_ADDR cmp_softmax_l1, GM_ADDR workspace,
    SFagTilingType ordTilingData, TPipe *pipeIn)
{
    // init current core tilingInfo
    if ASCEND_IS_AIV {
        vBlockIdx = GetBlockIdx();
        cBlockIdx = vBlockIdx / CV_CORE_RATIO;
        vSubBlockIdx = GetSubBlockIdx();
    } else {
        cBlockIdx = GetBlockIdx();
    }
    coreNum = AscendC::GetBlockNum();
    tilingData = ordTilingData;
    pipe = pipeIn;
    metadataGm = metadata;
    // fill constInfo
    SetConstInfo();

    actualCalcS1Token = constInfo.s1Token;
    actualCalcS2Token = constInfo.s2Token;

    actualSeqQlenAddr = cu_seqlens_q;
    actualSeqOriKvlenAddr = cu_seqlens_ori_kv;
    actualSeqCmpKvlenAddr = cu_seqlens_cmp_kv;
    usedSeqQlenAddr = seqused_q;
    usedSeqOriKvlenAddr = seqused_ori_kv;
    usedSeqCmpKvlenAddr = seqused_cmp_kv;
    oriTopkLengthAddr = ori_topk_length;
    cmpTopkLengthAddr = cmp_topk_length;
    cmpResidualKVAddr = cmp_residual_kv;

    InitCVCommonGlobalBuffer(workspace);
    InitCVCommonBuffer();

    // optional add
    SetOptionalInfo();

    // pass params to vector block
    vecBlock.SetVecBlockParams(pipeIn, tilingData, vBlockIdx, cBlockIdx, vSubBlockIdx, constInfo, attenMaskInfo,
                               pseInfo);
    vecBlock.InitUbBuffer(constInfo);
    vecBlock.InitGlobalBuffer(ori_kv, cmp_kv, d_out, out, ori_sparse_indices, cmp_sparse_indices, softmax_lse, sinks,
                              dsinks, ori_softmax_l1, cmp_softmax_l1, dq, dori_kv, dcmp_kv, cu_seqlens_q,
                              cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv,
                              cmp_residual_kv, ori_topk_length, cmp_topk_length, workspace, constInfo);
    // pass params to cube block
    cubeBlock.SetCubeBlockParams(pipeIn, tilingData, &l1BufferManager);
    cubeBlock.InitCubeBuffer(constInfo);
    cubeBlock.InitGlobalBuffer(query, ori_kv, cmp_kv, d_out, workspace);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Process()
{
    GetDerived()->Process();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonGlobalBuffer(
    GM_ADDR workspace)
{
    // init workspace address
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dOriKVWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dOriKVWorkSpaceOffset / sizeof(float));
    dCmpKVWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dCmpKVWorkSpaceOffset / sizeof(float));

    int64_t selectedKWorkSpaceOffset = tilingData->baseParams.selectedKWorkSpaceOffset / sizeof(INPUT_TYPE) +
                                       cBlockIdx * CUBE_BASEN * constInfo.commonConstInfo.dSize * 3;
    selectedKWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)workspace + selectedKWorkSpaceOffset);
    if constexpr (IsDETER) {
        mm4ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->baseParams.mm4ResWorkSpaceOffset / sizeof(float));
        mm5ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->baseParams.mm5ResWorkSpaceOffset / sizeof(float));
    } else {
        int64_t mm4ResWorkSpaceOffset = tilingData->baseParams.mm4ResWorkSpaceOffset / sizeof(float) +
                                        cBlockIdx * PROCESS_KV_SIZE_NO_DETER * constInfo.commonConstInfo.dSize * 2;
        mm4ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResWorkSpaceOffset);
        int64_t mm5ResWorkSpaceOffset = tilingData->baseParams.mm5ResWorkSpaceOffset / sizeof(float) +
                                        cBlockIdx * PROCESS_KV_SIZE_NO_DETER * constInfo.commonConstInfo.dSizeV * 2;
        mm5ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResWorkSpaceOffset);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonBuffer()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    dSL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE));
    pL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE));

    pipe->InitBuffer(mm1ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm1ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm2ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm2ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetOptionalInfo()
{}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetConstInfo()
{
    constInfo.hasUsedSeqQ = tilingData->baseParams.hasUsedSeqQ;
    constInfo.hasUsedSeqOriKV = tilingData->baseParams.hasUsedSeqOriKV;
    constInfo.hasUsedSeqCmpKV = tilingData->baseParams.hasUsedSeqCmpKV;
    constInfo.hasOriTopK = tilingData->baseParams.hasOriTopK;
    constInfo.hasCmpTopK = tilingData->baseParams.hasCmpTopK;
    constInfo.isSink = tilingData->baseParams.isSink;
    bIndex = constInfo.hasUsedSeqQ ? -1 : 0;
    constInfo.bSize = tilingData->baseParams.b;
    constInfo.n2Size = tilingData->baseParams.n2;
    constInfo.oriSelectedBlockCount = tilingData->baseParams.oriselectedBlockCount;
    constInfo.cmpSelectedBlockCount = tilingData->baseParams.cmpselectedBlockCount;
    constInfo.selectedBlockCount = constInfo.oriSelectedBlockCount + constInfo.cmpSelectedBlockCount;
    constInfo.commonConstInfo.gSize = tilingData->baseParams.g;
    constInfo.halfGRealSize = (constInfo.commonConstInfo.gSize + 1) >> 1;
    constInfo.firstHalfGRealSize = constInfo.halfGRealSize;
    if (vSubBlockIdx == 1) {
        constInfo.halfGRealSize = constInfo.commonConstInfo.gSize - constInfo.firstHalfGRealSize;
    }
    constInfo.commonConstInfo.s1Size = tilingData->baseParams.s1;
    constInfo.commonConstInfo.s2Size = tilingData->baseParams.s2;
    constInfo.s3Size = tilingData->baseParams.s3;
    constInfo.commonConstInfo.dSize = tilingData->baseParams.d;
    constInfo.commonConstInfo.dSizeV = tilingData->baseParams.d1;
    constInfo.dTotalSize = tilingData->baseParams.d;
    constInfo.commonConstInfo.layoutType = tilingData->baseParams.layout;

    constInfo.commonConstInfo.s1D = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.gS1D = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1D;
    constInfo.commonConstInfo.n2GS1D = constInfo.n2Size * constInfo.commonConstInfo.gS1D;
    constInfo.commonConstInfo.s2D = constInfo.commonConstInfo.s2Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.n2S2D = constInfo.n2Size * constInfo.commonConstInfo.s2D;
    constInfo.commonConstInfo.s1S2 = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.s2Size;
    constInfo.commonConstInfo.gS1 = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Size;
    constInfo.commonConstInfo.gD = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.n2D = constInfo.n2Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.bN2D = constInfo.bSize * constInfo.commonConstInfo.n2D;
    constInfo.commonConstInfo.n2G = constInfo.n2Size * constInfo.commonConstInfo.gSize;
    constInfo.commonConstInfo.n2GD = constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.bN2GD = constInfo.bSize * constInfo.commonConstInfo.n2GD;
    constInfo.commonConstInfo.gS2 = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s2Size;

    constInfo.commonConstInfo.s1Dv = constInfo.commonConstInfo.s1D;
    constInfo.commonConstInfo.gS1Dv = constInfo.commonConstInfo.gS1D;
    constInfo.commonConstInfo.n2GS1Dv = constInfo.commonConstInfo.n2GS1D;
    constInfo.commonConstInfo.s2Dv = constInfo.commonConstInfo.s2D;
    constInfo.commonConstInfo.n2S2Dv = constInfo.commonConstInfo.n2S2D;
    constInfo.commonConstInfo.gDv = constInfo.commonConstInfo.gD;
    constInfo.commonConstInfo.n2Dv = constInfo.commonConstInfo.n2D;
    constInfo.commonConstInfo.bN2Dv = constInfo.commonConstInfo.bN2D;
    constInfo.commonConstInfo.n2GDv = constInfo.commonConstInfo.n2GD;
    constInfo.commonConstInfo.bN2GDv = constInfo.commonConstInfo.bN2GD;

    constInfo.cmpRatio = static_cast<int64_t>(tilingData->baseParams.cmpRatio);
    constInfo.scaleValue = tilingData->baseParams.scaleValue;
    constInfo.n2GS1oS2o = constInfo.commonConstInfo.n2G * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.gS1oS2o = constInfo.commonConstInfo.gSize * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.s1oS2o = constInfo.s1Outer * constInfo.s2Outer;

    if ASCEND_IS_AIC {
        if constexpr (IS_TND) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSizeV;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSizeV;
            constInfo.mm2Ka = constInfo.commonConstInfo.dSize;
            constInfo.mm2Kb = constInfo.dTotalSize;
        } else {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSizeV;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSizeV;
            constInfo.mm2Ka = constInfo.commonConstInfo.dSize;
            constInfo.mm2Kb = constInfo.dTotalSize;
        }
        constInfo.mm3Ka = constInfo.mm2Ka;
        constInfo.mm4Kb = constInfo.mm2Kb;
    }
    constInfo.commonConstInfo.subBlockIdx = vSubBlockIdx;

    if ASCEND_IS_AIV {
        constInfo.sfmgMaxLoopSize = VECTOR_BASEM * VECTOR_BASEN / 512; // softmaxGrad每次最大能处理的m轴大小
        constInfo.dAlignToBlock = AlignTo(constInfo.commonConstInfo.dSizeV, INPUT_BLOCK_NUM);
        constInfo.dAlignToBlockForFp8 = AlignTo(constInfo.commonConstInfo.dSizeV, INPUT_BLOCK_NUM_FOR_FP8);
    }
    GetDerived()->SetUniqueConstInfo(constInfo);

    constInfo.totalNum = ((__gm__ uint32_t *)metadataGm)[0];
    int64_t totalNum = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[0]);
    int64_t formerCoreProcessNNum = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[1]);
    formerCoreProcessS1Num = formerCoreProcessNNum;
    int64_t remainCoreProcessNNum = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[2]);
    int64_t remainCoreNum = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[3]);
    usedCoreNum = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[4]);
    maxOriSeqKV = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[5]);
    maxCmpSeqKV = static_cast<int64_t>(((__gm__ int32_t *)metadataGm)[6]);
    int64_t totalCoreNum = static_cast<int64_t>(GetBlockNum());
    constInfo.usedCoreNum = usedCoreNum;
    if (cBlockIdx >= usedCoreNum) {
        processBS1ByCore = 0;
    } else if (cBlockIdx < totalCoreNum - remainCoreNum) {
        processBS1ByCore = formerCoreProcessNNum;
    } else {
        processBS1ByCore = remainCoreProcessNNum;
    }

    if constexpr (!IS_TND) {
        curS1 = constInfo.commonConstInfo.s1Size;
        curS2 = constInfo.commonConstInfo.s2Size;
        curS3 = constInfo.s3Size;
    }
    constInfo.selectedCountOffset = CUBE_BASEM;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBidx(
    int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actualSeqOriKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actualSeqOriKvlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqOriKvlenAddr)[bIdx - 1];
    }
    return;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <const bool IS_ORI_KV>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetRunInfo(
    FagRunInfo &runInfo, FagRunInfo &preRunInfo, int64_t taskId, int64_t sTaskId, int64_t deterTaskId, int64_t kvOffset)
{
    runInfo.t1Index = t1Index;
    runInfo.t2Index = t2Index;
    runInfo.t3Index = t3Index;
    runInfo.n2Index = n2Index;
    runInfo.blkCntOffset = IS_ORI_KV ? blkCntOffset : blkCntOffset - actualOriSelectedBlockCount;
    runInfo.commonRunInfo.boIdx = bIndex;
    runInfo.isOriKV = IS_ORI_KV;
    runInfo.selectedBlockCount = IS_ORI_KV ? constInfo.oriSelectedBlockCount : constInfo.cmpSelectedBlockCount;
    runInfo.isSparse = IS_ORI_KV ? IsOriKVSparse : IsCmpKVSparse;

    runInfo.commonRunInfo.actualS1Size = curS1;
    runInfo.commonRunInfo.actualS2Size = curS2;
    runInfo.actualSelectedBlockCount = IS_ORI_KV ? actualOriSelectedBlockCount : actualSelectedBlockCount;
    runInfo.actualCmpSelectedBlockCount = actualCmpSelectedBlockCount;

    runInfo.commonRunInfo.s2RealSize =
        blkCntOffset + constInfo.selectedCountOffset <= runInfo.actualSelectedBlockCount ?
            constInfo.selectedCountOffset :
            runInfo.actualSelectedBlockCount - blkCntOffset;
    runInfo.actualSelCntOffset = runInfo.commonRunInfo.s2RealSize;
    runInfo.sTaskId = sTaskId;
    runInfo.sTaskIdMod2 = sTaskId & 1;
    runInfo.deterTaskId = deterTaskId;
    runInfo.deterTaskIdMod2 = deterTaskId & 1;
    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
    runInfo.halfGRealSize = (constInfo.commonConstInfo.gSize + 1) >> 1;
    runInfo.firstHalfGRealSize = runInfo.halfGRealSize;
    if (vSubBlockIdx == 1) {
        runInfo.halfGRealSize = constInfo.commonConstInfo.gSize - runInfo.halfGRealSize;
    }

    // 当前S1内执行了一次task后，isS1IdxNoChange置为true
    runInfo.isS1IdxNoChange = t1Index == lastT1Index;
    if constexpr (IS_ORI_KV) {
        runInfo.isNextS1IdxNoChange =
            actualCmpSelectedBlockCount > 0 ||
            runInfo.blkCntOffset + constInfo.selectedCountOffset < actualOriSelectedBlockCount;
    } else {
        runInfo.isNextS1IdxNoChange =
            runInfo.blkCntOffset + constInfo.selectedCountOffset < actualCmpSelectedBlockCount;
    }
    preRunInfo.isSecondLast = !runInfo.isNextS1IdxNoChange;
    lastT1Index = t1Index;

    GetDerived()->SetUniqueRunInfo(runInfo);

    runInfo.commonRunInfo.queryOffset = GetQueryOffset(runInfo);
    runInfo.dyOffset = GetDxOffset(runInfo);
    runInfo.oriKVOffset = GetKeyOffset<true>(runInfo);
    runInfo.cmpKVOffset = GetKeyOffset<false>(runInfo);
    runInfo.commonRunInfo.keyOffset = IS_ORI_KV ? runInfo.oriKVOffset : runInfo.cmpKVOffset;
    runInfo.commonRunInfo.valueOffset = runInfo.commonRunInfo.keyOffset;
    runInfo.sparseIndicesOffset = runInfo.t1Index * (constInfo.n2Size * runInfo.selectedBlockCount) +
                                  runInfo.n2Index * runInfo.selectedBlockCount + runInfo.blkCntOffset;
    if constexpr (IS_ORI_KV) {
        if constexpr (IsOriKVSparse) {
            runInfo.kSelectedWsAddr = (taskId % 2) * CUBE_BASEN * constInfo.dTotalSize;
        } else {
            runInfo.kSelectedWsAddr =
                runInfo.commonRunInfo.keyOffset + (runInfo.blkCntOffset + oriWinStart) * constInfo.dTotalSize;
        }
    } else {
        if constexpr (IsCmpKVSparse) {
            runInfo.kSelectedWsAddr = (taskId % 2) * CUBE_BASEN * constInfo.dTotalSize;
        } else {
            runInfo.kSelectedWsAddr = runInfo.commonRunInfo.keyOffset + runInfo.blkCntOffset * constInfo.dTotalSize;
        }
    }
    if constexpr (IsDETER) {
        runInfo.mm4ResWsAddr = runInfo.deterTaskIdMod2 * PROCESS_KV_SIZE * constInfo.dTotalSize * coreNum +
                               cBlockIdx * PROCESS_KV_SIZE * constInfo.dTotalSize +
                               (runInfo.blkCntOffset - kvOffset) * constInfo.dTotalSize;
    } else {
        runInfo.mm4ResWsAddr = (taskId % 2) * CUBE_BASEN * constInfo.dTotalSize;
    }
    runInfo.mm5ResWsAddr = runInfo.mm4ResWsAddr;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <const bool IS_ORI_KV>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetDeterRunInfo(
    FagRunInfo &runInfo, int64_t sTaskId, int64_t deterTaskId, int64_t kvSeqOffset)
{
    runInfo.sTaskId = sTaskId;
    runInfo.sTaskIdMod2 = sTaskId & 1;
    runInfo.deterTaskId = deterTaskId;
    runInfo.deterTaskIdMod2 = deterTaskId & 1;
    runInfo.isOriKV = IS_ORI_KV;
    runInfo.isSparse = IS_ORI_KV ? IsOriKVSparse : IsCmpKVSparse;
    runInfo.kvSeqOffset = kvSeqOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <bool IS_MM1_MM2>
__aicore__ inline int64_t SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSize;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryRopeOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.dRopeSize;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetDxOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSizeV;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <const bool IS_ORI_KV>
__aicore__ inline int64_t SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyOffset(
    FagRunInfo &runInfo)
{
    if constexpr (IS_ORI_KV) {
        return runInfo.t2Index * constInfo.n2Size * constInfo.commonConstInfo.dSize;
    } else {
        return runInfo.t3Index * constInfo.n2Size * constInfo.commonConstInfo.dSize;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyRopeOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t2Index * constInfo.n2Size * constInfo.dRopeSize;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetValueOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t2Index * constInfo.n2Size * constInfo.commonConstInfo.dSizeV;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetActualSelCount(
    const int64_t t1Idx, const int64_t n2Idx, int64_t &actSelBlkCount)
{
    if (constInfo.hasOriTopK) {
        oriTopkLength = ((__gm__ int32_t *)oriTopkLengthAddr)[t1Index];
    } else if (IsOriKVSparse) {
        oriTopkLength = constInfo.oriSelectedBlockCount;
    } else {
        oriTopkLength = curS2;
    }
    if (constInfo.hasCmpTopK) {
        cmpTopkLength = ((__gm__ int32_t *)cmpTopkLengthAddr)[t1Index];
    } else if (IsCmpKVSparse) {
        cmpTopkLength = constInfo.cmpSelectedBlockCount;
    } else {
        cmpTopkLength = curS3;
    }
    actualOriSelectedBlockCount = Min(Min(oriTopkLength, curS2), usedS2);
    actualCmpSelectedBlockCount = Min(Min(cmpTopkLength, curS3), usedS3);
    actualSelectedBlockCount = actualOriSelectedBlockCount + actualCmpSelectedBlockCount;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetTndSeqLen(
    const int64_t t1Idx, int64_t &bIdx)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;
    int64_t t3Offset = 0;

    if (!constInfo.hasUsedSeqQ) {
        if constexpr (IS_TND) {
            int64_t curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            while (usedT1Index >= curT1) {
                curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[++bIndex];
            }
            bIndex = bIndex - 1;
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            s1Index = usedT1Index - t1Offset;
        } else {
            bIndex = usedT1Index / curS1;
            s1Index = usedT1Index % constInfo.commonConstInfo.s1Size;
        }
        t1Index = usedT1Index;
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
        t1Offset += s1Index;
        t1Index = t1Offset;
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
                t2Offset = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex];
                curS2 = ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex + 1] -
                        ((__gm__ int32_t *)actualSeqOriKvlenAddr)[bIndex];
            }
            if constexpr (IsCmpKVExist) {
                t3Offset = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex];
                curS3 = ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex + 1] -
                        ((__gm__ int32_t *)actualSeqCmpKvlenAddr)[bIndex];
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
        usedS2 = ((__gm__ int32_t *)usedSeqOriKvlenAddr)[bIndex];
    } else {
        usedS2 = curS2;
    }
    if (constInfo.hasUsedSeqCmpKV) {
        usedS3 = ((__gm__ int32_t *)usedSeqCmpKvlenAddr)[bIndex];
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
        int64_t oriUsedS3 = usedS3 * constInfo.cmpRatio + ((__gm__ int32_t *)cmpResidualKVAddr)[bIndex];
        usedS3 = Max((oriUsedS3 - usedS1 + s1Index + 1) / constInfo.cmpRatio, 0);
    }

    t2Index = t2Offset;
    t3Index = t3Offset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SyncALLCores()
{
    SyncAll<false>();
}
} // namespace SfagBaseApi
#endif
