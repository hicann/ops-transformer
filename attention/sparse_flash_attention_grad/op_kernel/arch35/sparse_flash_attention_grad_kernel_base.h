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

#ifndef SPARSE_FLASH_ATTENTION_GRAD_KERNEL_BASE_H
#define SPARSE_FLASH_ATTENTION_GRAD_KERNEL_BASE_H

#include "sparse_flash_attention_grad_common.h"
#include "sparse_flash_attention_grad_block_cube_nle64.h"

namespace SfagBaseApi {

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparse_indices, GM_ADDR dy,
                                GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR actualSeqQlen,
                                GM_ADDR actualSeqKvlen, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk,
                                GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR workspace,
                                SFagTilingType ordTilingData, TPipe *pipeIn, GM_ADDR sinks, GM_ADDR dSinks);
    __aicore__ inline void InitCVCommonBuffer();
    __aicore__ inline void InitCVCommonGlobalBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetOptionalInfo();
    __aicore__ inline void SetRunInfo(FagRunInfo &runInfo, int64_t taskId, int64_t sTaskId, int64_t deterTaskId);
    __aicore__ inline void SetDeterRunInfo(FagRunInfo &runInfo, int64_t sTaskId, int64_t deterTaskId);
    __aicore__ inline void Process();
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetQueryOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetQueryRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDxOffset(FagRunInfo &runInfo);
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetKeyOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetValueOffset(FagRunInfo &runInfo);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetActualSelCount(const int64_t t1Idx, const int64_t n2Idx, int64_t &actSelBlkCount);
    __aicore__ inline void GetTndSeqLen(const int64_t t1Idx, int64_t &bIdx);
    __aicore__ inline int64_t GetValidTndT1Size();
    __aicore__ inline bool IsValidTndT1Index(const int64_t t1Idx);
    __aicore__ inline int64_t GetMaxDeterEpochs();
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                        const GlobalTensor<INPUT_TYPE> &selectedVWorkSpaceGm, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo);
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                       const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
                                        BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo);
    __aicore__ inline ChildClass *GetDerived()
    {
        return static_cast<ChildClass *>(this);
    }

    constexpr static bool IS_D_NO_EQUAL = true;
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
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, mm4ResWorkSpaceGm, mm5ResWorkSpaceGm;
    GlobalTensor<INPUT_TYPE> selectedKWorkSpaceGm;
    // CV核间共享Buffer
    TBuf<> mm1ResBuf[2];
    TBuf<> mm2ResBuf[2];
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> pL1Buf;
    BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> dSL1Buf;

    GM_ADDR prefixNAddr;
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;

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
    int64_t bIndex = 0;
    int64_t s1Index = 0;

    SFagTilingType tilingData;
    FagConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;

    CubeBlockType cubeBlock;
    // AIV 上 CubeBlockType 是 Dummy：NLe64 也必须走 Dummy，否则会把 L0/mmad 编进 AIV。
    using CubeBlockNLe64Type =
        typename std::conditional<CubeBlockType::IS_CUBE_DUMMY, CubeBlockType,
                                  FAGBlockCubeNLe64<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_DROP, IS_TND, IS_ROPE, IS_DETER,
                                                    KV_MERGE, IS_SINKS, s2TemplateType, dTemplateType>>::type;
    CubeBlockNLe64Type cubeBlockNLe64;
    VecBlockType vecBlock;

    int64_t blkCntOffset = 0;
    int64_t actualSelectedBlockCount = 0;
    int64_t t1Index = 0;
    int64_t n2Index = 0;
    int64_t lastT1Index = -1;
    uint8_t qDxPingPongSeq = 0; // N<=64 Q/Dy L1 ping-pong，随 S1 翻转
    int64_t usedCoreNum = 0;
    int64_t processBS1ByCore = 0;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::AllocEventID()
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
        if constexpr (IS_DETER) {
            // 第一轮 AIC Wait(flag14) 的 initial credit。每个 AIV 各 Set 一次，对应 AIC 的 14 / 16+14。
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::FreeEventID()
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
        if constexpr (IS_DETER) {
            // 消费最后一轮 AIV Set(flag14)，或从未 scatter 时消费 AllocEventID 的 initial credit。
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax,
    GM_ADDR softmaxSum, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR workspace, SFagTilingType ordTilingData,
    TPipe *pipeIn, GM_ADDR sinks, GM_ADDR dSinks)
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

    // fill constInfo
    SetConstInfo();

    actualCalcS1Token = constInfo.s1Token;
    actualCalcS2Token = constInfo.s2Token;

    actualSeqQlenAddr = actualSeqQlen;
    actualSeqKvlenAddr = actualSeqKvlen;
    constInfo.seqS1_addr = actualSeqQlen;
    constInfo.seqS2_addr = actualSeqKvlen;

    InitCVCommonGlobalBuffer(dq, dk, dv, workspace);
    InitCVCommonBuffer();

    // optional add
    SetOptionalInfo();

    // pass params to vector block
    vecBlock.SetVecBlockParams(pipeIn, tilingData, vBlockIdx, cBlockIdx, vSubBlockIdx, constInfo, attenMaskInfo,
                               pseInfo);
    vecBlock.InitUbBuffer();
    vecBlock.InitGlobalBuffer(key, dy, y, sparseIndices, softmaxMax, softmaxSum, keyRope, dq, dk, dv, actualSeqQlen,
                              actualSeqKvlen, workspace, sinks, dSinks);

    if (constInfo.isHeadNLe64) {
        cubeBlockNLe64.SetCubeBlockParams(pipeIn, tilingData, &l1BufferManager);
        cubeBlockNLe64.InitCubeBuffer(constInfo);
        cubeBlockNLe64.InitGlobalBuffer(query, key, value, dy, queryRope, keyRope, dq, dk, dv, workspace);
    } else {
        cubeBlock.SetCubeBlockParams(pipeIn, tilingData, &l1BufferManager);
        cubeBlock.InitCubeBuffer(constInfo);
        cubeBlock.InitGlobalBuffer(query, key, value, dy, queryRope, keyRope, dq, dk, dv, workspace);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Process()
{
    GetDerived()->Process();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonGlobalBuffer(GM_ADDR dq,
                                                                                                     GM_ADDR dk,
                                                                                                     GM_ADDR dv,
                                                                                                     GM_ADDR workspace)
{
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    if constexpr (!KV_MERGE) {
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    }
    // init workspace address
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));

    int64_t selectedKWorkSpaceOffset =
        tilingData->baseParams.selectedKWorkSpaceOffset / sizeof(INPUT_TYPE) +
        cBlockIdx * CUBE_BASEN * (constInfo.commonConstInfo.dSize + constInfo.dRopeSize) * 3;
    selectedKWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)workspace + selectedKWorkSpaceOffset);
    if constexpr (IS_DETER) {
        mm4ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->baseParams.mm4ResWorkSpaceOffset / sizeof(float));
        mm5ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->baseParams.mm5ResWorkSpaceOffset / sizeof(float));
    } else {
        int64_t mm4ResWorkSpaceOffset = tilingData->baseParams.mm4ResWorkSpaceOffset / sizeof(float) +
                                        cBlockIdx * constInfo.selectedBlockCount * constInfo.selectedBlockSize *
                                            (constInfo.commonConstInfo.dSize + constInfo.dRopeSize) * 2;
        mm4ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResWorkSpaceOffset);
        int64_t mm5ResWorkSpaceOffset = tilingData->baseParams.mm5ResWorkSpaceOffset / sizeof(float) +
                                        cBlockIdx * constInfo.selectedBlockCount * constInfo.selectedBlockSize *
                                            constInfo.commonConstInfo.dSizeV * 2;
        mm5ResWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResWorkSpaceOffset);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonBuffer()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    if (constInfo.isHeadNLe64) {
        // gather 64×g：每槽 8KB（g=64 bf16），dS/P 双缓冲合计 32KB
        uint32_t dsPSize = AlignTo16(constInfo.commonConstInfo.gSize) * SFAG_GATHER_S2_HEAD_N * sizeof(INPUT_TYPE);
        dSL1Buf.Init(l1BufferManager, dsPSize);
        pL1Buf.Init(l1BufferManager, dsPSize);
    } else {
        dSL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE));
        pL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE));
    }

    pipe->InitBuffer(mm1ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm1ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm2ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm2ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetOptionalInfo()
{}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetConstInfo()
{
    constInfo.sparseMode = tilingData->baseParams.sparseMode;
    constInfo.bSize = tilingData->baseParams.b;
    constInfo.n2Size = tilingData->baseParams.n2;
    constInfo.selectedBlockCount = tilingData->baseParams.selectedBlockCount;
    constInfo.commonConstInfo.gSize = tilingData->baseParams.g;
    constInfo.isHeadNLe64 = tilingData->baseParams.isHeadNLe64 != 0;
    constInfo.commonConstInfo.s1Size = tilingData->baseParams.s1;
    constInfo.commonConstInfo.s2Size = tilingData->baseParams.s2;
    constInfo.commonConstInfo.dSize = IS_ROPE ? tilingData->baseParams.d - ROPE_D_64 : tilingData->baseParams.d;
    constInfo.commonConstInfo.dSizeV = tilingData->baseParams.d1;
    constInfo.dTotalSize = tilingData->baseParams.d;
    constInfo.commonConstInfo.layoutType = tilingData->baseParams.layout;
    constInfo.dRopeSize = IS_ROPE ? ROPE_D_64 : 0;

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
    // for D_V
    if constexpr (IS_D_NO_EQUAL) {
        constInfo.commonConstInfo.s1Dv = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.gS1Dv = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Dv;
        constInfo.commonConstInfo.n2GS1Dv = constInfo.n2Size * constInfo.commonConstInfo.gS1Dv;
        constInfo.commonConstInfo.s2Dv = constInfo.commonConstInfo.s2Size * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.n2S2Dv = constInfo.n2Size * constInfo.commonConstInfo.s2Dv;
        constInfo.commonConstInfo.gDv = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.n2Dv = constInfo.n2Size * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.bN2Dv = constInfo.bSize * constInfo.commonConstInfo.n2Dv;
        constInfo.commonConstInfo.n2GDv = constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSizeV;
        constInfo.commonConstInfo.bN2GDv = constInfo.bSize * constInfo.commonConstInfo.n2GDv;
        if constexpr (IS_ROPE) {
            constInfo.commonConstInfo.s1Dr = constInfo.commonConstInfo.s1Size * constInfo.dRopeSize;
            constInfo.commonConstInfo.gS1Dr = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Dr;
            constInfo.commonConstInfo.n2GS1Dr = constInfo.n2Size * constInfo.commonConstInfo.gS1Dr;
            constInfo.commonConstInfo.s2Dr = constInfo.commonConstInfo.s2Size * constInfo.dRopeSize;
            constInfo.commonConstInfo.n2S2Dr = constInfo.n2Size * constInfo.commonConstInfo.s2Dr;
            constInfo.commonConstInfo.gDr = constInfo.commonConstInfo.gSize * constInfo.dRopeSize;
            constInfo.commonConstInfo.n2Dr = constInfo.n2Size * constInfo.dRopeSize;
            constInfo.commonConstInfo.bN2Dr = constInfo.bSize * constInfo.commonConstInfo.n2Dr;
            constInfo.commonConstInfo.n2GDr = constInfo.commonConstInfo.n2G * constInfo.dRopeSize;
            constInfo.commonConstInfo.bN2GDr = constInfo.bSize * constInfo.commonConstInfo.n2GDr;
        }
    } else {
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
    }

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

    usedCoreNum = this->tilingData->baseParams.usedCoreNum;
    if (cBlockIdx >= usedCoreNum) {
        processBS1ByCore = 0;
    } else if (cBlockIdx < this->tilingData->baseParams.formerCoreNum) {
        processBS1ByCore = this->tilingData->baseParams.formerCoreProcessNNum;
    } else {
        processBS1ByCore = this->tilingData->baseParams.remainCoreProcessNNum;
    }

    if constexpr (!IS_TND) {
        curS1 = constInfo.commonConstInfo.s1Size;
        curS2 = constInfo.commonConstInfo.s2Size;
    }
    int64_t gatherS2 = constInfo.isHeadNLe64 ? SFAG_GATHER_S2_HEAD_N : CUBE_BASEM;
    constInfo.selectedCountOffset = gatherS2 / constInfo.selectedBlockSize;
    if (constInfo.selectedCountOffset < 1) {
        constInfo.selectedCountOffset = 1;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetRunInfo(
    FagRunInfo &runInfo, int64_t taskId, int64_t sTaskId, int64_t deterTaskId)
{
    runInfo.t1Index = t1Index;
    runInfo.n2Index = n2Index;
    runInfo.blkCntOffset = blkCntOffset;
    runInfo.commonRunInfo.boIdx = bIndex;
    if constexpr (IS_TND) {
        if (unlikely(bIndex == 0)) {
            runInfo.t2Index = 0;
        } else {
            runInfo.t2Index = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex - 1];
        }
    } else {
        runInfo.t2Index = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s2Size;
    }

    runInfo.commonRunInfo.actualS1Size = curS1;
    runInfo.commonRunInfo.actualS2Size = curS2;
    runInfo.commonRunInfo.s2RealSize = blkCntOffset + constInfo.selectedCountOffset <= actualSelectedBlockCount ?
                                           constInfo.selectedCountOffset :
                                           actualSelectedBlockCount - blkCntOffset;
    runInfo.actualSelCntOffset = runInfo.commonRunInfo.s2RealSize;
    runInfo.actualSelectedBlockCount = actualSelectedBlockCount;

    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
    runInfo.halfGRealSize = (constInfo.commonConstInfo.gSize + 1) >> 1;
    runInfo.halfGRealSize = runInfo.halfGRealSize;
    runInfo.firstHalfGRealSize = runInfo.halfGRealSize;
    if (vSubBlockIdx == 1) {
        runInfo.halfGRealSize = constInfo.commonConstInfo.gSize - runInfo.halfGRealSize;
    }

    // 当前S1内执行了一次task后，isS1IdxNoChange置为true
    runInfo.isS1IdxNoChange = t1Index == lastT1Index;
    runInfo.isNextS1IdxNoChange = blkCntOffset + constInfo.selectedCountOffset < actualSelectedBlockCount;
    if (!runInfo.isS1IdxNoChange && lastT1Index >= 0) {
        qDxPingPongSeq ^= 1;
    }
    runInfo.qDxPingPongIdx = qDxPingPongSeq;
    lastT1Index = t1Index;

    GetDerived()->SetUniqueRunInfo(runInfo);

    runInfo.commonRunInfo.queryOffset = GetQueryOffset(runInfo);
    runInfo.dyOffset = GetDxOffset(runInfo);
    runInfo.commonRunInfo.keyOffset = GetKeyOffset(runInfo);
    runInfo.commonRunInfo.valueOffset = GetValueOffset(runInfo);

    runInfo.queryOffsetWithRope = runInfo.commonRunInfo.queryOffset;
    runInfo.keyOffsetWithRope = runInfo.commonRunInfo.keyOffset;
    runInfo.queryOffsetWithRopeForMm12 = runInfo.commonRunInfo.queryOffset;
    runInfo.keyOffsetWithRopeForMm12 = runInfo.commonRunInfo.keyOffset;
    // Rope场景后面三个mm的GM offset不能和前面两个mm共用，因此需要重新计算
    if constexpr (IS_ROPE) {
        runInfo.queryOffsetWithRope = GetQueryOffset<false>(runInfo);
        runInfo.keyOffsetWithRope = GetKeyOffset<false>(runInfo);
        runInfo.queryOffsetWithRopeForMm12 = GetQueryOffset<true>(runInfo);
        runInfo.keyOffsetWithRopeForMm12 = GetKeyOffset<true>(runInfo);
        runInfo.commonRunInfo.qRopeOffset = GetQueryRopeOffset(runInfo);
        runInfo.commonRunInfo.kRopeOffset = GetKeyRopeOffset(runInfo);
    }
    runInfo.kSelectedWsAddr = (taskId % 2) * CUBE_BASEN * constInfo.dTotalSize;
    runInfo.mm4ResWsAddr = runInfo.kSelectedWsAddr;
    runInfo.mm5ResWsAddr = (taskId % 2) * CUBE_BASEN * constInfo.commonConstInfo.dSizeV;
    if constexpr (IS_DETER) {
        runInfo.sTaskId = sTaskId;
        runInfo.sTaskIdMod2 = sTaskId & 1;
        runInfo.deterTaskId = deterTaskId;
        runInfo.deterTaskIdMod2 = deterTaskId & 1;
        // deter workspace 布局是 [parity][core][block][d]：每个 epoch 每核只放一个 token 的贡献，
        // 没有 n2 维度。消费侧 ScatterAddDeter 的 topkIndices 寻址同样按 n2 == 1 展开。
        // host tiling 已经硬性拒绝 n2 != 1，这里依赖该前提。
        runInfo.mm4ResWsAddr = runInfo.deterTaskIdMod2 * constInfo.selectedBlockCount * constInfo.dTotalSize * coreNum +
                               cBlockIdx * constInfo.selectedBlockCount * constInfo.dTotalSize +
                               runInfo.blkCntOffset * constInfo.dTotalSize;
        runInfo.mm5ResWsAddr =
            runInfo.deterTaskIdMod2 * constInfo.selectedBlockCount * constInfo.commonConstInfo.dSizeV * coreNum +
            cBlockIdx * constInfo.selectedBlockCount * constInfo.commonConstInfo.dSizeV +
            runInfo.blkCntOffset * constInfo.commonConstInfo.dSizeV;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetDeterRunInfo(
    FagRunInfo &runInfo, int64_t sTaskId, int64_t deterTaskId)
{
    runInfo.sTaskId = sTaskId;
    runInfo.sTaskIdMod2 = sTaskId & 1;
    runInfo.deterTaskId = deterTaskId;
    runInfo.deterTaskIdMod2 = deterTaskId & 1;
    if constexpr (!IS_TND) {
        runInfo.commonRunInfo.actualS1Size = constInfo.commonConstInfo.s1Size;
        runInfo.commonRunInfo.actualS2Size = constInfo.commonConstInfo.s2Size;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <bool IS_MM1_MM2>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryOffset(
    FagRunInfo &runInfo)
{
    if constexpr (IS_MM1_MM2) {
        return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSize;
    } else {
        return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.dTotalSize;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryRopeOffset(FagRunInfo &runInfo)
{
    return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.dRopeSize;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetDxOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t1Index * constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSizeV;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <bool IS_MM1_MM2>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyOffset(
    FagRunInfo &runInfo)
{
    if constexpr (IS_MM1_MM2) {
        return runInfo.t2Index * constInfo.n2Size * constInfo.commonConstInfo.dSize;
    } else {
        return runInfo.t2Index * constInfo.n2Size * constInfo.dTotalSize;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyRopeOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t2Index * constInfo.n2Size * constInfo.dRopeSize;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetValueOffset(
    FagRunInfo &runInfo)
{
    return runInfo.t2Index * constInfo.n2Size * constInfo.commonConstInfo.dSizeV;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetActualSelCount(
    const int64_t t1Idx, const int64_t n2Idx, int64_t &actSelBlkCount)
{
    // t1Idx / n2Idx 不直接参与计算：因果上界由 GetTndSeqLen 先行写入的 curS1 / curS2 / s1Index 给出，
    // n2 方向不影响 sel 数（deter 路径也只支持 n2 == 1，见 host tiling 校验）。
    (void)t1Idx;
    (void)n2Idx;
    int64_t maxS2 = curS2;
    if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
        maxS2 = Max(curS2 - curS1 + s1Index + 1, 0);
    }
    // 与 FAGBlockVec::GetRunInfo 的消费侧公式必须逐字一致，否则 Scatter 会遍历到生产侧没写过的 block。
    actSelBlkCount = Min(constInfo.selectedBlockCount, Ceil<int64_t>(maxS2, constInfo.selectedBlockSize));
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetTndSeqLen(
    const int64_t t1Idx, int64_t &bIdx)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;
    if constexpr (IS_TND) {
        int64_t curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
        while (t1Idx >= curT1 && bIndex < constInfo.bSize - 1) {
            curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[++bIndex];
        }

        if (t1Idx >= curT1) {
            curS2 = 0;
            return;
        }

        if (unlikely(bIndex == 0)) {
            t1Offset = 0;
            t2Offset = 0;
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            curS2 = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex];
        } else {
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex - 1];
            t2Offset = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex - 1];
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex] - ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex - 1];
            curS2 = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex] - ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex - 1];
        }

        s1Index = t1Idx - t1Offset;
    } else {
        t1Offset = t1Idx;
        bIdx = t1Idx / curS1;
        s1Index = t1Idx % constInfo.commonConstInfo.s1Size;
        t2Offset = bIdx * curS2;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IsValidTndT1Index(
    const int64_t t1Idx)
{
    if constexpr (!IS_TND) {
        return true;
    }
    return t1Idx < GetValidTndT1Size();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetValidTndT1Size()
{
    if constexpr (!IS_TND) {
        return tilingData->baseParams.totalSize;
    }
    // 未校验前提（actual_seq_lengths_query 是 device 张量，tiling 期读不到，无法在 host 侧校验）：
    // 前缀和非负、单调不降，且末值 <= 物理 T1。契约见 docs/aclnnSparseFlashAttentionGrad.md。
    // 违反时尾 padding 的有效区间判断会失效。
    return ((__gm__ int32_t *)actualSeqQlenAddr)[constInfo.bSize - 1];
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetMaxDeterEpochs()
{
    // 所有核必须得到同一值：deterministic scatter epoch 总数。
    // 非 TND：tiling formerCoreProcessNNum = ceil(totalSize / usedCoreNum)，核间最多差 1 个 Q。
    // TND：用 packed validT1，不用物理 padding 长度，避免 physicalT1 与 validT1 把 epoch 数算分叉。
    if (usedCoreNum <= 0) {
        return 0;
    }
    int64_t tokenSize = GetValidTndT1Size();
    if (tokenSize <= 0) {
        return 0;
    }
    return (tokenSize + usedCoreNum - 1) / usedCoreNum;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IterateMmDyV(
    LocalTensor<CALC_TYPE> &mm1ResTensor, const GlobalTensor<INPUT_TYPE> &selectedVWorkSpaceGm, FagConstInfo &constInfo,
    FagRunInfo &runInfo)
{
    if (constInfo.isHeadNLe64) {
        cubeBlockNLe64.IterateMmDyV(mm1ResTensor, selectedVWorkSpaceGm, constInfo, runInfo);
    } else {
        cubeBlock.IterateMmDyV(mm1ResTensor, selectedVWorkSpaceGm, constInfo, runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IterateMmQK(
    LocalTensor<CALC_TYPE> &mm2ResTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm, FagConstInfo &constInfo,
    FagRunInfo &runInfo)
{
    if (constInfo.isHeadNLe64) {
        cubeBlockNLe64.IterateMmQK(mm2ResTensor, selectedKWorkSpaceGm, constInfo, runInfo);
    } else {
        cubeBlock.IterateMmQK(mm2ResTensor, selectedKWorkSpaceGm, constInfo, runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IterateMmDsK(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, const GlobalTensor<INPUT_TYPE> &selectedKWorkSpaceGm,
    BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf, FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (constInfo.isHeadNLe64) {
        cubeBlockNLe64.template IterateMmDsK<T, IS_WRITE_UB>(outTensor, selectedKWorkSpaceGm, dSL1Buf, constInfo,
                                                             runInfo);
    } else {
        cubeBlock.template IterateMmDsK<T, IS_WRITE_UB>(outTensor, selectedKWorkSpaceGm, dSL1Buf, constInfo, runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IterateMmDsQ(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (constInfo.isHeadNLe64) {
        cubeBlockNLe64.template IterateMmDsQ<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
    } else {
        cubeBlock.template IterateMmDsQ<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IterateMmPDy(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, BuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if (constInfo.isHeadNLe64) {
        cubeBlockNLe64.template IterateMmPDy<T, IS_WRITE_UB>(outTensor, pL1Buf, constInfo, runInfo);
    } else {
        cubeBlock.template IterateMmPDy<T, IS_WRITE_UB>(outTensor, pL1Buf, constInfo, runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SyncALLCores()
{
    SyncAll<false>();
}
} // namespace SfagBaseApi
#endif
