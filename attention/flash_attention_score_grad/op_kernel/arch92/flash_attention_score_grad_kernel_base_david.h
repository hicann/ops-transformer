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
 * \file flash_attention_score_grad_kernel_base_david.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_DAVID_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_DAVID_H

#include "flash_attention_score_grad_common_david.h"

namespace FagBaseApi {

struct TndValidCursor {
    int64_t globalTaskRank = -1;
    int64_t bIdx = 0;
    int64_t n2Idx = 0;
    int64_t s2Idx = -1;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    int64_t beginS1 = 0;
    int64_t endS1 = 0;
    int64_t innerTileIdx = 0;
    bool valid = false;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR deqScaleO,
                                GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR dpse, GM_ADDR dsink, GM_ADDR workspace, FagTilingType ordTilingData,
                                TPipe* pipeIn);
    __aicore__ inline void InitCVCommonBuffer();
    __aicore__ inline void InitCVCommonGlobalBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR deqScaleQ,
                                                    GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                                    GM_ADDR workspace);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetOptionalInfo();
    __aicore__ inline void SetSwizzleConstInfo();
    __aicore__ inline void SetRunInfo(FagRunInfo& runInfo, int64_t taskId, int64_t index);
    __aicore__ inline void Process();
    __aicore__ inline bool IsValid(FagRunInfo& runInfo, int64_t taskId, int64_t index);
    __aicore__ inline bool IsValidForDeter(FagRunInfo& runInfo, int64_t taskId, int64_t index);
    __aicore__ inline void UpdateToken(FagRunInfo& runInfo, int64_t bIdx);
    __aicore__ inline bool CheckIsValidBlock(FagRunInfo& runInfo, int64_t baseIdx, int64_t s1oDimIdx,
                                             int64_t s2oDimIdx);
    __aicore__ inline bool CheckIsValidBlockForDeter(FagRunInfo& runInfo, int64_t baseIdx, int64_t s1oDimIdx,
                                                     int64_t s2oDimIdx, int64_t taskId);
    __aicore__ inline int64_t GetNextValidIdx(FagRunInfo& runInfo, int64_t taskId, int64_t startIndex,
                                              int64_t curLoopIdx);
    __aicore__ inline int64_t GetNextValidIdxForSwizzle(FagRunInfo& runInfo, int64_t loopIdx);
    __aicore__ inline int64_t FloorDivSigned(int64_t x, int64_t y);
    __aicore__ inline int64_t CeilDivSigned(int64_t x, int64_t y);
    __aicore__ inline int64_t ClampSigned(int64_t x, int64_t low, int64_t high);
    __aicore__ inline int64_t ArithmeticPrefix(int64_t count, int64_t first, int64_t diff);
    __aicore__ inline int64_t SolveArithmeticOffset(int64_t compactIdx, int64_t first, int64_t diff, int64_t length);
    // tnd swizzle
    __aicore__ inline void LoadTndValidBatch(int64_t bIdx);
    __aicore__ inline void CalcTndValidColumnRange(FagRunInfo& runInfo);
    __aicore__ inline bool MoveToNextTndValidColumn(FagRunInfo& runInfo);
    __aicore__ inline bool LocateTndValidColumnTask(FagRunInfo& runInfo, int64_t taskRank);
    __aicore__ inline bool AdvanceTndValidColumnTask(FagRunInfo& runInfo, int64_t taskDelta);
    __aicore__ inline bool EmitCurrentTndValidColumnTile(FagRunInfo& runInfo, int64_t innerTileIdx);
    __aicore__ inline int64_t GetTndValidColumnTaskRank(int64_t round);
    __aicore__ inline int64_t GetNextValidIdxForTndBn2gs1s2Swizzle(FagRunInfo& runInfo, int64_t loopIdx);

    __aicore__ inline int64_t GetDeqScaleQOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t GetDeqScaleKOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t GetQueryOffset(FagRunInfo& runInfo, bool isMm1Mm2);
    __aicore__ inline int64_t GetQueryRopeOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t GetDxOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t GetKeyOffset(FagRunInfo& runInfo, bool isMm1Mm2);
    __aicore__ inline int64_t GetKeyRopeOffset(FagRunInfo& runInfo);
    __aicore__ inline int64_t GetValueOffset(FagRunInfo& runInfo);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t& actualSeqQlen, int64_t& actualSeqKvlen);
    __aicore__ inline void CheckS1RangeInBn2(int64_t taskId);
    __aicore__ inline ChildClass* GetDerived()
    {
        return static_cast<ChildClass*>(this);
    }
    constexpr static uint32_t QUANT_BLOCK_SIZE = 256;
    constexpr static bool IS_FP8_INPUT = is_f8_v<T1>;
    constexpr static bool IS_FP32_INPUT = is_f32_v<T1>;

    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = 64;
    constexpr static uint32_t VECTOR_BASEN = 128;
    constexpr static uint8_t ALIGN_NUM_32 = 32;
    constexpr static uint8_t VEC_CORE_NUM_64 = 64;
    constexpr static uint32_t BITS_EACH_UINT64 = 64;
    constexpr static uint32_t ROPE_D_RATIO = 3;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(T1);
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static int64_t OUTINDEX = -1;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(T1);
    constexpr static uint32_t UB_BUF_NUM = IS_FP32_INPUT ? 2 : 3;

    constexpr static uint8_t DETER_TILING_SPLIT_MODE =
        (CUBE_BASEM == CUBE_BASEN ? 0 : (CUBE_BASEM > CUBE_BASEN ? 2 : 1));
    constexpr static uint32_t DETER_CUBE_BASEM = CUBE_BASEM < CUBE_BASEN ? CUBE_BASEN : CUBE_BASEM;
    constexpr static uint32_t DETER_CUBE_BASEN = CUBE_BASEM > CUBE_BASEN ? CUBE_BASEM : CUBE_BASEN;
    constexpr static uint32_t IS_NORMAL_SWIZZLE = !IS_TND && SPLIT_AXIS == BN2GS1S2 && DETER_SPARSE_TYPE == NO_DETER;
    constexpr static uint32_t IS_TND_SWIZZLE =
        IS_TND && SPLIT_AXIS == BN2GS1S2 && IS_SWIZZLE && DETER_SPARSE_TYPE == NO_DETER;
    constexpr static uint32_t TND_VALID_S2_COLUMN_SIMPLE_CYCLIC = 0;
    constexpr static uint32_t TND_VALID_S2_COLUMN_SERPENTINE = 1;

protected:
    using DsType = std::conditional_t<is_e4m3_v<T1>, bfloat16_t, T1>;

    TPipe* pipe;
    TBuf<> mm1ResBuf[UB_BUF_NUM];
    TBuf<> mm2ResBuf[UB_BUF_NUM];
    __gm__ uint8_t* prefixNAddr;
    __gm__ uint8_t* actualSeqQlenAddr;
    __gm__ uint8_t* actualSeqKvlenAddr;

    uint32_t vBlockIdx = 0;
    uint32_t cBlockIdx = 0;
    uint32_t vSubBlockIdx = 0;
    int64_t lastS2oCvDimIdx = -1; // 上一次的s2方向基本块idx
    int64_t lastBdimIdx = -1;     // 上一次的b方向基本块idx
    int64_t lastN2dimIdx = -1;    // 上一次的n2方向基本块idx
    bool isLastLoop = false;
    int64_t s2CvBegin = 0;
    int64_t s2CvEnd = 0;
    int64_t actualCalcS1Token = 0; // 转换后实际计算使用的S1Token
    int64_t actualCalcS2Token = 0;
    uint32_t sinkOptional = 0;

    // BN2扩展模板判断S1轴有效始终位置
    typename std::conditional<IS_BN2_MULTIBLK, bool[3], std::nullptr_t>::type isLastS2Outer;
    typename std::conditional<IS_BN2_MULTIBLK, bool[3], std::nullptr_t>::type isFirstS2Outer;
    typename std::conditional<IS_BN2_MULTIBLK, Bn2MultiBlkInfo, std::nullptr_t>::type multiBlkInfo;

    FagTilingType tilingData;

    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    // input
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm, deqScaleOGm;

    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, sfmgWorkspaceGm;
    GlobalTensor<std::conditional_t<SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK, T1, float>> dvWorkSpaceGm;

    GlobalTensor<float> deterGm;
    GlobalTensor<int64_t> deterOffsetGm;

    // 同步事件
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t QKDXV_RES_READY[3] = {0, 1, 2};
    static constexpr uint64_t QKDXV_RES_RELEASE[3] = {3, 4, 5};
    static constexpr uint64_t DS_READY = 8;
    static constexpr uint64_t POUT_READY = 9;
    static constexpr uint64_t DS_RELEASE = 10;
    static constexpr uint64_t DETER_GM_INIT = 7;
    static constexpr uint64_t POUT_RELEASE = 11;
    static constexpr uint64_t ALL_DONE = 12;
    static constexpr uint64_t SYNC_C4_TO_V6_FLAG = 13;
    static constexpr uint64_t SYNC_DETER_FIX_FLAG = 15;
    // BN2BLK使用
    static constexpr uint64_t BN2_BLK_C3_TO_V3 = 6;  // dq to mulsAndCast
    static constexpr uint64_t BN2_BLK_C4_TO_V3 = 7;  // dk to mulsAndCast
    static constexpr uint64_t BN2_BLK_C5_TO_V3 = 14; // dv to cast
    static constexpr uint64_t BN2_BLK_V3_TO_C3 = 15; // dq mulsAndCast to dq fixp

    FagConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    DropMaskInfo dropInfo;

    int64_t deterGmOffset = 0;
    typename std::conditional<IS_TND_SWIZZLE, TndValidCursor, std::nullptr_t>::type tndValidCursor{};
    typename std::conditional<IS_TND_SWIZZLE, bool, std::nullptr_t>::type tndValidCursorInitialized{};
    int64_t swizzleLoopIdx = 0;
    bool enableDenseSwizzle = false;
    bool enableCasualSwizzle = false;
    bool filledBySwizzle = false;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR deqScaleO, GM_ADDR queryRope,
    GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dsink, GM_ADDR workspace,
    FagTilingType ordTilingData, TPipe* pipeIn)
{
    // init current core tilingInfo
    if ASCEND_IS_AIV {
        vBlockIdx = GetBlockIdx();
        cBlockIdx = vBlockIdx / CV_CORE_RATIO;
        vSubBlockIdx = get_subblockid();
    } else {
        cBlockIdx = GetBlockIdx();
    }
    tilingData = ordTilingData;
    pipe = pipeIn;

    // 填充基础参数
    SetConstInfo();
    actualCalcS1Token = constInfo.s1Token;
    actualCalcS2Token = constInfo.s2Token;

    prefixNAddr = prefixN;
    actualSeqQlenAddr = actualSeqQlen;
    actualSeqKvlenAddr = actualSeqKvlen;
    constInfo.seqS1_addr = actualSeqQlen;
    constInfo.seqS2_addr = actualSeqKvlen;
    InitCVCommonGlobalBuffer(dq, dk, dv, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, workspace);
    InitCVCommonBuffer();
    // optional add

    if ASCEND_IS_AIV {
        // pass params to vector block
        SetOptionalInfo();
        vecBlock.SetVecBlockParams(pipeIn, tilingData, vBlockIdx, cBlockIdx, vSubBlockIdx, attenMaskInfo, pseInfo,
                                   dropInfo);
        vecBlock.InitUbBuffer();
        vecBlock.InitGlobalBuffer(value, dy, y, pseShift, dropMask, attenMask, softmaxMax, softmaxSum, deqScaleQ,
                                  deqScaleK, deqScaleV, deqScaleDy, dq, dk, dv, sink, dsink, workspace);
        vecBlock.InitAsync();
    }

    if ASCEND_IS_AIC {
        // pass params to cube block
        cubeBlock.SetCubeBlockParams(pipeIn, tilingData);
        cubeBlock.InitCubeBuffer();
        cubeBlock.InitGlobalBuffer(query, key, value, dy, queryRope, keyRope, dq, dk, dv, workspace);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(DS_RELEASE);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(POUT_RELEASE);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Process()
{
    GetDerived()->Process();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonGlobalBuffer(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
    GM_ADDR workspace)
{
    deqScaleQGm.SetGlobalBuffer((__gm__ float*)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float*)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float*)deqScaleV);

    if constexpr (!IS_FP32_INPUT) {
        if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
            uint64_t qPostBlockTotal = CUBE_BASEM * HEAD_DIM_ALIGN * 2 * GetBlockNum(); // 2 pingpong
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + cBlockIdx * CUBE_BASEM * HEAD_DIM_ALIGN * 2);
            uint64_t workspaceOffsets = qPostBlockTotal * sizeof(float);
            dkWorkSpaceGm.SetGlobalBuffer(
                (__gm__ float*)workspace +
                (workspaceOffsets + cBlockIdx * CUBE_BASEN * HEAD_DIM_ALIGN * 2 * sizeof(T2)) / sizeof(T2));
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)dv);
        } else {
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                          tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                          tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace +
                                          tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));
            if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
                deterGm.SetGlobalBuffer((__gm__ float*)workspace +
                                        tilingData->postTilingData.deterGmOffset / sizeof(T2));
                deterOffsetGm.SetGlobalBuffer((__gm__ int64_t*)workspace +
                                              tilingData->postTilingData.deterWorkSpaceOffset / sizeof(int64_t));
            }
        }
    } else {
        // input type fp32, dq dk dv write to output gm directly
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)dq);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)dk);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)dv);
        if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            deterGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->postTilingData.deterGmOffset / sizeof(T2));
            deterOffsetGm.SetGlobalBuffer((__gm__ int64_t*)workspace +
                                          tilingData->postTilingData.deterWorkSpaceOffset / sizeof(int64_t));
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonBuffer()
{
    for (int32_t i = 0; i < UB_BUF_NUM; i++) {
        pipe->InitBuffer(mm1ResBuf[i], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
        pipe->InitBuffer(mm2ResBuf[i], VECTOR_BASEM * VECTOR_BASEN * sizeof(T2));
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetOptionalInfo()
{
    if constexpr (IS_ATTEN_MASK) {
        attenMaskInfo.attenMaskShapeType = tilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType;
        attenMaskInfo.compressMode = tilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
        attenMaskInfo.attenMaskS2Size = tilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;
        attenMaskInfo.preTokens = tilingData->s1s2BNGS1S2BaseParams.s1Token;
        attenMaskInfo.nextTokens = tilingData->s1s2BNGS1S2BaseParams.s2Token;
        attenMaskInfo.prefixNAddr = prefixNAddr;
        attenMaskInfo.bandIndex = tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx;
    }

    if ASCEND_IS_AIV {
        if constexpr (IS_PSE) {
            uint32_t pseShapeType = tilingData->s1s2BNGS1S2BaseParams.pseShapeType;
            pseInfo.pseBSize =
                (pseShapeType == PSE_SHAPE_TYPE_1NSS || pseShapeType == PSE_SHAPE_TYPE_1NHS) ? 1 : constInfo.bSize;
            pseInfo.pseS1Size = PSE_COMPRESS_H; // 1024
            pseInfo.pseS2Size = constInfo.commonConstInfo.s2Size;
            pseInfo.pseLayoutType = tilingData->s1s2BNGS1S2BaseParams.pseLayoutType;
            pseInfo.pseEncodeType =
                (pseShapeType == PSE_SHAPE_TYPE_BNHS || pseShapeType == PSE_SHAPE_TYPE_1NHS) ? pseEncodeALibiS2Full : 0;
            pseInfo.pseType = tilingData->s1s2BNGS1S2BaseParams.pseType;
            pseInfo.qStartIdx = tilingData->s1s2BNGS1S2BaseParams.qStartIdx;
            pseInfo.kvStartIdx = tilingData->s1s2BNGS1S2BaseParams.kvStartIdx;
        }

        if constexpr (IS_DROP) {
            dropInfo.dropMaskOuter = tilingData->s1s2BNGS1S2BaseParams.dropMaskOuter;
            dropInfo.seed = tilingData->s1s2BNGS1S2BaseParams.seed;
            dropInfo.offset = tilingData->s1s2BNGS1S2BaseParams.offset;
            dropInfo.keepProb = tilingData->s1s2BNGS1S2BaseParams.keepProb;
            dropInfo.keepProbUint8 = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.keepProbUint8);
            dropInfo.boolMode = tilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetSwizzleConstInfo()
{
    if constexpr (IS_TND_SWIZZLE) {
        constInfo.totalValidS2TaskNum = tilingData->tndSwizzleParam.totalValidS2TaskNum;
        constInfo.validS2ActiveCoreNum = tilingData->tndSwizzleParam.activeCoreNum;
        constInfo.columnScheduleMode = tilingData->tndSwizzleParam.columnScheduleMode;
    } else if constexpr (IS_NORMAL_SWIZZLE) {
        if constexpr (!IS_SWIZZLE) {
            return;
        }
        // swizzle
        enableDenseSwizzle = (!IS_ATTEN_MASK || tilingData->s1s2BNGS1S2BaseParams.sparseType ==
                                                    static_cast<uint8_t>(SparseType::DENSE)) &&
                             constInfo.commonConstInfo.s1Size > M_SWIZZLE_SIZE &&
                             constInfo.commonConstInfo.s2Size > N_SWIZZLE_SIZE;
        enableCasualSwizzle =
            IS_ATTEN_MASK && tilingData->s1s2BNGS1S2BaseParams.sparseType == static_cast<uint8_t>(SparseType::CASUAL) &&
            constInfo.commonConstInfo.s1Size > M_SWIZZLE_SIZE;

        uint32_t maxContinuousBlockNum =
            constInfo.commonConstInfo.s1Size < MIN_SWIZZLE_S1 ?
                MAX_CONTINUOUS_BLOCK_NUM :
                (constInfo.commonConstInfo.s1Size / MIN_SWIZZLE_S1) * BASE_SWIZZLE_BLOCK_NUM;
        if (enableDenseSwizzle || enableCasualSwizzle) {
            maxContinuousBlockNum = SWIZZLE_CONTINUOUS_BLOCK_NUM;
        }

        constInfo.continuousBlockNum = tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen > maxContinuousBlockNum ?
                                           maxContinuousBlockNum :
                                           tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetConstInfo()
{
    constInfo.s1Token = tilingData->s1s2BNGS1S2BaseParams.s1Token;
    constInfo.s2Token = tilingData->s1s2BNGS1S2BaseParams.s2Token;
    constInfo.sparseMode = tilingData->s1s2BNGS1S2BaseParams.sparseMode;
    // split info
    constInfo.s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    constInfo.s1CvTail = tilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    constInfo.s1Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s1Tail;
    constInfo.s2Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s2Tail;
    constInfo.s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;

    constInfo.commonConstInfo.s1BaseSize = CUBE_BASEM;
    constInfo.commonConstInfo.s1VecBaseSize = VECTOR_BASEM;
    constInfo.commonConstInfo.s2BaseSize = CUBE_BASEN;
    constInfo.bSize = tilingData->s1s2BNGS1S2BaseParams.b;
    constInfo.n2Size = tilingData->s1s2BNGS1S2BaseParams.n2;
    constInfo.commonConstInfo.gSize = tilingData->s1s2BNGS1S2BaseParams.g;
    constInfo.commonConstInfo.s1Size = tilingData->s1s2BNGS1S2BaseParams.s1;
    constInfo.commonConstInfo.s2Size = tilingData->s1s2BNGS1S2BaseParams.s2;
    constInfo.commonConstInfo.dSize = tilingData->s1s2BNGS1S2BaseParams.d;
    constInfo.commonConstInfo.dSizeV = tilingData->s1s2BNGS1S2BaseParams.d1;
    constInfo.commonConstInfo.layoutType = tilingData->s1s2BNGS1S2BaseParams.layout;

    constInfo.commonConstInfo.s1D = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.gS1D = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1D;
    constInfo.commonConstInfo.n2GS1D = constInfo.n2Size * constInfo.commonConstInfo.gS1D;
    constInfo.commonConstInfo.s2D = constInfo.commonConstInfo.s2Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.n2S2D = constInfo.n2Size * constInfo.commonConstInfo.s2D;
    constInfo.commonConstInfo.s1S2 = constInfo.commonConstInfo.s1Size * constInfo.commonConstInfo.s2Size;
    constInfo.commonConstInfo.gS1 = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s1Size;
    constInfo.commonConstInfo.gS1S2 = constInfo.commonConstInfo.gS1 * constInfo.commonConstInfo.s2Size;
    constInfo.commonConstInfo.n2GS1 = constInfo.n2Size * constInfo.commonConstInfo.gS1;
    constInfo.commonConstInfo.gD = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.n2D = constInfo.n2Size * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.bN2D = constInfo.bSize * constInfo.commonConstInfo.n2D;
    constInfo.commonConstInfo.n2G = constInfo.n2Size * constInfo.commonConstInfo.gSize;
    constInfo.commonConstInfo.n2GD = constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSize;
    constInfo.commonConstInfo.bN2GD = constInfo.bSize * constInfo.commonConstInfo.n2GD;
    constInfo.commonConstInfo.gS2 = constInfo.commonConstInfo.gSize * constInfo.commonConstInfo.s2Size;
    // for D_V
    if constexpr (IS_D_NO_EQUAL == 1) {
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
            constInfo.ropeConstInfo.s1Dr = constInfo.commonConstInfo.s1Size * constInfo.dRopeSize;
            constInfo.ropeConstInfo.gS1Dr = constInfo.commonConstInfo.gSize * constInfo.ropeConstInfo.s1Dr;
            constInfo.ropeConstInfo.n2GS1Dr = constInfo.n2Size * constInfo.ropeConstInfo.gS1Dr;
            constInfo.ropeConstInfo.s2Dr = constInfo.commonConstInfo.s2Size * constInfo.dRopeSize;
            constInfo.ropeConstInfo.n2S2Dr = constInfo.n2Size * constInfo.ropeConstInfo.s2Dr;
            constInfo.ropeConstInfo.gDr = constInfo.commonConstInfo.gSize * constInfo.dRopeSize;
            constInfo.ropeConstInfo.n2Dr = constInfo.n2Size * constInfo.dRopeSize;
            constInfo.ropeConstInfo.bN2Dr = constInfo.bSize * constInfo.ropeConstInfo.n2Dr;
            constInfo.ropeConstInfo.n2GDr = constInfo.commonConstInfo.n2G * constInfo.dRopeSize;
            constInfo.ropeConstInfo.bN2GDr = constInfo.bSize * constInfo.ropeConstInfo.n2GDr;
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

    constInfo.scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;
    constInfo.aicCoreNum = tilingData->s1s2BNGS1S2BaseParams.coreNum;
    constInfo.n2GS1oS2o = constInfo.commonConstInfo.n2G * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.gS1oS2o = constInfo.commonConstInfo.gSize * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.s1oS2o = constInfo.s1Outer * constInfo.s2Outer;

    sinkOptional = tilingData->s1s2BNGS1S2BaseParams.sinkOptional;
    constInfo.isSink = sinkOptional;
    constInfo.s1SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s1SinkOuter;
    constInfo.s2SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s2SinkOuter;

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        constInfo.deterConstInfo.noNeedDeter = static_cast<bool>(tilingData->s1s2BNGS1S2SplitCoreParams.noNeedDeter);
        constInfo.deterConstInfo.usedCubeCoreNum =
            static_cast<uint8_t>(tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter);
        // 确定性计算中会用满V核
        if (static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum) == VEC_CORE_NUM_64) {
            // 64V核场景确定性计算中会用满V核
            constInfo.deterConstInfo.usedVectorCoreNum =
                static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum);
        } else {
            // 非64V核场景确定性计算使用V核数量为往下32取整 兼容C核54核
            constInfo.deterConstInfo.usedVectorCoreNum =
                static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum) / ALIGN_NUM_32 * ALIGN_NUM_32;
        }
        // 确定性计算中每个v核处理两行s1
        constInfo.deterConstInfo.eachVecCoreS1Offset =
            static_cast<uint8_t>(CUBE_BASEM / constInfo.deterConstInfo.usedVectorCoreNum);
        constInfo.deterConstInfo.eachVecCoreS2Offset =
            static_cast<uint8_t>(CUBE_BASEN / constInfo.deterConstInfo.usedVectorCoreNum);
        // 确定性计算中，V核开满64核，按照CUBE_BASEM=128的基本块处理是能够整除的
        constInfo.deterConstInfo.dqEachVectorSize =
            static_cast<uint32_t>(BASE_DQ_SIZE / constInfo.deterConstInfo.usedVectorCoreNum);
        constInfo.deterConstInfo.dkvEachVectorSize =
            static_cast<uint32_t>(BASE_DKV_SIZE / constInfo.deterConstInfo.usedVectorCoreNum);
        if constexpr (IS_TND) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
            constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
            constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
            constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gD;
            constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.dSize;
            constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.n2GD;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSizeV;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSizeV;
                constInfo.mm2Ka = constInfo.commonConstInfo.dSize;
                constInfo.mm2Kb = constInfo.commonConstInfo.dSize;
                constInfo.deterConstInfo.deterBStride = constInfo.commonConstInfo.n2GS1D;
                constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gS1D;
                constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.s1D;
                constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.bN2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.bN2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.bN2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.bN2D;
                constInfo.deterConstInfo.deterBStride = constInfo.commonConstInfo.n2GD;
                constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gD;
                constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.dSize;
                constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.bN2GD;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
                constInfo.deterConstInfo.deterBStride = constInfo.commonConstInfo.n2GS1D;
                constInfo.deterConstInfo.deterN2Stride = constInfo.commonConstInfo.gD;
                constInfo.deterConstInfo.deterGStride = constInfo.commonConstInfo.dSize;
                constInfo.deterConstInfo.deterS1oStride = constInfo.commonConstInfo.n2GD;
            }
        }
        // arch35 kernel_base.h
        constInfo.deterConstInfo.deterDqkSrcStride =
            static_cast<uint32_t>((HEAD_DIM_ALIGN - constInfo.commonConstInfo.dSize) / FLOAT_BLOCK_SIZE);
        constInfo.deterConstInfo.deterDvSrcStride =
            static_cast<uint32_t>((HEAD_DIM_ALIGN - constInfo.commonConstInfo.dSizeV) / FLOAT_BLOCK_SIZE);
        constInfo.deterConstInfo.deterDqDstStride =
            static_cast<uint32_t>((constInfo.mm2Ka - constInfo.commonConstInfo.dSize) * sizeof(T2));
        constInfo.deterConstInfo.deterDkDstStride =
            static_cast<uint32_t>((constInfo.mm2Kb - constInfo.commonConstInfo.dSize) * sizeof(T2));
        constInfo.deterConstInfo.deterDvDstStride =
            static_cast<uint32_t>((constInfo.commonConstInfo.mm1Kb - constInfo.commonConstInfo.dSizeV) * sizeof(T2));
        constInfo.deterConstInfo.deterVecCoreS1Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS1Offset * constInfo.mm2Ka;
        constInfo.deterConstInfo.deterDkVecCoreS2Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS2Offset * constInfo.mm2Kb;
        constInfo.deterConstInfo.deterDvVecCoreS2Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS2Offset * constInfo.commonConstInfo.mm1Kb;

        constInfo.deterConstInfo.eventIDScalarToMte2 = (event_t)GetTPipePtr()->AllocEventID<HardEvent::S_MTE2>();
        constInfo.deterConstInfo.eventIDMte2ToScalar = (event_t)GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
        constInfo.deterConstInfo.eventIDScalarToMte3 = (event_t)GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>();
        constInfo.deterConstInfo.eventIDMte3ToScalar = (event_t)GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>();
        constInfo.deterConstInfo.eventIDMte3ToMte2 = (event_t)GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        constInfo.deterConstInfo.eventIDMte2ToMte3 = (event_t)GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
        if constexpr (IS_ROPE) {
            constInfo.mm2Ka = constInfo.mm2Ka / ROPE_D_RATIO << 1;
            constInfo.mm2Kb = constInfo.mm2Kb / ROPE_D_RATIO << 1;
        }
    } else {
        if constexpr (IS_TND) {
            constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
            constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
            constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
            constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
        } else {
            if (constInfo.commonConstInfo.layoutType == BNGSD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.dSizeV;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.dSizeV;
                constInfo.mm2Ka = constInfo.commonConstInfo.dSize;
                constInfo.mm2Kb = constInfo.commonConstInfo.dSize;
            } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.bN2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.bN2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.bN2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.bN2D;
            } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
                constInfo.commonConstInfo.mm1Ka = constInfo.commonConstInfo.n2GDv;
                constInfo.commonConstInfo.mm1Kb = constInfo.commonConstInfo.n2Dv;
                constInfo.mm2Ka = constInfo.commonConstInfo.n2GD;
                constInfo.mm2Kb = constInfo.commonConstInfo.n2D;
            }
        }
        if constexpr (IS_ROPE) {
            constInfo.mm2Ka = constInfo.mm2Ka / ROPE_D_RATIO << 1;
            constInfo.mm2Kb = constInfo.mm2Kb / ROPE_D_RATIO << 1;
        }
    }
    constInfo.commonConstInfo.subBlockIdx = vSubBlockIdx;

    uint32_t tmp = 0xFF7FFFFF;
    constInfo.attenMaskMinValue = *((float*)&tmp);
    constInfo.keepProb = tilingData->s1s2BNGS1S2BaseParams.keepProb;

    constInfo.sfmgMaxLoopSize = VECTOR_BASEM * VECTOR_BASEN / HEAD_DIM_ALIGN; // softmaxGrad每次最大能处理的m轴大小
    constInfo.dAlignToBlock = AlignTo(constInfo.commonConstInfo.dSizeV, INPUT_BLOCK_NUM);
    constInfo.tndMaxSumLayout = tilingData->s1s2BNGS1S2BaseParams.tndMaxSumLayout;
    SetSwizzleConstInfo();
    GetDerived()->SetUniqueConstInfo(constInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType,
                                                         VecBlockType>::GetSeqQlenKvlenByBidx(int64_t bIdx,
                                                                                              int64_t& actualSeqQlen,
                                                                                              int64_t& actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t*)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t*)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t*)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t*)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen = ((__gm__ int64_t*)actualSeqKvlenAddr)[bIdx] - ((__gm__ int64_t*)actualSeqKvlenAddr)[bIdx - 1];
    }
    return;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::UpdateToken(
    FagRunInfo& runInfo, int64_t bIdx)
{
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    if constexpr (IS_ATTEN_MASK) {
        int64_t actualS1Len;
        int64_t actualS2Len;
        if (constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND && bIdx != tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
            actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
        } else if (constInfo.sparseMode == BAND_LEFT_UP_CASUAL &&
                   bIdx != tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx) {
            actualCalcS1Token = INT32_MAX;
            actualCalcS2Token = 0;
        } else if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == BAND ||
                   (constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND &&
                    bIdx == tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx) ||
                   (constInfo.sparseMode == BAND_LEFT_UP_CASUAL &&
                    bIdx == tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx)) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            actualCalcS1Token = constInfo.s1Token + actualS1Len - actualS2Len;
            actualCalcS2Token = constInfo.s2Token - actualS1Len + actualS2Len;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CheckIsValidBlock(
    FagRunInfo& runInfo, int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oDimIdx)
{
    int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
    int64_t s2IdxRight = (s2oDimIdx + 1) * CUBE_BASEN;
    int64_t s2IgnoredEndLen =
        static_cast<int64_t>(constInfo.commonConstInfo.s1Size) - static_cast<int64_t>(CUBE_BASEM * (s1oDimIdx + 1));
    int64_t s2EndLen = 0;
    if (static_cast<int64_t>(constInfo.commonConstInfo.s2Size) > s2IgnoredEndLen) {
        s2EndLen = static_cast<int64_t>(constInfo.commonConstInfo.s2Size) - s2IgnoredEndLen;
    } else {
        s2EndLen = 0;
    }

    if (constInfo.sparseMode == PREFIX || constInfo.sparseMode == PREFIX_COMPRESS) {
        int64_t curBIdx = baseIdx / constInfo.n2GS1oS2o;
        s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t*)prefixNAddr)[curBIdx]),
                       static_cast<int64_t>(constInfo.commonConstInfo.s2Size));
    }
    bool isValid = s2IdxLeft < s2EndLen;
    if (isValid) {
        s2CvBegin = s2IdxLeft;
        s2CvEnd = s2CvBegin + CUBE_BASEN;         // 非尾块s2按照+CUBE_BASEN处理
        if (s2oDimIdx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
            s2CvEnd = s2CvBegin + constInfo.s2Tail;
        }
    }
    return isValid;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CheckIsValidBlockForDeter(
    FagRunInfo& runInfo, int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oDimIdx, int64_t taskId)
{
    int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
    int64_t s2IdxRight = (s2oDimIdx + 1) * CUBE_BASEN;
    int64_t s2IgnoredEndLen =
        static_cast<int64_t>(constInfo.commonConstInfo.s1Size) - static_cast<int64_t>(CUBE_BASEM * (s1oDimIdx + 1));
    int64_t s2EndLen = 0;
    if (static_cast<int64_t>(constInfo.commonConstInfo.s2Size) > s2IgnoredEndLen) {
        s2EndLen = static_cast<int64_t>(constInfo.commonConstInfo.s2Size) - s2IgnoredEndLen;
    } else {
        s2EndLen = 0;
    }

    if (constInfo.sparseMode == PREFIX || constInfo.sparseMode == PREFIX_COMPRESS) {
        int64_t curBIdx = baseIdx / constInfo.n2GS1oS2o;
        s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t*)prefixNAddr)[curBIdx]),
                       static_cast<int64_t>(constInfo.commonConstInfo.s2Size));
    } else {
        if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
            s2EndLen = Min(s2EndLen, constInfo.commonConstInfo.s2Size);
        }
    }
    bool isValid = s2IdxLeft < s2EndLen;
    if (isValid) {
        s2CvBegin = s2IdxLeft;
        s2CvEnd = s2CvBegin + CUBE_BASEN;         // 非尾块s2按照+CUBE_BASEN处理
        if (s2oDimIdx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
            s2CvEnd = s2CvBegin + constInfo.s2Tail;
        }
    }
    return isValid;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetRunInfo(
    FagRunInfo& runInfo, int64_t taskId, int64_t index)
{
    runInfo.s2CvBegin = s2CvBegin;
    runInfo.s2CvEnd = s2CvEnd;

    runInfo.taskPingPong = taskId % 2;
    if constexpr (IS_TND_SWIZZLE) {
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        int64_t tndS1PrefixSum =
            (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                                ((__gm__ int64_t*)actualSeqQlenAddr)[runInfo.commonRunInfo.boIdx - 1]);
        int64_t tndS2PrefixSum =
            (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                                ((__gm__ int64_t*)actualSeqKvlenAddr)[runInfo.commonRunInfo.boIdx - 1]);
        runInfo.lastBatchTotalS1BOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2GD;
        runInfo.lastBatchTotalS2BOffset = tndS2PrefixSum * constInfo.commonConstInfo.n2D;
        runInfo.lastBatchTotalS1BOffsetForDv = tndS1PrefixSum * constInfo.commonConstInfo.n2GDv;
        runInfo.lastBatchTotalS2BOffsetForDv = tndS2PrefixSum * constInfo.commonConstInfo.n2Dv;
        runInfo.lastBatchTotalS1S2SizeAlign =
            tilingData->tndSwizzleParam.tndSwizzleS1S2AlignPrefixSum[runInfo.commonRunInfo.boIdx];
        runInfo.lastBatchTotalS1S2Size =
            tilingData->tndSwizzleParam.tndSwizzleS1S2PrefixSum[runInfo.commonRunInfo.boIdx];
        runInfo.lastBatchTotalS2Size = tndS2PrefixSum;
        if constexpr (IS_ROPE) {
            runInfo.lastBatchTotalS1BRopeOffset = tndS1PrefixSum * constInfo.ropeConstInfo.n2GDr;
            runInfo.lastBatchTotalS2BRopeOffset = tndS2PrefixSum * constInfo.ropeConstInfo.n2Dr;
        }

        int64_t s1OuterTmp = (runInfo.commonRunInfo.actualS1Size + CUBE_BASEM - 1) / CUBE_BASEM;
        int64_t s2OuterTmp = (runInfo.commonRunInfo.actualS2Size + VECTOR_BASEN - 1) / VECTOR_BASEN;
        int64_t s1CvTailTmp = runInfo.commonRunInfo.actualS1Size - (s1OuterTmp - 1) * CUBE_BASEM;
        runInfo.commonRunInfo.s1RealSize = (runInfo.commonRunInfo.s1oIdx == s1OuterTmp - 1) ? s1CvTailTmp : CUBE_BASEM;
        runInfo.commonRunInfo.taskId = taskId;
        runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
        runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
        if ASCEND_IS_AIV {
            runInfo.commonRunInfo.halfS1RealSize = runInfo.commonRunInfo.s1RealSize;
            runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
            runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
            runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
            runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
        }
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        runInfo.commonRunInfo.s1SizeThisBatch = actualS1Len;
        runInfo.commonRunInfo.s2SizeThisBatch = actualS2Len;
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
    } else if constexpr (IS_TND) { // todo
        int64_t resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        for (int64_t bIdx = runInfo.lastBatchIdx; bIdx < constInfo.bSize; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + VECTOR_BASEN - 1) / VECTOR_BASEN;
            int64_t totalBaseIdx = constInfo.commonConstInfo.n2G * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * CUBE_BASEM;
                runInfo.commonRunInfo.boIdx = bIdx;
                runInfo.lastBatchIdx = bIdx;
                int64_t bDimTail = resbaseIdx;
                runInfo.commonRunInfo.n2oIdx = bDimTail / (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                int64_t n2DimTail = bDimTail % (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                runInfo.commonRunInfo.goIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                int64_t gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                runInfo.s2oIdx = gDimTail / s1OuterTmp;
                runInfo.commonRunInfo.s1oIdx = gDimTail % s1OuterTmp;
                runInfo.commonRunInfo.s1RealSize =
                    (runInfo.commonRunInfo.s1oIdx == s1OuterTmp - 1) ? s1CvTailTmp : CUBE_BASEM;
                runInfo.commonRunInfo.taskId = taskId;
                runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
                runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
                runInfo.commonRunInfo.halfS1RealSize = runInfo.commonRunInfo.s1RealSize;
                runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
                if (vSubBlockIdx == 1) {
                    runInfo.commonRunInfo.halfS1RealSize =
                        runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
                }
                break;
            } else {
                runInfo.lastBatchTotalBaseIdx += totalBaseIdx;
                resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
                runInfo.lastBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                runInfo.lastBatchTotalS2BOffset += actualS2Len * constInfo.commonConstInfo.n2D;
                runInfo.lastBatchTotalS1BOffsetForDv += actualS1Len * constInfo.commonConstInfo.n2GDv;
                runInfo.lastBatchTotalS2BOffsetForDv += actualS2Len * constInfo.commonConstInfo.n2Dv;
                runInfo.lastBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
                runInfo.lastBatchTotalS1S2Size += actualS1Len * actualS2Len;
                runInfo.lastBatchTotalS2Size += actualS2Len;
                if constexpr (IS_ROPE) {
                    runInfo.lastBatchTotalS1BRopeOffset += actualS1Len * constInfo.ropeConstInfo.n2GDr;
                    runInfo.lastBatchTotalS2BRopeOffset += actualS2Len * constInfo.ropeConstInfo.n2Dr;
                }
            }
        }
        GetSeqQlenKvlenByBidx(runInfo.commonRunInfo.boIdx, actualS1Len, actualS2Len);
        runInfo.commonRunInfo.s1SizeThisBatch = actualS1Len;
        runInfo.commonRunInfo.s2SizeThisBatch = actualS2Len;
        runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
        runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
        runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
    } else {
        if (!filledBySwizzle) {
            runInfo.commonRunInfo.boIdx = index / constInfo.n2GS1oS2o;
            int64_t bDimTail = index % constInfo.n2GS1oS2o;
            runInfo.commonRunInfo.n2oIdx = bDimTail / constInfo.gS1oS2o;
            int64_t n2DimTail = bDimTail % constInfo.gS1oS2o;
            runInfo.commonRunInfo.goIdx = n2DimTail / constInfo.s1oS2o;
            int64_t gDimTail = n2DimTail % constInfo.s1oS2o;
            runInfo.s2oIdx = gDimTail / constInfo.s1Outer;
            runInfo.commonRunInfo.s1oIdx = gDimTail % constInfo.s1Outer;
        }
        runInfo.commonRunInfo.s1RealSize =
            (runInfo.commonRunInfo.s1oIdx == constInfo.s1Outer - 1) ? constInfo.s1CvTail : CUBE_BASEM;
        runInfo.commonRunInfo.taskId = taskId;
        runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
        runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
        runInfo.commonRunInfo.halfS1RealSize = runInfo.commonRunInfo.s1RealSize;
        runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
        if (vSubBlockIdx == 1) {
            runInfo.commonRunInfo.halfS1RealSize =
                runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
        }
        // pse
        runInfo.commonRunInfo.b1SSOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1S2;
        runInfo.commonRunInfo.s2SizeAcc = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s2Size;
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
        runInfo.commonRunInfo.s1SizeThisBatch = constInfo.commonConstInfo.s1Size;
        runInfo.commonRunInfo.s2SizeThisBatch = constInfo.commonConstInfo.s2Size;
        runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1Size *
                                                AlignTo16(constInfo.commonConstInfo.s2Size);
        if constexpr (IS_FP8_INPUT) {
            int64_t deqScaleQGmOffset = GetDeqScaleQOffset(runInfo);
            int64_t deqScaleKGmOffset = GetDeqScaleKOffset(runInfo);
            int64_t deqScaleDyGmOffset = deqScaleQGmOffset;
            int64_t deqScaleVGmOffset = deqScaleKGmOffset;
            runInfo.quantScaleInfo.deqScaleQValue = deqScaleQGm.GetValue(deqScaleQGmOffset);
            runInfo.quantScaleInfo.deqScaleKValue = deqScaleKGm.GetValue(deqScaleKGmOffset);
            runInfo.quantScaleInfo.deqScaleDyValue = deqScaleDyGm.GetValue(deqScaleDyGmOffset);
            runInfo.quantScaleInfo.deqScaleVValue = deqScaleVGm.GetValue(deqScaleVGmOffset);
        }
    }
    // WriteOffsetToGM使用
    runInfo.queryGmOffset = GetQueryOffset(runInfo, true);
    runInfo.queryGmOffsetForBn2Blk = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN;
    runInfo.keyGmOffset = GetKeyOffset(runInfo, true);
    runInfo.keyGmOffsetForBn2Blk = runInfo.s2CvBegin * HEAD_DIM_ALIGN;
    runInfo.valueGmOffset = GetValueOffset(runInfo);
    runInfo.valueGmOffsetForBn2Blk = runInfo.s2CvBegin * HEAD_DIM_ALIGN;
    // BN2扩展模板专用
    if constexpr (IS_BN2_MULTIBLK) {
        runInfo.isLastS2Outer = isLastS2Outer[taskId % 3];
        runInfo.isFirstS2Outer = isFirstS2Outer[taskId % 3];
    }
    runInfo.isS2IdxNoChange = (lastS2oCvDimIdx == runInfo.s2oIdx && lastBdimIdx == runInfo.commonRunInfo.boIdx &&
                               lastN2dimIdx == runInfo.commonRunInfo.n2oIdx);
    if (!runInfo.isS2IdxNoChange) {
        lastS2oCvDimIdx = runInfo.s2oIdx;
        lastBdimIdx = runInfo.commonRunInfo.boIdx;
        lastN2dimIdx = runInfo.commonRunInfo.n2oIdx;
    }

    if (unlikely(sinkOptional)) {
        runInfo.sinkN1Idx =
            runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx;
        uint64_t s1oIdxSink = runInfo.commonRunInfo.s1oIdx;
        // [N, B, S1, S2]
        runInfo.dsinkWorkSpaceOffset =
            runInfo.sinkN1Idx * constInfo.bSize * constInfo.s1SinkOuter * constInfo.s2SinkOuter +
            runInfo.commonRunInfo.boIdx * constInfo.s1SinkOuter * constInfo.s2SinkOuter +
            s1oIdxSink * constInfo.s2SinkOuter + runInfo.s2oIdx;
    }
    GetDerived()->SetUniqueRunInfo(runInfo);
    runInfo.dqGmOffset = GetQueryOffset(runInfo, false);
    runInfo.dkGmOffset = GetKeyOffset(runInfo, false);
    if ASCEND_IS_AIV {
        return;
    }

    runInfo.dxGmOffset = GetDxOffset(runInfo);
    if constexpr (IS_ROPE) {
        runInfo.queryRopeGmOffset = GetQueryRopeOffset(runInfo);
        runInfo.keyRopeGmOffset = GetKeyRopeOffset(runInfo);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IsValid(
    FagRunInfo& runInfo, int64_t taskId, int64_t index)
{
    if constexpr (IS_TND) {
        int64_t resbaseIdx = index - runInfo.lastBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        for (int64_t bIdx = runInfo.lastBatchIdx; bIdx < constInfo.bSize; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            int64_t totalBaseIdx = constInfo.n2Size * constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
                int64_t s2oDimIdx = gDimTail / s1OuterTmp;
                int64_t s1oDimIdx = gDimTail % s1OuterTmp;
                int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
                int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, actualS2Len);
                if constexpr (IS_ATTEN_MASK) {
                    if (constInfo.sparseMode == PREFIX_COMPRESS) {
                        int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
                        int64_t s2IdxRight = (s2oDimIdx + 1) * CUBE_BASEN;
                        int64_t s2IgnoredEndLen = actualS1Len - static_cast<int64_t>(CUBE_BASEM * (s1oDimIdx + 1));
                        int64_t s2EndLen = 0;
                        if (actualS2Len > s2IgnoredEndLen) {
                            s2EndLen = actualS2Len - s2IgnoredEndLen;
                        }
                        if (constInfo.sparseMode == PREFIX || constInfo.sparseMode == PREFIX_COMPRESS) {
                            s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t*)prefixNAddr)[bIdx]), actualS2Len);
                        }
                        bool isValid = s2IdxLeft < s2EndLen;
                        if (isValid) {
                            s2CvBegin = s2IdxLeft;
                            s2CvEnd = s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
                            if (s2oDimIdx == s2OuterTmp - 1) {
                                s2CvEnd = s2CvBegin + actualS2Len - s2oDimIdx * CUBE_BASEN;
                            }
                        }
                        return isValid;
                    }

                    UpdateToken(runInfo, bIdx);
                    int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - actualCalcS1Token, 0);
                    s2SparseLeft = s2SparseLeft >> 6 << 6;
                    int64_t s2SparseRight = AlignTo64(
                        Min(CUBE_BASEM * (s1oDimIdx + 1), constInfo.commonConstInfo.s1Size) + actualCalcS2Token);
                    s2SparseRight = Min(s2SparseRight, actualS2Len);
                    if constexpr (IS_BN2_MULTIBLK) {
                        multiBlkInfo.s2oDimIdx = s2oDimIdx;
                        multiBlkInfo.s2OuterTmp = 0;
                        multiBlkInfo.s2SparseLeft = s2SparseLeft;
                        multiBlkInfo.s2SparseRight = s2SparseRight;
                        CheckS1RangeInBn2(taskId);
                    }
                    bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                    s2CvBegin = s2IdxLeft;
                    s2CvEnd = s2CvBegin + CUBE_BASEN;  // 非尾块s2按照+CUBE_BASEN处理
                    if (s2oDimIdx == s2OuterTmp - 1) { // 默认s2 cv tail相等
                        s2CvEnd = s2CvBegin + actualS2Len - s2oDimIdx * CUBE_BASEN;
                    }
                    return isValid;
                } else {
                    if constexpr (IS_BN2_MULTIBLK) {
                        multiBlkInfo.s2oDimIdx = s2oDimIdx;
                        multiBlkInfo.s2OuterTmp = constInfo.s2Outer;
                        multiBlkInfo.s2SparseLeft = 0;
                        multiBlkInfo.s2SparseRight = 0;
                        CheckS1RangeInBn2(taskId);
                    }
                    s2CvBegin = s2IdxLeft;
                    s2CvEnd = s2IdxRight;
                    return true;
                }
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        return false;
    } else {
        int64_t gDimTail = index % constInfo.s1oS2o;
        int64_t s2oDimIdx = gDimTail / constInfo.s1Outer;
        int64_t s1oDimIdx = gDimTail % constInfo.s1Outer;
        int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
        int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, constInfo.commonConstInfo.s2Size);
        if constexpr (IS_ATTEN_MASK) {
            if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == PREFIX ||
                constInfo.sparseMode == PREFIX_COMPRESS) {
                return CheckIsValidBlock(runInfo, index, s1oDimIdx, s2oDimIdx);
            } else {
                int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - constInfo.s1Token, 0);
                s2SparseLeft = s2SparseLeft >> 6 << 6;
                int64_t s2SparseRight =
                    AlignTo64(Min(CUBE_BASEM * (s1oDimIdx + 1), constInfo.commonConstInfo.s1Size) + constInfo.s2Token);
                s2SparseRight = Min(s2SparseRight, constInfo.commonConstInfo.s2Size);
                if constexpr (IS_BN2_MULTIBLK) {
                    multiBlkInfo.s2oDimIdx = s2oDimIdx;
                    multiBlkInfo.s2OuterTmp = 0;
                    multiBlkInfo.s2SparseLeft = s2SparseLeft;
                    multiBlkInfo.s2SparseRight = s2SparseRight;
                    CheckS1RangeInBn2(taskId);
                }
                bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                s2CvBegin = s2IdxLeft;
                s2CvEnd = s2CvBegin + CUBE_BASEN;         // 非尾块s2按照+CUBE_BASEN处理
                if (s2oDimIdx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
                    s2CvEnd = s2CvBegin + constInfo.s2Tail;
                }
                return isValid;
            }
        } else {
            if constexpr (IS_BN2_MULTIBLK) {
                multiBlkInfo.s2oDimIdx = s2oDimIdx;
                multiBlkInfo.s2OuterTmp = constInfo.s2Outer;
                multiBlkInfo.s2SparseLeft = 0;
                multiBlkInfo.s2SparseRight = 0;
                CheckS1RangeInBn2(taskId);
            }
            s2CvBegin = s2IdxLeft;
            s2CvEnd = s2IdxRight;
            return true;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IsValidForDeter(
    FagRunInfo& runInfo, int64_t taskId, int64_t index)
{
    int64_t gDimTail = index % constInfo.s1oS2o;
    int64_t s2oDimIdx = gDimTail / constInfo.s1Outer;
    int64_t s1oDimIdx = gDimTail % constInfo.s1Outer;
    int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
    int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, constInfo.commonConstInfo.s2Size);
    if constexpr (IS_ATTEN_MASK) {
        if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == PREFIX ||
            constInfo.sparseMode == PREFIX_COMPRESS) {
            return CheckIsValidBlockForDeter(runInfo, index, s1oDimIdx, s2oDimIdx, taskId);
        } else {
            uint64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - constInfo.s1Token, 0);
            s2SparseLeft = s2SparseLeft >> 6 << 6;
            uint64_t s2SparseRight =
                AlignTo64(Min(CUBE_BASEM * (s1oDimIdx + 1), constInfo.commonConstInfo.s1Size) + constInfo.s2Token);
            s2SparseRight = Min(s2SparseRight, constInfo.commonConstInfo.s2Size);
            bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
            s2CvBegin = s2IdxLeft;
            s2CvEnd = s2CvBegin + CUBE_BASEN;         // 非尾块s2按照+CUBE_BASEN处理
            if (s2oDimIdx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
                s2CvEnd = s2CvBegin + constInfo.s2Tail;
            }
            return isValid;
        }
    } else {
        s2CvBegin = s2IdxLeft;
        s2CvEnd = s2IdxRight;
        return true;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdx(
    FagRunInfo& runInfo, int64_t taskId, int64_t blockInnerIdx, int64_t curLoopIdx)
{
    if constexpr (IS_TND_SWIZZLE) {
        return GetNextValidIdxForTndBn2gs1s2Swizzle(runInfo, curLoopIdx);
    } else {
        int64_t nextValidBlockInnerIdx = 0;
        if constexpr (IS_NORMAL_SWIZZLE) {
            if constexpr (IS_SWIZZLE) {
                nextValidBlockInnerIdx = GetNextValidIdxForSwizzle(runInfo, curLoopIdx);
                return nextValidBlockInnerIdx;
            } else {
                nextValidBlockInnerIdx = blockInnerIdx;
                while (!IsValid(runInfo, taskId, nextValidBlockInnerIdx)) {
                    if (nextValidBlockInnerIdx >= tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
                        return -1;
                    }
                    nextValidBlockInnerIdx++;
                }
                if (nextValidBlockInnerIdx >= tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
                    return -1;
                }
                return nextValidBlockInnerIdx;
            }
        } else {
            nextValidBlockInnerIdx = blockInnerIdx;
            while (!IsValid(runInfo, taskId, nextValidBlockInnerIdx)) {
                if (nextValidBlockInnerIdx >= tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
                    return -1;
                }
                nextValidBlockInnerIdx++;
            }
            if (nextValidBlockInnerIdx >= tilingData->s1s2BNGS1S2BlockNumList.blockEnds[cBlockIdx]) {
                return -1;
            }
            return nextValidBlockInnerIdx;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CheckS1RangeInBn2(
    int64_t taskId)
{
    if constexpr (IS_ATTEN_MASK) {
        int64_t nextS2IdxLeft = (multiBlkInfo.s2oDimIdx + 1) * CUBE_BASEN;
        int64_t nextS2IdxRight = (multiBlkInfo.s2oDimIdx + 2) * CUBE_BASEN;
        if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == PREFIX ||
            constInfo.sparseMode == PREFIX_COMPRESS) {
            bool isNextValid = nextS2IdxLeft < multiBlkInfo.s2SparseRight;
            isLastS2Outer[taskId % 3] = !isNextValid;
            isFirstS2Outer[taskId % 3] = (multiBlkInfo.s2oDimIdx == 0);
        } else {
            bool isNextValid = nextS2IdxLeft < multiBlkInfo.s2SparseRight && nextS2IdxRight > multiBlkInfo.s2SparseLeft;
            isLastS2Outer[taskId % 3] = !isNextValid;
            if (multiBlkInfo.s2oDimIdx > 0) {
                int64_t preS2IdxLeft = (multiBlkInfo.s2oDimIdx - 1) * CUBE_BASEN;
                int64_t preS2IdxRight = multiBlkInfo.s2oDimIdx * CUBE_BASEN;
                bool isPreValid =
                    preS2IdxLeft < multiBlkInfo.s2SparseRight && preS2IdxRight > multiBlkInfo.s2SparseLeft;
                isFirstS2Outer[taskId % 3] = !isPreValid;
            } else {
                isFirstS2Outer[taskId % 3] = (multiBlkInfo.s2oDimIdx == 0);
            }
        }
    } else {
        isLastS2Outer[taskId % 3] = (multiBlkInfo.s2oDimIdx == multiBlkInfo.s2OuterTmp - 1) ? true : false;
        isFirstS2Outer[taskId % 3] = (multiBlkInfo.s2oDimIdx == 0);
    }
    return;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::FloorDivSigned(
    int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    if (r != 0 && ((r < 0) != (y < 0))) {
        --q;
    }
    return q;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CeilDivSigned(
    int64_t x, int64_t y)
{
    return -FloorDivSigned(-x, y);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::ClampSigned(
    int64_t x, int64_t low, int64_t high)
{
    return x < low ? low : (x > high ? high : x);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::ArithmeticPrefix(
    int64_t count, int64_t first, int64_t diff)
{
    if (count <= 0) {
        return 0;
    }
    return count * (NUM_TWO * first + (count - 1) * diff) / NUM_TWO;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SolveArithmeticOffset(int64_t compactIdx,
                                                                                                  int64_t first,
                                                                                                  int64_t diff,
                                                                                                  int64_t length)
{
    if (length <= 1) {
        return 0;
    }

    int64_t offset = 0;
    if (diff == 0) {
        offset = compactIdx / first;
    } else {
        float linear = static_cast<float>(NUM_TWO * first - diff);
        float discriminant = linear * linear + static_cast<float>(8 * diff) * static_cast<float>(compactIdx);
        discriminant = discriminant > 0.0f ? discriminant : 0.0f;
        // 4*g/(linear+sqrt(discriminant)) is the stable form of the smaller root.
        float denominator = linear + sqrt(discriminant);
        offset = denominator > 0.0f ? static_cast<int64_t>(4.0f * static_cast<float>(compactIdx) / denominator) : 0;
    }

    offset = ClampSigned(offset, 0, length - 1);
    // Fixed-count Newton refinement removes float-root error without data-dependent iteration.
    for (int64_t i = 0; i < NUM_THREE; ++i) {
        int64_t residual = ArithmeticPrefix(offset, first, diff) - compactIdx;
        int64_t derivativeTwice = NUM_TWO * first - diff + NUM_TWO * diff * offset;
        if (derivativeTwice <= 0) {
            break;
        }
        int64_t correction = NUM_TWO * residual / derivativeTwice;
        if (correction == 0) {
            break;
        }
        offset = ClampSigned(offset - correction, 0, length - 1);
    }
    // At most three exact unit corrections are needed after Newton refinement.
    for (int64_t i = 0; i < NUM_THREE; ++i) {
        if (offset + 1 < length && ArithmeticPrefix(offset + 1, first, diff) <= compactIdx) {
            ++offset;
        } else if (offset > 0 && ArithmeticPrefix(offset, first, diff) > compactIdx) {
            --offset;
        } else {
            break;
        }
    }
    return offset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForSwizzle(
    FagRunInfo& runInfo, int64_t loopIdx)
{
    int64_t blockGroupIdx = loopIdx / constInfo.continuousBlockNum;      // 第几组
    int64_t blockGroupInnerIdx = loopIdx % constInfo.continuousBlockNum; // 组内第几个

    int64_t globalIdx = constInfo.aicCoreNum * constInfo.continuousBlockNum * blockGroupIdx +
                        cBlockIdx * constInfo.continuousBlockNum + blockGroupInnerIdx;
    int64_t totalPerBatchNum = static_cast<int64_t>(tilingData->s1s2BNGS1S2BaseParams.totalPerBatchNum);
    uint8_t sparseType = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.sparseType);

    int64_t bIdx = globalIdx / totalPerBatchNum;
    if (bIdx >= constInfo.bSize * constInfo.n2Size * constInfo.commonConstInfo.gSize) {
        return -1;
    }
    int64_t gDimTail = globalIdx % totalPerBatchNum;
    int64_t s2Idx = 0;
    int64_t s1Idx = 0;
    if constexpr (!IS_ATTEN_MASK) {
        s2Idx = gDimTail / constInfo.s1Outer;
        s1Idx = gDimTail % constInfo.s1Outer;
    } else {
        if (sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
            s2Idx = gDimTail / constInfo.s1Outer;
            s1Idx = gDimTail % constInfo.s1Outer;
        } else if (sparseType == static_cast<uint8_t>(SparseType::CASUAL)) {
            int64_t m = constInfo.s1Outer;
            int64_t n = constInfo.s2Outer;
            int64_t s1Size = constInfo.commonConstInfo.s1Size;
            int64_t s2Size = constInfo.commonConstInfo.s2Size;
            bool isRightDownCasualAndS1LessThanS2 = constInfo.sparseMode == RIGHT_DOWN_CAUSAL &&
                                                    constInfo.commonConstInfo.s1Size < constInfo.commonConstInfo.s2Size;
            if (isRightDownCasualAndS1LessThanS2) {
                // 矩形段右界(含): s2o <= floor((BASEM-1 + s2 - s1)/BASEN), 即 s1oMin=0 的最后列
                int64_t rectEnd = (CUBE_BASEM - 1 + s2Size - s1Size) / CUBE_BASEN; // floor (分母正)
                rectEnd = rectEnd < 0 ? -1 : rectEnd;                              // 无矩形段
                int64_t rectEndClamp = rectEnd < constInfo.s2Outer - 1 ? rectEnd : (constInfo.s2Outer - 1);
                int64_t rectBlocks = (rectEndClamp + 1) * m; // 矩形段块数 (rectEndClamp+1 列 * m)
                if (rectEndClamp >= 0 && gDimTail < rectBlocks) {
                    // 矩形段: 每列 m 块, s1o in [0, m-1]
                    s2Idx = gDimTail / m;
                    s1Idx = gDimTail % m;
                } else {
                    int64_t g = gDimTail - rectBlocks;
                    int64_t triStart = rectEndClamp + 1;
                    int64_t s0 = (triStart * CUBE_BASEN - s2Size + s1Size + CUBE_BASEM) / CUBE_BASEM - 1;
                    s0 = s0 > 0 ? s0 : 0;
                    int64_t pairNum = (constInfo.s2Outer - triStart) / NUM_TWO;
                    int64_t c = NUM_TWO * (m - s0);
                    int64_t q = static_cast<int64_t>(((c + 1) - sqrt(((c + 1) * (c + 1) - 4 * g))) / NUM_TWO);
                    q = ClampSigned(q, 0, pairNum);

                    for (int64_t i = 0; i < NUM_THREE; ++i) {
                        int64_t prefix = q * (c - q + 1);
                        if (q > 0 && prefix > g) {
                            q--;
                        } else if (q < pairNum && (q + 1) * (c - q) <= g) {
                            q++;
                        } else {
                            break;
                        }
                    }
                    int64_t local = g - q * (c - q + 1);
                    int64_t columnHeight = m - s0 - q;
                    if (local < columnHeight) {
                        s2Idx = triStart + NUM_TWO * q;
                        s1Idx = s0 + q + local;
                    } else {
                        s2Idx = triStart + NUM_TWO * q + 1;
                        s1Idx = s0 + q + (local - columnHeight);
                    }
                }
            } else {
                int64_t q = static_cast<int64_t>(
                    ((NUM_TWO * m + 1) - sqrt(((NUM_TWO * m + 1) * (NUM_TWO * m + 1) - 4 * gDimTail))) / NUM_TWO);
                int64_t pairNum = constInfo.s2Outer / NUM_TWO;
                q = ClampSigned(q, 0, pairNum);
                while (NUM_TWO * (q + 1) <= (constInfo.s2Outer - NUM_TWO) && (q + 1) * (NUM_TWO * m - q) <= gDimTail) {
                    ++q;
                }
                while (q > 0 && q * (NUM_TWO * m + 1 - q) > gDimTail) {
                    --q;
                }
                int64_t off = gDimTail - q * (NUM_TWO * m + 1 - q);
                int64_t columnHeight = m - q;
                int parity = off / columnHeight;
                s2Idx = NUM_TWO * q + parity;
                s1Idx = q + off % columnHeight;
            }
        } else if (sparseType == static_cast<uint8_t>(SparseType::BAND)) {
            // A partial final S2 column is decoded after the six regular-column segments.
            int64_t validS2Outer = CeilDivSigned(constInfo.commonConstInfo.s1Size + actualCalcS2Token, CUBE_BASEN);
            validS2Outer = ClampSigned(validS2Outer, 0, constInfo.s2Outer);
            bool hasS2Tail = constInfo.commonConstInfo.s2Size % CUBE_BASEN != 0;
            int64_t regularS2Outer = hasS2Tail ? constInfo.s2Outer - 1 : constInfo.s2Outer;
            int64_t regularValidS2Outer = validS2Outer < regularS2Outer ? validS2Outer : regularS2Outer;
            int64_t bandCompactIdx = gDimTail;

            bool decoded = false;
            // parity=0: even increasing/rectangle/decreasing;
            // parity=1: odd increasing/rectangle/decreasing.
            for (int64_t parity = 0; parity < NUM_TWO && !decoded; ++parity) {
                int64_t parityColNum =
                    parity == 0 ? (regularValidS2Outer + 1) / NUM_TWO : regularValidS2Outer / NUM_TWO;
                if (parityColNum <= 0) {
                    continue;
                }

                // For j=2k+parity: L=max(0,k+a), U=min(m,k+b), H=U-L.
                int64_t a = FloorDivSigned(parity * static_cast<int64_t>(CUBE_BASEN) - actualCalcS2Token,
                                           static_cast<int64_t>(CUBE_BASEM));
                int64_t b = CeilDivSigned((parity + 1) * static_cast<int64_t>(CUBE_BASEN) + actualCalcS1Token,
                                          static_cast<int64_t>(CUBE_BASEM));
                int64_t width = b - a;
                if (width <= 0) {
                    continue;
                }

                // shape=0/1/2: increasing / rectangle / decreasing.
                for (int64_t shape = 0; shape < NUM_THREE; ++shape) {
                    int64_t segmentBegin = 0;
                    int64_t segmentEnd = -1;
                    int64_t firstHeight = 0;
                    int64_t diff = 0;
                    if (shape == 0) {
                        segmentBegin = 1 - b;
                        segmentEnd = width <= constInfo.s1Outer ? -a - 1 : constInfo.s1Outer - b - 1;
                        firstHeight = (segmentBegin > 0 ? segmentBegin : 0) + b;
                        diff = 1;
                    } else if (shape == 1) {
                        segmentBegin = width <= constInfo.s1Outer ? -a : constInfo.s1Outer - b;
                        segmentEnd = width <= constInfo.s1Outer ? constInfo.s1Outer - b : -a;
                        firstHeight = width <= constInfo.s1Outer ? width : constInfo.s1Outer;
                    } else {
                        segmentBegin = width <= constInfo.s1Outer ? constInfo.s1Outer - b + 1 : 1 - a;
                        segmentEnd = constInfo.s1Outer - a - 1;
                        firstHeight = constInfo.s1Outer - (segmentBegin > 0 ? segmentBegin : 0) - a;
                        diff = -1;
                    }

                    segmentBegin = segmentBegin > 0 ? segmentBegin : 0;
                    segmentEnd = segmentEnd < parityColNum - 1 ? segmentEnd : parityColNum - 1;
                    int64_t segmentCount = segmentEnd >= segmentBegin ? segmentEnd - segmentBegin + 1 : 0;
                    int64_t segmentBlockNum =
                        segmentCount > 0 && firstHeight > 0 ? ArithmeticPrefix(segmentCount, firstHeight, diff) : 0;
                    if (bandCompactIdx >= segmentBlockNum) {
                        bandCompactIdx -= segmentBlockNum;
                        continue;
                    }

                    int64_t localK = diff == 0 ? bandCompactIdx / firstHeight :
                                                 SolveArithmeticOffset(bandCompactIdx, firstHeight, diff, segmentCount);
                    int64_t k = segmentBegin + localK;
                    int64_t offset = bandCompactIdx - ArithmeticPrefix(localK, firstHeight, diff);
                    int64_t beginRow = ClampSigned(k + a, 0, constInfo.s1Outer);
                    s1Idx = beginRow + offset;
                    s2Idx = NUM_TWO * k + parity;
                    decoded = true;
                    break;
                }
            }

            if (!decoded && hasS2Tail && constInfo.s2Outer - 1 < validS2Outer) {
                int64_t tailS2Idx = constInfo.s2Outer - 1;
                int64_t tailBegin = ClampSigned(
                    FloorDivSigned(tailS2Idx * static_cast<int64_t>(CUBE_BASEN) - actualCalcS2Token, CUBE_BASEM), 0,
                    constInfo.s1Outer);
                int64_t tailRightAlign64 =
                    CeilDivSigned(constInfo.commonConstInfo.s2Size, BITS_EACH_UINT64) * BITS_EACH_UINT64;
                int64_t tailEnd =
                    ClampSigned(CeilDivSigned(tailRightAlign64 + actualCalcS1Token, CUBE_BASEM), 0, constInfo.s1Outer);
                int64_t tailHeight = tailEnd > tailBegin ? tailEnd - tailBegin : 0;
                if (bandCompactIdx < tailHeight) {
                    s1Idx = tailBegin + bandCompactIdx;
                    s2Idx = tailS2Idx;
                    decoded = true;
                }
            }

            if (!decoded) {
                return -1;
            }
        }
    }
    s2CvBegin = s2Idx * CUBE_BASEN;
    s2CvEnd = s2CvBegin + CUBE_BASEN;     // 非尾块s2按照+CUBE_BASEN处理
    if (s2Idx == constInfo.s2Outer - 1) { // 默认s2 cv tail相等
        s2CvEnd = s2CvBegin + constInfo.s2Tail;
    }
    filledBySwizzle = true;
    runInfo.commonRunInfo.boIdx = bIdx / constInfo.commonConstInfo.n2G;
    int64_t bDimTail = bIdx % constInfo.commonConstInfo.n2G;
    runInfo.commonRunInfo.n2oIdx = bDimTail / constInfo.commonConstInfo.gSize;
    runInfo.commonRunInfo.goIdx = bDimTail % constInfo.commonConstInfo.gSize;
    runInfo.commonRunInfo.s1oIdx = s1Idx;
    runInfo.s2oIdx = s2Idx;
    return bIdx * constInfo.s1Outer * constInfo.s2Outer + s2Idx * constInfo.s1Outer + s1Idx;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryOffset(
    FagRunInfo& runInfo, bool isMm1Mm2)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t n2GD = constInfo.commonConstInfo.n2GD;
    int64_t gD = constInfo.commonConstInfo.gD;
    int64_t dSize = constInfo.commonConstInfo.dSize;
    int64_t n2GS1D = constInfo.commonConstInfo.n2GS1D;
    int64_t gS1D = constInfo.commonConstInfo.gS1D;
    int64_t s1D = constInfo.commonConstInfo.s1D;
    int64_t bN2GD = constInfo.commonConstInfo.bN2GD;
    int64_t bOffsetTmp = runInfo.lastBatchTotalS1BOffset;
    if constexpr (IS_ROPE) {
        if (isMm1Mm2) {
            n2GD = (constInfo.commonConstInfo.n2GD / ROPE_D_RATIO) << 1;
            gD = (constInfo.commonConstInfo.gD / ROPE_D_RATIO) << 1;
            dSize = (constInfo.commonConstInfo.dSize / ROPE_D_RATIO) << 1;
            n2GS1D = (constInfo.commonConstInfo.n2GS1D / ROPE_D_RATIO) << 1;
            gS1D = (constInfo.commonConstInfo.gS1D / ROPE_D_RATIO) << 1;
            s1D = (constInfo.commonConstInfo.s1D / ROPE_D_RATIO) << 1;
            bN2GD = (constInfo.commonConstInfo.bN2GD / ROPE_D_RATIO) << 1;
            bOffsetTmp = (bOffsetTmp / ROPE_D_RATIO) << 1;
        }
    }
    if constexpr (IS_TND) {
        bOffset = bOffsetTmp;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * n2GD;
        n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
        gOffset = runInfo.commonRunInfo.goIdx * dSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2GS1D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * gS1D;
            gOffset = runInfo.commonRunInfo.goIdx * s1D;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * bN2GD;
            bOffset = runInfo.commonRunInfo.boIdx * n2GD;
            n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
            gOffset = runInfo.commonRunInfo.goIdx * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2GS1D;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * n2GD;
            n2Offset = runInfo.commonRunInfo.n2oIdx * gD;
            gOffset = runInfo.commonRunInfo.goIdx * dSize;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryRopeOffset(FagRunInfo& runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS1BRopeOffset;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.ropeConstInfo.n2GDr;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.ropeConstInfo.gDr;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.ropeConstInfo.n2GS1Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.ropeConstInfo.gS1Dr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.ropeConstInfo.s1Dr;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.ropeConstInfo.bN2GDr;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.ropeConstInfo.n2GDr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.ropeConstInfo.gDr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.ropeConstInfo.n2GS1Dr;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.ropeConstInfo.n2GDr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.ropeConstInfo.gDr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetDxOffset(
    FagRunInfo& runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS1BOffsetForDv;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1Dv;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Dv;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GDv;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GDv;
            n2Offset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
            gOffset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyOffset(
    FagRunInfo& runInfo, bool isMm1Mm2)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t n2D = constInfo.commonConstInfo.n2D;
    int64_t n2S2D = constInfo.commonConstInfo.n2S2D;
    int64_t dSize = constInfo.commonConstInfo.dSize;
    int64_t bN2D = constInfo.commonConstInfo.bN2D;
    int64_t s2D = constInfo.commonConstInfo.s2D;
    int64_t s1D = constInfo.commonConstInfo.s1D;
    int64_t bN2GD = constInfo.commonConstInfo.bN2GD;
    int64_t bOffsetTmp = runInfo.lastBatchTotalS2BOffset;
    if constexpr (IS_ROPE) {
        if (isMm1Mm2) {
            n2D = (constInfo.commonConstInfo.n2D / ROPE_D_RATIO) << 1;
            n2S2D = (constInfo.commonConstInfo.n2S2D / ROPE_D_RATIO) << 1;
            dSize = (constInfo.commonConstInfo.dSize / ROPE_D_RATIO) << 1;
            bN2D = (constInfo.commonConstInfo.bN2D / ROPE_D_RATIO) << 1;
            s2D = (constInfo.commonConstInfo.s2D / ROPE_D_RATIO) << 1;
            s1D = (constInfo.commonConstInfo.s1D / ROPE_D_RATIO) << 1;
            bN2GD = (constInfo.commonConstInfo.bN2GD / ROPE_D_RATIO) << 1;
            bOffsetTmp = (bOffsetTmp / ROPE_D_RATIO) << 1;
        }
    }
    if constexpr (IS_TND) {
        bOffset = bOffsetTmp;
        s2Offset = runInfo.s2CvBegin * n2D;
        n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * s2D;
            s2Offset = runInfo.s2CvBegin * dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * bN2D;
            bOffset = runInfo.commonRunInfo.boIdx * n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
            s2Offset = runInfo.s2CvBegin * n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyRopeOffset(
    FagRunInfo& runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS2BRopeOffset;
        s2Offset = runInfo.s2CvBegin * constInfo.ropeConstInfo.n2Dr;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.ropeConstInfo.n2S2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.ropeConstInfo.s2Dr;
            s2Offset = runInfo.s2CvBegin * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.ropeConstInfo.bN2Dr;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.ropeConstInfo.n2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.ropeConstInfo.n2S2Dr;
            s2Offset = runInfo.s2CvBegin * constInfo.ropeConstInfo.n2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetValueOffset(
    FagRunInfo& runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS2BOffsetForDv;
        s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Dv;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.bN2Dv;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetDeqScaleQOffset(FagRunInfo& runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t scaleNumPerS1 = math::Ceil<int64_t>(constInfo.commonConstInfo.s1Size, QUANT_BLOCK_SIZE);
    bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2G * scaleNumPerS1;
    n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize * scaleNumPerS1;
    gOffset = runInfo.commonRunInfo.goIdx * scaleNumPerS1;
    s1Offset = runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType / QUANT_BLOCK_SIZE;
    scaleOffset = bOffset + n2Offset + gOffset + s1Offset;
    return scaleOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetDeqScaleKOffset(FagRunInfo& runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t scaleNumPerS2 = math::Ceil<int64_t>(constInfo.commonConstInfo.s2Size, QUANT_BLOCK_SIZE);
    bOffset = runInfo.commonRunInfo.boIdx * constInfo.n2Size * scaleNumPerS2;
    n2Offset = runInfo.commonRunInfo.n2oIdx * scaleNumPerS2;
    s2Offset = runInfo.s2oIdx * (uint32_t)s2TemplateType / QUANT_BLOCK_SIZE;
    scaleOffset = bOffset + n2Offset + s2Offset;
    return scaleOffset;
}

// tnd swizzle 优化
template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::LoadTndValidBatch(
    int64_t bIdx)
{
    tndValidCursor.bIdx = bIdx;
    GetSeqQlenKvlenByBidx(bIdx, tndValidCursor.actualS1Len, tndValidCursor.actualS2Len);
    tndValidCursor.s1Outer = (tndValidCursor.actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
    tndValidCursor.s2Outer = (tndValidCursor.actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CalcTndValidColumnRange(FagRunInfo& runInfo)
{
    UpdateToken(runInfo, tndValidCursor.bIdx);
    uint8_t sparseType = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.sparseType);
    if (sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
        tndValidCursor.beginS1 = 0;
        tndValidCursor.endS1 = tndValidCursor.s1Outer;
        return;
    }

    int64_t s2Left = tndValidCursor.s2Idx * CUBE_BASEN;
    int64_t s2Remain = tndValidCursor.actualS2Len - s2Left;
    int64_t s2Width = s2Remain < CUBE_BASEN ? s2Remain : CUBE_BASEN;

    int64_t left = s2Left - actualCalcS2Token;
    left = left > 0 ? left : 0;
    tndValidCursor.beginS1 = left > tndValidCursor.actualS1Len ? tndValidCursor.s1Outer : left / CUBE_BASEM;

    int64_t right = s2Left + s2Width + actualCalcS1Token;
    right = right > 0 ? right : 0;
    right = right < tndValidCursor.actualS1Len ? right : tndValidCursor.actualS1Len;
    tndValidCursor.endS1 = (right + CUBE_BASEM - 1) / CUBE_BASEM;

    tndValidCursor.beginS1 =
        tndValidCursor.beginS1 < 0 ?
            0 :
            (tndValidCursor.beginS1 > tndValidCursor.s1Outer ? tndValidCursor.s1Outer : tndValidCursor.beginS1);
    tndValidCursor.endS1 =
        tndValidCursor.endS1 < tndValidCursor.beginS1 ?
            tndValidCursor.beginS1 :
            (tndValidCursor.endS1 > tndValidCursor.s1Outer ? tndValidCursor.s1Outer : tndValidCursor.endS1);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType,
                                                         VecBlockType>::MoveToNextTndValidColumn(FagRunInfo& runInfo)
{
    while (true) {
        ++tndValidCursor.s2Idx;
        if (tndValidCursor.s2Idx >= tndValidCursor.s2Outer) {
            tndValidCursor.s2Idx = 0;
            ++tndValidCursor.n2Idx;
            if (tndValidCursor.n2Idx >= constInfo.n2Size) {
                tndValidCursor.n2Idx = 0;
                ++tndValidCursor.bIdx;
                if (tndValidCursor.bIdx >= constInfo.bSize) {
                    tndValidCursor.valid = false;
                    return false;
                }
                LoadTndValidBatch(tndValidCursor.bIdx);
            }
        }

        CalcTndValidColumnRange(runInfo);
        if (tndValidCursor.endS1 > tndValidCursor.beginS1) {
            tndValidCursor.valid = true;
            return true;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType,
                                                         VecBlockType>::LocateTndValidColumnTask(FagRunInfo& runInfo,
                                                                                                 int64_t taskRank)
{
    if (taskRank < 0 || static_cast<uint64_t>(taskRank) >= constInfo.totalValidS2TaskNum || constInfo.n2Size == 0) {
        return false;
    }

    int64_t left = 0;
    int64_t right = constInfo.bSize;
    while (left < right) {
        int64_t middle = (left + right) >> 1;
        if (tilingData->tndSwizzleParam.tndValidBlockPrefixSum[middle + 1] <= static_cast<uint64_t>(taskRank)) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }

    int64_t bIdx = left;
    uint64_t batchBegin = tilingData->tndSwizzleParam.tndValidBlockPrefixSum[bIdx];
    uint64_t batchEnd = tilingData->tndSwizzleParam.tndValidBlockPrefixSum[bIdx + 1];
    uint64_t validS2PerN2 = (batchEnd - batchBegin) / constInfo.n2Size;
    if (validS2PerN2 == 0) {
        return false;
    }

    uint64_t localRank = static_cast<uint64_t>(taskRank) - batchBegin;
    tndValidCursor.n2Idx = static_cast<int64_t>(localRank / validS2PerN2);
    uint64_t validS2Ordinal = localRank % validS2PerN2;
    LoadTndValidBatch(bIdx);
    tndValidCursor.s2Idx = -1;
    tndValidCursor.valid = true;
    for (uint64_t ordinal = 0; ordinal <= validS2Ordinal; ++ordinal) {
        if (!MoveToNextTndValidColumn(runInfo)) {
            return false;
        }
    }

    tndValidCursor.globalTaskRank = taskRank;
    tndValidCursor.innerTileIdx = 0;
    return true;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType,
                                                         VecBlockType>::AdvanceTndValidColumnTask(FagRunInfo& runInfo,
                                                                                                  int64_t taskDelta)
{
    if (taskDelta < 0) {
        return false;
    }
    int64_t targetRank = tndValidCursor.globalTaskRank + taskDelta;
    if (static_cast<uint64_t>(targetRank) >= constInfo.totalValidS2TaskNum) {
        tndValidCursor.valid = false;
        return false;
    }

    for (int64_t step = 0; step < taskDelta; ++step) {
        if (!MoveToNextTndValidColumn(runInfo)) {
            return false;
        }
    }
    tndValidCursor.globalTaskRank = targetRank;
    return true;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::EmitCurrentTndValidColumnTile(
    FagRunInfo& runInfo, int64_t innerTileIdx)
{
    int64_t validS1 = tndValidCursor.endS1 - tndValidCursor.beginS1;
    int64_t tileNum = validS1 * constInfo.commonConstInfo.gSize;
    if (innerTileIdx < 0 || innerTileIdx >= tileNum) {
        return false;
    }

    int64_t localS1 = innerTileIdx / constInfo.commonConstInfo.gSize;
    int64_t gIdx = innerTileIdx % constInfo.commonConstInfo.gSize;
    int64_t s1Idx = tndValidCursor.beginS1 + localS1;
    int64_t s2End = (tndValidCursor.s2Idx + 1) * CUBE_BASEN;
    s2End = s2End < tndValidCursor.actualS2Len ? s2End : tndValidCursor.actualS2Len;

    s2CvBegin = tndValidCursor.s2Idx * CUBE_BASEN;
    s2CvEnd = s2End;

    runInfo.commonRunInfo.boIdx = tndValidCursor.bIdx;
    runInfo.commonRunInfo.n2oIdx = tndValidCursor.n2Idx;
    runInfo.commonRunInfo.goIdx = gIdx;
    runInfo.commonRunInfo.s1oIdx = s1Idx;
    runInfo.s2oIdx = tndValidCursor.s2Idx;
    runInfo.commonRunInfo.actualS1Size = tndValidCursor.actualS1Len;
    runInfo.commonRunInfo.actualS2Size = tndValidCursor.actualS2Len;
    return true;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetTndValidColumnTaskRank(int64_t round)
{
    if (round < 0) {
        return -1;
    }
    if (constInfo.columnScheduleMode != TND_VALID_S2_COLUMN_SIMPLE_CYCLIC &&
        constInfo.columnScheduleMode != TND_VALID_S2_COLUMN_SERPENTINE) {
        return -1;
    }

    const uint64_t activeCoreNum = constInfo.validS2ActiveCoreNum;
    const uint64_t base = static_cast<uint64_t>(round) * activeCoreNum;
    if (base >= constInfo.totalValidS2TaskNum) {
        return -1;
    }
    const uint64_t remain = constInfo.totalValidS2TaskNum - base;
    const uint64_t waveWidth = remain < activeCoreNum ? remain : activeCoreNum;
    const uint64_t coreIdx = static_cast<uint64_t>(cBlockIdx);
    if (coreIdx >= waveWidth) {
        return -1;
    }

    if (constInfo.columnScheduleMode == TND_VALID_S2_COLUMN_SIMPLE_CYCLIC || (round & 1) == 0) {
        return static_cast<int64_t>(base + coreIdx);
    }
    if (constInfo.columnScheduleMode == TND_VALID_S2_COLUMN_SERPENTINE) {
        // The final partial wave must use its real width W, not activeCoreNum.
        return static_cast<int64_t>(base + waveWidth - 1 - coreIdx);
    }
    return -1;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForTndBn2gs1s2Swizzle(
    FagRunInfo& runInfo, int64_t loopIdx)
{
    if (!tndValidCursorInitialized) {
        const int64_t firstTaskRank = GetTndValidColumnTaskRank(0);
        if (firstTaskRank < 0 || !LocateTndValidColumnTask(runInfo, firstTaskRank)) {
            return -1;
        }
        tndValidCursorInitialized = true;
    }

    while (tndValidCursor.valid) {
        int64_t tileNum = (tndValidCursor.endS1 - tndValidCursor.beginS1) * constInfo.commonConstInfo.gSize;
        if (tndValidCursor.innerTileIdx < tileNum) {
            bool emitted = EmitCurrentTndValidColumnTile(runInfo, tndValidCursor.innerTileIdx);
            ++tndValidCursor.innerTileIdx;
            return emitted ? loopIdx : -1;
        }

        const int64_t currentRound = tndValidCursor.globalTaskRank / constInfo.validS2ActiveCoreNum;
        const int64_t nextTaskRank = GetTndValidColumnTaskRank(currentRound + 1);
        if (nextTaskRank < 0) {
            tndValidCursor.valid = false;
            return -1;
        }
        const int64_t taskDelta = nextTaskRank - tndValidCursor.globalTaskRank;
        if (taskDelta <= 0 || !AdvanceTndValidColumnTask(runInfo, taskDelta)) {
            return -1;
        }
        tndValidCursor.innerTileIdx = 0;
    }
    return -1;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SyncALLCores()
{
    SyncAll<>();
}

} // namespace FagBaseApi
#endif
