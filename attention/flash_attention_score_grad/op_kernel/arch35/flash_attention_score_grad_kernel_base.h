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

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_BASE_H

#include "flash_attention_score_grad_common.h"
#include "cube_api/mutex_buffers_policy.h"
namespace FagBaseApi {

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreGradKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift,
                                GM_ADDR dropMask, GM_ADDR attenMask, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen, GM_ADDR deqScaleQ,
                                GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope,
                                GM_ADDR keyRope, GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse,
                                GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink, GM_ADDR workspace,
                                FagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void InitCVCommonBuffer();
    __aicore__ inline void InitCVCommonGlobalBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR deqScaleQ,
                                                    GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy,
                                                    GM_ADDR workspace);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void SetOptionalInfo();
    __aicore__ inline void SetSwizzleConstInfo();
    __aicore__ inline void SetAxisRunInfo(FagRunInfo &runInfo, int64_t s2CvBegin, int64_t s2CvEnd, int64_t boIdx,
                                          int64_t n2oIdx, int64_t goIdx, int64_t s1oIdx, int64_t s2oIdx);
    __aicore__ inline void SetRunInfo(FagRunInfo &runInfo, FagRunInfo &nextRunInfo, int64_t taskId, int64_t index,
                                      int64_t nextIndex = -1);
    __aicore__ inline void Process();
    __aicore__ inline bool IsValid(FagRunInfo &runInfo, int64_t taskId, int64_t index);
    __aicore__ inline bool IsValidForTND(FagRunInfo &runInfo, int64_t taskId, int64_t index);
    __aicore__ inline bool IsValidForDeter(FagRunInfo &runInfo, int64_t taskId, int64_t index);
    __aicore__ inline void UpdateToken(FagRunInfo &runInfo, int64_t bIdx);
    __aicore__ inline bool CheckIsValidBlock(FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oDimIdx,
                                             int64_t taskId);
    __aicore__ inline bool CheckIsValidBlockForDeter(FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx,
                                                     int64_t s2oDimIdx, int64_t taskId);
    __aicore__ inline int64_t GetNextValidIdx(FagRunInfo &runInfo, int64_t taskId, int64_t startIndex,
                                              int64_t loopIdx = 0);
    __aicore__ inline int64_t GetNextValidIdxForTndSwizzleDense(FagRunInfo &runInfo, int64_t loopIdx);
    __aicore__ inline int64_t GetNextValidIdxForTndSwizzleCasual(FagRunInfo &runInfo, int64_t loopIdx);
    __aicore__ inline int64_t GetNextValidIdxForTndSwizzleBand(FagRunInfo &runInfo, int64_t loopIdx);
    __aicore__ inline int64_t GetNextValidIdxForSwizzle(FagRunInfo &runInfo, int64_t loopIdx);
    __aicore__ inline int64_t GetNextValidIdxForSwizzleDense(FagRunInfo &runInfo, int64_t loopIdx);
    __aicore__ inline int64_t GetNextValidIdxForSwizzleCasual(FagRunInfo &runInfo, int64_t loopIdx);
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetQueryOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetQueryRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDxOffset(FagRunInfo &runInfo);
    template <bool IS_MM1_MM2 = true>
    __aicore__ inline int64_t GetKeyOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetKeyRopeOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetValueOffset(FagRunInfo &runInfo);
    __aicore__ inline void GetNextDxAndQueryOffset(FagRunInfo &runInfo, FagRunInfo &nextRunInfo, int64_t nextIndex,
                                                   PreloadArgs<IS_ROPE> &preloadArgs);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);
    __aicore__ inline void CheckS1RangeInBn2(int64_t taskId);
    template <const bool IS_DENSE = false>
    __aicore__ inline bool GetLocalS1S2Idx(int64_t loppIdx, int64_t m, int64_t n, int64_t &localS1Idx,
                                           int64_t &localS2Idx);
    __aicore__ inline void UpdateMNPQ(int64_t &m, int64_t &n, int64_t &p, int64_t &q, int64_t &mGap, int64_t &nGap);
    __aicore__ inline ChildClass *GetDerived()
    {
        return static_cast<ChildClass *>(this);
    }
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint8_t ALIGN_NUM_32 = 32;
    constexpr static uint8_t VEC_CORE_NUM_64 = 64;
    constexpr static uint32_t BITS_EACH_UINT64 = 64;
    constexpr static uint32_t MAX_BITS_IN_TILING = 32 * BITS_EACH_UINT64;
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INPUT_BLOCK_NUM = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_FP8 = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t BASE_DQ_SIZE = CUBE_BASEM * HEAD_DIM_ALIGN;
    constexpr static uint32_t BASE_DKV_SIZE = CUBE_BASEN * HEAD_DIM_ALIGN;
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE = 32 / sizeof(INPUT_TYPE);

    constexpr static bool IS_DQ_RES_EXCEED_UB = HEAD_DIM_ALIGN > VECTOR_BASEN;
    constexpr static bool IS_DKV_RES_EXCEED_UB =
        VECTOR_BASEN / CV_CORE_RATIO * HEAD_DIM_ALIGN > VECTOR_BASEM *VECTOR_BASEN;
    constexpr static bool IS_DQ_WRITE_UB = (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK && !IS_DQ_RES_EXCEED_UB);
    constexpr static bool IS_DK_WRITE_UB =
        (((SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) || (SPLIT_AXIS == BN2S2 && DETER_SPARSE_TYPE == NO_DETER)) &&
         !IS_DKV_RES_EXCEED_UB);
    constexpr static bool IS_DV_WRITE_UB =
        ((SPLIT_AXIS == BN2S2 && DETER_SPARSE_TYPE == NO_DETER) && !IS_DKV_RES_EXCEED_UB);
    constexpr static uint32_t M_SWIZZLE_BLOCK_SIZE = M_SWIZZLE_SIZE / CUBE_BASEM;
    constexpr static uint32_t N_SWIZZLE_BLOCK_SIZE = N_SWIZZLE_SIZE / CUBE_BASEN;
    constexpr static uint8_t DETER_TILING_SPLIT_MODE =
        (CUBE_BASEM == CUBE_BASEN ? 0 : (CUBE_BASEM > CUBE_BASEN ? 2 : 1));
    constexpr static uint32_t DETER_CUBE_BASEM = CUBE_BASEM < CUBE_BASEN ? CUBE_BASEN : CUBE_BASEM;
    constexpr static uint32_t DETER_CUBE_BASEN = CUBE_BASEM > CUBE_BASEN ? CUBE_BASEM : CUBE_BASEN;
    constexpr static uint32_t IS_NORMAL_SWIZZLE =
        !IS_TND && CUBE_BASEM == CUBE_BASEN && SPLIT_AXIS == BN2GS1S2 && DETER_SPARSE_TYPE == NO_DETER;
    constexpr static bool IS_PRELOAD_TWO_TIMES = HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned128) &&
                                                 !IS_DROP && SPLIT_AXIS == BN2GS1S2 && DETER_SPARSE_TYPE == NO_DETER &&
                                                 !IS_FP32_INPUT;

protected:
    TPipe *pipe;

    // output global mmemory
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    // CV核间共享Buffer
    TBuf<> mm1ResBuf[2];
    TBuf<> mm2ResBuf[2];
    MutexBufferManager<BufferType::L1> l1BufferManager;
    typename std::conditional<IS_PRELOAD_TWO_TIMES, MutexBuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC>,
                              MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC>>::type pL1Buf;
    typename std::conditional<IS_PRELOAD_TWO_TIMES, MutexBuffersPolicyDB<BufferType::L1, SyncType::NO_SYNC>,
                              MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC>>::type dSL1Buf;

    GM_ADDR prefixNAddr;
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;

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

    // BN2S2模板判断是否有无效S2列
    int64_t curS2oIdx = -1;
    int64_t curS2InvalidTotalNum = 0;

    // BN2扩展模板判断S1轴有效始终位置
    typename std::conditional<IS_BN2_MULTIBLK, bool[2], std::nullptr_t>::type isLastS1Outer;
    typename std::conditional<IS_BN2_MULTIBLK, bool[2], std::nullptr_t>::type isFirstS1Outer;
    typename std::conditional<IS_BN2_MULTIBLK, Bn2MultiBlkInfo, std::nullptr_t>::type multiBlkInfo;

    FagTilingType tilingData;
    FagConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    DropMaskInfo dropInfo;
    PreloadArgs<IS_ROPE> preloadArgs;

    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    // for record tnd offset or index info
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchIdx;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalBaseIdx;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS1BOffset;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS1BRopeOffset;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS1BOffsetForDv;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS2BOffset;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS2BRopeOffset;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS2BOffsetForDv;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS1S2SizeAlign;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS1S2Size;
    typename std::conditional<IS_TND, int64_t, std::nullptr_t>::type curBatchTotalS2Size;

    typename std::conditional<IS_TND_SWIZZLE, int64_t, std::nullptr_t>::type deltaCnt{};
    typename std::conditional<IS_TND_SWIZZLE, int64_t, std::nullptr_t>::type bandLoopIdx{};
    int64_t swizzleLoopIdx = 0;
    bool enableDenseSwizzle = false;
    bool enableCasualSwizzle = false;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR sink, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR dsink,
    GM_ADDR workspace, FagTilingType ordTilingData, TPipe *pipeIn)
{
    // init current core tilingInfo
    if ASCEND_IS_AIV {
        vBlockIdx = GetBlockIdx();
        cBlockIdx = vBlockIdx / CV_CORE_RATIO;
        vSubBlockIdx = GetSubBlockIdx();
    } else {
        cBlockIdx = GetBlockIdx();
    }
    tilingData = ordTilingData;
    pipe = pipeIn;

    // fill constInfo
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
    SetOptionalInfo();

    // pass params to vector block
    vecBlock.SetVecBlockParams(pipeIn, tilingData, vBlockIdx, cBlockIdx, vSubBlockIdx, attenMaskInfo, pseInfo,
                               dropInfo);
    vecBlock.InitUbBuffer();
    vecBlock.InitGlobalBuffer(value, dy, y, pseShift, dropMask, attenMask, softmaxMax, softmaxSum, deqScaleQ, deqScaleK,
                              deqScaleV, deqScaleDy, dq, dk, dv, dqRope, dkRope, sink, dsink, workspace);
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE) && IS_FP32_INPUT) {
        vecBlock.SetOldDeterFp32Param(constInfo);
    }

    // pass params to cube block
    cubeBlock.SetCubeBlockParams(pipeIn, tilingData, &l1BufferManager);
    cubeBlock.InitCubeBuffer(constInfo);
    cubeBlock.InitGlobalBuffer(query, key, value, dy, queryRope, keyRope, dq, dk, dv, workspace);
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
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    // init workspace address
    if constexpr (!IS_FP32_INPUT) {
        if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
            uint64_t qPostBlockTotal = CUBE_BASEM * HEAD_DIM_ALIGN * MAX_CUBE_CORE_NUM;
            uint64_t kPostBlockTotal = CUBE_BASEN * HEAD_DIM_ALIGN * MAX_CUBE_CORE_NUM;
            uint64_t workspaceOffsets = RESERVED_WORKSPACE_SIZE;
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(CALC_TYPE));
            workspaceOffsets = workspaceOffsets + qPostBlockTotal * sizeof(float);
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(CALC_TYPE));
            workspaceOffsets = workspaceOffsets + kPostBlockTotal * sizeof(float);
        } else {
            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        }
    } else {
        // input type fp32, dq dk dv write to output gm directly
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dq);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dk);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dv);
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitCVCommonBuffer()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    if constexpr ((DETER_SPARSE_TYPE) == DETER_OLD) {
        dSL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE) * NUM_TWO);
    } else {
        dSL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(INPUT_TYPE));
    }
    pL1Buf.Init(l1BufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(OUTDTYPE));
    pipe->InitBuffer(mm1ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm1ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm2ResBuf[0], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
    pipe->InitBuffer(mm2ResBuf[1], VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE));
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetOptionalInfo()
{
    if constexpr (!IS_DETER_NEW(DETER_SPARSE_TYPE) && IS_TND && !IS_TND_SWIZZLE) {
        curBatchIdx = tilingData->tndParam.tndStartBIdx[cBlockIdx];
        int64_t tndS1PrefixSum = (curBatchIdx == 0 ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[curBatchIdx - 1]);
        int64_t tndS2PrefixSum = (curBatchIdx == 0 ? 0 : ((__gm__ int64_t *)actualSeqKvlenAddr)[curBatchIdx - 1]);
        curBatchTotalBaseIdx = tilingData->tndParam.tndPrefixSum[cBlockIdx] * constInfo.commonConstInfo.n2G;
        curBatchTotalS1BOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2GD;
        curBatchTotalS2BOffset = tndS2PrefixSum * constInfo.commonConstInfo.n2D;
        curBatchTotalS1BOffsetForDv = tndS1PrefixSum * constInfo.commonConstInfo.n2GDv;
        curBatchTotalS2BOffsetForDv = tndS2PrefixSum * constInfo.commonConstInfo.n2Dv;
        curBatchTotalS1S2SizeAlign = tilingData->tndParam.tndS1S2AlignPrefixSum[cBlockIdx];
        curBatchTotalS1S2Size = tilingData->tndParam.tndS1S2PrefixSum[cBlockIdx];
        curBatchTotalS2Size = tndS2PrefixSum;
        if constexpr (IS_ROPE) {
            curBatchTotalS1BRopeOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2GDr;
            curBatchTotalS2BRopeOffset = tndS2PrefixSum * constInfo.commonConstInfo.n2Dr;
        }
    }

    if constexpr (IS_ATTEN_MASK) {
        attenMaskInfo.preTokens = tilingData->s1s2BNGS1S2BaseParams.s1Token;
        attenMaskInfo.nextTokens = tilingData->s1s2BNGS1S2BaseParams.s2Token;
        attenMaskInfo.bandIndex = tilingData->s1s2BNGS1S2SplitCoreParams.bandIdx;
    }

    if ASCEND_IS_AIV {
        if constexpr (IS_ATTEN_MASK) {
            attenMaskInfo.attenMaskShapeType = tilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType;
            attenMaskInfo.compressMode = tilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
            attenMaskInfo.attenMaskS2Size = tilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;
            attenMaskInfo.prefixNAddr = prefixNAddr;
        }
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
            dropInfo.seed = tilingData->s1s2BNGS1S2BaseParams.seed;
            dropInfo.offset = tilingData->s1s2BNGS1S2BaseParams.offset;
            dropInfo.keepProbUint8 = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.keepProbUint8);
            dropInfo.dropMaskOuter = tilingData->s1s2BNGS1S2BaseParams.dropMaskOuter;
            dropInfo.boolMode = tilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetSwizzleConstInfo()
{
    if constexpr (IS_NORMAL_SWIZZLE) {
        if (!tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx) {
            return;
        }
        // swizzle
        enableDenseSwizzle = (!IS_ATTEN_MASK || tilingData->s1s2BNGS1S2BaseParams.sparseType ==
                                                    static_cast<uint8_t>(SparseType::DENSE)) &&
                             constInfo.commonConstInfo.s1Size > M_SWIZZLE_SIZE &&
                             constInfo.commonConstInfo.s2Size > N_SWIZZLE_SIZE;
        enableCasualSwizzle =
            IS_ATTEN_MASK && tilingData->s1s2BNGS1S2BaseParams.sparseType == static_cast<uint8_t>(SparseType::CASUAL) &&
            constInfo.commonConstInfo.s1Size == constInfo.commonConstInfo.s2Size &&
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
        constInfo.mSwizzleBlockNum = constInfo.s1Outer / M_SWIZZLE_BLOCK_SIZE;     // m1
        constInfo.mSwizzleBlockNumTail = constInfo.s1Outer % M_SWIZZLE_BLOCK_SIZE; // m2
        uint64_t allCoreOnceBlockNum = constInfo.aicCoreNum * constInfo.continuousBlockNum;
        if (enableDenseSwizzle) {
            constInfo.nSwizzleBlockNum = constInfo.s2Outer / N_SWIZZLE_BLOCK_SIZE;     // n1
            constInfo.nSwizzleBlockNumTail = constInfo.s2Outer % N_SWIZZLE_BLOCK_SIZE; // n2
            // swizzle block指经过第一次大块切分后的单个块
            // 计算单个核左上单个swizzle block内整块基本块总轮数
            uint64_t c1Tmp = (M_SWIZZLE_BLOCK_SIZE * N_SWIZZLE_BLOCK_SIZE) % allCoreOnceBlockNum;
            uint64_t c1 = c1Tmp < constInfo.continuousBlockNum ? c1Tmp : constInfo.continuousBlockNum;
            constInfo.leftUpTotalRound =
                ((M_SWIZZLE_BLOCK_SIZE * N_SWIZZLE_BLOCK_SIZE) / allCoreOnceBlockNum) * constInfo.continuousBlockNum +
                c1;

            // 计算单个核左下（s1方向可能存在尾块）单个swizzle block内基本块总轮数
            uint64_t c2Tmp = (constInfo.mSwizzleBlockNumTail * N_SWIZZLE_BLOCK_SIZE) % allCoreOnceBlockNum;
            uint64_t c2 = c2Tmp < constInfo.continuousBlockNum ? c2Tmp : constInfo.continuousBlockNum;
            constInfo.leftDownTotalRound =
                ((constInfo.mSwizzleBlockNumTail * N_SWIZZLE_BLOCK_SIZE) / allCoreOnceBlockNum) *
                    constInfo.continuousBlockNum +
                c2;

            // 计算单个核右上（s2方向可能存在尾块）单个swizzle block内（可能存在尾块）基本块总轮数
            uint64_t c3Tmp = (M_SWIZZLE_BLOCK_SIZE * constInfo.nSwizzleBlockNumTail) % allCoreOnceBlockNum;
            uint64_t c3 = c3Tmp < constInfo.continuousBlockNum ? c3Tmp : constInfo.continuousBlockNum;
            constInfo.rightUpTotalRound =
                ((M_SWIZZLE_BLOCK_SIZE * constInfo.nSwizzleBlockNumTail) / allCoreOnceBlockNum) *
                    constInfo.continuousBlockNum +
                c3;

            // 计算单个核右下（s1s2方向都可能存在尾块）单个swizzle block内（可能存在尾块）基本块总轮数
            uint64_t c4Tmp = (constInfo.mSwizzleBlockNumTail * constInfo.nSwizzleBlockNumTail) % allCoreOnceBlockNum;
            uint64_t c4 = c4Tmp < constInfo.continuousBlockNum ? c4Tmp : constInfo.continuousBlockNum;
            constInfo.rightDownTotalRound =
                ((constInfo.mSwizzleBlockNumTail * constInfo.nSwizzleBlockNumTail) / allCoreOnceBlockNum) *
                    constInfo.continuousBlockNum +
                c4;

            // 左半部分单个核列swizzle block内总基本块个数
            constInfo.leftSingleColTotalRound =
                constInfo.leftUpTotalRound * constInfo.mSwizzleBlockNum + constInfo.leftDownTotalRound;
            // 左半部分单个核所有列swizzle block内总基本块个数
            constInfo.leftTotalRound = constInfo.leftSingleColTotalRound * constInfo.nSwizzleBlockNum;
            // s1s2内总轮数
            constInfo.batchTotalRound = constInfo.leftTotalRound +
                                        constInfo.rightUpTotalRound * constInfo.mSwizzleBlockNum +
                                        constInfo.rightDownTotalRound;
        } else if (enableCasualSwizzle) {
            uint64_t triangleBlockNum = M_SWIZZLE_BLOCK_SIZE * (M_SWIZZLE_BLOCK_SIZE + 1) >> 1;
            uint64_t c1Tmp = triangleBlockNum % allCoreOnceBlockNum;
            uint64_t c1 = c1Tmp < constInfo.continuousBlockNum ? c1Tmp : constInfo.continuousBlockNum;
            constInfo.leftUpTotalRound = triangleBlockNum / allCoreOnceBlockNum * constInfo.continuousBlockNum + c1;

            uint64_t c2Tmp = (M_SWIZZLE_BLOCK_SIZE * M_SWIZZLE_BLOCK_SIZE) % allCoreOnceBlockNum;
            uint64_t c2 = c2Tmp < constInfo.continuousBlockNum ? c2Tmp : constInfo.continuousBlockNum;
            constInfo.leftDownTotalRound =
                ((M_SWIZZLE_BLOCK_SIZE * M_SWIZZLE_BLOCK_SIZE) / allCoreOnceBlockNum) * constInfo.continuousBlockNum +
                c2;

            uint64_t c3Tmp = (constInfo.mSwizzleBlockNumTail * M_SWIZZLE_BLOCK_SIZE) % allCoreOnceBlockNum;
            uint64_t c3 = c3Tmp < constInfo.continuousBlockNum ? c3Tmp : constInfo.continuousBlockNum;
            constInfo.rightUpTotalRound =
                ((constInfo.mSwizzleBlockNumTail * M_SWIZZLE_BLOCK_SIZE) / allCoreOnceBlockNum) *
                    constInfo.continuousBlockNum +
                c3;

            triangleBlockNum = (constInfo.mSwizzleBlockNumTail + 1) * constInfo.mSwizzleBlockNumTail >> 1;
            uint64_t c4Tmp = triangleBlockNum % allCoreOnceBlockNum;
            uint64_t c4 = c4Tmp < constInfo.continuousBlockNum ? c4Tmp : constInfo.continuousBlockNum;
            constInfo.rightDownTotalRound =
                (triangleBlockNum / allCoreOnceBlockNum) * constInfo.continuousBlockNum + c4;

            constInfo.leftTotalRound =
                (constInfo.leftUpTotalRound + constInfo.rightUpTotalRound) * constInfo.mSwizzleBlockNum +
                ((constInfo.mSwizzleBlockNum - 1) * constInfo.mSwizzleBlockNum >> 1) * constInfo.leftDownTotalRound;
            constInfo.batchTotalRound = constInfo.leftTotalRound + constInfo.rightDownTotalRound;
        } else if (tilingData->s1s2BNGS1S2BaseParams.sparseType == static_cast<uint8_t>(SparseType::BAND)) {
            int64_t m = 0;
            int64_t n = 0;
            int64_t p = 0;    // 第一列有效基本块的数量
            int64_t q = 0;    // 第一行有效基本块的数量
            int64_t mGap = 0; // 无效行数
            int64_t nGap = 0; // 无效列数
            UpdateMNPQ(m, n, p, q, mGap, nGap);
            constInfo.actualS1Outer = m;
            constInfo.actualS2Outer = n;
            constInfo.firstValidColBlockNum = p;
            constInfo.firstValidRowBlockNum = q;
            constInfo.mGap = mGap;
            constInfo.nGap = nGap;
            if (p + q > m) {
                constInfo.s2FirstWidth = m - p;
                constInfo.s2SecondWidth = p + q - m - 1;
                constInfo.s2FirstBlockNum = (p + m - 1) * constInfo.s2FirstWidth >> 1; // 梯形
                constInfo.s2SecondBlockNum = m * constInfo.s2SecondWidth;              // 长方形
            } else {
                constInfo.s2FirstWidth = q - 1;
                constInfo.s2SecondWidth = Min(n - q + 1, m + NUM_TWO - p - q);
                constInfo.s2FirstBlockNum = (((p - 1) << 1) + q) * (q - 1) >> 1;
                constInfo.s2SecondBlockNum = (p + q - 1) * constInfo.s2SecondWidth;
            }
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetConstInfo()
{
    // split info
    constInfo.s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    constInfo.s1CvTail = tilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    constInfo.s2Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s2Tail;
    constInfo.s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;

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
    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) {
        constInfo.isSink = tilingData->s1s2BNGS1S2BaseParams.sinkOptional;
        constInfo.s1SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s1SinkOuter;
        constInfo.s2SinkOuter = tilingData->s1s2BNGS1S2BaseParams.s2SinkOuter;
    }
    constInfo.n2GS1oS2o = constInfo.commonConstInfo.n2G * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.gS1oS2o = constInfo.commonConstInfo.gSize * constInfo.s1Outer * constInfo.s2Outer;
    constInfo.s1oS2o = constInfo.s1Outer * constInfo.s2Outer;
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        constInfo.deterConstInfo.noNeedDeter = static_cast<bool>(tilingData->baseDeterParam.noNeedDeter);
        constInfo.deterConstInfo.usedCubeCoreNum =
            static_cast<uint8_t>(tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter);
        // 确定性计算中会用满V核
        if (static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum) == VEC_CORE_NUM_64) {
            // 64V核场景确定性计算中会用满V核
            constInfo.deterConstInfo.usedVectorCoreNum =
                static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.coreNum);
        } else {
            // 非64V核场景确定性计算使用V核数量为往下32取整 兼容C核28核
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
        constInfo.deterConstInfo.deterDqkSrcStride =
            static_cast<uint32_t>((HEAD_DIM_ALIGN - constInfo.commonConstInfo.dSize) / FLOAT_BLOCK_SIZE);
        constInfo.deterConstInfo.deterDvSrcStride =
            static_cast<uint32_t>((HEAD_DIM_ALIGN - constInfo.commonConstInfo.dSizeV) / FLOAT_BLOCK_SIZE);
        constInfo.deterConstInfo.deterDqDstStride =
            static_cast<uint32_t>((constInfo.mm2Ka - constInfo.commonConstInfo.dSize) * sizeof(CALC_TYPE));
        constInfo.deterConstInfo.deterDkDstStride =
            static_cast<uint32_t>((constInfo.mm2Kb - constInfo.commonConstInfo.dSize) * sizeof(CALC_TYPE));
        constInfo.deterConstInfo.deterDvDstStride = static_cast<uint32_t>(
            (constInfo.commonConstInfo.mm1Kb - constInfo.commonConstInfo.dSizeV) * sizeof(CALC_TYPE));
        constInfo.deterConstInfo.deterVecCoreS1Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS1Offset * constInfo.mm2Ka;
        constInfo.deterConstInfo.deterDkVecCoreS2Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS2Offset * constInfo.mm2Kb;
        constInfo.deterConstInfo.deterDvVecCoreS2Offset =
            vBlockIdx * constInfo.deterConstInfo.eachVecCoreS2Offset * constInfo.commonConstInfo.mm1Kb;
        constInfo.deterConstInfo.eventIDScalarToMte2 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        constInfo.deterConstInfo.eventIDMte2ToScalar =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        constInfo.deterConstInfo.eventIDScalarToMte3 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        constInfo.deterConstInfo.eventIDMte3ToScalar =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        constInfo.deterConstInfo.eventIDMte3ToMte2 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        constInfo.deterConstInfo.eventIDMte2ToMte3 =
            static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        constInfo.mm3Ka = constInfo.mm2Ka;
        constInfo.mm4Kb = constInfo.mm2Kb;
        if constexpr (IS_ROPE) {
            constInfo.mm2Ka = constInfo.mm2Ka / 3 << 1;
            constInfo.mm2Kb = constInfo.mm2Kb / 3 << 1;
        }
    } else {
        if ASCEND_IS_AIC {
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
            constInfo.mm3Ka = constInfo.mm2Ka;
            constInfo.mm4Kb = constInfo.mm2Kb;
            if constexpr (IS_ROPE) {
                constInfo.mm2Ka = constInfo.mm2Ka / 3 << 1;
                constInfo.mm2Kb = constInfo.mm2Kb / 3 << 1;
            }
        }
    }
    if constexpr (IS_ATTEN_MASK) {
        constInfo.s1Token = tilingData->s1s2BNGS1S2BaseParams.s1Token;
        constInfo.s2Token = tilingData->s1s2BNGS1S2BaseParams.s2Token;
        constInfo.sparseMode = tilingData->s1s2BNGS1S2BaseParams.sparseMode;
    }
    constInfo.commonConstInfo.subBlockIdx = vSubBlockIdx;
    constInfo.scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;
    constInfo.aicCoreNum = tilingData->s1s2BNGS1S2BaseParams.coreNum >> 1;
    constInfo.enablePreSfmg = ((uint32_t)dTemplateType > 64) && tilingData->s1s2BNGS1S2BaseParams.enablePreSfmg;
    constInfo.dAlign16 = AlignTo16(constInfo.commonConstInfo.dSize);
    constInfo.dvAlign16 = AlignTo16(constInfo.commonConstInfo.dSizeV);
    uint32_t tmp = 0xFF7FFFFF;
    if ASCEND_IS_AIV {
        constInfo.commonConstInfo.s1BaseSize = CUBE_BASEM;
        constInfo.commonConstInfo.s2BaseSize = CUBE_BASEN;
        constInfo.attenMaskMinValue = *((float *)&tmp);
        constInfo.commonConstInfo.keepProb = tilingData->s1s2BNGS1S2BaseParams.keepProb;
        constInfo.sfmgMaxLoopSize = VECTOR_BASEM * VECTOR_BASEN / HEAD_DIM_ALIGN; // softmaxGrad每次最大能处理的m轴大小
        constInfo.dAlignToBlock = AlignTo(constInfo.commonConstInfo.dSizeV, INPUT_BLOCK_NUM);
        constInfo.tndMaxSumLayout = tilingData->s1s2BNGS1S2BaseParams.tndMaxSumLayout;
    }
    SetSwizzleConstInfo();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType,
                                                         VecBlockType>::GetSeqQlenKvlenByBidx(int64_t bIdx,
                                                                                              int64_t &actualSeqQlen,
                                                                                              int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx] - ((__gm__ int64_t *)actualSeqKvlenAddr)[bIdx - 1];
    }
    return;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::UpdateToken(
    FagRunInfo &runInfo, int64_t bIdx)
{
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    if constexpr (IS_ATTEN_MASK) {
        int64_t actualS1Len;
        int64_t actualS2Len;
        if (constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND && bIdx != attenMaskInfo.bandIndex) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
            actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
        } else if (constInfo.sparseMode == BAND_LEFT_UP_CASUAL && bIdx != attenMaskInfo.bandIndex) {
            actualCalcS1Token = INT32_MAX;
            actualCalcS2Token = 0;
        } else if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == BAND ||
                   (constInfo.sparseMode == RIGHT_DOWN_CASUAL_BAND && bIdx == attenMaskInfo.bandIndex) ||
                   (constInfo.sparseMode == BAND_LEFT_UP_CASUAL && bIdx == attenMaskInfo.bandIndex)) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            actualCalcS1Token = constInfo.s1Token + actualS1Len - actualS2Len;
            actualCalcS2Token = constInfo.s2Token - actualS1Len + actualS2Len;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CheckIsValidBlock(
    FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oDimIdx, int64_t taskId)
{
    int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
    int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, constInfo.commonConstInfo.s2Size);
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
        s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t *)prefixNAddr)[curBIdx]),
                       static_cast<int64_t>(constInfo.commonConstInfo.s2Size));
    } else {
        if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
            s2EndLen = Min(s2EndLen, constInfo.commonConstInfo.s2Size);
        }
    }
    if constexpr (IS_BN2_MULTIBLK) {
        multiBlkInfo.s2oDimIdx = s2oDimIdx;
        multiBlkInfo.s2OuterTmp = 0;
        multiBlkInfo.s2SparseLeft = 0;
        multiBlkInfo.s2SparseRight = s2EndLen;
        CheckS1RangeInBn2(taskId);
    }
    bool isValid = s2IdxLeft < s2EndLen;
    if (isValid) {
        int64_t bDimTail = baseIdx % constInfo.n2GS1oS2o;
        int64_t n2DimTail = bDimTail % constInfo.gS1oS2o;
        SetAxisRunInfo(runInfo, s2IdxLeft, s2IdxRight, baseIdx / constInfo.n2GS1oS2o, bDimTail / constInfo.gS1oS2o,
                       n2DimTail / constInfo.s1oS2o, s1oDimIdx, s2oDimIdx);
    }
    return isValid;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::CheckIsValidBlockForDeter(
    FagRunInfo &runInfo, int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oDimIdx, int64_t taskId)
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
        s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t *)prefixNAddr)[curBIdx]),
                       static_cast<int64_t>(constInfo.commonConstInfo.s2Size));
    } else {
        if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
            s2EndLen = Min(s2EndLen, constInfo.commonConstInfo.s2Size);
        }
    }
    if constexpr (IS_BN2_MULTIBLK) {
        multiBlkInfo.s2oDimIdx = s2oDimIdx;
        multiBlkInfo.s2OuterTmp = 0;
        multiBlkInfo.s2SparseLeft = 0;
        multiBlkInfo.s2SparseRight = s2EndLen;
        CheckS1RangeInBn2(taskId);
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
    FagRunInfo &runInfo, FagRunInfo &nextRunInfo, int64_t taskId, int64_t index, int64_t nextIndex)
{
    if constexpr (IS_TND_SWIZZLE) {
        int64_t tndS1PrefixSum =
            (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                                ((__gm__ int64_t *)actualSeqQlenAddr)[runInfo.commonRunInfo.boIdx - 1]);
        int64_t tndS2PrefixSum = (runInfo.commonRunInfo.boIdx == 0 ?
                                      0 :
                                      ((__gm__ int64_t *)actualSeqKvlenAddr)[runInfo.commonRunInfo.boIdx - 1]);
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
            runInfo.lastBatchTotalS1BRopeOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2GDr;
            runInfo.lastBatchTotalS2BRopeOffset = tndS2PrefixSum * constInfo.commonConstInfo.n2Dr;
        }

        int64_t s1OuterTmp = (runInfo.commonRunInfo.actualS1Size + CUBE_BASEM - 1) / CUBE_BASEM;
        int64_t s2OuterTmp = (runInfo.commonRunInfo.actualS2Size + VECTOR_BASEN - 1) / VECTOR_BASEN;
        int64_t s1CvTailTmp = runInfo.commonRunInfo.actualS1Size - (s1OuterTmp - 1) * CUBE_BASEM;
        runInfo.commonRunInfo.s1RealSize = (runInfo.commonRunInfo.s1oIdx == s1OuterTmp - 1) ? s1CvTailTmp : CUBE_BASEM;
        runInfo.commonRunInfo.taskId = taskId;
        runInfo.commonRunInfo.taskIdMod2 = taskId & 1;
        runInfo.commonRunInfo.s2RealSize = runInfo.s2CvEnd - runInfo.s2CvBegin; // 真实s2基本块大小
        if ASCEND_IS_AIV {
            runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
            runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
            runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
            runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
            if (vSubBlockIdx == 1) {
                runInfo.commonRunInfo.halfS1RealSize =
                    runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
                runInfo.halfS2RealSize = runInfo.commonRunInfo.s2RealSize - runInfo.halfS2RealSize;
            }
            runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
            runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
            runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
            runInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.commonRunInfo.b1SSOffset;
            runInfo.commonRunInfo.vecCoreOffset = vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
        }
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
    } else if constexpr (IS_TND) {
        int64_t resbaseIdx = index - curBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        uint64_t startBIdx = curBatchIdx;

        runInfo.lastBatchTotalBaseIdx = curBatchTotalBaseIdx;
        runInfo.lastBatchTotalS1BOffset = curBatchTotalS1BOffset;
        runInfo.lastBatchTotalS2BOffset = curBatchTotalS2BOffset;
        runInfo.lastBatchTotalS1BOffsetForDv = curBatchTotalS1BOffsetForDv;
        runInfo.lastBatchTotalS2BOffsetForDv = curBatchTotalS2BOffsetForDv;
        runInfo.lastBatchTotalS1S2SizeAlign = curBatchTotalS1S2SizeAlign;
        runInfo.lastBatchTotalS1S2Size = curBatchTotalS1S2Size;
        runInfo.lastBatchTotalS2Size = curBatchTotalS2Size;
        if constexpr (IS_ROPE) {
            runInfo.lastBatchTotalS1BRopeOffset = curBatchTotalS1BRopeOffset;
            runInfo.lastBatchTotalS2BRopeOffset = curBatchTotalS2BRopeOffset;
        }

        for (int64_t bIdx = startBIdx; bIdx < constInfo.bSize; bIdx++) {
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
                if ASCEND_IS_AIV {
                    runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
                    runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
                    runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
                    runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
                    if (vSubBlockIdx == 1) {
                        runInfo.commonRunInfo.halfS1RealSize =
                            runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
                        runInfo.halfS2RealSize = runInfo.commonRunInfo.s2RealSize - runInfo.halfS2RealSize;
                    }
                }
                curBatchIdx = bIdx;

                int64_t batchRemainBlockNum =
                    (s1OuterTmp - runInfo.commonRunInfo.s1oIdx - 1) + (s2OuterTmp - runInfo.s2oIdx - 1) * s1OuterTmp +
                    (totalBaseIdx - (runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize +
                                     runInfo.commonRunInfo.goIdx + 1) *
                                        s1OuterTmp * s2OuterTmp);
                if ((totalBaseIdx - resbaseIdx == 1) || (nextIndex - index > batchRemainBlockNum)) {
                    curBatchIdx = bIdx + 1;
                    curBatchTotalBaseIdx += totalBaseIdx;
                    curBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                    curBatchTotalS2BOffset += actualS2Len * constInfo.commonConstInfo.n2D;
                    curBatchTotalS1BOffsetForDv += actualS1Len * constInfo.commonConstInfo.n2GDv;
                    curBatchTotalS2BOffsetForDv += actualS2Len * constInfo.commonConstInfo.n2Dv;
                    curBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
                    curBatchTotalS1S2Size += actualS1Len * actualS2Len;
                    curBatchTotalS2Size += actualS2Len;
                    if constexpr (IS_ROPE) {
                        curBatchTotalS1BRopeOffset += actualS1Len * constInfo.commonConstInfo.n2GDr;
                        curBatchTotalS2BRopeOffset += actualS2Len * constInfo.commonConstInfo.n2Dr;
                    }
                }
                break;
            } else {
                runInfo.lastBatchTotalBaseIdx = curBatchTotalBaseIdx;
                runInfo.lastBatchTotalS1BOffset = curBatchTotalS1BOffset;
                runInfo.lastBatchTotalS2BOffset = curBatchTotalS2BOffset;
                runInfo.lastBatchTotalS1BOffsetForDv = curBatchTotalS1BOffsetForDv;
                runInfo.lastBatchTotalS2BOffsetForDv = curBatchTotalS2BOffsetForDv;
                runInfo.lastBatchTotalS1S2SizeAlign = curBatchTotalS1S2SizeAlign;
                runInfo.lastBatchTotalS1S2Size = curBatchTotalS1S2Size;
                runInfo.lastBatchTotalS2Size = curBatchTotalS2Size;
                if constexpr (IS_ROPE) {
                    runInfo.lastBatchTotalS1BRopeOffset = curBatchTotalS1BRopeOffset;
                    runInfo.lastBatchTotalS2BRopeOffset = curBatchTotalS2BRopeOffset;
                }

                resbaseIdx = index - curBatchTotalBaseIdx;
                curBatchTotalBaseIdx += totalBaseIdx;
                curBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                curBatchTotalS2BOffset += actualS2Len * constInfo.commonConstInfo.n2D;
                curBatchTotalS1BOffsetForDv += actualS1Len * constInfo.commonConstInfo.n2GDv;
                curBatchTotalS2BOffsetForDv += actualS2Len * constInfo.commonConstInfo.n2Dv;
                curBatchTotalS1S2SizeAlign += actualS1Len * AlignTo16(actualS2Len);
                curBatchTotalS1S2Size += actualS1Len * actualS2Len;
                curBatchTotalS2Size += actualS2Len;
                if constexpr (IS_ROPE) {
                    curBatchTotalS1BRopeOffset += actualS1Len * constInfo.commonConstInfo.n2GDr;
                    curBatchTotalS2BRopeOffset += actualS2Len * constInfo.commonConstInfo.n2Dr;
                }
                // when s1 or s2 = 0, bIdx changed, need to update runInfo
                if (totalBaseIdx == 0) {
                    runInfo.lastBatchTotalBaseIdx = curBatchTotalBaseIdx;
                    runInfo.lastBatchTotalS1BOffset = curBatchTotalS1BOffset;
                    runInfo.lastBatchTotalS2BOffset = curBatchTotalS2BOffset;
                    runInfo.lastBatchTotalS1BOffsetForDv = curBatchTotalS1BOffsetForDv;
                    runInfo.lastBatchTotalS2BOffsetForDv = curBatchTotalS2BOffsetForDv;
                    runInfo.lastBatchTotalS1S2SizeAlign = curBatchTotalS1S2SizeAlign;
                    runInfo.lastBatchTotalS1S2Size = curBatchTotalS1S2Size;
                    runInfo.lastBatchTotalS2Size = curBatchTotalS2Size;
                    if constexpr (IS_ROPE) {
                        runInfo.lastBatchTotalS1BRopeOffset = curBatchTotalS1BRopeOffset;
                        runInfo.lastBatchTotalS2BRopeOffset = curBatchTotalS2BRopeOffset;
                    }
                }
            }
        }
        runInfo.commonRunInfo.actualS1Size = actualS1Len;
        runInfo.commonRunInfo.actualS2Size = actualS2Len;
        runInfo.commonRunInfo.s2SizeAcc = runInfo.lastBatchTotalS2Size;
        runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.lastBatchTotalS1S2SizeAlign;
        runInfo.commonRunInfo.b1SSOffset = runInfo.lastBatchTotalS1S2Size;
        runInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.commonRunInfo.b1SSOffset;
        runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
        runInfo.commonRunInfo.vecCoreOffset = vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
    } else {
        if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
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
        if ASCEND_IS_AIV {
            runInfo.halfS2RealSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1;
            runInfo.firstHalfS2RealSize = runInfo.halfS2RealSize;
            runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
            runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
            if (vSubBlockIdx == 1) {
                runInfo.commonRunInfo.halfS1RealSize =
                    runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
                runInfo.halfS2RealSize = runInfo.commonRunInfo.s2RealSize - runInfo.halfS2RealSize;
            }
            // pse
            runInfo.commonRunInfo.b1SSOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1S2;
            runInfo.commonRunInfo.b1SSAttenMaskOffset = runInfo.commonRunInfo.b1SSOffset;
            runInfo.commonRunInfo.s2SizeAcc = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s2Size;
            runInfo.commonRunInfo.s2StartIdx = runInfo.s2CvBegin;
            runInfo.commonRunInfo.vecCoreOffset = vSubBlockIdx * runInfo.commonRunInfo.firstHalfS1RealSize;
            runInfo.commonRunInfo.b1SSOffsetAlign = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.s1Size *
                                                    AlignTo16(constInfo.commonConstInfo.s2Size);
            runInfo.commonRunInfo.preTokensPerBatch = attenMaskInfo.preTokens;
            runInfo.commonRunInfo.nextTokensPerBatch = attenMaskInfo.nextTokens;
        }
        runInfo.commonRunInfo.s2AlignedSize = AlignTo16(runInfo.commonRunInfo.s2RealSize);
        runInfo.commonRunInfo.actualS1Size = constInfo.commonConstInfo.s1Size;
        runInfo.commonRunInfo.actualS2Size = constInfo.commonConstInfo.s2Size;
    }

    runInfo.s1AlignedSize = AlignTo16(runInfo.commonRunInfo.s1RealSize);
    // BN2扩展模板专用
    if constexpr (IS_BN2_MULTIBLK) {
        runInfo.isLastS1Outer = isLastS1Outer[taskId & 1];
        runInfo.isFirstS1Outer = isFirstS1Outer[taskId & 1];
    }

    runInfo.isS2IdxNoChange = (lastS2oCvDimIdx == runInfo.s2oIdx && lastBdimIdx == runInfo.commonRunInfo.boIdx &&
                               lastN2dimIdx == runInfo.commonRunInfo.n2oIdx);
    if (!runInfo.isS2IdxNoChange) {
        lastS2oCvDimIdx = runInfo.s2oIdx;
        lastBdimIdx = runInfo.commonRunInfo.boIdx;
        lastN2dimIdx = runInfo.commonRunInfo.n2oIdx;
    }
    if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) {
        runInfo.sinkN1Idx =
            runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx;
        uint64_t s1oIdxSink =
            vSubBlockIdx == 1 ? runInfo.commonRunInfo.s1oIdx * 2 + 1 : runInfo.commonRunInfo.s1oIdx * 2;
        // [N, B, S1, S2]
        runInfo.dsinkWorkSpaceOffset =
            runInfo.sinkN1Idx * constInfo.bSize * constInfo.s1SinkOuter * constInfo.s2SinkOuter +
            runInfo.commonRunInfo.boIdx * constInfo.s1SinkOuter * constInfo.s2SinkOuter +
            s1oIdxSink * constInfo.s2SinkOuter + runInfo.s2oIdx;
    }

    // preload next query and dy offset for l1 preload
    if (taskId == 0) {
        runInfo.commonRunInfo.queryOffset = GetQueryOffset(runInfo);
        runInfo.dyOffset = runInfo.commonRunInfo.queryOffset;
        if constexpr (IS_D_NO_EQUAL) {
            runInfo.dyOffset = GetDxOffset(runInfo);
        }
        GetNextDxAndQueryOffset(runInfo, nextRunInfo, nextIndex,
                                preloadArgs); // get nextQueryOffset, nextDyOffset, nextMorN
    } else {
        runInfo.commonRunInfo.queryOffset = preloadArgs.nextQueryOffset;
        runInfo.dyOffset = preloadArgs.nextDyOffset;
        GetNextDxAndQueryOffset(runInfo, nextRunInfo, nextIndex,
                                preloadArgs); // get nextQueryOffset, nextDyOffset, nextMorN
    }

    if constexpr (SPLIT_AXIS == BN2GS1S2 && !IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if ASCEND_IS_AIV {
            return;
        }
    }
    runInfo.commonRunInfo.keyOffset = GetKeyOffset(runInfo);
    runInfo.commonRunInfo.valueOffset = runInfo.commonRunInfo.keyOffset;
    if constexpr (IS_D_NO_EQUAL) {
        runInfo.commonRunInfo.valueOffset = GetValueOffset(runInfo);
    }

    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        // deter old 场景 aic/aiv以下值都需要获取
        runInfo.queryOffsetWithRope = runInfo.commonRunInfo.queryOffset;
        runInfo.keyOffsetWithRope = runInfo.commonRunInfo.keyOffset;
        // Rope场景后面三个mm的GM offset不能和前面两个mm共用，因此需要重新计算
        if constexpr (IS_ROPE) {
            runInfo.queryOffsetWithRope = GetQueryOffset<false>(runInfo);
            runInfo.keyOffsetWithRope = GetKeyOffset<false>(runInfo);
            runInfo.queryOffsetWithRopeForMm12 = GetQueryOffset<true>(runInfo);
            runInfo.keyOffsetWithRopeForMm12 = GetKeyOffset<true>(runInfo);
            runInfo.commonRunInfo.qRopeOffset = GetQueryRopeOffset(runInfo);
            runInfo.commonRunInfo.kRopeOffset = GetKeyRopeOffset(runInfo);
        }
    } else {
        if ASCEND_IS_AIC {
            runInfo.queryOffsetWithRope = runInfo.commonRunInfo.queryOffset;
            runInfo.keyOffsetWithRope = runInfo.commonRunInfo.keyOffset;
            // Rope场景后面三个mm的GM offset不能和前面两个mm共用，因此需要重新计算
            if constexpr (IS_ROPE) {
                runInfo.queryOffsetWithRope = GetQueryOffset<false>(runInfo);
                runInfo.keyOffsetWithRope = GetKeyOffset<false>(runInfo);
                runInfo.queryOffsetWithRopeForMm12 = GetQueryOffset<true>(runInfo);
                runInfo.keyOffsetWithRopeForMm12 = GetKeyOffset<true>(runInfo);
                runInfo.commonRunInfo.qRopeOffset = GetQueryRopeOffset(runInfo);
                runInfo.commonRunInfo.kRopeOffset = GetKeyRopeOffset(runInfo);
            }
        } else {
            if constexpr (SPLIT_AXIS == BN2 && IS_ROPE) {
                runInfo.queryOffsetWithRopeForMm12 = GetQueryOffset<true>(runInfo);
                runInfo.keyOffsetWithRopeForMm12 = GetKeyOffset<true>(runInfo);
                runInfo.commonRunInfo.qRopeOffset = GetQueryRopeOffset(runInfo);
                runInfo.commonRunInfo.kRopeOffset = GetKeyRopeOffset(runInfo);
            }
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetAxisRunInfo(
    FagRunInfo &runInfo, int64_t s2CvBegin, int64_t s2CvEnd, int64_t boIdx, int64_t n2oIdx, int64_t goIdx,
    int64_t s1oIdx, int64_t s2oIdx)
{
    runInfo.s2CvBegin = s2CvBegin;
    runInfo.s2CvEnd = s2CvEnd;
    runInfo.commonRunInfo.boIdx = boIdx;
    runInfo.commonRunInfo.n2oIdx = n2oIdx;
    runInfo.commonRunInfo.goIdx = goIdx;
    runInfo.commonRunInfo.s1oIdx = s1oIdx;
    runInfo.s2oIdx = s2oIdx;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IsValidForTND(
    FagRunInfo &runInfo, int64_t taskId, int64_t index)
{
    int64_t resbaseIdx = 0;
    int64_t startBIdx = 0;
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        resbaseIdx = index - tilingData->tndParam.tndPrefixSum[cBlockIdx] * constInfo.commonConstInfo.n2G;
        startBIdx = tilingData->tndParam.tndStartBIdx[cBlockIdx];
    } else {
        resbaseIdx = index - curBatchTotalBaseIdx;
        startBIdx = curBatchIdx;
    }
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    for (int64_t bIdx = startBIdx; bIdx < constInfo.bSize; bIdx++) {
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        int64_t s2OuterTmp = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        int64_t totalBaseIdx = constInfo.commonConstInfo.n2G * s1OuterTmp * s2OuterTmp;
        if (resbaseIdx < totalBaseIdx) {
            int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
            int64_t s2oDimIdx = gDimTail / s1OuterTmp;
            int64_t s1oDimIdx = gDimTail % s1OuterTmp;
            int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
            int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, actualS2Len);
            if constexpr (SPLIT_AXIS == BN2S2) {
                if (curS2oIdx == -1 || curS2oIdx != s2oDimIdx) {
                    curS2oIdx = s2oDimIdx;
                    curS2InvalidTotalNum = 0;
                }
            }
            if constexpr (IS_ATTEN_MASK) {
                if (constInfo.sparseMode == PREFIX_COMPRESS) {
                    int64_t s2IgnoredEndLen = actualS1Len - static_cast<int64_t>(CUBE_BASEM * (s1oDimIdx + 1));
                    int64_t s2EndLen = 0;
                    if (actualS2Len > s2IgnoredEndLen) {
                        s2EndLen = actualS2Len - s2IgnoredEndLen;
                    }
                    if (constInfo.sparseMode == PREFIX || constInfo.sparseMode == PREFIX_COMPRESS) {
                        s2EndLen = Min(Max(s2EndLen, ((__gm__ int64_t *)prefixNAddr)[bIdx]), actualS2Len);
                    }
                    bool isValid = s2IdxLeft < s2EndLen;
                    if (isValid) {
                        runInfo.s2CvBegin = s2IdxLeft;
                        runInfo.s2CvEnd = s2IdxRight;
                    }
                    if constexpr (IS_BN2_MULTIBLK) {
                        multiBlkInfo.s2oDimIdx = s2oDimIdx;
                        multiBlkInfo.s2OuterTmp = 0;
                        multiBlkInfo.s2SparseLeft = 0;
                        multiBlkInfo.s2SparseRight = s2EndLen;
                        CheckS1RangeInBn2(taskId);
                    }
                    if constexpr (SPLIT_AXIS == BN2S2) {
                        if (!isValid) {
                            curS2InvalidTotalNum += 1;
                        }
                        if (curS2InvalidTotalNum * CUBE_BASEM >= actualS1Len) {
                            runInfo.s2CvBegin = s2IdxLeft;
                            runInfo.s2CvEnd = s2IdxRight;
                            return true;
                        }
                    }
                    return isValid;
                }

                UpdateToken(runInfo, bIdx);
                int64_t s2SparseLeft = Max(CUBE_BASEM * s1oDimIdx - actualCalcS1Token, 0);
                s2SparseLeft = s2SparseLeft >> 6 << 6;
                int64_t s2SparseRight =
                    AlignTo64(Min(CUBE_BASEM * (s1oDimIdx + 1), constInfo.commonConstInfo.s1Size) + actualCalcS2Token);
                s2SparseRight = Min(s2SparseRight, actualS2Len);
                bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                runInfo.s2CvBegin = s2IdxLeft;
                runInfo.s2CvEnd = s2IdxRight;
                if constexpr (IS_BN2_MULTIBLK) {
                    multiBlkInfo.s2oDimIdx = s2oDimIdx;
                    multiBlkInfo.s2OuterTmp = 0;
                    multiBlkInfo.s2SparseLeft = s2SparseLeft;
                    multiBlkInfo.s2SparseRight = s2SparseRight;
                    CheckS1RangeInBn2(taskId);
                }
                if constexpr (SPLIT_AXIS == BN2S2) {
                    if (!isValid) {
                        curS2InvalidTotalNum += 1;
                    }
                    if (curS2InvalidTotalNum * CUBE_BASEM >= actualS1Len) {
                        return true;
                    }
                }
                return isValid;
            } else {
                runInfo.s2CvBegin = s2IdxLeft;
                runInfo.s2CvEnd = s2IdxRight;
                if constexpr (IS_BN2_MULTIBLK) {
                    multiBlkInfo.s2oDimIdx = s2oDimIdx;
                    multiBlkInfo.s2OuterTmp = s2OuterTmp;
                    multiBlkInfo.s2SparseLeft = 0;
                    multiBlkInfo.s2SparseRight = 0;
                    CheckS1RangeInBn2(taskId);
                }
                return true;
            }
        } else {
            resbaseIdx -= totalBaseIdx;
        }
    }
    return false;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IsValid(
    FagRunInfo &runInfo, int64_t taskId, int64_t index)
{
    if constexpr (IS_TND) {
        return IsValidForTND(runInfo, taskId, index);
    } else {
        int64_t gDimTail = index % constInfo.s1oS2o;
        int64_t s2oDimIdx = gDimTail / constInfo.s1Outer;
        int64_t s1oDimIdx = gDimTail % constInfo.s1Outer;
        int64_t s2IdxLeft = s2oDimIdx * CUBE_BASEN;
        int64_t s2IdxRight = Min((s2oDimIdx + 1) * CUBE_BASEN, constInfo.commonConstInfo.s2Size);
        if constexpr (IS_ATTEN_MASK) {
            if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL || constInfo.sparseMode == PREFIX ||
                constInfo.sparseMode == PREFIX_COMPRESS) {
                return CheckIsValidBlock(runInfo, index, s1oDimIdx, s2oDimIdx, taskId);
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

                // set axis runInfo
                if (isValid) {
                    int64_t bDimTail = index % constInfo.n2GS1oS2o;
                    int64_t n2DimTail = bDimTail % constInfo.gS1oS2o;
                    SetAxisRunInfo(runInfo, s2IdxLeft, s2IdxRight, index / constInfo.n2GS1oS2o,
                                   bDimTail / constInfo.gS1oS2o, n2DimTail / constInfo.s1oS2o, s1oDimIdx, s2oDimIdx);
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
            int64_t bDimTail = index % constInfo.n2GS1oS2o;
            int64_t n2DimTail = bDimTail % constInfo.gS1oS2o;
            SetAxisRunInfo(runInfo, s2IdxLeft, s2IdxRight, index / constInfo.n2GS1oS2o, bDimTail / constInfo.gS1oS2o,
                           n2DimTail / constInfo.s1oS2o, s1oDimIdx, s2oDimIdx);
            return true;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::IsValidForDeter(
    FagRunInfo &runInfo, int64_t taskId, int64_t index)
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
        s2CvBegin = s2IdxLeft;
        s2CvEnd = s2IdxRight;
        if constexpr (IS_BN2_MULTIBLK) {
            multiBlkInfo.s2oDimIdx = s2oDimIdx;
            multiBlkInfo.s2OuterTmp = constInfo.s2Outer;
            multiBlkInfo.s2SparseLeft = 0;
            multiBlkInfo.s2SparseRight = 0;
            CheckS1RangeInBn2(taskId);
        }
        return true;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdx(
    FagRunInfo &runInfo, int64_t taskId, int64_t blockInnerIdx, int64_t curLoopIdx)
{
    if constexpr (IS_TND_SWIZZLE) {
        uint8_t sparseType = static_cast<uint8_t>(tilingData->s1s2BNGS1S2BaseParams.sparseType);
        if (sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
            return GetNextValidIdxForTndSwizzleDense(runInfo, curLoopIdx);
        } else if (sparseType == static_cast<uint8_t>(SparseType::CASUAL)) {
            return GetNextValidIdxForTndSwizzleCasual(runInfo, curLoopIdx);
        } else {
            return GetNextValidIdxForTndSwizzleBand(runInfo, curLoopIdx);
        }
    } else {
        int64_t nextValidBlockInnerIdx = 0;
        if constexpr (IS_NORMAL_SWIZZLE) {
            if (tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx) {
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
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForTndSwizzleDense(
    FagRunInfo &runInfo, int64_t loopIdx)
{
    bool isEnd = true;
    for (int64_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
        if (loopIdx < tilingData->tndSwizzleParam.tndS2BlockPrefixSum[bIdx + 1]) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            int64_t delta = loopIdx - tilingData->tndSwizzleParam.tndS2BlockPrefixSum[bIdx] + deltaCnt;
            // 更正delta
            if (delta < 0) {
                deltaCnt += (-delta);
                delta = 0;
            }
            int64_t s1Idx = delta % s1OuterTmp;
            // delta / s1OuterTmp表示在此bIdx下，s2的绝对idx
            int64_t s2IdxTmp = delta / s1OuterTmp * constInfo.aicCoreNum + cBlockIdx;
            if (s2IdxTmp >= s2OuterTmp * constInfo.commonConstInfo.n2G) {
                continue;
            }
            int64_t n1Idx = s2IdxTmp / s2OuterTmp;
            int64_t s2Idx = s2IdxTmp - s2OuterTmp * n1Idx;
            SetAxisRunInfo(runInfo, s2Idx * CUBE_BASEN, min((s2Idx + 1) * CUBE_BASEN, actualS2Len), bIdx,
                           n1Idx / constInfo.commonConstInfo.gSize, n1Idx % constInfo.commonConstInfo.gSize, s1Idx,
                           s2Idx);
            runInfo.commonRunInfo.actualS1Size = actualS1Len;
            runInfo.commonRunInfo.actualS2Size = actualS2Len;
            isEnd = false;
            break;
        }
    }
    if (isEnd) {
        return -1;
    }
    return loopIdx;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForTndSwizzleCasual(
    FagRunInfo &runInfo, int64_t loopIdx)
{
    bool isEnd = true;
    int64_t halfN1 = constInfo.commonConstInfo.n2G >> 1;
    int64_t s1OuterTmp = 0;
    int64_t s2OuterTmp = 0;
    for (int64_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
        if (loopIdx < tilingData->tndSwizzleParam.tndS2BlockPrefixSum[bIdx + 1]) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            int64_t s1Outer = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2Outer = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            // 处理无效列场景
            if (constInfo.sparseMode == LEFT_UP_CAUSAL && s1Outer < s2Outer) {
                s2Outer = s1Outer;
            } else if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL && s1Outer > s2Outer) {
                s1Outer = s2Outer;
            }
            s1OuterTmp = s1Outer;
            s2OuterTmp = s2Outer;
            if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
                s2OuterTmp = (s2Outer << 1) - s1Outer + 1;
            } else {
                s1OuterTmp = (s1Outer << 1) - s2Outer + 1;
            }
            int64_t delta = loopIdx - tilingData->tndSwizzleParam.tndS2BlockPrefixSum[bIdx] + deltaCnt;
            // 更正delta
            if (delta < 0) {
                deltaCnt += (-delta);
                delta = 0;
            }
            int64_t s1Idx = delta % s1OuterTmp;
            // delta / s1OuterTmp表示在此bIdx下，s2的idx
            int64_t s2IdxTmp = delta / s1OuterTmp * constInfo.aicCoreNum + cBlockIdx;
            if (s2IdxTmp >= s2OuterTmp * halfN1) {
                continue;
            }
            int64_t n1Idx = s2IdxTmp / s2OuterTmp;
            int64_t s2Idx = s2IdxTmp - s2OuterTmp * n1Idx; // 相邻N拼起来后的s2Idx

            if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
                if (s2Idx >= s1Idx + s2Outer - s1Outer + 1) {
                    n1Idx = (n1Idx << 1) + 1;
                    if (s2Idx < s2Outer) {
                        s2Idx = (s2Outer << 1) - s1Outer - s2Idx;
                        s1Idx = s1Idx + s2Idx - s2Outer + s1Outer;
                    } else {
                        s2Idx = (s2Outer << 1) - s1Outer - s2Idx;
                    }
                } else {
                    n1Idx = (n1Idx << 1);
                }
            } else {
                if (s1Idx >= s2Idx + s1Outer - s2Outer + 1) {
                    s1Idx = s1Idx - (s1Outer - s2Outer + 1);
                    n1Idx = (n1Idx << 1) + 1;
                } else {
                    s1Idx = s1Outer - 1 - s1Idx;
                    s2Idx = s2Outer - 1 - s2Idx;
                    n1Idx = n1Idx << 1;
                }
            }
            SetAxisRunInfo(runInfo, s2Idx * CUBE_BASEN, min((s2Idx + 1) * CUBE_BASEN, actualS2Len), bIdx,
                           n1Idx / constInfo.commonConstInfo.gSize, n1Idx % constInfo.commonConstInfo.gSize, s1Idx,
                           s2Idx);
            runInfo.commonRunInfo.actualS1Size = actualS1Len;
            runInfo.commonRunInfo.actualS2Size = actualS2Len;
            isEnd = false;
            break;
        }
    }
    if (isEnd) {
        return -1;
    }
    return loopIdx;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForTndSwizzleBand(
    FagRunInfo &runInfo, int64_t loopIdx)
{
    int64_t oriPreTokenBlock = (attenMaskInfo.preTokens + CUBE_BASEM - 1) / CUBE_BASEM + 1;
    int64_t oriNextTokenBlock = (attenMaskInfo.nextTokens + CUBE_BASEN - 1) / CUBE_BASEN + 1;
    int64_t preTokenBlock = 0;
    int64_t nextTokenBlock = 0;
    int64_t s2IncreaseValidBlockLen = 0;
    int64_t s2EqualValidBlockLen = 0;
    int64_t s2DecreaseValidBlockLen = 0;
    int64_t s2MaxValidBlockLen = 0;
    int64_t n1Idx = 0;
    int64_t s1Idx = 0;
    int64_t s2Idx = 0;
    int64_t s1IdxTmp = 0;
    int64_t s2IdxTmp1;
    int64_t s2IdxTmp2;
    int64_t lastBIdx = 0;
    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    int64_t s1OuterTmp = 0;
    int64_t s2OuterTmp = 0;
    // while循环跳过可能的轮空
    while (true) {
        bool isEnd = true;
        for (int64_t bIdx = lastBIdx; bIdx < constInfo.bSize; bIdx++) {
            if (bandLoopIdx < tilingData->tndSwizzleParam.tndS2BlockPrefixSum[bIdx + 1]) {
                int64_t actualS1Len = 0;
                int64_t actualS2Len = 0;
                GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
                s1Outer = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
                s2Outer = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
                preTokenBlock = oriPreTokenBlock > s1Outer ? s1Outer : oriPreTokenBlock;
                nextTokenBlock = oriNextTokenBlock > s2Outer ? s2Outer : oriNextTokenBlock;
                int64_t delta = bandLoopIdx - tilingData->tndSwizzleParam.tndS2BlockPrefixSum[bIdx] + deltaCnt;
                // 更正delta
                if (delta < 0) {
                    deltaCnt += (-delta);
                    delta = 0;
                }
                if (preTokenBlock + nextTokenBlock <= s1Outer) {
                    // s2方向有效块递增序列的宽度
                    s2IncreaseValidBlockLen = nextTokenBlock - 1;
                    // s2方向有效块相等序列的宽度
                    s2EqualValidBlockLen =
                        Min(s2Outer - nextTokenBlock + 1, s1Outer + NUM_TWO - preTokenBlock - nextTokenBlock);
                    // s2方向有效块递减序列的宽度
                    s2DecreaseValidBlockLen =
                        Max(0, Min(preTokenBlock + s2Outer - s1Outer - 1, preTokenBlock + nextTokenBlock - NUM_TWO));
                    if (s2DecreaseValidBlockLen == 0) {
                        s1OuterTmp = preTokenBlock + nextTokenBlock + s2EqualValidBlockLen - NUM_TWO;
                    } else {
                        s1OuterTmp = s1Outer;
                    }
                    s2OuterTmp = s2IncreaseValidBlockLen + s2EqualValidBlockLen + s2DecreaseValidBlockLen;
                    // batch内最长的列的有效长度
                    s2MaxValidBlockLen = preTokenBlock + nextTokenBlock - 1;
                    s2IdxTmp1 = delta / s2MaxValidBlockLen * constInfo.aicCoreNum + cBlockIdx;
                    // 无法斜向拼接
                    if (s2DecreaseValidBlockLen == 0 || s2OuterTmp - s1OuterTmp < 1) {
                        s1Idx = delta % s2MaxValidBlockLen;
                        if (s2IdxTmp1 >= s2OuterTmp * constInfo.commonConstInfo.n2G) {
                            continue;
                        }
                        n1Idx = s2IdxTmp1 / s2OuterTmp;
                        s2Idx = s2IdxTmp1 - s2OuterTmp * n1Idx;
                        s1Idx += (s2Idx - nextTokenBlock + 1);
                    }
                    // 斜向拼接
                    if (s2DecreaseValidBlockLen > 0 && s2OuterTmp - s1OuterTmp >= 1) {
                        s2IdxTmp2 = delta % s2MaxValidBlockLen;
                        s1IdxTmp = s2IdxTmp2 + s2IdxTmp1 + 1 - nextTokenBlock;
                        if (s1IdxTmp < 0) {
                            n1Idx = constInfo.commonConstInfo.n2G - 1;
                            s2Idx = s1IdxTmp + s1OuterTmp;
                            s1Idx = s2IdxTmp1 + s1OuterTmp;
                        } else {
                            n1Idx = s1IdxTmp / s1OuterTmp;
                            s1Idx = s1IdxTmp % s1OuterTmp;
                            s2Idx = s1Idx + nextTokenBlock - 1 - s2IdxTmp2;
                            if (n1Idx == constInfo.commonConstInfo.n2G - 1 && s2Idx >= s1OuterTmp) {
                                continue;
                            }
                        }
                    }
                } else {
                    s1OuterTmp = s1Outer;
                    s2OuterTmp = Min(s1Outer - 1 + nextTokenBlock, s2Outer);

                    s1Idx = delta % s1OuterTmp;
                    s2IdxTmp1 = delta / s1OuterTmp * constInfo.aicCoreNum + cBlockIdx;
                    if (s2IdxTmp1 >= s2OuterTmp * constInfo.commonConstInfo.n2G) {
                        continue;
                    }
                    n1Idx = s2IdxTmp1 / s1OuterTmp;
                    s2Idx = s2IdxTmp1 - s2OuterTmp * n1Idx;
                }

                SetAxisRunInfo(runInfo, s2Idx * CUBE_BASEN, min((s2Idx + 1) * CUBE_BASEN, actualS2Len), bIdx,
                               n1Idx / constInfo.commonConstInfo.gSize, n1Idx % constInfo.commonConstInfo.gSize, s1Idx,
                               s2Idx);
                runInfo.commonRunInfo.actualS1Size = actualS1Len;
                runInfo.commonRunInfo.actualS2Size = actualS2Len;
                isEnd = false;
                lastBIdx = bIdx;
                break;
            }
        }
        bandLoopIdx++;
        if (isEnd) {
            return -1;
        }
        if (s1Idx < 0 || s2Idx < 0 || s1Idx >= s1Outer || s2Idx >= s2Outer || s1Idx - s2Idx >= preTokenBlock ||
            s2Idx - s1Idx >= nextTokenBlock) {
            continue;
        } else {
            break;
        }
    }
    return loopIdx;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForSwizzleDense(
    FagRunInfo &runInfo, int64_t loopIdx)
{
    while (true) {
        int64_t bIdx = swizzleLoopIdx / constInfo.batchTotalRound;
        loopIdx = swizzleLoopIdx % constInfo.batchTotalRound;
        if (bIdx >= constInfo.bSize * constInfo.n2Size * constInfo.commonConstInfo.gSize) {
            return -1;
        }
        int64_t globalS2Idx = 0;
        int64_t globalS1Idx = 0;
        int64_t localS1Idx = 0;
        int64_t localS2Idx = 0;
        bool isEnd = false;
        if (loopIdx < constInfo.leftTotalRound) {
            globalS2Idx = loopIdx / constInfo.leftSingleColTotalRound;
            int64_t tmp = loopIdx % constInfo.leftSingleColTotalRound;
            globalS1Idx = tmp / constInfo.leftUpTotalRound;
            int64_t rTmp =
                loopIdx - globalS2Idx * constInfo.leftSingleColTotalRound - globalS1Idx * constInfo.leftUpTotalRound;
            if (globalS1Idx == constInfo.mSwizzleBlockNum) {
                isEnd = GetLocalS1S2Idx<true>(rTmp, constInfo.mSwizzleBlockNumTail, N_SWIZZLE_BLOCK_SIZE, localS1Idx,
                                              localS2Idx);
            } else {
                isEnd = GetLocalS1S2Idx<true>(rTmp, M_SWIZZLE_BLOCK_SIZE, N_SWIZZLE_BLOCK_SIZE, localS1Idx, localS2Idx);
            }
        } else {
            loopIdx = loopIdx - constInfo.leftTotalRound;
            globalS2Idx = constInfo.nSwizzleBlockNum;
            globalS1Idx = loopIdx / constInfo.rightUpTotalRound;
            int64_t rTmp = loopIdx - globalS1Idx * constInfo.rightUpTotalRound;
            if (globalS1Idx == constInfo.mSwizzleBlockNum) {
                isEnd = GetLocalS1S2Idx<true>(rTmp, constInfo.mSwizzleBlockNumTail, constInfo.nSwizzleBlockNumTail,
                                              localS1Idx, localS2Idx);
            } else {
                isEnd = GetLocalS1S2Idx<true>(rTmp, M_SWIZZLE_BLOCK_SIZE, constInfo.nSwizzleBlockNumTail, localS1Idx,
                                              localS2Idx);
            }
        }
        swizzleLoopIdx++;
        if (isEnd) {
            continue;
        }

        int64_t s1Idx = M_SWIZZLE_BLOCK_SIZE * globalS1Idx + localS1Idx;
        int64_t s2Idx = N_SWIZZLE_BLOCK_SIZE * globalS2Idx + localS2Idx;
        if (s1Idx < 0 || s1Idx >= constInfo.s1Outer || s2Idx < 0 || s2Idx >= constInfo.s2Outer) {
            continue;
        }
        runInfo.s2CvBegin = s2Idx * CUBE_BASEN;
        runInfo.s2CvEnd = runInfo.s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
        if (s2Idx == constInfo.s2Outer - 1) {             // 默认s2 cv tail相等
            runInfo.s2CvEnd = runInfo.s2CvBegin + constInfo.s2Tail;
        }
        runInfo.commonRunInfo.boIdx = bIdx / constInfo.commonConstInfo.n2G;
        int64_t bDimTail = bIdx % constInfo.commonConstInfo.n2G;
        runInfo.commonRunInfo.n2oIdx = bDimTail / constInfo.commonConstInfo.gSize;
        runInfo.commonRunInfo.goIdx = bDimTail % constInfo.commonConstInfo.gSize;
        runInfo.commonRunInfo.s1oIdx = s1Idx;
        runInfo.s2oIdx = s2Idx;
        return 0;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForSwizzleCasual(
    FagRunInfo &runInfo, int64_t loopIdx)
{
    while (true) {
        int64_t bIdx = swizzleLoopIdx / constInfo.batchTotalRound;
        loopIdx = swizzleLoopIdx % constInfo.batchTotalRound;
        if (bIdx >= constInfo.bSize * constInfo.n2Size * constInfo.commonConstInfo.gSize) {
            return -1;
        }
        int64_t globalS2Idx = 0;
        int64_t globalS1Idx = 0;
        int64_t localS1Idx = 0;
        int64_t localS2Idx = 0;
        bool isEnd = false;
        if (loopIdx < constInfo.leftTotalRound) {
            int64_t tmp = constInfo.leftUpTotalRound + constInfo.rightUpTotalRound +
                          constInfo.leftDownTotalRound * (constInfo.mSwizzleBlockNum - 1);
            // 求根公式求解二次方程
            int64_t a = (-constInfo.leftDownTotalRound / NUM_TWO);
            int64_t b = constInfo.leftDownTotalRound / NUM_TWO - tmp;
            int64_t c = tmp - loopIdx - 1;
            globalS2Idx = Ceil<int64_t>(-b - sqrt(b * b - ((a * c) << NUM_TWO)), constInfo.leftDownTotalRound);
            loopIdx = loopIdx - (globalS2Idx * (constInfo.leftUpTotalRound + constInfo.rightUpTotalRound) +
                                 (NUM_TWO * constInfo.mSwizzleBlockNum - 1 - globalS2Idx) * globalS2Idx / NUM_TWO *
                                     constInfo.leftDownTotalRound);
            if (loopIdx < constInfo.leftUpTotalRound) {
                globalS1Idx = globalS2Idx;
                isEnd =
                    GetLocalS1S2Idx<false>(loopIdx, M_SWIZZLE_BLOCK_SIZE, M_SWIZZLE_BLOCK_SIZE, localS1Idx, localS2Idx);
            } else if (constInfo.leftUpTotalRound <= loopIdx &&
                       loopIdx < (constInfo.leftUpTotalRound +
                                  constInfo.leftDownTotalRound * (constInfo.mSwizzleBlockNum - 1 - globalS2Idx))) {
                loopIdx = loopIdx - constInfo.leftUpTotalRound;
                globalS1Idx = globalS2Idx + loopIdx / constInfo.leftDownTotalRound + 1;
                loopIdx = loopIdx % constInfo.leftDownTotalRound;
                isEnd =
                    GetLocalS1S2Idx<true>(loopIdx, M_SWIZZLE_BLOCK_SIZE, M_SWIZZLE_BLOCK_SIZE, localS1Idx, localS2Idx);
            } else {
                globalS1Idx = constInfo.mSwizzleBlockNum;
                loopIdx = loopIdx - (constInfo.leftUpTotalRound +
                                     constInfo.leftDownTotalRound * (constInfo.mSwizzleBlockNum - 1 - globalS2Idx));
                isEnd = GetLocalS1S2Idx<true>(loopIdx, constInfo.mSwizzleBlockNumTail, M_SWIZZLE_BLOCK_SIZE, localS1Idx,
                                              localS2Idx);
            }
        } else {
            globalS1Idx = constInfo.mSwizzleBlockNum;
            globalS2Idx = constInfo.mSwizzleBlockNum;
            loopIdx = loopIdx - constInfo.leftTotalRound;
            isEnd = GetLocalS1S2Idx<false>(loopIdx, constInfo.mSwizzleBlockNumTail, constInfo.mSwizzleBlockNumTail,
                                           localS1Idx, localS2Idx);
        }
        swizzleLoopIdx++;
        if (isEnd) {
            continue;
        }

        int64_t s1Idx = M_SWIZZLE_BLOCK_SIZE * globalS1Idx + localS1Idx;
        int64_t s2Idx = M_SWIZZLE_BLOCK_SIZE * globalS2Idx + localS2Idx;
        if (s1Idx < 0 || s1Idx >= constInfo.s1Outer || s2Idx < 0 || s2Idx >= constInfo.s2Outer) {
            continue;
        }
        runInfo.s2CvBegin = s2Idx * CUBE_BASEN;
        runInfo.s2CvEnd = runInfo.s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
        if (s2Idx == constInfo.s2Outer - 1) {             // 默认s2 cv tail相等
            runInfo.s2CvEnd = runInfo.s2CvBegin + constInfo.s2Tail;
        }
        runInfo.commonRunInfo.boIdx = bIdx / constInfo.commonConstInfo.n2G;
        int64_t bDimTail = bIdx % constInfo.commonConstInfo.n2G;
        runInfo.commonRunInfo.n2oIdx = bDimTail / constInfo.commonConstInfo.gSize;
        runInfo.commonRunInfo.goIdx = bDimTail % constInfo.commonConstInfo.gSize;
        runInfo.commonRunInfo.s1oIdx = s1Idx;
        runInfo.s2oIdx = s2Idx;
        return 0;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <const bool IS_DENSE>
__aicore__ inline bool FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetLocalS1S2Idx(
    int64_t loopIdx, int64_t m, int64_t n, int64_t &localS1Idx, int64_t &localS2Idx)
{
    int64_t r1 = loopIdx / constInfo.continuousBlockNum;
    int64_t r2 = loopIdx % constInfo.continuousBlockNum;
    int64_t globalIdx =
        constInfo.aicCoreNum * constInfo.continuousBlockNum * r1 + cBlockIdx * constInfo.continuousBlockNum + r2;
    if constexpr (IS_DENSE) {
        if (globalIdx >= m * n) {
            return true;
        }
        localS1Idx = globalIdx % m;
        localS2Idx = globalIdx / m;
    } else {
        if (globalIdx >= m * (NUM_TWO * m - n + 1) / NUM_TWO) {
            return true;
        }
        localS2Idx =
            Ceil((NUM_TWO * m - 1) - sqrt((NUM_TWO * m - 1) * (NUM_TWO * m - 1) + MULTIPLY_COEF * (m - 1 - globalIdx)),
                 NUM_TWO);
        localS1Idx = globalIdx - (NUM_TWO * m - localS2Idx - 1) * localS2Idx / NUM_TWO;
    }
    return false;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextValidIdxForSwizzle(
    FagRunInfo &runInfo, int64_t loopIdx)
{
    if (enableDenseSwizzle) {
        return GetNextValidIdxForSwizzleDense(runInfo, loopIdx);
    } else if (enableCasualSwizzle) {
        return GetNextValidIdxForSwizzleCasual(runInfo, loopIdx);
    } else {
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
                float sqrt_delta = 0.0;
                bool isRightDownCasualAndS1LessThanS2 =
                    constInfo.sparseMode == RIGHT_DOWN_CAUSAL &&
                    constInfo.commonConstInfo.s1Size < constInfo.commonConstInfo.s2Size;
                if (isRightDownCasualAndS1LessThanS2) {
                    int64_t rectangleNum = constInfo.s1Outer * (constInfo.s2Outer - constInfo.s1Outer + 1);
                    if (gDimTail < rectangleNum) {
                        s1Idx = gDimTail % constInfo.s1Outer;
                        s2Idx = gDimTail / constInfo.s1Outer;
                    } else {
                        gDimTail = gDimTail - rectangleNum;
                        sqrt_delta = sqrt(((constInfo.s1Outer << 1) - 1) * (((constInfo.s1Outer << 1) - 1)) +
                                          ((constInfo.s1Outer - 1 - gDimTail) << kShiftToMultiplyByEight));
                        s2Idx = Ceil<int64_t>(((constInfo.s1Outer << 1) - 1) - sqrt_delta, NUM_TWO);
                        s1Idx = gDimTail - ((((constInfo.s1Outer << 1) - 1 - s2Idx) * s2Idx) >> 1);
                        s2Idx = s2Idx + constInfo.s2Outer - constInfo.s1Outer + 1;
                    }
                } else {
                    sqrt_delta = sqrt(((constInfo.s1Outer << 1) - 1) * (((constInfo.s1Outer << 1) - 1)) +
                                      ((constInfo.s1Outer - 1 - gDimTail) << kShiftToMultiplyByEight));
                    s2Idx = Ceil<int64_t>(((constInfo.s1Outer << 1) - 1) - sqrt_delta, NUM_TWO);
                    s1Idx = gDimTail - ((((constInfo.s1Outer << 1) - 1 - s2Idx) * s2Idx) >> 1);
                }
            } else {
                int64_t tmp = 0;
                if (constInfo.firstValidColBlockNum + constInfo.firstValidRowBlockNum > constInfo.actualS1Outer) {
                    if (gDimTail < constInfo.s2FirstBlockNum) {
                        tmp = (constInfo.firstValidColBlockNum << 1) - 1;
                        s2Idx = (sqrt(tmp * tmp + (gDimTail << NUM_THREE)) - tmp) >> 1;
                        s1Idx = gDimTail - (((tmp + s2Idx) * s2Idx) >> 1);
                    } else if (constInfo.s2FirstBlockNum <= gDimTail &&
                               gDimTail < constInfo.s2FirstBlockNum + constInfo.s2SecondBlockNum) {
                        gDimTail = gDimTail - constInfo.s2FirstBlockNum;
                        s2Idx = gDimTail / constInfo.actualS1Outer + constInfo.s2FirstWidth;
                        s1Idx = gDimTail % constInfo.actualS1Outer;
                    } else {
                        gDimTail = gDimTail - constInfo.s2FirstBlockNum - constInfo.s2SecondBlockNum;
                        tmp = (constInfo.actualS1Outer << 1) - 1;
                        s2Idx = Ceil<int64_t>(
                            (tmp - sqrt(tmp * tmp + ((constInfo.actualS1Outer - 1 - gDimTail) << NUM_THREE))), NUM_TWO);
                        s1Idx = gDimTail - (((tmp - s2Idx) * s2Idx) >> 1);
                        s2Idx = s2Idx + constInfo.firstValidRowBlockNum - 1;
                    }
                } else {
                    if (gDimTail < constInfo.s2FirstBlockNum) {
                        tmp = (constInfo.firstValidColBlockNum << 1) - 1;
                        s2Idx = (sqrt(tmp * tmp + (gDimTail << NUM_THREE)) - tmp) >> 1;
                        s1Idx = gDimTail - (((tmp + s2Idx) * s2Idx) >> 1);
                    } else if (constInfo.s2FirstBlockNum <= gDimTail &&
                               gDimTail < constInfo.s2FirstBlockNum + constInfo.s2SecondBlockNum) {
                        gDimTail = gDimTail - constInfo.s2FirstBlockNum;
                        s2Idx = gDimTail / (constInfo.firstValidColBlockNum + constInfo.firstValidRowBlockNum - 1);
                        s1Idx =
                            gDimTail % (constInfo.firstValidColBlockNum + constInfo.firstValidRowBlockNum - 1) + s2Idx;
                        s2Idx = s2Idx + constInfo.s2FirstWidth;
                    } else {
                        gDimTail = gDimTail - constInfo.s2FirstBlockNum - constInfo.s2SecondBlockNum;
                        int64_t s1OuterTmp =
                            constInfo.firstValidColBlockNum + constInfo.firstValidRowBlockNum - NUM_TWO;
                        tmp = (s1OuterTmp << 1) - 1;
                        s2Idx = Ceil<int64_t>((tmp - sqrt(tmp * tmp + ((s1OuterTmp - 1 - gDimTail) << NUM_THREE))),
                                              NUM_TWO);
                        s1Idx = gDimTail - (((tmp - s2Idx) * s2Idx) >> 1) + constInfo.s2SecondWidth;
                        s2Idx = s2Idx + constInfo.s2FirstWidth + constInfo.s2SecondWidth;
                    }
                }
                s1Idx += constInfo.mGap;
                s2Idx += constInfo.nGap;
                if (s1Idx >= constInfo.s1Outer || s2Idx >= constInfo.s2Outer) {
                    return -1;
                }
            }
        }
        runInfo.s2CvBegin = s2Idx * CUBE_BASEN;
        runInfo.s2CvEnd = runInfo.s2CvBegin + CUBE_BASEN; // 非尾块s2按照+CUBE_BASEN处理
        if (s2Idx == constInfo.s2Outer - 1) {             // 默认s2 cv tail相等
            runInfo.s2CvEnd = runInfo.s2CvBegin + constInfo.s2Tail;
        }
        runInfo.commonRunInfo.boIdx = bIdx / constInfo.commonConstInfo.n2G;
        int64_t bDimTail = bIdx % constInfo.commonConstInfo.n2G;
        runInfo.commonRunInfo.n2oIdx = bDimTail / constInfo.commonConstInfo.gSize;
        runInfo.commonRunInfo.goIdx = bDimTail % constInfo.commonConstInfo.gSize;
        runInfo.commonRunInfo.s1oIdx = s1Idx;
        runInfo.s2oIdx = s2Idx;
        return bIdx * constInfo.s1Outer * constInfo.s2Outer + s2Idx * constInfo.s1Outer + s1Idx;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
template <bool IS_MM1_MM2>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryOffset(
    FagRunInfo &runInfo)
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
    int64_t bOffsetTmp = 0;
    if constexpr (IS_TND) {
        bOffsetTmp = runInfo.lastBatchTotalS1BOffset;
    }
    if constexpr (IS_ROPE) {
        if constexpr (IS_MM1_MM2) {
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
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetQueryRopeOffset(FagRunInfo &runInfo)
{
    int64_t leftMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS1BRopeOffset;
        s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDr;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDr;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1Dr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Dr;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GDr;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GDr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dr;
            s1Offset = runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDr;
            gOffset = runInfo.commonRunInfo.goIdx * constInfo.dRopeSize;
        }
    }
    leftMatrixOffset = bOffset + n2Offset + gOffset + s1Offset;
    return leftMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetDxOffset(
    FagRunInfo &runInfo)
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
template <bool IS_MM1_MM2>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyOffset(
    FagRunInfo &runInfo)
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
    int64_t bOffsetTmp = 0;
    if constexpr (IS_TND) {
        bOffsetTmp = runInfo.lastBatchTotalS2BOffset;
    }
    if constexpr (IS_ROPE) {
        if (IS_MM1_MM2) {
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
        if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
            runInfo.kGmS2SplitOffset = CUBE_BASEN / NUM_TWO * n2D;
            runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * n2D;
        }
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * s2D;
            s2Offset = runInfo.s2CvBegin * dSize;
            if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
                runInfo.kGmS2SplitOffset = CUBE_BASEN / NUM_TWO * dSize;
                runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * dSize;
            }
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * bN2D;
            bOffset = runInfo.commonRunInfo.boIdx * n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
            if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
                runInfo.kGmS2SplitOffset = CUBE_BASEN / NUM_TWO * bN2D;
                runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * bN2D;
            }
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * n2S2D;
            s2Offset = runInfo.s2CvBegin * n2D;
            n2Offset = runInfo.commonRunInfo.n2oIdx * dSize;
            if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
                runInfo.kGmS2SplitOffset = CUBE_BASEN / NUM_TWO * n2D;
                runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * n2D;
            }
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetKeyRopeOffset(
    FagRunInfo &runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS2BRopeOffset;
        s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dr;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Dr;
            s2Offset = runInfo.s2CvBegin * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.bN2Dr;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dr;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dr;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.dRopeSize;
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetValueOffset(
    FagRunInfo &runInfo)
{
    int64_t rightMatrixOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (IS_TND) {
        bOffset = runInfo.lastBatchTotalS2BOffsetForDv;
        s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
        if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
            runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * constInfo.commonConstInfo.n2Dv;
        }
    } else {
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Dv;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.dSizeV;
            if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
                runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * constInfo.commonConstInfo.dSizeV;
            }
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.bN2Dv;
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
            if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
                runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * constInfo.commonConstInfo.bN2Dv;
            }
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2S2Dv;
            s2Offset = runInfo.s2CvBegin * constInfo.commonConstInfo.n2Dv;
            n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.dSizeV;
            if constexpr (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) {
                runInfo.vGmS2SplitOffset = CUBE_BASEN / NUM_TWO * constInfo.commonConstInfo.n2Dv;
            }
        }
    }
    rightMatrixOffset = bOffset + n2Offset + s2Offset;
    return rightMatrixOffset;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetNextDxAndQueryOffset(
    FagRunInfo &runInfo, FagRunInfo &nextRunInfo, int64_t nextIndex, PreloadArgs<IS_ROPE> &preloadArgs)
{
    preloadArgs.copyNext = !(nextIndex == -1);
    preloadArgs.copyCurrent = (runInfo.commonRunInfo.taskId == 0);
    if (!preloadArgs.copyNext) {
        runInfo.isNextS2IdxNoChange = false;
        return;
    }
    int64_t nextBoIdx = 0;
    int64_t bDimTail = 0;
    int64_t nextN2oIdx = 0;
    int64_t n2DimTail = 0;
    int64_t nextGoIdx = 0;
    int64_t gDimTail = 0;
    int64_t nextS1oIdx = 0;
    int64_t nextS2oIdx = 0;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t bOffsetDv = 0;
    int64_t n2OffsetDv = 0;
    int64_t gOffsetDv = 0;
    int64_t s1OffsetDv = 0;

    if constexpr (IS_TND_SWIZZLE) {
        nextBoIdx = nextRunInfo.commonRunInfo.boIdx;
        nextN2oIdx = nextRunInfo.commonRunInfo.n2oIdx;
        nextGoIdx = nextRunInfo.commonRunInfo.goIdx;
        nextS1oIdx = nextRunInfo.commonRunInfo.s1oIdx;
        nextS2oIdx = nextRunInfo.s2oIdx;
        int64_t tndS1PrefixSum = (nextBoIdx == 0 ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[nextBoIdx - 1]);
        int64_t tndS2PrefixSum = (nextBoIdx == 0 ? 0 : ((__gm__ int64_t *)actualSeqKvlenAddr)[nextBoIdx - 1]);
        bOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2GD;
        s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
        n2Offset = nextN2oIdx * constInfo.commonConstInfo.gD;
        gOffset = nextGoIdx * constInfo.commonConstInfo.dSize;
        if constexpr (IS_D_NO_EQUAL) {
            bOffsetDv = tndS1PrefixSum * constInfo.commonConstInfo.n2GDv;
            s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
            n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
            gOffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
        }
        int64_t s1OuterTmp = (nextRunInfo.commonRunInfo.actualS1Size + CUBE_BASEM - 1) / CUBE_BASEM;
        int64_t s1CvTail = nextRunInfo.commonRunInfo.actualS1Size - (s1OuterTmp - 1) * CUBE_BASEM;
        preloadArgs.nextMOrN = (nextS1oIdx == s1OuterTmp - 1) ? s1CvTail : CUBE_BASEM;
    } else if constexpr (IS_TND) {
        int64_t lastBatchTotalS1BOffset = curBatchTotalS1BOffset;
        int64_t lastBatchTotalS1BOffsetForDv = curBatchTotalS1BOffsetForDv;
        int64_t lastBatchTotalBaseIdx = curBatchTotalBaseIdx;
        int64_t resbaseIdx = nextIndex - curBatchTotalBaseIdx;
        int64_t actualS1Len = 0;
        int64_t actualS2Len = 0;
        int64_t s1CvTail = 0;
        int64_t s1OuterTmp = 0;
        int64_t s2OuterTmp = 0;

        for (int64_t bIdx = curBatchIdx; bIdx < constInfo.bSize; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            s2OuterTmp = (actualS2Len + VECTOR_BASEN - 1) / VECTOR_BASEN;
            int64_t totalBaseIdx = constInfo.commonConstInfo.n2G * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                s1CvTail = actualS1Len - (s1OuterTmp - 1) * CUBE_BASEM;
                nextBoIdx = bIdx;
                bDimTail = resbaseIdx;
                nextN2oIdx = bDimTail / (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                n2DimTail = bDimTail % (constInfo.commonConstInfo.gSize * s1OuterTmp * s2OuterTmp);
                nextGoIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                nextS1oIdx = gDimTail % s1OuterTmp;
                nextS2oIdx = (resbaseIdx / s1OuterTmp) % s2OuterTmp;
                break;
            } else {
                lastBatchTotalBaseIdx += totalBaseIdx;
                resbaseIdx = nextIndex - lastBatchTotalBaseIdx;
                lastBatchTotalS1BOffset += actualS1Len * constInfo.commonConstInfo.n2GD;
                lastBatchTotalS1BOffsetForDv += actualS1Len * constInfo.commonConstInfo.n2GDv;
            }
        }
        bOffset = lastBatchTotalS1BOffset;
        s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
        n2Offset = nextN2oIdx * constInfo.commonConstInfo.gD;
        gOffset = nextGoIdx * constInfo.commonConstInfo.dSize;
        if constexpr (IS_D_NO_EQUAL) {
            bOffsetDv = lastBatchTotalS1BOffsetForDv;
            s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
            n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
            gOffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
        }
        preloadArgs.nextMOrN = (nextS1oIdx == s1OuterTmp - 1) ? s1CvTail : CUBE_BASEM;
    } else {
        if constexpr (IS_DETER_NEW(DETER_SPARSE_TYPE)) {
            nextBoIdx = nextIndex / constInfo.n2GS1oS2o;
            bDimTail = nextIndex % constInfo.n2GS1oS2o;
            nextN2oIdx = bDimTail / constInfo.gS1oS2o;
            n2DimTail = bDimTail % constInfo.gS1oS2o;
            nextGoIdx = n2DimTail / constInfo.s1oS2o;
            gDimTail = n2DimTail % constInfo.s1oS2o;
            nextS1oIdx = gDimTail % constInfo.s1Outer;
            nextS2oIdx = (nextIndex / constInfo.s1Outer) % constInfo.s2Outer;
        } else {
            nextBoIdx = nextRunInfo.commonRunInfo.boIdx;
            nextN2oIdx = nextRunInfo.commonRunInfo.n2oIdx;
            nextGoIdx = nextRunInfo.commonRunInfo.goIdx;
            nextS1oIdx = nextRunInfo.commonRunInfo.s1oIdx;
            nextS2oIdx = nextRunInfo.s2oIdx;
        }
        if (constInfo.commonConstInfo.layoutType == BNGSD) {
            if constexpr (IS_D_NO_EQUAL) {
                bOffsetDv = nextBoIdx * constInfo.commonConstInfo.n2GS1Dv;
                n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gS1Dv;
                gOffsetDv = nextGoIdx * constInfo.commonConstInfo.s1Dv;
                s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSizeV;
            }

            bOffset = nextBoIdx * constInfo.commonConstInfo.n2GS1D;
            n2Offset = nextN2oIdx * constInfo.commonConstInfo.gS1D;
            gOffset = nextGoIdx * constInfo.commonConstInfo.s1D;
            s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.dSize;
        } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
            if constexpr (IS_D_NO_EQUAL) {
                s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GDv;
                bOffsetDv = nextBoIdx * constInfo.commonConstInfo.n2GDv;
                n2OffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
                gOffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
            }

            s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.bN2GD;
            bOffset = nextBoIdx * constInfo.commonConstInfo.n2GD;
            n2Offset = nextGoIdx * constInfo.commonConstInfo.dSize;
            gOffset = nextN2oIdx * constInfo.commonConstInfo.gD;
        } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
            if constexpr (IS_D_NO_EQUAL) {
                bOffsetDv = nextBoIdx * constInfo.commonConstInfo.n2GS1Dv;
                s1OffsetDv = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GDv;
                n2OffsetDv = nextN2oIdx * constInfo.commonConstInfo.gDv;
                gOffsetDv = nextGoIdx * constInfo.commonConstInfo.dSizeV;
            }

            bOffset = nextBoIdx * constInfo.commonConstInfo.n2GS1D;
            s1Offset = nextS1oIdx * CUBE_BASEM * constInfo.commonConstInfo.n2GD;
            n2Offset = nextN2oIdx * constInfo.commonConstInfo.gD;
            gOffset = nextGoIdx * constInfo.commonConstInfo.dSize;
        }
        preloadArgs.nextMOrN = (nextS1oIdx == constInfo.s1Outer - 1) ? constInfo.s1CvTail : CUBE_BASEM;
    }
    preloadArgs.nextQueryOffset = bOffset + n2Offset + gOffset + s1Offset;
    if constexpr (IS_ROPE) {
        preloadArgs.nextQueryRopeOffset = preloadArgs.nextQueryOffset / 3;
        preloadArgs.nextQueryOffset = (preloadArgs.nextQueryOffset / 3) << 1;
    }
    if constexpr (IS_D_NO_EQUAL) {
        preloadArgs.nextDyOffset = bOffsetDv + n2OffsetDv + gOffsetDv + s1OffsetDv;
    } else {
        preloadArgs.nextDyOffset = preloadArgs.nextQueryOffset;
    }

    runInfo.isNextS2IdxNoChange = (nextS2oIdx == runInfo.s2oIdx && nextN2oIdx == runInfo.commonRunInfo.n2oIdx &&
                                   nextBoIdx == runInfo.commonRunInfo.boIdx);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::SyncALLCores()
{
    SyncAll<false>();
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
            isLastS1Outer[taskId & 1] = !isNextValid;
            isFirstS1Outer[taskId & 1] = (multiBlkInfo.s2oDimIdx == 0);
        } else {
            bool isNextValid = nextS2IdxLeft < multiBlkInfo.s2SparseRight && nextS2IdxRight > multiBlkInfo.s2SparseLeft;
            isLastS1Outer[taskId & 1] = !isNextValid;
            if (multiBlkInfo.s2oDimIdx > 0) {
                int64_t preS2IdxLeft = (multiBlkInfo.s2oDimIdx - 1) * CUBE_BASEN;
                int64_t preS2IdxRight = multiBlkInfo.s2oDimIdx * CUBE_BASEN;
                bool isPreValid =
                    preS2IdxLeft < multiBlkInfo.s2SparseRight && preS2IdxRight > multiBlkInfo.s2SparseLeft;
                isFirstS1Outer[taskId & 1] = !isPreValid;
            } else {
                isFirstS1Outer[taskId & 1] = (multiBlkInfo.s2oDimIdx == 0);
            }
        }
    } else {
        isLastS1Outer[taskId & 1] = (multiBlkInfo.s2oDimIdx == multiBlkInfo.s2OuterTmp - 1) ? true : false;
        isFirstS1Outer[taskId & 1] = (multiBlkInfo.s2oDimIdx == 0);
    }
    return;
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>::UpdateMNPQ(
    int64_t &m, int64_t &n, int64_t &p, int64_t &q, int64_t &mGap, int64_t &nGap)
{
    // 注意：与tiling中GetTotalPerBatchNum函数的有效块计算逻辑要保持一致
    m = constInfo.s1Outer;
    n = constInfo.s2Outer;
    p = Ceil<int64_t>(constInfo.s1Token, CUBE_BASEM) + 1;
    q = Ceil<int64_t>(constInfo.s2Token, CUBE_BASEN) + 1;
    p = p > constInfo.s1Outer ? constInfo.s1Outer : p;
    q = q > constInfo.s2Outer ? constInfo.s2Outer : q;

    int64_t actualM, actualN, actualP, actualQ;
    if (p < 0) {
        actualM = m;
        actualN = n + p - 1;
        actualP = 1;
        actualQ = p + q - 1;
        mGap = 0;
        nGap = -p + 1;
    } else if (q < 0) {
        actualM = m + q - 1;
        actualN = n;
        actualP = p + q - 1;
        actualQ = 1;
        mGap = -q + 1;
        nGap = 0;
    } else {
        actualM = m;
        actualN = n;
        actualP = p;
        actualQ = q;
        mGap = 0;
        nGap = 0;
    }
    m = actualM;
    n = actualN;
    p = actualP;
    q = actualQ;
}

} // namespace FagBaseApi
#endif
