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
 * \file quant_block_sparse_attn_kernel.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_KERNEL_H_
#define QUANT_BLOCK_SPARSE_ATTN_KERNEL_H_
#include "quant_block_sparse_attn_block_cube.h"
#include "quant_block_sparse_attn_block_vec.h"
#include "quant_block_sparse_attn_common_arch35.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_block_sparse_attn_attenmask.h"

#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"

#include "quant_block_sparse_attn_kvcache.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/utils/init_global_memory.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
namespace QBSAKernelBase {
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type =
        std::conditional_t<(useDn && isFp8), BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                           BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};
} // namespace QBSAKernelBase

template <typename CubeBlockType, typename VecBlockType>
class QuantBlockSparseAttnKernel {
public:
    ARGS_TRAITS;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    __aicore__ inline QuantBlockSparseAttnKernel(){};

    __aicore__ inline void InitBaseAPI(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                       __gm__ uint8_t *attenMask, __gm__ uint8_t *metadata, __gm__ uint8_t *cuSeqQlen,
                                       __gm__ uint8_t *cuSeqKvlen, __gm__ uint8_t *seqUsedQlen,
                                       __gm__ uint8_t *seqUsedKvlen, __gm__ uint8_t *blockTable,
                                       __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                       __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                                       __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                       __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                       __gm__ uint8_t *workspace,
                                       const QuantBlockSparseAttnTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *metadata, __gm__ uint8_t *cuSeqQlen, __gm__ uint8_t *cuSeqKvlen,
        __gm__ uint8_t *seqUsedQlen, __gm__ uint8_t *seqUsedKvlen, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
        __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
        __gm__ uint8_t *workspace, const QuantBlockSparseAttnTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void AnalysisSparseIndices();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx, uint64_t sparseBase);
    __aicore__ inline void SetSparseIndicesRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t s2LoopCount,
                                                   int64_t s2LoopLimit, uint64_t sparseBase);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);
    __aicore__ inline void ProcessMainLoop();

    TPipe *pipe;

    const QuantBlockSparseAttnTilingData *__restrict tilingData;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr bool isFp8 = CubeBlockType::isFp8;
    static constexpr bool isMxfp8FullQuant = s1BaseSize == 128 && s2BaseSize == 512;
    static constexpr TPosition bmm2OutPos = CubeBlockType::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = CubeBlockType::bmm2Write2Ub;
    static constexpr bool splitD = CubeBlockType::splitD;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    BufferManager<BufferType::GM> gmBufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    using bmm2ResBufferType = typename QBSAKernelBase::Bmm2ResBuffSel<useDn, isFp8>::Type;
    bmm2ResBufferType bmm2Buffers;

    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    CVSharedParams<isPa> sharedParams;
    using keyGmType = GlobalTensor<INPUT_T>;
    keyGmType keyGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    GlobalTensor<int32_t> metadataGm;
    GlobalTensor<int32_t> blockTableGm;
    __gm__ int32_t *cuSeqQlenAddr;
    __gm__ int32_t *seqUsedKvlenAddr;
    int32_t aicIdx;

    ConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;

    int64_t actualKVPrefixLen = 0;

    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::InitBaseAPI(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseSeqLen, __gm__ uint8_t *attenMask, __gm__ uint8_t *metadata, __gm__ uint8_t *cuSeqQlen,
    __gm__ uint8_t *cuSeqKvlen, __gm__ uint8_t *seqUsedQlen, __gm__ uint8_t *seqUsedKvlen, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ,
    __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const QuantBlockSparseAttnTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    // tilingData 必须在 AIC/AIV 两侧都设置：InitVecBlock 与 ComputeConstexpr 在两核均会解引用
    // this->tilingData，若仅 AIV 赋值，AIC 将解引用未初始化指针导致访存异常（507015）。
    this->tilingData = tiling;
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
    }
    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx,
                          attenMaskInfo);
    vecBlock.CleanOutput(softmaxLse, attentionOut, constInfo);
    InitMMResBuf();
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value, blockTable);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t *>(0);
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams<isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }

    this->ComputeConstexpr();
    this->InitGlobalBuffer(key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqQlen, cuSeqKvlen,
                           seqUsedQlen, seqUsedKvlen, deqScaleQ, deqScaleK, deqScaleV, pScale, blockTable,
                           queryPaddingSize, kvPaddingSize, softmaxMax, softmaxSum, workspace, tiling, tPipe);
    this->InitLocalBuffer();
    this->AnalysisSparseIndices();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *metadata, __gm__ uint8_t *cuSeqQlen, __gm__ uint8_t *,
    __gm__ uint8_t *, __gm__ uint8_t *seqUsedKvlen, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
    __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
    const QuantBlockSparseAttnTilingData *__restrict tiling, TPipe *tPipe)
{
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)(key));
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t *)sparseSeqLen);
    metadataGm.SetGlobalBuffer((__gm__ int32_t *)metadata);
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    if constexpr (layout == QBSALayout::TND || layout == QBSALayout::NTD) {
        cuSeqQlenAddr = (__gm__ int32_t *)cuSeqQlen;
        seqUsedKvlenAddr = (__gm__ int32_t *)seqUsedKvlen;
    }

    uint64_t singleCoreOffset = 0;
    if constexpr (!bmm2Write2Ub) {
        int64_t bmm2ResBlock = this->sharedParams.dSizeV;
        if constexpr (splitD) {
            bmm2ResBlock = (int64_t)dVTemplateType;
        }
        int64_t mm2ResultSize = (s1BaseSize)*bmm2ResBlock;
        int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * sizeof(T);
        int64_t vec2ResultSize = (s1BaseSize)*constInfo.dBasicBlock;
        int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
        singleCoreOffset = mm2Offset;
        if constexpr (splitD) {
            singleCoreOffset = mm2Offset + vec2Offset;
        }
        int64_t totalOffset = this->aicIdx * 3 * singleCoreOffset;
        gmBufferManager.Init(workspace + totalOffset);
        bmm2ResGmBuffers.Init(gmBufferManager, mm2Offset);
        workspace += (totalOffset + mm2Offset * 3);
    }
    __gm__ uint8_t *prefix = nullptr;
    __gm__ uint8_t *pScalePtr = (this->tilingData->inputParamsRegbase.pScaleShapeSize == 0U) ? nullptr : pScale;
    vecBlock.InitGlobalBuffer(deqScaleQ, deqScaleK, deqScaleV, pScalePtr, prefix, attenMask,
                              blockTable, queryPaddingSize, kvPaddingSize, softmaxMax, softmaxSum, workspace,
                              singleCoreOffset, this->aicIdx, constInfo);
    // NTD 的 query 走 offsetCalculator(NGTD)，其 ACCUM(no-zero-head) parser 期望"不含前导0的累加序列"
    // [s0, s0+s1, ...]：GetTSize() 取末元素=总 token 数(决定 head/n2 的 stride=d*T、d*T*g)，
    // GetTBase(b)=arr[b-1]=batch b 的 token 起始。cuSeqlens 含前导0 [0,s0,s0+s1,...]，故传 cuSeqQlenAddr+1
    // 去掉前导0。否则 B>1 时 GetTSize 取到单 batch 长度而非总 T，head>=1 的 query 读取整体错位，精度大面积错。
    // (KV 为 PA，靠 blockTable 寻址，不依赖该 parser，保持 seqUsedKvlenAddr。)
    cubeBlock.InitCubeInput(key, value, &sharedParams, &attenMaskInfo, cuSeqQlenAddr + 1, seqUsedKvlenAddr);
    cubeBlock.InitDequantParams(deqScaleQ, deqScaleK, deqScaleV);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    uint32_t mm1OutDtype = sizeof(T);
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
    if constexpr (isMxfp8FullQuant) {
        mm1ResultSize /= 2;
    }
    constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(INPUT_T);
    l1BufferManager.Init(pipe, 524288);
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    if constexpr (bmm2Write2Ub) {
        if constexpr (useDn && isFp8) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
        }
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        if ASCEND_IS_AIV {
            bmm2Buffers.Get().SetCrossCore();
            if constexpr (!(useDn && isFp8)) {
                bmm2Buffers.Get().SetCrossCore();
            }
        }
    } else {
        ubBufferManager.Init(pipe, mm1ResultSize * 2);
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::AnalysisSparseIndices()
{
    constInfo.maxQb = this->tilingData->sparseParams.sparseSeqLenStride;
    constInfo.maxKb = this->tilingData->sparseParams.sparseIndicesStride;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;

    constInfo.bSize = sharedParams.bSize;
    constInfo.t1Size = sharedParams.t1Size;
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeV = sharedParams.dSizeV;
    constInfo.combineDim = (constInfo.dSize + constInfo.dSizeV + sizeof(float) + 31U) / 32U * 32U;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    constInfo.dSizeRope = 0;
    constInfo.gSize = sharedParams.gSize;
    constInfo.n1Size = constInfo.n2Size * constInfo.gSize;
    constInfo.s1OuterSize = sharedParams.s1OuterSize;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
    constInfo.layoutType = sharedParams.layoutType;

    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.maxBlockNumPerBatch = sharedParams.blockTableDim2;

    if constexpr (layout == QBSALayout::TND) {
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if ASCEND_IS_AIV {
            // QBSA: 输出 TND [T, N1, Dv]，同一 head 相邻 token 间隔 (n2G-1)*Dv 个元素（跳过其余 head）。
            // 每核独立计算一个 head，不做 FIA GS1 合轴/连续 GQA 输出，固定使用 per-head stride。
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (layout == QBSALayout::NTD) {
        constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
        constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }

    if ASCEND_IS_AIV {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = sharedParams.preTokens;
            attenMaskInfo.nextTokens = sharedParams.nextTokens;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::SetSparseIndicesRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t s2LoopCount, int64_t s2LoopLimit, uint64_t sparseBase)
{
    uint64_t sparseLoopCountOffset = sparseBase + s2LoopCount * 2;
    int32_t remainingSparseBlocks = runParam.actSparseLen - static_cast<int32_t>(s2LoopCount * 2);
    runInfo.sparseBlkIdx1 = remainingSparseBlocks > 0 ? sparseIndicesGm.GetValue(sparseLoopCountOffset) : -1;
    runInfo.sparseBlkIdx2 = remainingSparseBlocks > 1 ? sparseIndicesGm.GetValue(sparseLoopCountOffset + 1) : -1;

    if (unlikely(runInfo.sparseBlkIdx1 < 0)) {
        runInfo.s2SparseBlk1RealSize = 0;
        runInfo.s2SparseBlk1RealAlignedSize = 0;
        runInfo.sparseBlk1PartialMask = false;
    } else if (unlikely((runInfo.sparseBlkIdx1 + 1) * constInfo.kvSparseBlockSize > runParam.actualS2Size)) {
        runInfo.s2SparseBlk1RealSize = runParam.actualS2Size - runInfo.sparseBlkIdx1 * constInfo.kvSparseBlockSize;
        runInfo.s2SparseBlk1RealAlignedSize = Align(runInfo.s2SparseBlk1RealSize);
    } else {
        runInfo.s2SparseBlk1RealSize = constInfo.kvSparseBlockSize;
        runInfo.s2SparseBlk1RealAlignedSize = runInfo.s2SparseBlk1RealSize;
    }
    if (unlikely(runInfo.sparseBlkIdx2 < 0)) {
        runInfo.s2SparseBlk2RealSize = 0;
        runInfo.s2SparseBlk2RealAlignedSize = 0;
        runInfo.sparseBlk2PartialMask = false;
    } else if (unlikely((runInfo.sparseBlkIdx2 + 1) * constInfo.kvSparseBlockSize > runParam.actualS2Size)) {
        runInfo.s2SparseBlk2RealSize = runParam.actualS2Size - runInfo.sparseBlkIdx2 * constInfo.kvSparseBlockSize;
        runInfo.s2SparseBlk2RealAlignedSize = Align(runInfo.s2SparseBlk2RealSize);
    } else {
        runInfo.s2SparseBlk2RealSize = constInfo.kvSparseBlockSize;
        runInfo.s2SparseBlk2RealAlignedSize = runInfo.s2SparseBlk2RealSize;
    }

    // 将sparseBlkIdx映射为phyBlkNumIdx  并按照phyBlkNumIdx的大小顺序排列
    IdxSortBySparseIdx(runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx, uint64_t sparseBase)
{
    runInfo.s1oIdx = runParam.s1oIdx;
    runInfo.boIdx = runParam.boIdx;
    runInfo.n2oIdx = runParam.n2oIdx;
    runInfo.goIdx = runParam.goIdx;
    runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    runInfo.multiCoreIdxMod2 = static_cast<uint64_t>(multiCoreInnerIdx) & 1;
    runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;

    // sparseIndices 相关计算
    SetSparseIndicesRunInfo(runInfo, runParam, s2LoopCount, s2LoopLimit, sparseBase);
    runInfo.s2StartIdx = 0; // sparsemode仅支持0/3  且  逻辑分核不会对S2进行分核
    runInfo.s2LoopCount = s2LoopCount;
    runInfo.qBOffset = runParam.qBOffset;
    runInfo.qBScalarOffset = runParam.qBScalarOffset;
    runInfo.actSparseLen = runParam.actSparseLen;

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = static_cast<uint64_t>(taskId) & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo &runInfo,
                                                                                                RunParamStr &runParam)
{
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;

    runInfo.s2RealSize = runInfo.s2SparseBlk1RealSize + runInfo.s2SparseBlk2RealSize;
    runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.seqUsedQlenSize = this->sharedParams.seqUsedQlenSize;
    this->constInfo.seqUsedKvlenSize = this->sharedParams.seqUsedKvlenSize;
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.kvSparseBlockSize = this->sharedParams.kvSparseBlockSize;
        this->constInfo.qSparseBlockSize = this->sharedParams.qSparseBlockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
        this->constInfo.paBlockStride = this->sharedParams.paBlockStride;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS>(runParam, runInfo);
    runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if (this->aicIdx >= actualCoreNums || static_cast<uint32_t>(this->aicIdx) >= QBSA_FA_AIC_CORE_NUM) {
        return;
    }

    uint32_t sectionNum = GetBsaSectionNum(metadataGm);
    uint32_t lastValidSectionIdx =
        GetBsaLastValidSectionIdx(metadataGm, sectionNum, static_cast<uint32_t>(this->aicIdx));
    if (lastValidSectionIdx == sectionNum) {
        return;
    }

    int64_t s2LoopLimit;
    int64_t taskId = 0;
    bool notLast = true;
    RunInfo runInfo[4];
    RunParamStr runParam;

    int64_t multiCoreInnerIdx = 1;
    uint64_t sparseBase = 0;
    for (uint32_t sectionIdx = 0U; sectionIdx < sectionNum; ++sectionIdx) {
        BsaFaCoreMetadata coreMetadata =
            GetBsaCoreMetadata(metadataGm, sectionIdx, static_cast<uint32_t>(this->aicIdx));
        if (coreMetadata.coreEnable == 0U) {
            continue;
        }
        uint32_t bn1StartIdx = coreMetadata.bn1StartIdx;
        uint32_t bn1EndIdx = coreMetadata.bn1EndIdx;
        uint32_t s1StartIdx = coreMetadata.s1StartIdx;
        uint32_t s1EndIdx = coreMetadata.s1EndIdx;
        if (s1EndIdx != 0U) {
            bn1EndIdx++;
        }
        if (bn1EndIdx <= bn1StartIdx) {
            continue;
        }

        for (uint32_t bn1Idx = bn1StartIdx; bn1Idx < bn1EndIdx; ++bn1Idx) {
            bool sectionLastBN = (bn1Idx == bn1EndIdx - 1);
            bool coreLastBN = sectionLastBN && (sectionIdx == lastValidSectionIdx);
            runParam.boIdx = bn1Idx / this->constInfo.n1Size;
            runParam.n1oIdx = bn1Idx % this->constInfo.n1Size;
            runParam.n2oIdx = runParam.n1oIdx / constInfo.gSize;
            runParam.goIdx = runParam.n1oIdx % constInfo.gSize;
            ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS>(runParam, this->constInfo, this->attenMaskInfo, this->keyGm,
                                                        this->cuSeqQlenAddr, this->seqUsedKvlenAddr);
            ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS>(runParam, constInfo, s1StartIdx, s1EndIdx,
                                                        bn1Idx == bn1StartIdx, sectionLastBN);
            int64_t temps1End = coreLastBN ? (runParam.s1LoopEnd + 3) : runParam.s1LoopEnd;
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            for (int64_t s1Index = runParam.s1LoopStart; s1Index < temps1End; ++s1Index) {
                if (coreLastBN) {
                    UpdateLoopFlagsBasedOnS1Index(s1Index - runParam.s1LoopEnd, notLast, notLastTwoLoop,
                                                  notLastThreeLoop);
                }
                if (notLastThreeLoop) {
                    runParam.s1oIdx = s1Index;
                    ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS>(runParam, this->constInfo, s1Index, this->cuSeqQlenAddr);
                    bool s2NoNeedCalc =
                        ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS>(runParam, constInfo, sparseSeqLenGm, s1Index);
                    if (s2NoNeedCalc) {
                        continue;
                    }
                    sparseBase = (static_cast<uint64_t>(runParam.boIdx) * constInfo.n1Size + runParam.n1oIdx) *
                                     constInfo.maxQb * constInfo.maxKb +
                                 static_cast<uint64_t>(s1Index) * constInfo.maxKb;
                }
                s2LoopLimit = notLastThreeLoop ? runParam.s2LoopEndIdx - 1 : 0;
                for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                    if (notLastThreeLoop) {
                        RunInfo &runInfo1 = runInfo[static_cast<uint64_t>(taskId) & 3];
                        this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx,
                                         sparseBase);
                        if ASCEND_IS_AIC {
                            this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                        }
                        if ASCEND_IS_AIV {
                            int64_t delta = runInfo1.nextTokensPerBatch;
                            uint32_t sparseBlk1S1Offset = runInfo1.s1oIdx * constInfo.s1BaseSize;
                            uint32_t sparseBlk2S1Offset = sparseBlk1S1Offset;
                            if (likely(runInfo1.sparseBlkIdx1 >= 0)) {
                                uint32_t sparseBlk1S2Offset = runInfo1.sparseBlkIdx1 * constInfo.kvSparseBlockSize;
                                AttentionMaskFullProcessingOrRequired(
                                    sparseBlk1S1Offset, sparseBlk1S2Offset, delta, runInfo1.s2SparseBlk1RealSize,
                                    runInfo1.sparseBlkIdx1, runInfo1.sparseBlk1PartialMask);
                            }
                            if (likely(runInfo1.sparseBlkIdx2 >= 0)) {
                                uint32_t sparseBlk2S2Offset = runInfo1.sparseBlkIdx2 * constInfo.kvSparseBlockSize;
                                AttentionMaskFullProcessingOrRequired(
                                    sparseBlk2S1Offset, sparseBlk2S2Offset, delta, runInfo1.s2SparseBlk2RealSize,
                                    runInfo1.sparseBlkIdx2, runInfo1.sparseBlk2PartialMask);
                            }
                        }
                    }
                    if (likely(taskId > 0 && notLastTwoLoop)) {
                        if ASCEND_IS_AIV {
                            auto &runInfo3 = runInfo[static_cast<uint64_t>(taskId + 3) & 3];
                            this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                                                       this->constInfo);
                        }
                    }
                    if (likely(taskId > 1 && notLast)) {
                        if ASCEND_IS_AIC {
                            RunInfo &runInfo2 = runInfo[static_cast<uint64_t>(taskId + 2) & 3];
                            if constexpr (bmm2Write2Ub) {
                                this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                                            this->constInfo);
                            } else {
                                this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                                            this->constInfo);
                            }
                        }
                    }
                    if (likely(taskId > 2)) {
                        if ASCEND_IS_AIV {
                            RunInfo &runInfo3 = runInfo[static_cast<uint64_t>(taskId + 1) & 3];
                            if constexpr (bmm2Write2Ub) {
                                this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                            } else {
                                this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                            }
                        }
                    }
                    ++taskId;
                }
                ++multiCoreInnerIdx;
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>::Process()
{
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
}
} // namespace BaseApi
#endif
