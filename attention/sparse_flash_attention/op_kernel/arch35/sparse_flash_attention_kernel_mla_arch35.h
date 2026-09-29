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
 * \file sparse_flash_attention_kernel_mla_arch35.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_KERNEL_MLA_ARCH35_H
#define SPARSE_FLASH_ATTENTION_KERNEL_MLA_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_service_cube_mla_arch35.h"
#include "sparse_flash_attention_service_vector_mla_arch35.h"
#include "sparse_flash_attention_common_arch35.h"
#include "sparse_flash_attention_kvcache.h"

#include "../../common/op_kernel/CopyInL1.h"
#include "../../common/op_kernel/matmul.h"
#include "../../common/op_kernel/FixpipeOut.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class SparseFlashAttentionKernelMla {
public:
    ARGS_TRAITS;

    __aicore__ inline SparseFlashAttentionKernelMla(){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* actualSeqLengthsQ,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* blockTable, __gm__ uint8_t* queryRope,
                                __gm__ uint8_t* keyRope, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxMax,
                                __gm__ uint8_t* softmaxSum, __gm__ uint8_t* sinks, __gm__ uint8_t* workspace,
                                const SparseFlashAttentionTilingDataMla* __restrict tiling, __gm__ uint8_t* gmTiling,
                                TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void FreeEvent();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                            __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
                                            __gm__ uint8_t* sparseIndices, __gm__ uint8_t* blockTable,
                                            __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
                                            __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
                                            __gm__ uint8_t* sinks, __gm__ uint8_t* workspace,
                                            const SparseFlashAttentionTilingDataMla* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t* workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo& sfaRunInfo, RunParamStr& sfaRunParam, int64_t taskId,
                                      int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo& sfaRunInfo, RunParamStr& sfaRunParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr& sfaRunParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr& sfaRunParam, RunInfo& sfaRunInfo);

    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t>& actualSeqLengths, uint32_t bIdx);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev, uint32_t s2EndPrev);

    TPipe* sfaPipe;

    const SparseFlashAttentionTilingDataMla* __restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1RightBuffers;
    CVSharedParams sharedParams;
    /* GM信息 */
    __gm__ int32_t* actualSeqKvlenAddr = nullptr;
    __gm__ int32_t* actualSeqQlenAddr = nullptr;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    uint32_t usedCoreNum = 0U;

    GlobalTensor<int32_t> oriTopkLengthGm;
    bool hasOriTopkLength = false;
    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 切G时最大s2Loop */
    int64_t maxS2LoopCnt;

    /* 初始化后不变的信息 */
    ConstInfo sfaConstInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    uint32_t crossCoreSyncBufId = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* sparseIndices,
    __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxMax,
    __gm__ uint8_t* softmaxSum, __gm__ uint8_t* sinks, __gm__ uint8_t* workspace,
    const SparseFlashAttentionTilingDataMla* __restrict tiling, __gm__ uint8_t* gmTiling, TPipe* tPipe)
{
    fa_base_matmul::ResetIdCounter();
    sfaConstInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        sfaConstInfo.aicIdx = this->aicIdx;
        sfaConstInfo.aivIdx = 0;
    } else {
        sfaConstInfo.aivIdx = GetBlockIdx();
        this->aicIdx = sfaConstInfo.aivIdx >> 1;
        sfaConstInfo.aicIdx = this->aicIdx;
        this->tilingData = tiling;
    }

    sfaConstInfo.s1BaseSize = 64;
    sfaConstInfo.s2BaseSize = 128;

    this->sfaPipe = tPipe;
    int64_t dSizeRope = 0;
    if constexpr (HAS_ROPE) {
        dSizeRope = 64; // 64: 编码维度
    }
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, sfaConstInfo.subBlockIdx,
                          actualSeqLengthsQ, actualSeqLengths, dSizeRope);
    if ASCEND_IS_AIV {
        sfaConstInfo.bSize = this->sharedParams.bSize;
        sfaConstInfo.n2Size = this->sharedParams.n2Size;
        sfaConstInfo.gSize = this->sharedParams.gSize;
        sfaConstInfo.s1Size = this->sharedParams.s1Size;
        sfaConstInfo.dSizeV = 512;
        sfaConstInfo.needInit = this->sharedParams.needInit;
        sfaConstInfo.returnSoftmaxLse = this->sharedParams.returnSoftmaxLse;
    }
    vecBlock.CleanOutput(attentionOut, softmaxMax, softmaxSum, sfaConstInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(sfaPipe, l1BufferManager, query, queryRope);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto sfaTempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t*>(&sharedParams);
#pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++sfaTempTilingSSbuf, ++tempTiling) {
            *tempTiling = *sfaTempTilingSSbuf;
        }
    }
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, queryRope, keyRope, sparseIndices, blockTable, actualSeqLengthsQ,
                           actualSeqLengths, softmaxMax, softmaxSum, sinks, workspace, tiling, tPipe); // gm设置
    this->InitCalcParamsEach();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitCalcParamsEach()
{
    // 计算总的基本块
    maxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t sfaTotalBaseNum = 0;
    uint32_t s1GBaseSize = sfaConstInfo.gSize;
    uint32_t actBatchS2 = 1;
    uint32_t coreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，coreNum减半
    uint32_t sfaCurrCoreIdx = aicIdx;
    if constexpr (IS_SPLIT_G) {
        coreNum = coreNum >> 1;
        sfaCurrCoreIdx = sfaCurrCoreIdx >> 1;
    }
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < sfaConstInfo.bSize; bIdx++) {
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx); // 不切S2，只关注S1
        if (actBatchS1 < sfaConstInfo.s1Size) {
            sfaConstInfo.needInit = true;
        }
        sfaTotalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t sfaAvgBaseNum = 1;
    if (sfaTotalBaseNum > coreNum) {
        sfaAvgBaseNum = (sfaTotalBaseNum + coreNum - 1) / coreNum;
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = (sfaTotalBaseNum + sfaAvgBaseNum - 1) / sfaAvgBaseNum << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = sfaTotalBaseNum << 1;
        } else {
            usedCoreNum = sfaTotalBaseNum;
        }
    }

    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = sfaAvgBaseNum *
                       (Min(sfaConstInfo.sparseBlockCount, sfaConstInfo.s2Size) + sfaConstInfo.s2BaseSize - 1) /
                       sfaConstInfo.s2BaseSize;
    }

    if (aicIdx >= usedCoreNum) {
        return;
    }
    // 计算当前核的基本块
    uint32_t sfaAccumBaseNum = 0; // sfa 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t sfaLastValidBIdx = 0;
    uint32_t lastValidactBatchS1 = 0;
    bool setStart = false;
    targetBaseNum = (sfaCurrCoreIdx + 1) * sfaAvgBaseNum; // 计算当前的目标权重
    uint32_t targetStartBaseNum = targetBaseNum - sfaAvgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < sfaConstInfo.bSize * sfaConstInfo.n2Size; bN2Idx++) {
        uint32_t bIdx = bN2Idx / sfaConstInfo.n2Size;
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            sfaAccumBaseNum += 1;
            if (!setStart && sfaAccumBaseNum >= targetStartBaseNum) {
                sfaConstInfo.bN2Start = bN2Idx;
                sfaConstInfo.gS1Start = s1GIdx;
                setStart = true;
            }
            if (sfaAccumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                sfaConstInfo.bN2End = bN2Idx;
                sfaConstInfo.gS1End = s1GIdx;
                sfaConstInfo.s2End = 0;
                if (sfaCurrCoreIdx != 0) {
                    GetAxisStartIdx(sfaConstInfo.bN2Start, sfaConstInfo.gS1Start, 0);
                }
                return;
            }
        }
        if ((actBatchS1 > 0) && (actBatchS2 > 0)) {
            sfaLastValidBIdx = bIdx;
            lastValidactBatchS1 = actBatchS1;
        }
    }
    if (!setStart) {
        sfaConstInfo.bN2Start = sfaLastValidBIdx;
        sfaConstInfo.gS1Start = lastValidactBatchS1 - 1;
    }
    if (sfaAccumBaseNum < targetBaseNum) {
        // 更新最后一个核的End分核信息
        sfaConstInfo.bN2End = sfaLastValidBIdx;
        sfaConstInfo.gS1End = lastValidactBatchS1 - 1;
        sfaConstInfo.s2End = 0;
        if (sfaCurrCoreIdx != 0) {
            GetAxisStartIdx(sfaConstInfo.bN2Start, sfaConstInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline uint64_t SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::GetBalanceActualSeqLengths(
    GlobalTensor<int32_t>& actualSeqLengths, uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
        } else if (bIdx == 0) {
            return actualSeqQlenAddr[0];
        } else {
            return 0;
        }
    } else {
        if (sfaConstInfo.isActualLenDimsNull == 1) {
            return sfaConstInfo.s1Size;
        } else {
            return actualSeqQlenAddr[bIdx];
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                                   uint32_t s1GEndPrev,
                                                                                                   uint32_t s2EndPrev)
{
    uint32_t sfaBEndPrev = bN2EndPrev / sfaConstInfo.n2Size;
    uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, sfaBEndPrev);
    uint32_t s1GPrevBaseNum = actualSeqQPrev;
    sfaConstInfo.bN2Start = bN2EndPrev;
    sfaConstInfo.gS1Start = s1GEndPrev;

    sfaConstInfo.s2Start = 0;
    if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        sfaConstInfo.gS1Start = 0;
        sfaConstInfo.bN2Start++;
    } else {
        sfaConstInfo.gS1Start++;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* queryRope,
    __gm__ uint8_t* keyRope, __gm__ uint8_t* sparseIndices, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* softmaxMax,
    __gm__ uint8_t* softmaxSum, __gm__ uint8_t* sinks, __gm__ uint8_t* workspace,
    const SparseFlashAttentionTilingDataMla* __restrict tiling, TPipe* tPipe)
{
    if (actualSeqLengthsQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t*)actualSeqLengthsQ;
    }
    if (actualSeqLengths != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t*)actualSeqLengths;
    }

    vecBlock.InitGlobalBuffer(key, value, keyRope, sparseIndices, blockTable, softmaxMax, softmaxSum, sinks);
    cubeBlock.InitCubeInput(key, keyRope, sparseIndices, blockTable, actualSeqLengthsQ, sfaConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitMMResBuf(
    __gm__ uint8_t* workspace)
{
    uint32_t sfaMm1ResultSize = sfaConstInfo.s1BaseSize / CV_RATIO * sfaConstInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = sfaConstInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = sfaConstInfo.s1BaseSize * sfaConstInfo.s2BaseSize * sizeof(Q_T);
    // P 复用 rope 槽位，L1 right 两侧都按 576 分配；none 实例 KV 仍按 512 pitch 搬入
    uint32_t mm1RightSize = sfaConstInfo.s2BaseSize * 576 * sizeof(Q_T);
    l1BufferManager.Init(sfaPipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1RightBuffers.Init(l1BufferManager, mm1RightSize);
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    ubBufferManager.Init(sfaPipe, sfaMm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    bmm2Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, sfaMm1ResultSize);
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }

    constexpr uint32_t kvTokenWidth = HAS_ROPE ? 576U : 512U;
    uint32_t v0ResSize = sfaConstInfo.s2BaseSize * kvTokenWidth * sizeof(Q_T);
    int64_t totalOffset;
    if constexpr (IS_SPLIT_G) {
        totalOffset = v0ResSize * 3 * (aicIdx >> 1U);
    } else {
        totalOffset = v0ResSize * 3 * aicIdx;
    }
    gmBufferManager.Init(workspace + totalOffset);
    v0ResGmBuffers.Init(gmBufferManager, v0ResSize);
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(sfaPipe, sfaConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    usedCoreNum = sharedParams.usedCoreNum;

    if ASCEND_IS_AIC {
        sfaConstInfo.bSize = this->sharedParams.bSize;
        sfaConstInfo.gSize = this->sharedParams.gSize;
        sfaConstInfo.s1Size = this->sharedParams.s1Size;
        sfaConstInfo.dSizeV = 512;
        sfaConstInfo.needInit = this->sharedParams.needInit;
    }
    sfaConstInfo.n2Size = sharedParams.n2Size;
    sfaConstInfo.s2Size = sharedParams.s2Size;
    sfaConstInfo.dSize = sharedParams.dSize;
    sfaConstInfo.dSizeVInput = sharedParams.dSizeVInput;
    if constexpr (HAS_ROPE) {
        sfaConstInfo.dSizeRope = 64;
    } else {
        sfaConstInfo.dSizeRope = 0;
    }
    sfaConstInfo.dSizeNope = 512;
    sfaConstInfo.tileSize = sharedParams.tileSize;
    sfaConstInfo.sparseBlockCount = sharedParams.sparseBlockCount;
    sfaConstInfo.sparseBlockSize = 1;
    sfaConstInfo.sparseMode = sharedParams.maskMode;
    sfaConstInfo.n2G = sfaConstInfo.n2Size * sfaConstInfo.gSize;
    sfaConstInfo.s1Dv = sfaConstInfo.s1Size * sfaConstInfo.dSizeV;
    sfaConstInfo.s2Dv = sfaConstInfo.s2Size * sfaConstInfo.dSizeV;
    sfaConstInfo.n2Dv = sfaConstInfo.n2Size * sfaConstInfo.dSizeV;
    sfaConstInfo.gDv = sfaConstInfo.gSize * sfaConstInfo.dSizeV;
    sfaConstInfo.isActualLenDimsNull = sharedParams.isActualSeqLengthsNull;
    sfaConstInfo.isActualLenDimsKVNull = sharedParams.isActualSeqLengthsKVNull;
    sfaConstInfo.n2S2Dv = sfaConstInfo.n2Size * sfaConstInfo.s2Dv;
    sfaConstInfo.n2GDv = sfaConstInfo.n2Size * sfaConstInfo.gDv;
    sfaConstInfo.s2BaseN2Dv = sfaConstInfo.s2BaseSize * sfaConstInfo.n2Dv;
    sfaConstInfo.layoutType = sharedParams.layoutType;

    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        // (BS)ND
        sfaConstInfo.s1BaseN2GDv = sfaConstInfo.s1BaseSize * sfaConstInfo.n2GDv;
        sfaConstInfo.mm1Ka = sfaConstInfo.n2Size * sfaConstInfo.dSize;
        if ASCEND_IS_AIV {
            sfaConstInfo.attentionOutStride =
                (sfaConstInfo.n2G - sfaConstInfo.gSize) * sfaConstInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
        // BSH/BSNGD
        sfaConstInfo.s1BaseN2GDv = sfaConstInfo.s1BaseSize * sfaConstInfo.n2GDv;
        sfaConstInfo.mm1Ka = sfaConstInfo.n2Size * sfaConstInfo.dSize;
        if ASCEND_IS_AIV {
            sfaConstInfo.attentionOutStride =
                (sfaConstInfo.n2G - sfaConstInfo.gSize) * sfaConstInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
    if ASCEND_IS_AIV {
        sfaConstInfo.softmaxScale = sharedParams.softmaxScale;
        sfaConstInfo.blockSize = sharedParams.blockSize;
        sfaConstInfo.maxBlockNumPerBatch = sharedParams.maxBlockNumPerBatch;
    }

    if ASCEND_IS_AIV {
        sfaConstInfo.keyStride0 = this->tilingData->baseParams.keyStride0;
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    // bsize + 1-> bsize
    this->sfaConstInfo.actualSeqLenSize = this->sharedParams.bSize;
    this->sfaConstInfo.actualSeqLenKVSize = this->sharedParams.bSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }

    ProcessMainLoop();
    FreeEvent();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::FreeEvent()
{
    if ASCEND_IS_AIC {
        cubeBlock.UninitLocalBuffer();
        bmm1Buffers.Get().WaitCrossCore();
        bmm1Buffers.Get().WaitCrossCore();
        bmm2Buffers.Get().WaitCrossCore();
    }
    if ASCEND_IS_AIV {
        vecBlock.UninitLocalBuffer();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    bool hasLoad = aicIdx < usedCoreNum;
    if (!hasLoad) {
        if ASCEND_IS_AIV {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<SFA_SYNC_MODE0, PIPE_MTE3>(15);
                    CrossCoreWaitFlag<SFA_SYNC_MODE0, PIPE_MTE3>(15);
                }
            }
        }
        return;
    }

    // 适配分核左闭右开
    uint32_t bIdx = sfaConstInfo.bN2End / sfaConstInfo.n2Size;
    uint32_t sfaActS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
    uint32_t sfaGS1max = sfaActS1Size;
    /* constInfo.gS1End != gS1max时，gS1End需要往后加一格, bN2End不变 */
    if (sfaConstInfo.gS1End + 1 < sfaGS1max) {
        sfaConstInfo.gS1End = sfaConstInfo.gS1End + 1;
    } else {
        /* constInfo.gS1End == gS1max，bN2End需要往后加一格，bN2End变为0，以代表末尾 */
        sfaConstInfo.bN2End = sfaConstInfo.bN2End + 1;
        sfaConstInfo.gS1End = 0;
    }

    // 分核信息
    uint32_t sfaBN2StartIdx = sfaConstInfo.bN2Start;
    uint32_t bN2EndIdx = sfaConstInfo.bN2End;
    uint32_t sfaGS1StartIdx = sfaConstInfo.gS1Start;
    uint32_t nextGs1Idx = sfaConstInfo.gS1End;
    uint32_t s2StartIdx = 0;
    uint32_t s2EndIdx = 0;
    uint32_t s2LoopLimit = 0;

    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    int64_t taskId = 0;
    bool sfaHasPendingPipelineTask = true;
    RunInfo sfaRunInfo[3];
    RunParamStr sfaRunParam;
    int64_t multiCoreInnerIdx = 1;

    for (int64_t sfaBnIdx = sfaBN2StartIdx; sfaBnIdx < bN2EndIdx; sfaBnIdx++) {
        bool lastBN = (sfaBnIdx == bN2EndIdx - 1);
        sfaRunParam.boIdx = sfaBnIdx;
        sfaRunParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo, this->actualSeqQlenAddr,
                                              this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo, lastBN, nextGs1Idx, sfaGS1StartIdx);

        int64_t gS1LoopEnd = lastBN ? (sfaRunParam.gs1LoopEndIdx + PRELOAD_NUM) : sfaRunParam.gs1LoopEndIdx;
        for (int64_t gS1Index = sfaRunParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool sfaHasTwoPipelineStages = true;
            if (lastBN) {
                int32_t sfaExtraGS1 = gS1Index - sfaRunParam.gs1LoopEndIdx;
                switch (sfaExtraGS1) {
                    case 0:
                        sfaHasTwoPipelineStages = false;
                        break;
                    case 1:
                        sfaHasPendingPipelineTask = false;
                        sfaHasTwoPipelineStages = false;
                        break;
                    default:
                        break;
                }
            }
            if (sfaHasTwoPipelineStages) {
                this->ComputeAxisIdxByBnAndGs1(sfaBnIdx, gS1Index, sfaRunParam);
                bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo, gS1Index,
                                                                       this->actualSeqQlenAddr);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(sfaRunParam, this->sfaConstInfo);
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = sfaRunParam.s2LoopEndIdx - 1;
                if constexpr (IS_SPLIT_G) {
                    maxS2LoopCnt -= (s2LoopLimit + 1);
                }
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (sfaHasTwoPipelineStages) {
                    RunInfo& runInfo1 = sfaRunInfo[taskId % 3];
                    this->SetRunInfo(runInfo1, sfaRunParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(),
                                                    v0ResGmBuffers.Get(), runInfo1, this->sfaConstInfo);
                    } else {
                        this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), v0ResGmBuffers.Get(), runInfo1,
                                                   this->sfaConstInfo, 0);
                    }
                } else {
                    if ASCEND_IS_AIV {
                        if constexpr (IS_SPLIT_G) {
                            if (maxS2LoopCnt > 0) {
                                maxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
                            }
                        }
                    }
                }
                if (taskId > 0 && sfaHasPendingPipelineTask) {
                    auto& sfaRunInfo2 = sfaRunInfo[(taskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec1(this->l1RightBuffers.GetReused(), this->bmm1Buffers.Get(),
                                                   sfaRunInfo2, this->sfaConstInfo);
                    } else {
                        RunInfo& sfaRunInfo2 = sfaRunInfo[(taskId + 2) % 3];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1RightBuffers,
                                                    this->l1RightBuffers.GetReused(), sfaRunInfo2, this->sfaConstInfo);
                    }
                }
                if (taskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo& sfaRunInfo3 = sfaRunInfo[(taskId + 1) % 3];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), sfaRunInfo3, this->sfaConstInfo);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        sfaGS1StartIdx = 0;
    }
    if ASCEND_IS_AIV {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr& sfaRunParam)
{
    // GS1合轴, 不切G, 只切S1
    sfaRunParam.s1oIdx = gS1Index * sfaRunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        uint32_t firstHalfG = (sfaConstInfo.gSize + 1) >> 1;
        sfaRunParam.goIdx =
            (aicIdx % 2 == 0) ?
                0 :
                firstHalfG; // N1>64场景，相邻cube核处理一个s1，第一个cube核承担前一半，第二个cube核承担后一半
    } else {
        sfaRunParam.goIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo& sfaRunInfo, RunParamStr& sfaRunParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < sfaRunParam.kvLoopEndIdx) {
        sfaRunInfo.s2StartIdx = sfaRunParam.s2LineStartIdx;
        sfaRunInfo.s2EndIdx = sfaRunParam.s2LineEndIdx;
    }
    sfaRunInfo.s2LoopCount = s2LoopCount;
    if (sfaRunInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        sfaRunInfo.s1oIdx = sfaRunParam.s1oIdx;
        sfaRunInfo.boIdx = sfaRunParam.boIdx;
        sfaRunInfo.n2oIdx = sfaRunParam.n2oIdx;
        sfaRunInfo.goIdx = sfaRunParam.goIdx;
        sfaRunInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        sfaRunInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        sfaRunInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    sfaRunInfo.taskId = taskId;
    sfaRunInfo.taskIdMod2 = taskId & 1;
    sfaRunInfo.taskIdMod3 = taskId % 3;
    sfaRunInfo.s2LoopLimit = s2LoopLimit;

    sfaRunInfo.actualS1Size = sfaRunParam.actualS1Size;
    sfaRunInfo.actualS2Size = sfaRunParam.actualS2Size;
    sfaRunInfo.attentionOutOffset = sfaRunParam.attentionOutOffset;
    sfaRunInfo.sOuterOffset = sfaRunParam.sOuterOffset;
    this->ComputeBmm1Tail(sfaRunInfo, sfaRunParam);
    InitUniqueRunInfo(sfaRunParam, sfaRunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr& sfaRunParam, RunInfo& sfaRunInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(sfaRunParam, sfaRunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo& sfaRunInfo, RunParamStr& sfaRunParam)
{
    // ------------------------S1 Base Related---------------------------
    sfaRunInfo.s1RealSize = sfaRunParam.s1RealSize;
    sfaRunInfo.halfS1RealSize = sfaRunParam.halfS1RealSize;
    sfaRunInfo.firstHalfS1RealSize = sfaRunParam.firstHalfS1RealSize;
    sfaRunInfo.mRealSize = sfaRunParam.mRealSize;
    sfaRunInfo.halfMRealSize = sfaRunParam.halfMRealSize;
    sfaRunInfo.firstHalfMRealSize = sfaRunParam.firstHalfMRealSize;

    sfaRunInfo.vec2S1BaseSize = sfaRunInfo.halfS1RealSize;
    sfaRunInfo.vec2MBaseSize = sfaRunInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    sfaRunInfo.s2RealSize = sfaConstInfo.s2BaseSize;
    sfaRunInfo.s2AlignedSize = sfaRunInfo.s2RealSize;
    int64_t curS2LoopCnt = sfaRunInfo.s2LoopCount;
    if (sfaRunInfo.s2StartIdx + (curS2LoopCnt + 1) * sfaRunInfo.s2RealSize > sfaRunInfo.s2EndIdx) {
        sfaRunInfo.s2RealSize = sfaRunInfo.s2EndIdx - curS2LoopCnt * sfaRunInfo.s2RealSize - sfaRunInfo.s2StartIdx;
        sfaRunInfo.s2AlignedSize = Align(sfaRunInfo.s2RealSize);
    }
}

} // namespace BaseApi
#endif // SPARSE_FLASH_ATTENTION_KERNEL_MLA_ARCH35_H
