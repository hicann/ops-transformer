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
 * \file quant_block_sparse_attn_mx_kernel.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 kernel 入口。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_

#include "kernel_operator.h"
#include "../quant_block_sparse_attn_mx_tiling_data.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_attenmask.h"
#include "quant_block_sparse_attn_kvcache.h"
#include "quant_block_sparse_attn_mx_block_cube.h"
#include "quant_block_sparse_attn_mx_block_vec.h"
#include "common/buffer_manager.h"
#include "common/buffers_policy.h"
#include "common/util_regbase_mx.h"

using namespace AscendC;
using namespace optiling;

namespace BaseApi {
// MX 独立 tiling/kernel 路径：两个 256 C1/V1 subLoop 生成 P[128,512]，再执行一次 C2/V2。
template <typename inputType, typename mmType, typename outputType, QBSALayout layout, QBSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool hasAtten, bool hasLse, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxKernel {
public:
    using INPUT_T = inputType;
    using MM_T = mmType;
    using OUTPUT_T = outputType;
    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_ATTEN = hasAtten;
    static constexpr bool HAS_LSE = hasLse;
    static constexpr bool IS_PA = isPa;
    static constexpr QBSALayout LAYOUT = layout;
    static constexpr QBSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == QBSALayout::TND && KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MXFullQuantMode currently only supports TND query and PA_BNBD KV");
    static_assert(M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MXFullQuantMode currently only supports S1=128, S2=512, D=128 and DV=128");
    // 延迟两个 task 执行 C2/V2。
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1U;

    __aicore__ inline QuantBlockSparseAttnMxKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *,
                                __gm__ uint8_t *, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *qScale, __gm__ uint8_t *kScale, __gm__ uint8_t *vScale,
                                __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *, const QuantBlockSparseAttnMxTilingData *__restrict tiling,
                                TPipe *tPipe)
    {
        // 跨核 Buffer ID 由核内状态生成，ACL 图重放会复用该状态；每次启动时需重置计数器，
        // 保证 AIC 与 AIV 使用相同的事件 ID 序列。
        fa_base_matmul::idCounterNum = 0;
        // Q/K/V 在 cube 中结合 block_table 完成 PA 搬运。
        pipe = tPipe;
        tilingData = tiling;
        queryPtr = query;
        keyPtr = key;
        valuePtr = value;
        sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t *)sparseSeqLen);
        metadataGm.SetGlobalBuffer((__gm__ int32_t *)metadata);
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        seqUsedKvGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv);
        // 预取序列长度元数据，重叠 InitBuffers 期间的 HBM 延迟。
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(cuSeqlensQ), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(seqUsedKv), 0);
        InitConstInfo();
        InitMMResBuf();
        cubeBlock.Init(pipe, &l1BufferManager, queryPtr, keyPtr, valuePtr, blockTable, qScale, kScale, vScale);
        __gm__ uint8_t *pScalePtr = (tilingData->scaleParams.pScaleShapeSize == 0U) ? nullptr : pScale;
        vecBlock.Init(pipe, pScalePtr, softmaxLse, attentionOut, attenMask, tilingData->attenMaskParams.attenMaskS2Size,
                      tilingData->scaleParams.pScaleDtype);
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx >= constInfo.coreNum || constInfo.aicIdx >= QBSA_FA_AIC_CORE_NUM) {
            return;
        }
        CrossCoreBufferInit();
        if ASCEND_IS_AIC {
            cubeBlock.InitBuffers();
        } else {
            vecBlock.InitBuffers();
            vecBlock.ClearOutput(tilingData->emptyTensorParams, constInfo);
        }
        ProcessMainLoop();
        CrossCoreBufferUninit();
    }

private:
    __aicore__ inline void InitConstInfo()
    {
        // 两个 AIV subBlock 共享逻辑 AIC id。
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        const auto &baseParams = tilingData->baseParams;
        const auto &pageAttentionParams = tilingData->pageAttentionParams;
        const auto &sparseParams = tilingData->sparseParams;
        const auto &scaleParams = tilingData->scaleParams;

        constInfo.n2Size = baseParams.n2Size;
        constInfo.gSize = baseParams.gSize;
        constInfo.realN2Size = baseParams.n2Size * baseParams.gSize;
        constInfo.dSize = baseParams.dSize;
        constInfo.dSizeV = baseParams.dSizeV;
        constInfo.scaleValue = baseParams.scaleValue;
        constInfo.coreNum = baseParams.coreNum;
        constInfo.n2GD = constInfo.realN2Size * constInfo.dSize;
        constInfo.attentionOutStride = (constInfo.realN2Size - 1U) * constInfo.dSizeV * sizeof(bfloat16_t);
        constInfo.softmaxLseStride = (constInfo.realN2Size - 1U) * sizeof(float);

        constInfo.maxBlockNumPerBatch = pageAttentionParams.maxBlockNumPerBatch;
        constInfo.paBlockStride = pageAttentionParams.paBlockStride;
        constInfo.qSparseBlockSize = pageAttentionParams.qBlockSize;
        constInfo.kvSparseBlockSize = pageAttentionParams.kvBlockSize;
        constInfo.paBlockSize = pageAttentionParams.blockSize;

        constInfo.maxQb = sparseParams.maxQb;
        constInfo.maxKb = sparseParams.maxKb;

        // QScale 为 [T,N,D/64,2]；K/V scale 在 PA 场景按 BNBD 映射。
        constInfo.scaleLastDim = scaleParams.scaleLastDim;
        constInfo.queryScaleDSize = scaleParams.queryScaleDSize;
        constInfo.keyScaleDSize = scaleParams.keyScaleDSize;
        constInfo.valueScaleDSize = scaleParams.valueScaleDSize;
        constInfo.qScaleN1D = constInfo.realN2Size * constInfo.queryScaleDSize * constInfo.scaleLastDim;
        constInfo.kScaleN2D = static_cast<uint32_t>(baseParams.kScaleStrides.n2Stride);
        constInfo.valueScaleN2D = static_cast<uint32_t>(baseParams.vScaleStrides.n2Stride);
    }

    __aicore__ inline void InitMMResBuf()
    {
        // C1 UB 为 [64,256] fp32，C2 UB 为 [64,128] fp32。
        constexpr uint32_t mm1ResultSize = M_BASE / CV_RATIO * S2_BASE * sizeof(MM_T) / 2U;
        constexpr uint32_t mm2ResultSize = M_BASE / CV_RATIO * DV_BASE * sizeof(MM_T);
        // L1 连续保存 P(e4m3) 和 PScale(e8m0)，供 C2 随路反量化。
        constexpr uint32_t mm2LeftSize = M_BASE * S2_BASE * sizeof(INPUT_T) + M_BASE * S2_BASE / 32U;
        l1BufferManager.Init(pipe, 512U * 1024U);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        ubBufferManager.Init(pipe, mm1ResultSize * 2U + mm2ResultSize);
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
            bmm2Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUninit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            bmm2Buffers.Get().WaitCrossCore();
        }
    }

    __aicore__ inline uint32_t GetActualQSeqLen(uint32_t bIdx)
    {
        return static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx + 1U) - cuSeqlensQGm.GetValue(bIdx));
    }

    __aicore__ inline uint32_t GetActualKvSeqLen(uint32_t bIdx)
    {
        return static_cast<uint32_t>(seqUsedKvGm.GetValue(bIdx));
    }

    __aicore__ inline uint32_t GetQTokenBase(uint32_t bIdx)
    {
        return static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx));
    }

    __aicore__ inline uint32_t GetS1LoopStart(uint32_t bn1Idx, uint32_t bn1StartIdx, uint32_t s1StartIdx) const
    {
        return bn1Idx == bn1StartIdx ? s1StartIdx : 0U;
    }

    __aicore__ inline uint32_t GetS1LoopEnd(uint32_t bn1Idx, uint32_t bn1EndIdx, uint32_t s1EndIdx,
                                            uint32_t actualS1Size) const
    {
        const uint32_t loopEnd = (actualS1Size + constInfo.qSparseBlockSize - 1U) / constInfo.qSparseBlockSize;
        if (s1EndIdx != 0U && bn1Idx == bn1EndIdx - 1U) {
            return s1EndIdx;
        }
        return loopEnd;
    }

    __aicore__ inline bool ShouldMoveSparseBlockBack(int64_t previousBlockIdx, int64_t currentBlockIdx) const
    {
        // 有效 block 升序，无效 block 放末尾。
        if (previousBlockIdx < 0) {
            return currentBlockIdx >= 0;
        }
        if (currentBlockIdx < 0) {
            return false;
        }
        return previousBlockIdx > currentBlockIdx;
    }

    __aicore__ inline void SortSparseBlocks(MxRunInfo &runInfo) const
    {
        // 最多 8 个 block，排序后再统一生成 offset 与 mask 状态。
        for (uint32_t i = 1U; i < runInfo.sparseBlockCount; ++i) {
            const int64_t currentBlockIdx = runInfo.sparseBlockIdx[i];
            uint32_t insertPos = i;
            while (insertPos > 0U &&
                   ShouldMoveSparseBlockBack(runInfo.sparseBlockIdx[insertPos - 1U], currentBlockIdx)) {
                runInfo.sparseBlockIdx[insertPos] = runInfo.sparseBlockIdx[insertPos - 1U];
                --insertPos;
            }
            runInfo.sparseBlockIdx[insertPos] = currentBlockIdx;
        }
    }

    __aicore__ inline void FillRunInfo(MxRunInfo &runInfo, uint64_t loop, uint32_t mLoop, uint32_t bIdx, uint32_t n1Idx,
                                       uint32_t n2Idx, uint32_t queryTokenBase, uint32_t actualS1Size,
                                       uint32_t actualS2Size, uint32_t s1OuterIdx, uint32_t s2LoopIdx,
                                       uint32_t s2LoopEnd, uint32_t sparseBlockCount, uint64_t sparseBase,
                                       bool isFirstValidS2Loop)
    {
        // 构造一个 512-token logical S2 tile 的运行参数。
        // 仅清零必要的字段
        runInfo.actSingleLoopS2Size = 0U;
        runInfo.loop = static_cast<uint32_t>(loop);
        runInfo.mLoop = mLoop;
        runInfo.isValid = true;
        runInfo.bIdx = bIdx;
        runInfo.realN2Idx = n1Idx;
        runInfo.n2Idx = n2Idx;
        // s1OuterIdx 是 sparse Q block 下标。MX 的计算上限仍为 M=128，
        // q_block_size=64 时每个 task 只处理对应的 64 行，避免跨两个 sparse 列表。
        runInfo.s1Idx = s1OuterIdx * constInfo.qSparseBlockSize;
        runInfo.queryTokenBase = queryTokenBase;
        runInfo.actS1Size = actualS1Size;
        runInfo.actS2Size = actualS2Size;
        runInfo.actMSize = constInfo.qSparseBlockSize;
        if (unlikely(runInfo.s1Idx + constInfo.qSparseBlockSize > runInfo.actS1Size)) {
            runInfo.actMSize = static_cast<uint32_t>(runInfo.actS1Size) - runInfo.s1Idx;
        }
        runInfo.actMSizeAlign32 = (runInfo.actMSize + 31U) >> 5 << 5;
        runInfo.actVecMSize = runInfo.actMSize <= 16U ? runInfo.actMSize : (runInfo.actMSizeAlign32 >> 1U);
        runInfo.vecMbaseIdx = 0U;
        if (constInfo.subBlockIdx == 1U) {
            runInfo.vecMbaseIdx = runInfo.actVecMSize;
            runInfo.actVecMSize = runInfo.actMSize - runInfo.actVecMSize;
        }
        runInfo.sparseBlockCount = sparseBlockCount;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            runInfo.sparseBlockIdx[i] = sparseIndicesGm.GetValue(sparseBase + s2LoopIdx + i);
        }
        SortSparseBlocks(runInfo);
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const int64_t sparseBlkIdx = runInfo.sparseBlockIdx[i];
            uint64_t sparseTokenOffset = 0U;
            uint32_t sparseBlockRealSize = 0U;
            runInfo.sparseBlockTileOffset[i] = runInfo.actSingleLoopS2Size;
            // 负 index 不参与拼接。
            if (sparseBlkIdx >= 0) {
                sparseTokenOffset = static_cast<uint64_t>(sparseBlkIdx) * constInfo.kvSparseBlockSize;
                const int64_t remainS2 =
                    static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(sparseTokenOffset);
                sparseBlockRealSize = remainS2 > static_cast<int64_t>(constInfo.kvSparseBlockSize) ?
                                          constInfo.kvSparseBlockSize :
                                          (remainS2 > 0 ? static_cast<uint32_t>(remainS2) : 0U);
            }
            runInfo.sparseBlockTokenOffset[i] = sparseTokenOffset;
            runInfo.sparseBlockRealSize[i] = sparseBlockRealSize;
            runInfo.sparseBlockPartialMask[i] = false;
            if constexpr (HAS_ATTEN) {
                if (likely(sparseBlockRealSize != 0U)) {
                    const int64_t causalDelta =
                        static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
                    int64_t sparseBlkIdxForMask = sparseBlkIdx;
                    AttentionMaskFullProcessingOrRequired(
                        static_cast<int64_t>(runInfo.s1Idx), static_cast<int64_t>(sparseTokenOffset), causalDelta,
                        sparseBlockRealSize, sparseBlkIdxForMask, runInfo.sparseBlockPartialMask[i]);
                }
            }
            runInfo.actSingleLoopS2Size += sparseBlockRealSize;
        }
        runInfo.actSingleLoopS2SizeAlign = (runInfo.actSingleLoopS2Size + 63U) >> 6 << 6;
        runInfo.isFirstS2Loop = isFirstValidS2Loop;
        runInfo.isLastS2Loop = s2LoopIdx + runInfo.sparseBlockCount - 1U >= s2LoopEnd;
        runInfo.queryOffset =
            (runInfo.queryTokenBase + runInfo.s1Idx) * constInfo.n2GD + runInfo.realN2Idx * constInfo.dSize;
        runInfo.queryScaleOffset = (runInfo.queryTokenBase + runInfo.s1Idx) * constInfo.qScaleN1D +
                                   runInfo.realN2Idx * constInfo.queryScaleDSize * constInfo.scaleLastDim;
    }

    __aicore__ inline bool HasValidTask(MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE]) const
    {
        for (uint32_t i = 0U; i < PRELOAD_TASK_CACHE_SIZE; ++i) {
            if (taskRunInfo[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        // 当前 task 做 C1/V1，PRELOAD_N 之前的 task 做 C2/V2。
        MxRunInfo &runInfoCur = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        MxRunInfo &runInfoPre = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
        if (likely(runInfoCur.isValid)) {
            const uint32_t c1v1Loop = (runInfoCur.actSingleLoopS2Size + S2_SPLIT - 1U) / S2_SPLIT;
            if ASCEND_IS_AIC {
                for (uint32_t subLoop = 0U; subLoop < c1v1Loop; ++subLoop) {
                    cubeBlock.IterateBmm1(bmm1Buffers.Get(), runInfoCur, constInfo, subLoop);
                }
            } else {
                Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &pBuf = l1PBuffers.GetVec();
                for (uint32_t subLoop = 0U; subLoop < c1v1Loop; ++subLoop) {
                    vecBlock.ProcessVec1(pBuf, bmm1Buffers.Get(), runInfoCur, constInfo, subLoop);
                }
            }
        }
        if (likely(loop >= PRELOAD_N && runInfoPre.isValid)) {
            if ASCEND_IS_AIC {
                cubeBlock.IterateBmm2(bmm2Buffers.Get(), l1PBuffers, runInfoPre, constInfo);
            } else {
                vecBlock.ProcessVec2(bmm2Buffers.Get(), runInfoPre, constInfo);
            }
            runInfoPre.isValid = false;
        }
    }

    __aicore__ inline void ProcessMainLoop()
    {
        // 三槽环形缓存组织 C1/V1 与延迟 C2/V2。
        MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        uint64_t loop = 0U;
        uint32_t mLoop = 0U;
        // S2_BASE=512, kvSparseBlockSize=64/128(host 侧已拦截), 结果必为 8 或 4。
        const uint32_t sparseBlocksPerTask = S2_BASE / constInfo.kvSparseBlockSize;
        // metadata 可包含多个 section；每个 section 为当前 AIC 提供一段独立的 BN1/S1 边界。
        const uint32_t sectionNum = GetBsaSectionNum(metadataGm);
        for (uint32_t sectionIdx = 0U; sectionIdx < sectionNum; ++sectionIdx) {
            const BsaFaCoreMetadata coreMetadata = GetBsaCoreMetadata(metadataGm, sectionIdx, constInfo.aicIdx);
            if (unlikely(coreMetadata.coreEnable == 0U)) {
                continue;
            }
            const uint32_t bn1StartIdx = coreMetadata.bn1StartIdx;
            uint32_t bn1EndIdx = coreMetadata.bn1EndIdx;
            const uint32_t s1StartIdx = coreMetadata.s1StartIdx;
            const uint32_t s1EndIdx = coreMetadata.s1EndIdx;
            if (s1EndIdx != 0U) {
                ++bn1EndIdx;
            }
            if (unlikely(bn1EndIdx <= bn1StartIdx)) {
                continue;
            }

            for (uint32_t bn1Idx = bn1StartIdx; bn1Idx < bn1EndIdx; ++bn1Idx) {
                const uint32_t bIdx = bn1Idx / constInfo.realN2Size;
                const uint32_t n1Idx = bn1Idx % constInfo.realN2Size;
                const uint32_t n2Idx = n1Idx / constInfo.gSize;
                const uint32_t queryTokenBase = GetQTokenBase(bIdx);
                const uint32_t actualS1Size = GetActualQSeqLen(bIdx);
                const uint32_t actualS2Size = GetActualKvSeqLen(bIdx);
                const uint32_t s1LoopStart = GetS1LoopStart(bn1Idx, bn1StartIdx, s1StartIdx);
                const uint32_t s1LoopEnd = GetS1LoopEnd(bn1Idx, bn1EndIdx, s1EndIdx, actualS1Size);
                for (uint32_t s1OuterIdx = s1LoopStart; s1OuterIdx < s1LoopEnd; ++s1OuterIdx) {
                    // 当前 query block 的有效 sparse block 数。
                    const int32_t sparseLen = sparseSeqLenGm.GetValue(
                        (static_cast<uint64_t>(bIdx) * constInfo.realN2Size + n1Idx) * constInfo.maxQb + s1OuterIdx);
                    if (unlikely(sparseLen <= 0)) {
                        continue;
                    }
                    // sparseBase 指向 sparseIndices[B,N1,Qb,0]。
                    const uint64_t sparseBase = (static_cast<uint64_t>(bIdx) * constInfo.realN2Size + n1Idx) *
                                                    constInfo.maxQb * constInfo.maxKb +
                                                static_cast<uint64_t>(s1OuterIdx) * constInfo.maxKb;
                    const uint32_t s2LoopEnd = static_cast<uint32_t>(sparseLen - 1);
                    bool hasValidS2Task = false;
                    for (uint32_t s2LoopIdx = 0U; s2LoopIdx <= s2LoopEnd; s2LoopIdx += sparseBlocksPerTask) {
                        MxRunInfo &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
                        const uint32_t remainSparseBlockCount = static_cast<uint32_t>(sparseLen) - s2LoopIdx;
                        const uint32_t sparseBlockCount =
                            remainSparseBlockCount > sparseBlocksPerTask ? sparseBlocksPerTask : remainSparseBlockCount;
                        FillRunInfo(runInfo, loop, mLoop, bIdx, n1Idx, n2Idx, queryTokenBase, actualS1Size,
                                    actualS2Size, s1OuterIdx, s2LoopIdx, s2LoopEnd, sparseBlockCount, sparseBase,
                                    !hasValidS2Task);
                        if (likely(runInfo.actSingleLoopS2Size != 0U && runInfo.actMSize != 0U)) {
                            ExecuteTask(loop, taskRunInfo);
                            hasValidS2Task = true;
                            ++loop;
                        } else {
                            runInfo.isValid = false;
                        }
                    }
                    ++mLoop;
                }
            }
        }
        // 排空尚未执行 C2/V2 的尾部 task。
        while (HasValidTask(taskRunInfo)) {
            ExecuteTask(loop, taskRunInfo);
            ++loop;
        }
    }

    TPipe *pipe = nullptr;
    const QuantBlockSparseAttnMxTilingData *__restrict tilingData = nullptr;
    regbasemx::MxConstInfo constInfo;
    __gm__ uint8_t *queryPtr = nullptr;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    GlobalTensor<int32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> seqUsedKvGm;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    QuantBlockSparseAttnMxBlockCube<INPUT_T, MM_T, OUTPUT_T, LAYOUT, KV_LAYOUT, s1TemplateType, s2TemplateType,
                                    dTemplateType, dVTemplateType, IS_PA, USE_DN>
        cubeBlock;
    QuantBlockSparseAttnMxBlockVec<INPUT_T, MM_T, OUTPUT_T, LAYOUT, KV_LAYOUT, s1TemplateType, s2TemplateType,
                                   dTemplateType, dVTemplateType, HAS_ATTEN, HAS_LSE, IS_PA, USE_DN>
        vecBlock;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
