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
#include "../common/quant_block_sparse_attn_common_arch35.h"
#include "../common/quant_block_sparse_attn_attenmask.h"
#include "../common/quant_block_sparse_attn_kvcache.h"
#include "quant_block_sparse_attn_mx_block_cube.h"
#include "quant_block_sparse_attn_mx_block_vec.h"
#include "quant_block_sparse_attn_mx_metadata.h"
#include "quant_block_sparse_attn_mx_indices_cache.h"
#include "../common/buffer_manager.h"
#include "../common/buffers_policy.h"
#include "../common/util_regbase_mx.h"

// [PLAN_A_PERF] Keep cross-section overlap by default. STC060 was a metadata-layout
// mismatch, not a reason to drain every section. Set to 1 only for diagnostic A/B.
#ifndef QBSA_9102_MX_DIAG_DRAIN_SECTION
#define QBSA_9102_MX_DIAG_DRAIN_SECTION 0
#endif
static_assert(QBSA_9102_MX_DIAG_DRAIN_SECTION == 0 || QBSA_9102_MX_DIAG_DRAIN_SECTION == 1,
              "QBSA_9102_MX_DIAG_DRAIN_SECTION must be 0 or 1");

using namespace AscendC;
using namespace optiling;

namespace BaseApi {
// MX 独立 tiling/kernel 路径：两个 256 C1/V1 subLoop 生成 P[64,512]，再执行一次 C2/V2。
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
    static constexpr uint32_t LOGICAL_M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t M_BASE = 64U;
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == QBSALayout::TND && KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MXFullQuantMode currently only supports TND query and PA_BNBD KV");
    static_assert(LOGICAL_M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MXFullQuantMode currently only supports logical S1=128, S2=512, D=128 and DV=128");
    // 延迟两个 task 执行 C2/V2。
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1U;

    __aicore__ inline QuantBlockSparseAttnMxKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseSeqLen, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t*, __gm__ uint8_t*,
                                __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* blockTable, __gm__ uint8_t* qScale,
                                __gm__ uint8_t* kScale, __gm__ uint8_t* vScale, __gm__ uint8_t* pScale,
                                __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut, __gm__ uint8_t*,
                                const QuantBlockSparseAttnMxTilingData* __restrict tiling, TPipe* tPipe)
    {
        // Do not allow block-local storage retained by a previous launch to become this launch's defaults.
        // InitConstInfo overwrites the fields used by the active sub-core; the aggregate reset also covers
        // fields that are intentionally unused on AIC or AIV.
        constInfo = {};
        // 跨核 Buffer ID 由核内状态生成，ACL 图重放会复用该状态；每次启动时需重置计数器，
        // 保证 AIC 与 AIV 使用相同的事件 ID 序列。
        fa_base_matmul::idCounterNum = 0;
        // Q/K/V 在 cube 中结合 block_table 完成 PA 搬运。
        pipe = tPipe;
        tilingData = tiling;
        queryPtr = query;
        keyPtr = key;
        valuePtr = value;
        sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t*)sparseSeqLen);
        metadataGm.SetGlobalBuffer((__gm__ int32_t*)metadata);
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
        seqUsedKvGm.SetGlobalBuffer((__gm__ int32_t*)seqUsedKv);
        // 预取序列长度元数据，重叠 InitBuffers 期间的 HBM 延迟。
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(cuSeqlensQ), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(seqUsedKv), 0);
        InitConstInfo();
        InitMMResBuf();
        cubeBlock.Init(pipe, &l1BufferManager, queryPtr, keyPtr, valuePtr, blockTable, qScale, kScale, vScale);
        __gm__ uint8_t* pScalePtr = (tilingData->scaleParams.pScaleShapeSize == 0U) ? nullptr : pScale;
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
            // [PLAN_A_RESIDENT_PSCALE_OPT] Complete one-time L1 fills before Q/K/V reserve local events.
            cubeBlock.InitResidentPScale(l1PBuffers);
            cubeBlock.InitBuffers();
        } else {
            vecBlock.InitBuffers();
            // [PLAN_A_INDICES_UB_OPT] Append after existing allocations: no matrix UB offset changes.
            indicesCache.Init(pipe, constInfo.maxKb);
        }
        ProcessMainLoop();
        // Drain both directions of every cross-core slot before releasing local HardEvent IDs.  In particular,
        // the final bmm2 forward flag below is an AIC->AIV acknowledgement that all reverse release flags have
        // been consumed; without it a following kernel launch can reuse the same 0..10 cross-core IDs too early.
        CrossCoreBufferUninit();
        if ASCEND_IS_AIC {
            cubeBlock.FreeEventID();
        } else {
            vecBlock.FreeEventID();
            indicesCache.FreeEventID(pipe);
        }
        // task_reset must observe no pending instruction or release flag on either sub-core.
        SyncAll();
        fa_base_matmul::idCounterNum = 0;
    }

private:
    __aicore__ inline void InitConstInfo()
    {
        // 9102r1 一个 AIV 对应一个 AIC。
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx();
        }

        const auto& baseParams = tilingData->baseParams;
        const auto& pageAttentionParams = tilingData->pageAttentionParams;
        const auto& sparseParams = tilingData->sparseParams;
        const auto& scaleParams = tilingData->scaleParams;

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
        constexpr uint32_t mm1ResultSize = M_BASE * S2_SPLIT * sizeof(MM_T);
        constexpr uint32_t mm2ResultSize = M_BASE * DV_BASE * sizeof(MM_T);
        // L1 连续保存 P(e4m3) 和 PScale(e8m0)，供 C2 随路反量化。
        constexpr uint32_t mm2LeftSize = M_BASE * S2_BASE * sizeof(INPUT_T) + M_BASE * S2_BASE / 32U;
        l1BufferManager.Init(pipe, 512U * 1024U);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        // C1 三槽覆盖 PRELOAD_N+1。
        // 1:1 路径 3*2(C1) + 1*2(C2) + 3*1(P-L1) 正好使用 11 个跨核事件 ID。
        constexpr uint32_t c1BufferNum = 3U;
        ubBufferManager.Init(pipe, mm1ResultSize * c1BufferNum + mm2ResultSize);
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
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
            bmm1Buffers.Get().WaitCrossCore();
            auto& tailSyncBuf = bmm2Buffers.Get();
            tailSyncBuf.WaitCrossCore();
            // All AIV->AIC release flags are now consumed.  Return one final acknowledgement on the already
            // drained bmm2 forward channel so AIV cannot leave Process() before AIC finishes the cross-core drain.
            tailSyncBuf.SetCrossCore();
        } else {
            // The regular bmm2 forward flag was consumed by the last V2.  This wait consumes only the tail ack.
            bmm2Buffers.Get().WaitCrossCore();
        }
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

    __aicore__ inline void SortSparseBlocks(MxRunInfo& runInfo) const
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

    __aicore__ inline uint32_t GetLastValidS2LoopIdx(uint64_t sparseBase, uint32_t sparseLen,
                                                     uint32_t sparseBlocksPerTask, uint32_t actualS2Size)
    {
        // sparse_seq_len normally covers only valid indices. Keep the kernel robust when its tail contains
        // a negative or out-of-range block: the final pipeline task must be the final group that contributes data,
        // otherwise Q is never released and V2/LSE never perform their final update.
        uint32_t groupStart = (sparseLen - 1U) / sparseBlocksPerTask * sparseBlocksPerTask;
        while (true) {
            const uint32_t remainSparseBlockCount = sparseLen - groupStart;
            const uint32_t groupBlockCount =
                remainSparseBlockCount > sparseBlocksPerTask ? sparseBlocksPerTask : remainSparseBlockCount;
            for (uint32_t i = 0U; i < groupBlockCount; ++i) {
                const int64_t sparseBlockIdx = indicesCache.Read(sparseIndicesGm, sparseBase, groupStart + i);
                if (sparseBlockIdx >= 0 &&
                    static_cast<uint64_t>(sparseBlockIdx) * constInfo.kvSparseBlockSize < actualS2Size) {
                    return groupStart;
                }
            }
            if (groupStart == 0U) {
                break;
            }
            groupStart -= sparseBlocksPerTask;
        }
        // No loop starts at sparseLen, so this sentinel cannot mark an empty group as the final task.
        return sparseLen;
    }

    __aicore__ inline void FillRunInfo(MxRunInfo& runInfo, uint64_t loop, uint32_t mLoop, uint32_t bIdx, uint32_t n1Idx,
                                       uint32_t n2Idx, uint32_t queryTokenBase, uint32_t actualS1Size, uint32_t mPart,
                                       uint32_t actualS2Size, uint32_t s1OuterIdx, uint32_t s2LoopIdx,
                                       uint32_t sparseBlockCount, uint64_t sparseBase, bool isFirstValidS2Loop,
                                       bool isLastValidS2Loop)
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
        // s1OuterIdx 是 sparse Q block 下标；每个 M-part 固定最多处理 64 行。
        runInfo.s1Idx = s1OuterIdx * constInfo.qSparseBlockSize + mPart * M_BASE;
        runInfo.queryTokenBase = queryTokenBase;
        runInfo.actS1Size = actualS1Size;
        runInfo.actS2Size = actualS2Size;
        const uint32_t mPartStart = mPart * M_BASE;
        const uint32_t blockRemain =
            constInfo.qSparseBlockSize > mPartStart ? constInfo.qSparseBlockSize - mPartStart : 0U;
        runInfo.actMSize = blockRemain > M_BASE ? M_BASE : blockRemain;
        if (unlikely(runInfo.s1Idx >= runInfo.actS1Size)) {
            runInfo.actMSize = 0U;
        } else if (unlikely(runInfo.s1Idx + runInfo.actMSize > runInfo.actS1Size)) {
            runInfo.actMSize = static_cast<uint32_t>(runInfo.actS1Size) - runInfo.s1Idx;
        }
        runInfo.actVecMSize = runInfo.actMSize;
        if constexpr (HAS_ATTEN) {
            if ASCEND_IS_AIV {
                // Reset on every ring-slot reuse, including empty/invalid tasks.
                runInfo.attenMaskSubLoopBits = 0U;
            }
        }
        runInfo.sparseBlockCount = sparseBlockCount;
        indicesCache.ReadGroup(sparseIndicesGm, sparseBase, s2LoopIdx, runInfo.sparseBlockCount,
                               runInfo.sparseBlockIdx);
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
            if constexpr (HAS_ATTEN) {
                if ASCEND_IS_AIV {
                    runInfo.sparseBlockPartialMask[i] = false;
                    if (likely(sparseBlockRealSize != 0U)) {
                        const int64_t causalDelta =
                            static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
                        int64_t sparseBlkIdxForMask = sparseBlkIdx;
                        AttentionMaskFullProcessingOrRequired(
                            static_cast<int64_t>(runInfo.s1Idx), static_cast<int64_t>(sparseTokenOffset), causalDelta,
                            sparseBlockRealSize, sparseBlkIdxForMask, runInfo.sparseBlockPartialMask[i]);
                        // [PLAN_A_PERF] Match master: skip the mask VF only when
                        // every real sparse segment in this subLoop is visible.
                        // Use packed tile offsets, not the original KV indices.
                        if (unlikely(runInfo.sparseBlockPartialMask[i])) {
                            const uint32_t blockTileStart = runInfo.sparseBlockTileOffset[i];
                            const uint32_t blockTileEnd = blockTileStart + sparseBlockRealSize;
                            if (blockTileStart < S2_SPLIT) {
                                runInfo.attenMaskSubLoopBits |= 0x1U;
                            }
                            if (blockTileEnd > S2_SPLIT) {
                                runInfo.attenMaskSubLoopBits |= 0x2U;
                            }
                        }
                    }
                }
            }
            runInfo.actSingleLoopS2Size += sparseBlockRealSize;
        }
        runInfo.actSingleLoopS2SizeAlign = (runInfo.actSingleLoopS2Size + 63U) >> 6 << 6;
        runInfo.isFirstS2Loop = isFirstValidS2Loop;
        runInfo.isLastS2Loop = isLastValidS2Loop;
        if ASCEND_IS_AIC {
            const uint64_t queryTokenOffset = runInfo.queryTokenBase + runInfo.s1Idx;
            runInfo.queryOffset =
                queryTokenOffset * constInfo.n2GD + static_cast<uint64_t>(runInfo.realN2Idx) * constInfo.dSize;
            runInfo.queryScaleOffset =
                queryTokenOffset * constInfo.qScaleN1D +
                static_cast<uint64_t>(runInfo.realN2Idx) * constInfo.queryScaleDSize * constInfo.scaleLastDim;
        }
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
        MxRunInfo& runInfoCur = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        MxRunInfo& runInfoPre = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
        const bool hasBmm2 = loop >= PRELOAD_N && runInfoPre.isValid;
        if ASCEND_IS_AIC {
            // [PLAN_A_PREFETCH_OPT] Local handles cannot survive a task/section/kernel boundary.
            // [PLAN_A_C1_FIRST_OPT] Submit current C1 before preparing delayed V.
            // Prepare Q/K0 here; paired C1 prepares K1 after its first half.
            // V is still prepared before the C2 P-ready wait.
            // Both consumers below use the exact handles acquired here, never another Get().
            typename decltype(cubeBlock)::Bmm1Prefetch c1Inputs{};
            typename decltype(cubeBlock)::Bmm2Prefetch c2Inputs{};
            if (likely(runInfoCur.isValid)) {
                cubeBlock.PrefetchBmm1Inputs(c1Inputs, runInfoCur, constInfo);
            }
            if (likely(runInfoCur.isValid)) {
                if (likely(runInfoCur.actSingleLoopS2Size > S2_SPLIT)) {
                    auto& c1Out0 = bmm1Buffers.Get();
                    auto& c1Out1 = bmm1Buffers.Get();
                    cubeBlock.IterateBmm1ReuseQ(c1Out0, c1Out1, runInfoCur, constInfo, c1Inputs);
                } else {
                    cubeBlock.IterateBmm1(bmm1Buffers.Get(), runInfoCur, constInfo, c1Inputs);
                }
            }
            if (likely(hasBmm2)) {
                cubeBlock.PrefetchBmm2Inputs(c2Inputs, runInfoPre, constInfo);
                cubeBlock.IterateBmm2(bmm2Buffers.Get(), l1PBuffers, runInfoPre, constInfo, c2Inputs);
            }
        } else {
            if (likely(runInfoCur.isValid)) {
                Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U>& pBuf = l1PBuffers.GetVec();
                vecBlock.template ProcessVec1<0U>(pBuf, bmm1Buffers.Get(), runInfoCur, constInfo);
                if (likely(runInfoCur.actSingleLoopS2Size > S2_SPLIT)) {
                    vecBlock.template ProcessVec1<1U>(pBuf, bmm1Buffers.Get(), runInfoCur, constInfo);
                }
            }
            if (likely(hasBmm2)) {
                vecBlock.ProcessVec2(bmm2Buffers.Get(), runInfoPre, constInfo);
            }
        }
        if (likely(hasBmm2)) {
            runInfoPre.isValid = false;
        }
    }

    __aicore__ inline void DrainPendingTasks(uint64_t& loop, MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        while (HasValidTask(taskRunInfo)) {
            ExecuteTask(loop, taskRunInfo);
            ++loop;
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
        const uint32_t layoutAicCoreNum = regbasemx::GetMxMetadataAicCoreNum(metadataGm);
        uint32_t cachedBIdx = static_cast<uint32_t>(-1);
        uint32_t queryTokenBase = 0U;
        uint32_t actualS1Size = 0U;
        uint32_t actualS2Size = 0U;
        for (uint32_t sectionIdx = 0U; sectionIdx < sectionNum; ++sectionIdx) {
            // A metadata slot is not necessarily a launched physical core.
            // Visit every slot exactly once even for 64-slot CPU metadata on
            // 54 launched cores, or a small case that launches fewer cores.
            for (uint32_t metadataSlot = constInfo.aicIdx; metadataSlot < layoutAicCoreNum;
                 metadataSlot += constInfo.coreNum) {
                // [PLAN_A_PERF] Read enable first, then only the four MX bounds.
                // The stride is still the metadata layout count, NOT coreNum.
                const uint64_t metadataBase =
                    regbasemx::GetMxCoreMetadataOffset(sectionIdx, metadataSlot, layoutAicCoreNum);
                if (unlikely(metadataGm.GetValue(metadataBase + QBSA_FA_CORE_ENABLE_INDEX) == 0U)) {
                    continue;
                }
                const uint32_t bn1StartIdx = metadataGm.GetValue(metadataBase + QBSA_FA_BN1_START_INDEX);
                uint32_t bn1EndIdx = metadataGm.GetValue(metadataBase + QBSA_FA_BN1_END_INDEX);
                const uint32_t s1StartIdx = metadataGm.GetValue(metadataBase + QBSA_FA_S1_START_INDEX);
                const uint32_t s1EndIdx = metadataGm.GetValue(metadataBase + QBSA_FA_S1_END_INDEX);
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
                    // Reuse lengths across heads, slots and sections of a batch.
                    // This cache is local to ProcessMainLoop, never retained across launches.
                    if (unlikely(bIdx != cachedBIdx)) {
                        queryTokenBase = static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx));
                        actualS1Size = static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx + 1U)) - queryTokenBase;
                        actualS2Size = static_cast<uint32_t>(seqUsedKvGm.GetValue(bIdx));
                        cachedBIdx = bIdx;
                    }
                    const uint32_t s1LoopStart = GetS1LoopStart(bn1Idx, bn1StartIdx, s1StartIdx);
                    const uint32_t s1LoopEnd = GetS1LoopEnd(bn1Idx, bn1EndIdx, s1EndIdx, actualS1Size);
                    for (uint32_t s1OuterIdx = s1LoopStart; s1OuterIdx < s1LoopEnd; ++s1OuterIdx) {
                        // 当前 query block 的有效 sparse block 数。
                        const int32_t sparseLen = sparseSeqLenGm.GetValue(
                            (static_cast<uint64_t>(bIdx) * constInfo.realN2Size + n1Idx) * constInfo.maxQb +
                            s1OuterIdx);
                        if (unlikely(sparseLen <= 0)) {
                            if ASCEND_IS_AIV {
                                vecBlock.ClearEmptyQBlock(n1Idx, queryTokenBase, actualS1Size, s1OuterIdx, constInfo);
                            }
                            continue;
                        }
                        // sparseBase 指向 sparseIndices[B,N1,Qb,0]。
                        const uint64_t sparseBase = (static_cast<uint64_t>(bIdx) * constInfo.realN2Size + n1Idx) *
                                                        constInfo.maxQb * constInfo.maxKb +
                                                    static_cast<uint64_t>(s1OuterIdx) * constInfo.maxKb;
                        const uint32_t s2LoopEnd = static_cast<uint32_t>(sparseLen - 1);
                        if ASCEND_IS_AIV {
                            // One exact-length DMA per eligible Q row, reused by last-valid scan and M-parts.
                            indicesCache.Prepare(sparseIndicesGm, sparseBase, static_cast<uint32_t>(sparseLen),
                                                 constInfo.maxKb);
                        }
                        const uint32_t lastValidS2LoopIdx = GetLastValidS2LoopIdx(
                            sparseBase, static_cast<uint32_t>(sparseLen), sparseBlocksPerTask, actualS2Size);
                        const uint32_t mPartCount = (constInfo.qSparseBlockSize + M_BASE - 1U) / M_BASE;
                        bool hasValidQBlockTask = false;
                        for (uint32_t mPart = 0U; mPart < mPartCount; ++mPart) {
                            bool hasValidS2Task = false;
                            for (uint32_t s2LoopIdx = 0U; s2LoopIdx <= s2LoopEnd; s2LoopIdx += sparseBlocksPerTask) {
                                MxRunInfo& runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
                                const uint32_t remainSparseBlockCount = static_cast<uint32_t>(sparseLen) - s2LoopIdx;
                                const uint32_t sparseBlockCount = remainSparseBlockCount > sparseBlocksPerTask ?
                                                                      sparseBlocksPerTask :
                                                                      remainSparseBlockCount;
                                FillRunInfo(runInfo, loop, mLoop, bIdx, n1Idx, n2Idx, queryTokenBase, actualS1Size,
                                            mPart, actualS2Size, s1OuterIdx, s2LoopIdx, sparseBlockCount, sparseBase,
                                            !hasValidS2Task, s2LoopIdx == lastValidS2LoopIdx);
                                if (likely(runInfo.actSingleLoopS2Size != 0U && runInfo.actMSize != 0U)) {
                                    ExecuteTask(loop, taskRunInfo);
                                    hasValidS2Task = true;
                                    ++loop;
                                } else {
                                    runInfo.isValid = false;
                                }
                            }
                            // softmaxSum/Max use mLoop % (PRELOAD_N + 1). An empty M-part must not consume a slot,
                            // otherwise a later V1 can overwrite state that a delayed V2 still needs.
                            if (hasValidS2Task) {
                                hasValidQBlockTask = true;
                                ++mLoop;
                            }
                        }
                        // sparseSeqLen may contain only negative/out-of-range indices. Such a Q block also has no
                        // producer task and must follow the empty-row output contract.
                        if (unlikely(!hasValidQBlockTask)) {
                            if ASCEND_IS_AIV {
                                vecBlock.ClearEmptyQBlock(n1Idx, queryTokenBase, actualS1Size, s1OuterIdx, constInfo);
                            }
                        }
                    }
                }
            } // metadata slots assigned to this physical core
            if constexpr (QBSA_9102_MX_DIAG_DRAIN_SECTION != 0) {
                // Complete this core's queued tasks before entering the next section.
                // Preserve loop/mLoop, BufferPolicy cursors and all sync events: drain
                // bubbles advance the task clock, not the P/C1 producer/consumer cursors.
                // Do not add a global barrier: enabled sections may differ across cores.
                DrainPendingTasks(loop, taskRunInfo);
            }
        }
        // 排空尚未执行 C2/V2 的尾部 task。
        DrainPendingTasks(loop, taskRunInfo);
    }

    TPipe* pipe = nullptr;
    const QuantBlockSparseAttnMxTilingData* __restrict tilingData = nullptr;
    regbasemx::MxConstInfo constInfo;
    __gm__ uint8_t* queryPtr = nullptr;
    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    GlobalTensor<int32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> seqUsedKvGm;
    MxIndicesUbCache indicesCache;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    using Bmm1BuffersPolicy = BuffersPolicy3buff<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U>;
    Bmm1BuffersPolicy bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH, 1U> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD, 1U> l1PBuffers;
    QuantBlockSparseAttnMxBlockCube<INPUT_T, MM_T, OUTPUT_T, LAYOUT, KV_LAYOUT, s1TemplateType, s2TemplateType,
                                    dTemplateType, dVTemplateType, IS_PA, USE_DN>
        cubeBlock;
    QuantBlockSparseAttnMxBlockVec<INPUT_T, MM_T, OUTPUT_T, LAYOUT, KV_LAYOUT, s1TemplateType, s2TemplateType,
                                   dTemplateType, dVTemplateType, HAS_ATTEN, HAS_LSE, IS_PA, USE_DN>
        vecBlock;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
