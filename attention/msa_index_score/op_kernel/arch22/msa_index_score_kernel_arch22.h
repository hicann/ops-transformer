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
 * \file msa_index_score_kernel_arch22.h
 * \brief MsaIndexScore Atlas A2/A3 核函数：AIC 做 paged QKᵀ，AIV 做（可选反量化）+ mask + 分段 RowMax。
 *
 * 非量化：query/key 同为 bf16|fp16，score 侧不乘 scale。
 * int8 量化：key 为 int8；AIV 将 page DataCopy+Cast 到四槽 fp 暂存后通知 AIC 做 Mmad；
 * gather(st+4) 叠在 cube(st+1..st+3) 上；两 AIV 对分 8 page（每页仍串行 MTE2→V→MTE3）。
 * 同 AIV 上 MTE2∥MTE3 / 多 page 并发 MTE2 会写坏多 page int8。S 侧仍为 fp32 反量化。
 *
 * ENABLE_TOPK：score 仍走 L0C→GM，写到 workspace；全部核 SyncAll 后，AIV 用与 950 相同的
 * Sort32 TopK 写出 indices/values。A2/A3 没有 L0C→UB，也不做 fp8。
 */

#ifndef MSA_INDEX_SCORE_KERNEL_ARCH22_H
#define MSA_INDEX_SCORE_KERNEL_ARCH22_H

#include "kernel_operator.h"

#include "../catlass/msa_catlass.hpp"
#include "../catlass/arch/msa_arch.hpp"
#include "../catlass/arch/msa_resource.hpp"
#include "../catlass/arch/msa_cross_core_sync.hpp"
#include "../catlass/msa_coord.hpp"
#include "../catlass/msa_gemm_coord.hpp"
#include "../catlass/msa_matrix_coord.hpp"
#include "../catlass/layout/msa_layout.hpp"
#include "../catlass/gemm/msa_gemm_type.hpp"

#include "../msa_index_score_common.h"
#include "msa_block_mmad_arch22.h"
#include "../msa_index_score_task.h"
#include "msa_seg_row_max_epilogue_arch22.h"
#include "msa_index_score_topk_arch22.h"

namespace MsaIndexScoreNs {

enum class MsaBatchPacketPolicy : uint8_t {
    Disabled,
    Uniform
};

template <class ElementQ_, bool IS_QUANT, bool NATIVE_A8C8 = false, bool ENABLE_PAIRED_NATIVE = false,
          bool GROUP_MASK = false, bool WIDE_FIXED_SCORE = false, bool MIX_ONE_WIDE_POLICY = false,
          bool REQUEST_PLAN = false, MsaBatchPacketPolicy BATCH_PACKET_POLICY = MsaBatchPacketPolicy::Disabled,
          bool ENABLE_TOPK = false>
class MsaIndexScoreKernel {
public:
    static constexpr bool ENABLE_BATCH_PACKET = BATCH_PACKET_POLICY == MsaBatchPacketPolicy::Uniform;
    static_assert(!ENABLE_BATCH_PACKET || NATIVE_A8C8);
    using ElementQ = ElementQ_;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementQ, LayoutK>;
    using Epilogue = MsaSegRowMaxEpilogue<IS_QUANT, ENABLE_PAIRED_NATIVE, NATIVE_A8C8, GROUP_MASK, WIDE_FIXED_SCORE,
                                          MIX_ONE_WIDE_POLICY, ENABLE_TOPK>;
    using ElementS = typename Epilogue::ElementS;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmad = MsaBlockMmad<QType, KType, SType, MSA_L1B_STAGES_FP16, true, NATIVE_A8C8, NATIVE_A8C8>;

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static_assert(MSA_K_SCRATCH_STAGES_A2 == 4, "A2 int8 K scratch handshake assumes 4 slots");
    // int8：cast 放 epilogue 尾部。EVENT_ID2 每页成对 MTE2_V / V_MTE3 / MTE3_MTE2。
    // 非量化 TND 仍占用 epilogue S 区（gather 在 WaitS 之前，S 尚未载入）。
    static constexpr uint32_t UB_OFF_CAST_I8 = (IS_QUANT && !NATIVE_A8C8) ? Epilogue::UB_OFF_TAIL : 0U;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    static_assert(NATIVE_A8C8 || UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE, "cast UB out of bounds");
    // Legacy cast buffers live after the epilogue. Native padding instead
    // aliases the epilogue S buffer and is capacity-checked by the native epilogue layout.
    static_assert(NATIVE_A8C8 || !IS_QUANT || UB_OFF_CAST_I8 >= MsaSegRowMaxEpilogue<true>::UB_TOTAL,
                  "int8 cast UB overlaps epilogue");

    __aicore__ inline MsaIndexScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale, GM_ADDR actualSeqQlen,
                                GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score, GM_ADDR sparseIndices,
                                GM_ADDR sparseValues, GM_ADDR workspace,
                                const MsaIndexScoreTilingData* __restrict tiling, GM_ADDR queryScale = nullptr)
    {
        tiling_ = tiling;
        gQuery_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(query));
        if constexpr (IS_QUANT_V) {
            gKeyInt8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        } else {
            gKeyFp_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(key));
        }
        if constexpr (NATIVE_A8C8) {
            // A BBND physical page index multiplies the aligned page stride.
            // Its page address therefore has the same alignment as the base.
            allPagedKeysDirect_ = tiling_->keyLayout == MSA_KEY_LAYOUT_BBND &&
                                  tiling_->paddedHeadDim == tiling_->headDim &&
                                  tiling_->strideKvToken % MSA_DATABLOCK_BYTES == 0U &&
                                  tiling_->strideKvBlock % MSA_DATABLOCK_BYTES == 0U &&
                                  reinterpret_cast<uint64_t>(gKeyInt8_.GetPhyAddr()) % MSA_DATABLOCK_BYTES == 0U;
        }
        if constexpr (NATIVE_A8C8) {
            gKeyScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspace) + tiling_->kScratchOffsetBytes);
            gQueryScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workspace) + tiling_->qScratchOffsetBytes);
            gQueryScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryScale));
        } else if (IS_QUANT_V || MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            gKeyScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(reinterpret_cast<__gm__ float*>(workspace) +
                                                                            tiling_->kScratchOffsetElems));
        }
        if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
            gBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        }
        gScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
        gActualSeqQlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(actualSeqQlen));
        gActualSeqKlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(actualSeqKlen));
        gStartLoc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(startLoc));
        if constexpr (ENABLE_TOPK) {
            // 用户没有 score。归约结果写到 S/K scratch 之后的 fp32 workspace，再做 TopK。
            gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + tiling_->scoreWorkspaceElems);
            gSparseIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices));
            if (tiling_->returnChoice == MSA_RETURN_INDICES_VALUES) {
                gSparseValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sparseValues));
            }
        } else {
            (void)sparseIndices;
            (void)sparseValues;
            gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score));
        }
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS*>(workspace));

        uint32_t requestPlanTaskIdx = 0U;
        uint32_t requestPlanCores = 0U;
        if constexpr (REQUEST_PLAN) {
            if ((tiling_->headDim == 64U || tiling_->headDim == 96U || tiling_->headDim == MSA_K_TILE) &&
                tiling_->queryLayout == MSA_QUERY_LAYOUT_TND && tiling_->keyLayout == MSA_KEY_LAYOUT_BBND &&
                tiling_->pageSize == MSA_BLOCK_SIZE && tiling_->chunkBudget == 0U && tiling_->totalQ > 0U) {
                requestPlanCores = AscendC::GetBlockNum();
            }
            if ASCEND_IS_AIC {
                requestPlanTaskIdx = AscendC::GetBlockIdx();
            }
            if ASCEND_IS_AIV {
                requestPlanTaskIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            }
        }
        bool checkBatchPacketRows = false;
        if constexpr (ENABLE_BATCH_PACKET) {
            if ASCEND_IS_AIC {
                checkBatchPacketRows = allPagedKeysDirect_ && (tiling_->gqaGroup & MSA_PACKET_CANDIDATE) != 0U;
            }
        }
        scheduler_.template Init<REQUEST_PLAN>(
            tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch, tiling_->scoreBlockStride,
            tiling_->sparseMode, tiling_->initBlocks, tiling_->localBlocks, tiling_->keyLayout, tiling_->kvChunks,
            gActualSeqQlen_, gActualSeqKlen_, gStartLoc_, true, tiling_->qMaxSeq, tiling_->kMaxSeq, tiling_->tableWidth,
            0U, MSA_BLOCK_SIZE, tiling_->totalQ, 1U, 0U, requestPlanCores, requestPlanTaskIdx, checkBatchPacketRows);
        if constexpr (ENABLE_BATCH_PACKET) {
            batchPacket_ = scheduler_.BatchPacketRows();
        }
        if constexpr (NATIVE_A8C8 && GROUP_MASK && !ENABLE_PAIRED_NATIVE && !WIDE_FIXED_SCORE) {
            if ((tiling_->headDim == 64U || tiling_->headDim == 96U) && tiling_->queryLayout == MSA_QUERY_LAYOUT_TND &&
                tiling_->keyLayout == MSA_KEY_LAYOUT_BBND && tiling_->poolBlockSize == MSA_BLOCK_SIZE &&
                tiling_->returnChoice == MSA_RETURN_SCORE && tiling_->strideOutToken > 64U &&
                tiling_->strideOutToken <= 80U && tiling_->chunkBudget == 0U) {
                scheduler_.ConfigureTailWaveSplit(AscendC::GetBlockNum());
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessCube();
        }
        if ASCEND_IS_AIV {
            ProcessVector();
        }
    }

private:
    static constexpr uint32_t CAUSAL_SNAKE_CORES = 24U;
    static constexpr int32_t CAUSAL_SNAKE_MAX_Q = 8192;

    __aicore__ inline bool UseCausalSnake(uint32_t coreNum) const
    {
        if constexpr (NATIVE_A8C8 && GROUP_MASK && ENABLE_PAIRED_NATIVE) {
            return coreNum == CAUSAL_SNAKE_CORES && tiling_->headDim == MSA_K_TILE &&
                   tiling_->returnChoice == MSA_RETURN_SCORE && tiling_->queryLayout == MSA_QUERY_LAYOUT_TND &&
                   tiling_->keyLayout == MSA_KEY_LAYOUT_BBND && tiling_->pageSize == MSA_BLOCK_SIZE &&
                   tiling_->poolBlockSize == MSA_BLOCK_SIZE && tiling_->initBlocks == 0U &&
                   tiling_->localBlocks == 0U && scheduler_.IsSingleFullCausalPrefill(coreNum, CAUSAL_SNAKE_MAX_Q);
        }
        return false;
    }

    __aicore__ inline uint32_t CausalSnakeRank(uint32_t assignment, uint32_t coreIdx) const
    {
        const uint32_t wave = assignment / CAUSAL_SNAKE_CORES;
        const uint32_t lane = (wave & 1U) == 0U ? coreIdx : CAUSAL_SNAKE_CORES - 1U - coreIdx;
        return wave * CAUSAL_SNAKE_CORES + lane;
    }

    __aicore__ inline void ProcessCube()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK0{MSA_FLAG_K_READY};
        Catlass::Arch::CrossCoreFlag flagK1{MSA_FLAG_K_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK2{MSA_FLAG_K_READY_2};
        Catlass::Arch::CrossCoreFlag flagK3{MSA_FLAG_K_READY_3};

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint32_t kStages = IS_QUANT_V ? MSA_K_SCRATCH_STAGES_A2 : 1U;
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * kStages * MSA_K_SCRATCH_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        const bool causalSnake = UseCausalSnake(coreNum);
        const uint32_t leadPad =
            causalSnake ? (CAUSAL_SNAKE_CORES - totalTasks % CAUSAL_SNAKE_CORES) % CAUSAL_SNAKE_CORES : 0U;
        const uint32_t assignmentEnd = totalTasks + leadPad;
        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t assignment = coreIdx; assignment < assignmentEnd; assignment += coreNum) {
            const uint32_t rank = causalSnake ? CausalSnakeRank(assignment, coreIdx) : assignment;
            if (rank < leadPad) {
                continue;
            }
            scheduler_.Decode(rank - leadPad, task);
            // block_table 按 256B 预取，盖住后续逐页 GetValue 的 HBM 延迟。
            PreloadBlockTable(task);
            // 只对可见 S-tile 做 QKᵀ + 握手；因果不可见尾由 AIV 直接写 -inf，避免空转同步。
            bool needLoadQ = true;
            for (uint32_t st = task.sStileBegin; st < task.sStileEnd; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                if (needKScratch) {
                    // MIX 1AIC:2AIV：AIV→AIC 的 0x2 flag 需两个 AIV 都 Set 后才放行。
                    // 绑 PIPE_MTE2，避免 PIPE_ALL 把上一 stile 的 cube/fixpipe 冲掉。
                    WaitKScratchReady(flagK0, flagK1, flagK2, flagK3, st);
                }
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                ComputeSTile(blockMmad, task, st * MSA_BLOCKS_PER_STILE, sBase, KScratchSlot(coreKScratch, st),
                             needLoadQ);
                needLoadQ = false;
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagSReady);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (ENABLE_TOPK) {
            // Cube 与 Vector 都要进 SyncAll，否则 AIV 读 score 时 AIC 可能还没写完。
            AscendC::SyncAll<false>();
        }
    }

    __aicore__ inline uint64_t KScratchSlot(uint64_t coreKScratch, uint32_t st) const
    {
        if constexpr (IS_QUANT_V) {
            return coreKScratch + static_cast<uint64_t>(st % MSA_K_SCRATCH_STAGES_A2) * MSA_K_SCRATCH_ELEM_NUM;
        }
        return coreKScratch;
    }

    __aicore__ inline void WaitKScratchReady(Catlass::Arch::CrossCoreFlag& flagK0, Catlass::Arch::CrossCoreFlag& flagK1,
                                             Catlass::Arch::CrossCoreFlag& flagK2, Catlass::Arch::CrossCoreFlag& flagK3,
                                             uint32_t st)
    {
        if constexpr (IS_QUANT_V) {
            const uint32_t slot = st % MSA_K_SCRATCH_STAGES_A2;
            if (slot == 0U) {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK0);
            } else if (slot == 1U) {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK1);
            } else if (slot == 2U) {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK2);
            } else {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK3);
            }
        } else {
            (void)flagK1;
            (void)flagK2;
            (void)flagK3;
            Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK0);
        }
    }

    __aicore__ inline bool KeyBlockNeedsScratch(const MsaTask& task, uint32_t blk) const
    {
        if constexpr (NATIVE_A8C8) {
            if (allPagedKeysDirect_) {
                return false;
            }
        }
        if constexpr (NATIVE_A8C8) {
            // A complete C0-aligned INT8 page can use the same GM -> L1
            // prefetch path as non-quantized K. Keep padding and partial
            // sequence pages in scratch so Cube never reads beyond a request.
            const bool aligned =
                tiling_->paddedHeadDim == tiling_->headDim && tiling_->strideKvToken % MSA_DATABLOCK_BYTES == 0U &&
                (!MsaIsPaKeyLayout(tiling_->keyLayout) || tiling_->strideKvBlock % MSA_DATABLOCK_BYTES == 0U);
            if (!aligned || KeyBlockValidTokens(task, blk) != MSA_BLOCK_SIZE) {
                return true;
            }
            const uint64_t address = reinterpret_cast<uint64_t>(gKeyInt8_.GetPhyAddr()) + KeyBlockGmOffset(task, blk);
            return address % MSA_DATABLOCK_BYTES != 0U;
        } else if constexpr (IS_QUANT_V) {
            return true;
        }
        if (!MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            return false;
        }
        // TND fp16：完整 128-token 页与 BBND 一样直接从 packed GM mmad。
        // 仅最后一条请求的尾页可能越过 T2，需要 AIV 填零后再走 scratch。
        const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
        return (nValidTok > 0) && (nValidTok < MSA_BLOCK_SIZE) && (task.batchIdx + 1U == tiling_->batch);
    }

    __aicore__ inline bool StileNeedsKScratch(const MsaTask& task, uint32_t blkBase) const
    {
        if constexpr (NATIVE_A8C8) {
            if (allPagedKeysDirect_) {
                return false;
            }
        }
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (KeyBlockNeedsScratch(task, blk)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint32_t KeyBlockValidTokens(const MsaTask& task, uint32_t blk) const
    {
        if (!MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            return MSA_BLOCK_SIZE;
        }
        const uint32_t seqTok = blk * tiling_->poolBlockSize;
        if (task.kvLen <= 0 || seqTok >= static_cast<uint32_t>(task.kvLen)) {
            return 0;
        }
        const uint32_t seqRemain = static_cast<uint32_t>(task.kvLen) - seqTok;
        return seqRemain < MSA_BLOCK_SIZE ? seqRemain : MSA_BLOCK_SIZE;
    }

    /// score 末维是 16 对齐后的输出宽度。分页页表的行距是 tableWidth，不能用 score 末维。
    __aicore__ inline uint32_t BlockTableRowStride() const
    {
        return tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
    }

    /// PA block_table 一行可能有数百个 int32。dc_preload 每次 256B，在 QK 开始前发出。
    /// 只发完整落在 block_table 分配内的 256B；短尾不再补一笔，避免探出 GM 分配。
    __aicore__ inline void PreloadBlockTable(const MsaTask& task)
    {
        if (MsaIsSeqKeyLayout(tiling_->keyLayout) || task.sBlkEnd <= task.sBlkBegin) {
            return;
        }
        const uint32_t rowStride = BlockTableRowStride();
        if (rowStride == 0U || task.batchIdx >= tiling_->batch) {
            return;
        }
        const uint64_t elemOff = static_cast<uint64_t>(task.batchIdx) * rowStride + task.sBlkBegin;
        const uint64_t tensorElems = static_cast<uint64_t>(tiling_->batch) * rowStride;
        if (elemOff >= tensorElems) {
            return;
        }
        const uint64_t availBytes = (tensorElems - elemOff) * sizeof(int32_t);
        const uint64_t sliceBytes = static_cast<uint64_t>(task.sBlkEnd - task.sBlkBegin) * sizeof(int32_t);
        // 分配基址按 8B 对齐。elemOff 为奇数时起点只 4B 对齐，dc_preload 需要 uint64 指针，先跳过 1 个 int32。
        const uint32_t leadElems = static_cast<uint32_t>(elemOff & 1ULL);
        const uint64_t leadBytes = static_cast<uint64_t>(leadElems) * sizeof(int32_t);
        if (leadBytes >= availBytes || leadBytes >= sliceBytes) {
            return;
        }
        const uint64_t alignedAvail = availBytes - leadBytes;
        uint64_t cover = sliceBytes - leadBytes;
        if (cover > alignedAvail) {
            cover = alignedAvail;
        }
        const __gm__ int32_t* alignedRow = gBlockTable_.GetPhyAddr() + (elemOff + leadElems);
        auto* base = reinterpret_cast<__gm__ uint64_t*>(const_cast<__gm__ int32_t*>(alignedRow));
        constexpr uint64_t kPreloadBytes = 256ULL;
        for (uint64_t off = 0; off < cover; off += kPreloadBytes) {
            if (off + kPreloadBytes > alignedAvail) {
                break;
            }
            dc_preload(base, static_cast<int64_t>(off));
        }
    }

    __aicore__ inline uint64_t KeyBlockGmOffset(const MsaTask& task, uint32_t blk) const
    {
        if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            return MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, blk * tiling_->poolBlockSize,
                                      tiling_->numKvHeads, tiling_->headDim, tiling_->kMaxSeq);
        }
        const int32_t page = gBlockTable_.GetValue(task.batchIdx * BlockTableRowStride() + blk);
        return static_cast<uint64_t>(page) * tiling_->strideKvBlock;
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbPartial(const AscendC::LocalTensor<T>& ub, const AscendC::GlobalTensor<T>& gm,
                                             uint32_t nElem)
    {
        if (nElem == 0) {
            return;
        }
        const uint32_t bytes = nElem * static_cast<uint32_t>(sizeof(T));
        if ((bytes % MSA_DATABLOCK_BYTES) == 0U) {
            AscendC::DataCopy(ub, gm, nElem);
            return;
        }
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = bytes;
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = 0;
        AscendC::DataCopyPad(ub, gm, params, pad);
    }

    // A8C8-only member implementations.
#include "msa_index_score_a8c8_kernel_arch22.h"

    /// int8 gather：单 i8 + 单 fp16，每页 MTE2→V→MTE3 串行。subIdx/subBlockNum 对分 page。
    __aicore__ inline void GatherKeySTileQuantPiped(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                    uint32_t blkBase, uint64_t kScratchBase, uint32_t subIdx,
                                                    uint32_t subBlockNum)
    {
        AscendC::LocalTensor<int8_t> ubI8 = resource.ubBuf.template GetBufferByByte<int8_t>(UB_OFF_CAST_I8);
        AscendC::LocalTensor<half> ubHalf = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        const uint32_t split = (subBlockNum == 0U) ? 1U : subBlockNum;

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            if ((j % split) != subIdx) {
                continue;
            }
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                continue;
            }
            if (!KeyBlockNeedsScratch(task, blk)) {
                continue;
            }
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
            const uint32_t nValid = nValidTok * headDim;

            uint32_t copyN = nElem;
            if (tiling_->keyLayout == MSA_KEY_LAYOUT_TND) {
                const uint32_t tokenStart = task.cuKStart + blk * tiling_->poolBlockSize;
                const uint32_t packedEnd = tokenStart + MSA_BLOCK_SIZE;
                if (tiling_->totalK > 0 && packedEnd > tiling_->totalK) {
                    copyN = nValid;
                }
            }
            if (copyN < nElem) {
                AscendC::Duplicate(ubHalf, static_cast<half>(0), nElem);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
            if (copyN > 0) {
                if (copyN == nElem) {
                    AscendC::DataCopy(ubI8, gKeyInt8_[kOffset], nElem);
                } else {
                    CopyGmToUbPartial(ubI8, gKeyInt8_[kOffset], copyN);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::Cast(ubHalf, ubI8, AscendC::RoundMode::CAST_NONE, copyN);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopy(gKeyScratch_[scratchOff], ubHalf, nElem);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
    }

    /// AIV：把本 S-tile 可见 block 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    /// int8：两 AIV 按 j%subBlockNum 对分 page；非量化仍由调用方只让 AIV0 进来。
    __aicore__ inline void GatherKeySTileToScratch(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                   uint32_t blkBase, uint64_t kScratchBase, uint32_t subIdx,
                                                   uint32_t subBlockNum)
    {
        if constexpr (NATIVE_A8C8) {
            GatherNativeKey(resource, task, blkBase, kScratchBase, subIdx, subBlockNum);
        } else {
            if constexpr (IS_QUANT_V) {
                GatherKeySTileQuantPiped(resource, task, blkBase, kScratchBase, subIdx, subBlockNum);
                return;
            }
            AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
            const uint32_t headDim = tiling_->headDim;
            const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
            const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
            const uint32_t split = (subBlockNum == 0U) ? 1U : subBlockNum;

            for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                if ((j % split) != subIdx) {
                    continue;
                }
                const uint32_t blk = blkBase + j;
                if (blk >= task.visibleEndBlk) {
                    continue;
                }
                if (!KeyBlockNeedsScratch(task, blk)) {
                    continue;
                }
                const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
                const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
                const uint32_t nValid = nValidTok * headDim;

                if (nValid < nElem) {
                    AscendC::Duplicate(ubK, static_cast<ElementQ>(0), nElem);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                if (nValid > 0) {
                    CopyGmToUbPartial(ubK, gKeyFp_[kOffset], nValid);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopy(gKeyScratch_[scratchOff], ubK, nElem);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void NotifyKScratchReady(Catlass::Arch::CrossCoreFlag& flagK0,
                                               Catlass::Arch::CrossCoreFlag& flagK1,
                                               Catlass::Arch::CrossCoreFlag& flagK2,
                                               Catlass::Arch::CrossCoreFlag& flagK3, uint32_t st)
    {
        if constexpr (IS_QUANT_V) {
            const uint32_t slot = st % MSA_K_SCRATCH_STAGES_A2;
            if (slot == 0U) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK0);
            } else if (slot == 1U) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK1);
            } else if (slot == 2U) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK2);
            } else {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK3);
            }
        } else {
            (void)flagK1;
            (void)flagK2;
            (void)flagK3;
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK0);
        }
    }

    __aicore__ inline void IssueKGatherAndNotify(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                 uint32_t st, uint64_t coreKScratch, uint32_t subIdx,
                                                 uint32_t subBlockNum, Catlass::Arch::CrossCoreFlag& flagK0,
                                                 Catlass::Arch::CrossCoreFlag& flagK1,
                                                 Catlass::Arch::CrossCoreFlag& flagK2,
                                                 Catlass::Arch::CrossCoreFlag& flagK3)
    {
        if constexpr (IS_QUANT_V) {
            GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, KScratchSlot(coreKScratch, st), subIdx,
                                    subBlockNum);
        } else if (subIdx == 0U) {
            GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, KScratchSlot(coreKScratch, st), 0U, 1U);
        }
        NotifyKScratchReady(flagK0, flagK1, flagK2, flagK3, st);
    }

    __aicore__ inline void ComputeSTile(BlockMmad& blockMmad, const MsaTask& task, uint32_t blkBase, uint64_t sBase,
                                        uint64_t kScratchBase, bool needLoadL1)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = NATIVE_A8C8 ? tiling_->paddedHeadDim : tiling_->headDim;
        const bool queryPadded = NATIVE_A8C8 && tiling_->qScratchBytesPerCore != 0U;
        const LayoutQ layoutQ(mActual, headDim, queryPadded ? headDim : tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const LayoutS layoutS(mActual, MSA_BLOCK_SIZE, STILE_WIDTH);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const uint64_t qOffset =
            MsaQueryTileElem(tiling_->queryLayout, task.batchIdx, task.cuQStart, task.mStart, tiling_->numQHeads,
                             tiling_->headDim, tiling_->qMaxSeq, tiling_->strideQn);
        auto queryTile = gQuery_[qOffset];
        if constexpr (NATIVE_A8C8) {
            if (queryPadded) {
                queryTile =
                    gQueryScratch_[static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling_->qScratchBytesPerCore];
            }
        }
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        // Full native tiles keep the eight-page INT32 S layout. Two adjacent
        // physical C regions share one Fixpipe; ring indices persist across tiles.
        if constexpr (NATIVE_A8C8 && ENABLE_PAIRED_NATIVE) {
            static_assert(BlockMmad::STAGES >= 2, "paired Fixpipe requires at least two existing L1 slots");
            if (mActual == MSA_ROW_TILE_M && headDim == MSA_K_TILE) {
                ComputeNativePairedSTile(blockMmad, task, blkBase, sBase, kScratchBase, needLoadL1, queryTile, layoutQ,
                                         layoutKScratch, layoutKGm, layoutS, shape);
                return;
            }
        }
        // headDim<=128 时一页 K 正好一块 L1。先把本 stile 的页号收齐，再让最多 3 页 MTE2 在飞，
        // GetValue 不再卡在每一页的 WaitFlag(MTE2) 前面。
        // 其余满 128 行场景保留单页 BlockMmad；小 M 场景沿用逐页消费。
        const bool pipeK =
            (headDim <= MSA_K_TILE) && (mActual > 0U) && (mActual <= 32U || (ENABLE_BATCH_PACKET && batchPacket_));
        if (!pipeK) {
            bool pairPending = false;
            for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                const uint32_t blk = blkBase + j;
                if (blk >= task.visibleEndBlk) {
                    break;
                }
                if constexpr (NATIVE_A8C8 && !ENABLE_PAIRED_NATIVE && (GROUP_MASK || WIDE_FIXED_SCORE)) {
                    const bool eligiblePair =
                        mActual == MSA_ROW_TILE_M && headDim == tiling_->headDim && (headDim == 64U || headDim == 96U);
                    const bool completePair = pairPending;
                    const bool startPair = eligiblePair && !pairPending && j + 1U < MSA_BLOCKS_PER_STILE &&
                                           blk + 1U < task.visibleEndBlk && blockMmad.CanConsumePair();
                    if (startPair || completePair) {
                        const uint32_t outputPage = completePair ? j - 1U : j;
                        auto pairedOutput = gWorkspace_[sBase + static_cast<uint64_t>(outputPage) * MSA_BLOCK_SIZE];
                        if (KeyBlockNeedsScratch(task, blk)) {
                            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
                            blockMmad.template operator()<true>(queryTile, layoutQ, gKeyScratch_[scratchOff],
                                                                layoutKScratch, pairedOutput, layoutS, shape,
                                                                needLoadL1, completePair);
                        } else {
                            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                            blockMmad.template operator()<true>(queryTile, layoutQ, gKeyInt8_[kOffset], layoutKGm,
                                                                pairedOutput, layoutS, shape, needLoadL1, completePair);
                        }
                        pairPending = startPair;
                        needLoadL1 = false;
                        continue;
                    }
                }
                if (KeyBlockNeedsScratch(task, blk)) {
                    const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
                    blockMmad(queryTile, layoutQ, gKeyScratch_[scratchOff], layoutKScratch,
                              gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape,
                              needLoadL1);
                } else if constexpr (NATIVE_A8C8) {
                    const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                    blockMmad(queryTile, layoutQ, gKeyInt8_[kOffset], layoutKGm,
                              gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape,
                              needLoadL1);
                } else if constexpr (!IS_QUANT_V) {
                    const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                    blockMmad(queryTile, layoutQ, gKeyFp_[kOffset], layoutKGm,
                              gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape,
                              needLoadL1);
                }
                needLoadL1 = false;
            }
            return;
        }

        uint64_t pageOff[MSA_BLOCKS_PER_STILE];
        uint8_t pageScratch[MSA_BLOCKS_PER_STILE];
        uint32_t nPage = 0;
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            pageScratch[nPage] = KeyBlockNeedsScratch(task, blk) ? 1U : 0U;
            if (pageScratch[nPage] != 0U) {
                pageOff[nPage] = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            } else if constexpr (NATIVE_A8C8 || !IS_QUANT_V) {
                pageOff[nPage] = KeyBlockGmOffset(task, blk);
            } else {
                pageOff[nPage] = 0;
            }
            ++nPage;
        }
        if (nPage == 0U) {
            return;
        }
        if (needLoadL1) {
            blockMmad.LoadQ(queryTile, layoutQ, shape);
        }
        const uint32_t depth = (nPage < BlockMmad::STAGES) ? nPage : BlockMmad::STAGES;
        for (uint32_t i = 0; i < depth; ++i) {
            IssuePageK(blockMmad, pageScratch[i], pageOff[i], shape, layoutKScratch, layoutKGm);
        }
        if constexpr (ENABLE_BATCH_PACKET) {
            if (batchPacket_) {
                for (uint32_t first = 0; first < nPage; first += 2U) {
                    const uint32_t pages = Min(2U, nPage - first);
                    blockMmad.BeginBatchPacket();
                    for (uint32_t i = first; i < first + pages; ++i) {
                        blockMmad.ConsumeBatchPacketPage(shape, i - first);
                        const uint32_t nxt = i + depth;
                        if (nxt < nPage) {
                            IssuePageK(blockMmad, pageScratch[nxt], pageOff[nxt], shape, layoutKScratch, layoutKGm);
                        }
                    }
                    auto output = gWorkspace_[sBase + static_cast<uint64_t>(first) * MSA_BLOCK_SIZE];
                    if (pages == 2U) {
                        blockMmad.template PublishBatchPacket<2U>(output, layoutS, mActual);
                    } else {
                        blockMmad.template PublishBatchPacket<1U>(output, layoutS, mActual);
                    }
                }
                return;
            }
        }
        for (uint32_t i = 0; i < nPage; ++i) {
            blockMmad.ConsumeK(gWorkspace_[sBase + static_cast<uint64_t>(i) * MSA_BLOCK_SIZE], layoutS, shape);
            const uint32_t nxt = i + depth;
            if (nxt < nPage) {
                IssuePageK(blockMmad, pageScratch[nxt], pageOff[nxt], shape, layoutKScratch, layoutKGm);
            }
        }
    }

    __aicore__ inline void IssuePageK(BlockMmad& blockMmad, uint8_t scratch, uint64_t off,
                                      const Catlass::GemmCoord& shape, const LayoutK& layoutKScratch,
                                      const LayoutK& layoutKGm)
    {
        if (scratch != 0U) {
            blockMmad.IssueK(gKeyScratch_[off], layoutKScratch, shape);
        } else if constexpr (NATIVE_A8C8) {
            blockMmad.IssueK(gKeyInt8_[off], layoutKGm, shape);
        } else if constexpr (!IS_QUANT_V) {
            blockMmad.IssueK(gKeyFp_[off], layoutKGm, shape);
        }
    }

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        Epilogue epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->tableWidth, tiling_->strideScalePage,
                      tiling_->strideScaleHead, IS_QUANT_V && !NATIVE_A8C8, tiling_->keyLayout, tiling_->totalK,
                      gScore_, gScale_, gBlockTable_);
        if constexpr (NATIVE_A8C8) {
            epilogue.SetQueryScale(gQueryScale_);
            if constexpr (MIX_ONE_WIDE_POLICY) {
                epilogue.SetWideStageAllowed(tiling_->headDim == MSA_K_TILE);
            }
        }
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK0{MSA_FLAG_K_READY};
        Catlass::Arch::CrossCoreFlag flagK1{MSA_FLAG_K_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK2{MSA_FLAG_K_READY_2};
        Catlass::Arch::CrossCoreFlag flagK3{MSA_FLAG_K_READY_3};

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint32_t kStages = IS_QUANT_V ? MSA_K_SCRATCH_STAGES_A2 : 1U;
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * kStages * MSA_K_SCRATCH_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        bool allowWideGroupStage = false;
        bool allowNarrowGroupedStore = false;
        bool allowTwoWindowCompactConsume = false;
        if constexpr (Epilogue::WIDE_GROUP_SCORE) {
            const bool groupedScoreLayout =
                tiling_->batch == 1U && (tiling_->headDim == 64U || tiling_->headDim == 96U) &&
                tiling_->returnChoice == MSA_RETURN_SCORE && tiling_->queryLayout == MSA_QUERY_LAYOUT_TND &&
                tiling_->keyLayout == MSA_KEY_LAYOUT_BBND && tiling_->pageSize == MSA_BLOCK_SIZE &&
                tiling_->poolBlockSize == MSA_BLOCK_SIZE && tiling_->sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN &&
                tiling_->initBlocks == 0U && tiling_->localBlocks == 0U;
            const bool wideScore = tiling_->strideOutToken > Epilogue::MSA_STAGE_BLOCKS;
            allowWideGroupStage = groupedScoreLayout && wideScore;
            allowNarrowGroupedStore = groupedScoreLayout && !wideScore;
            constexpr uint32_t compactScoreLimit = Epilogue::MSA_STAGE_BLOCKS + 2U * MSA_BLOCKS_PER_STILE;
            allowTwoWindowCompactConsume = allowWideGroupStage && tiling_->strideOutToken <= compactScoreLimit;
        }

        bool allowBatchedCompact = false;
        if constexpr (WIDE_FIXED_SCORE) {
            constexpr uint32_t compactScoreLimit = Epilogue::MSA_STAGE_BLOCKS + 2U * MSA_BLOCKS_PER_STILE;
            allowBatchedCompact =
                tiling_->batch > 1U && (tiling_->headDim == 64U || tiling_->headDim == 96U) &&
                tiling_->returnChoice == MSA_RETURN_SCORE && tiling_->queryLayout == MSA_QUERY_LAYOUT_TND &&
                tiling_->keyLayout == MSA_KEY_LAYOUT_BBND && tiling_->pageSize == MSA_BLOCK_SIZE &&
                tiling_->poolBlockSize == MSA_BLOCK_SIZE && tiling_->sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN &&
                tiling_->initBlocks == 0U && tiling_->localBlocks == 0U &&
                tiling_->strideOutToken > Epilogue::MSA_STAGE_BLOCKS && tiling_->strideOutToken <= compactScoreLimit;
        }

        const bool causalSnake = UseCausalSnake(coreNum);
        const uint32_t leadPad =
            causalSnake ? (CAUSAL_SNAKE_CORES - totalTasks % CAUSAL_SNAKE_CORES) % CAUSAL_SNAKE_CORES : 0U;
        const uint32_t assignmentEnd = totalTasks + leadPad;
        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t assignment = coreIdx; assignment < assignmentEnd; assignment += coreNum) {
            const uint32_t rank = causalSnake ? CausalSnakeRank(assignment, coreIdx) : assignment;
            if (rank < leadPad) {
                continue;
            }
            scheduler_.Decode(rank - leadPad, task);
            epilogue.BeginTask(task, subIdx, subBlockNum, allowWideGroupStage || allowBatchedCompact);
            const bool useNarrowGroupedScore =
                allowNarrowGroupedStore && task.qLen >= static_cast<int32_t>(MSA_WIDE_SCORE_MIN_Q_TOKENS);
            const bool useUnifiedConsumer =
                useNarrowGroupedScore || (allowTwoWindowCompactConsume && epilogue.ScoreStaged()) ||
                (allowBatchedCompact && task.qLen >= 32 && epilogue.SubRows() >= 32U && epilogue.ScoreStaged());
            if constexpr (NATIVE_A8C8) {
                // Both AIVs finish their own rows before publishing the first K_READY.
                if (task.sStileBegin < task.sStileEnd && tiling_->qScratchBytesPerCore != 0U) {
                    PadNativeQuery(resource, task, coreIdx, subIdx, subBlockNum);
                }
            }
            if constexpr (IS_QUANT_V) {
                // Native KV chunks use exactly the Cube's global stile range. In particular,
                // a chunk need not begin at slot zero; K scratch slots retain global st % 4.
                // Legacy quantization keeps its existing range and numerical path.
                const uint32_t stBegin = NATIVE_A8C8 ? task.sStileBegin : 0U;
                const uint32_t stEnd = NATIVE_A8C8 ? task.sStileEnd : task.numComputeSTiles;
                const uint32_t nSt = stEnd - stBegin;
                uint32_t nPref = (nSt < MSA_K_SCRATCH_STAGES_A2) ? nSt : MSA_K_SCRATCH_STAGES_A2;
                if constexpr (NATIVE_A8C8) {
                    if (allPagedKeysDirect_) {
                        nPref = 0U;
                    }
                }
                for (uint32_t p = 0; p < nPref; ++p) {
                    const uint32_t st = stBegin + p;
                    if (StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, st, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                }
                if constexpr (NATIVE_A8C8) {
                    // Publish the initial K scratch tiles before sharing MTE2
                    // with the first scale cache. S16 still waits on S_READY.
                    epilogue.PrefetchFirstScales(task);
                }
                for (uint32_t st = stBegin; st < stEnd; ++st) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                    const uint32_t stPref = st + MSA_K_SCRATCH_STAGES_A2;
                    const uint64_t sBase =
                        coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                    if (stPref < stEnd && StileNeedsKScratch(task, stPref * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, stPref, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                    epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE, useUnifiedConsumer);
                    ++tileSeq;
                }
            } else {
                for (uint32_t st = task.sStileBegin; st < task.sStileEnd; ++st) {
                    if (StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, st, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                    const uint64_t sBase =
                        coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                    epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE);
                    ++tileSeq;
                }
            }
            // 因果不可见尾不握手。窄 score 仍逐 tile 写进暂存窗；宽 score 的 -inf 尾
            // 按 chunk 连续写回，避免 8 列循环把墙钟拉到可见 KV 计算量的数倍。
            if (epilogue.ScoreStaged()) {
                if constexpr ((NATIVE_A8C8 && ENABLE_PAIRED_NATIVE) || WIDE_FIXED_SCORE || Epilogue::WIDE_GROUP_SCORE) {
                    if (!epilogue.TryFinishStagedTail(task, useNarrowGroupedScore) && task.writeTailFill) {
                        for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                            epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE);
                        }
                    }
                } else {
                    if (task.writeTailFill) {
                        for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                            epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE);
                        }
                    }
                }
            } else {
                epilogue.FillInvalidTail(task);
            }
            // Narrow stages flush only at EndTask. Keep window-advance and tail handling unchanged.
            epilogue.EndTask(useNarrowGroupedScore);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (ENABLE_TOPK) {
            AscendC::SyncAll<false>();
            MsaIndexScoreTopk<> topk;
            auto ubTrunk = resource.ubBuf.template GetBufferByByte<float>(0);
            topk.Run(tiling_, gScore_, gSparseIndices_, gSparseValues_, gActualSeqQlen_, gActualSeqKlen_,
                     tiling_->returnChoice == MSA_RETURN_INDICES_VALUES, ubTrunk);
        }
    }

    const MsaIndexScoreTilingData* __restrict tiling_ = nullptr;
    bool allPagedKeysDirect_ = false;
    bool batchPacket_ = false;
    MsaTaskScheduler<MSA_BLOCK_SIZE> scheduler_;

    AscendC::GlobalTensor<ElementQ> gQuery_;
    AscendC::GlobalTensor<int8_t> gQueryScratch_;
    AscendC::GlobalTensor<float> gQueryScale_;
    AscendC::GlobalTensor<ElementQ> gKeyFp_;
    AscendC::GlobalTensor<int8_t> gKeyInt8_;
    AscendC::GlobalTensor<ElementQ> gKeyScratch_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gActualSeqQlen_;
    AscendC::GlobalTensor<int32_t> gActualSeqKlen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;
    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<int32_t> gSparseIndices_;
    AscendC::GlobalTensor<float> gSparseValues_;
    AscendC::GlobalTensor<ElementS> gWorkspace_;
};

static_assert(MSA_TOPK_LEN_FLOAT_OFF_A2 >= 14336U, "A2 topk length table overlaps the sort buffer");
static_assert(MSA_TOPK_META_FLOAT_OFF_A2 + MSA_TOPK_META_CAP * 3U <= (192U * 1024U) / sizeof(float),
              "A2 topk metadata exceeds 192KB UB");

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_KERNEL_ARCH22_H
