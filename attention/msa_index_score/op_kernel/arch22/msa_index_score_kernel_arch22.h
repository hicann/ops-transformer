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

namespace MsaIndexScoreNs {

template <class ElementQ_, bool IS_QUANT>
class MsaIndexScoreKernel {
public:
    using ElementQ = ElementQ_;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementQ, LayoutK>;
    // S workspace 元素：非量化 fp16（fixpipe F322F16 直接写，写读流量减半），int8 fp32
    // （S 幅值大、1e-3 容差下 fp16 精度余量不足 15%，保持 fp32 反量化）。
    using ElementS = std::conditional_t<IS_QUANT, float, half>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;

    // int8：三槽 K scratch 后与非量化同样 3 级 L1B + unit flag（单槽时 3 级会与下一 stile
    // 的 cast 重写竞态，950 arch35 仍保持 2 级 / 关 unit flag）。
    using BlockMmad = MsaBlockMmad<QType, KType, SType, MSA_L1B_STAGES_FP16, true>;

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static_assert(MSA_K_SCRATCH_STAGES_A2 == 4, "A2 int8 K scratch handshake assumes 4 slots");
    // int8：cast 放 epilogue 尾部。EVENT_ID2 每页成对 MTE2_V / V_MTE3 / MTE3_MTE2。
    // 非量化 TND 仍占用 epilogue S 区（gather 在 WaitS 之前，S 尚未载入）。
    static constexpr uint32_t UB_OFF_CAST_I8 = IS_QUANT ? MsaSegRowMaxEpilogue<true>::UB_OFF_TAIL : 0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    static_assert(UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE, "cast UB out of bounds");
    static_assert(!IS_QUANT || UB_OFF_CAST_I8 >= MsaSegRowMaxEpilogue<true>::UB_TOTAL,
                  "int8 cast UB overlaps epilogue");

    __aicore__ inline MsaIndexScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale, GM_ADDR actualSeqQlen,
                                GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score, GM_ADDR workspace,
                                const MsaIndexScoreTilingData* __restrict tiling)
    {
        tiling_ = tiling;
        gQuery_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(query));
        if constexpr (IS_QUANT_V) {
            gKeyInt8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        } else {
            gKeyFp_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(key));
        }
        if (IS_QUANT_V || MsaIsSeqKeyLayout(tiling_->keyLayout)) {
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
        gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score));
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS*>(workspace));

        scheduler_.Init(tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch, tiling_->scoreBlockStride,
                        tiling_->sparseMode, tiling_->initBlocks, tiling_->localBlocks, tiling_->keyLayout,
                        tiling_->kvChunks, gActualSeqQlen_, gActualSeqKlen_, gStartLoc_, true, tiling_->qMaxSeq,
                        tiling_->kMaxSeq, tiling_->tableWidth);
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

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
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
        if constexpr (IS_QUANT_V) {
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
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const LayoutS layoutS(mActual, MSA_BLOCK_SIZE, STILE_WIDTH);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const uint64_t qOffset = MsaQueryTileElem(tiling_->queryLayout, task.batchIdx, task.cuQStart, task.mStart,
                                                  tiling_->numQHeads, headDim, tiling_->qMaxSeq, tiling_->strideQn);
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        // headDim<=128 时一页 K 正好一块 L1。先把本 stile 的页号收齐，再让最多 3 页 MTE2 在飞，
        // GetValue 不再卡在每一页的 WaitFlag(MTE2) 前面。
        // 满 128 行的 prefill 沿用原来的单页 BlockMmad。M=128 时 3 级预取更慢。
        const bool pipeK = (headDim <= MSA_K_TILE) && (mActual > 0U) && (mActual <= 32U);
        if (!pipeK) {
            for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                const uint32_t blk = blkBase + j;
                if (blk >= task.visibleEndBlk) {
                    break;
                }
                if (KeyBlockNeedsScratch(task, blk)) {
                    const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
                    blockMmad(gQuery_[qOffset], layoutQ, gKeyScratch_[scratchOff], layoutKScratch,
                              gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape,
                              needLoadL1);
                } else if constexpr (!IS_QUANT_V) {
                    const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                    blockMmad(gQuery_[qOffset], layoutQ, gKeyFp_[kOffset], layoutKGm,
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
            } else if constexpr (!IS_QUANT_V) {
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
            blockMmad.LoadQ(gQuery_[qOffset], layoutQ, shape);
        }
        const uint32_t depth = (nPage < BlockMmad::STAGES) ? nPage : BlockMmad::STAGES;
        for (uint32_t i = 0; i < depth; ++i) {
            IssuePageK(blockMmad, pageScratch[i], pageOff[i], shape, layoutKScratch, layoutKGm);
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
        } else if constexpr (!IS_QUANT_V) {
            blockMmad.IssueK(gKeyFp_[off], layoutKGm, shape);
        }
    }

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        MsaSegRowMaxEpilogue<IS_QUANT_V> epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->tableWidth, tiling_->strideScalePage,
                      tiling_->strideScaleHead, IS_QUANT_V, tiling_->keyLayout, tiling_->totalK, gScore_, gScale_,
                      gBlockTable_);
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

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            epilogue.BeginTask(task, subIdx, subBlockNum);
            if constexpr (IS_QUANT_V) {
                // 四槽：先填 0/1/2/3。WaitS(st) 后 slot[st%4] 空闲，gather(st+4) 叠在
                // cube(st+1..st+3) 上。两 AIV 对分 page，每页串行 MTE2→V→MTE3。
                const uint32_t nSt = task.numComputeSTiles;
                const uint32_t nPref = (nSt < MSA_K_SCRATCH_STAGES_A2) ? nSt : MSA_K_SCRATCH_STAGES_A2;
                for (uint32_t p = 0; p < nPref; ++p) {
                    if (StileNeedsKScratch(task, p * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, p, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                }
                for (uint32_t st = 0; st < nSt; ++st) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                    const uint32_t stPref = st + MSA_K_SCRATCH_STAGES_A2;
                    const uint64_t sBase =
                        coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                    if (stPref < nSt && StileNeedsKScratch(task, stPref * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, stPref, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                    epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE);
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
                if (task.writeTailFill) {
                    for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                        epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE);
                    }
                }
            } else {
                epilogue.FillInvalidTail(task);
            }
            epilogue.EndTask();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    const MsaIndexScoreTilingData* __restrict tiling_ = nullptr;
    MsaTaskScheduler<MSA_BLOCK_SIZE> scheduler_;

    AscendC::GlobalTensor<ElementQ> gQuery_;
    AscendC::GlobalTensor<ElementQ> gKeyFp_;
    AscendC::GlobalTensor<int8_t> gKeyInt8_;
    AscendC::GlobalTensor<ElementQ> gKeyScratch_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gActualSeqQlen_;
    AscendC::GlobalTensor<int32_t> gActualSeqKlen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;
    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<ElementS> gWorkspace_;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_KERNEL_ARCH22_H
