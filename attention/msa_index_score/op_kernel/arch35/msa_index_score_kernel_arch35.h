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
 * \file msa_index_score_kernel_arch35.h
 * \brief MsaIndexScore Ascend 950：计算路径照搬 A2 核函数：AIC 做 paged QKᵀ，AIV 做（可选反量化）+ mask + 分段 RowMax。
 *        第一阶段：计算/握手照搬 A2，ArchTag=AtlasA5。
 *
 * 非量化：query/key 同为 bf16|fp16|fp8，score 侧不乘 scale。
 * int8 量化：key 为 int8；AIV 将 page DataCopy+Cast 到 fp 暂存后通知 AIC 做 Mmad；
 * AIV 在 mask 前按 scale[NP,N_kv,P] 对 S 做列乘完成 per-token 反量化。
 *
 * MIX 编译：`if ASCEND_IS_AIC` 拦不住模板实例化。AIV 再实例化 FP8 Cube
 *（Mmad / LoadData2DParamsV2）会让 bisheng 栈砸（exit 134）。Cube 路径仅在
 * `__DAV_C310_CUBE__` / `__DAV_310R6_CUBE__` 下编译。
 */

#ifndef MSA_INDEX_SCORE_KERNEL_ARCH35_H
#define MSA_INDEX_SCORE_KERNEL_ARCH35_H

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
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
#include "msa_block_mmad_arch35.h"
#endif
#include "../msa_index_score_task.h"
#include "msa_seg_row_max_epilogue_arch35.h"

namespace MsaIndexScoreNs {

/// IS_BLOCK16：逻辑块长 MSA_BLOCK16_SIZE。BSND/BNSD、BBND/BNBD 页长 16，或 TND 显式 pool_block_size=16（两级
/// maxpool）。 一次 128 列 GEMM 覆盖 MSA_BLOCKS_PER_STILE 个逻辑 block（= 一个「窗」）， 归约时多做一次 head 间
/// maxpool。Cube 的 N 仍是 MSA_BLOCK_SIZE，与 128 路径共用全部流水。
template <class ElementQ_, bool IS_QUANT, bool IS_BLOCK16 = false>
class MsaIndexScoreKernel {
public:
    using ElementQ = ElementQ_;
    using ArchTag = Catlass::Arch::AtlasA5;
    /// 逻辑 block 的 token 数：决定 scheduler 的块算术，与 Cube 的 N 无关。
    static constexpr uint32_t LOGICAL_BLOCK_SIZE = IS_BLOCK16 ? MSA_BLOCK16_SIZE : MSA_BLOCK_SIZE;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementQ, LayoutK>;
    // S workspace 元素：非量化 fp16（fixpipe F322F16 直接写，写读流量减半），int8 fp32
    // （S 幅值大、1e-3 容差下 fp16 精度余量不足 15%，保持 fp32 反量化）。
    using ElementS = std::conditional_t<IS_QUANT, float, half>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    // int8 的 K 源是每 stile 复用的 per-core scratch，保持 2 级 L1B 流水（3 级实测竞态崩溃）；
    // 同样出于该 scratch 的时序保守性，unit flag 只在非量化路径开启。
    using BlockMmad = MsaBlockMmad<QType, KType, SType, IS_QUANT ? MSA_L1B_STAGES_INT8 : MSA_L1B_STAGES_FP16, false>;
#endif

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static constexpr bool USE_C2UB = (MSA_A5_USE_C2UB != 0);
    using Epilogue = MsaSegRowMaxEpilogue<IS_QUANT_V, IS_BLOCK16>;
    // 块长 16 与 host 的两槽 workspace 对齐，按 ping 选择当前窗 / 预取窗。
    static constexpr uint32_t K_SCRATCH_SLOTS = (USE_C2UB && IS_BLOCK16) ? MSA_A5_S_STAGES : 1U;
    static constexpr uint32_t K_SCRATCH_ELEMS =
        (USE_C2UB ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM) * K_SCRATCH_SLOTS;
    // C_to_UB：cast 必须在 epilogue stage 之后。放在 ping 64KB 处会盖住 T64/deq/stage，
    // 下一页 gather 会把尚未 Flush 的 score 清掉（TND/int8 多 page：1e29→0、fill 变 0）。
    static constexpr uint32_t UB_OFF_CAST_I8 =
        USE_C2UB ? (((Epilogue::UB_TOTAL + MSA_UB_ALIGN_BYTES - 1U) / MSA_UB_ALIGN_BYTES) * MSA_UB_ALIGN_BYTES) : 0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    static_assert(UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE, "int8 cast UB out of bounds");

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
        if (IS_QUANT_V || MsaIsSeqKeyLayout(tiling_->keyLayout) || IS_BLOCK16) {
            gKeyScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(reinterpret_cast<__gm__ float*>(workspace) +
                                                                            tiling_->kScratchOffsetElems));
        }
        if (PackQuery()) {
            gQScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(workspace) + tiling_->qScratchElems);
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
                        tiling_->kvChunks, gActualSeqQlen_, gActualSeqKlen_, gStartLoc_, false, tiling_->qMaxSeq,
                        tiling_->kMaxSeq, tiling_->tableWidth, tiling_->chunkBudget);
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
        ProcessCube();
#else
        ProcessVector();
#endif
    }

private:
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    __aicore__ inline void ProcessCube()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * K_SCRATCH_ELEMS;
        const uint32_t totalTasks = scheduler_.TotalTasks();
        if constexpr (USE_C2UB) {
            ProcessCubeC2Ub(resource, blockMmad, coreIdx, coreNum, coreKScratch, totalTasks);
        } else {
            ProcessCubeGm(blockMmad, coreIdx, coreNum, coreKScratch, totalTasks);
        }
    }

    __aicore__ inline void ProcessCubeGm(BlockMmad& blockMmad, uint32_t coreIdx, uint32_t coreNum,
                                         uint64_t coreKScratch, uint32_t totalTasks)
    {
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagKReady{MSA_FLAG_K_READY};
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            // block_table 按 256B 预取，盖住后续逐页 GetValue 的 HBM 延迟。
            PreloadBlockTable(task);
            bool needLoadQ = true;
            bool qSynced = false;
            for (uint32_t st = task.sStileBegin; st < task.sStileEnd; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                // BSND/BNSD 的 Q 只在本任务第一拍收集。之后各 stile 只在 K scratch 需要时再同步。
                const bool needQSync = PackQuery() && !qSynced;
                if (needKScratch || needQSync) {
                    Catlass::Arch::CrossCoreWaitFlag(flagKReady);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    qSynced = true;
                }
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                ComputeSTile(blockMmad, task, st * MSA_BLOCKS_PER_STILE, sBase, coreKScratch, needLoadQ);
                needLoadQ = false;
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagSReady);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessCubeC2Ub(Catlass::Arch::Resource<ArchTag>& resource, BlockMmad& blockMmad,
                                           uint32_t coreIdx, uint32_t coreNum, uint64_t coreKScratch,
                                           uint32_t totalTasks)
    {
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            if (task.sBlkBegin >= task.sBlkEnd) {
                continue;
            }
            // block_table 按 256B 预取，盖住后续逐页 GetValue 的 HBM 延迟。
            PreloadBlockTable(task);
            bool needLoadQ = true;
            uint32_t pageSeq = 0;
            if constexpr (IS_BLOCK16) {
                // Q 一任务只打包一次。K 旗只用于 scratch 窗，直读窗不占这面旗。
                if (PackQuery()) {
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd; ++win) {
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    if (!WindowDirectGm(task, win)) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                            MSA_A5_AIV_FLAG_OFFSET);
                    auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                    if (WindowDirectGm(task, win)) {
                        ComputeWindowFromGm(blockMmad, task, win, ubS, needLoadQ);
                    } else {
                        ComputeWindowToUb(blockMmad, task, ubS, KScratchSlot(coreKScratch, ping), needLoadQ);
                    }
                    needLoadQ = false;
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping +
                                                                           MSA_A5_AIV_FLAG_OFFSET);
                    ++pageSeq;
                }
                continue;
            }
            // BSND/BNSD：本 M-tile 的 Q 只收集一次，进 block 循环前等完。
            // 不能并进每个 block 的 K scratch 旗，否则满页不等 K 时 Cube 会多等一拍。
            if (PackQuery()) {
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd; ++blk) {
                const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                if (KeyBlockNeedsScratch(task, blk)) {
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                    // 与 GM 路径一致：等 AIV MTE3 写完 scratch 再 MTE2 进 L1，否则 int8/TND 读到 0。
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping + MSA_A5_AIV_FLAG_OFFSET);
                auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                ComputePageToUb(blockMmad, task, blk, ubS, coreKScratch, needLoadQ);
                needLoadQ = false;
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping + MSA_A5_AIV_FLAG_OFFSET);
                ++pageSeq;
            }
        }
        // 循环前每个 AIV 子块都预置了 VC slot。这里两个子块都要收回，
        // 否则子块 1（flag+16）的计数留到下一次 launch，Cube 会提前放行并踩到还在用的 UB。
        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + 0);
        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + 0 + MSA_A5_AIV_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + 1);
        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + 1 + MSA_A5_AIV_FLAG_OFFSET);
    }
#endif

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
    /// 行距必须用 tableWidth。按 score 末维寻址会在 batch>0 时落到页表分配外。
    /// 每一笔 256B 必须完整落在分配内，短尾不再补一笔。
    __aicore__ inline void PreloadBlockTable(const MsaTask& task)
    {
        // TND/BSND/BNSD 没有 block_table，gBlockTable_ 未绑定。对空地址 dc_preload 会把设备打成 507015。
        if (!MsaIsPaKeyLayout(tiling_->keyLayout) || task.sBlkEnd <= task.sBlkBegin) {
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
        // 存储页表按页粒度：打分块首 token 定位页与页内偏移。页长与打分块长解耦；
        // 页长 = 打分块长（128 路径或 M3 页 16）时退化为逐块 1:1 寻址。
        // 可见块已停在页表宽度；这里只在页号越界时钳到最后一页，避免读出页表。
        const uint32_t tok = blk * tiling_->poolBlockSize;
        uint32_t pageIdx = tok / tiling_->pageSize;
        const uint32_t maxPageIdx = tiling_->tableWidth > 0U ? tiling_->tableWidth - 1U : 0U;
        if (pageIdx > maxPageIdx) {
            pageIdx = maxPageIdx;
        }
        const int32_t page = gBlockTable_.GetValue(task.batchIdx * tiling_->tableWidth + pageIdx);
        return static_cast<uint64_t>(page) * tiling_->strideKvBlock +
               static_cast<uint64_t>(tok % tiling_->pageSize) * tiling_->strideKvToken;
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbPartial(const AscendC::LocalTensor<T>& ub, const AscendC::GlobalTensor<T>& gm,
                                             uint32_t nElem)
    {
        if (nElem == 0) {
            return;
        }
        if constexpr (sizeof(T) == 1U && !std::is_same_v<T, uint8_t>) {
            CopyGmToUbPartial(ub.template ReinterpretCast<uint8_t>(), gm.template ReinterpretCast<uint8_t>(), nElem);
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

    /// AIV：把一个可见 page 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    __aicore__ inline void GatherKeyPageToScratch(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                  uint32_t blk, uint64_t scratchOff)
    {
        AscendC::LocalTensor<int8_t> ubI8 = resource.ubBuf.template GetBufferByByte<int8_t>(UB_OFF_CAST_I8);
        AscendC::LocalTensor<ElementQ> ubFp = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint64_t kOffset = KeyBlockGmOffset(task, blk);
        const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
        const uint32_t nValid = nValidTok * headDim;

        if constexpr (IS_QUANT_V) {
            uint32_t copyN = nElem;
            if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                const uint32_t tokenStart = task.cuKStart + blk * tiling_->poolBlockSize;
                const uint32_t packedEnd = tokenStart + MSA_BLOCK_SIZE;
                if (tiling_->totalK > 0 && packedEnd > tiling_->totalK) {
                    copyN = nValid;
                }
            }
            if (copyN < nElem) {
                AscendC::Duplicate(ubFp, static_cast<ElementQ>(0), nElem);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            if (copyN > 0) {
                if (copyN == nElem) {
                    AscendC::DataCopy(ubI8, gKeyInt8_[kOffset], nElem);
                } else {
                    CopyGmToUbPartial(ubI8, gKeyInt8_[kOffset], copyN);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(ubFp, ubI8, AscendC::RoundMode::CAST_NONE, copyN);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(gKeyScratch_[scratchOff], ubFp, nElem);
        } else {
            AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
            if (nValid < nElem) {
                if constexpr (sizeof(ElementQ) == 1U) {
                    AscendC::Duplicate(ubK.template ReinterpretCast<uint8_t>(), static_cast<uint8_t>(0), nElem);
                } else {
                    AscendC::Duplicate(ubK, static_cast<ElementQ>(0), nElem);
                }
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
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// 本窗 128 个 token 都落在 kv 内、整窗可见、且在 GM 上按 token 步长连成一段时，
    /// Cube 直接读，省掉 GM→UB→scratch→L1 这一趟搬运。
    /// 顺序布局天然连续；分页布局要么整窗落在同一页，要么页号连续且页间步长正好等于页内跨度。
    __aicore__ inline bool WindowDirectGm(const MsaTask& task, uint32_t win) const
    {
        if (tiling_->strideKvToken != tiling_->headDim || tiling_->headDim == 0U || task.kvLen <= 0) {
            return false;
        }
        const uint32_t blkBase = win * MSA_BLOCKS_PER_STILE;
        if (blkBase + MSA_BLOCKS_PER_STILE > task.visibleEndBlk) {
            return false;
        }
        const uint64_t tokEnd = (static_cast<uint64_t>(blkBase) + MSA_BLOCKS_PER_STILE) * MSA_BLOCK16_SIZE;
        if (tokEnd > static_cast<uint64_t>(task.kvLen)) {
            return false;
        }
        if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            return true;
        }
        return PagedWindowContiguous(task, blkBase);
    }

    /// 分页布局下本窗跨到的页是否在 GM 上接得上。
    __aicore__ inline bool PagedWindowContiguous(const MsaTask& task, uint32_t blkBase) const
    {
        const uint32_t pageSize = tiling_->pageSize;
        const uint32_t tableWidth = tiling_->tableWidth;
        if (pageSize == 0U || tableWidth == 0U) {
            return false;
        }
        const uint32_t tok0 = blkBase * MSA_BLOCK16_SIZE;
        const uint32_t firstPage = tok0 / pageSize;
        const uint32_t lastPage = (tok0 + MSA_BLOCK_SIZE - 1U) / pageSize;
        if (lastPage >= tableWidth) {
            return false;
        }
        if (firstPage == lastPage) {
            return true;
        }
        if (static_cast<uint64_t>(tiling_->strideKvBlock) !=
            static_cast<uint64_t>(pageSize) * static_cast<uint64_t>(tiling_->strideKvToken)) {
            return false;
        }
        const uint32_t tableBase = task.batchIdx * tableWidth;
        const int32_t page0 = gBlockTable_.GetValue(tableBase + firstPage);
        for (uint32_t p = firstPage + 1U; p <= lastPage; ++p) {
            if (gBlockTable_.GetValue(tableBase + p) != page0 + static_cast<int32_t>(p - firstPage)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t KScratchSlot(uint64_t coreKScratch, uint32_t ping) const
    {
        return coreKScratch + static_cast<uint64_t>(ping) * MSA_A5_K_SCRATCH_ELEM_NUM;
    }

    __aicore__ inline void SignalKeyReady() const
    {
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
    }

    /// 块长 16 的一个逻辑块里，落在 kv_len 内的 token 数。
    __aicore__ inline uint32_t KeyPage16ValidTokens(const MsaTask& task, uint32_t blk) const
    {
        if (task.kvLen <= 0) {
            return 0;
        }
        const uint32_t tok = blk * MSA_BLOCK16_SIZE;
        if (tok >= static_cast<uint32_t>(task.kvLen)) {
            return 0;
        }
        const uint32_t remain = static_cast<uint32_t>(task.kvLen) - tok;
        return remain < MSA_BLOCK16_SIZE ? remain : MSA_BLOCK16_SIZE;
    }

    /// AIV（块长 16）：把一窗 8 个逻辑块收成 128 个 token。
    /// BBND/BNBD 经 block_table 寻址；BSND/BNSD 按连续 token 寻址。越界补零。
    __aicore__ inline void GatherKeyWindowToScratch(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                    uint32_t win, uint64_t scratchOff)
    {
        AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint32_t blkBase = win * MSA_BLOCKS_PER_STILE;
        uint32_t blkLimit = tiling_->maxBlocksPerBatch;
        if (MsaIsPaKeyLayout(tiling_->keyLayout) && tiling_->tableWidth > 0U && tiling_->tableWidth < blkLimit) {
            blkLimit = tiling_->tableWidth;
        }
        bool windowFull = true;
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= blkLimit || KeyPage16ValidTokens(task, blk) != MSA_BLOCK16_SIZE) {
                windowFull = false;
                break;
            }
        }
        if (!windowFull) {
            if constexpr (sizeof(ElementQ) == 1U) {
                AscendC::Duplicate(ubK.template ReinterpretCast<uint8_t>(), static_cast<uint8_t>(0), nElem);
            } else {
                AscendC::Duplicate(ubK, static_cast<ElementQ>(0), nElem);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        }
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= blkLimit) {
                break;
            }
            const uint32_t nTok = KeyPage16ValidTokens(task, blk);
            if (nTok == 0U) {
                continue;
            }
            uint64_t kOffset = 0;
            if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                kOffset = MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, blk * MSA_BLOCK16_SIZE,
                                             tiling_->numKvHeads, tiling_->headDim, tiling_->kMaxSeq);
            } else {
                const uint32_t tok = blk * MSA_BLOCK16_SIZE;
                const int32_t page =
                    gBlockTable_.GetValue(task.batchIdx * tiling_->tableWidth + tok / tiling_->pageSize);
                kOffset = static_cast<uint64_t>(page) * tiling_->strideKvBlock +
                          static_cast<uint64_t>(tok % tiling_->pageSize) * tiling_->strideKvToken;
            }
            auto dst = ubK[static_cast<uint32_t>(j * MSA_BLOCK16_SIZE * headDim)];
            if (tiling_->strideKvToken == headDim) {
                CopyGmToUbPartial(dst, gKeyFp_[kOffset], nTok * headDim);
            } else {
                AscendC::DataCopyExtParams params;
                params.blockCount = static_cast<uint16_t>(nTok);
                params.blockLen = headDim * static_cast<uint32_t>(sizeof(ElementQ));
                params.srcStride = (tiling_->strideKvToken - headDim) * static_cast<uint32_t>(sizeof(ElementQ));
                params.dstStride = 0;
                AscendC::DataCopyPadExtParams<ElementQ> pad;
                pad.isPad = false;
                pad.leftPadding = 0;
                pad.rightPadding = 0;
                pad.paddingValue = 0;
                AscendC::DataCopyPad(dst, gKeyFp_[kOffset], params, pad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        AscendC::DataCopy(gKeyScratch_[scratchOff], ubK, nElem);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    /// AIV：把本 S-tile 可见 block 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    __aicore__ inline void GatherKeySTileToScratch(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                   uint32_t blkBase, uint64_t kScratchBase)
    {
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (!KeyBlockNeedsScratch(task, blk)) {
                continue;
            }
            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            GatherKeyPageToScratch(resource, task, blk, scratchOff);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// TND 的 M-tile 在 GM 上已是连续 (token, head)。BSND/BNSD 先收进 per-core scratch。
    __aicore__ inline bool PackQuery() const
    {
        return tiling_->queryLayout == MSA_QUERY_LAYOUT_BSND || tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD;
    }

    __aicore__ inline uint64_t QueryElemOffset(const MsaTask& task) const
    {
        if (PackQuery()) {
            return static_cast<uint64_t>(AscendC::GetBlockIdx()) * MSA_ROW_TILE_M * tiling_->headDim;
        }
        return MsaQueryTileElem(tiling_->queryLayout, task.batchIdx, task.cuQStart, task.mStart, tiling_->numQHeads,
                                tiling_->headDim, tiling_->qMaxSeq, tiling_->strideQn);
    }

    __aicore__ inline AscendC::GlobalTensor<ElementQ> QueryGm(const MsaTask& task) const
    {
        const uint64_t elem = QueryElemOffset(task);
        if (PackQuery()) {
            return gQScratch_[elem];
        }
        return gQuery_[elem];
    }

    /// BSND 以及 N1=1 的 BNSD：逻辑行已经连续，一次搬完。BNSD 且 N1>1 时按 (token, head) 逐行收。
    __aicore__ inline bool QueryRowsContiguous() const
    {
        return tiling_->queryLayout == MSA_QUERY_LAYOUT_BSND || tiling_->numQHeads <= 1U;
    }

    __aicore__ inline uint64_t QuerySrcElem(const MsaTask& task, uint32_t localRow) const
    {
        const uint32_t n = tiling_->numQHeads;
        const uint32_t d = tiling_->headDim;
        const uint32_t seq = tiling_->qMaxSeq;
        if (tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD) {
            // [B, N, S, D]。N=1 时 token 沿 S 连续；N>1 时每个 (token, head) 单独寻址。
            if (n <= 1U) {
                return (static_cast<uint64_t>(task.batchIdx) * seq + localRow) * d;
            }
            const uint32_t token = localRow / n;
            const uint32_t head = localRow - token * n;
            return (static_cast<uint64_t>(task.batchIdx) * n * seq + static_cast<uint64_t>(head) * seq + token) * d;
        }
        // BSND：[B, S, N, D]，batch 内 (token, head) 连续。
        return MsaQueryTileElem(tiling_->queryLayout, task.batchIdx, task.cuQStart, localRow, n, d, seq,
                                tiling_->strideQn);
    }

    /// BNSD 的 head 在 S 之外，Cube 的连续行是 (token, head)。先收成一块连续 Q。
    /// 逐行 DataCopy 会把 MTE3 的完成信号拆碎，后面的 subblock barrier 等不到；
    /// 与 K scratch 一样：先收进 UB，再一次对齐长度写回 GM。
    __aicore__ inline void GatherQueryTile(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                           uint64_t dstBase)
    {
        const uint32_t n = tiling_->numQHeads;
        const uint32_t d = tiling_->headDim;
        if (task.mActual == 0U || d == 0U || n == 0U) {
            return;
        }
        AscendC::LocalTensor<ElementQ> ub = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
        if (QueryRowsContiguous()) {
            const uint64_t src = QuerySrcElem(task, task.mStart);
            CopyGmToUbPartial(ub, gQuery_[src], task.mActual * d);
        } else {
            for (uint32_t row = 0; row < task.mActual; ++row) {
                const uint64_t src = QuerySrcElem(task, task.mStart + row);
                CopyGmToUbPartial(ub[row * d], gQuery_[src], d);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        uint32_t nCopy = task.mActual * d;
        nCopy = (nCopy + 15U) & ~15U;
        const uint32_t cap = MSA_ROW_TILE_M * d;
        if (nCopy > cap) {
            nCopy = cap;
        }
        AscendC::DataCopy(gQScratch_[dstBase], ub, nCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
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
        const auto queryGm = QueryGm(task);
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (KeyBlockNeedsScratch(task, blk)) {
                const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
                blockMmad(queryGm, layoutQ, gKeyScratch_[scratchOff], layoutKScratch,
                          gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape, needLoadL1);
            } else if constexpr (!IS_QUANT_V) {
                const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                blockMmad(queryGm, layoutQ, gKeyFp_[kOffset], layoutKGm,
                          gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape, needLoadL1);
            }
            needLoadL1 = false;
        }
    }

    __aicore__ inline void ComputePageToUb(BlockMmad& blockMmad, const MsaTask& task, uint32_t blk,
                                           const AscendC::LocalTensor<float>& ubS, uint64_t kScratchBase,
                                           bool needLoadL1)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const auto queryGm = QueryGm(task);
        if (KeyBlockNeedsScratch(task, blk)) {
            blockMmad.ComputeToUb(queryGm, layoutQ, gKeyScratch_[kScratchBase], layoutKScratch, ubS, shape, needLoadL1);
        } else if constexpr (!IS_QUANT_V) {
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            blockMmad.ComputeToUb(queryGm, layoutQ, gKeyFp_[kOffset], layoutKGm, ubS, shape, needLoadL1);
        }
    }

    /// 块长 16：一次 GEMM 读 scratch 里已经拼好的 128 列。Fixpipe 按 numQHeads 整组切给两个 AIV。
    __aicore__ inline void ComputeWindowToUb(BlockMmad& blockMmad, const MsaTask& task,
                                             const AscendC::LocalTensor<float>& ubS, uint64_t kScratchBase,
                                             bool needLoadL1)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const auto queryGm = QueryGm(task);
        const uint32_t split = tiling_->numQHeads;
        blockMmad.ComputeToUb(queryGm, layoutQ, gKeyScratch_[kScratchBase], layoutKScratch, ubS, shape, needLoadL1,
                              split);
    }

    /// 块长 16 的完整顺序窗：128 个 token 在 GM 上连续，按 token 步长直接进 GEMM。
    __aicore__ inline void ComputeWindowFromGm(BlockMmad& blockMmad, const MsaTask& task, uint32_t win,
                                               const AscendC::LocalTensor<float>& ubS, bool needLoadL1)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const auto queryGm = QueryGm(task);
        const uint32_t split = tiling_->numQHeads;
        const uint64_t kOffset = KeyBlockGmOffset(task, win * MSA_BLOCKS_PER_STILE);
        blockMmad.ComputeToUb(queryGm, layoutQ, gKeyFp_[kOffset], layoutKGm, ubS, shape, needLoadL1, split);
    }
#endif

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        Epilogue epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->tableWidth, tiling_->poolBlockSize, tiling_->pageSize,
                      tiling_->strideScalePage, tiling_->strideScaleHead, IS_QUANT_V, tiling_->keyLayout,
                      tiling_->totalK, gScore_, gScale_, gBlockTable_);
        const uint32_t subBlockNum = MSA_AIV_PER_AIC;
        const uint32_t subIdx = AscendC::GetBlockIdx() % subBlockNum;
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * K_SCRATCH_ELEMS;
        const uint32_t totalTasks = scheduler_.TotalTasks();
        if constexpr (USE_C2UB) {
            ProcessVectorC2Ub(resource, epilogue, subIdx, subBlockNum, coreIdx, coreNum, coreKScratch, totalTasks);
        } else {
            ProcessVectorGm(resource, epilogue, subIdx, subBlockNum, coreIdx, coreNum, coreKScratch, totalTasks);
        }
    }

    __aicore__ inline void ProcessVectorGm(Catlass::Arch::Resource<ArchTag>& resource, Epilogue& epilogue,
                                           uint32_t subIdx, uint32_t subBlockNum, uint32_t coreIdx, uint32_t coreNum,
                                           uint64_t coreKScratch, uint32_t totalTasks)
    {
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagKReady{MSA_FLAG_K_READY};
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            epilogue.BeginTask(task, subIdx, subBlockNum);
            const bool gatherQ = PackQuery();
            bool qSynced = false;
            const uint64_t qScratch = static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim;
            for (uint32_t st = task.sStileBegin; st < task.sStileEnd; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                const bool needQSync = gatherQ && !qSynced;
                if (needKScratch || needQSync) {
                    if (subIdx == 0) {
                        if (needQSync) {
                            GatherQueryTile(resource, task, qScratch);
                        }
                        if (needKScratch) {
                            GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, coreKScratch);
                        }
                    }
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagKReady);
                    qSynced = true;
                }
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE);
                ++tileSeq;
            }
            if (task.writeTailFill) {
                for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                    epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE);
                }
            }
            epilogue.EndTask();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessVectorC2Ub(Catlass::Arch::Resource<ArchTag>& resource, Epilogue& epilogue,
                                             uint32_t subIdx, uint32_t subBlockNum, uint32_t coreIdx, uint32_t coreNum,
                                             uint64_t coreKScratch, uint32_t totalTasks)
    {
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + 0);
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + 1);
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            if (task.sBlkBegin >= task.sBlkEnd && !task.writeTailFill) {
                continue;
            }
            epilogue.BeginTask(task, subIdx, subBlockNum);
            uint32_t pageSeq = 0;
            if (task.sBlkBegin < task.sBlkEnd && PackQuery() && !IS_BLOCK16) {
                if (subIdx == 0) {
                    GatherQueryTile(resource, task, static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim);
                }
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
            }
            if constexpr (IS_BLOCK16) {
                if (task.sBlkBegin < task.sBlkEnd && PackQuery()) {
                    if (subIdx == 0) {
                        GatherQueryTile(resource, task,
                                        static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim);
                    }
                    SignalKeyReady();
                }
                bool prefetched = false;
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd; ++win) {
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    if (!WindowDirectGm(task, win) && !prefetched) {
                        if (subIdx == 0) {
                            GatherKeyWindowToScratch(resource, task, win, KScratchSlot(coreKScratch, ping));
                        }
                        SignalKeyReady();
                    }
                    prefetched = false;
                    if (win + 1U < task.sStileEnd && !WindowDirectGm(task, win + 1U)) {
                        const uint32_t nextPing = (pageSeq + 1U) % MSA_A5_S_STAGES;
                        if (subIdx == 0) {
                            GatherKeyWindowToScratch(resource, task, win + 1U, KScratchSlot(coreKScratch, nextPing));
                        }
                        SignalKeyReady();
                        prefetched = true;
                    }
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                    epilogue.ProcessWindowUb(ping, task, win);
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + ping);
                    ++pageSeq;
                }
                epilogue.EndTask();
                continue;
            }
            for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd; ++blk) {
                const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                if (KeyBlockNeedsScratch(task, blk)) {
                    if (subIdx == 0) {
                        GatherKeyPageToScratch(resource, task, blk, coreKScratch);
                    }
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
                }
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                epilogue.ProcessPageUb(ping, task, blk);
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + ping);
                ++pageSeq;
            }
            epilogue.EndTask();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    const MsaIndexScoreTilingData* __restrict tiling_ = nullptr;
    MsaTaskScheduler<LOGICAL_BLOCK_SIZE> scheduler_;

    AscendC::GlobalTensor<ElementQ> gQuery_;
    AscendC::GlobalTensor<ElementQ> gQScratch_;
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

#endif // MSA_INDEX_SCORE_KERNEL_ARCH35_H
