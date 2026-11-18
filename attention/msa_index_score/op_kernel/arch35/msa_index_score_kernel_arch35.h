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

#include "c_api/asc_simd.h"
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
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
#include "msa_index_score_topk_arch35.h"
#endif

namespace MsaIndexScoreNs {

/// IS_BLOCK16：逻辑块长 MSA_BLOCK16_SIZE。BSND/BNSD、950 fp16/bf16 的 TND，或 BBND/BNBD 页长 16。
/// 一次 128 列 GEMM 覆盖 MSA_BLOCKS_PER_STILE 个逻辑 block（= 一个「窗」），
/// 归约时多做一次 head 间 maxpool。Cube 的 N 仍是 MSA_BLOCK_SIZE，与 128 路径共用全部流水。
template <class ElementQ_, bool IS_QUANT, bool IS_BLOCK16 = false, bool USE_C2UB_IN = (MSA_A5_USE_C2UB != 0),
          bool IS_MXFP4 = false, bool MXFP4_N512 = false, bool ENABLE_TOPK = false, bool MX_FUSED_STAGE = false,
          bool MX_LATENCY_TND = false>
class MsaIndexScoreKernel {
public:
    static_assert(!(IS_BLOCK16 && ENABLE_TOPK), "block size 16 does not implement topk");
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
    // WIDE 双路 Fixpipe(subBlock) 配 unitFlag 实测 launch 卡死，保持关闭、继续用 M_FIX/FIX_M。
    using BlockMmad =
        MsaBlockMmad<QType, KType, SType,
                     IS_MXFP4 ? (IS_BLOCK16 ? MSA_L1B_STAGES_INT8 : MSA_L1B_STAGES_FP16) :
                                (IS_QUANT ? MSA_L1B_STAGES_INT8 : (IS_BLOCK16 ? MSA_L1B_STAGES_FP16 : 4U)),
                     false, IS_MXFP4, IS_MXFP4 && MXFP4_N512,
                     IS_MXFP4 && !IS_BLOCK16 && USE_C2UB_IN && !MX_LATENCY_TND>;
#endif

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static constexpr bool USE_C2UB = USE_C2UB_IN;
    using Epilogue = MsaSegRowMaxEpilogue<IS_QUANT_V, IS_BLOCK16, IS_MXFP4, MX_FUSED_STAGE>;
    static constexpr uint32_t K_SCRATCH_SLOTS = (USE_C2UB && IS_BLOCK16 && !IS_MXFP4) ? MSA_A5_S_STAGES : 1U;
    static constexpr uint32_t K_SCRATCH_ELEMS =
        (USE_C2UB ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM) * K_SCRATCH_SLOTS;
    // C_to_UB：cast 必须在 epilogue stage 之后。放在 ping 64KB 处会盖住 T64/deq/stage，
    // 下一页 gather 会把尚未 Flush 的 score 清掉（TND/int8 多 page：1e29→0、fill 变 0）。
    static constexpr uint32_t UB_OFF_CAST_I8 =
        USE_C2UB ? (((Epilogue::UB_TOTAL + MSA_UB_ALIGN_BYTES - 1U) / MSA_UB_ALIGN_BYTES) * MSA_UB_ALIGN_BYTES) : 0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    // Native block16 also uses these buffers for paged/tail K and BSND/BNSD Q.
    // MXFP4 block16 has no GM gather; it only needs the small Q-scale tile.
    static_assert((IS_BLOCK16 && IS_MXFP4) || (UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE),
                  "Q/K gather or int8 cast UB out of bounds");

    static constexpr bool IS_TND_TND = MX_LATENCY_TND;
    static_assert(!IS_TND_TND || (IS_MXFP4 && !IS_BLOCK16 && !IS_QUANT && USE_C2UB && !ENABLE_TOPK),
                  "latency specialization requires TND MXFP4 block128 score C2UB");

    __aicore__ inline MsaIndexScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale, GM_ADDR actualSeqQlen,
                                GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score, GM_ADDR sparseIndices,
                                GM_ADDR sparseValues, GM_ADDR workspace,
                                const MsaIndexScoreTilingData* __restrict tiling)
    {
        tiling_ = tiling;
        gQuery_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(query));
        if constexpr (IS_QUANT_V) {
            gKeyInt8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        } else {
            gKeyFp_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(key));
        }
        if (IS_QUANT_V || IS_TND_TND || MsaIsSeqKeyLayout(tiling_->keyLayout) || IS_BLOCK16) {
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
        if constexpr (ENABLE_TOPK) {
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

        scheduler_.Init(tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch, tiling_->scoreBlockStride,
                        tiling_->sparseMode, tiling_->initBlocks, tiling_->localBlocks,
                        IS_TND_TND ? MSA_KEY_LAYOUT_TND : tiling_->keyLayout, tiling_->kvChunks, gActualSeqQlen_,
                        gActualSeqKlen_, gStartLoc_, false, IS_TND_TND ? 0U : tiling_->qMaxSeq, tiling_->kMaxSeq,
                        tiling_->tableWidth, tiling_->chunkBudget, tiling_->pageSize, tiling_->totalQ);
    }

    /// MXFP4 direct 精简启动：不走通用 Init 里的 int8/Q-scratch/dequant scale 分支，减少热路径活对象。
    __aicore__ inline void InitMx(GM_ADDR query, GM_ADDR key, GM_ADDR queryScale, GM_ADDR keyScale, GM_ADDR blockTable,
                                  GM_ADDR actualSeqQlen, GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score,
                                  GM_ADDR workspace, const MsaIndexScoreTilingData* __restrict tiling,
                                  GM_ADDR sparseIndices = nullptr, GM_ADDR sparseValues = nullptr)
    {
        tiling_ = tiling;
        gQuery_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(query));
        gKeyFp_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ*>(key));
        if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
            gBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        }
        gActualSeqQlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(actualSeqQlen));
        gActualSeqKlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(actualSeqKlen));
        gStartLoc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(startLoc));
        if constexpr (ENABLE_TOPK) {
            gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + tiling_->scoreWorkspaceElems);
            gSparseIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices));
            if (tiling_->returnChoice == MSA_RETURN_INDICES_VALUES) {
                gSparseValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sparseValues));
            }
        } else {
            gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score));
        }
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS*>(workspace));
        scheduler_.Init(tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch, tiling_->scoreBlockStride,
                        tiling_->sparseMode, tiling_->initBlocks, tiling_->localBlocks,
                        IS_TND_TND ? MSA_KEY_LAYOUT_TND : tiling_->keyLayout, tiling_->kvChunks, gActualSeqQlen_,
                        gActualSeqKlen_, gStartLoc_, false, IS_TND_TND ? 0U : tiling_->qMaxSeq, tiling_->kMaxSeq,
                        tiling_->tableWidth, tiling_->chunkBudget, tiling_->pageSize, tiling_->totalQ);
        gQueryScaleMx_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(queryScale));
        gKeyScaleMx_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(keyScale));
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
    // Causal single-request prefill has increasing work per M tile. Alternate
    // complete waves to pair early/light and late/heavy tiles without changing
    // the scheduler's monotonically increasing per-core task order.
    __aicore__ inline bool BalanceCausalTaskWaves(uint32_t coreNum, uint32_t totalTasks) const
    {
        if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && USE_C2UB) {
            return tiling_->batch == 1U && tiling_->sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN &&
                   tiling_->kvChunks == 1U && coreNum > 0U && totalTasks > coreNum && totalTasks % coreNum == 0U;
        }
        return false;
    }

    __aicore__ inline uint32_t TaskOfWave(uint32_t slot, uint32_t wave, uint32_t coreIdx, uint32_t coreNum,
                                          bool balance) const
    {
        return balance && (wave & 1U) != 0U ? slot - coreIdx + coreNum - 1U - coreIdx : slot;
    }

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
        if constexpr (ENABLE_TOPK) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
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
            if constexpr (IS_MXFP4) {
                blockMmad.SetBnsdQueryGather(DirectBnsdQuery() ? tiling_->numQHeads : 0U, tiling_->qMaxSeq,
                                             task.mStart);
            }
            PreloadBlockTable(task);
            bool needLoadQ = true;
            bool qSynced = false;
            for (uint32_t st = task.sStileBegin; st < task.sStileEnd; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                // BSND/BNSD 的 Q 只在本任务第一拍收集。之后各 stile 只在 K scratch 需要时再同步。
                const bool needQSync = NeedQueryPrepare() && !qSynced;
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
        bool balanceWaves = BalanceCausalTaskWaves(coreNum, totalTasks);
        uint32_t wave = 0U;
        for (uint32_t taskSlot = coreIdx; taskSlot < totalTasks; taskSlot += coreNum, ++wave) {
            const uint32_t taskIdx = TaskOfWave(taskSlot, wave, coreIdx, coreNum, balanceWaves);
            scheduler_.Decode(taskIdx, task);
            if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && USE_C2UB) {
                // Wave zero is always the identity mapping. Reuse its decoded
                // device lengths once; nearly rectangular requests stay in the
                // original order without introducing a host length sync.
                if (wave == 0U && balanceWaves) {
                    balanceWaves = task.kvLen > 0 && task.qLen >= task.kvLen / 2 + task.kvLen % 2;
                }
            }
            if (task.sBlkBegin >= task.sBlkEnd) {
                continue;
            }
            // block_table 按 256B 预取，盖住后续逐页 GetValue 的 HBM 延迟。
            if constexpr (IS_MXFP4) {
                blockMmad.SetBnsdQueryGather(DirectBnsdQuery() ? tiling_->numQHeads : 0U, tiling_->qMaxSeq,
                                             task.mStart);
            }
            PreloadBlockTable(task);
            bool needLoadQ = true;
            uint32_t pageSeq = 0;
            if constexpr (IS_BLOCK16 && !IS_MXFP4) {
                // Q 一任务只打包一次。K 旗只用于 scratch 窗，直读窗不占这面旗。
                if (PackQuery()) {
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd; ++win) {
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    if (!NativeWindowDirectGm(task, win)) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                            MSA_A5_AIV_FLAG_OFFSET);
                    auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                    if (NativeWindowDirectGm(task, win)) {
                        NativeComputeWindowFromGm(blockMmad, task, win, ubS, needLoadQ);
                    } else {
                        NativeComputeWindowToUb(blockMmad, task, ubS, KScratchSlot(coreKScratch, ping), needLoadQ);
                    }
                    needLoadQ = false;
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping +
                                                                           MSA_A5_AIV_FLAG_OFFSET);
                    ++pageSeq;
                }
                continue;
            }
            if constexpr (IS_BLOCK16 && IS_MXFP4) {
                // direct 后续窗跳过 FLAG_K；earlyCv 在 Fixpipe 两路间放行 AIV；
                // 本窗结束后预取下一窗 K→L1（短 seq 中性，长 seq 有收益）。
                // 注意：单级 L0C 下「Mmad 先于 Wait VC」实测劣化（mte2↑），勿再套 block128 次序。
                bool bReady = false;
                uint32_t readyNWin = 1U;
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd;) {
                    const uint32_t nWin = Block16WinGroup(task, win);
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    const bool direct = WindowKeyDirectGm(task, win);
                    if (bReady && (readyNWin != nWin || !direct)) {
                        bReady = false;
                    }
                    const bool needK = !direct || ((pageSeq == 0U) && (!IS_MXFP4 || NeedQueryPrepare()));
                    if (needK) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                            MSA_A5_AIV_FLAG_OFFSET);
                    auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * Epilogue::S_PING_BYTES);
                    if (nWin > 1U || direct) {
                        ComputeWindowDirectGm(blockMmad, task, win, ubS, needLoadQ, nWin > 1U ? nWin : 1U, ping,
                                              bReady);
                    }
                    needLoadQ = false;
                    bReady = false;
                    const uint32_t nextWin = win + nWin;
                    if constexpr (IS_MXFP4) {
                        if (nextWin < task.sStileEnd && WindowKeyDirectGm(task, nextWin)) {
                            readyNWin = Block16WinGroup(task, nextWin);
                            LoadWindowToL1(blockMmad, task, nextWin, readyNWin);
                            bReady = true;
                        }
                    }
                    ++pageSeq;
                    win += nWin;
                }
                continue;
            }
            // BSND/BNSD：本 M-tile 的 Q 只收集一次，进 block 循环前等完。
            // 不能并进每个 block 的 K scratch 旗，否则满页不等 K 时 Cube 会多等一拍。
            if (NeedQueryPrepare()) {
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            if constexpr (IS_MXFP4) {
                // VC 下沉：LoadB → MMAD 到空闲 L0C → 等 VC → Fixpipe。
                // 双 L0C 让本组 Cube/MTE1 可与上一组 AIV Vector 重叠；CV 仍在 Store 之后发。
                bool bReady = false;
                uint32_t nPages = 1U;
                uint32_t gemmCols = MSA_BLOCK_SIZE;
                uint32_t loadCols = MSA_BLOCK_SIZE;
                for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd;) {
                    if (!bReady) {
                        MxFp4Consume(task, blk, nPages, gemmCols, loadCols);
                        if (!KeyBlockNeedsScratch(task, blk)) {
                            LoadPageToL1(blockMmad, task, blk, gemmCols, loadCols);
                            bReady = true;
                        }
                    }
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    MmadPageToL0C(blockMmad, task, blk, coreKScratch, needLoadQ, gemmCols, loadCols, bReady);
                    needLoadQ = false;
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                            MSA_A5_AIV_FLAG_OFFSET);
                    auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                    // 提前 CV：FlushStore 内在两路 F322F16 之间分别发 CV0/CV1。
                    blockMmad.FlushStoreToUb(ubS, ping, true);
                    ++pageSeq;
                    blk += nPages;
                    bReady = false;
                    if (blk < task.sBlkEnd) {
                        MxFp4Consume(task, blk, nPages, gemmCols, loadCols);
                        if (!KeyBlockNeedsScratch(task, blk)) {
                            LoadPageToL1(blockMmad, task, blk, gemmCols, loadCols);
                            bReady = true;
                        }
                    }
                }
            } else {
                for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd; ++blk) {
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    if (KeyBlockNeedsScratch(task, blk)) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                        // 与 GM 路径一致：等 AIV MTE3 写完 scratch 再 MTE2 进 L1，否则 int8/TND 读到 0。
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    if constexpr (IS_QUANT_V) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                                MSA_A5_AIV_FLAG_OFFSET);
                    }
                    auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                    ComputePageToUb(blockMmad, task, blk, ubS, coreKScratch, needLoadQ,
                                    IS_QUANT_V ? -1 : static_cast<int32_t>(ping));
                    needLoadQ = false;
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping +
                                                                           MSA_A5_AIV_FLAG_OFFSET);
                    ++pageSeq;
                }
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
        if constexpr (IS_MXFP4) {
            // Packed FP4 is read directly; partial pages limit loadCols and are masked before reduction.
            return false;
        }
        if (!MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            return false;
        }
        const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
        return (nValidTok > 0) && (nValidTok < MSA_BLOCK_SIZE) && (task.batchIdx + 1U == tiling_->batch);
    }

    // 每个 AIV 的 fp16 行要放进 32KB ping。M=128 时四页是 64KB，退回两页；短 M 用四页。
    __aicore__ inline bool MxFp4GroupFits(const MsaTask& task, uint32_t blk, uint32_t pages) const
    {
        uint32_t m0 = 0U;
        uint32_t m1 = 0U;
        MsaC2UbHalfSplit(task.mActual, m0, m1);
        const uint32_t rows = m0 > m1 ? m0 : m1;
        const uint32_t room = MSA_BLOCKS_PER_STILE - (blk % MSA_BLOCKS_PER_STILE);
        const uint32_t remain = (blk < task.sBlkEnd) ? (task.sBlkEnd - blk) : 0U;
        const uint64_t bytes = static_cast<uint64_t>(rows) * pages * MSA_BLOCK_SIZE * sizeof(half);
        return pages <= room && pages <= remain && bytes <= MSA_A5_S_PING_BYTES;
    }

    __aicore__ inline uint32_t MxFp4PageGroup(const MsaTask& task, uint32_t blk) const
    {
        if constexpr (!IS_MXFP4) {
            return 1U;
        }
        if constexpr (MXFP4_N512) {
            if (MxFp4GroupFits(task, blk, 4U)) {
                return 4U;
            }
        }
        if (MxFp4GroupFits(task, blk, 2U)) {
            return 2U;
        }
        return 1U;
    }

    // 分页时物理页号必须连续，这一段才能合成一条 MTE。TND 的 token 本来就连续。
    __aicore__ inline int32_t PhysicalPageOfBlock(const MsaTask& task, uint32_t blk) const
    {
        const uint32_t pSize = tiling_->pageSize > 0U ? tiling_->pageSize : tiling_->poolBlockSize;
        const uint32_t tWidth = tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
        const uint32_t tok = blk * tiling_->poolBlockSize;
        uint32_t pageIdx = pSize == 0U ? 0U : tok / pSize;
        const uint32_t maxPageIdx = tWidth > 0U ? tWidth - 1U : 0U;
        if (pageIdx > maxPageIdx) {
            pageIdx = maxPageIdx;
        }
        return gBlockTable_.GetValue(task.batchIdx * tWidth + pageIdx);
    }

    __aicore__ inline uint32_t DirectPageRun(const MsaTask& task, uint32_t blk, uint32_t maxPages) const
    {
        if constexpr (IS_TND_TND) {
            (void)task;
            (void)blk;
            return maxPages;
        } else {
            if (maxPages <= 1U || MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                return maxPages;
            }
            // Consecutive page IDs are not contiguous in memory when the PA cache has a dim0 gap.
            const uint64_t compactPageElems =
                static_cast<uint64_t>(tiling_->pageSize) * tiling_->numKvHeads * tiling_->headDim;
            if (tiling_->strideKvBlock != compactPageElems) {
                return 1U;
            }
            uint32_t run = 1U;
            int32_t prev = PhysicalPageOfBlock(task, blk);
            for (uint32_t i = 1U; i < maxPages; ++i) {
                const int32_t cur = PhysicalPageOfBlock(task, blk + i);
                if (cur != prev + 1) {
                    break;
                }
                prev = cur;
                ++run;
            }
            return run;
        }
    }

    // 整页能合成一组。遇到不满 128 的尾块就停，尾块单独做，避免读出 key 末尾。
    // 序列布局可合并多个逻辑页；尾组仅从GM读取实际可见token，GEMM仍按整组计算。
    __aicore__ inline void MxFp4Consume(const MsaTask& task, uint32_t blk, uint32_t& nPages, uint32_t& gemmCols,
                                        uint32_t& loadCols) const
    {
        if (IS_TND_TND || MsaIsSeqKeyLayout(tiling_->keyLayout)) {
            const uint32_t nWant = MxFp4PageGroup(task, blk);
            nPages = (nWant == 0U) ? 1U : nWant;
            gemmCols = nPages * MSA_BLOCK_SIZE;
            const uint64_t tokenBase = static_cast<uint64_t>(blk) * MSA_BLOCK_SIZE;
            const uint64_t remaining = task.kvLen > 0 && tokenBase < static_cast<uint64_t>(task.kvLen) ?
                                           static_cast<uint64_t>(task.kvLen) - tokenBase :
                                           0U;
            loadCols = remaining < gemmCols ? static_cast<uint32_t>(remaining) : gemmCols;
            return;
        }
        // Gather supports arbitrary physical page order and dim0 gaps, only for one logical block per page.
        // DirectPageRun retains its original physical-contiguity meaning.
        const uint32_t pageGroup = MxFp4PageGroup(task, blk);
        const uint32_t nWant = (!MX_LATENCY_TND && USE_C2UB && !IS_BLOCK16 && tiling_->pageSize == MSA_BLOCK_SIZE &&
                                tiling_->poolBlockSize == MSA_BLOCK_SIZE) ?
                                   pageGroup :
                                   DirectPageRun(task, blk, pageGroup);
        uint32_t nFull = 0U;
        for (uint32_t i = 0U; i < nWant; ++i) {
            if (KeyBlockValidTokens(task, blk + i) < MSA_BLOCK_SIZE) {
                break;
            }
            ++nFull;
        }
        if (nFull == 0U) {
            nPages = 1U;
            gemmCols = MSA_BLOCK_SIZE;
            loadCols = KeyBlockValidTokens(task, blk);
            if (loadCols == 0U) {
                loadCols = MSA_BLOCK_SIZE;
            }
        } else {
            nPages = nFull;
            gemmCols = nFull * MSA_BLOCK_SIZE;
            loadCols = gemmCols;
        }
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
        if constexpr (IS_TND_TND) {
            const uint32_t seqTok = blk * MSA_BLOCK_SIZE;
            if (task.kvLen <= 0 || seqTok >= static_cast<uint32_t>(task.kvLen)) {
                return 0U;
            }
            const uint32_t remain = static_cast<uint32_t>(task.kvLen) - seqTok;
            return remain < MSA_BLOCK_SIZE ? remain : MSA_BLOCK_SIZE;
        } else {
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
        if constexpr (IS_TND_TND) {
            return static_cast<uint64_t>(task.cuKStart + blk * MSA_BLOCK_SIZE) * tiling_->numKvHeads * tiling_->headDim;
        } else {
            if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                return MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart,
                                          blk * tiling_->poolBlockSize, tiling_->numKvHeads, tiling_->headDim,
                                          tiling_->kMaxSeq);
            }
            const uint32_t tok = blk * tiling_->poolBlockSize;
            const uint32_t pSize = tiling_->pageSize > 0U ? tiling_->pageSize : tiling_->poolBlockSize;
            const uint32_t tWidth = tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
            uint32_t pageIdx = tok / pSize;
            const uint32_t maxPageIdx = tWidth > 0U ? tWidth - 1U : 0U;
            if (pageIdx > maxPageIdx) {
                pageIdx = maxPageIdx;
            }
            const int32_t page = gBlockTable_.GetValue(task.batchIdx * tWidth + pageIdx);
            return static_cast<uint64_t>(page) * tiling_->strideKvBlock +
                   static_cast<uint64_t>(tok % pSize) * tiling_->strideKvToken;
        }
    }

    // key may have gaps between physical PA pages; AutoContiguous key_scale never has those gaps.
    __aicore__ inline uint64_t MxKeyScaleOffset(uint64_t keyElem) const
    {
        if constexpr (IS_TND_TND) {
            return keyElem / MSA_MX_GROUP;
        } else {
            if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                return keyElem / MSA_MX_GROUP;
            }
            const uint64_t pageStride = tiling_->strideKvBlock;
            const uint64_t compactPageElems =
                static_cast<uint64_t>(tiling_->pageSize) * tiling_->numKvHeads * tiling_->headDim;
            if (pageStride == compactPageElems) {
                return keyElem / MSA_MX_GROUP;
            }
            const uint64_t compactElem = (keyElem / pageStride) * compactPageElems + keyElem % pageStride;
            return compactElem / MSA_MX_GROUP;
        }
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

    // Preserve the merged PR15456 native window loader and scratch pipeline.
    __aicore__ inline bool NativeWindowDirectGm(const MsaTask& task, uint32_t win) const
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

#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    __aicore__ inline void ComputePageToUb(BlockMmad& blockMmad, const MsaTask& task, uint32_t blk,
                                           const AscendC::LocalTensor<float>& ubS, uint64_t kScratchBase,
                                           bool needLoadL1, int32_t vcPing)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const auto queryGm = QueryGm(task);
        if (KeyBlockNeedsScratch(task, blk)) {
            blockMmad.ComputeToUbNative(queryGm, layoutQ, gKeyScratch_[kScratchBase], layoutKScratch, ubS, shape,
                                        needLoadL1, 1U, vcPing);
        } else if constexpr (!IS_QUANT_V) {
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            blockMmad.ComputeToUbNative(queryGm, layoutQ, gKeyFp_[kOffset], layoutKGm, ubS, shape, needLoadL1, 1U,
                                        vcPing);
        }
    }

    __aicore__ inline void NativeComputeWindowFromGm(BlockMmad& blockMmad, const MsaTask& task, uint32_t win,
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

    __aicore__ inline void NativeComputeWindowToUb(BlockMmad& blockMmad, const MsaTask& task,
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

#endif

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

    /// 块长 16 且 key 按 token 连续摆放时，一窗 8 个逻辑块就是 GM 上连续的 128 个 token，
    /// 其 layout 与 scratch 的 (kGemm, 128, kGemm) 完全一致，可以让 Cube 直接读原 key。
    /// MXFP4 tails load valid tokens only; non-MX tails retain the scratch fallback.
    __aicore__ inline bool WindowKeyDirectGm(const MsaTask& task, uint32_t win) const
    {
        if constexpr (!IS_BLOCK16) {
            return false;
        }
        if constexpr (IS_MXFP4) {
            // Host guarantees the direct-loader layout/stride contract. Scheduler
            // only emits windows intersecting valid keys; tails use bounded loads.
            return true;
        }
        if (!MsaIsSeqKeyLayout(tiling_->keyLayout) || tiling_->strideKvToken != tiling_->headDim) {
            return false;
        }
        const uint32_t blkEnd = (win + 1U) * MSA_BLOCKS_PER_STILE;
        if (blkEnd > task.visibleEndBlk || task.kvLen <= 0) {
            return false;
        }
        return static_cast<uint64_t>(blkEnd) * MSA_BLOCK16_SIZE <= static_cast<uint64_t>(task.kvLen);
    }

    /// 一组窗的 S 要放进一个 S_PING_BYTES_B16 的 ping（C2UB 后是 fp16），行数按两个 AIV 中较多的那半算。
    __aicore__ inline bool Block16GroupFits(const MsaTask& task, uint32_t nWin) const
    {
        uint32_t m0 = 0U;
        uint32_t m1 = 0U;
        // Fixpipe keeps a token's heads together; H128 can put all rows on one AIV.
        MsaC2UbGroupSplit(task.mActual, tiling_->numQHeads, m0, m1);
        const uint32_t rows = m0 > m1 ? m0 : m1;
        const uint64_t bytes = static_cast<uint64_t>(rows) * nWin * MSA_BLOCK_SIZE * sizeof(half);
        return bytes <= MSA_A5_S_PING_BYTES_B16;
    }

    /// 从 win 起能并成一次 MMAD 的窗数。block128 靠 MxFp4PageGroup 一次吃 2~4 页，block16 原本
    /// 固定一窗，于是每 128 列就要付一整轮跨核握手和 epilogue 尾处理，窗数多出几倍。
    /// Sequence windows are contiguous; MXFP4 PA windows gather pages directly into L1.
    /// AIC and AIV must use the same grouping and actual head-group split for UB capacity.
    __aicore__ inline uint32_t Block16WinGroup(const MsaTask& task, uint32_t win) const
    {
        if constexpr (!IS_BLOCK16 || !IS_MXFP4) {
            return 1U;
        }
        // 组起点对齐到 MSA_BLOCK16_MAX_WIN_GROUP，这样组覆盖的 block 区间不会跨过 score 暂存窗
        // （窗宽 MSA_STAGE_BLOCKS）的边界，AdvanceStageWindow 只需在组起点判一次。
        if (((win % MSA_BLOCK16_MAX_WIN_GROUP) != 0U) || !WindowKeyDirectGm(task, win)) {
            return 1U;
        }
        uint32_t n = 1U;
        while (n < MSA_BLOCK16_MAX_WIN_GROUP && (win + n) < task.sStileEnd && WindowKeyDirectGm(task, win + n) &&
               Block16GroupFits(task, n + 1U)) {
            ++n;
        }
        return n;
    }

    /// AIV（块长 16）：把一窗 8 个逻辑块收成 128 个 token。
    /// BBND/BNBD 经 block_table 寻址；BSND/BNSD 按连续 token 寻址。越界补零。
    __aicore__ inline void GatherKeyWindowToScratch(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                    uint32_t win, uint64_t scratchOff)
    {
        const uint32_t headDim = tiling_->headDim;
        const uint32_t tokenBytes = headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * tokenBytes;
        AscendC::LocalTensor<uint8_t> ubBytes = resource.ubBuf.template GetBufferByByte<uint8_t>(UB_OFF_CAST_FP);
        AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
        if constexpr (sizeof(ElementQ) == 1U) {
            AscendC::Duplicate(ubBytes, static_cast<uint8_t>(0), nElem);
        } else {
            AscendC::Duplicate(ubK, static_cast<ElementQ>(0), MSA_BLOCK_SIZE * headDim);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        const uint32_t blkBase = win * MSA_BLOCKS_PER_STILE;
        const uint32_t pSize = tiling_->pageSize > 0U ? tiling_->pageSize : MSA_BLOCK16_SIZE;
        const uint32_t tWidth = tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
        uint32_t blkLimit = tiling_->maxBlocksPerBatch;
        if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
            const uint32_t paBlocks = tWidth * (pSize / MSA_BLOCK16_SIZE);
            if (paBlocks > 0U && paBlocks < blkLimit) {
                blkLimit = paBlocks;
            }
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
                const int32_t page = gBlockTable_.GetValue(task.batchIdx * tWidth + tok / pSize);
                kOffset = static_cast<uint64_t>(page) * tiling_->strideKvBlock +
                          static_cast<uint64_t>(tok % pSize) * tiling_->strideKvToken;
            }
            if (tiling_->strideKvToken == headDim) {
                CopyGmToUbPartial(ubK[j * MSA_BLOCK16_SIZE * headDim], gKeyFp_[kOffset], nTok * headDim);
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
                AscendC::DataCopyPad(ubK[j * MSA_BLOCK16_SIZE * headDim], gKeyFp_[kOffset], params, pad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);

        AscendC::DataCopy(gKeyScratch_[scratchOff], ubK, MSA_BLOCK_SIZE * headDim);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
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

    __aicore__ inline bool DirectBnsdQuery() const
    {
        return IS_MXFP4 && tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD && tiling_->qMaxSeq > 1U &&
               tiling_->numQHeads > 1U;
    }

    __aicore__ inline bool NeedMxScaleL1() const
    {
        return DirectBnsdQuery() && (tiling_->numQHeads % 16U != 0U);
    }

    __aicore__ inline bool NeedQueryPrepare() const
    {
        return PackQuery() || NeedMxScaleL1();
    }

    /// MXFP4 Q never uses a GM preprocessing result.
    __aicore__ inline bool PackQuery() const
    {
        if constexpr (IS_TND_TND) {
            return false;
        } else {
            if constexpr (IS_MXFP4) {
                return false;
            }
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BSND || tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD;
        }
    }

    __aicore__ inline uint64_t QueryElemOffset(const MsaTask& task) const
    {
        if constexpr (IS_TND_TND) {
            return static_cast<uint64_t>(task.cuQStart * tiling_->numQHeads + task.mStart) * tiling_->strideQn;
        } else {
            if (PackQuery()) {
                return static_cast<uint64_t>(AscendC::GetBlockIdx()) * MSA_ROW_TILE_M * tiling_->headDim;
            }
            if constexpr (IS_MXFP4) {
                if (tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD) {
                    const uint64_t base = static_cast<uint64_t>(task.batchIdx) * tiling_->numQHeads * tiling_->qMaxSeq;
                    if (DirectBnsdQuery()) {
                        return base * tiling_->headDim;
                    }
                    return (base + static_cast<uint64_t>(task.mStart % tiling_->numQHeads) * tiling_->qMaxSeq +
                            task.mStart / tiling_->numQHeads) *
                           tiling_->headDim;
                }
            }
            return MsaQueryTileElem(tiling_->queryLayout, task.batchIdx, task.cuQStart, task.mStart, tiling_->numQHeads,
                                    tiling_->headDim, tiling_->qMaxSeq, tiling_->strideQn);
        }
    }

    __aicore__ inline AscendC::GlobalTensor<ElementQ> QueryGm(const MsaTask& task) const
    {
        uint64_t elem = QueryElemOffset(task);
        if constexpr (IS_MXFP4) {
            elem >>= 1;
        }
        if (PackQuery()) {
            return gQScratch_[elem];
        }
        return gQuery_[elem];
    }

    __aicore__ inline AscendC::GlobalTensor<fp8_e8m0_t> QueryScaleGm(const MsaTask& task) const
    {
        const uint64_t scaleOff = QueryElemOffset(task) / MSA_MX_GROUP;
        return gQueryScaleMx_[scaleOff];
    }

    /// BSND 以及 N1=1 的 BNSD：逻辑行已经连续，一次搬完。BNSD 且 N1>1 时按 (token, head) 逐行收。
    __aicore__ inline bool QueryRowsContiguous() const
    {
        if constexpr (IS_TND_TND) {
            return true;
        } else {
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BSND || tiling_->numQHeads <= 1U;
        }
    }

    __aicore__ inline uint64_t QuerySrcElem(const MsaTask& task, uint32_t localRow) const
    {
        if constexpr (IS_TND_TND) {
            return static_cast<uint64_t>(task.cuQStart * tiling_->numQHeads + localRow) * tiling_->strideQn;
        } else {
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
    }

    // Only the scale tile needs AIV repacking when BNSD head segments do not
    // start on16-row NZ boundaries. Q data is still gathered by AIC GM->L1.
    __aicore__ inline void PrepareQuery(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                        uint64_t scratch)
    {
        if constexpr (IS_MXFP4) {
            if (!NeedMxScaleL1()) {
                return;
            }
            const uint32_t groups = tiling_->headDim / MSA_MX_GROUP;
            const uint32_t rows = (task.mActual + 15U) & ~15U;
            auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(UB_OFF_CAST_I8);
            AscendC::Duplicate(ub, static_cast<uint8_t>(0), rows * groups);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t r = 0U; r < task.mActual; ++r) {
                const uint64_t src = QuerySrcElem(task, task.mStart + r) / MSA_MX_GROUP;
                for (uint32_t g = 0U; g < groups; ++g) {
                    const uint32_t dst = (r / 16U) * (16U * groups) + (g / 2U) * 32U + (r % 16U) * 2U + g % 2U;
                    ub.SetValue(dst, gQueryScaleMx_.template ReinterpretCast<uint8_t>().GetValue(src + g));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            constexpr uint32_t stages = IS_BLOCK16 ? MSA_L1B_STAGES_INT8 : MSA_L1B_STAGES_FP16;
            constexpr uint32_t offset = MsaMxL1Layout<stages, MXFP4_N512>::SCALE_A_OFFSET;
            // Resource::l1Buf is intentionally unbound on AIV; use an explicit
            // A1 handle for the paired AIC's shared L1 address.
            AscendC::LocalTensor<uint8_t> l1(AscendC::TPosition::A1, offset, rows * groups);
            AscendC::DataCopy(l1, ub, rows * groups);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        } else {
            GatherQueryTile(resource, task, scratch);
        }
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
        const uint32_t kGemm = IS_MXFP4 ? (headDim >> 1) : headDim;
        const uint32_t strideQ = IS_MXFP4 ? (tiling_->strideQn >> 1) : tiling_->strideQn;
        const uint32_t strideK = IS_MXFP4 ? (tiling_->strideKvToken >> 1) : tiling_->strideKvToken;
        const LayoutQ layoutQ(mActual, kGemm, strideQ);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(kGemm, MSA_BLOCK_SIZE, strideK);
        const LayoutS layoutS(mActual, MSA_BLOCK_SIZE, STILE_WIDTH);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, kGemm};
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
                const uint64_t kValue = IS_MXFP4 ? (kOffset >> 1) : kOffset;
                blockMmad(queryGm, layoutQ, gKeyFp_[kValue], layoutKGm,
                          gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape, needLoadL1);
            }
            needLoadL1 = false;
        }
    }

    __aicore__ inline void LoadPageToL1(BlockMmad& blockMmad, const MsaTask& task, uint32_t blk, uint32_t nCols,
                                        uint32_t loadCols)
    {
        // 热字段进局部，避免循环内反复 tiling_-> 间接读。
        const uint32_t headDim = tiling_->headDim;
        const uint32_t strideKvToken = tiling_->strideKvToken;
        const uint32_t kGemm = headDim >> 1;
        const uint32_t strideK = strideKvToken >> 1;
        const uint32_t nLoad = (loadCols == 0U || loadCols > nCols) ? nCols : loadCols;
        const LayoutK layoutKGm(kGemm, nLoad, strideK);
        const Catlass::GemmCoord shape{task.mActual, nCols, kGemm};
        if constexpr (IS_MXFP4 && !IS_BLOCK16 && USE_C2UB && !MX_LATENCY_TND) {
            const uint32_t pages = nCols / MSA_BLOCK_SIZE;
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && tiling_->pageSize == MSA_BLOCK_SIZE &&
                tiling_->poolBlockSize == MSA_BLOCK_SIZE && pages > 1U && DirectPageRun(task, blk, pages) < pages) {
                const LayoutK layoutPage(kGemm, MSA_BLOCK_SIZE, strideK);
                for (uint32_t i = 0U; i < pages; ++i) {
                    const uint64_t pageKey = KeyBlockGmOffset(task, blk + i);
                    const uint64_t pageScale = MxKeyScaleOffset(pageKey);
                    blockMmad.LoadBPageToL1(gKeyFp_[pageKey >> 1], layoutPage, gKeyScaleMx_[pageScale], kGemm,
                                            i * MSA_BLOCK_SIZE, i == 0U, i + 1U == pages);
                }
                return;
            }
        }
        const uint64_t kOffset = KeyBlockGmOffset(task, blk);
        const uint64_t kScaleOff = MxKeyScaleOffset(kOffset);
        blockMmad.LoadBToL1(gKeyFp_[kOffset >> 1], layoutKGm, gKeyScaleMx_[kScaleOff], shape, nLoad);
    }

    /// block16 direct-GM：把 nWin 个连续窗的 K（及 MX scale）预取进 L1B，供下一轮 bLoaded 复用。
    __aicore__ inline void LoadPaWindowToL1(BlockMmad& blockMmad, const MsaTask& task, uint32_t win, uint32_t nWin)
    {
        const uint32_t d = tiling_->headDim;
        const uint32_t kp = d / 2U;
        const uint32_t page = tiling_->pageSize;
        const uint32_t cols = Min(nWin * MSA_BLOCK_SIZE, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE);
        for (uint32_t n = 0U; n < cols; n += page) {
            const uint32_t rows = Min(page, cols - n);
            const LayoutK srcLayout(kp, rows, tiling_->strideKvToken / 2U);
            const uint32_t token = win * MSA_BLOCK_SIZE + n;
            const int32_t id = gBlockTable_.GetValue(task.batchIdx * tiling_->tableWidth + token / page);
            const uint64_t off = static_cast<uint64_t>(id) * tiling_->strideKvBlock;
            blockMmad.LoadBPageToL1(gKeyFp_[off / 2U], srcLayout, gKeyScaleMx_[MxKeyScaleOffset(off)], kp, n, n == 0U,
                                    n + rows == cols, rows);
        }
    }

    __aicore__ inline void LoadWindowToL1(BlockMmad& blockMmad, const MsaTask& task, uint32_t win, uint32_t nWin)
    {
        if constexpr (!IS_BLOCK16 || !IS_MXFP4) {
            return;
        }
        if constexpr (IS_MXFP4 && IS_BLOCK16) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
                LoadPaWindowToL1(blockMmad, task, win, nWin);
                return;
            }
        }
        const uint32_t headDim = tiling_->headDim;
        const uint32_t kGemm = IS_MXFP4 ? (headDim >> 1) : headDim;
        const uint32_t strideK = IS_MXFP4 ? (tiling_->strideKvToken >> 1) : tiling_->strideKvToken;
        const uint32_t nCols = nWin * MSA_BLOCK_SIZE;
        const LayoutK layoutKGm(kGemm, nCols, strideK);
        const Catlass::GemmCoord shape{task.mActual, nCols, kGemm};
        const uint64_t kOff = MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, win * MSA_BLOCK_SIZE,
                                                 tiling_->numKvHeads, headDim, tiling_->kMaxSeq);
        blockMmad.LoadBToL1(gKeyFp_[IS_MXFP4 ? (kOff >> 1) : kOff], layoutKGm, gKeyScaleMx_[MxKeyScaleOffset(kOff)],
                            shape, Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE));
    }

    /// VC 下沉前半段：只把 QK 算进空闲 L0C（bLoaded 时不再搬 K）。
    __aicore__ inline void MmadPageToL0C(BlockMmad& blockMmad, const MsaTask& task, uint32_t blk, uint64_t kScratchBase,
                                         bool needLoadL1, uint32_t nCols, uint32_t loadCols, bool bLoaded)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const uint32_t kGemm = IS_MXFP4 ? (headDim >> 1) : headDim;
        const uint32_t strideQ = IS_MXFP4 ? (tiling_->strideQn >> 1) : tiling_->strideQn;
        const uint32_t strideK = IS_MXFP4 ? (tiling_->strideKvToken >> 1) : tiling_->strideKvToken;
        const uint32_t nLoad = (loadCols == 0U || loadCols > nCols) ? nCols : loadCols;
        const LayoutQ layoutQ(mActual, kGemm, strideQ);
        const LayoutK layoutKGm(kGemm, nLoad, strideK);
        const Catlass::GemmCoord shape{mActual, nCols, kGemm};
        const auto queryGm = QueryGm(task);
        if constexpr (IS_MXFP4) {
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            const uint64_t kScaleOff = MxKeyScaleOffset(kOffset);
            blockMmad.MmadToL0CMx(queryGm, layoutQ, gKeyFp_[kOffset >> 1], layoutKGm, QueryScaleGm(task),
                                  gKeyScaleMx_[kScaleOff], shape, needLoadL1, 1U, nLoad, bLoaded);
        } else {
            (void)kScratchBase;
        }
    }

    /// MXFP4 块长 16：Cube 直读原 key，分页 K 按页装入 L1，不经过 GM scratch。
    /// Fixpipe 按 numQHeads 整组切给两个 AIV。
    /// nWin 个连续 direct-GM 窗并成一次 MMAD：它们的 key token 在 GM 上首尾相接，只需把 N 加宽。
    /// MXFP4_N512：L0C 单级吃满 mActual×512，不再 M-split。earlyCv：Fixpipe 两路之间放行 AIV。
    /// bLoaded：上一轮已 LoadWindowToL1，跳过本轮 GM→L1B。
    __aicore__ inline void ComputeWindowDirectGm(BlockMmad& blockMmad, const MsaTask& task, uint32_t win,
                                                 const AscendC::LocalTensor<float>& ubS, bool needLoadL1, uint32_t nWin,
                                                 uint32_t cvPing = 0U, bool bLoaded = false)
    {
        if constexpr (!IS_BLOCK16) {
            return;
        }
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const uint32_t kGemm = IS_MXFP4 ? (headDim >> 1) : headDim;
        const uint32_t strideQ = IS_MXFP4 ? (tiling_->strideQn >> 1) : tiling_->strideQn;
        const uint32_t strideK = IS_MXFP4 ? (tiling_->strideKvToken >> 1) : tiling_->strideKvToken;
        const uint32_t nCols = nWin * MSA_BLOCK_SIZE;
        const LayoutK layoutKGm(kGemm, nCols, strideK);
        const uint32_t split = tiling_->numQHeads;
        const uint64_t kOff = MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, win * MSA_BLOCK_SIZE,
                                                 tiling_->numKvHeads, headDim, tiling_->kMaxSeq);
        const LayoutQ layoutQ(mActual, kGemm, strideQ);
        const Catlass::GemmCoord shape{mActual, nCols, kGemm};
        if constexpr (IS_MXFP4) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && !bLoaded) {
                LoadPaWindowToL1(blockMmad, task, win, nWin);
                bLoaded = true;
            }
        }
        if constexpr (IS_MXFP4) {
            blockMmad.ComputeToUbMx(QueryGm(task), layoutQ, gKeyFp_[IS_MXFP4 ? (kOff >> 1) : kOff], layoutKGm,
                                    QueryScaleGm(task), gKeyScaleMx_[MxKeyScaleOffset(kOff)], ubS, shape, needLoadL1,
                                    split, Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE),
                                    bLoaded, true, cvPing);
        } else {
            blockMmad.ComputeToUb(QueryGm(task), layoutQ, gKeyFp_[kOff], layoutKGm, ubS, shape, needLoadL1, split, 0U,
                                  bLoaded, true, cvPing);
        }
    }

#endif

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        Epilogue epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->tableWidth, tiling_->poolBlockSize, tiling_->pageSize,
                      tiling_->strideScalePage, tiling_->strideScaleHead, IS_QUANT_V,
                      IS_TND_TND ? MSA_KEY_LAYOUT_TND : tiling_->keyLayout, tiling_->totalK, gScore_, gScale_,
                      gBlockTable_);
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
        if constexpr (ENABLE_TOPK) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
            MsaIndexScoreTopk topk;
            // 阶段 A 已经结束，阶段 B 复用整段 UB 做 Sort32。只在 ENABLE_TOPK 实例里走到。
            auto ubTrunk = resource.ubBuf.template GetBufferByByte<float>(0);
            topk.Run(tiling_, gScore_, gSparseIndices_, gSparseValues_, gActualSeqQlen_, gActualSeqKlen_,
                     tiling_->returnChoice == MSA_RETURN_INDICES_VALUES, ubTrunk);
#endif
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
            const bool gatherQ = NeedQueryPrepare();
            bool qSynced = false;
            const uint64_t qScratch = static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim;
            for (uint32_t st = task.sStileBegin; st < task.sStileEnd; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                const bool needQSync = gatherQ && !qSynced;
                if (needKScratch || needQSync) {
                    if (subIdx == 0) {
                        if (needQSync) {
                            PrepareQuery(resource, task, qScratch);
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
        bool balanceWaves = BalanceCausalTaskWaves(coreNum, totalTasks);
        uint32_t wave = 0U;
        for (uint32_t taskSlot = coreIdx; taskSlot < totalTasks; taskSlot += coreNum, ++wave) {
            const uint32_t taskIdx = TaskOfWave(taskSlot, wave, coreIdx, coreNum, balanceWaves);
            scheduler_.Decode(taskIdx, task);
            if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && USE_C2UB) {
                // Wave zero is always the identity mapping. Reuse its decoded
                // device lengths once; nearly rectangular requests stay in the
                // original order without introducing a host length sync.
                if (wave == 0U && balanceWaves) {
                    balanceWaves = task.kvLen > 0 && task.qLen >= task.kvLen / 2 + task.kvLen % 2;
                }
            }
            if (task.sBlkBegin >= task.sBlkEnd && !task.writeTailFill) {
                continue;
            }
            epilogue.BeginTask(task, subIdx, subBlockNum);
            uint32_t pageSeq = 0;
            if (task.sBlkBegin < task.sBlkEnd && NeedQueryPrepare() && !IS_BLOCK16) {
                if (subIdx == 0) {
                    PrepareQuery(resource, task, static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim);
                }
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
            }
            if constexpr (IS_BLOCK16 && !IS_MXFP4) {
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
                    if (!NativeWindowDirectGm(task, win) && !prefetched) {
                        if (subIdx == 0) {
                            GatherKeyWindowToScratch(resource, task, win, KScratchSlot(coreKScratch, ping));
                        }
                        SignalKeyReady();
                    }
                    prefetched = false;
                    if (win + 1U < task.sStileEnd && !NativeWindowDirectGm(task, win + 1U)) {
                        const uint32_t nextPing = (pageSeq + 1U) % MSA_A5_S_STAGES;
                        if (subIdx == 0) {
                            GatherKeyWindowToScratch(resource, task, win + 1U, KScratchSlot(coreKScratch, nextPing));
                        }
                        SignalKeyReady();
                        prefetched = true;
                    }
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                    epilogue.ProcessNativeWindowUb(ping, task, win);
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + ping);
                    ++pageSeq;
                }
                epilogue.EndTask();
                continue;
            }
            if constexpr (IS_BLOCK16 && IS_MXFP4) {
                // Q 在一个 task 内按需整理 scale 到 L1，首窗用 FLAG_K 同步；K 由 Cube 直搬 L1。
                if (subIdx == 0 && NeedQueryPrepare() && task.sStileBegin < task.sStileEnd) {
                    PrepareQuery(resource, task, static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim);
                }
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd;) {
                    const uint32_t nWin = Block16WinGroup(task, win);
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    const bool direct = WindowKeyDirectGm(task, win);
                    if (subIdx == 0 && !direct) {
                        GatherKeyWindowToScratch(resource, task, win, coreKScratch);
                    }
                    const bool needK = !direct || ((pageSeq == 0U) && (!IS_MXFP4 || NeedQueryPrepare()));
                    if (needK) {
                        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
                    }
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                    // ProcessWindowUb 在读完 S ping 后提前 ReleaseSPing，勿再 Set VC。
                    (void)epilogue.ProcessWindowUb(ping, task, win, nWin);
                    ++pageSeq;
                    win += nWin;
                }
                epilogue.EndTask();
                continue;
            }
            if constexpr (IS_MXFP4) {
                for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd;) {
                    uint32_t nPages = 1U;
                    uint32_t gemmCols = MSA_BLOCK_SIZE;
                    uint32_t loadCols = MSA_BLOCK_SIZE;
                    MxFp4Consume(task, blk, nPages, gemmCols, loadCols);
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                    if (nPages > 1U) {
                        epilogue.ProcessPageGroupUb(ping, task, blk, nPages);
                    } else {
                        epilogue.ProcessPageUb(ping, task, blk);
                    }
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + ping);
                    ++pageSeq;
                    blk += nPages;
                }
            } else {
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
    AscendC::GlobalTensor<fp8_e8m0_t> gQueryScaleMx_;
    AscendC::GlobalTensor<fp8_e8m0_t> gKeyScaleMx_;
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

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_KERNEL_ARCH35_H
