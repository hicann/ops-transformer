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

#include "catlass/msa_catlass.hpp"
#include "catlass/arch/msa_arch.hpp"
#include "catlass/arch/msa_resource.hpp"
#include "catlass/arch/msa_cross_core_sync.hpp"
#include "catlass/msa_coord.hpp"
#include "catlass/msa_gemm_coord.hpp"
#include "catlass/msa_matrix_coord.hpp"
#include "catlass/layout/msa_layout.hpp"
#include "catlass/gemm/msa_gemm_type.hpp"

#include "../msa_index_score_common.h"
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
#include "msa_block_mmad_arch35.h"
#endif
#include "../msa_index_score_task.h"
#include "msa_seg_row_max_epilogue_arch35.h"
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
#include "msa_block16_select_arch35.h"
#include "msa_index_score_topk_arch35.h"
#endif

namespace MsaIndexScoreNs {

/// IS_BLOCK16：逻辑块长 MSA_BLOCK16_SIZE。BSND/BNSD、950 fp16/bf16 的 TND，或 BBND/BNBD 页长 16。
/// 一次 128 列 GEMM 覆盖 MSA_BLOCKS_PER_STILE 个逻辑 block（= 一个「窗」），
/// 归约时多做一次 head 间 maxpool。Cube 的 N 仍是 MSA_BLOCK_SIZE，与 128 路径共用全部流水。
template <class ElementQ_, bool IS_QUANT, bool IS_BLOCK16 = false, bool USE_C2UB_IN = (MSA_A5_USE_C2UB != 0),
          bool IS_MXFP4 = false, bool MXFP4_N512 = false, bool ENABLE_TOPK = false, bool MX_FUSED_STAGE = false,
          bool MX_LATENCY_TND = false, bool COMPRESSED_PAGES = false, bool DIRECT_FLOAT = false>
class MsaIndexScoreKernel {
public:
    static constexpr bool IS_TND_TND = MX_LATENCY_TND;
    using ElementQ = ElementQ_;
    static constexpr bool FLOAT_DIRECT =
        DIRECT_FLOAT && IS_BLOCK16 && USE_C2UB_IN && !IS_QUANT && !IS_MXFP4 && sizeof(ElementQ) == sizeof(half);

    using ArchTag = Catlass::Arch::AtlasA5;
    /// 逻辑 block 的 token 数：决定 scheduler 的块算术，与 Cube 的 N 无关。
    static constexpr uint32_t LOGICAL_BLOCK_SIZE = IS_BLOCK16 ? MSA_BLOCK16_SIZE : MSA_BLOCK_SIZE;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementQ, LayoutK>;
    // 块长 16 的 fp16/bf16/MXFP4：Fixpipe 保持 FP32，head max 和块内 max 都在 FP32 上做。
    // MXFP4 宽窗一次 MMAD 仍是 N=512，Fixpipe 按 128 列切片，ping 里只放一片。
    static constexpr bool BLOCK16_FP32_MAX = IS_BLOCK16 && !IS_QUANT;
    // S workspace 元素：非量化 fp16（fixpipe F322F16 直接写，写读流量减半），int8 fp32
    // （S 幅值大、1e-3 容差下 fp16 精度余量不足 15%，保持 fp32 反量化）。
    using ElementS = std::conditional_t<IS_QUANT || BLOCK16_FP32_MAX, float, half>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;

    // 950 非量化、非 TopK 的 FP8 block128：PA 的相邻两个逻辑页可合成一次 MMAD。
    static constexpr bool FP8_SCORE_PIPE =
        USE_C2UB_IN && !IS_QUANT && !IS_MXFP4 && !IS_BLOCK16 && !ENABLE_TOPK && sizeof(ElementQ) == 1U;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    // int8 的 K 源是每 stile 复用的 per-core scratch，保持 2 级 L1B 流水（3 级实测竞态崩溃）；
    // 同样出于该 scratch 的时序保守性，unit flag 只在非量化路径开启。
    using BlockMmad =
        MsaBlockMmad<QType, KType, SType,
                     IS_MXFP4 ? (IS_BLOCK16 ? MSA_L1B_STAGES_INT8 : MSA_L1B_STAGES_FP16) :
                                (IS_QUANT ? MSA_L1B_STAGES_INT8 : (IS_BLOCK16 ? MSA_L1B_STAGES_FP16 : 4U)),
                     false, IS_MXFP4, false, IS_MXFP4, IS_MXFP4 && MXFP4_N512, BLOCK16_FP32_MAX, FLOAT_DIRECT,
                     FP8_SCORE_PIPE, IS_MXFP4 && !IS_BLOCK16 && USE_C2UB_IN && !MX_LATENCY_TND>;
#endif

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static constexpr bool USE_C2UB = USE_C2UB_IN;
    using Epilogue =
        MsaSegRowMaxEpilogue<IS_QUANT_V, IS_BLOCK16, BLOCK16_FP32_MAX, ENABLE_TOPK, IS_MXFP4, MX_FUSED_STAGE>;
    static constexpr uint32_t K_SCRATCH_ELEMS = USE_C2UB ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM;
    // C_to_UB：cast 必须在 epilogue stage 之后。放在 ping 64KB 处会盖住 T64/deq/stage，
    // 下一页 gather 会把尚未 Flush 的 score 清掉（TND/int8 多 page：1e29→0、fill 变 0）。
    static constexpr uint32_t UB_OFF_CAST_I8 =
        USE_C2UB ? (((Epilogue::UB_TOTAL + MSA_UB_ALIGN_BYTES - 1U) / MSA_UB_ALIGN_BYTES) * MSA_UB_ALIGN_BYTES) : 0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    // block16 nWin=4 的 epilogue 占满 S ping 之后，cast 区放不下。块长 16 不跑 int8 cast。
    // 收 Q/K 发生在 Cube 写 S 之前，复用两级 S ping：数据在 ping0，MX scale 在 ping1。
    static constexpr uint32_t UB_OFF_B16_GATHER = 0U;
    static constexpr uint32_t UB_OFF_B16_SCALE = Epilogue::S_PING_BYTES;
    static_assert(IS_BLOCK16 || (UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE), "int8 cast UB out of bounds");
    static_assert(!IS_BLOCK16 || (MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half) <= Epilogue::S_PING_BYTES),
                  "block16 gather does not fit in the S ping");

    __aicore__ inline MsaIndexScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale, GM_ADDR actualSeqQlen,
                                GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score, GM_ADDR sparseIndices,
                                GM_ADDR sparseValues, GM_ADDR workspace,
                                const MsaIndexScoreTilingData* __restrict tiling)
    {
        tiling_ = tiling;
        runtimeScoreStride_ = tiling_->scoreBlockStride;
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
        gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score));
        if constexpr (ENABLE_TOPK) {
            gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + tiling_->scoreWorkspaceElems);
            gTopkIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices));
            if (tiling_->returnChoice == MSA_RETURN_INDICES_VALUES) {
                gTopkVal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sparseValues));
            }
        }
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS*>(workspace));
        if constexpr (IS_BLOCK16) {
            if (sparseIndices != nullptr) {
                gTopkIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices),
                                          tiling_->totalQ * MSA_BLOCK16_TOPK);
            }
            if (sparseValues != nullptr) {
                gTopkVal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sparseValues),
                                          tiling_->totalQ * MSA_BLOCK16_TOPK);
            }
            if (tiling_->returnChoice != MSA_RETURN_SCORE) {
                // Immutable per-request metadata gives every core the same compact row extent.
                if (tiling_->keyLayout == MSA_KEY_LAYOUT_TND) {
                    runtimeScoreStride_ = ResolveRuntimeScoreStride();
                }
                // Same raw FP32 score producer as score-only; only its GM destination differs.
                gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + tiling_->scoreWorkspaceElems);
            }
        }

        const uint32_t scoreBlockCap =
            IS_BLOCK16 && tiling_->returnChoice != MSA_RETURN_SCORE && tiling_->keyLayout == MSA_KEY_LAYOUT_TND ?
                runtimeScoreStride_ :
                tiling_->maxBlocksPerBatch;
        scheduler_.Init(tiling_->batch, tiling_->numQHeads, scoreBlockCap, runtimeScoreStride_, tiling_->sparseMode,
                        tiling_->initBlocks, tiling_->localBlocks, IS_TND_TND ? MSA_KEY_LAYOUT_TND : tiling_->keyLayout,
                        tiling_->kvChunks, gActualSeqQlen_, gActualSeqKlen_, gStartLoc_, false,
                        IS_TND_TND ? 0U : tiling_->qMaxSeq, tiling_->kMaxSeq, tiling_->tableWidth, tiling_->chunkBudget,
                        tiling_->pageSize, tiling_->totalQ, IS_BLOCK16 ? tiling_->compressFactor : 1U,
                        IS_BLOCK16 ? tiling_->totalTasks : 0U);
    }

    __aicore__ inline void InitMx(GM_ADDR query, GM_ADDR key, GM_ADDR queryScale, GM_ADDR keyScale, GM_ADDR blockTable,
                                  GM_ADDR actualSeqQlen, GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score,
                                  GM_ADDR workspace, const MsaIndexScoreTilingData* __restrict tiling,
                                  GM_ADDR sparseIndices = nullptr, GM_ADDR sparseValues = nullptr)
    {
        if constexpr (IS_BLOCK16) {
            Init(query, key, blockTable, nullptr, actualSeqQlen, actualSeqKlen, startLoc, score, sparseIndices,
                 sparseValues, workspace, tiling);
            auto* scaleScratch = reinterpret_cast<__gm__ fp8_e8m0_t*>(workspace + tiling_->mxScaleScratchOffset);
            gKeyScaleScratchMx_.SetGlobalBuffer(scaleScratch);
            gQScaleScratchMx_.SetGlobalBuffer(scaleScratch + MSA_BLOCK16_MX_SCALE_SCRATCH_BYTES / 2U);
            gQueryScaleMx_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(queryScale));
            gKeyScaleMx_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(keyScale));
            return;
        }
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
            gTopkIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sparseIndices));
            if (tiling_->returnChoice == MSA_RETURN_INDICES_VALUES) {
                gTopkVal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sparseValues));
            }
        } else {
            gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score));
        }
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS*>(workspace));
        scheduler_.Init(tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch, tiling_->scoreBlockStride,
                        tiling_->sparseMode, tiling_->initBlocks, tiling_->localBlocks,
                        IS_TND_TND ? MSA_KEY_LAYOUT_TND : tiling_->keyLayout, tiling_->kvChunks, gActualSeqQlen_,
                        gActualSeqKlen_, gStartLoc_, false, IS_TND_TND ? 0U : tiling_->qMaxSeq, tiling_->kMaxSeq,
                        tiling_->tableWidth, tiling_->chunkBudget, tiling_->pageSize, tiling_->totalQ, 1U, 0U, 0U, 0U,
                        false, USE_C2UB && !MX_FUSED_STAGE && tiling_->headDim == 128U && tiling_->numKvHeads == 1U);
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
        if constexpr (ENABLE_TOPK) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
            Catlass::Arch::Resource<ArchTag> resource;
            auto ubTrunk = resource.ubBuf.template GetBufferByByte<float>(0);
            MsaIndexScoreTopk topk;
            topk.Run(tiling_, gScore_, gTopkIdx_, gTopkVal_, gActualSeqQlen_, gActualSeqKlen_,
                     tiling_->returnChoice == MSA_RETURN_INDICES_VALUES, ubTrunk);
#endif
        }
        if constexpr (IS_BLOCK16) {
            if (tiling_->returnChoice != MSA_RETURN_SCORE) {
                AscendC::SyncAll();
#if !defined(__DAV_C310_CUBE__) && !defined(__DAV_310R6_CUBE__)
                MsaBlock16Select::FusedSelect::VisibleMeta meta;
                meta.actualSeqQ = reinterpret_cast<GM_ADDR>(const_cast<__gm__ int32_t*>(gActualSeqQlen_.GetPhyAddr()));
                meta.actualSeqK = reinterpret_cast<GM_ADDR>(const_cast<__gm__ int32_t*>(gActualSeqKlen_.GetPhyAddr()));
                meta.batch = tiling_->batch;
                meta.sparseMode = tiling_->sparseMode;
                meta.initBlocks = tiling_->initBlocks;
                meta.localBlocks = tiling_->localBlocks;
                meta.compressFactor = tiling_->compressFactor;
                meta.keyLayout = tiling_->keyLayout;
                meta.qMaxSeq = tiling_->qMaxSeq;
                meta.kMaxSeq = tiling_->kMaxSeq;
                meta.maxBlocksPerBatch = tiling_->maxBlocksPerBatch;
                meta.tableWidth = tiling_->tableWidth;
                meta.pageSize = tiling_->pageSize;
                MsaBlock16Select::FusedSelect::Run(
                    reinterpret_cast<GM_ADDR>(const_cast<__gm__ float*>(gScore_.GetPhyAddr())),
                    reinterpret_cast<GM_ADDR>(const_cast<__gm__ int32_t*>(gTopkIdx_.GetPhyAddr())),
                    tiling_->returnChoice == MSA_RETURN_INDICES_VALUES ?
                        reinterpret_cast<GM_ADDR>(const_cast<__gm__ float*>(gTopkVal_.GetPhyAddr())) :
                        nullptr,
                    tiling_->totalQ, runtimeScoreStride_, AscendC::GetBlockNum(),
                    tiling_->returnChoice == MSA_RETURN_INDICES_VALUES, meta);
#endif
            }
        }
    }

private:
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

    // 等长满 M 的 decode 按连续 N256 tile 均分，每核只在请求边界重新初始化。
    // 同时支持连续和分页 K；变长、尾行沿用原调度，页地址仍由原窗口路径解析。
    __aicore__ inline uint32_t InitBalancedTasks(uint32_t coreNum, uint32_t totalTasks)
    {
        // MX 与 bf16/fp16 的块长 16 都按这段均分。N512 宽窗不走这里。
        if constexpr (IS_BLOCK16 && !MXFP4_N512) {
            const bool seqKey = MsaIsSeqKeyLayout(tiling_->keyLayout);
            if ((!seqKey && !MsaIsPaKeyLayout(tiling_->keyLayout)) || tiling_->numQHeads != 16U ||
                tiling_->headDim != 128U || tiling_->compressFactor != 4U || tiling_->batch == 0U || coreNum == 0U ||
                gActualSeqQlen_.GetValue(0) != 0 || (seqKey && gActualSeqKlen_.GetValue(0) != 0) ||
                (tiling_->qMaxSeq > 0U && tiling_->qMaxSeq < 8U)) {
                return totalTasks;
            }
            const int32_t kTokens = gActualSeqKlen_.GetValue(seqKey ? 1U : 0U);
            if (kTokens < 1024 || kTokens % 1024 != 0 ||
                (tiling_->kMaxSeq > 0U && static_cast<uint32_t>(kTokens / 4) > tiling_->kMaxSeq) ||
                static_cast<uint32_t>(kTokens / 16) > runtimeScoreStride_ ||
                static_cast<uint32_t>(kTokens / 16) > tiling_->maxBlocksPerBatch) {
                return totalTasks;
            }
            for (uint32_t b = 0U; b < tiling_->batch; ++b) {
                if (gActualSeqQlen_.GetValue(b + 1U) != static_cast<int32_t>((b + 1U) * 8U) ||
                    static_cast<int64_t>(gActualSeqKlen_.GetValue(seqKey ? b + 1U : b)) !=
                        (seqKey ? static_cast<int64_t>(b + 1U) * kTokens : kTokens)) {
                    return totalTasks;
                }
            }
            if (totalTasks % tiling_->batch != 0U) {
                return totalTasks;
            }
            balancedDirectFp4_ = IS_MXFP4 && (seqKey || tiling_->pageSize == 128U);
            balancedTilesPerRequest_ = static_cast<uint32_t>(kTokens) / 1024U;
            balancedTilesPerCore_ = MsaCeilDiv(tiling_->batch * balancedTilesPerRequest_, coreNum);
            balancedOldChunks_ = totalTasks / tiling_->batch;
            return coreNum * (MsaCeilDiv(balancedTilesPerCore_, balancedTilesPerRequest_) + 1U);
        }
        return totalTasks;
    }

    __aicore__ inline bool DecodeBalancedTask(uint32_t taskIdx, uint32_t coreNum, MsaTask& task)
    {
        if (balancedTilesPerRequest_ == 0U) {
            scheduler_.Decode(taskIdx, task);
            return true;
        }
        const uint32_t core = taskIdx % coreNum;
        const uint32_t part = taskIdx / coreNum;
        const uint32_t coreBegin = core * balancedTilesPerCore_;
        const uint32_t coreEnd =
            MsaMinU32(coreBegin + balancedTilesPerCore_, tiling_->batch * balancedTilesPerRequest_);
        const uint32_t batch = coreBegin / balancedTilesPerRequest_ + part;
        const uint32_t requestBegin = batch * balancedTilesPerRequest_;
        const uint32_t requestEnd = requestBegin + balancedTilesPerRequest_;
        const uint32_t begin = part == 0U ? coreBegin : requestBegin;
        const uint32_t end = MsaMinU32(coreEnd, requestEnd);
        if (begin >= end || batch >= tiling_->batch) {
            return false;
        }
        scheduler_.Decode(batch * balancedOldChunks_, task);
        task.sStileBegin = (begin - requestBegin) * 2U;
        task.sStileEnd = (end - requestBegin) * 2U;
        task.sBlkBegin = task.sStileBegin * 32U;
        task.sBlkEnd = task.sStileEnd * 32U;
        const uint32_t firstCore = requestBegin / balancedTilesPerCore_;
        task.sChunkIdx = core - firstCore;
        task.sChunks = (requestEnd - 1U) / balancedTilesPerCore_ - firstCore + 1U;
        task.writeTailFill = end == requestEnd;
        return true;
    }

    // Match MsaTaskScheduler::LoadBatchMeta/LoadCompressedBatch, BlocksOfRows and BlockCap.
    // Host allocation remains conservative. Only private TopK row addressing is compacted.
    __aicore__ inline uint32_t ResolveRuntimeScoreStride() const
    {
        const uint32_t cf = tiling_->compressFactor < 1U ? 1U : tiling_->compressFactor;
        uint32_t rowsPerBlock = MSA_BLOCK16_SIZE / cf;
        if (rowsPerBlock == 0U) {
            rowsPerBlock = 1U;
        }
        uint64_t blockCap = tiling_->maxBlocksPerBatch;
        if (MsaIsPaKeyLayout(tiling_->keyLayout) && tiling_->tableWidth > 0U) {
            const uint64_t tableBlocks =
                static_cast<uint64_t>(tiling_->tableWidth) * (tiling_->pageSize / rowsPerBlock);
            if (tableBlocks < blockCap) {
                blockCap = tableBlocks;
            }
        }
        uint64_t maxBlocks = 0U;
        for (uint32_t b = 0U; b < tiling_->batch; ++b) {
            int32_t tokens;
            if (MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                tokens = static_cast<int32_t>(gActualSeqKlen_.GetValue(b + 1U)) -
                         static_cast<int32_t>(gActualSeqKlen_.GetValue(b));
            } else {
                tokens = gActualSeqKlen_.GetValue(b);
            }
            // Metadata describes original token lengths; key slots describe compressed rows.
            uint64_t rows = tokens > 0 ? (static_cast<uint64_t>(tokens) + cf - 1U) / cf : 0U;
            if (tiling_->kMaxSeq > 0U && rows > tiling_->kMaxSeq) {
                rows = tiling_->kMaxSeq;
            }
            const uint32_t blockRows = cf == 1U ? MSA_BLOCK16_SIZE : rowsPerBlock;
            uint64_t blocks = (rows + blockRows - 1U) / blockRows;
            if (blocks > blockCap) {
                blocks = blockCap;
            }
            if (blocks > maxBlocks) {
                maxBlocks = blocks;
            }
        }
        const uint64_t aligned = maxBlocks == 0U ? 16U : (maxBlocks + 15U) / 16U * 16U;
        return aligned < tiling_->scoreBlockStride ? static_cast<uint32_t>(aligned) : tiling_->scoreBlockStride;
    }
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    __aicore__ inline void ProcessCube()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        blockMmad.SetBnsdQuery(FastBnsdQuery() ? tiling_->qMaxSeq : 0U);
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
            if constexpr (IS_MXFP4) {
                blockMmad.SetBnsdQueryGather(DirectBnsdQuery() ? tiling_->numQHeads : 0U, tiling_->qMaxSeq,
                                             task.mStart);
            } else if constexpr (FLOAT_DIRECT) {
                const bool gather =
                    DirectBnsdQuery() && !FastBnsdQuery() && tiling_->numQHeads > 1U && tiling_->qMaxSeq > 1U;
                blockMmad.SetBnsdQueryGather(gather ? tiling_->numQHeads : 0U, tiling_->qMaxSeq, task.mStart);
            }
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
        if constexpr (IS_BLOCK16) {
            totalTasks = InitBalancedTasks(coreNum, totalTasks);
        }
        MsaTask task;
        bool balanceWaves = BalanceCausalTaskWaves(coreNum, totalTasks);
        uint32_t wave = 0U;
        for (uint32_t taskSlot = coreIdx; taskSlot < totalTasks; taskSlot += coreNum, ++wave) {
            const uint32_t taskIdx = TaskOfWave(taskSlot, wave, coreIdx, coreNum, balanceWaves);
            if constexpr (IS_BLOCK16) {
                if (!DecodeBalancedTask(taskIdx, coreNum, task)) {
                    continue;
                }
            } else if (!scheduler_.DecodeScoreSlot(taskIdx, task)) {
                continue;
            }
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
            if constexpr (IS_MXFP4) {
                blockMmad.SetBnsdQueryGather(DirectBnsdQuery() ? tiling_->numQHeads : 0U, tiling_->qMaxSeq,
                                             task.mStart);
            } else if constexpr (FLOAT_DIRECT) {
                const bool gather =
                    DirectBnsdQuery() && !FastBnsdQuery() && tiling_->numQHeads > 1U && tiling_->qMaxSeq > 1U;
                blockMmad.SetBnsdQueryGather(gather ? tiling_->numQHeads : 0U, tiling_->qMaxSeq, task.mStart);
            }
            if constexpr (!IS_BLOCK16) {
                PreloadBlockTable(task);
            }
            bool needLoadQ = true;
            uint32_t pageSeq = 0;
            if constexpr (IS_BLOCK16) {
                // 连续 direct-GM 窗并成一次宽 N。earlyCv 在 Fixpipe 里放行 AIV，这里不再 Set CV。
                bool bReady = false;
                uint32_t readyNWin = 1U;
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd;) {
                    const uint32_t nWin = Block16WinGroup(task, win);
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    const bool direct = WindowDirectGm(task, win);
                    if (bReady && (readyNWin != nWin || !direct)) {
                        bReady = false;
                    }
                    const bool needK = (IS_MXFP4 || FLOAT_DIRECT) ? ((pageSeq == 0U && NeedQueryPrepare()) || !direct) :
                                                                    ((pageSeq == 0U) || !direct);
                    if (needK) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                    // 整段 FP32 放不进一个 ping 时，先做完 N=512 的 MMAD，再按 ping 切片 Fixpipe。
                    // 这样下一次 MMAD 只等最后一片交还 L0C，能叠在 vector 归约上。
                    bool sliced = false;
                    if constexpr (IS_MXFP4 && BLOCK16_FP32_MAX) {
                        if (direct && nWin > Block16Fp32WinsPerPing(task)) {
                            pageSeq += ComputeWindowFp32Slices(blockMmad, resource, task, win, needLoadQ, nWin, pageSeq,
                                                               bReady);
                            sliced = true;
                        }
                    }
                    if (!sliced) {
#if defined(ORIG_DTYPE_QUERY) && (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1)
                        // MMAD only needs an idle L0C; wait for the AIV UB immediately before Fixpipe.
                        const bool deferVc = IS_MXFP4 && !MXFP4_N512 && direct && balancedDirectFp4_;
                        if (!deferVc) {
                            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                                    MSA_A5_AIV_FLAG_OFFSET);
                        }
                        auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * Epilogue::S_PING_BYTES);
                        if (direct) {
                            ComputeWindowDirectGm(blockMmad, task, win, ubS, needLoadQ, nWin, ping, bReady, deferVc);
                        } else {
                            ComputeWindowToUb(blockMmad, task, ubS, coreKScratch, needLoadQ, false, 0U, ping);
                            bReady = false;
                        }
#else
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping +
                                                                                MSA_A5_AIV_FLAG_OFFSET);
                        auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * Epilogue::S_PING_BYTES);
                        if (direct) {
                            ComputeWindowDirectGm(blockMmad, task, win, ubS, needLoadQ, nWin, ping, bReady);
                        } else {
                            ComputeWindowToUb(blockMmad, task, ubS, coreKScratch, needLoadQ, false, 0U, ping);
                            bReady = false;
                        }
#endif
                        ++pageSeq;
                    }
                    needLoadQ = false;
                    bReady = false;
                    const uint32_t nextWin = win + nWin;
                    if (nextWin < task.sStileEnd && WindowDirectGm(task, nextWin)) {
                        readyNWin = Block16WinGroup(task, nextWin);
                        LoadWindowToL1(blockMmad, task, nextWin, readyNWin);
                        bReady = true;
                    }
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
            } else if constexpr (FP8_SCORE_PIPE) {
                // 两页合成一次 MMAD：标量 / Fixpipe / 跨核握手按组付一次。页在 GM 上不连续，
                // 仍按页搬进同一块 L1 的相邻 N。顺序布局尾块继续逐页（可能要 K scratch）。
                if (!MsaIsSeqKeyLayout(tiling_->keyLayout) && task.sBlkBegin < task.sBlkEnd) {
                    for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd;) {
                        const uint32_t nPages = Fp8PageGroup(task, blk);
                        const uint32_t headDim = tiling_->headDim;
                        const LayoutK layoutPage(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
                        for (uint32_t i = 0; i < nPages; ++i) {
                            const uint64_t kOffset = KeyBlockGmOffset(task, blk + i);
                            blockMmad.AppendBPage(gKeyFp_[kOffset], layoutPage, i * MSA_BLOCK_SIZE, i == 0U,
                                                  i + 1U == nPages);
                        }
                        const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                        MmadGroupedToL0C(blockMmad, task, needLoadQ, nPages);
                        needLoadQ = false;
                        auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                        blockMmad.FlushStoreToUb(ubS, ping, false, static_cast<int32_t>(ping));
                        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping +
                                                                               MSA_A5_AIV_FLAG_OFFSET);
                        ++pageSeq;
                        blk += nPages;
                    }
                } else {
                    for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd; ++blk) {
                        const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                        if (KeyBlockNeedsScratch(task, blk)) {
                            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K +
                                                                                    MSA_A5_AIV_FLAG_OFFSET);
                            AscendC::PipeBarrier<PIPE_ALL>();
                        }
                        auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                        ComputePageToUb(blockMmad, task, blk, ubS, coreKScratch, needLoadQ, static_cast<int32_t>(ping));
                        needLoadQ = false;
                        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping +
                                                                               MSA_A5_AIV_FLAG_OFFSET);
                        ++pageSeq;
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

    // AIC 与 AIV 必须采用相同的页组和 ping 顺序。
    __aicore__ inline uint32_t Fp8PageGroup(const MsaTask& task, uint32_t blk) const
    {
        if constexpr (!FP8_SCORE_PIPE) {
            return 1U;
        }
        if (MsaIsSeqKeyLayout(tiling_->keyLayout) || blk >= task.sBlkEnd) {
            return 1U;
        }
        const uint32_t room = MSA_BLOCKS_PER_STILE - (blk % MSA_BLOCKS_PER_STILE);
        const uint32_t remain = task.sBlkEnd - blk;
        if (room < 2U || remain < 2U) {
            return 1U;
        }
        uint32_t m0 = 0U;
        uint32_t m1 = 0U;
        MsaC2UbGroupSplit(task.mActual, 1U, m0, m1);
        const uint32_t rows = m0 > m1 ? m0 : m1;
        const uint64_t bytes = static_cast<uint64_t>(rows) * 2U * MSA_BLOCK_SIZE * sizeof(half);
        return bytes <= MSA_A5_S_PING_BYTES ? 2U : 1U;
    }

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

    /// 块长 16：一个打分块占多少 key 行。key 未压缩时就是 MSA_BLOCK16_SIZE。
    __aicore__ inline uint32_t RowsPerBlock16() const
    {
        return tiling_->rowsPerBlock > 0U ? tiling_->rowsPerBlock : MSA_BLOCK16_SIZE;
    }

    /// 一窗固定 MSA_BLOCK_SIZE 个 key 行（Cube 的 N），压缩后装得下更多打分块。
    __aicore__ inline uint32_t BlocksPerWin16() const
    {
        return MSA_BLOCK_SIZE / RowsPerBlock16();
    }

    /// A128-row window must be inside one page or contiguous compressed pages.
    /// Fragmented pages and incomplete windows use the gather fallback.
    __aicore__ inline bool PaWindowInPage(const MsaTask& task, uint32_t win) const
    {
        const uint32_t pSize = tiling_->pageSize;
        if constexpr (COMPRESSED_PAGES && IS_BLOCK16 && !IS_QUANT && !IS_MXFP4) {
            // A128-row window can span several compressed physical pages.
            // Direct GM is safe only when every page is present, physically
            // adjacent and has no per-page/head padding. Fragmented pages and
            // incomplete tail windows retain the existing gather path.
            if (pSize < MSA_BLOCK_SIZE && pSize > 0U && tiling_->compressFactor > 1U &&
                pSize == MSA_BLOCK_SIZE / tiling_->compressFactor) {
                const uint32_t row0 = win * MSA_BLOCK_SIZE;
                if (tiling_->strideKvToken != tiling_->headDim || task.kvLen <= 0 ||
                    row0 + MSA_BLOCK_SIZE > static_cast<uint32_t>(task.kvLen) ||
                    tiling_->strideKvBlock != static_cast<uint64_t>(pSize) * tiling_->strideKvToken) {
                    return false;
                }
                const uint32_t page0 = row0 / pSize;
                const uint32_t count = MSA_BLOCK_SIZE / pSize;
                if (page0 + count > tiling_->tableWidth) {
                    return false;
                }
                const uint32_t table0 = task.batchIdx * tiling_->tableWidth + page0;
                const int32_t first = gBlockTable_.GetValue(table0);
                if (first < 0 || static_cast<uint32_t>(first) >= tiling_->numPages) {
                    return false;
                }
                for (uint32_t i = 1U; i < count; ++i) {
                    const int32_t page = gBlockTable_.GetValue(table0 + i);
                    if (page != first + static_cast<int32_t>(i) || static_cast<uint32_t>(page) >= tiling_->numPages) {
                        return false;
                    }
                }
                return true;
            }
        }
        if (pSize < MSA_BLOCK_SIZE || tiling_->strideKvToken != tiling_->headDim || task.kvLen <= 0) {
            return false;
        }
        const uint32_t row0 = win * MSA_BLOCK_SIZE;
        if (static_cast<uint32_t>(task.kvLen) < row0 + MSA_BLOCK_SIZE) {
            return false;
        }
        const uint32_t pageIdx = pSize == MSA_BLOCK_SIZE ? win : row0 / pSize;
        const uint32_t inPage = pSize == MSA_BLOCK_SIZE ? 0U : row0 - pageIdx * pSize;
        if (inPage + MSA_BLOCK_SIZE > pSize) {
            return false;
        }
        const uint32_t tWidth = tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
        if (pageIdx >= tWidth) {
            return false;
        }
        const int32_t page = gBlockTable_.GetValue(task.batchIdx * tWidth + pageIdx);
        return page >= 0;
    }

    /// 从 win 起的 nWin 个窗在 GM 上首尾相接。同一页内相邻，或物理页号连续，都满足。
    __aicore__ inline bool PaWindowsContiguous(const MsaTask& task, uint32_t win, uint32_t nWin) const
    {
        if (nWin <= 1U) {
            return true;
        }
        const uint64_t base = PaWindowOffset(task, win);
        const uint64_t step = static_cast<uint64_t>(MSA_BLOCK_SIZE) * tiling_->strideKvToken;
        for (uint32_t i = 1U; i < nWin; ++i) {
            if (!PaWindowInPage(task, win + i) ||
                PaWindowOffset(task, win + i) != base + static_cast<uint64_t>(i) * step) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t PaWindowOffset(const MsaTask& task, uint32_t win) const
    {
        const uint32_t pSize = tiling_->pageSize;
        const uint32_t row0 = win * MSA_BLOCK_SIZE;
        const uint32_t pageIdx = pSize == MSA_BLOCK_SIZE ? win : (pSize == 0U ? 0U : row0 / pSize);
        const uint32_t inPage = pSize == MSA_BLOCK_SIZE ? 0U : (pSize == 0U ? row0 : row0 - pageIdx * pSize);
        const uint32_t tWidth = tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
        const int32_t page = gBlockTable_.GetValue(task.batchIdx * tWidth + pageIdx);
        return static_cast<uint64_t>(page) * tiling_->strideKvBlock +
               static_cast<uint64_t>(inPage) * tiling_->strideKvToken;
    }

    /// 一个 128 行的窗整段落在本请求的 KV 内、且 token 步长等于 headDim 时，
    /// Cube 可以直接从 GM 读 K：省掉「Vector 读 → 写 workspace → Cube 再读」的两趟。
    /// 顺序布局看打包后的行是否连续。分页布局只在整窗落在同一存储页内时直读。
    /// 尾窗仍走 scratch，避免把下一个请求或张量尾部读进来。
    __aicore__ inline bool WindowDirectGm(const MsaTask& task, uint32_t win) const
    {
        if constexpr (FLOAT_DIRECT) {
            // Head dimensions beyond one K tile keep the existing fallback.
            if (tiling_->headDim <= MSA_K_TILE && tiling_->strideKvToken >= tiling_->headDim && task.kvLen > 0) {
                const uint32_t row0 = win * MSA_BLOCK_SIZE;
                if (MsaIsSeqKeyLayout(tiling_->keyLayout) ||
                    (MsaIsPaKeyLayout(tiling_->keyLayout) && tiling_->pageSize > 0U &&
                     (MSA_BLOCK_SIZE % tiling_->pageSize == 0U))) {
                    return row0 < static_cast<uint32_t>(task.kvLen);
                }
            }
        }
        if constexpr (IS_BLOCK16 && IS_MXFP4) {
            // DN2NZ scale segments require 16-row boundaries. Compressed 8-row
            // pages retain the packed scratch gather rather than misplacing scales.
            return !MsaIsPaKeyLayout(tiling_->keyLayout) || tiling_->pageSize >= 16U;
        }
        if constexpr (IS_BLOCK16 && IS_MXFP4 && !MXFP4_N512) {
            // Balanced tasks already validated q=8, H=16, D=128, cf=4 and full 256-row tiles.
            // Every 128-row window is direct for sequence keys and physical page=128.
            if (balancedDirectFp4_) {
                return true;
            }
        }

        if constexpr (!IS_BLOCK16 || IS_QUANT_V) {
            return false;
        }
        if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
            return PaWindowInPage(task, win);
        }
        if (!MsaIsSeqKeyLayout(tiling_->keyLayout) || tiling_->strideKvToken != tiling_->headDim || task.kvLen <= 0) {
            return false;
        }
        const uint32_t row0 = win * MSA_BLOCK_SIZE;
        return (row0 + MSA_BLOCK_SIZE) <= static_cast<uint32_t>(task.kvLen);
    }

    /// 满窗且 Q/K 都在 GM 上时，Vector 没有 scratch 要交给 Cube。
    /// 这时再等 K 旗并排空全部流水，会把下一窗的 MTE2 挡在上一窗 FIXP 之后。
    __aicore__ inline bool WindowNeedsKHandoff(const MsaTask& task, uint32_t win) const
    {
        return PackQuery() || !WindowDirectGm(task, win);
    }

    __aicore__ inline uint64_t WindowDirectGmOffset(const MsaTask& task, uint32_t win) const
    {
        if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
            return PaWindowOffset(task, win);
        }
        return MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, win * MSA_BLOCK_SIZE,
                                  tiling_->numKvHeads, tiling_->headDim, tiling_->kMaxSeq);
    }

    /// 一组窗的 fp16 S 要放进一个加宽 ping。行数按两个 AIV 中较多的那半算。
    __aicore__ inline bool Block16GroupFits(const MsaTask& task, uint32_t nWin) const
    {
        uint32_t m0 = 0U;
        uint32_t m1 = 0U;
        MsaC2UbHalfSplit(task.mActual, m0, m1);
        const uint32_t rows = m0 > m1 ? m0 : m1;
        // MXFP4 宽窗按 128 列把 FP32 点积切片写入 ping，不按整段 N=512 占 UB。
        if constexpr (IS_MXFP4 && BLOCK16_FP32_MAX) {
            const uint64_t slice = static_cast<uint64_t>(rows) * MSA_BLOCK_SIZE * sizeof(float);
            return slice <= MSA_A5_S_PING_BYTES_B16;
        }
        const uint64_t bytes = static_cast<uint64_t>(rows) * nWin * MSA_BLOCK_SIZE * sizeof(ElementS);
        return bytes <= MSA_A5_S_PING_BYTES_B16;
    }

    /// 一个 S ping 放得下几个 128 列 FP32 窗。热路径 m=128、N1=16 时每 AIV 64 行，两窗正好 64KB。
    __aicore__ inline uint32_t Block16Fp32WinsPerPing(const MsaTask& task) const
    {
        uint32_t m0 = 0U;
        uint32_t m1 = 0U;
        MsaC2UbGroupSplit(task.mActual, tiling_->numQHeads, m0, m1);
        const uint32_t rows = m0 > m1 ? m0 : m1;
        if (rows == 0U) {
            return MSA_BLOCK16_MAX_WIN_GROUP;
        }
        const uint64_t one = static_cast<uint64_t>(rows) * MSA_BLOCK_SIZE * sizeof(float);
        if (one > MSA_A5_S_PING_BYTES_B16) {
            return 1U;
        }
        uint32_t fit = static_cast<uint32_t>(MSA_A5_S_PING_BYTES_B16 / one);
        if (fit > MSA_BLOCK16_MAX_WIN_GROUP) {
            fit = MSA_BLOCK16_MAX_WIN_GROUP;
        }
        return fit == 0U ? 1U : fit;
    }

    /// 从 win 起能并成一次 MMAD 的窗数。压缩时一个逻辑块不是 16 行，epilogue 按 rowsPerBlock 归约。
    /// 只合并连续的 direct-GM 窗；走 scratch 的尾窗逐窗处理。
    __aicore__ inline uint32_t Block16WinGroup(const MsaTask& task, uint32_t win) const
    {
        if constexpr (IS_BLOCK16 && IS_MXFP4 && !MXFP4_N512) {
            // Balanced tasks already validated q=8, H=16, D=128, cf=4 and full 256-row tiles.
            // Every 128-row window is direct for sequence keys and physical page=128.
            if (balancedDirectFp4_) {
                return 2U;
            }
        }

        if constexpr (!IS_BLOCK16) {
            return 1U;
        }
        const uint32_t rowsPerBlk = RowsPerBlock16();
        if (rowsPerBlk == 0U || (MSA_BLOCK_SIZE % rowsPerBlk) != 0U) {
            return 1U;
        }
        const uint32_t blocksPerWin = MSA_BLOCK_SIZE / rowsPerBlk;
        if ((blocksPerWin % MSA_BLOCKS_PER_STILE) != 0U) {
            return 1U;
        }
        // N512 单槽按 4 窗对齐。块长 16 的窄 MXFP4 用 N=256、双 L0C，按 2 窗对齐，
        // 下一半 MMAD 才能叠在上一半 Fixpipe 上。bf16 的槽仍是 N=128，不并窗。
        constexpr uint32_t kGroupAlign =
            (IS_MXFP4 && MXFP4_N512) ? MSA_BLOCK16_MAX_WIN_GROUP : ((IS_BLOCK16 && IS_MXFP4) ? 2U : 1U);
        if ((kGroupAlign > 1U && (win % kGroupAlign) != 0U) || !WindowDirectGm(task, win)) {
            return 1U;
        }
        constexpr uint32_t kMaxCols = (IS_MXFP4 && MXFP4_N512) ?
                                          (MSA_BLOCK_SIZE * MSA_BLOCK16_MAX_WIN_GROUP) :
                                          ((IS_BLOCK16 && IS_MXFP4) ? (MSA_BLOCK_SIZE * 2U) : MSA_BLOCK_SIZE);
        uint32_t n = 1U;
        while (n < MSA_BLOCK16_MAX_WIN_GROUP && (n + 1U) * MSA_BLOCK_SIZE <= kMaxCols && (win + n) < task.sStileEnd &&
               WindowDirectGm(task, win + n) && Block16GroupFits(task, n + 1U) &&
               (!MsaIsPaKeyLayout(tiling_->keyLayout) || (IS_MXFP4 && !MXFP4_N512) ||
                (IS_MXFP4 || PaWindowsContiguous(task, win, n + 1U)))) {
            ++n;
        }
        return n;
    }

    __aicore__ inline bool StileNeedsKScratch(const MsaTask& task, uint32_t blkBase) const
    {
        const uint32_t nBlk = IS_BLOCK16 ? BlocksPerWin16() : MSA_BLOCKS_PER_STILE;
        for (uint32_t j = 0; j < nBlk; ++j) {
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

    __aicore__ inline uint32_t BlockTableRowStride() const
    {
        return tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
    }

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

    /// MX scale 一块经常只有 16B，源和目的都不是 32B 对齐。非原子 MTE 要求地址 32B 对齐，
    /// 所以先搬到对齐的临时槽，再用标量把有效字节放到块槽里。返回 true 表示走过标量。
    __aicore__ inline bool CopyScaleBytes(const AscendC::LocalTensor<uint8_t>& ubSc, uint32_t dstElem, uint64_t srcElem,
                                          uint32_t nBytes, uint32_t stageElem)
    {
        if (nBytes == 0U) {
            return false;
        }
        const uintptr_t ubBase = reinterpret_cast<uintptr_t>(ubSc.GetPhyAddr());
        const uintptr_t gmBase = reinterpret_cast<uintptr_t>(gKeyScaleMx_.GetPhyAddr());
        const uint32_t dstMis = static_cast<uint32_t>((ubBase + dstElem) & 31U);
        const uint32_t srcMis = static_cast<uint32_t>((gmBase + srcElem) & 31U);
        if (dstMis == 0U && srcMis == 0U) {
            CopyGmToUbPartial(ubSc[dstElem], gKeyScaleMx_[srcElem].template ReinterpretCast<uint8_t>(), nBytes);
            return false;
        }
        const uint32_t back = srcMis;
        CopyGmToUbPartial(ubSc[stageElem], gKeyScaleMx_[srcElem - back].template ReinterpretCast<uint8_t>(),
                          back + nBytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        for (uint32_t i = 0U; i < nBytes; ++i) {
            ubSc.SetValue(dstElem + i, ubSc.GetValue(stageElem + back + i));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
        return true;
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
            if constexpr (IS_MXFP4) {
                const uint32_t groups = headDim / MSA_MX_GROUP;
                const uint32_t nScaleElem = MSA_BLOCK_SIZE * groups;
                const uint32_t nValidScale = nValidTok * groups;
                const uint64_t scaleOff = IS_MXFP4 ? MxKeyScaleOffset(kOffset) : kOffset / MSA_MX_GROUP;
                const uint64_t scaleScratchOff = (scratchOff / MSA_K_TILE) * groups;
                AscendC::LocalTensor<uint8_t> ubSc = resource.ubBuf.template GetBufferByByte<uint8_t>(UB_OFF_CAST_I8);
                if (nValidScale < nScaleElem) {
                    AscendC::Duplicate(ubSc, static_cast<uint8_t>(0), nScaleElem);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                if (nValidScale > 0) {
                    CopyGmToUbPartial(ubSc, gKeyScaleMx_[scaleOff].template ReinterpretCast<uint8_t>(), nValidScale);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopy(gKeyScaleScratchMx_[scaleScratchOff].template ReinterpretCast<uint8_t>(), ubSc,
                                  nScaleElem);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// 块长 16 的一个逻辑块里，落在 kv_len 内的 key 行数。kv_len 与行号都按压缩行计。
    __aicore__ inline uint32_t KeyPage16ValidTokens(const MsaTask& task, uint32_t blk) const
    {
        if (task.kvLen <= 0) {
            return 0;
        }
        const uint32_t rows = RowsPerBlock16();
        const uint32_t tok = blk * rows;
        if (tok >= static_cast<uint32_t>(task.kvLen)) {
            return 0;
        }
        const uint32_t remain = static_cast<uint32_t>(task.kvLen) - tok;
        return remain < rows ? remain : rows;
    }

    /// AIV（块长 16）：把一窗 8 个逻辑块收成 128 个 token。
    /// BBND/BNBD 经 block_table 寻址；BSND/BNSD 按连续 token 寻址。越界补零。
    __aicore__ inline void GatherKeyWindowToScratch(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                                    uint32_t win, uint64_t scratchOff)
    {
        const uint32_t headDim = tiling_->headDim;
        const uint32_t tokenBytes = IS_MXFP4 ? (headDim >> 1U) : headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * tokenBytes;
        const uint32_t gatherOff = IS_BLOCK16 ? UB_OFF_B16_GATHER : UB_OFF_CAST_FP;
        AscendC::LocalTensor<uint8_t> ubBytes = resource.ubBuf.template GetBufferByByte<uint8_t>(gatherOff);
        AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(gatherOff);
        if constexpr (IS_MXFP4 || sizeof(ElementQ) == 1U) {
            AscendC::Duplicate(ubBytes, static_cast<uint8_t>(0), nElem);
        } else {
            AscendC::Duplicate(ubK, static_cast<ElementQ>(0), MSA_BLOCK_SIZE * headDim);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        const uint32_t rowsPerBlk = RowsPerBlock16();
        const uint32_t blocksPerWin = BlocksPerWin16();
        const uint32_t blkBase = win * blocksPerWin;
        const uint32_t pSize = tiling_->pageSize > 0U ? tiling_->pageSize : rowsPerBlk;
        const uint32_t tWidth = tiling_->tableWidth > 0U ? tiling_->tableWidth : tiling_->maxBlocksPerBatch;
        const bool selection = IS_BLOCK16 && tiling_->returnChoice != MSA_RETURN_SCORE;
        uint32_t blkLimit =
            selection && tiling_->keyLayout == MSA_KEY_LAYOUT_TND ? runtimeScoreStride_ : tiling_->maxBlocksPerBatch;
        if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
            const uint32_t paBlocks = tWidth * (pSize / rowsPerBlk);
            if (paBlocks > 0U && paBlocks < blkLimit) {
                blkLimit = paBlocks;
            }
        }
        for (uint32_t j = 0; j < blocksPerWin; ++j) {
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
                kOffset = MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, blk * rowsPerBlk,
                                             tiling_->numKvHeads, tiling_->headDim, tiling_->kMaxSeq);
            } else {
                const uint32_t tok = blk * rowsPerBlk;
                const int32_t page = gBlockTable_.GetValue(task.batchIdx * tWidth + tok / pSize);
                kOffset = static_cast<uint64_t>(page) * tiling_->strideKvBlock +
                          static_cast<uint64_t>(tok % pSize) * tiling_->strideKvToken;
            }
            if constexpr (IS_MXFP4) {
                const uint64_t packed = kOffset >> 1U;
                auto dst = ubBytes[j * rowsPerBlk * tokenBytes];
                CopyGmToUbPartial(dst, gKeyFp_[packed].template ReinterpretCast<uint8_t>(), nTok * tokenBytes);
            } else if (tiling_->strideKvToken == headDim) {
                CopyGmToUbPartial(ubK[j * rowsPerBlk * headDim], gKeyFp_[kOffset], nTok * headDim);
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
                AscendC::DataCopyPad(ubK[j * rowsPerBlk * headDim], gKeyFp_[kOffset], params, pad);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
        if constexpr (IS_MXFP4) {
            AscendC::DataCopy(gKeyScratch_[scratchOff >> 1U].template ReinterpretCast<uint8_t>(), ubBytes, nElem);
        } else {
            AscendC::DataCopy(gKeyScratch_[scratchOff], ubK, MSA_BLOCK_SIZE * headDim);
        }
        if constexpr (IS_MXFP4) {
            const uint32_t groups = headDim / MSA_MX_GROUP;
            const uint32_t nScaleElem = MSA_BLOCK_SIZE * groups;
            const uint64_t scaleScratchOff = (scratchOff / MSA_K_TILE) * groups;
            const uint32_t scaleUbOff = IS_BLOCK16 ? UB_OFF_B16_SCALE : UB_OFF_CAST_I8;
            AscendC::LocalTensor<uint8_t> ubSc = resource.ubBuf.template GetBufferByByte<uint8_t>(scaleUbOff);
            AscendC::Duplicate(ubSc, static_cast<uint8_t>(0), nScaleElem);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            const uint32_t scaleStage = (nScaleElem + 31U) & ~31U;
            bool scalarScale = false;
            for (uint32_t j = 0; j < blocksPerWin; ++j) {
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
                    kOffset = MsaSeqKeyBlockElem(tiling_->keyLayout, task.batchIdx, task.cuKStart, blk * rowsPerBlk,
                                                 tiling_->numKvHeads, tiling_->headDim, tiling_->kMaxSeq);
                } else {
                    const uint32_t tok = blk * rowsPerBlk;
                    const int32_t page = gBlockTable_.GetValue(task.batchIdx * tWidth + tok / pSize);
                    kOffset = static_cast<uint64_t>(page) * tiling_->strideKvBlock +
                              static_cast<uint64_t>(tok % pSize) * tiling_->strideKvToken;
                }
                const uint64_t scaleOff = IS_MXFP4 ? MxKeyScaleOffset(kOffset) : kOffset / MSA_MX_GROUP;
                const uint32_t dstElem = j * rowsPerBlk * groups;
                if (CopyScaleBytes(ubSc, dstElem, scaleOff, nTok * groups, scaleStage)) {
                    scalarScale = true;
                }
            }
            if (scalarScale) {
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
            }
            AscendC::DataCopy(gKeyScaleScratchMx_[scaleScratchOff].template ReinterpretCast<uint8_t>(), ubSc,
                              nScaleElem);
        }
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

    /// TND 的 M-tile 在 GM 上已是连续 (token, head)。BSND/BNSD 先收进 per-core scratch。
    __aicore__ inline bool FastBnsdQuery() const
    {
        if constexpr (IS_BLOCK16 && !IS_QUANT_V) {
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD && tiling_->numQHeads == 16U &&
                   tiling_->headDim == 128U && tiling_->qMaxSeq == 8U;
        }
        return false;
    }

    __aicore__ inline bool DirectBnsdQuery() const
    {
        if constexpr (FLOAT_DIRECT) {
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD;
        }
        if constexpr (IS_MXFP4) {
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD && tiling_->qMaxSeq > 1U && tiling_->numQHeads > 1U;
        }
        if constexpr (IS_BLOCK16 && !IS_QUANT_V) {
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD && tiling_->numQHeads == 16U &&
                   tiling_->headDim == 128U && tiling_->qMaxSeq == 8U;
        }
        return false;
    }

    __aicore__ inline bool NeedMxScaleL1() const
    {
        return IS_MXFP4 && DirectBnsdQuery() && (tiling_->numQHeads % 16U != 0U);
    }

    __aicore__ inline bool NeedQueryPrepare() const
    {
        return PackQuery() || NeedMxScaleL1();
    }

    __aicore__ inline void PrepareQuery(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                        uint64_t scratch)
    {
        if constexpr (IS_MXFP4) {
            if (!NeedMxScaleL1()) {
                return;
            }
            const uint32_t groups = tiling_->headDim / MSA_MX_GROUP;
            const uint32_t rows = (task.mActual + 15U) & ~15U;
            auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(IS_BLOCK16 ? UB_OFF_B16_GATHER : UB_OFF_CAST_I8);
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

    __aicore__ inline bool PackQuery() const
    {
        if constexpr (IS_TND_TND) {
            return false;
        }
        if constexpr (IS_MXFP4) {
            return false;
        }
        if (DirectBnsdQuery()) {
            return false;
        }
        if constexpr (IS_BLOCK16 && !IS_QUANT_V) {
            // BSND 的 (token, head) 行与 Cube 的 M 顺序一致，可直接读取。
            return tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD;
        }
        return tiling_->queryLayout == MSA_QUERY_LAYOUT_BSND || tiling_->queryLayout == MSA_QUERY_LAYOUT_BNSD;
    }

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

    __aicore__ inline uint64_t QueryElemOffset(const MsaTask& task) const
    {
        if constexpr (IS_TND_TND) {
            return static_cast<uint64_t>(task.cuQStart * tiling_->numQHeads + task.mStart) * tiling_->strideQn;
        }
        if constexpr (FLOAT_DIRECT) {
            if (DirectBnsdQuery() && !FastBnsdQuery() && tiling_->numQHeads > 1U && tiling_->qMaxSeq > 1U) {
                return static_cast<uint64_t>(task.batchIdx) * tiling_->numQHeads * tiling_->qMaxSeq * tiling_->headDim;
            }
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
        if (PackQuery()) {
            return static_cast<uint64_t>(AscendC::GetBlockIdx()) * MSA_ROW_TILE_M * tiling_->headDim;
        }
        if (DirectBnsdQuery()) {
            return QuerySrcElem(task, task.mStart);
        }
        return MsaQueryTileElem(tiling_->queryLayout, task.batchIdx, task.cuQStart, task.mStart, tiling_->numQHeads,
                                tiling_->headDim, tiling_->qMaxSeq, tiling_->strideQn);
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
        if (PackQuery()) {
            return gQScaleScratchMx_[scaleOff];
        }
        return gQueryScaleMx_[scaleOff];
    }

    /// BSND 以及 N1=1 的 BNSD：逻辑行已经连续，一次搬完。BNSD 且 N1>1 时按 (token, head) 逐行收。
    __aicore__ inline bool QueryRowsContiguous() const
    {
        if constexpr (IS_TND_TND) {
            return true;
        }
        return tiling_->queryLayout == MSA_QUERY_LAYOUT_BSND || tiling_->numQHeads <= 1U;
    }

    __aicore__ inline uint64_t QuerySrcElem(const MsaTask& task, uint32_t localRow) const
    {
        if constexpr (IS_TND_TND) {
            return static_cast<uint64_t>(task.cuQStart * tiling_->numQHeads + task.mStart + localRow) *
                   tiling_->strideQn;
        }
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

    // 一次 2D MTE 收齐同一 token 的多个 head，UB 中每行按 32B 对齐。
    template <bool COMPACT = false>
    __aicore__ inline void GatherBnsdRows(const AscendC::LocalTensor<uint8_t>& ub,
                                          const AscendC::GlobalTensor<uint8_t>& gm, const MsaTask& task,
                                          uint32_t rowBytes, uint32_t srcDiv, uint32_t dstRowBytes,
                                          uint32_t srcMul = 1U)
    {
        uint32_t row = 0U;
        const uint32_t n = tiling_->numQHeads;
        while (row < task.mActual) {
            const uint32_t head = (task.mStart + row) % n;
            const uint32_t count = MsaMinU32(n - head, task.mActual - row);
            const uint64_t src = QuerySrcElem(task, task.mStart + row) / srcDiv * srcMul;
            AscendC::DataCopyExtParams copy{static_cast<uint16_t>(count), rowBytes,
                                            static_cast<uint32_t>((tiling_->qMaxSeq - 1U) * rowBytes), 0U, 0U};
            AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0U, 0U, 0U};
            AscendC::DataCopyPad<uint8_t, COMPACT ? AscendC::PaddingMode::Compact : AscendC::PaddingMode::Normal>(
                ub[row * dstRowBytes], gm[src], copy, pad);
            row += count;
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
        // Q reuses the S ping area after the previous task's vector reduction/store.
        // A faster gather must explicitly wait for both UB readers before overwriting it.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        const uint32_t gatherOff = IS_BLOCK16 ? UB_OFF_B16_GATHER : UB_OFF_CAST_FP;
        if constexpr (IS_MXFP4) {
            const uint32_t pack = d >> 1U;
            AscendC::LocalTensor<uint8_t> ub = resource.ubBuf.template GetBufferByByte<uint8_t>(gatherOff);
            if (QueryRowsContiguous()) {
                const uint64_t src = QuerySrcElem(task, task.mStart) >> 1U;
                CopyGmToUbPartial(ub, gQuery_[src].template ReinterpretCast<uint8_t>(), task.mActual * pack);
            } else {
                if (pack % 32U == 0U) {
                    GatherBnsdRows(ub, gQuery_.template ReinterpretCast<uint8_t>(), task, pack, 2U, pack);
                } else {
                    for (uint32_t row = 0; row < task.mActual; ++row) {
                        const uint64_t src = QuerySrcElem(task, task.mStart + row) >> 1U;
                        CopyGmToUbPartial(ub[row * pack], gQuery_[src].template ReinterpretCast<uint8_t>(), pack);
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
            uint32_t nCopy = task.mActual * pack;
            nCopy = (nCopy + 15U) & ~15U;
            const uint32_t cap = MSA_ROW_TILE_M * pack;
            if (nCopy > cap) {
                nCopy = cap;
            }
            AscendC::DataCopy(gQScratch_[dstBase >> 1U].template ReinterpretCast<uint8_t>(), ub, nCopy);
        } else {
            AscendC::LocalTensor<ElementQ> ub = resource.ubBuf.template GetBufferByByte<ElementQ>(gatherOff);
            if (QueryRowsContiguous()) {
                const uint64_t src = QuerySrcElem(task, task.mStart);
                CopyGmToUbPartial(ub, gQuery_[src], task.mActual * d);
            } else {
                GatherBnsdRows(ub.template ReinterpretCast<uint8_t>(), gQuery_.template ReinterpretCast<uint8_t>(),
                               task, d * sizeof(ElementQ), 1U, d * sizeof(ElementQ), sizeof(ElementQ));
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
        }
        if constexpr (IS_MXFP4) {
            const uint32_t groups = d / MSA_MX_GROUP;
            const uint32_t scaleUbOff = IS_BLOCK16 ? UB_OFF_B16_SCALE : UB_OFF_CAST_I8;
            AscendC::LocalTensor<uint8_t> ubSc = resource.ubBuf.template GetBufferByByte<uint8_t>(scaleUbOff);
            auto packedSc = ubSc;
            if (QueryRowsContiguous()) {
                const uint64_t srcSc = QuerySrcElem(task, task.mStart) / MSA_MX_GROUP;
                CopyGmToUbPartial(ubSc, gQueryScaleMx_[srcSc].template ReinterpretCast<uint8_t>(),
                                  task.mActual * groups);
            } else {
                if ((n * groups) % 32U == 0U && task.mStart % n == 0U && task.mActual % n == 0U) {
                    // 950 compact mode packs the short scale rows without 32B gaps.
                    GatherBnsdRows<true>(ubSc, gQueryScaleMx_.template ReinterpretCast<uint8_t>(), task, groups,
                                         MSA_MX_GROUP, groups);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
                } else {
                    // Partial head groups retain the original scalar copy semantics.
                    for (uint32_t row = 0; row < task.mActual; ++row) {
                        const uint64_t srcSc = QuerySrcElem(task, task.mStart + row) / MSA_MX_GROUP;
                        for (uint32_t g = 0; g < groups; ++g) {
                            ubSc.SetValue(row * groups + g,
                                          gQueryScaleMx_[srcSc].template ReinterpretCast<uint8_t>().GetValue(g));
                        }
                    }
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t nScCopy = (task.mActual * groups + 31U) & ~31U;
            const uint32_t scCap = MSA_ROW_TILE_M * groups;
            if (nScCopy > scCap) {
                nScCopy = scCap;
            }
            AscendC::DataCopy(gQScaleScratchMx_[dstBase / MSA_MX_GROUP].template ReinterpretCast<uint8_t>(), packedSc,
                              nScCopy);
        }
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

    __aicore__ inline void MmadGroupedToL0C(BlockMmad& blockMmad, const MsaTask& task, bool needLoadL1, uint32_t nPages)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nCols = nPages * MSA_BLOCK_SIZE;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKGm(headDim, nCols, tiling_->strideKvToken);
        const Catlass::GemmCoord shape{mActual, nCols, headDim};
        blockMmad.MmadToL0C(QueryGm(task), layoutQ, gKeyFp_, layoutKGm, shape, needLoadL1, 1U, nCols, true);
    }

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
                if constexpr (!ENABLE_TOPK) {
                    constexpr uint32_t compactPageElems = MSA_BLOCK_SIZE * 128U;
                    constexpr int32_t maxStridedCopyKvTokens = 131072;
                    if (headDim == 128U && strideKvToken == 128U && tiling_->numQHeads == 1U &&
                        tiling_->numKvHeads == 1U && task.kvLen <= maxStridedCopyKvTokens && nLoad == nCols &&
                        tiling_->strideKvBlock == 2U * compactPageElems) {
                        const int32_t page = PhysicalPageOfBlock(task, blk);
                        uint32_t run = 1U;
                        for (; run < pages; ++run) {
                            if (PhysicalPageOfBlock(task, blk + run) != static_cast<int64_t>(page) + run) {
                                break;
                            }
                        }
                        if (run == pages) {
                            const uint64_t compactKey = static_cast<uint64_t>(page) * compactPageElems;
                            blockMmad.LoadStridedBPagesToL1(gKeyFp_[compactKey],
                                                            gKeyScaleMx_[compactKey / MSA_MX_GROUP], pages);
                            return;
                        }
                    }
                }
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

    __aicore__ inline void LoadFloatWindowToL1(BlockMmad& blockMmad, const MsaTask& task, uint32_t win)
    {
        const uint32_t row0 = win * MSA_BLOCK_SIZE;
        const uint32_t cols = Min(MSA_BLOCK_SIZE, static_cast<uint32_t>(task.kvLen) - row0);
        const uint32_t page = tiling_->pageSize;
        uint32_t done = 0U;
        while (done < cols) {
            const uint32_t token = row0 + done;
            const uint32_t inPage = token % page;
            const uint32_t rows = Min(page - inPage, cols - done);
            const int32_t id = gBlockTable_.GetValue(task.batchIdx * tiling_->tableWidth + token / page);
            const uint64_t off = static_cast<uint64_t>(id) * tiling_->strideKvBlock +
                                 static_cast<uint64_t>(inPage) * tiling_->strideKvToken;
            const LayoutK srcLayout(tiling_->headDim, rows, tiling_->strideKvToken);
            blockMmad.AppendBPage(gKeyFp_[off], srcLayout, done, done == 0U, done + rows == cols);
            done += rows;
        }
    }

    __aicore__ inline void LoadWindowToL1(BlockMmad& blockMmad, const MsaTask& task, uint32_t win, uint32_t nWin)
    {
        if constexpr (FLOAT_DIRECT) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && !PaWindowInPage(task, win)) {
                LoadFloatWindowToL1(blockMmad, task, win);
                return;
            }
        }
        if constexpr (IS_BLOCK16 && IS_MXFP4) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout)) {
                LoadPaWindowToL1(blockMmad, task, win, nWin);
                return;
            }
        }
        if constexpr (!IS_BLOCK16) {
            return;
        }
        const uint32_t headDim = tiling_->headDim;
        const uint32_t kGemm = IS_MXFP4 ? (headDim >> 1) : headDim;
        const uint32_t strideK = IS_MXFP4 ? (tiling_->strideKvToken >> 1) : tiling_->strideKvToken;
        const uint32_t nCols = nWin * MSA_BLOCK_SIZE;
        const LayoutK layoutKGm(kGemm, nCols, strideK);
        const Catlass::GemmCoord shape{task.mActual, nCols, kGemm};
        const uint64_t kOff = WindowDirectGmOffset(task, win);
        if constexpr (IS_MXFP4 && !MXFP4_N512) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && nWin == 2U) {
                const uint64_t nextOff = WindowDirectGmOffset(task, win + 1U);
                if (nextOff != kOff + static_cast<uint64_t>(MSA_BLOCK_SIZE) * tiling_->strideKvToken) {
                    blockMmad.LoadBPairToL1(gKeyFp_[kOff >> 1U], gKeyFp_[nextOff >> 1U],
                                            gKeyScaleMx_[MxKeyScaleOffset(kOff)],
                                            gKeyScaleMx_[MxKeyScaleOffset(nextOff)], kGemm, strideK);
                    return;
                }
            }
        }
        const uint32_t loadCols =
            (IS_MXFP4 || FLOAT_DIRECT) ? Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE) : nCols;
        blockMmad.LoadBToL1(gKeyFp_[IS_MXFP4 ? (kOff >> 1) : kOff], layoutKGm, gKeyScaleMx_[MxKeyScaleOffset(kOff)],
                            shape, loadCols);
    }

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

    /// 块长 16 且整窗 key 在 GM 上连续：nWin 个窗并成一次 MMAD。earlyCv 在 Fixpipe 两路之间放行 AIV。
    __aicore__ inline void ComputeWindowDirectGm(BlockMmad& blockMmad, const MsaTask& task, uint32_t win,
                                                 const AscendC::LocalTensor<float>& ubS, bool needLoadL1, uint32_t nWin,
                                                 uint32_t cvPing, bool bLoaded
#if defined(ORIG_DTYPE_QUERY) && (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1)
                                                 ,
                                                 bool deferVc = false
#endif
    )
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
        const uint64_t kOff = WindowDirectGmOffset(task, win);
        if constexpr (IS_MXFP4) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && !bLoaded) {
                LoadPaWindowToL1(blockMmad, task, win, nWin);
                bLoaded = true;
            }
        }

        if constexpr (FLOAT_DIRECT) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && !bLoaded && !PaWindowInPage(task, win)) {
                LoadFloatWindowToL1(blockMmad, task, win);
                bLoaded = true;
            }
        }
        const LayoutQ layoutQ(mActual, kGemm, strideQ);
        const Catlass::GemmCoord shape{mActual, nCols, kGemm};
        if constexpr (IS_MXFP4 && !MXFP4_N512) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && nWin == 2U && !bLoaded) {
                LoadWindowToL1(blockMmad, task, win, nWin);
                bLoaded = true;
            }
        }
#if defined(ORIG_DTYPE_QUERY) && (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1)
        if constexpr (IS_MXFP4 && !MXFP4_N512) {
            if (deferVc) {
                blockMmad.MmadToL0CMx(QueryGm(task), layoutQ, gKeyFp_[kOff >> 1], layoutKGm, QueryScaleGm(task),
                                      gKeyScaleMx_[MxKeyScaleOffset(kOff)], shape, needLoadL1, split,
                                      Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE), bLoaded);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + cvPing);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + cvPing +
                                                                        MSA_A5_AIV_FLAG_OFFSET);
                blockMmad.FlushStoreToUb(ubS, cvPing, true);
                return;
            }
        }
#endif
        if constexpr (IS_MXFP4) {
            blockMmad.ComputeToUbMx(QueryGm(task), layoutQ, gKeyFp_[IS_MXFP4 ? (kOff >> 1) : kOff], layoutKGm,
                                    QueryScaleGm(task), gKeyScaleMx_[MxKeyScaleOffset(kOff)], ubS, shape, needLoadL1,
                                    split, Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE),
                                    bLoaded, 0U, true, cvPing);
        } else {
            blockMmad.ComputeToUb(
                QueryGm(task), layoutQ, gKeyFp_[kOff], layoutKGm, ubS, shape, needLoadL1, split,
                FLOAT_DIRECT ? Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE) : 0U, bLoaded, 0U,
                true, cvPing);
        }
    }

    /// MXFP4 宽窗：一次 MMAD 仍写满 N=nWin*128 的 FP32，再按 ping 容量把列切开。
    /// 调用方还没等 VC。MMAD 先占用 L0C，每片等自己的 ping，最后一片才交还 L0C。
    /// 返回占用的 ping 数，调用方据此推进 pageSeq。
    __aicore__ inline uint32_t ComputeWindowFp32Slices(BlockMmad& blockMmad, Catlass::Arch::Resource<ArchTag>& resource,
                                                       const MsaTask& task, uint32_t win, bool needLoadL1,
                                                       uint32_t nWin, uint32_t basePage, bool bLoaded)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const uint32_t kGemm = headDim >> 1;
        const uint32_t strideQ = tiling_->strideQn >> 1;
        const uint32_t strideK = tiling_->strideKvToken >> 1;
        const uint32_t nCols = nWin * MSA_BLOCK_SIZE;
        const LayoutK layoutKGm(kGemm, nCols, strideK);
        const LayoutQ layoutQ(mActual, kGemm, strideQ);
        const Catlass::GemmCoord shape{mActual, nCols, kGemm};
        const uint64_t kOff = WindowDirectGmOffset(task, win);
        if constexpr (IS_MXFP4) {
            if (MsaIsPaKeyLayout(tiling_->keyLayout) && !bLoaded) {
                LoadPaWindowToL1(blockMmad, task, win, nWin);
                bLoaded = true;
            }
        }

        blockMmad.MmadToL0CMx(QueryGm(task), layoutQ, gKeyFp_[kOff >> 1], layoutKGm, QueryScaleGm(task),
                              gKeyScaleMx_[MxKeyScaleOffset(kOff)], shape, needLoadL1, tiling_->numQHeads,
                              Min(nCols, static_cast<uint32_t>(task.kvLen) - win * MSA_BLOCK_SIZE), bLoaded, 0U);
        uint32_t fit = Block16Fp32WinsPerPing(task);
        if (fit > nWin) {
            fit = nWin;
        }
        if (fit == 0U) {
            fit = 1U;
        }
        uint32_t done = 0U;
        uint32_t slice = 0U;
        while (done < nWin) {
            uint32_t w = fit;
            if (w > nWin - done) {
                w = nWin - done;
            }
            const uint32_t sping = (basePage + slice) % MSA_A5_S_STAGES;
            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + sping);
            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + sping + MSA_A5_AIV_FLAG_OFFSET);
            auto ubS = resource.ubBuf.template GetBufferByByte<float>(sping * Epilogue::S_PING_BYTES);
            const bool last = (done + w) == nWin;
            blockMmad.FlushFp32Slice(ubS, done * MSA_BLOCK_SIZE, w * MSA_BLOCK_SIZE, sping, slice == 0U, last);
            done += w;
            ++slice;
        }
        return slice;
    }

    /// 块长 16：一次 GEMM 读 scratch 里已经拼好的 128 列。Fixpipe 按 numQHeads 整组切给两个 AIV。
    /// cvPing 有效时 earlyCv 由 Fixpipe 发 CV，调用方不要再 Set。
    __aicore__ inline void ComputeWindowToUb(BlockMmad& blockMmad, const MsaTask& task,
                                             const AscendC::LocalTensor<float>& ubS, uint64_t kScratchBase,
                                             bool needLoadL1, bool directK = false, uint64_t kGmOffset = 0U,
                                             uint32_t cvPing = 0U)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const uint32_t kGemm = IS_MXFP4 ? (headDim >> 1) : headDim;
        const uint32_t strideQ = IS_MXFP4 ? (tiling_->strideQn >> 1) : tiling_->strideQn;
        const LayoutQ layoutQ(mActual, kGemm, strideQ);
        const LayoutK layoutKScratch(kGemm, MSA_BLOCK_SIZE, kGemm);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, kGemm};
        const auto queryGm = QueryGm(task);
        const uint32_t split = tiling_->numQHeads;
        const uint64_t kBase = IS_MXFP4 ? (kScratchBase >> 1) : kScratchBase;
        if constexpr (IS_MXFP4) {
            const uint32_t groups = headDim / MSA_MX_GROUP;
            const uint64_t kScaleScratchOff = (kScratchBase / MSA_K_TILE) * groups;
            blockMmad.ComputeToUbMx(queryGm, layoutQ, gKeyScratch_[kBase], layoutKScratch, QueryScaleGm(task),
                                    gKeyScaleScratchMx_[kScaleScratchOff], ubS, shape, needLoadL1, split, 0U, false, 0U,
                                    true, cvPing);
        } else {
            // strideKvToken == headDim 时 layoutKScratch 与 GM 上的窗布局一致，换指针即可。
            if (directK) {
                blockMmad.ComputeToUb(queryGm, layoutQ, gKeyFp_[kGmOffset], layoutKScratch, ubS, shape, needLoadL1,
                                      split, 0U, false, 0U, true, cvPing);
                return;
            }
            blockMmad.ComputeToUb(queryGm, layoutQ, gKeyScratch_[kScratchBase], layoutKScratch, ubS, shape, needLoadL1,
                                  split, 0U, false, 0U, true, cvPing);
        }
    }
#endif

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        Epilogue epilogue;
        const bool selection = IS_BLOCK16 && tiling_->returnChoice != MSA_RETURN_SCORE;
        const uint32_t producerStride = selection ? runtimeScoreStride_ : tiling_->strideOutToken;
        const uint32_t producerHeadStride = selection ? tiling_->totalQ * runtimeScoreStride_ : tiling_->strideOutHead;
        const uint32_t producerBlockCap =
            selection && tiling_->keyLayout == MSA_KEY_LAYOUT_TND ? runtimeScoreStride_ : tiling_->maxBlocksPerBatch;
        epilogue.Init(resource, tiling_->numQHeads, producerHeadStride, producerStride, producerBlockCap,
                      tiling_->tableWidth, tiling_->poolBlockSize, tiling_->pageSize, tiling_->strideScalePage,
                      tiling_->strideScaleHead, IS_QUANT_V, IS_TND_TND ? MSA_KEY_LAYOUT_TND : tiling_->keyLayout,
                      tiling_->totalK, tiling_->headDim, tiling_->rowsPerBlock, tiling_->compressFactor, gScore_,
                      gScale_, gBlockTable_, IS_BLOCK16);
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
        if constexpr (IS_BLOCK16) {
            totalTasks = InitBalancedTasks(coreNum, totalTasks);
        }
        if constexpr (IS_BLOCK16 && IS_MXFP4 && !MXFP4_N512) {
            epilogue.EnableFusedHalfStage(balancedDirectFp4_);
        }
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + 0);
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + 1);
        MsaTask task;
        bool balanceWaves = BalanceCausalTaskWaves(coreNum, totalTasks);
        uint32_t wave = 0U;
        for (uint32_t taskSlot = coreIdx; taskSlot < totalTasks; taskSlot += coreNum, ++wave) {
            const uint32_t taskIdx = TaskOfWave(taskSlot, wave, coreIdx, coreNum, balanceWaves);
            if constexpr (IS_BLOCK16) {
                if (!DecodeBalancedTask(taskIdx, coreNum, task)) {
                    continue;
                }
            } else if (!scheduler_.DecodeScoreSlot(taskIdx, task)) {
                continue;
            }
            if constexpr (IS_MXFP4 && MX_FUSED_STAGE && !IS_BLOCK16 && USE_C2UB) {
                // Wave zero is always the identity mapping. Reuse its decoded
                // device lengths once; nearly rectangular requests stay in the
                // original order without introducing a host length sync.
                if (wave == 0U && balanceWaves) {
                    balanceWaves = task.kvLen > 0 && task.qLen >= task.kvLen / 2 + task.kvLen % 2;
                }
            }
            if (task.sBlkBegin >= task.sBlkEnd && !task.writeTailFill) {
                if constexpr (IS_BLOCK16) {
                    if (task.sChunks > 1U) {
                        epilogue.BeginTask(task, subIdx, subBlockNum);
                        epilogue.EndTask();
                    }
                }
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
            if constexpr (IS_BLOCK16) {
                // Q 在一个 task 内不变，整个 task 只收一次。首窗用 FLAG_K 让 Cube 看见它。
                if (subIdx == 0 && NeedQueryPrepare() && task.sStileBegin < task.sStileEnd) {
                    PrepareQuery(resource, task, static_cast<uint64_t>(coreIdx) * MSA_ROW_TILE_M * tiling_->headDim);
                }
                for (uint32_t win = task.sStileBegin; win < task.sStileEnd;) {
                    const uint32_t nWin = Block16WinGroup(task, win);
                    const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                    const bool direct = WindowDirectGm(task, win);
                    if (subIdx == 0 && !direct) {
                        GatherKeyWindowToScratch(resource, task, win, coreKScratch);
                    }
                    const bool needK = (IS_MXFP4 || FLOAT_DIRECT) ? ((pageSeq == 0U && NeedQueryPrepare()) || !direct) :
                                                                    ((pageSeq == 0U) || !direct);
                    if (needK) {
                        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
                    }
                    // 与 cube 的切片 Fixpipe 对齐：一片一次 CV，归约完再还对应的 ping。
                    bool sliced = false;
                    if constexpr (IS_MXFP4 && BLOCK16_FP32_MAX) {
                        uint32_t fit = Block16Fp32WinsPerPing(task);
                        if (fit > nWin) {
                            fit = nWin;
                        }
                        // 整组放得进一个 ping 时也走宽折叠。N=256 正好一 ping，逐窗 ProcessWindowUb 会把 Vector 拉长。
                        if (direct && nWin > 1U && fit >= nWin && epilogue.Fp32WideGroupFits(task, win, nWin)) {
                            AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                            epilogue.AbsorbFp32Slice(ping, task, win, nWin, 0U, nWin * MSA_BLOCK_SIZE,
                                                     epilogue.Fp32Block4Fusable(task, win, nWin));
                            epilogue.FinishFp32Group(task, win, nWin);
                            ++pageSeq;
                            sliced = true;
                        } else if (direct && fit < nWin && fit > 0U) {
                            const bool wide = epilogue.Fp32WideGroupFits(task, win, nWin);
                            const bool fuse = wide && epilogue.Fp32Block4Fusable(task, win, nWin);
                            uint32_t done = 0U;
                            uint32_t slice = 0U;
                            while (done < nWin) {
                                uint32_t w = fit;
                                if (w > nWin - done) {
                                    w = nWin - done;
                                }
                                const uint32_t sping = (pageSeq + slice) % MSA_A5_S_STAGES;
                                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + sping);
                                if (wide) {
                                    epilogue.AbsorbFp32Slice(sping, task, win + done, w, done * MSA_BLOCK_SIZE,
                                                             nWin * MSA_BLOCK_SIZE, fuse);
                                } else {
                                    epilogue.ProcessWindowUb(sping, task, win + done, w);
                                }
                                done += w;
                                ++slice;
                            }
                            if (wide) {
                                epilogue.FinishFp32Group(task, win, nWin);
                            }
                            pageSeq += slice;
                            sliced = true;
                        }
                    }
                    // bf16/fp16 的 Cube 仍是一窗 N=128。连续 direct 窗在 vector 侧折进同一条 T64，
                    // 每窗折完就还 ping，整组再做一次块 max。cf=1 或放不下时 Fp32WideGroupFits 为 false，走逐窗。
                    if constexpr (BLOCK16_FP32_MAX && !IS_MXFP4) {
                        if (!sliced && direct && nWin == 1U) {
                            uint32_t group = 1U;
                            while (group < MSA_BLOCK16_MAX_WIN_GROUP && (win + group) < task.sStileEnd &&
                                   WindowDirectGm(task, win + group) && Block16WinGroup(task, win + group) == 1U &&
                                   epilogue.Fp32WideGroupFits(task, win, group + 1U)) {
                                ++group;
                            }
                            if (group > 1U) {
                                const uint32_t totalCols = group * MSA_BLOCK_SIZE;
                                const bool fuse = epilogue.Fp32Block4Fusable(task, win, group);
                                for (uint32_t i = 0U; i < group; ++i) {
                                    const uint32_t sping = pageSeq % MSA_A5_S_STAGES;
                                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + sping);
                                    epilogue.AbsorbFp32Slice(sping, task, win + i, 1U, i * MSA_BLOCK_SIZE, totalCols,
                                                             fuse);
                                    ++pageSeq;
                                }
                                epilogue.FinishFp32Group(task, win, group);
                                win += group;
                                continue;
                            }
                        }
                    }
                    if (!sliced) {
                        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                        // ProcessWindowUb 读完 S ping 后自己还 VC。
                        epilogue.ProcessWindowUb(ping, task, win, nWin);
                        ++pageSeq;
                    }
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
            } else if constexpr (FP8_SCORE_PIPE) {
                if (!MsaIsSeqKeyLayout(tiling_->keyLayout)) {
                    for (uint32_t blk = task.sBlkBegin; blk < task.sBlkEnd;) {
                        const uint32_t nPages = Fp8PageGroup(task, blk);
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
    MsaTaskScheduler<LOGICAL_BLOCK_SIZE, IS_MXFP4 && !IS_BLOCK16 && !ENABLE_TOPK && !MX_LATENCY_TND> scheduler_;

    AscendC::GlobalTensor<ElementQ> gQuery_;
    AscendC::GlobalTensor<ElementQ> gQScratch_;
    AscendC::GlobalTensor<ElementQ> gKeyFp_;
    AscendC::GlobalTensor<int8_t> gKeyInt8_;
    AscendC::GlobalTensor<ElementQ> gKeyScratch_;
    AscendC::GlobalTensor<fp8_e8m0_t> gQueryScaleMx_;
    AscendC::GlobalTensor<fp8_e8m0_t> gQScaleScratchMx_;
    AscendC::GlobalTensor<fp8_e8m0_t> gKeyScaleMx_;
    AscendC::GlobalTensor<fp8_e8m0_t> gKeyScaleScratchMx_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gActualSeqQlen_;
    AscendC::GlobalTensor<int32_t> gActualSeqKlen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;
    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<int32_t> gTopkIdx_;
    AscendC::GlobalTensor<float> gTopkVal_;
    AscendC::GlobalTensor<ElementS> gWorkspace_;
    uint32_t runtimeScoreStride_ = 0U;
    bool balancedDirectFp4_ = false;
    uint32_t balancedTilesPerRequest_ = 0U;
    uint32_t balancedTilesPerCore_ = 0U;
    uint32_t balancedOldChunks_ = 0U;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_KERNEL_ARCH35_H
