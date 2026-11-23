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
 * \file text_gated_residual_grad_kernel_common.h
 * \brief Shared TextGatedResidualGrad initialization and Cube/Vector execution.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H
#define TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "text_gated_residual_grad_tiling_data_arch35.h"
#include "text_gated_residual_grad_pipeline_arch35.h"
#include "text_gated_residual_grad_chunk_state.h"
#include "text_gated_residual_grad_vector_arch35.h"
#include "text_gated_residual_grad_credit_sync.h"

namespace TextGatedResidualGrad {

// Ascend 950 and Atlas A2/A3 share Init/Process/RunCube. Each entry header supplies its Cube backend
// and tiles. USE_DUP_T/G56_PACKED_T remain available for transpose fallback paths.
template <typename D_T, int HAS_GAMMA, class GemmT, bool USE_DUP_T, bool G56_PACKED_T = true>
class KernelTextGatedResidualGrad {
public:
    __aicore__ inline void Init(GM_ADDR gradMixed, GM_ADDR x, GM_ADDR wDown, GM_ADDR wUp, GM_ADDR invRms,
                                GM_ADDR downPreAct, GM_ADDR up, GM_ADDR gradInject, GM_ADDR wGate, GM_ADDR gamma,
                                GM_ADDR inject, GM_ADDR dx, GM_ADDR dwDown, GM_ADDR dwUp, GM_ADDR dwGate,
                                GM_ADDR dgamma, GM_ADDR workspace,
                                const TextGatedResidualGradTilingDataArch35* tilingData, AscendC::TPipe* pipe)
    {
        taskRows_ = static_cast<uint32_t>(tilingData->taskRows);
        fullBatchRows_ = static_cast<uint32_t>(tilingData->fullBatchRows);
        lowpDx_ = tilingData->lowpDx != 0;
        gmDxOutput_ = dx;
        n_ = static_cast<uint32_t>(tilingData->n);
        c_ = static_cast<uint32_t>(tilingData->c);
        r_ = static_cast<uint32_t>(tilingData->r);
        d_ = static_cast<uint32_t>(tilingData->d);
        taskCount_ = static_cast<uint32_t>(tilingData->taskCount);
        workspaceSlots_ = static_cast<uint32_t>(tilingData->workspaceSlots);
        splitkFactor_ = static_cast<uint32_t>(tilingData->splitkFactor);
        hasGate_ = static_cast<uint32_t>(tilingData->hasGate);
        scheduleBeatCount_ = static_cast<uint32_t>(tilingData->scheduleBeatCount);
        aicNum_ = static_cast<uint32_t>(tilingData->aicNum);
        workspace_ = workspace;
        fullBatch_ = tilingData->scheduleMode == HC_GRAD_FULL_BATCH_SCHEDULE;
        producerConsumer_ = (tilingData->scheduleMode != 0) && !USE_DUP_T && !G56_PACKED_T;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        deferWeightGrads_ = tilingData->scheduleMode == HC_GRAD_A5_FULL_K_SCHEDULE;
#endif
        singleTask_ = taskCount_ == 1U && tilingData->nnMatmul == 0;
        gmWUp_ = wUp;
        gmWDown_ = wDown;
        gmWGate_ = wGate;
        const bool retainWeightOperands = fullBatch_ || deferWeightGrads_;
        const uint32_t retainedRows = fullBatch_ ? fullBatchRows_ : n_;
        const uint32_t denseColumns = r_ + (hasGate_ != 0U ? c_ : 0U);
        const uint32_t dxColumns =
            HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5 && fullBatch_ ? HcGradAlign16(r_ + HcGradAlign16(c_)) : denseColumns;
        layout_ = BuildHcGradWorkspaceLayout(
            workspaceSlots_, splitkFactor_, static_cast<uint32_t>(tilingData->aivNum), c_,
            static_cast<uint32_t>(tilingData->h), r_, d_, sizeof(D_T), HAS_GAMMA != 0, USE_DUP_T, taskRows_,
            retainWeightOperands ? retainedRows : 0U, fullBatch_, dxColumns, tilingData->recomputeDirect != 0,
            fullBatch_, fullBatch_ && !lowpDx_, fullBatch_);
        gmDwDown_ = Pointer(layout_.dwDownAccum);
        gmDwUp_ = Pointer(layout_.dwUpAccum);
        gmDwGate_ = Pointer(layout_.dwGateAccum);
        g56Rows_ = layout_.g56Rows;
        gateRows_ = layout_.gateRows;
        credit_.Init(workspace_, 0U, taskCount_, workspaceSlots_, aicNum_, static_cast<uint32_t>(tilingData->aivNum));
        if ASCEND_IS_AIC {
            gemm_.Init(aicNum_, tilingData, workspace);
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            gemm_.Init(aicNum_, tilingData, workspace);
#elif HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
            if (fullBatch_) {
                gemm_.Init(aicNum_, tilingData, workspace);
            }
#endif
            vector_.Init(gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInject, wGate, gamma, inject, dx, dwDown,
                         dwUp, dwGate, dgamma, workspace, layout_, tilingData, pipe);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            vector_.SetBlockedG56Xn(gemm_.UsesBlockedG56Xn(), gemm_.UsesPreparedG56A());
#endif
        }
    }

    __aicore__ inline void Process()
    {
        // Empty token input still requires zeroing nonempty dw/dgamma outputs, but must bypass credit/ring scheduling.
        // With workspaceSlots=0, TaskPointer modulo operations or a DGAMMA step can hang the kernel.
        if (taskCount_ == 0U) {
            if ASCEND_IS_AIV {
                vector_.Prepare();
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
            AscendC::SyncAll<false>();
#endif
            return;
        }
        if ASCEND_IS_AIV {
            vector_.Prepare();
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            if (!fullBatch_) {
                vector_.PrepareWeights();
            }
#else
            vector_.PrepareWeights();
#endif
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        if (fullBatch_) {
            // The first V1 publication orders all writes from Prepare and V1.
            ProcessA2ChunkCredits();
            return;
        }
        if (!producerConsumer_) {
            AscendC::SyncAll<false>();
        }
#elif HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (fullBatch_) {
            ProcessChunkedFullBatch();
            return;
        }
#else
        AscendC::SyncAll<false>();
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        // V1 and V3 publish the first Cube operands and the prepared weights.
        if (!producerConsumer_ || deferWeightGrads_) {
            AscendC::SyncAll<false>();
        }
        if (deferWeightGrads_) {
            ProcessTokenPipeline<true>();
        } else
#endif
        {
            ProcessTokenPipeline<false>();
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (producerConsumer_ && deferWeightGrads_) {
            AscendC::SyncAll<false>();
        }
        if (deferWeightGrads_) {
            // The final DX beat already launched the full-K dW GEMMs; publish their result.
            ComputeFullWeightGradients(true);
        }
#elif HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2
        if (producerConsumer_) {
            AscendC::SyncAll<false>();
        }
#endif
        // Bounded mode has already reduced every per-task dW partial into the FP32 accumulators.
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            // Reduction and cast assign different ranges to each Vector core.
            credit_.JoinAivWritesForReads();
#endif
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        AscendC::SyncAll<false>();
#endif
    }

private:
    template <bool DEFER_WEIGHT_GRADS>
    __aicore__ inline void ProcessTokenPipeline()
    {
        HcGradScheduler<USE_DUP_T, G56_PACKED_T, DEFER_WEIGHT_GRADS> scheduler(taskCount_, workspaceSlots_,
                                                                               HAS_GAMMA != 0);
        for (uint32_t beatIndex = 0; beatIndex < scheduleBeatCount_ && !scheduler.Done(); ++beatIndex) {
            const HcGradBeat beat = scheduler.Next();
            if constexpr (!DEFER_WEIGHT_GRADS) {
                if (!producerConsumer_) {
                    ProcessSynchronizedBeat(beat);
                    continue;
                }
            }
            // Both sides share the beat sequence; credits provide cross-side readiness and slot reuse.
            ProcessCubeCredits<DEFER_WEIGHT_GRADS>(beat);
            ProcessNnVector(beat);
            ProcessVectorCredits<DEFER_WEIGHT_GRADS>(beat);
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2
            SynchronizeDgamma(beat);
#endif
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void FinishFullBatchStage()
    {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SyncAll<false>();
    }

    __aicore__ inline void ConvertFullBatchInputs(uint32_t stage, GM_ADDR a, GM_ADDR b)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        const uint32_t plan = gemm_.ConversionPlan(chunk_.rows);
        if (!gemm_.NeedsNnConversion(stage, plan)) {
            // V3 and DX already published their outputs immediately before these two calls.
            if (stage != HC_GRAD_NN_STAGE_DX && stage != HC_GRAD_NN_STAGE_C1B) {
                FinishFullBatchStage();
            }
            return;
        }
        if ASCEND_IS_AIV {
            gemm_.ConvertNnInputs(stage, plan, a, b, vector_.NnScratch());
            vector_.FinishNnScratch();
        }
#else
        // A5 Blaze consumes the retained ND operands directly. Its callers already
        // publish V1/V3/DX at the enclosing stage boundary, so no conversion barrier is needed.
        (void)stage;
        (void)a;
        (void)b;
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        FinishFullBatchStage();
#endif
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    static constexpr uint32_t C1A_SLABS = 2U;

    __aicore__ inline void InitC1aSlab(const TextGatedResidualGradTilingDataArch35* data)
    {
        if (fullBatchRows_ < 16384U || fullBatchRows_ > 49152U || c_ != 4U || d_ != 10240U || r_ != 320U ||
            data->aivNum != HC_GRAD_AIV_PER_AIC * data->aicNum || data->aivNum % c_ != 0U ||
            data->vectorUbBytes < HC_GRAD_A2_NN_UB_BYTES || splitkFactor_ != 1U || (data->nnMixMask[0] & 1U) == 0U) {
            return;
        }
        const auto& nn = data->nnBase[0][0];
        const uint32_t tiles = nn.tileL2cacheTiling.mTileCntL2;
        const uint32_t rowsPerTile = nn.tileL2cacheTiling.mTileBlock * nn.matmulTiling.singleCoreM;
        if (nn.matmulRunInfo.nd2nzA == 0U && nn.matmulRunInfo.nd2nzB == 1U && nn.matmulRunInfo.transA == 0U &&
            nn.matmulRunInfo.transB == 0U && nn.matmulTiling.M == fullBatchRows_ && nn.matmulTiling.N == r_ &&
            nn.matmulTiling.singleCoreK == d_ && tiles != 0U && tiles % C1A_SLABS == 0U &&
            static_cast<uint64_t>(tiles) * rowsPerTile == fullBatchRows_) {
            c1aTilesPerSlab_ = tiles / C1A_SLABS;
        }
    }

    __aicore__ inline void ProcessC1aSlabs()
    {
        // WUp needs one conversion. NN scratch is released before V1 takes ownership of UB.
        ConvertFullBatchInputs(HC_GRAD_NN_STAGE_C1A, TaskPointer(layout_.dup, 0U, d_), gmWUp_);
        if ASCEND_IS_AIV {
            const uint32_t rows = chunk_.rows / C1A_SLABS;
            for (uint32_t slab = 0; slab < C1A_SLABS; ++slab) {
                vector_.ProcessFullV1Slab(slab * rows, rows);
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                credit_.SameKindBarrier();
                credit_.MarkC1aSlabReady();
            }
            // C1a only reads dup. C1b first reads act after the enclosing stage barrier.
            vector_.ProcessFullV1Act();
        } else {
            gemm_.RunC1aSlabs(TaskPointer(layout_.dup, 0U, d_), Pointer(layout_.dactPartial), c1aTilesPerSlab_,
                              HcGradCreditSync::C1A_SLAB_READY);
        }
        FinishFullBatchStage();
    }
#endif

    __aicore__ inline void SetChunkState(uint32_t rowBegin, uint32_t chunk)
    {
        const uint32_t remaining = n_ - rowBegin;
        chunk_.rowBegin = rowBegin;
        chunk_.rows = remaining < fullBatchRows_ ? remaining : fullBatchRows_;
        chunk_.taskCount = (chunk_.rows + taskRows_ - 1U) / taskRows_;
        chunk_.parity = chunk & 1U;
        if ASCEND_IS_AIV {
            vector_.SetChunk(rowBegin, chunk_.rows, chunk_.taskCount, chunk_.parity);
        }
    }

    __aicore__ inline void ProduceV1()
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        if (!layout_.duplicateV1Operands && c1aTilesPerSlab_ != 0U && chunk_.rows == fullBatchRows_) {
            ProcessC1aSlabs();
        } else
#endif
            if ASCEND_IS_AIV {
            if (!vector_.TryProcessFullV1()) {
                for (uint32_t task = 0; task < chunk_.taskCount; ++task) {
                    vector_.Process(HcGradVectorOp::V1, task);
                }
            }
        }
    }

    __aicore__ inline void ComputeC1a()
    {
        ConvertFullBatchInputs(HC_GRAD_NN_STAGE_C1A, TaskPointer(layout_.dup, 0U, d_), gmWUp_);
        if ASCEND_IS_AIC {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_V1_READY);
#endif
            RunC1(0U, chunk_.rows);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_C1A_READY>();
#endif
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        FinishFullBatchStage();
#endif
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void ProcessC1aV3Gm()
    {
        if (layout_.duplicateV1Operands || c1aTilesPerSlab_ == 0U || chunk_.rows != fullBatchRows_) {
            ComputeC1a();
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            // At chunk i>0 every publishing AIC has also finished G56(i-1).
            // V3 may now reuse the common packed storage.
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_C1A_READY);
#endif
            if (!vector_.TryProcessV3FullBatch()) {
                for (uint32_t task = 0; task < chunk_.taskCount; ++task) {
                    vector_.Process(HcGradVectorOp::REDUCE_DACT, task);
                    vector_.Process(HcGradVectorOp::V3, task);
                }
            }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V3_READY);
#endif
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        FinishFullBatchStage();
#endif
    }

    __aicore__ inline void ProcessC1aV3Ub()
    {
        const bool needsPreviousG56 = chunk_.rowBegin != 0U;
        if ASCEND_IS_AIC {
            if (needsPreviousG56) {
                credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_G56_READ_DONE>();
            }
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_V1_READY);
            gemm_.RunC1aToV3Ub(TaskPointer(layout_.dup, 0U, d_), gmWUp_);
        }
        if ASCEND_IS_AIV {
            const uint32_t cube = AscendC::GetBlockIdx() / 2U;
            if (cube < HC_GRAD_C1A_UB_CORES) {
                // Drain V1 and opening WConcat before granting write access
                // to the local UB. The next chunk issues a fresh credit.
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(HC_GRAD_C1A_UB_FREE);
            }
            if (needsPreviousG56) {
                // The previous G56 owns the shared prepared operand until its final read.
                credit_.WaitChunkStage(HcGradCreditSync::CHUNK_G56_READ_DONE);
            }
            if (cube < HC_GRAD_C1A_UB_CORES) {
                const uint32_t tileBegin = cube * HC_GRAD_C1A_UB_TILE_M;
                const uint32_t pairRows = AscendC::Std::min(HC_GRAD_C1A_UB_TILE_M, chunk_.rows - tileBegin);
                const uint32_t rows = pairRows / 2U;
                const uint32_t begin = tileBegin + AscendC::GetSubBlockIdx() * rows;
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(HC_GRAD_C1A_UB_READY);
                vector_.ProcessV3FromC1aUb(begin, rows);
            }
            credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V3_READY);
        }
    }
#endif

    __aicore__ inline void ProcessAfterV1()
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            if (gemm_.CanRunC1aToV3Ub(chunk_.rows)) {
                ProcessC1aV3Ub();
            } else {
                ProcessC1aV3Gm();
            }
        } else {
#endif
            if (layout_.duplicateV1Operands || c1aTilesPerSlab_ == 0U || chunk_.rows != fullBatchRows_) {
                ComputeC1a();
            }
            if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                // At chunk i>0 every publishing AIC has also finished G56(i-1).
                // V3 may now reuse the common packed storage.
                credit_.WaitChunkStage(HcGradCreditSync::CHUNK_C1A_READY);
#endif
                if (!vector_.TryProcessV3FullBatch()) {
                    for (uint32_t task = 0; task < chunk_.taskCount; ++task) {
                        vector_.Process(HcGradVectorOp::REDUCE_DACT, task);
                        vector_.Process(HcGradVectorOp::V3, task);
                    }
                }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V3_READY);
#endif
            }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
            FinishFullBatchStage();
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        }
#endif
        ConvertFullBatchInputs(HC_GRAD_NN_STAGE_DX, DxMatrix(0U), hasGate_ != 0U ? Pointer(layout_.wConcat) : gmWDown_);
        if ASCEND_IS_AIC {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_V3_READY);
#endif
            RunDx(0U, chunk_.rows);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_DX_READY>();
#endif
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        // Publish the intermediate DX before V6 reads it.
        FinishFullBatchStage();
#endif
        ConvertFullBatchInputs(HC_GRAD_NN_STAGE_C1B, TaskPointer(layout_.dup, 0U, d_),
                               TaskPointer(layout_.act, 0U, r_));
        if ASCEND_IS_AIC {
            RunC1b();
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_DX_READY);
#endif
            vector_.ResumeDgammaAfterNn(chunk_.rowBegin != 0U);
            vector_.ProcessFullV6();
            if constexpr (HAS_GAMMA) {
                vector_.Process(HcGradVectorOp::DGAMMA, HC_GRAD_NO_TASK);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if (chunk_.rows == n_ - chunk_.rowBegin) {
                    credit_.JoinAivWritesForReads();
                }
#endif
            }
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        FinishFullBatchStage();
#endif
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    template <uint32_t STAGE, uint16_t READY, bool READS_PEER_OPERANDS = false>
    __aicore__ inline void PrepareA2ChunkGemm(GM_ADDR a, GM_ADDR b)
    {
        if ASCEND_IS_AIV {
            const uint32_t plan = gemm_.ConversionPlan(chunk_.rows);
            if (gemm_.NeedsNnConversion(STAGE, plan)) {
                if constexpr (READS_PEER_OPERANDS) {
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                    credit_.SameKindBarrier();
                }
                gemm_.ConvertNnInputs(STAGE, plan, a, b, vector_.NnScratch());
                vector_.FinishNnScratch();
            }
            credit_.PublishA2ChunkEdge(READY);
        } else {
            credit_.WaitA2ChunkEdge(READY);
        }
    }

    __aicore__ inline uint32_t PrefetchA2NextV1(uint32_t nextRow, uint32_t nextChunk)
    {
        if (!layout_.recomputeDirect || hasGate_ == 0U || !HAS_GAMMA || nextRow >= n_ || !vector_.CanUseFullV1Slabs()) {
            return 0U;
        }
        const uint32_t rows = AscendC::Std::min(fullBatchRows_, n_ - nextRow);
        const uint32_t tasks = (rows + taskRows_ - 1U) / taskRows_;
        constexpr uint32_t PREFETCH_ROWS = 1024U;
        const uint32_t head = AscendC::Std::min(rows, PREFETCH_ROWS);
        vector_.SetChunk(nextRow, rows, tasks, nextChunk & 1U);
        vector_.ProcessFullV1Slab(0U, head);
        vector_.SetChunk(chunk_.rowBegin, chunk_.rows, chunk_.taskCount, chunk_.parity);
        return head;
    }

    __aicore__ inline void ProcessA2ChunkCredits()
    {
        SetChunkState(0U, 0U);
        ProduceV1();
        for (uint32_t row = 0U, index = 0U; row < n_; row += fullBatchRows_, ++index) {
            const uint32_t next = row + fullBatchRows_;
            const bool hasNext = next < n_;
            PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_C1A, HcGradCreditSync::A2_C1A_INPUT_READY, true>(
                TaskPointer(layout_.dup, 0U, d_), gmWUp_);
            if ASCEND_IS_AIC {
                RunC1(0U, chunk_.rows);
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1A_OUTPUT_READY);
            } else {
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1A_OUTPUT_READY);
                if (!vector_.TryProcessV3FullBatch()) {
                    for (uint32_t task = 0U; task < chunk_.taskCount; ++task) {
                        vector_.Process(HcGradVectorOp::REDUCE_DACT, task);
                        vector_.Process(HcGradVectorOp::V3, task);
                    }
                }
            }
            PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_DX, HcGradCreditSync::A2_DX_INPUT_READY, true>(
                DxMatrix(0U), hasGate_ != 0U ? Pointer(layout_.wConcat) : gmWDown_);
            uint32_t prefetchedRows = 0U;
            if ASCEND_IS_AIC {
                RunDx(0U, chunk_.rows);
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_DX_OUTPUT_READY);
            } else {
                // The old owner of the next parity released it at the previous
                // G56 edge. Recomputed direct gradients need no shared V1 buffer.
                prefetchedRows = PrefetchA2NextV1(next, index + 1U);
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_DX_OUTPUT_READY);
            }
            PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_C1B, HcGradCreditSync::A2_C1B_INPUT_READY>(
                TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_));
            if ASCEND_IS_AIC {
                RunC1b();
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1B_READ_DONE);
            } else {
                vector_.ResumeDgammaAfterNn(row != 0U);
                vector_.ProcessFullV6();
                if constexpr (HAS_GAMMA) {
                    vector_.Process(HcGradVectorOp::DGAMMA, HC_GRAD_NO_TASK);
                    if (!hasNext) {
                        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                        credit_.SameKindBarrier();
                        vector_.Process(HcGradVectorOp::REDUCE_DGAMMA, HC_GRAD_NO_TASK);
                    }
                }
                // C1b still owns the shared NN scratch until its last read.
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1B_READ_DONE);
            }
            PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_G56, HcGradCreditSync::A2_G56_INPUT_READY>(
                G56Matrix(), TaskPointer(layout_.xn, 0U, d_));
            if ASCEND_IS_AIC {
                RunG56(index != 0U);
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_G56_READ_DONE);
                if (hasNext) {
                    SetChunkState(next, index + 1U);
                }
            } else {
                if (hasNext) {
                    SetChunkState(next, index + 1U);
                    if (prefetchedRows != 0U) {
                        vector_.ProcessFullV1Act();
                        vector_.ProcessFullV1Slab(prefetchedRows, chunk_.rows - prefetchedRows);
                    } else {
                        ProduceV1();
                    }
                }
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_G56_READ_DONE);
            }
        }
        if ASCEND_IS_AIV {
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }
#endif

    __aicore__ inline void ProcessChunkedFullBatch()
    {
        SetChunkState(0U, 0U);
        ProduceV1();
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if ASCEND_IS_AIV {
            credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V1_READY);
            // V3 publishes the DX weights after C1a has consumed WUp.
            AscendC::PipeBarrier<PIPE_ALL>();
            vector_.PrepareWeights();
        }
#endif
        for (uint32_t rowBegin = 0U, chunk = 0U; rowBegin < n_; rowBegin += fullBatchRows_, ++chunk) {
            ProcessAfterV1();
            if ASCEND_IS_AIV {
                if constexpr (HAS_GAMMA) {
                    if (chunk_.rows == n_ - rowBegin) {
                        vector_.Process(HcGradVectorOp::REDUCE_DGAMMA, HC_GRAD_NO_TASK);
                    }
                }
            }
            const uint32_t nextRowBegin = rowBegin + fullBatchRows_;
            ConvertFullBatchInputs(HC_GRAD_NN_STAGE_G56, G56Matrix(), TaskPointer(layout_.xn, 0U, d_));
            if (nextRowBegin < n_) {
                if ASCEND_IS_AIC {
                    RunG56(chunk != 0U);
                }
                if ASCEND_IS_AIV {
                    // The next V1 fills the opposite operand bank while Cube reads this bank.
                    SetChunkState(nextRowBegin, chunk + 1U);
                    ProduceV1();
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                    credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V1_READY);
#endif
                }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
                FinishFullBatchStage();
#endif
                SetChunkState(nextRowBegin, chunk + 1U);
            } else {
                if ASCEND_IS_AIC {
                    RunG56(chunk != 0U);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                    credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_DW_READY>();
#endif
                }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
                FinishFullBatchStage();
#endif
            }
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            // All Cube dW writers have drained before Vector reads their accumulators.
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_DW_READY);
#endif
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        AscendC::SyncAll<false>();
#endif
    }
#endif

    __aicore__ inline void RunC1b()
    {
        if (fullBatch_) {
            gemm_.RunC1b(TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_),
                         Pointer(layout_.dwUpPartial), chunk_.rows, r_, d_, chunk_.rowBegin != 0U);
        } else {
            gemm_.RunC1bDirect(TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_), gmDwUp_, n_, r_, d_);
        }
    }

    __aicore__ inline void RunG56(bool accumulate = false)
    {
        const uint32_t stride = fullBatch_ && hasGate_ == 0 ? r_ : g56Rows_;
        const uint32_t rows = fullBatch_ ? chunk_.rows : n_;
        gemm_.RunG56MergedPacked(G56Matrix(), TaskPointer(layout_.xn, 0U, d_), Pointer(layout_.dwPartial), rows, r_, d_,
                                 stride, gateRows_, hasGate_ != 0, accumulate);
    }

    __aicore__ inline GM_ADDR G56Matrix() const
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (gemm_.UsesPreparedG56A()) {
            return Pointer(layout_.dact);
        }
#endif
        // Without a gate, G56 and DX both consume the same dense ddown matrix.
        return fullBatch_ && hasGate_ == 0 ? DxMatrix(0U) : TaskPointer(layout_.packed, 0U, g56Rows_);
    }

    __aicore__ inline void RunFullWeightMatmuls()
    {
        // Each dW output tile has one AIC owner with K=[0,n_).
        // The host selects a deterministic NN path without an AIV epilogue.
        RunC1b();
        RunG56();
    }

    __aicore__ inline void ComputeFullWeightGradients(bool weightsComputed)
    {
        if ASCEND_IS_AIC {
            if (!weightsComputed) {
                RunFullWeightMatmuls();
            }
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            vector_.CopyFullWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }

    __aicore__ inline bool CubeNeedsVector(HcGradCubeOp op, uint32_t task)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (op == HcGradCubeOp::NONE)
            return false;
        const uint32_t rows = Rows(task);
        if (op == HcGradCubeOp::C1) {
            return gemm_.NeedsVector(0, rows) || (!deferWeightGrads_ && gemm_.NeedsVector(1, rows));
        }
        if (op == HcGradCubeOp::G56)
            return !deferWeightGrads_ && gemm_.NeedsVector(HC_GRAD_NN_STAGE_G56, rows);
        if (op == HcGradCubeOp::DX)
            return gemm_.NeedsVector(HC_GRAD_NN_STAGE_DX, rows);
#endif
        return false;
    }
    __aicore__ inline void ProcessNnVector(const HcGradBeat& beat)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if ASCEND_IS_AIV {
            const bool first = CubeNeedsVector(beat.cubeOp, beat.cubeTask);
            const bool second = CubeNeedsVector(beat.cubeOp2, beat.cubeTask2);
            if (!first && !second)
                return;
            AscendC::PipeBarrier<PIPE_ALL>();
            if (first)
                RunCube(beat.cubeOp, beat.cubeTask);
            if (second)
                RunCube(beat.cubeOp2, beat.cubeTask2);
            vector_.FinishNnScratch();
        }
#endif
    }
    __aicore__ inline void ProcessSynchronizedBeat(const HcGradBeat& beat)
    {
        ProcessNnVector(beat);
        if ASCEND_IS_AIC {
            RunCube(beat.cubeOp, beat.cubeTask);
            RunCube(beat.cubeOp2, beat.cubeTask2);
        }
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                if (beat.vectorOp[i] != HcGradVectorOp::NONE) {
                    vector_.Process(beat.vectorOp[i], beat.vectorTask[i]);
                }
            }
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }

    template <bool DEFER_WEIGHT_GRADS>
    __aicore__ inline void ProcessCubeCredits(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIC {
            const bool hasCube = (beat.cubeOp != HcGradCubeOp::NONE) || (beat.cubeOp2 != HcGradCubeOp::NONE);
            credit_.WaitCubeDeps<DEFER_WEIGHT_GRADS>(beat.cubeOp, beat.cubeTask);
            RunCube(beat.cubeOp, beat.cubeTask);
            credit_.WaitCubeDeps<DEFER_WEIGHT_GRADS>(beat.cubeOp2, beat.cubeTask2);
            RunCube(beat.cubeOp2, beat.cubeTask2);
            if (hasCube) {
                credit_.SameKindBarrier();
                credit_.MarkCube<DEFER_WEIGHT_GRADS>(beat.cubeOp, beat.cubeTask);
                credit_.MarkCube<DEFER_WEIGHT_GRADS>(beat.cubeOp2, beat.cubeTask2);
            }
            if constexpr (DEFER_WEIGHT_GRADS) {
                if (beat.cubeOp == HcGradCubeOp::DX && beat.cubeTask + 1U == taskCount_) {
                    // The last DX acquired V3's final publication. All retained dW
                    // operands are visible, while the AIVs can consume the last dx2 slot.
                    RunFullWeightMatmuls();
                }
            }
        }
    }

    template <bool DEFER_WEIGHT_GRADS>
    __aicore__ inline void ProcessVectorCredits(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIV {
            bool hasVector = false;
            for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                if (beat.vectorOp[i] != HcGradVectorOp::NONE) {
                    credit_.WaitVectorDeps<DEFER_WEIGHT_GRADS>(beat.vectorOp[i], beat.vectorTask[i]);
                    vector_.Process(beat.vectorOp[i], beat.vectorTask[i]);
                    hasVector = true;
                }
            }
            if (hasVector) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                credit_.SameKindBarrier();
                for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                    credit_.MarkVector<DEFER_WEIGHT_GRADS>(beat.vectorOp[i], beat.vectorTask[i]);
                }
            }
        }
    }

    __aicore__ inline void SynchronizeDgamma(const HcGradBeat& beat)
    {
        // DGAMMA publishes per-core partials before the next beat reads them.
        for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
            if (beat.vectorOp[i] == HcGradVectorOp::DGAMMA) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if ASCEND_IS_AIV {
                    credit_.JoinAivWritesForReads();
                }
#else
                AscendC::SyncAll<false>();
#endif
                break;
            }
        }
    }

    __aicore__ inline GM_ADDR TaskPointer(uint64_t offset, uint32_t task, uint32_t rowStride) const
    {
        return workspace_ + layout_.TaskBufferOffset(offset, task, workspaceSlots_,
                                                     static_cast<uint64_t>(taskRows_) * rowStride * sizeof(D_T),
                                                     chunk_.parity);
    }

    __aicore__ inline GM_ADDR TaskPointerF32(uint64_t offset, uint32_t task, uint32_t rowStride) const
    {
        return workspace_ + layout_.TaskBufferOffset(offset, task, workspaceSlots_,
                                                     static_cast<uint64_t>(taskRows_) * rowStride * sizeof(float),
                                                     chunk_.parity);
    }

    __aicore__ inline GM_ADDR SlotPointerF32(uint64_t offset, uint32_t task, uint64_t elements) const
    {
        return workspace_ + layout_.TaskBufferOffset(offset, task, workspaceSlots_, elements * sizeof(float));
    }

    __aicore__ inline GM_ADDR Pointer(uint64_t offset) const
    {
        return workspace_ + offset;
    }

    __aicore__ inline uint32_t Rows(uint32_t task) const
    {
        const uint32_t rowBegin = task * taskRows_;
        return n_ - rowBegin < taskRows_ ? n_ - rowBegin : taskRows_;
    }

    __aicore__ inline void RunC1(uint32_t task, uint32_t rows)
    {
        GM_ADDR dactPartial =
            SlotPointerF32(layout_.dactPartial, task, static_cast<uint64_t>(splitkFactor_) * taskRows_ * r_);
        gemm_.RunC1aSplitK(TaskPointer(layout_.dup, task, d_), gmWUp_, dactPartial, rows, r_, d_, splitkFactor_);
        if (fullBatch_ || deferWeightGrads_) {
            return; // Full-K modes consume retained operands after the token pipeline drains.
        }
        // Bounded mode finishes dW for this task before its ring slot can be reused.
        const uint64_t c1bA = USE_DUP_T ? layout_.dupT : layout_.dup;
        if (singleTask_) {
            gemm_.RunC1bDirect(TaskPointer(c1bA, task, d_), TaskPointer(layout_.act, task, r_), gmDwUp_, rows, r_, d_);
        } else {
            gemm_.RunC1b(TaskPointer(c1bA, task, d_), TaskPointer(layout_.act, task, r_),
                         SlotPointerF32(layout_.dwPartial, task, layout_.dwSlotElems), rows, r_, d_);
        }
    }

    __aicore__ inline GM_ADDR DxMatrix(uint32_t task) const
    {
        return layout_.denseDx != 0U ? Pointer(layout_.denseDx) : TaskPointer(layout_.packed, task, g56Rows_);
    }

    __aicore__ inline void RunDx(uint32_t task, uint32_t rows)
    {
        const bool dense = layout_.denseDx != 0U;
        const uint32_t stride = dense ? r_ + (hasGate_ != 0U ? c_ : 0U) : g56Rows_;
        GM_ADDR dxOut = TaskPointerF32(layout_.dx2, task, d_);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (fullBatch_ && lowpDx_) {
            dxOut = gmDxOutput_ + static_cast<uint64_t>(chunk_.rowBegin) * d_ * sizeof(D_T);
        }
#endif
        gemm_.RunDx(DxMatrix(task), gmWDown_, gmWGate_, Pointer(layout_.wConcat), dxOut, rows, d_, r_, c_, stride,
                    hasGate_ != 0U);
    }

    __aicore__ inline void RunCube(HcGradCubeOp op, uint32_t task)
    {
        const uint32_t rows = Rows(task);
        switch (op) {
            case HcGradCubeOp::C1: {
                RunC1(task, rows);
                break;
            }
            case HcGradCubeOp::G56: {
                if (fullBatch_ || deferWeightGrads_) {
                    break; // Full-K modes retire this marker without overwriting retained inputs.
                }
                GM_ADDR packedT = TaskPointer(layout_.packedT, task, g56Rows_);
                GM_ADDR packed = TaskPointer(layout_.packed, task, g56Rows_);
                GM_ADDR xn = TaskPointer(layout_.xn, task, d_);
                GM_ADDR dwPartial = SlotPointerF32(layout_.dwPartial, task, layout_.dwSlotElems);
                if constexpr (G56_PACKED_T) {
                    if (singleTask_) {
                        gemm_.RunG56Direct(packedT, xn, gmDwDown_, dwPartial + layout_.dwGateRowElems * sizeof(float),
                                           rows, r_, d_, gateRows_, hasGate_ != 0);
                    } else {
                        gemm_.RunG56Merged(packedT, xn, dwPartial, rows, r_, d_, gateRows_, hasGate_ != 0);
                    }
                } else {
                    if (singleTask_) {
                        gemm_.RunG56DirectPacked(packed, xn, gmDwDown_,
                                                 dwPartial + layout_.dwGateRowElems * sizeof(float), rows, r_, d_,
                                                 g56Rows_, gateRows_, hasGate_ != 0);
                    } else {
                        gemm_.RunG56MergedPacked(packed, xn, dwPartial, rows, r_, d_, g56Rows_, gateRows_,
                                                 hasGate_ != 0);
                    }
                }
                break;
            }
            case HcGradCubeOp::DX:
                RunDx(task, rows);
                break;
            default:
                break;
        }
    }

    uint32_t n_{0};
    uint32_t c_{0};
    uint32_t r_{0};
    uint32_t d_{0};
    uint32_t taskRows_{HC_GRAD_TASK_ROWS};
    uint32_t taskCount_{0};
    uint32_t workspaceSlots_{0};
    uint32_t splitkFactor_{0};
    uint32_t hasGate_{0};
    uint32_t g56Rows_{0};
    uint32_t gateRows_{0};
    uint32_t scheduleBeatCount_{0};
    uint32_t aicNum_{0};
    GM_ADDR workspace_{nullptr};
    GM_ADDR gmDxOutput_{nullptr};
    GM_ADDR gmWUp_{nullptr};
    GM_ADDR gmWDown_{nullptr};
    GM_ADDR gmWGate_{nullptr};
    GM_ADDR gmDwUp_{nullptr};
    GM_ADDR gmDwDown_{nullptr};
    GM_ADDR gmDwGate_{nullptr};
    uint32_t c1aTilesPerSlab_{0};
    uint32_t fullBatchRows_{0};
    ChunkState chunk_{};
    bool lowpDx_{false};
    bool singleTask_{false};
    bool fullBatch_{false};
    bool deferWeightGrads_{false};
    bool producerConsumer_{false};
    HcGradWorkspaceLayout layout_;
    HcGradCreditSync credit_;
    GemmT gemm_;
    HcGradVector<D_T, HAS_GAMMA> vector_;
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H
