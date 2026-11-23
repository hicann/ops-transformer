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
 * \file text_gated_residual_grad_vector_arch35.h
 * \brief TextGatedResidualGrad vector computation, reductions and output conversion.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_VECTOR_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_VECTOR_ARCH35_H

#include "../text_gated_residual_grad_constants.h"

#include <type_traits>

#include "kernel_operator.h"
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
#include "reg_compute/kernel_reg_compute_intf.h"
#endif
#include "text_gated_residual_grad_pipeline_arch35.h"
#include "text_gated_residual_grad_tiling_data_arch35.h"

namespace TextGatedResidualGrad {

using namespace AscendC;

// DataCopyExtParams uses an architecture-dependent stride type.
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
// A5 reads packed directly as ColumnMajor and reserves no transpose scratch.
using HcGradCopyStride = int64_t;
constexpr uint64_t HC_GRAD_FIXED_VECTOR_BYTES = HC_GRAD_SIGMOID_SCRATCH;
#else
using HcGradCopyStride = uint32_t;
constexpr uint64_t HC_GRAD_FIXED_VECTOR_BYTES = HC_GRAD_TRANSPOSE_SCRATCH + HC_GRAD_SIGMOID_SCRATCH;
#endif

template <typename D_T, int HAS_GAMMA>
class HcGradVector {
public:
    __aicore__ inline void Init(GM_ADDR gradMixed, GM_ADDR x, GM_ADDR wDown, GM_ADDR wUp, GM_ADDR invRms,
                                GM_ADDR downPreAct, GM_ADDR up, GM_ADDR gradInject, GM_ADDR wGate, GM_ADDR gamma,
                                GM_ADDR inject, GM_ADDR dx, GM_ADDR dwDown, GM_ADDR dwUp, GM_ADDR dwGate,
                                GM_ADDR dgamma, GM_ADDR workspace, const HcGradWorkspaceLayout& layout,
                                const TextGatedResidualGradTilingDataArch35* tilingData, TPipe* pipe)
    {
        gmGradMixed_ = gradMixed;
        gmGradInject_ = gradInject;
        gmX_ = x;
        gmWDown_ = wDown;
        gmWGate_ = wGate;
        gmGamma_ = gamma;
        gmInvRms_ = invRms;
        gmInject_ = inject;
        gmDownPreAct_ = downPreAct;
        gmUp_ = up;
        gmDx_ = dx;
        gmDwDownOutput_ = dwDown;
        gmDwUpOutput_ = dwUp;
        gmDwGateOutput_ = dwGate;
        gmDgamma_ = dgamma;
        workspace_ = workspace;
        layout_ = layout;
        gmDwDown_ = Pointer(layout_.dwDownAccum);
        gmDwUp_ = Pointer(layout_.dwUpAccum);
        gmDwGate_ = Pointer(layout_.dwGateAccum);
        taskRows_ = static_cast<uint32_t>(tilingData->taskRows);
        directWeights_ = tilingData->taskCount == 1 && tilingData->nnMatmul == 0;
        recomputeDirect_ = tilingData->recomputeDirect != 0;
        lowpDx_ = tilingData->lowpDx != 0;
        n_ = static_cast<uint32_t>(tilingData->n);
        c_ = static_cast<uint32_t>(tilingData->c);
        h_ = static_cast<uint32_t>(tilingData->h);
        r_ = static_cast<uint32_t>(tilingData->r);
        d_ = static_cast<uint32_t>(tilingData->d);
        taskCount_ = static_cast<uint32_t>(tilingData->taskCount);
        workspaceSlots_ = static_cast<uint32_t>(tilingData->workspaceSlots);
        splitkFactor_ = static_cast<uint32_t>(tilingData->splitkFactor);
        hasGate_ = static_cast<uint32_t>(tilingData->hasGate);
        g56Rows_ = layout_.g56Rows;
        gateRows_ = layout_.gateRows;
        InitBufferLayout(tilingData);
        aivId_ = GetBlockIdx();
        aivNum_ = static_cast<uint32_t>(tilingData->aivNum);
        InitGammaBuffers(pipe);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void SetBlockedG56Xn(bool enabled, bool prepareA)
    {
        blockedG56Xn_ = enabled;
        preparedG56A_ = prepareA;
    }
#endif

    // Concatenate w_down and w_gate along K into wConcat [R+C, D] for one DX GEMM with K=R+C.
    // One L0C accumulation and FIXP avoid an FP32 [N,D] dxGate write/read. Weights are shared by all tasks.
    __aicore__ inline void PrepareWeights()
    {
        if (hasGate_ == 0 || d_ == 0U || r_ == 0U) {
            return;
        }
        GM_ADDR dst = Pointer(layout_.wConcat);
        const uint32_t totalRows = r_ + c_;
        const uint32_t chunk = (bufCount_ < HC_GRAD_WEIGHT_COPY_CHUNK) ? bufCount_ : HC_GRAD_WEIGHT_COPY_CHUNK;
        auto buf = HalfBuf(HC_GRAD_HALF_INPUT, chunk);
        for (uint32_t row = aivId_; row < totalRows; row += aivNum_) {
            const bool isGate = row >= r_;
            GM_ADDR src = isGate ? gmWGate_ : gmWDown_;
            const uint64_t srcRow = isGate ? static_cast<uint64_t>(row - r_) : static_cast<uint64_t>(row);
            for (uint32_t off = 0; off < d_; off += chunk) {
                const uint32_t n = (d_ - off < chunk) ? (d_ - off) : chunk;
                CopyIn<D_T>(src, srcRow * d_ + off, buf, n);
                CopyOut<D_T>(dst, static_cast<uint64_t>(row) * d_ + off, buf, n);
            }
        }
    }

    __aicore__ inline void Prepare()
    {
        // V3 initializes each packed row, including gate padding, on its owning AIV.
        if (taskCount_ == 0U) {
            ZeroRangeT<D_T>(gmDwDownOutput_, static_cast<uint64_t>(r_) * d_);
            ZeroRangeT<D_T>(gmDwUpOutput_, static_cast<uint64_t>(d_) * r_);
            if (hasGate_ != 0) {
                ZeroRangeT<D_T>(gmDwGateOutput_, static_cast<uint64_t>(c_) * d_);
            }
        }
        if constexpr (HAS_GAMMA) {
            ZeroRangeT<float>(gmDgamma_, static_cast<uint64_t>(c_) * h_);
        }
    }

    __aicore__ inline void StoreWeightGradients()
    {
        CastWeightGradient(gmDwDown_, gmDwDownOutput_, static_cast<uint64_t>(r_) * d_);
        CastWeightGradient(gmDwUp_, gmDwUpOutput_, static_cast<uint64_t>(d_) * r_);
        if (hasGate_ != 0U) {
            CastWeightGradient(gmDwGate_, gmDwGateOutput_, static_cast<uint64_t>(c_) * d_);
        }
    }

    __aicore__ inline void CopyFullWeightGradients(bool accumulate = false)
    {
        // Copy or accumulate one complete chunk result into the fixed FP32 workspaces.
        CopyWeightRows(gmDwDown_, 0U, static_cast<uint64_t>(r_) * d_, accumulate);
        if (hasGate_ != 0U) {
            CopyWeightRows(gmDwGate_, layout_.dwGateRowElems, static_cast<uint64_t>(c_) * d_, accumulate);
        }
    }

    // Each core caches H gamma elements for its channel and reloads them only when the channel changes.
    __aicore__ inline void EnsureGammaCache(uint32_t channel)
    {
        if (!useGammaCache_ || gammaCachedCh_ == static_cast<int32_t>(channel)) {
            return;
        }
        const uint32_t chunk = bufCount_;
        for (uint32_t off = 0; off < h_; off += chunk) {
            const uint32_t n = (h_ - off < chunk) ? (h_ - off) : chunk;
            auto dst = ub_.GetWithOffset<float>(n, GammaOffset() + off * sizeof(float));
            CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_ + off, dst, n);
            Adds(dst, dst, 1.0f, n);
        }
        gammaCachedCh_ = static_cast<int32_t>(channel);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void SetChunk(uint32_t rowBegin, uint32_t rows, uint32_t taskCount, uint32_t parity)
    {
        const int64_t rowsMoved = static_cast<int64_t>(rowBegin) - fullBatchRowBegin_;
        const int64_t dataBytes = rowsMoved * static_cast<int64_t>(sizeof(D_T));
        gmGradMixed_ += dataBytes * h_;
        gmX_ += dataBytes * d_;
        gmUp_ += dataBytes * d_;
        gmDx_ += dataBytes * d_;
        gmInvRms_ += rowsMoved * c_ * static_cast<int64_t>(sizeof(float));
        gmDownPreAct_ += dataBytes * r_;
        if (hasGate_ != 0U) {
            gmGradInject_ += dataBytes * c_;
            gmInject_ += dataBytes * c_;
        }
        fullBatchRowBegin_ = rowBegin;
        parity_ = parity;
        n_ = rows;
        taskCount_ = taskCount;
    }

    __aicore__ inline TBuf<TPosition::VECCALC>& NnScratch()
    {
        PipeBarrier<PIPE_ALL>();
        return ub_;
    }

    __aicore__ inline void FinishNnScratch()
    {
        PipeBarrier<PIPE_ALL>();
        gammaCachedCh_ = -1;
    }

    __aicore__ inline bool TryProcessV3FullBatch()
    {
        const uint32_t blockRows = V3BlockRows();
        if (splitkFactor_ != 1U || !CanPipelineV3(blockRows * HC_GRAD_GATE_ALIGN)) {
            return false;
        }
        const uint32_t rowsPerCore = (n_ + aivNum_ - 1U) / aivNum_;
        const uint32_t begin = aivId_ * rowsPerCore;
        if (begin < n_) {
            const uint32_t end = n_ - begin < rowsPerCore ? n_ : begin + rowsPerCore;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            const uint32_t block = SelectA5V3BlockRows(blockRows, end - begin);
            if constexpr (HAS_GAMMA) {
                if (CanFuseV3Act()) {
                    ProcessV3Pipe<true, true>(0U, begin, end, block, block * HC_GRAD_GATE_ALIGN,
                                              Pointer(layout_.dactPartial),
                                              TaskPointer(layout_.packed, 0U, g56Rows_, sizeof(D_T)));
                    return true;
                }
            }
#else
            const uint32_t block = blockRows;
#endif
            ProcessV3Pipe<true>(0U, begin, end, block, block * HC_GRAD_GATE_ALIGN, Pointer(layout_.dactPartial),
                                TaskPointer(layout_.packed, 0U, g56Rows_, sizeof(D_T)));
        }
        return true;
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void ProcessV3FromC1aUb(uint32_t begin, uint32_t rows)
    {
        gammaCachedCh_ = -1;
        ProcessV3Ub(begin, begin + rows);
    }
#endif

    __aicore__ inline void ResumeDgammaAfterNn(bool hasPreviousChunk)
    {
        if constexpr (HAS_GAMMA) {
            // V1 and NN borrow UB. Each AIV restores its own sum and Kahan compensation.
            if (hasPreviousChunk) {
                TransferDgammaPartial<true>();
            } else {
                ZeroDgammaAccumulators();
            }
        }
    }

#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
    __aicore__ inline bool CanUseFullV1Slabs() const
    {
        return h_ == 2560U && c_ == 4U && r_ == 320U && aivNum_ >= c_ && aivNum_ % c_ == 0U &&
               ubBytes_ >= FullV1RequiredBytes();
    }
#endif

    __aicore__ inline bool TryProcessFullV1()
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            if (CanUseFullV1Rows()) {
                if (!CanFuseV3Act()) {
                    ProcessV1Act(0U, n_);
                }
                ProcessFullV1Rows();
                return true;
            }
            if (CanUseV1ChannelPairs()) {
                ProcessV1(0U, n_);
                return true;
            }
        }
        return false;
#else
        // This layout packs two independent nBurst rows into the UB budget.
        if (!CanUseFullV1Slabs()) {
            return false;
        }
        ProcessFullV1Act();
        ProcessFullV1Slab(0U, n_);
        return true;
#endif
    }

    __aicore__ inline void ProcessFullV1Act()
    {
        ProcessV1Act(0U, n_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessFullV1Slab(uint32_t firstRow, uint32_t rows)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        const uint32_t channel = aivId_ % c_;
        const uint32_t lanes = aivNum_ / c_;
        const uint32_t rowsPerLane = (rows + lanes - 1U) / lanes;
        const uint32_t begin = (aivId_ / c_) * rowsPerLane;
        if (begin < rows) {
            const uint32_t end = (rows - begin < rowsPerLane) ? rows : begin + rowsPerLane;
            RunFullV1Rows(firstRow + begin, firstRow + end, channel);
        }
        gammaCachedCh_ = -1;
#else
        (void)firstRow;
        (void)rows;
#endif
    }

    __aicore__ inline void ProcessFullV6()
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
            const bool supportedRowLayout = rowBlock_ == 1U;
#else
            const bool supportedRowLayout = true;
#endif
            if (c_ == 4U && h_ == 2560U && supportedRowLayout && pipeStages_ == HC_GRAD_PIPE_MAX_STAGES &&
                useGammaCache_ && kahanDgamma_ && ubBytes_ >= V6TwoRowStorageBytes() && aivId_ < aivNum_) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if (layout_.fullBatch && recomputeDirect_ && !lowpDx_) {
                    ProcessV6PackedRows();
                    return;
                }
#endif
                ProcessV6TwoRows();
                return;
            }
        }
#endif
        if (pipeStages_ != 0U && (HAS_GAMMA == 0 || useGammaCache_) && aivId_ < aivNum_) {
            const uint32_t channel = aivId_ % c_;
            const uint32_t lane = aivId_ / c_;
            const uint32_t laneCount = (aivNum_ - 1U - channel) / c_ + 1U;
            uint32_t stage = 0;
            PipePrime();
            for (uint32_t task = 0; task < taskCount_; ++task) {
                const uint32_t remaining = n_ - task * taskRows_;
                const uint32_t rows = remaining < taskRows_ ? remaining : taskRows_;
                const uint32_t rowsPerCore = (rows + laneCount - 1U) / laneCount;
                const uint32_t rowBegin = lane * rowsPerCore;
                const uint32_t rowEnd = rowBegin + rowsPerCore < rows ? rowBegin + rowsPerCore : rows;
                stage = ProcessV6Blocks(task, rowBegin, rowEnd, channel, stage);
            }
            PipeDrain();
            return;
        }
        for (uint32_t task = 0; task < taskCount_; ++task) {
            const uint32_t remaining = n_ - task * taskRows_;
            ProcessV6(task, remaining < taskRows_ ? remaining : taskRows_);
        }
    }
#endif

    __aicore__ inline void Process(HcGradVectorOp op, uint32_t task)
    {
        if (op == HcGradVectorOp::DGAMMA) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            // The token pipeline publishes its partial at the end of each V6 beat.
            if (!layout_.fullBatch) {
                return;
            }
#endif
            TransferDgammaPartial<false>();
            return;
        }
        if (op == HcGradVectorOp::REDUCE_DGAMMA) {
            if constexpr (HAS_GAMMA) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if (!layout_.fullBatch) {
                    return;
                }
                if constexpr (AscendC::Std::is_same_v<D_T, bfloat16_t>) {
                    if (TryReduceDgammaGather()) {
                        return;
                    }
                }
#endif
                ReduceDgamma();
            }
            return;
        }
        const uint32_t rowBegin = task * taskRows_;
        const uint32_t rows = n_ - rowBegin < taskRows_ ? n_ - rowBegin : taskRows_;
        switch (op) {
            case HcGradVectorOp::V1:
                ProcessV1(task, rows);
                break;
            case HcGradVectorOp::V3:
                ProcessV3(task, rows);
                break;
            case HcGradVectorOp::TRANSPOSE:
                TransposePacked(task, rows);
                break;
            case HcGradVectorOp::TRANSPOSE_DUP:
                TransposeDup(task, rows);
                break;
            case HcGradVectorOp::V6:
                ProcessV6(task, rows);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if constexpr (HAS_GAMMA) {
                    AccumulateBoundedDgamma(task, rows);
                }
#endif
                break;
            case HcGradVectorOp::REDUCE_DACT:
                ReduceDact(task, rows);
                break;
            default:
                ProcessWeightReduction(op, task);
                break;
        }
    }

private:
    __aicore__ inline void CastWeightGradient(GM_ADDR src, GM_ADDR dst, uint64_t count)
    {
        // Reuse the serial scratch after the pipeline drains; no additional UB is required.
        const uint32_t chunk = bufCount_ < HC_GRAD_WEIGHT_COPY_CHUNK ? bufCount_ : HC_GRAD_WEIGHT_COPY_CHUNK;
        auto accum = FloatBuf(HC_GRAD_SCRATCH_BASE, chunk);
        auto output = HalfBuf(HC_GRAD_HALF_INPUT, chunk);
        const uint64_t stride = static_cast<uint64_t>(aivNum_) * chunk;
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunk; offset < count; offset += stride) {
            const uint32_t current = count - offset < chunk ? static_cast<uint32_t>(count - offset) : chunk;
            CopyIn<float>(src, offset, accum, current);
            Cast(output, accum, RoundMode::CAST_RINT, current);
            CopyOut<D_T>(dst, offset, output, current);
        }
    }

    __aicore__ inline void InitBufferLayout(const TextGatedResidualGradTilingDataArch35* tilingData)
    {
        // Each AIV handles one channel. V1/V6 process rowBlock_ rows per nBurst transfer to amortize
        // GM transfer waits. Resident dgamma, Kahan compensation and gamma storage remain proportional to H.
        ubBytes_ = (tilingData->vectorUbBytes > 0) ? static_cast<uint32_t>(tilingData->vectorUbBytes) :
                                                     HC_GRAD_VECTOR_UB_FALLBACK;
        // The host computes the row block with the same UB budget and the layout selected by pipeStages.
        // Use that value directly to keep host and kernel buffer sizes consistent.
        pipeStages_ = static_cast<uint32_t>(tilingData->pipeStages);
        if (pipeStages_ > HC_GRAD_PIPE_MAX_STAGES) {
            pipeStages_ = HC_GRAD_PIPE_MAX_STAGES;
        }
        rowBlock_ = (tilingData->vectorRowBlock > 0) ? static_cast<uint32_t>(tilingData->vectorRowBlock) : 1U;
        if (rowBlock_ > HC_GRAD_MAX_ROW_BLOCK) {
            rowBlock_ = HC_GRAD_MAX_ROW_BLOCK;
        }
        {
            const uint32_t blockElems = rowBlock_ * h_ > r_ ? rowBlock_ * h_ : r_;
            // bufCount_ is a multiple of 16 so both FP32 and 16-bit buffer strides are aligned to 32 bytes.
            bufCount_ = (blockElems + HC_GRAD_VECTOR_ALIGN_MASK) & ~HC_GRAD_VECTOR_ALIGN_MASK;
        }
        // Keep the pool size consistent with PoolBytes(rowBlock) in host tiling.
        if (pipeStages_ == 0U) {
            poolBytes_ = HC_GRAD_SERIAL_POOL_BYTES * bufCount_;
        } else {
            const uint32_t v1 = HC_GRAD_V1_POOL_BYTES * bufCount_;
            uint32_t v6 = HC_GRAD_V6_POOL_BYTES * bufCount_ + HC_GRAD_V6_ROW_BYTES * h_;
            if (recomputeDirect_) {
                v6 += HC_GRAD_V6_DIRECT_RECOMP_EXTRA_BYTES * bufCount_;
            }
            poolBytes_ = v1 > v6 ? v1 : v6;
        }
        // Each stage stores rowBlock*C FP32 inv_rms values in a 32-byte-aligned segment.
        invStride_ = ((rowBlock_ * c_ * static_cast<uint32_t>(sizeof(float))) + HC_GRAD_DATA_BLOCK_MASK) &
                     ~HC_GRAD_DATA_BLOCK_MASK;
        scalarBytes_ = HC_GRAD_SCALAR_SLOTS * rowBlock_ * HC_GRAD_DATA_BLOCK_BYTES;
    }

    __aicore__ inline void InitGammaBuffers(TPipe* pipe)
    {
        const bool cacheWanted = (HAS_GAMMA != 0) && taskCount_ >= HC_GRAD_GAMMA_CACHE_MIN_TASKS;
        const uint64_t ubBytes = static_cast<uint64_t>(ubBytes_);
        dgammaBytes_ = HAS_GAMMA ? h_ * sizeof(float) : 0U;
        // Resident Kahan compensation follows the dgamma accumulator; other buffers follow both.
        // Each core stores H values for its channel: accumulator, compensation and gamma cache use 12 bytes per
        // element.
        kahanDgamma_ = false;
        useGammaCache_ = false;
        if constexpr (HAS_GAMMA) {
            if (poolBytes_ + static_cast<uint64_t>(HC_GRAD_GAMMA_RESIDENT_BYTES) * h_ + HC_GRAD_FIXED_VECTOR_BYTES +
                    scalarBytes_ <=
                ubBytes) {
                kahanDgamma_ = true;
                useGammaCache_ = cacheWanted;
            } else if (poolBytes_ + static_cast<uint64_t>(HC_GRAD_KAHAN_RESIDENT_BYTES) * h_ +
                           HC_GRAD_FIXED_VECTOR_BYTES + scalarBytes_ <=
                       ubBytes) {
                kahanDgamma_ = true;
            } else if (poolBytes_ + static_cast<uint64_t>(sizeof(float)) * h_ + HC_GRAD_FIXED_VECTOR_BYTES +
                           scalarBytes_ <=
                       ubBytes) {
                useGammaCache_ = cacheWanted;
            }
        }
        ubBufBase_ = dgammaBytes_ * (kahanDgamma_ ? HC_GRAD_DGAMMA_SLOTS : 1U);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        // Two xn/dup pairs are the larger of the V1 and V6 output allocations.
        const uint64_t outputBytes = 2U * HC_GRAD_PIPE_MAX_STAGES * sizeof(D_T) * bufCount_;
        independentPipeOutput_ =
            pipeStages_ != 0U && IndependentPipeOutputOffset() + outputBytes + HC_GRAD_TRANSPOSE_SCRATCH <= ubBytes;
#endif
        pipe->InitBuffer(ub_, ubBytes_);
        ZeroDgammaAccumulators();
    }

    __aicore__ inline void ZeroDgammaAccumulators()
    {
        if constexpr (HAS_GAMMA) {
            for (uint32_t off = 0; off < h_; off += HC_GRAD_DGAMMA_COPY_CHUNK) {
                const uint32_t n = h_ - off < HC_GRAD_DGAMMA_COPY_CHUNK ? h_ - off : HC_GRAD_DGAMMA_COPY_CHUNK;
                auto zero = ub_.GetWithOffset<float>(n, off * sizeof(float));
                Duplicate(zero, 0.0f, n);
            }
            if (kahanDgamma_) {
                for (uint32_t off = 0; off < h_; off += HC_GRAD_DGAMMA_COPY_CHUNK) {
                    const uint32_t n = h_ - off < HC_GRAD_DGAMMA_COPY_CHUNK ? h_ - off : HC_GRAD_DGAMMA_COPY_CHUNK;
                    auto zero = ub_.GetWithOffset<float>(n, dgammaBytes_ + off * sizeof(float));
                    Duplicate(zero, 0.0f, n);
                }
            }
        }
    }

    template <bool LOAD>
    __aicore__ inline void TransferDgammaPartial()
    {
        if constexpr (HAS_GAMMA) {
            GM_ADDR partial = Pointer(layout_.dgammaPartial) +
                              static_cast<uint64_t>(aivId_) * HC_GRAD_DGAMMA_SLOTS * h_ * sizeof(float);
            const uint32_t count = ubBufBase_ / sizeof(float);
            auto resident = ub_.GetWithOffset<float>(count, 0U);
            if constexpr (LOAD) {
                CopyIn<float>(partial, 0U, resident, count);
            } else {
                CopyOut<float>(partial, 0U, resident, count);
            }
        }
    }

    __aicore__ inline void ProcessWeightReduction(HcGradVectorOp op, uint32_t task)
    {
        switch (op) {
            case HcGradVectorOp::REDUCE_WU2:
                if (!directWeights_) {
                    ReducePartialAtomic2(layout_.dwPartial, gmDwUp_, task, task + 1U, layout_.dwSlotElems, 0,
                                         static_cast<uint64_t>(r_) * d_);
                }
                break;
            case HcGradVectorOp::REDUCE_WD2:
                if (!directWeights_) {
                    ReducePartialAtomic2(layout_.dwPartial, gmDwDown_, task, task + 1U, layout_.dwSlotElems, 0,
                                         static_cast<uint64_t>(r_) * d_);
                }
                if (hasGate_ != 0) {
                    ReducePartialFloat2(layout_.dwPartial, gmDwGate_, task, task + 1U, layout_.dwSlotElems,
                                        static_cast<uint64_t>(c_) * d_, layout_.dwGateRowElems);
                }
                break;
            case HcGradVectorOp::REDUCE_WU:
                if (!directWeights_) {
                    ReducePartialAtomic(layout_.dwPartial, gmDwUp_, task, layout_.dwSlotElems, 0,
                                        static_cast<uint64_t>(r_) * d_);
                }
                break;
            case HcGradVectorOp::REDUCE_WD:
                // A single task can write dwDown directly. Otherwise reduce each ring slot's
                // [R+gateRows,D] partial into the fixed FP32 output accumulators.
                if (!directWeights_) {
                    ReducePartialAtomic(layout_.dwPartial, gmDwDown_, task, layout_.dwSlotElems, 0,
                                        static_cast<uint64_t>(r_) * d_);
                }
                if (hasGate_ != 0) {
                    ReducePartialFloat(layout_.dwPartial, gmDwGate_, task, layout_.dwSlotElems,
                                       static_cast<uint64_t>(c_) * d_, layout_.dwGateRowElems);
                }
                break;
            default:
                break;
        }
    }

    __aicore__ inline GM_ADDR Pointer(uint64_t offset) const
    {
        return workspace_ + offset;
    }

    __aicore__ inline GM_ADDR TaskPointer(uint64_t offset, uint32_t task, uint32_t rowStride,
                                          uint32_t elementSize) const
    {
        return Pointer(layout_.TaskBufferOffset(offset, task, workspaceSlots_,
                                                static_cast<uint64_t>(taskRows_) * rowStride * elementSize, parity_));
    }

    __aicore__ inline uint32_t DenseDxColumns() const
    {
        return r_ + (hasGate_ != 0U ? c_ : 0U);
    }

    __aicore__ inline GM_ADDR DenseDxTaskPointer(uint32_t task) const
    {
        return TaskPointer(layout_.denseDx, task, DenseDxColumns(), sizeof(D_T));
    }

    __aicore__ inline GM_ADDR Dx2TaskPointer(uint32_t task) const
    {
        if (lowpDx_) {
            // BF16 DX is produced directly into the public output buffer.
            return gmDx_ + static_cast<uint64_t>(task) * taskRows_ * d_ * sizeof(D_T);
        }
        return TaskPointer(layout_.dx2, task, d_, sizeof(float));
    }

    __aicore__ inline LocalTensor<float> DgammaComp(uint32_t count, uint32_t offsetBytes)
    {
        return ub_.GetWithOffset<float>(count, dgammaBytes_ + offsetBytes);
    }

    // Accumulate per-core dgamma with resident Kahan compensation, or ordinary Add when disabled.
    __aicore__ inline void DgammaAccumAdd(const LocalTensor<float>& value, uint32_t count, uint32_t offsetBytes,
                                          const LocalTensor<float>& tmp, const LocalTensor<float>& tmp2)
    {
        auto accum = ub_.GetWithOffset<float>(count, offsetBytes);
        if (!kahanDgamma_) {
            Add(accum, accum, value, count);
            return;
        }
        auto comp = DgammaComp(count, offsetBytes);
        Sub(tmp, value, comp, count);
        Add(tmp2, accum, tmp, count);
        Sub(comp, tmp2, accum, count);
        Sub(comp, comp, tmp, count);
        Adds(accum, tmp2, 0.0f, count);
    }

    __aicore__ inline LocalTensor<float> FloatBuf(uint32_t index, uint32_t count)
    {
        // DataCopyPad rounds a short GM->UB transfer up to a 32-byte block.
        // Each pool slot already reserves bufCount_ aligned elements; expose that
        // padded capacity in the view without changing the logical compute count.
        const uint32_t dmaCount =
            ((count * sizeof(float) + HC_GRAD_DATA_BLOCK_MASK) & ~HC_GRAD_DATA_BLOCK_MASK) / sizeof(float);
        return ub_.GetWithOffset<float>(dmaCount, ubBufBase_ + index * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<D_T> HalfBuf(uint32_t index, uint32_t count)
    {
        const uint32_t dmaCount =
            ((count * sizeof(D_T) + HC_GRAD_DATA_BLOCK_MASK) & ~HC_GRAD_DATA_BLOCK_MASK) / sizeof(D_T);
        return ub_.GetWithOffset<D_T>(dmaCount, ubBufBase_ + HC_GRAD_SERIAL_FLOAT_SLOTS * bufCount_ * sizeof(float) +
                                                    index * bufCount_ * sizeof(D_T));
    }

    __aicore__ inline uint32_t GammaOffset() const
    {
        return ubBufBase_ + poolBytes_;
    }

    __aicore__ inline uint32_t SigmoidOffset() const
    {
        return GammaOffset() + (useGammaCache_ ? h_ * sizeof(float) : 0U);
    }

    // The scalar area stores row sums followed by centered residual sums in 32-byte-aligned slots.
    // inv_rms uses the contiguous InvOffset region. V_S synchronizes scalar reads after each reduction.
    __aicore__ inline uint32_t ScalarOffset() const
    {
        return SigmoidOffset() + HC_GRAD_SIGMOID_SCRATCH;
    }

    // A block of n inv_rms rows contains n*C contiguous FP32 values in GM. Each pipeline stage has
    // a separate 32-byte-aligned segment immediately after the scalar area.
    __aicore__ inline uint32_t InvOffset(uint32_t stage = 0U) const
    {
        return ScalarOffset() + scalarBytes_ + stage * invStride_;
    }

    __aicore__ inline uint32_t IndependentPipeOutputOffset() const
    {
        return InvOffset(HC_GRAD_PIPE_MAX_STAGES);
    }

    __aicore__ inline LocalTensor<float> SumSlot(uint32_t row)
    {
        return ub_.GetWithOffset<float>(1, ScalarOffset() + row * HC_GRAD_DATA_BLOCK_BYTES);
    }

    __aicore__ inline LocalTensor<float> ResidualSlot(uint32_t row)
    {
        return ub_.GetWithOffset<float>(1, ScalarOffset() + (rowBlock_ + row) * HC_GRAD_DATA_BLOCK_BYTES);
    }

    __aicore__ inline void ReduceRmsProjection(LocalTensor<float> result, LocalTensor<float> product,
                                               LocalTensor<float> scratch, uint32_t width)
    {
        if (width == 2560U) {
            // Fold both aligned halves into the existing scratch. Leave the products
            // available for the centered pass and use disjoint scratch for ReduceSum.
            constexpr uint32_t HALF = 1280U;
            Add(scratch, product, product[HALF], HALF);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceSum(result, scratch, scratch[HALF], HALF);
        } else {
            ReduceSum(result, product, scratch, width);
        }
    }

    // Keep the mean and centered residual separate to avoid rounding a small correction back into one FP32 mean.
    // For non-finite means, retain the original projection without centering Inf/NaN values.
    __aicore__ inline void ReduceRmsResidual(LocalTensor<float> product, LocalTensor<float> scratch, uint32_t row)
    {
        const float mean = SumSlot(row).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
        if (mean <= HC_GRAD_MAX_FINITE_FLOAT && mean >= -HC_GRAD_MAX_FINITE_FLOAT) {
            Adds(product, product, -mean, h_);
            ReduceRmsProjection(ResidualSlot(row), product, scratch, h_);
        } else {
            Duplicate(ResidualSlot(row), 0.0f, 1U);
        }
    }

    // V1/V6 alternate two input stages. With separate output slots, only Vector waits
    // for MTE3 before overwriting an output. Otherwise MTE2 waits before reusing the aliased slot.
    // MTE2_S is stage-specific so scalar waits do not include the next stage.
    // Events start at HC_GRAD_PIPE_EVENT_BASE; event zero belongs to the serial path.
    // Pipeline prologue/epilogue set and drain all flags used across iterations.
    __aicore__ inline int32_t PipeInEvent(uint32_t stage) const
    {
        return HC_GRAD_PIPE_EVENT_BASE + static_cast<int32_t>(stage);
    }

    __aicore__ inline int32_t PipeOutEvent(uint32_t stage) const
    {
        // HardEvent includes the pipe type, so different event kinds in one stage can share this ID
        // without allocating additional event resources.
        return PipeInEvent(stage);
    }

    // Use a separate MTE2-to-Scalar event per stage. Sharing event 0 would make current computation
    // wait for the next block's prefetch as well, serializing the pipeline at this scalar handshake.
    __aicore__ inline int32_t PipeScalarEvent(uint32_t stage) const
    {
        return PipeInEvent(stage);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr uint32_t V1_EXTRA_STAGE = HC_GRAD_PIPE_MAX_STAGES;

    __aicore__ inline uint32_t V1ExtraInvOffset() const
    {
        return IndependentPipeOutputOffset() + HC_GRAD_V1_INPUT_SLOTS * bufCount_ * sizeof(D_T);
    }

    __aicore__ inline bool CanUseV1ThreeSlots() const
    {
        return pipeStages_ == HC_GRAD_PIPE_MAX_STAGES && !independentPipeOutput_ && useGammaCache_ && rowBlock_ == 2U &&
               c_ == 4U && h_ == 2560U && r_ == 320U && V1ExtraInvOffset() + invStride_ <= ubBytes_;
    }

    __aicore__ inline uint32_t V1InvOffset(uint32_t stage) const
    {
        if (stage == V1_EXTRA_STAGE) {
            return V1ExtraInvOffset();
        }
        return InvOffset(stage);
    }
#endif

    // V1 pool offsets.
    __aicore__ inline LocalTensor<float> V1Scratch(uint32_t index, uint32_t count)
    {
        // Block-sized FP32 slots: 0 = x to xn, 1 = gate, 2 = temp.
        // DataCopyPad rounds a short GM->UB transfer up to a 32-byte block.
        // Each pool slot already reserves bufCount_ aligned elements; expose that
        // padded capacity in the view without changing the logical compute count.
        const uint32_t dmaCount =
            ((count * sizeof(float) + HC_GRAD_DATA_BLOCK_MASK) & ~HC_GRAD_DATA_BLOCK_MASK) / sizeof(float);
        return ub_.GetWithOffset<float>(dmaCount, ubBufBase_ + index * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<float> V1DirectOut(uint32_t stage, uint32_t count)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (stage == V1_EXTRA_STAGE) {
            // In the three-slot path, grad_mixed goes directly to directOut and dup reuses x.
            return V1Scratch(HC_GRAD_V1_PIPE_TEMP, count);
        }
#endif
        return ub_.GetWithOffset<float>(count,
                                        ubBufBase_ + (HC_GRAD_V1_SCRATCH_SLOTS + stage) * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<D_T> V1InHalf(uint32_t stage, uint32_t index, uint32_t count)
    {
        // Three input slots per stage: x, up and grad_mixed.
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (stage == V1_EXTRA_STAGE) {
            return ub_.GetWithOffset<D_T>(count, IndependentPipeOutputOffset() + index * bufCount_ * sizeof(D_T));
        }
#endif
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + HC_GRAD_V1_FLOAT_SLOTS * bufCount_ * sizeof(float) +
                                                 (stage * HC_GRAD_V1_INPUT_SLOTS + index) * bufCount_ * sizeof(D_T));
    }

    __aicore__ inline LocalTensor<D_T> V1OutHalf(uint32_t stage, uint32_t index, uint32_t count)
    {
        if (independentPipeOutput_) {
            const uint32_t offset = IndependentPipeOutputOffset() + (2U * stage + index) * bufCount_ * sizeof(D_T);
            return ub_.GetWithOffset<D_T>(count, offset);
        }
        return V1InHalf(stage, index, count);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr uint32_t V6_EXTRA_STAGE = HC_GRAD_PIPE_MAX_STAGES;

    __aicore__ inline uint32_t V6ExtraHalfOffset() const
    {
        return IndependentPipeOutputOffset() + HC_GRAD_V6_INPUT_SLOTS * bufCount_ * sizeof(float);
    }

    __aicore__ inline uint32_t V6ExtraInvOffset() const
    {
        return V6ExtraHalfOffset() + bufCount_ * sizeof(D_T);
    }

    __aicore__ inline bool CanUseV6ThreeSlots() const
    {
        return HAS_GAMMA && useGammaCache_ && !independentPipeOutput_ && !recomputeDirect_ && !lowpDx_ &&
               pipeStages_ == HC_GRAD_PIPE_MAX_STAGES && rowBlock_ == 2U && c_ == 4U && h_ == 2560U && r_ == 320U &&
               static_cast<uint64_t>(V6ExtraInvOffset()) + invStride_ <= ubBytes_;
    }

    __aicore__ inline uint32_t V6InvOffset(uint32_t stage) const
    {
        if (stage == V6_EXTRA_STAGE) {
            return V6ExtraInvOffset();
        }
        return InvOffset(stage);
    }
#endif

    // V6 pool offsets.
    __aicore__ inline LocalTensor<float> V6InFloat(uint32_t stage, uint32_t index, uint32_t count)
    {
        // Two FP32 input slots per stage: direct and dx2.
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (stage == V6_EXTRA_STAGE) {
            return ub_.GetWithOffset<float>(count, IndependentPipeOutputOffset() + index * bufCount_ * sizeof(float));
        }
#endif
        return ub_.GetWithOffset<float>(
            count, ubBufBase_ + (stage * HC_GRAD_V6_INPUT_SLOTS + index) * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<float> V6X(uint32_t count)
    {
        return ub_.GetWithOffset<float>(count, ubBufBase_ + HC_GRAD_V6_X_SLOT * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<float> V6Row(uint32_t index, uint32_t count)
    {
        // Four row-sized slots: gn, product, pending dgamma and reduceWork.
        return ub_.GetWithOffset<float>(
            count, ubBufBase_ + HC_GRAD_V6_FLOAT_SLOTS * bufCount_ * sizeof(float) + index * h_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<D_T> V6InHalf(uint32_t stage, uint32_t count)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (stage == V6_EXTRA_STAGE) {
            return ub_.GetWithOffset<D_T>(count, V6ExtraHalfOffset());
        }
#endif
        const uint32_t base =
            HC_GRAD_V6_FLOAT_SLOTS * bufCount_ * sizeof(float) + HC_GRAD_V6_ROW_SLOTS * h_ * sizeof(float);
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + base + stage * bufCount_ * sizeof(D_T));
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline LocalTensor<D_T> V6DirectHalf(uint32_t stage, uint32_t index, uint32_t count)
    {
        const uint32_t base =
            HC_GRAD_V6_FLOAT_SLOTS * bufCount_ * sizeof(float) + HC_GRAD_V6_ROW_SLOTS * h_ * sizeof(float);
        const uint32_t slot = HC_GRAD_PIPE_MAX_STAGES + stage * 2U + index;
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + base + slot * bufCount_ * sizeof(D_T));
    }
#endif

    __aicore__ inline LocalTensor<D_T> V6OutHalf(uint32_t stage, uint32_t count)
    {
        if (independentPipeOutput_) {
            const uint32_t offset = IndependentPipeOutputOffset() + stage * bufCount_ * sizeof(D_T);
            return ub_.GetWithOffset<D_T>(count, offset);
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        // dx reuses this stage's x input slot after Cast consumes xHalf.
        return V6InHalf(stage, count);
#else
        const uint32_t base =
            HC_GRAD_V6_FLOAT_SLOTS * bufCount_ * sizeof(float) + HC_GRAD_V6_ROW_SLOTS * h_ * sizeof(float);
        // dx reuses this stage's x input slot after Cast consumes xHalf, avoiding another UB output buffer.
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + base + stage * bufCount_ * sizeof(D_T));
#endif
    }

    __aicore__ inline void SetPipeOutputReady(uint32_t stage)
    {
        if (independentPipeOutput_) {
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        } else {
            SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        }
    }

    // Prime the reuse events for each stage before its first operation.
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void PipePrime(uint32_t stages = HC_GRAD_PIPE_MAX_STAGES)
    {
        for (uint32_t s = 0U; s < stages; ++s) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(s));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(s));
        }
        for (uint32_t s = 0U; s < stages; ++s) {
            SetPipeOutputReady(s);
        }
    }

    __aicore__ inline void PipeDrain(uint32_t stages = HC_GRAD_PIPE_MAX_STAGES)
    {
        for (uint32_t s = 0U; s < stages; ++s) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(s));
            WaitFlag<HardEvent::S_MTE2>(PipeInEvent(s));
        }
        for (uint32_t s = 0U; s < stages; ++s) {
            if (independentPipeOutput_) {
                WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(s));
            } else {
                WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(s));
            }
        }
        // The next operation may reuse the pool for vector writes, such as clearing pending dgamma in V6.
        // MTE3_MTE2 protects subsequent input DMA only; it does not protect vector writes from outstanding MTE3 reads.
        SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
    }
#else
    __aicore__ inline void PipePrime()
    {
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(s));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(s));
        }
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            SetPipeOutputReady(s);
        }
    }

    __aicore__ inline void PipeDrain()
    {
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(s));
            WaitFlag<HardEvent::S_MTE2>(PipeInEvent(s));
        }
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            if (independentPipeOutput_) {
                WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(s));
            } else {
                WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(s));
            }
        }
        // The next operation may reuse the pool for vector writes, such as clearing pending dgamma in V6.
        // MTE3_MTE2 protects subsequent input DMA only; it does not protect vector writes from outstanding MTE3 reads.
        SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
    }
#endif

    // DMA helpers issue no flags; the pipeline controls synchronization.
    template <typename T, bool STREAM = false>
    __aicore__ inline void IssueCopyInRows(GM_ADDR src, uint64_t srcRow0ElementOffset, uint32_t rowStride,
                                           const LocalTensor<T>& dst, uint32_t n, uint32_t count)
    {
        if (n == 0U || count == 0U) {
            return;
        }
        GlobalTensor<T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        if constexpr (STREAM) {
            source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyExtParams params{static_cast<uint16_t>(n), static_cast<uint32_t>(count * sizeof(T)),
                                 static_cast<HcGradCopyStride>((rowStride - count) * sizeof(T)), 0, 0};
        DataCopyPad(dst, source[srcRow0ElementOffset], params, pad);
    }

    template <typename T, bool STREAM = false>
    __aicore__ inline void IssueCopyIn(GM_ADDR src, uint64_t srcElementOffset, const LocalTensor<T>& dst,
                                       uint32_t count)
    {
        if (count == 0U) {
            return;
        }
        GlobalTensor<T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        if constexpr (STREAM) {
            source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, source[srcElementOffset], params, pad);
    }

    template <typename T, bool STREAM = false>
    __aicore__ inline void IssueCopyOutRows(GM_ADDR dst, uint64_t dstRow0ElementOffset, uint32_t rowStride,
                                            const LocalTensor<T>& src, uint32_t n, uint32_t count)
    {
        if (n == 0U || count == 0U) {
            return;
        }
        GlobalTensor<T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
        if constexpr (STREAM) {
            target.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        DataCopyExtParams params{static_cast<uint16_t>(n), static_cast<uint32_t>(count * sizeof(T)), 0,
                                 static_cast<HcGradCopyStride>((rowStride - count) * sizeof(T)), 0};
        DataCopyPad(target[dstRow0ElementOffset], src, params);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void IssueBlockedG56XnRows(uint32_t task, uint32_t row, uint32_t col, const LocalTensor<D_T>& src,
                                                 uint32_t rows, uint32_t cols)
    {
        constexpr uint32_t TILE = HC_GRAD_G56_BLOCKED_XN_N;
        constexpr uint32_t PITCH = HC_GRAD_CHUNK_ROWS;
        constexpr uint32_t TILE_BLOCKS = TILE * sizeof(D_T) / HC_GRAD_DATA_BLOCK_BYTES;
        static_assert(TILE * sizeof(D_T) % HC_GRAD_DATA_BLOCK_BYTES == 0U);
        static_assert((PITCH - 1U) * TILE_BLOCKS <= 65535U);
        const uint64_t firstRow = static_cast<uint64_t>(task) * taskRows_ + row;
        const uint32_t endCol = col + cols;
        const uint32_t aligned = (col + TILE - 1U) / TILE * TILE;
        const uint32_t firstFullCol = aligned < endCol ? aligned : endCol;
        const uint32_t prefix = firstFullCol - col;
        const uint32_t fullTiles = (endCol - firstFullCol) / TILE;
        const uint32_t suffixCol = firstFullCol + fullTiles * TILE;
        const uint32_t suffix = endCol - suffixCol;
        const uint32_t prefixTileCol = col / TILE * TILE;
        const uint32_t prefixTileN = d_ - prefixTileCol < TILE ? d_ - prefixTileCol : TILE;
        const uint32_t suffixTileN = d_ - suffixCol < TILE ? d_ - suffixCol : TILE;
        GlobalTensor<D_T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(TaskPointer(layout_.xn, 0U, d_, sizeof(D_T))),
                               static_cast<uint64_t>(PITCH) * d_);
        target.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        // The final 64-column panel has a 64-element row stride, so there is
        // no padding or extra workspace. Adjacent channel pairs can share a
        // panel but own disjoint, 32B-aligned intervals within each row.
        const DataCopyParams full{static_cast<uint16_t>(fullTiles), static_cast<uint16_t>(TILE_BLOCKS), 0U,
                                  static_cast<uint16_t>((PITCH - 1U) * TILE_BLOCKS)};
        const DataCopyParams first{1U, static_cast<uint16_t>(prefix * sizeof(D_T) / HC_GRAD_DATA_BLOCK_BYTES), 0U, 0U};
        const DataCopyParams last{1U, static_cast<uint16_t>(suffix * sizeof(D_T) / HC_GRAD_DATA_BLOCK_BYTES), 0U, 0U};
        for (uint32_t r = 0U; r < rows; ++r) {
            const uint64_t token = firstRow + r;
            const uint32_t in = r * cols;
            if (prefix != 0U) {
                DataCopy(
                    target[static_cast<uint64_t>(prefixTileCol) * PITCH + token * prefixTileN + col - prefixTileCol],
                    src[in], first);
            }
            if (fullTiles != 0U) {
                DataCopy(target[static_cast<uint64_t>(firstFullCol) * PITCH + token * TILE], src[in + prefix], full);
            }
            if (suffix != 0U) {
                DataCopy(target[static_cast<uint64_t>(suffixCol) * PITCH + token * suffixTileN],
                         src[in + prefix + fullTiles * TILE], last);
            }
        }
    }

    __aicore__ inline bool TryIssueBlockedG56XnRows(uint32_t task, uint32_t row, uint32_t col,
                                                    const LocalTensor<D_T>& src, uint32_t rows, uint32_t cols)
    {
        if (blockedG56Xn_) {
            IssueBlockedG56XnRows(task, row, col, src, rows, cols);
            return true;
        }
        return false;
    }
#endif

    // For power-of-two C, multiply by its exact reciprocal; otherwise use Div with the second sigmoid scratch buffer.
    // The latter requires 512 FP32 scratch elements.
    __aicore__ inline void ScaleByInvC(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
    {
        // Convert uint32_t C through int32_t: A2 does not support a direct unsigned-to-float cast here.
        const float divisor = static_cast<float>(static_cast<int32_t>(c_));
        if ((c_ & (c_ - 1U)) == 0U) {
            Muls(dst, src, 1.0f / divisor, count);
            return;
        }
        auto denom =
            ub_.GetWithOffset<float>(HC_GRAD_SIGMOID_CHUNK, SigmoidOffset() + HC_GRAD_SIGMOID_CHUNK * sizeof(float));
        Duplicate(denom, divisor, HC_GRAD_SIGMOID_CHUNK);
        LocalTensor<float> dstH = dst;
        LocalTensor<float> srcH = src;
        for (uint32_t off = 0; off < count; off += HC_GRAD_SIGMOID_CHUNK) {
            const uint32_t n = (count - off < HC_GRAD_SIGMOID_CHUNK) ? (count - off) : HC_GRAD_SIGMOID_CHUNK;
            Div(dstH[off], srcH[off], denom, static_cast<int32_t>(n));
        }
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Scratch(uint32_t index, uint32_t count)
    {
        if constexpr (std::is_same_v<T, float>) {
            return FloatBuf(index, count);
        } else {
            return HalfBuf(index, count);
        }
    }

    // Serial V6 float slots 0/1/2 map to FloatBuf 0/3/4; its half input uses slot 0 and ignores stage.
    // The pipeline instead uses V6InFloat/V6InHalf for stage-specific offsets.
    __aicore__ inline LocalTensor<float> InFloat(uint32_t stage, uint32_t index, uint32_t count)
    {
        (void)stage;
        const uint32_t slot =
            (index == HC_GRAD_V6_DIRECT_INPUT) ?
                HC_GRAD_V6_SERIAL_TOTAL :
                ((index == HC_GRAD_V6_DX2_INPUT) ? HC_GRAD_V6_SERIAL_PRODUCT : HC_GRAD_V6_SERIAL_CORRECTION);
        return ub_.GetWithOffset<float>(count, ubBufBase_ + slot * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<D_T> InHalf(uint32_t stage, uint32_t count)
    {
        (void)stage;
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + HC_GRAD_SERIAL_FLOAT_SLOTS * bufCount_ * sizeof(float));
    }

    template <typename T>
    __aicore__ inline void CopyInRows(GM_ADDR src, uint64_t srcRow0ElementOffset, uint32_t rowStride,
                                      LocalTensor<T>& dst, uint32_t n, uint32_t count, bool waitAfter = true)
    {
        if (n == 0U || count == 0U) {
            return;
        }
        GlobalTensor<T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        SetFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        // Shared multi-burst DMA helpers: DataCopyPad uses byte strides for the source,
        // while the destination is contiguous across rows.
        DataCopyExtParams params{static_cast<uint16_t>(n), static_cast<uint32_t>(count * sizeof(T)),
                                 static_cast<HcGradCopyStride>((rowStride - count) * sizeof(T)), 0, 0};
        DataCopyPad(dst, source[srcRow0ElementOffset], params, pad);
        if (waitAfter) {
            SetFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
        }
    }

    template <typename T>
    __aicore__ inline void CopyOutRows(GM_ADDR dst, uint64_t dstRow0ElementOffset, uint32_t rowStride,
                                       const LocalTensor<T>& src, uint32_t n, uint32_t count)
    {
        if (n == 0U || count == 0U) {
            return;
        }
        GlobalTensor<T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
        SetFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
        DataCopyExtParams params{static_cast<uint16_t>(n), static_cast<uint32_t>(count * sizeof(T)), 0,
                                 static_cast<HcGradCopyStride>((rowStride - count) * sizeof(T)), 0};
        DataCopyPad(target[dstRow0ElementOffset], src, params);
        SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
    }

    template <typename T>
    __aicore__ inline void CopyIn(GM_ADDR src, uint64_t srcElementOffset, LocalTensor<T>& dst, uint32_t count,
                                  bool waitAfter = true)
    {
        if (count == 0) {
            return;
        }
        GlobalTensor<T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        // Workspace producers and cross-core consumers require L2 coherence; bypassing L2 can expose stale data.
        // Reads that bypass L2 may see old data before writeback.
        SetFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
        // Use the 32-bit blockLen in DataCopyExtParams to avoid truncating transfers larger than 64 KiB.
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, source[srcElementOffset], params, pad);
        if (waitAfter) {
            SetFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
        }
    }

    // Issue all DMA transfers for the block before one shared wait.
    __aicore__ inline void SyncCopy()
    {
        SetFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
    }

    template <typename T, bool ATOMIC_ADD = false>
    __aicore__ inline void CopyOut(GM_ADDR dst, uint64_t dstElementOffset, const LocalTensor<T>& src, uint32_t count)
    {
        if (count == 0) {
            return;
        }
        GlobalTensor<T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
        SetFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        if constexpr (ATOMIC_ADD) {
            SetAtomicAdd<float>();
            PipeBarrier<PIPE_MTE3>();
        }
        DataCopyPad(target[dstElementOffset], src, params);
        SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        if constexpr (ATOMIC_ADD) {
            DisableDmaAtomic();
        }
    }

    template <typename T>
    __aicore__ inline void ZeroRangeT(GM_ADDR dst, uint64_t count)
    {
        // Zero at most min(H, bufCount_, 8192) elements per block.
        uint32_t chunk = h_ < bufCount_ ? h_ : bufCount_;
        if (chunk > HC_GRAD_ZERO_CHUNK) {
            chunk = HC_GRAD_ZERO_CHUNK;
        }
        const uint64_t stride = static_cast<uint64_t>(aivNum_) * chunk;
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunk; offset < count; offset += stride) {
            const uint32_t current = count - offset < chunk ? static_cast<uint32_t>(count - offset) : chunk;
            auto zero = Scratch<T>(HC_GRAD_SCRATCH_BASE, current);
            Duplicate(zero, static_cast<T>(0), current);
            CopyOut<T>(dst, offset, zero, current);
        }
    }

    // Hardware Exp and Div implement sigmoid; bitwise equality with CPU results is not guaranteed.
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void SigmoidFp32(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count,
                                       uint32_t scratchOffset = 0U)
    {
        const uint32_t base = scratchOffset != 0U ? scratchOffset : SigmoidOffset();
#else
    __aicore__ inline void SigmoidFp32(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
    {
        const uint32_t base = SigmoidOffset();
#endif
        auto ones = ub_.GetWithOffset<float>(HC_GRAD_SIGMOID_CHUNK, base);
        auto denom = ub_.GetWithOffset<float>(HC_GRAD_SIGMOID_CHUNK, base + HC_GRAD_SIGMOID_CHUNK * sizeof(float));
        LocalTensor<float> dstH = dst;
        LocalTensor<float> srcH = src;
        Duplicate(ones, 1.0f, HC_GRAD_SIGMOID_CHUNK);
        for (uint32_t off = 0; off < count; off += HC_GRAD_SIGMOID_CHUNK) {
            const uint32_t n = (count - off < HC_GRAD_SIGMOID_CHUNK) ? (count - off) : HC_GRAD_SIGMOID_CHUNK;
            Muls(denom, srcH[off], -1.0f, n);
            Exp(denom, denom, n);
            Adds(denom, denom, 1.0f, n);
            Div(dstH[off], ones, denom, static_cast<int32_t>(n));
        }
    }

    // A2 Transpose supports half/int16/uint16, but not bfloat16 directly. Reinterpret the same 16-bit
    // data as uint16_t for a bit-preserving transpose on both architectures.
    template <typename T>
    __aicore__ inline void Transpose16(const LocalTensor<T>& dst, const LocalTensor<T>& src)
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
            Transpose(dst, src);
        } else {
            Transpose(dst.template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>());
        }
    }

    __aicore__ inline void NormalizeV1Rows(uint32_t n, uint32_t channel, LocalTensor<float> x,
                                           LocalTensor<float> xnFloat, LocalTensor<float> invBuf)
    {
        LocalTensor<float> gammaCache;
        if constexpr (HAS_GAMMA) {
            EnsureGammaCache(channel);
            if (useGammaCache_) {
                gammaCache = ub_.GetWithOffset<float>(h_, GammaOffset());
            } else {
                // Reuse temp for gamma before temp holds computed values.
                auto gamma = FloatBuf(HC_GRAD_V1_TEMP, h_);
                CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_, gamma, h_);
                Adds(gamma, gamma, 1.0f, h_);
                gammaCache = gamma;
            }
        }
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            const float inv = invBuf.GetValue(r * c_ + channel);
            Muls(xnFloat[off], x[off], inv, h_);
            if constexpr (HAS_GAMMA) {
                Mul(xnFloat[off], xnFloat[off], gammaCache, h_);
            }
        }
    }

    // For n rows of one channel, transfer x/up/xn/direct/dup with nBurst;
    // only row-specific scalars and multiply/add operations remain inside the row loop.
    __aicore__ inline void ProcessV1Block(uint32_t task, uint32_t rowBegin, uint32_t n, uint32_t channel)
    {
        const uint32_t len = n * h_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + rowBegin;
        const uint64_t inputOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        const uint64_t outputOffset = static_cast<uint64_t>(rowBegin) * d_ + static_cast<uint64_t>(channel) * h_;
        GM_ADDR xn = TaskPointer(layout_.xn, task, d_, sizeof(D_T));
        GM_ADDR dup = TaskPointer(layout_.dup, task, d_, sizeof(D_T));
        auto x = FloatBuf(HC_GRAD_V1_X, len);
        auto up = FloatBuf(HC_GRAD_V1_UP, len);
        auto gradMixed = FloatBuf(HC_GRAD_V1_GRAD_MIXED, len);
        auto xnFloat = FloatBuf(HC_GRAD_V1_XN, len);
        auto gate = FloatBuf(HC_GRAD_V1_GATE, len);
        auto temp = FloatBuf(HC_GRAD_V1_TEMP, len);
        auto xHalf = HalfBuf(HC_GRAD_HALF_INPUT, len);
        auto upHalf = HalfBuf(HC_GRAD_HALF_SECOND_INPUT, len);
        // Do not immediately reuse an output buffer for CopyIn: this serial helper has no MTE3_MTE2 handshake.
        auto xnHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
        auto dupHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
        auto gmHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
        CopyInRows<D_T>(gmX_, inputOffset, d_, xHalf, n, h_, false);
        IssueCopyInRows<D_T>(gmUp_, inputOffset, d_, upHalf, n, h_);
        // grad_mixed [B,S,H] is contiguous across the n rows.
        IssueCopyIn<D_T>(gmGradMixed_, token * h_, gmHalf, len);
        // inv_rms contains n*C contiguous values and needs one transfer.
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset());
        IssueCopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_);
        SyncCopy();
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
        Cast(up, upHalf, RoundMode::CAST_NONE, len);
        Cast(gradMixed, gmHalf, RoundMode::CAST_NONE, len);
        SetFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
        NormalizeV1Rows(n, channel, x, xnFloat, invBuf);
        Cast(xnHalf, xnFloat, RoundMode::CAST_RINT, len);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (blockedG56Xn_) {
            SetFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
            IssueBlockedG56XnRows(task, rowBegin, channel * h_, xnHalf, n, h_);
            SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        } else
#endif
        {
            CopyOutRows<D_T>(xn, outputOffset, d_, xnHalf, n, h_);
        }
        SigmoidFp32(gate, up, len);
        Mul(temp, gradMixed, gate, len);
        ScaleByInvC(temp, temp, len);
        if (!recomputeDirect_) {
            CopyOutRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), outputOffset, d_, temp, n, h_);
        }
        Mul(temp, gradMixed, xnFloat, len);
        ScaleByInvC(temp, temp, len);
        Mul(temp, temp, gate, len);
        Muls(up, gate, -1.0f, len);
        Adds(up, up, 1.0f, len);
        Mul(temp, temp, up, len);
        Cast(dupHalf, temp, RoundMode::CAST_RINT, len);
        CopyOutRows<D_T>(dup, outputOffset, d_, dupHalf, n, h_);
    }

    // V1 pipeline with stage-specific input and output buffers.
    __aicore__ inline void IssueV1In(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t channel,
                                     uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
        const uint64_t inputOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto xHalf = V1InHalf(stage, HC_GRAD_V1_X, len);
        auto upHalf = V1InHalf(stage, HC_GRAD_V1_UP, len);
        auto gmHalf = V1InHalf(stage, HC_GRAD_V1_GRAD_MIXED, len);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), V1InvOffset(stage));
#else
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset(stage));
#endif
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        // Scalar reads of inv_rms require S_MTE2 before overwrite; V_MTE2 protects vector reads only.
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        if (!independentPipeOutput_) {
            WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        }
        IssueCopyInRows<D_T>(gmX_, inputOffset, d_, xHalf, n, h_);
        IssueCopyInRows<D_T>(gmUp_, inputOffset, d_, upHalf, n, h_);
        // grad_mixed [B,S,H] is contiguous across the n rows.
        IssueCopyIn<D_T>(gmGradMixed_, token * h_, gmHalf, len);
        IssueCopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        SetFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
    }

    __aicore__ inline void NormalizeV1PipeRows(uint32_t n, uint32_t channel, LocalTensor<float> x,
                                               LocalTensor<float> directOut, LocalTensor<float> invBuf)
    {
        LocalTensor<float> gammaRow;
        if constexpr (HAS_GAMMA) {
            EnsureGammaCache(channel);
            if (useGammaCache_) {
                gammaRow = ub_.GetWithOffset<float>(h_, GammaOffset());
            } else {
                // Without a gamma cache, temporarily store 1+gamma in the output slot. Ordered vector operations
                // allow direct to overwrite it after xn has consumed it.
                gammaRow = directOut;
                CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_, gammaRow, h_);
                Adds(gammaRow, gammaRow, 1.0f, h_);
            }
        }
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            const float inv = invBuf.GetValue(r * c_ + channel);
            Muls(x[off], x[off], inv, h_);
            if constexpr (HAS_GAMMA) {
                Mul(x[off], x[off], gammaRow, h_);
            }
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static __simd_vf__ inline void A5V1GateAndWeightGrad(__ubuf__ float* direct, __ubuf__ float* dup,
                                                         __ubuf__ float* gradMixed, __ubuf__ float* up,
                                                         __ubuf__ float* xn, uint16_t repeats)
    {
        namespace Reg = AscendC::Reg;
        constexpr uint32_t FP32_LANES = 64U;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> ones;
        Reg::Duplicate(ones, 1.0f);
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_LANES;
            Reg::RegTensor<float> sigmoid, grad, normalized, directVal;
            Reg::LoadAlign(sigmoid, up + offset);
            Reg::LoadAlign(grad, gradMixed + offset);
            Reg::LoadAlign(normalized, xn + offset);
            Reg::Muls(sigmoid, sigmoid, -1.0f, mask);
            Reg::Exp(sigmoid, sigmoid, mask);
            Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
            Reg::Div(sigmoid, ones, sigmoid, mask);
            Reg::Mul(directVal, grad, sigmoid, mask);
            Reg::Muls(directVal, directVal, 0.25f, mask);
            Reg::StoreAlign(direct + offset, directVal, mask);

            Reg::Mul(grad, directVal, normalized, mask);
            Reg::Muls(sigmoid, sigmoid, -1.0f, mask);
            Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
            Reg::Mul(grad, grad, sigmoid, mask);
            Reg::StoreAlign(dup + offset, grad, mask);
        }
    }
#endif

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr AscendC::Reg::CastTrait A5_V1_B16_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait A5_V1_FP32_TO_B16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};

    // The tuned chunk path only exports xn and dup. Keep the cast, normalization,
    // and gate intermediates in registers instead of round-tripping through the UB.
    static __simd_vf__ inline void A5V1PackedXnDupRow(__ubuf__ D_T* xIn, __ubuf__ D_T* upIn, __ubuf__ D_T* mixedIn,
                                                      __ubuf__ float* gamma, __ubuf__ D_T* xnOut, __ubuf__ D_T* dupOut,
                                                      float inv)
    {
        namespace Reg = AscendC::Reg;
        constexpr uint32_t ROW_ELEMS = 2560U;
        constexpr uint32_t LANES = 64U;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> ones;
        Reg::Duplicate(ones, 1.0f);
        for (uint16_t repeat = 0; repeat < ROW_ELEMS / LANES; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(repeat) * LANES;
            Reg::RegTensor<D_T> packed;
            Reg::RegTensor<float> normalized, gammaVal, sigmoid, grad;
            // xnOut and dupOut may reuse xIn and upIn. Read the entire
            // current vector before either store; different iterations are disjoint.
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, xIn + offset);
            Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(normalized, packed, mask);
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, upIn + offset);
            Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(sigmoid, packed, mask);
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, mixedIn + offset);
            Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(grad, packed, mask);
            Reg::LoadAlign(gammaVal, gamma + offset);
            Reg::Muls(normalized, normalized, inv, mask);
            Reg::Mul(normalized, normalized, gammaVal, mask);
            Reg::Muls(sigmoid, sigmoid, -1.0f, mask);
            Reg::Exp(sigmoid, sigmoid, mask);
            Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
            Reg::Div(sigmoid, ones, sigmoid, mask);
            Reg::Mul(grad, grad, sigmoid, mask);
            Reg::Muls(grad, grad, 0.25f, mask);
            // dup consumes the full FP32 normalized value, before xn is rounded.
            Reg::Mul(grad, grad, normalized, mask);
            Reg::Muls(sigmoid, sigmoid, -1.0f, mask);
            Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
            Reg::Mul(grad, grad, sigmoid, mask);
            Reg::Cast<D_T, float, A5_V1_FP32_TO_B16>(packed, normalized, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(xnOut + offset, packed, mask);
            Reg::Cast<D_T, float, A5_V1_FP32_TO_B16>(packed, grad, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(dupOut + offset, packed, mask);
        }
    }

    template <bool THREE_SLOTS>
    __aicore__ inline void PrefetchForV1(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk,
                                         uint32_t blocks, uint32_t channel, uint32_t stage)
    {
        if constexpr (THREE_SLOTS) {
            if (blk + 2U < blocks) {
                const uint32_t futureStage = stage == 0U ? V1_EXTRA_STAGE : stage - 1U;
                IssueV1In(task, rowBegin, rowEnd, blk + 2U, channel, futureStage);
            }
        } else if (!independentPipeOutput_ && blk + 1U < blocks) {
            IssueV1In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
    }

    template <bool THREE_SLOTS>
    __aicore__ inline void ComputeV1PackedFp16(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk,
                                               uint32_t blocks, uint32_t channel, uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        auto xHalf = V1InHalf(stage, HC_GRAD_V1_X, len);
        auto upHalf = V1InHalf(stage, HC_GRAD_V1_UP, len);
        auto gmHalf = V1InHalf(stage, HC_GRAD_V1_GRAD_MIXED, len);
        auto xnHalf = V1OutHalf(stage, HC_GRAD_V1_X, len);
        auto dupHalf = V1OutHalf(stage, HC_GRAD_V1_UP, len);
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), V1InvOffset(stage));
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        PrefetchForV1<THREE_SLOTS>(task, rowBegin, rowEnd, blk, blocks, channel, stage);
        EnsureGammaCache(channel);
        auto gammaRow = ub_.GetWithOffset<float>(h_, GammaOffset());
        auto xAddr = reinterpret_cast<__ubuf__ D_T*>(xHalf.GetPhyAddr());
        auto upAddr = reinterpret_cast<__ubuf__ D_T*>(upHalf.GetPhyAddr());
        auto mixedAddr = reinterpret_cast<__ubuf__ D_T*>(gmHalf.GetPhyAddr());
        auto gammaAddr = reinterpret_cast<__ubuf__ float*>(gammaRow.GetPhyAddr());
        auto xnAddr = reinterpret_cast<__ubuf__ D_T*>(xnHalf.GetPhyAddr());
        auto dupAddr = reinterpret_cast<__ubuf__ D_T*>(dupHalf.GetPhyAddr());
        for (uint32_t row = 0U; row < n; ++row) {
            const uint32_t rowOffset = row * h_;
            const float inv = invBuf.GetValue(row * c_ + channel);
            asc_vf_call<A5V1PackedXnDupRow>(xAddr + rowOffset, upAddr + rowOffset, mixedAddr + rowOffset, gammaAddr,
                                            xnAddr + rowOffset, dupAddr + rowOffset, inv);
        }
    }

    template <bool THREE_SLOTS = false>
    __aicore__ inline void ComputeV1(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t blocks,
                                     uint32_t channel, uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        auto xHalf = V1InHalf(stage, HC_GRAD_V1_X, len);
        auto upHalf = V1InHalf(stage, HC_GRAD_V1_UP, len);
        auto gmHalf = V1InHalf(stage, HC_GRAD_V1_GRAD_MIXED, len);
        auto xnHalf = V1OutHalf(stage, HC_GRAD_V1_X, len);
        auto dupHalf = V1OutHalf(stage, HC_GRAD_V1_UP, len);
        auto x = V1Scratch(HC_GRAD_V1_PIPE_X, len);       // x → xn
        auto gate = V1Scratch(HC_GRAD_V1_PIPE_GATE, len); // gate → (1-gate)
        auto directOut = V1DirectOut(stage, len);
        auto gradMixed = THREE_SLOTS ? directOut : V1Scratch(HC_GRAD_V1_PIPE_TEMP, len);
        auto dup = THREE_SLOTS ? x : gradMixed;
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), V1InvOffset(stage));
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
        Cast(gate, upHalf, RoundMode::CAST_NONE, len);
        Cast(gradMixed, gmHalf, RoundMode::CAST_NONE, len);
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        if constexpr (THREE_SLOTS) {
            if (blk + 2U < blocks) {
                const uint32_t futureStage = stage == 0U ? V1_EXTRA_STAGE : stage - 1U;
                IssueV1In(task, rowBegin, rowEnd, blk + 2U, channel, futureStage);
            }
        } else if (!independentPipeOutput_ && blk + 1U < blocks) {
            IssueV1In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
        if (independentPipeOutput_) {
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        NormalizeV1PipeRows(n, channel, x, directOut, invBuf);
        Cast(xnHalf, x, RoundMode::CAST_RINT, len);
        if (independentPipeOutput_) {
            const uint64_t outputOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            if (!TryIssueBlockedG56XnRows(task, base, channel * h_, xnHalf, n, h_)) {
                IssueCopyOutRows<D_T>(TaskPointer(layout_.xn, task, d_, sizeof(D_T)), outputOffset, d_, xnHalf, n, h_);
            }
        }
        if (c_ == 4U && h_ == 2560U && r_ == 320U && !independentPipeOutput_) {
            constexpr uint32_t FP32_LANES = 64U;
            asc_vf_call<A5V1GateAndWeightGrad>(reinterpret_cast<__ubuf__ float*>(directOut.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(dup.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(gradMixed.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(gate.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr()),
                                               static_cast<uint16_t>(len / FP32_LANES));
            Cast(dupHalf, dup, RoundMode::CAST_RINT, len);
            return;
        }
        SigmoidFp32(gate, gate, len);
        // direct = grad_mixed * gate / C. temp already holds grad_mixed, so write direct to its output buffer
        // without another block copy or cast.
        Mul(directOut, gradMixed, gate, len);
        ScaleByInvC(directOut, directOut, len);
        if (independentPipeOutput_) {
            const uint64_t outputOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            if (!recomputeDirect_) {
                IssueCopyOutRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), outputOffset, d_,
                                        directOut, n, h_);
            }
        }
        // dup = gm * xn / C * gate * (1 - gate) = direct * xn * (1 - gate):
        // Reuse direct to avoid multiplying grad_mixed and 1/C again.
        Mul(dup, directOut, x, len);
        Muls(gate, gate, -1.0f, len);
        Adds(gate, gate, 1.0f, len);
        Mul(dup, dup, gate, len);
        Cast(dupHalf, dup, RoundMode::CAST_RINT, len);
    }
#else
    __aicore__ inline void ComputeV1(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t blocks,
                                     uint32_t channel, uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        auto xHalf = V1InHalf(stage, HC_GRAD_V1_X, len);
        auto upHalf = V1InHalf(stage, HC_GRAD_V1_UP, len);
        auto gmHalf = V1InHalf(stage, HC_GRAD_V1_GRAD_MIXED, len);
        auto xnHalf = V1OutHalf(stage, HC_GRAD_V1_X, len);
        auto dupHalf = V1OutHalf(stage, HC_GRAD_V1_UP, len);
        auto x = V1Scratch(HC_GRAD_V1_PIPE_X, len);       // x → xn
        auto gate = V1Scratch(HC_GRAD_V1_PIPE_GATE, len); // gate → (1-gate)
        auto temp = V1Scratch(HC_GRAD_V1_PIPE_TEMP, len); // gradMixed / temp
        auto directOut = V1DirectOut(stage, len);
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset(stage));
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
        Cast(gate, upHalf, RoundMode::CAST_NONE, len);
        Cast(temp, gmHalf, RoundMode::CAST_NONE, len);
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        if (!independentPipeOutput_ && blk + 1U < blocks) {
            IssueV1In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
        if (independentPipeOutput_) {
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        NormalizeV1PipeRows(n, channel, x, directOut, invBuf);
        Cast(xnHalf, x, RoundMode::CAST_RINT, len);
        if (independentPipeOutput_) {
            const uint64_t outputOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            IssueCopyOutRows<D_T>(TaskPointer(layout_.xn, task, d_, sizeof(D_T)), outputOffset, d_, xnHalf, n, h_);
        }
        SigmoidFp32(gate, gate, len);
        // direct = grad_mixed * gate / C. temp already holds grad_mixed, so write direct to its output buffer
        // without another block copy or cast.
        Mul(directOut, temp, gate, len);
        ScaleByInvC(directOut, directOut, len);
        if (independentPipeOutput_) {
            const uint64_t outputOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            if (!recomputeDirect_) {
                IssueCopyOutRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), outputOffset, d_,
                                        directOut, n, h_);
            }
        }
        // dup = gm * xn / C * gate * (1 - gate) = direct * xn * (1 - gate):
        // Reuse direct to avoid multiplying grad_mixed and 1/C again.
        Mul(temp, directOut, x, len);
        Muls(gate, gate, -1.0f, len);
        Adds(gate, gate, 1.0f, len);
        Mul(temp, temp, gate, len);
        Cast(dupHalf, temp, RoundMode::CAST_RINT, len);
    }
#endif

    __aicore__ inline void IssueV1Out(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t channel,
                                      uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        const uint64_t outputOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
        auto xnHalf = V1OutHalf(stage, HC_GRAD_V1_X, len);
        auto dupHalf = V1OutHalf(stage, HC_GRAD_V1_UP, len);
        auto directOut = V1DirectOut(stage, len);
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        if (!independentPipeOutput_) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            if (!TryIssueBlockedG56XnRows(task, base, channel * h_, xnHalf, n, h_))
#endif
            {
                IssueCopyOutRows<D_T>(TaskPointer(layout_.xn, task, d_, sizeof(D_T)), outputOffset, d_, xnHalf, n, h_);
            }
            if (!recomputeDirect_) {
                IssueCopyOutRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), outputOffset, d_,
                                        directOut, n, h_);
            }
        }
        IssueCopyOutRows<D_T>(TaskPointer(layout_.dup, task, d_, sizeof(D_T)), outputOffset, d_, dupHalf, n, h_);
        SetPipeOutputReady(stage);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    template <bool THREE_SLOTS = false>
    __aicore__ inline void ProcessV1Pipe(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t channel)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
        const uint32_t blocks = (rowEnd - rowBegin + rowBlock_ - 1U) / rowBlock_;
        constexpr uint32_t stages = HC_GRAD_PIPE_MAX_STAGES + static_cast<uint32_t>(THREE_SLOTS);
        PipePrime(stages);
        IssueV1In(task, rowBegin, rowEnd, 0U, channel, 0U);
        if constexpr (THREE_SLOTS) {
            if (blocks > 1U) {
                IssueV1In(task, rowBegin, rowEnd, 1U, channel, 1U);
            }
        }
        uint32_t stage = 0U;
        for (uint32_t blk = 0U; blk < blocks; ++blk) {
            if (independentPipeOutput_ && blk + 1U < blocks) {
                IssueV1In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
            }
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            if constexpr (HAS_GAMMA && AscendC::Std::is_same_v<D_T, half>) {
                if (layout_.fullBatch && recomputeDirect_ && useGammaCache_ && !independentPipeOutput_ &&
                    rowBlock_ == 2U && c_ == 4U && h_ == 2560U && r_ == 320U) {
                    ComputeV1PackedFp16<THREE_SLOTS>(task, rowBegin, rowEnd, blk, blocks, channel, stage);
                } else {
                    ComputeV1<THREE_SLOTS>(task, rowBegin, rowEnd, blk, blocks, channel, stage);
                }
            } else {
                ComputeV1<THREE_SLOTS>(task, rowBegin, rowEnd, blk, blocks, channel, stage);
            }
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV1Out(task, rowBegin, rowEnd, blk, channel, stage);
            if constexpr (THREE_SLOTS) {
                stage = stage + 1U == stages ? 0U : stage + 1U;
            } else {
                stage ^= HC_GRAD_PIPE_STAGE_MASK;
            }
        }
        PipeDrain(stages);
    }
#else
    __aicore__ inline void ProcessV1Pipe(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t channel)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
        const uint32_t blocks = (rowEnd - rowBegin + rowBlock_ - 1U) / rowBlock_;
        PipePrime();
        IssueV1In(task, rowBegin, rowEnd, 0U, channel, 0U);
        for (uint32_t blk = 0U; blk < blocks; ++blk) {
            const uint32_t stage = blk & HC_GRAD_PIPE_STAGE_MASK;
            if (independentPipeOutput_ && blk + 1U < blocks) {
                IssueV1In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
            }
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            ComputeV1(task, rowBegin, rowEnd, blk, blocks, channel, stage);
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV1Out(task, rowBegin, rowEnd, blk, channel, stage);
        }
        PipeDrain();
    }
#endif

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline LocalTensor<D_T> V1ActHalf(uint32_t slot, uint32_t count)
    {
        // Act uses three FP32 buffers. Place both DMA slots after those buffers in the phase pool.
        const uint32_t offset = ubBufBase_ + bufCount_ * (3U * sizeof(float) + slot * sizeof(D_T));
        return ub_.GetWithOffset<D_T>(count, offset);
    }

    __aicore__ inline void IssueV1ActIn(uint32_t task, uint32_t base, uint32_t rows, uint32_t stage)
    {
        const uint32_t len = rows * r_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        IssueCopyIn<D_T>(gmDownPreAct_, token * r_, V1ActHalf(stage, len), len);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
    }

    __aicore__ inline void ProcessV1ActPipe(uint32_t task, uint32_t begin, uint32_t end, uint32_t block, GM_ADDR act)
    {
        for (uint32_t stage = 0; stage < HC_GRAD_PIPE_MAX_STAGES; ++stage) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        uint32_t stage = 0;
        uint32_t rows = end - begin < block ? end - begin : block;
        IssueV1ActIn(task, begin, rows, stage);
        for (uint32_t base = begin; base < end;) {
            const uint32_t next = base + rows;
            const uint32_t nextRows = end - next < block ? end - next : block;
            const uint32_t nextStage = stage ^ HC_GRAD_PIPE_STAGE_MASK;
            if (nextRows != 0U) {
                IssueV1ActIn(task, next, nextRows, nextStage);
            }
            const uint32_t len = rows * r_;
            auto down = FloatBuf(HC_GRAD_ACT_DOWN, len);
            auto sigmoid = FloatBuf(HC_GRAD_ACT_SIGMOID, len);
            auto actFloat = FloatBuf(HC_GRAD_ACT_OUTPUT, len);
            auto actHalf = V1ActHalf(HC_GRAD_PIPE_MAX_STAGES + stage, len);
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            Cast(down, V1ActHalf(stage, len), RoundMode::CAST_NONE, len);
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            ScaleByInvC(down, down, len);
            SigmoidFp32(sigmoid, down, len);
            Mul(actFloat, down, sigmoid, len);
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            Cast(actHalf, actFloat, RoundMode::CAST_RINT, len);
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            IssueCopyOutRows<D_T>(act, static_cast<uint64_t>(base) * r_, len, actHalf, 1U, len);
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            base = next;
            rows = nextRows;
            stage = nextStage;
        }
        for (uint32_t stage = 0; stage < HC_GRAD_PIPE_MAX_STAGES; ++stage) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
    }
#endif

    __aicore__ inline void ProcessV1Act(uint32_t task, uint32_t rows)
    {
        GM_ADDR act = TaskPointer(layout_.act, task, r_, sizeof(D_T));
        // down_pre_act and act [token,R] are contiguous across n rows.
        const uint32_t actRowsPerCore = (rows + aivNum_ - 1U) / aivNum_;
        const uint32_t actBegin = aivId_ * actRowsPerCore;
        const uint32_t actEnd = (actBegin + actRowsPerCore < rows) ? (actBegin + actRowsPerCore) : rows;
        uint32_t blk = bufCount_ / r_;
        if (blk < 1U) {
            blk = 1U;
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        constexpr uint32_t ACT_SLOTS_BYTES = 3U * sizeof(float) + 2U * HC_GRAD_PIPE_MAX_STAGES * sizeof(D_T);
        if (c_ == 4U && r_ == 320U && actBegin < actEnd && actEnd - actBegin > blk && r_ <= bufCount_ &&
            ACT_SLOTS_BYTES * bufCount_ <= poolBytes_) {
            ProcessV1ActPipe(task, actBegin, actEnd, blk, act);
            return;
        }
#endif
        for (uint32_t base = actBegin; base < actEnd; base += blk) {
            const uint32_t nb = (actEnd - base < blk) ? (actEnd - base) : blk;
            const uint32_t len = nb * r_;
            const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
            auto down = FloatBuf(HC_GRAD_ACT_DOWN, len);
            auto sigmoid = FloatBuf(HC_GRAD_ACT_SIGMOID, len);
            auto actFloat = FloatBuf(HC_GRAD_ACT_OUTPUT, len);
            auto downHalf = HalfBuf(HC_GRAD_HALF_INPUT, len);
            auto actHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
            CopyIn<D_T>(gmDownPreAct_, token * r_, downHalf, len);
            Cast(down, downHalf, RoundMode::CAST_NONE, len);
            ScaleByInvC(down, down, len);
            SigmoidFp32(sigmoid, down, len);

            Mul(actFloat, down, sigmoid, len);
            Cast(actHalf, actFloat, RoundMode::CAST_RINT, len);
            CopyOut<D_T>(act, static_cast<uint64_t>(base) * r_, actHalf, len);
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr uint32_t V1_PAIR_H = 2560U;
    static constexpr uint32_t V1_PAIR_CHANNELS = 2U;
    static constexpr uint32_t V1_PAIR_ROWS = 2U;
    static constexpr uint32_t V1_PAIR_ROW_ELEMS = V1_PAIR_CHANNELS * V1_PAIR_H;
    static constexpr uint32_t V1_PAIR_TILE_ELEMS = V1_PAIR_ROWS * V1_PAIR_ROW_ELEMS;
    static constexpr uint32_t V1_PAIR_UP_OFFSET = V1_PAIR_TILE_ELEMS * sizeof(D_T);
    static constexpr uint32_t V1_PAIR_MIXED_OFFSET = 2U * V1_PAIR_UP_OFFSET;
    static constexpr uint32_t V1_PAIR_INV_OFFSET = V1_PAIR_MIXED_OFFSET + V1_PAIR_ROWS * V1_PAIR_H * sizeof(D_T);
    static constexpr uint32_t V1_PAIR_STAGE_BYTES = V1_PAIR_INV_OFFSET + HC_GRAD_DATA_BLOCK_BYTES;
    static constexpr uint32_t V1_PAIR_NORM_OFFSET = HC_GRAD_PIPE_MAX_STAGES * V1_PAIR_STAGE_BYTES;
    static constexpr uint32_t V1_PAIR_GAMMA1_OFFSET = V1_PAIR_NORM_OFFSET + V1_PAIR_TILE_ELEMS * sizeof(float);
    static constexpr uint32_t V1_PAIR_POOL_BYTES = V1_PAIR_GAMMA1_OFFSET + V1_PAIR_H * sizeof(float);

    static constexpr uint32_t V1_FULL_C = 4U;
    static constexpr uint32_t V1_FULL_D = V1_FULL_C * V1_PAIR_H;
    static constexpr uint32_t V1_FULL_GAMMA_BYTES = V1_FULL_D * sizeof(float);
    static constexpr uint32_t V1_FULL_UP_OFFSET = V1_PAIR_ROWS * V1_FULL_D * sizeof(D_T);
    static constexpr uint32_t V1_FULL_MIXED_OFFSET = 2U * V1_FULL_UP_OFFSET;
    static constexpr uint32_t V1_FULL_INV_OFFSET = V1_FULL_MIXED_OFFSET + V1_PAIR_ROWS * V1_PAIR_H * sizeof(D_T);
    static constexpr uint32_t V1_FULL_STAGE_BYTES = V1_FULL_INV_OFFSET + HC_GRAD_DATA_BLOCK_BYTES;

    __aicore__ inline bool CanUseFullV1Rows() const
    {
        return layout_.fullBatch && recomputeDirect_ && c_ == V1_FULL_C && h_ == V1_PAIR_H && r_ == 320U &&
               pipeStages_ == HC_GRAD_PIPE_MAX_STAGES && !independentPipeOutput_ &&
               ubBytes_ >= V1_FULL_GAMMA_BYTES + HC_GRAD_PIPE_MAX_STAGES * V1_FULL_STAGE_BYTES;
    }

    static __simd_vf__ inline void A5V1FullRow(__ubuf__ D_T* xn, __ubuf__ D_T* up, __ubuf__ D_T* mixed,
                                               __ubuf__ float* gamma, __ubuf__ float* inv)
    {
        namespace Reg = AscendC::Reg;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> one;
        Reg::Duplicate(one, 1.0f);
        for (uint16_t i = 0; i < V1_PAIR_H / 64U; ++i) {
            const uint32_t col = i * 64U;
            Reg::RegTensor<D_T> packed;
            Reg::RegTensor<float> mixedVal;
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, mixed + col);
            Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(mixedVal, packed, mask);
#pragma unroll
            for (uint16_t ch = 0U; ch < V1_FULL_C; ++ch) {
                const uint32_t off = ch * V1_PAIR_H + col;
                Reg::RegTensor<float> normalized, scale, invVal, sig, grad;
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, xn + off);
                Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(normalized, packed, mask);
                Reg::LoadAlign(scale, gamma + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(invVal, inv + ch);
                Reg::Mul(normalized, normalized, invVal, mask);
                Reg::Mul(normalized, normalized, scale, mask);
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, up + off);
                Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(sig, packed, mask);
                Reg::Muls(sig, sig, -1.0f, mask);
                Reg::Exp(sig, sig, mask);
                Reg::Adds(sig, sig, 1.0f, mask);
                Reg::Div(sig, one, sig, mask);
                Reg::Mul(grad, mixedVal, sig, mask);
                Reg::Muls(grad, grad, 0.25f, mask);
                Reg::Mul(grad, grad, normalized, mask);
                Reg::Muls(sig, sig, -1.0f, mask);
                Reg::Adds(sig, sig, 1.0f, mask);
                Reg::Mul(grad, grad, sig, mask);
                Reg::Cast<D_T, float, A5_V1_PAIR_FP32_TO_B16>(packed, normalized, mask);
                Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(xn + off, packed, mask);
                Reg::Cast<D_T, float, A5_V1_PAIR_FP32_TO_B16>(packed, grad, mask);
                Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(up + off, packed, mask);
            }
        }
    }

    template <class T>
    __aicore__ inline LocalTensor<T> FullV1Stage(uint32_t stage, uint32_t offset, uint32_t elements)
    {
        return ub_.GetWithOffset<T>(elements, V1_FULL_GAMMA_BYTES + stage * V1_FULL_STAGE_BYTES + offset);
    }

    __aicore__ inline void LoadFullV1Rows(uint32_t stage, uint32_t row, uint32_t rows)
    {
        const uint64_t token = row;
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        IssueCopyIn<D_T, true>(gmX_, token * d_, FullV1Stage<D_T>(stage, 0U, rows * d_), rows * d_);
        IssueCopyIn<D_T, true>(gmUp_, token * d_, FullV1Stage<D_T>(stage, V1_FULL_UP_OFFSET, rows * d_), rows * d_);
        IssueCopyIn<D_T, true>(gmGradMixed_, token * h_, FullV1Stage<D_T>(stage, V1_FULL_MIXED_OFFSET, rows * h_),
                               rows * h_);
        IssueCopyIn<float>(gmInvRms_, token * c_, FullV1Stage<float>(stage, V1_FULL_INV_OFFSET, 8U), rows * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
    }

    __aicore__ __attribute__((always_inline)) inline void ProcessFullV1Rows()
    {
        const uint32_t perCore = (n_ + aivNum_ - 1U) / aivNum_;
        const uint32_t begin = aivId_ * perCore;
        if (begin >= n_) {
            return;
        }
        const uint32_t end = n_ - begin < perCore ? n_ : begin + perCore;
        const uint32_t first = end - begin < V1_PAIR_ROWS ? end - begin : V1_PAIR_ROWS;
        auto gamma = ub_.GetWithOffset<float>(V1_FULL_D, 0U);
        CopyIn<float>(gmGamma_, 0U, gamma, V1_FULL_D);
        Adds(gamma, gamma, 1.0f, V1_FULL_D);
        GM_ADDR gmXn = TaskPointer(layout_.xn, 0U, d_, sizeof(D_T));
        GM_ADDR gmDup = TaskPointer(layout_.dup, 0U, d_, sizeof(D_T));
        uint32_t stage = 0U;
        PipePrime();
        LoadFullV1Rows(stage, begin, first);
        for (uint32_t row = begin; row < end; row += V1_PAIR_ROWS) {
            const uint32_t rows = end - row < V1_PAIR_ROWS ? end - row : V1_PAIR_ROWS;
            const uint32_t next = row + rows;
            if (next < end) {
                LoadFullV1Rows(stage ^ 1U, next, end - next < V1_PAIR_ROWS ? end - next : V1_PAIR_ROWS);
            }
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            auto xn = FullV1Stage<D_T>(stage, 0U, rows * d_);
            auto up = FullV1Stage<D_T>(stage, V1_FULL_UP_OFFSET, rows * d_);
            auto mixed = FullV1Stage<D_T>(stage, V1_FULL_MIXED_OFFSET, rows * h_);
            auto inv = FullV1Stage<float>(stage, V1_FULL_INV_OFFSET, 8U);
            for (uint32_t i = 0U; i < rows; ++i) {
                asc_vf_call<A5V1FullRow>(reinterpret_cast<__ubuf__ D_T*>(xn.GetPhyAddr()) + i * d_,
                                         reinterpret_cast<__ubuf__ D_T*>(up.GetPhyAddr()) + i * d_,
                                         reinterpret_cast<__ubuf__ D_T*>(mixed.GetPhyAddr()) + i * h_,
                                         reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(inv.GetPhyAddr()) + i * c_);
            }
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            const uint64_t off = static_cast<uint64_t>(row) * d_;
            if (!TryIssueBlockedG56XnRows(0U, row, 0U, xn, rows, d_)) {
                IssueCopyOutRows<D_T>(gmXn, off, d_, xn, rows, d_);
            }
            IssueCopyOutRows<D_T>(gmDup, off, d_, up, rows, d_);
            SetPipeOutputReady(stage);
            stage ^= 1U;
        }
        PipeDrain();
        gammaCachedCh_ = -1;
    }

    __aicore__ inline bool CanUseV1ChannelPairs() const
    {
        return layout_.fullBatch && layout_.duplicateV1Operands && recomputeDirect_ && useGammaCache_ && kahanDgamma_ &&
               !independentPipeOutput_ && pipeStages_ == HC_GRAD_PIPE_MAX_STAGES && !CanUseV1ThreeSlots() &&
               aivNum_ == 56U && rowBlock_ == V1_PAIR_ROWS && c_ == 4U && h_ == V1_PAIR_H && r_ == 320U &&
               d_ == c_ * h_ && V1_PAIR_POOL_BYTES <= poolBytes_ && GammaOffset() + h_ * sizeof(float) <= ubBytes_;
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> V1PairStage(uint32_t stage, uint32_t offset, uint32_t count)
    {
        return ub_.GetWithOffset<T>(count, ubBufBase_ + stage * V1_PAIR_STAGE_BYTES + offset);
    }

    __aicore__ inline void IssueV1PairIn(uint32_t task, uint32_t row, uint32_t n, uint32_t channel, uint32_t stage)
    {
        const uint32_t elements = n * V1_PAIR_ROW_ELEMS;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + row;
        const uint64_t offset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto x = V1PairStage<D_T>(stage, 0U, elements);
        auto up = V1PairStage<D_T>(stage, V1_PAIR_UP_OFFSET, elements);
        auto mixed = V1PairStage<D_T>(stage, V1_PAIR_MIXED_OFFSET, n * h_);
        auto inv = V1PairStage<float>(stage, V1_PAIR_INV_OFFSET, HC_GRAD_DATA_BLOCK_BYTES / sizeof(float));
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        IssueCopyInRows<D_T>(gmX_, offset, d_, x, n, V1_PAIR_ROW_ELEMS);
        IssueCopyInRows<D_T>(gmUp_, offset, d_, up, n, V1_PAIR_ROW_ELEMS);
        IssueCopyIn<D_T>(gmGradMixed_, token * h_, mixed, n * h_);
        IssueCopyIn<float>(gmInvRms_, token * c_, inv, n * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        SetFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
    }

    static constexpr AscendC::Reg::CastTrait A5_V1_PAIR_FP32_TO_B16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};

    static __simd_vf__ inline void A5V1NormalizeGatePair(__ubuf__ D_T* xn, __ubuf__ D_T* up, __ubuf__ D_T* mixed,
                                                         __ubuf__ float* gamma0, __ubuf__ float* gamma1, float inv0,
                                                         float inv1)
    {
        namespace Reg = AscendC::Reg;
        constexpr uint32_t FP32_LANES = 64U;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> ones;
        Reg::Duplicate(ones, 1.0f);
        for (uint16_t repeat = 0; repeat < V1_PAIR_H / FP32_LANES; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_LANES;
            Reg::RegTensor<D_T> packed;
            Reg::RegTensor<float> mixedVal;
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, mixed + offset);
            Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(mixedVal, packed, mask);
            // The two channels share grad_mixed and each keeps xn in FP32 until dup is computed.
#pragma unroll
            for (uint16_t channel = 0U; channel < V1_PAIR_CHANNELS; ++channel) {
                const uint32_t pairOff = offset + channel * V1_PAIR_H;
                __ubuf__ float* gamma = channel == 0U ? gamma0 : gamma1;
                const float inv = channel == 0U ? inv0 : inv1;
                Reg::RegTensor<float> sigmoid, normalized, gammaVal, directVal, grad;
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, xn + pairOff);
                Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(normalized, packed, mask);
                Reg::LoadAlign(gammaVal, gamma + offset);
                Reg::Muls(normalized, normalized, inv, mask);
                Reg::Mul(normalized, normalized, gammaVal, mask);
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, up + pairOff);
                Reg::Cast<float, D_T, A5_V1_B16_TO_FP32>(sigmoid, packed, mask);
                Reg::Muls(sigmoid, sigmoid, -1.0f, mask);
                Reg::Exp(sigmoid, sigmoid, mask);
                Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
                Reg::Div(sigmoid, ones, sigmoid, mask);
                Reg::Mul(directVal, mixedVal, sigmoid, mask);
                Reg::Muls(directVal, directVal, 0.25f, mask);
                Reg::Mul(grad, directVal, normalized, mask);
                Reg::Muls(sigmoid, sigmoid, -1.0f, mask);
                Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
                Reg::Mul(grad, grad, sigmoid, mask);
                Reg::Cast<D_T, float, A5_V1_PAIR_FP32_TO_B16>(packed, normalized, mask);
                Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(xn + pairOff, packed, mask);
                Reg::Cast<D_T, float, A5_V1_PAIR_FP32_TO_B16>(packed, grad, mask);
                Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(up + pairOff, packed, mask);
            }
        }
    }

    __aicore__ __attribute__((noinline)) void ProcessV1ChannelPairs(uint32_t task, uint32_t rows)
    {
        const uint32_t pairs = c_ / V1_PAIR_CHANNELS;
        const uint32_t channel = (aivId_ % pairs) * V1_PAIR_CHANNELS;
        const uint32_t lanes = aivNum_ / pairs;
        const uint32_t laneRows = (rows + lanes - 1U) / lanes;
        const uint32_t begin = (aivId_ / pairs) * laneRows;
        const uint32_t end = begin + laneRows < rows ? begin + laneRows : rows;
        if (begin >= end) {
            return;
        }

        EnsureGammaCache(channel);
        auto gamma0 = ub_.GetWithOffset<float>(h_, GammaOffset());
        // Keep both gamma vectors live without borrowing the resident dgamma slots.
        auto gamma1 = ub_.GetWithOffset<float>(h_, ubBufBase_ + V1_PAIR_GAMMA1_OFFSET);
        CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel + 1U) * h_, gamma1, h_);
        Adds(gamma1, gamma1, 1.0f, h_);

        GM_ADDR xn = TaskPointer(layout_.xn, task, d_, sizeof(D_T));
        GM_ADDR dup = TaskPointer(layout_.dup, task, d_, sizeof(D_T));
        uint32_t stage = 0U;
        uint32_t n = end - begin < V1_PAIR_ROWS ? end - begin : V1_PAIR_ROWS;
        PipePrime();
        IssueV1PairIn(task, begin, n, channel, stage);
        for (uint32_t row = begin; row < end; row += V1_PAIR_ROWS) {
            const uint32_t elements = n * V1_PAIR_ROW_ELEMS;
            auto xnHalf = V1PairStage<D_T>(stage, 0U, elements);
            auto upHalf = V1PairStage<D_T>(stage, V1_PAIR_UP_OFFSET, elements);
            auto mixedHalf = V1PairStage<D_T>(stage, V1_PAIR_MIXED_OFFSET, n * h_);
            auto inv = V1PairStage<float>(stage, V1_PAIR_INV_OFFSET, HC_GRAD_DATA_BLOCK_BYTES / sizeof(float));
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
            const uint32_t nextStage = stage ^ HC_GRAD_PIPE_STAGE_MASK;
            const uint32_t nextRow = row + n;
            const uint32_t nextCount = end - nextRow < V1_PAIR_ROWS ? end - nextRow : V1_PAIR_ROWS;
            if (nextCount != 0U) {
                IssueV1PairIn(task, nextRow, nextCount, channel, nextStage);
            }

            auto xnAddr = reinterpret_cast<__ubuf__ D_T*>(xnHalf.GetPhyAddr());
            auto upAddr = reinterpret_cast<__ubuf__ D_T*>(upHalf.GetPhyAddr());
            auto mixedAddr = reinterpret_cast<__ubuf__ D_T*>(mixedHalf.GetPhyAddr());
            auto gammaAddr0 = reinterpret_cast<__ubuf__ float*>(gamma0.GetPhyAddr());
            auto gammaAddr1 = reinterpret_cast<__ubuf__ float*>(gamma1.GetPhyAddr());
            for (uint32_t r = 0U; r < n; ++r) {
                const uint32_t pairOff = r * V1_PAIR_ROW_ELEMS;
                const float inv0 = inv.GetValue(r * c_ + channel);
                const float inv1 = inv.GetValue(r * c_ + channel + 1U);
                asc_vf_call<A5V1NormalizeGatePair>(xnAddr + pairOff, upAddr + pairOff, mixedAddr + r * h_, gammaAddr0,
                                                   gammaAddr1, inv0, inv1);
            }

            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            const uint64_t offset = static_cast<uint64_t>(row) * d_ + static_cast<uint64_t>(channel) * h_;
            if (!TryIssueBlockedG56XnRows(task, row, channel * h_, xnHalf, n, V1_PAIR_ROW_ELEMS)) {
                IssueCopyOutRows<D_T>(xn, offset, d_, xnHalf, n, V1_PAIR_ROW_ELEMS);
            }
            IssueCopyOutRows<D_T>(dup, offset, d_, upHalf, n, V1_PAIR_ROW_ELEMS);
            SetPipeOutputReady(stage);
            stage = nextStage;
            n = nextCount;
        }
        PipeDrain();
    }

#endif

    __aicore__ inline void ProcessV1(uint32_t task, uint32_t rows)
    {
        bool produceAct = true;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            produceAct = !CanFuseV3Act();
        }
#endif
        if (produceAct) {
            ProcessV1Act(task, rows);
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            if (CanUseV1ChannelPairs()) {
                ProcessV1ChannelPairs(task, rows);
                return;
            }
        }
#endif
        for (uint32_t channel = aivId_ % c_; channel < c_; channel += aivNum_) {
            const bool firstWave = channel < aivNum_;
            const uint32_t lane = firstWave ? aivId_ / c_ : 0U;
            const uint32_t laneCount = firstWave ? (aivNum_ - 1U - channel) / c_ + 1U : 1U;
            // Partition rows contiguously across cores; row counts differ by at most one.
            const uint32_t rowsPerCore = (rows + laneCount - 1U) / laneCount;
            const uint32_t rowBegin = lane * rowsPerCore;
            const uint32_t rowEnd = (rowBegin + rowsPerCore < rows) ? (rowBegin + rowsPerCore) : rows;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            if (CanUseV1ThreeSlots()) {
                ProcessV1Pipe<true>(task, rowBegin, rowEnd, channel);
                continue;
            }
#endif
            if (pipeStages_ != 0U) {
                ProcessV1Pipe(task, rowBegin, rowEnd, channel);
                continue;
            }
            for (uint32_t base = rowBegin; base < rowEnd; base += rowBlock_) {
                const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
                ProcessV1Block(task, base, n, channel);
            }
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    static constexpr uint32_t FULL_V1_ROWS = 2U;
    static constexpr uint32_t FULL_V1_F32_SLOTS = 4U;
    static constexpr uint32_t FULL_V1_INPUT_SLOTS = 3U;
    static constexpr uint32_t FULL_V1_OUTPUT_SLOTS = 2U;

    __aicore__ inline uint32_t FullV1BlockElems() const
    {
        return FULL_V1_ROWS * h_;
    }

    __aicore__ inline uint32_t FullV1GammaOffset() const
    {
        return FullV1BlockElems() *
               (FULL_V1_F32_SLOTS * sizeof(float) +
                HC_GRAD_PIPE_MAX_STAGES * (FULL_V1_INPUT_SLOTS + FULL_V1_OUTPUT_SLOTS) * sizeof(D_T));
    }

    __aicore__ inline uint32_t FullV1InvOffset(uint32_t stage) const
    {
        return FullV1GammaOffset() + HAS_GAMMA * h_ * sizeof(float) + HC_GRAD_DATA_BLOCK_BYTES * stage;
    }

    __aicore__ inline uint32_t FullV1RequiredBytes() const
    {
        return FullV1InvOffset(HC_GRAD_PIPE_MAX_STAGES) + 64U * sizeof(float);
    }

    __aicore__ inline LocalTensor<D_T> FullV1Half(uint32_t buf, uint32_t n)
    {
        return ub_.GetWithOffset<D_T>(n, (FULL_V1_F32_SLOTS * sizeof(float) + buf * sizeof(D_T)) * FullV1BlockElems());
    }

    __aicore__ inline void FullV1Load(uint32_t row, uint32_t n, uint32_t channel, uint32_t stage)
    {
        const uint32_t len = n * h_;
        const uint64_t offset = static_cast<uint64_t>(row) * d_ + channel * h_;
        const uint32_t inputSlot = stage * FULL_V1_INPUT_SLOTS;
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        IssueCopyInRows<D_T>(gmX_, offset, d_, FullV1Half(inputSlot, len), n, h_);
        IssueCopyInRows<D_T>(gmUp_, offset, d_, FullV1Half(inputSlot + 1U, len), n, h_);
        IssueCopyIn<D_T>(gmGradMixed_, static_cast<uint64_t>(row) * h_, FullV1Half(inputSlot + 2U, len), len);
        auto inv = ub_.GetWithOffset<float>(HC_GRAD_DATA_BLOCK_BYTES / sizeof(float), FullV1InvOffset(stage));
        IssueCopyIn<float>(gmInvRms_, static_cast<uint64_t>(row) * c_, inv, n * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        SetFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
    }

    __aicore__ inline void FullV1Compute(uint32_t row, uint32_t n, uint32_t channel, uint32_t stage,
                                         LocalTensor<float> gamma)
    {
        constexpr uint32_t elementsPerRepeat = 64U;
        constexpr uint32_t blocksPerRepeat = 8U;
        const BinaryRepeatParams strides{1U, 1U, 1U, blocksPerRepeat, 0U, blocksPerRepeat};
        const uint32_t len = n * h_;
        const uint32_t buffer = FullV1BlockElems() * sizeof(float);
        const uint32_t inputSlot = stage * FULL_V1_INPUT_SLOTS;
        const uint32_t outputSlot = HC_GRAD_PIPE_MAX_STAGES * FULL_V1_INPUT_SLOTS + stage * FULL_V1_OUTPUT_SLOTS;
        auto x = ub_.GetWithOffset<float>(len, 0U);
        auto gate = ub_.GetWithOffset<float>(len, buffer);
        auto direct = ub_.GetWithOffset<float>(len, (2U + stage) * buffer);
        auto xn = FullV1Half(outputSlot, len);
        auto dup = FullV1Half(outputSlot + 1U, len);
        auto ones = ub_.GetWithOffset<float>(elementsPerRepeat, FullV1InvOffset(HC_GRAD_PIPE_MAX_STAGES));
        auto inv = ub_.GetWithOffset<float>(HC_GRAD_DATA_BLOCK_BYTES / sizeof(float), FullV1InvOffset(stage));

        WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        Cast(x, FullV1Half(inputSlot, len), RoundMode::CAST_NONE, len);
        Cast(gate, FullV1Half(inputSlot + 1U, len), RoundMode::CAST_NONE, len);
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t off = i * h_;
            const float scale = inv.GetValue(i * c_ + channel);
            Muls(x[off], x[off], scale, h_);
            if constexpr (HAS_GAMMA) {
                Mul(x[off], x[off], gamma, h_);
            }
        }
        const uint64_t offset = static_cast<uint64_t>(row) * d_ + channel * h_;
        WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        Cast(xn, x, RoundMode::CAST_RINT, len);
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        IssueCopyOutRows<D_T>(TaskPointer(layout_.xn, 0U, d_, sizeof(D_T)), offset, d_, xn, n, h_);
        Cast(direct, FullV1Half(inputSlot + 2U, len), RoundMode::CAST_NONE, len);
        SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));

        Muls(gate, gate, -1.0f, len);
        Exp(gate, gate, len);
        Adds(gate, gate, 1.0f, len);
        Div(gate, ones, gate, elementsPerRepeat, static_cast<uint8_t>(len / elementsPerRepeat), strides);
        Mul(direct, direct, gate, len);
        Muls(direct, direct, 0.25f, len);
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        if (!recomputeDirect_) {
            IssueCopyOutRows<float>(Pointer(layout_.direct), offset, d_, direct, n, h_);
        }

        Mul(x, direct, x, len);
        Muls(gate, gate, -1.0f, len);
        Adds(gate, gate, 1.0f, len);
        Mul(x, x, gate, len);
        Cast(dup, x, RoundMode::CAST_RINT, len);
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        IssueCopyOutRows<D_T>(TaskPointer(layout_.dup, 0U, d_, sizeof(D_T)), offset, d_, dup, n, h_);
        SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
    }

    __aicore__ inline void RunFullV1Rows(uint32_t begin, uint32_t end, uint32_t channel)
    {
        auto gamma = ub_.GetWithOffset<float>(h_, FullV1GammaOffset());
        if constexpr (HAS_GAMMA) {
            CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_, gamma, h_);
            Adds(gamma, gamma, 1.0f, h_);
        }
        auto ones = ub_.GetWithOffset<float>(64U, FullV1InvOffset(HC_GRAD_PIPE_MAX_STAGES));
        Duplicate(ones, 1.0f, 64U);
        for (uint32_t stage = 0; stage < HC_GRAD_PIPE_MAX_STAGES; ++stage) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        uint32_t stage = 0;
        FullV1Load(begin, end - begin < FULL_V1_ROWS ? end - begin : FULL_V1_ROWS, channel, stage);
        for (uint32_t row = begin; row < end; row += FULL_V1_ROWS) {
            const uint32_t n = end - row < FULL_V1_ROWS ? end - row : FULL_V1_ROWS;
            const uint32_t nextStage = stage ^ HC_GRAD_PIPE_STAGE_MASK;
            const uint32_t nextRow = row + FULL_V1_ROWS;
            if (nextRow < end) {
                const uint32_t nextN = end - nextRow < FULL_V1_ROWS ? end - nextRow : FULL_V1_ROWS;
                FullV1Load(nextRow, nextN, channel, nextStage);
            }
            FullV1Compute(row, n, channel, stage, gamma);
            stage = nextStage;
        }
        for (uint32_t i = 0; i < HC_GRAD_PIPE_MAX_STAGES; ++i) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(i));
            WaitFlag<HardEvent::S_MTE2>(PipeInEvent(i));
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(i));
        }
        PipeBarrier<PIPE_ALL>();
    }
#endif

    __aicore__ inline void ComputeV3Gate(const LocalTensor<float>& gradInject, const LocalTensor<float>& inject,
                                         const LocalTensor<float>& temp, const LocalTensor<D_T>& gradHalf,
                                         const LocalTensor<D_T>& injectHalf, uint32_t count)
    {
        Cast(gradInject, gradHalf, RoundMode::CAST_NONE, count);
        Cast(inject, injectHalf, RoundMode::CAST_NONE, count);
        Muls(inject, inject, HC_GRAD_INJECT_TO_SIGMOID, count);
        Muls(temp, inject, -1.0f, count);
        Adds(temp, temp, 1.0f, count);
        Muls(inject, inject, HC_GRAD_SIGMOID_TO_INJECT, count);
        Mul(temp, inject, temp, count);
        Mul(gradInject, gradInject, temp, count);
        ScaleByInvC(gradInject, gradInject, count);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void ComputeV3Down(const LocalTensor<float>& dact, const LocalTensor<float>& down,
                                         const LocalTensor<float>& sigmoid, const LocalTensor<float>& derivative,
                                         const LocalTensor<D_T>& downHalf, uint32_t count, uint32_t sigmoidScratch = 0U)
    {
        Cast(down, downHalf, RoundMode::CAST_NONE, count);
        ScaleByInvC(down, down, count);
        SigmoidFp32(sigmoid, down, count, sigmoidScratch);
#else
    __aicore__ inline void ComputeV3Down(const LocalTensor<float>& dact, const LocalTensor<float>& down,
                                         const LocalTensor<float>& sigmoid, const LocalTensor<float>& derivative,
                                         const LocalTensor<D_T>& downHalf, uint32_t count)
    {
        Cast(down, downHalf, RoundMode::CAST_NONE, count);
        ScaleByInvC(down, down, count);
        SigmoidFp32(sigmoid, down, count);
#endif
        Mul(dact, dact, sigmoid, count);
        Muls(derivative, sigmoid, -1.0f, count);
        Adds(derivative, derivative, 1.0f, count);
        Mul(derivative, derivative, down, count);
        Adds(derivative, derivative, 1.0f, count);
        Mul(dact, dact, derivative, count);
        ScaleByInvC(dact, dact, count);
    }

    __aicore__ inline void ProcessGateLogits(uint32_t task, uint32_t base, uint32_t nb, GM_ADDR packed)
    {
        const uint32_t gateChunk = c_ < bufCount_ ? c_ : bufCount_;
        auto gradInject = FloatBuf(HC_GRAD_GATE_GRAD_INJECT, gateChunk);
        auto inject = FloatBuf(HC_GRAD_GATE_INJECT, gateChunk);
        auto temp = FloatBuf(HC_GRAD_GATE_TEMP, gateChunk);
        auto giHalf = HalfBuf(HC_GRAD_HALF_INPUT, gateChunk);
        auto injectHalf = HalfBuf(HC_GRAD_HALF_SECOND_INPUT, gateChunk);
        auto dlogitsHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, gateChunk);
        for (uint32_t rr = 0; rr < nb; ++rr) {
            const uint32_t row = base + rr;
            const uint64_t tok = static_cast<uint64_t>(task) * taskRows_ + row;
            for (uint32_t cOff = 0; cOff < c_; cOff += gateChunk) {
                const uint32_t count = c_ - cOff < gateChunk ? c_ - cOff : gateChunk;
                // One producer fence and one completion wait cover both independent input copies.
                CopyIn<D_T>(gmGradInject_, tok * c_ + cOff, giHalf, count, false);
                IssueCopyIn<D_T>(gmInject_, tok * c_ + cOff, injectHalf, count);
                SyncCopy();
                ComputeV3Gate(gradInject, inject, temp, giHalf, injectHalf, count);
                // The final gate chunk owns the entire aligned row tail. Initialize
                // padding here instead of zeroing packed on unrelated cores in Prepare.
                const uint32_t storeCount = HcGradAlign16(count);
                if (storeCount != count) {
                    Duplicate(dlogitsHalf, static_cast<D_T>(0), storeCount);
                }
                Cast(dlogitsHalf, gradInject, RoundMode::CAST_RINT, count);
                CopyOut<D_T>(packed, static_cast<uint64_t>(row) * g56Rows_ + r_ + cOff, dlogitsHalf, storeCount);
                if (layout_.denseDx != 0U) {
                    CopyOut<D_T>(DenseDxTaskPointer(task), static_cast<uint64_t>(row) * DenseDxColumns() + r_ + cOff,
                                 dlogitsHalf, count);
                }
            }
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    // V3 has two dact/down input slots and two independent FP16/BF16 output slots.
    // Three shared FP32 scratch buffers serve both the down and the gate math.
    static constexpr uint32_t V3_PIPE_FLOAT_SLOTS = 5U;
    static constexpr uint32_t V3_PIPE_HALF_SLOTS = 4U;
    static constexpr uint32_t V3_GATE_HALF_SLOTS = 3U;
    static constexpr uint32_t V3_PIPE_GATE_COLS = 4U;

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline LocalTensor<float> V3PipeFloat(uint32_t slot, uint32_t count, uint32_t capacity)
    {
        return ub_.GetWithOffset<float>(count, ubBufBase_ + slot * capacity * sizeof(float));
    }

    __aicore__ inline uint32_t V3PipeHalfBase(uint32_t capacity) const
    {
        return ubBufBase_ + V3_PIPE_FLOAT_SLOTS * capacity * sizeof(float);
    }

    __aicore__ inline uint32_t V3PipeGateBase(uint32_t capacity) const
    {
        return V3PipeHalfBase(capacity) + V3_PIPE_HALF_SLOTS * capacity * sizeof(D_T);
    }

    __aicore__ inline LocalTensor<D_T> V3PipeHalf(uint32_t slot, uint32_t count, uint32_t capacity)
    {
        return ub_.GetWithOffset<D_T>(count, V3PipeHalfBase(capacity) + slot * capacity * sizeof(D_T));
    }

    __aicore__ inline LocalTensor<D_T> V3PipeGate(uint32_t slot, uint32_t count, uint32_t gateCapacity,
                                                  uint32_t capacity)
    {
        return ub_.GetWithOffset<D_T>(count, V3PipeGateBase(capacity) + slot * gateCapacity * sizeof(D_T));
    }
#else
    __aicore__ inline uint32_t V3PipeHalfBase() const
    {
        return ubBufBase_ + V3_PIPE_FLOAT_SLOTS * bufCount_ * sizeof(float);
    }

    __aicore__ inline uint32_t V3PipeGateBase() const
    {
        return V3PipeHalfBase() + V3_PIPE_HALF_SLOTS * bufCount_ * sizeof(D_T);
    }

    __aicore__ inline LocalTensor<D_T> V3PipeHalf(uint32_t slot, uint32_t count)
    {
        return ub_.GetWithOffset<D_T>(count, V3PipeHalfBase() + slot * bufCount_ * sizeof(D_T));
    }

    __aicore__ inline LocalTensor<D_T> V3PipeGate(uint32_t slot, uint32_t count, uint32_t capacity)
    {
        return ub_.GetWithOffset<D_T>(count, V3PipeGateBase() + slot * capacity * sizeof(D_T));
    }
#endif

    __aicore__ inline void IssueV3GateIn(GM_ADDR src, uint64_t token, uint32_t rows, const LocalTensor<D_T>& dst)
    {
        GlobalTensor<D_T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(src));
        // Separate GM rows become 32-byte UB rows, so the padded result can be
        // copied straight to the aligned gate columns without per-row vector work.
        DataCopyExtParams params{static_cast<uint16_t>(rows), V3_PIPE_GATE_COLS * sizeof(D_T), 0, 0, 0};
        DataCopyPadExtParams<D_T> pad{true, 0, static_cast<uint8_t>(HC_GRAD_GATE_ALIGN - V3_PIPE_GATE_COLS),
                                      static_cast<D_T>(0)};
        DataCopyPad(dst, source[token * V3_PIPE_GATE_COLS], params, pad);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    template <bool FUSE_ACT = false>
    __aicore__ inline void IssueV3In(uint32_t task, uint32_t base, uint32_t rows, uint32_t stage, uint32_t gateCapacity,
                                     uint32_t capacity, GM_ADDR dact)
    {
        const uint32_t count = rows * r_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        IssueCopyIn<float>(dact, static_cast<uint64_t>(base) * r_, V3PipeFloat(stage, count, capacity), count);
        if constexpr (FUSE_ACT) {
            // Only down reuses the buffer that the Act store still reads.
            WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        }
        IssueCopyIn<D_T>(gmDownPreAct_, token * r_, V3PipeHalf(stage, count, capacity), count);
        if (hasGate_ != 0U) {
            const uint32_t gateCount = rows * HC_GRAD_GATE_ALIGN;
            IssueV3GateIn(gmGradInject_, token, rows,
                          V3PipeGate(stage * V3_GATE_HALF_SLOTS, gateCount, gateCapacity, capacity));
            IssueV3GateIn(gmInject_, token, rows,
                          V3PipeGate(stage * V3_GATE_HALF_SLOTS + 1U, gateCount, gateCapacity, capacity));
        }
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
    }

    __aicore__ inline bool CanPipelineV3(uint32_t gateCapacity) const
    {
        // Enable the A5 path for the 320-column packed layout.
        if (r_ != 320U) {
            return false;
        }
        const uint32_t gateBytes =
            hasGate_ != 0U ? HC_GRAD_PIPE_MAX_STAGES * V3_GATE_HALF_SLOTS * gateCapacity * sizeof(D_T) : 0U;
        return c_ == V3_PIPE_GATE_COLS && r_ != 0U && r_ % HC_GRAD_VECTOR_ALIGN == 0U &&
               (hasGate_ == 0U || (gateRows_ == HC_GRAD_GATE_ALIGN && gateCapacity <= bufCount_)) &&
               V3PipeGateBase(bufCount_) + gateBytes <= GammaOffset();
    }

    __aicore__ inline bool CanFuseV3Act() const
    {
        // V1 may omit Act only when TryProcessV3FullBatch will produce it.
        return layout_.fullBatch && layout_.duplicateV1Operands && hasGate_ != 0U && h_ == 2560U &&
               splitkFactor_ == 1U && CanPipelineV3(V3BlockRows() * HC_GRAD_GATE_ALIGN);
    }

    __aicore__ inline uint32_t V3WideSigmoidOffset() const
    {
        return (ubBytes_ & ~HC_GRAD_DATA_BLOCK_MASK) - HC_GRAD_SIGMOID_SCRATCH;
    }

    __aicore__ inline void IssuePreparedG56A(uint32_t base, uint32_t rows, const LocalTensor<D_T>& down,
                                             const LocalTensor<D_T>& gate)
    {
        constexpr uint32_t C0 = HC_GRAD_DATA_BLOCK_BYTES / sizeof(D_T);
        static_assert(336U * sizeof(D_T) <= 320U * sizeof(float));
        GlobalTensor<D_T> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(Pointer(layout_.dact)), static_cast<uint64_t>(n_) * g56Rows_);
        const uint64_t row = static_cast<uint64_t>(base) * C0;
        const DataCopyExtParams copy{static_cast<uint16_t>(rows), HC_GRAD_DATA_BLOCK_BYTES,
                                     static_cast<HcGradCopyStride>(r_ / C0 - 1U), 0U, 0U};
        // Down and gate form one ZN matrix whose logical axes are output rows and tokens.
        const LoopModeParams loop{r_ / C0, 1U, HC_GRAD_DATA_BLOCK_BYTES, n_ * HC_GRAD_DATA_BLOCK_BYTES, 0U, 0U};
        SetLoopModePara(loop, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(dst[row], down, copy);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        DataCopy(dst[static_cast<uint64_t>(r_) * n_ + row], gate, rows * C0);
    }

    // The fused Act path has C=4 and R=320. Every row, including the
    // final tile, contains a whole number of FP32 vector repeats.
    static __simd_vf__ inline void A5V3DownAct(__ubuf__ float* dact, __ubuf__ D_T* downAct, __ubuf__ D_T* ddown,
                                               uint16_t repeats)
    {
        namespace Reg = AscendC::Reg;
        constexpr uint32_t FP32_LANES = 64U;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> zero, ones;
        Reg::Duplicate(zero, 0.0f);
        Reg::Duplicate(ones, 1.0f);
        for (uint16_t repeat = 0U; repeat < repeats; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(repeat) * FP32_LANES;
            Reg::RegTensor<float> grad, down, sigmoid, derivative;
            Reg::RegTensor<D_T> packed;
            Reg::LoadAlign(grad, dact + offset);
            Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, downAct + offset);
            Reg::Cast<float, D_T, A5_V6_B16_TO_FP32>(down, packed, mask);
            // Keep the same FP32 operations as the full-K dact sum, SigmoidFp32
            // and ComputeV3Down, including the +0 operand and operation order.
            Reg::Add(grad, zero, grad, mask);
            Reg::Muls(down, down, 0.25f, mask);
            Reg::Muls(sigmoid, down, -1.0f, mask);
            Reg::Exp(sigmoid, sigmoid, mask);
            Reg::Adds(sigmoid, sigmoid, 1.0f, mask);
            Reg::Div(sigmoid, ones, sigmoid, mask);
            Reg::Mul(grad, grad, sigmoid, mask);
            Reg::Muls(derivative, sigmoid, -1.0f, mask);
            Reg::Adds(derivative, derivative, 1.0f, mask);
            Reg::Mul(derivative, derivative, down, mask);
            Reg::Adds(derivative, derivative, 1.0f, mask);
            Reg::Mul(grad, grad, derivative, mask);
            Reg::Muls(grad, grad, 0.25f, mask);
            Reg::Cast<D_T, float, A5_V6_FP32_TO_B16>(packed, grad, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(ddown + offset, packed, mask);
            Reg::Mul(derivative, down, sigmoid, mask);
            Reg::Cast<D_T, float, A5_V6_FP32_TO_B16>(packed, derivative, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(downAct + offset, packed, mask);
        }
    }

    static constexpr uint32_t V3_UB_BLOCK_ROWS = 24U;
    static constexpr uint32_t V3_UB_HALF_ELEMS = V3_UB_BLOCK_ROWS * HC_GRAD_C1A_UB_TILE_N;
    static constexpr uint32_t V3_UB_GATE_ELEMS = V3_UB_BLOCK_ROWS * HC_GRAD_GATE_ALIGN;
    static constexpr uint32_t V3_UB_HALF_BYTES = V3_PIPE_HALF_SLOTS * V3_UB_HALF_ELEMS * sizeof(D_T);
    static constexpr uint32_t V3_UB_GATE_BYTES =
        HC_GRAD_PIPE_MAX_STAGES * V3_GATE_HALF_SLOTS * V3_UB_GATE_ELEMS * sizeof(D_T);
    static_assert(HC_GRAD_DGAMMA_SLOTS * 2560U * sizeof(float) + V3_UB_HALF_BYTES + V3_UB_GATE_BYTES +
                      3U * V3_UB_GATE_ELEMS * sizeof(float) <=
                  HC_GRAD_C1A_UB_OFFSET);

    __aicore__ inline LocalTensor<D_T> V3UbHalf(uint32_t slot, uint32_t count)
    {
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + slot * V3_UB_HALF_ELEMS * sizeof(D_T));
    }

    __aicore__ inline LocalTensor<D_T> V3UbGate(uint32_t slot, uint32_t count)
    {
        const uint32_t offset = ubBufBase_ + V3_UB_HALF_BYTES + slot * V3_UB_GATE_ELEMS * sizeof(D_T);
        return ub_.GetWithOffset<D_T>(count, offset);
    }

    __aicore__ inline LocalTensor<float> V3UbGateFloat(uint32_t slot, uint32_t count)
    {
        const uint32_t offset =
            ubBufBase_ + V3_UB_HALF_BYTES + V3_UB_GATE_BYTES + slot * V3_UB_GATE_ELEMS * sizeof(float);
        return ub_.GetWithOffset<float>(count, offset);
    }

    __aicore__ inline void IssueV3UbIn(uint32_t base, uint32_t rows, uint32_t stage)
    {
        const uint32_t count = rows * r_;
        const uint64_t token = base;
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        IssueCopyIn<D_T>(gmDownPreAct_, token * r_, V3UbHalf(stage, count), count);
        if (hasGate_ != 0U) {
            const uint32_t gateCount = rows * HC_GRAD_GATE_ALIGN;
            const uint32_t slot = stage * V3_GATE_HALF_SLOTS;
            IssueV3GateIn(gmGradInject_, token, rows, V3UbGate(slot, gateCount));
            IssueV3GateIn(gmInject_, token, rows, V3UbGate(slot + 1U, gateCount));
        }
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
    }

    __aicore__ inline void ProcessV3Ub(uint32_t begin, uint32_t end)
    {
        constexpr uint32_t block = V3_UB_BLOCK_ROWS;
        GM_ADDR act = TaskPointer(layout_.act, 0U, r_, sizeof(D_T));
        GM_ADDR packed = TaskPointer(layout_.packed, 0U, g56Rows_, sizeof(D_T));
        for (uint32_t stage = 0; stage < HC_GRAD_PIPE_MAX_STAGES; ++stage) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        }
        uint32_t stage = 0U;
        uint32_t rows = (end - begin < block) ? end - begin : block;
        IssueV3UbIn(begin, rows, stage);
        for (uint32_t base = begin; base < end;) {
            const uint32_t next = base + rows;
            const uint32_t nextRows = (end - next < block) ? end - next : block;
            const uint32_t nextStage = stage ^ HC_GRAD_PIPE_STAGE_MASK;
            if (nextRows != 0U) {
                IssueV3UbIn(next, nextRows, nextStage);
            }
            const uint32_t count = rows * r_;
            auto ddownHalf = V3UbHalf(stage + HC_GRAD_PIPE_MAX_STAGES, count);
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            auto directDact =
                ub_.GetWithOffset<float>(count, HC_GRAD_C1A_UB_OFFSET + (base - begin) * r_ * sizeof(float));
            auto downHalf = V3UbHalf(stage, count);
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            asc_vf_call<A5V3DownAct>(reinterpret_cast<__ubuf__ float*>(directDact.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ D_T*>(downHalf.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ D_T*>(ddownHalf.GetPhyAddr()),
                                     static_cast<uint16_t>(count / 64U));
            const uint32_t gateCount = rows * HC_GRAD_GATE_ALIGN;
            const uint32_t gateSlot = stage * V3_GATE_HALF_SLOTS;
            if (hasGate_ != 0U) {
                auto grad = V3UbGateFloat(0U, gateCount);
                ComputeV3Gate(grad, V3UbGateFloat(1U, gateCount), V3UbGateFloat(2U, gateCount),
                              V3UbGate(gateSlot, gateCount), V3UbGate(gateSlot + 1U, gateCount), gateCount);
                Cast(V3UbGate(gateSlot + 2U, gateCount), grad, RoundMode::CAST_RINT, gateCount);
            }
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            auto actHalf = V3UbHalf(stage, count);
            IssueCopyOutRows<D_T>(act, static_cast<uint64_t>(base) * r_, count, actHalf, 1U, count);
            SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
            const uint64_t offset = static_cast<uint64_t>(base) * g56Rows_;
            IssueCopyOutRows<D_T>(packed, offset, g56Rows_, ddownHalf, rows, r_);
            if (hasGate_ != 0U) {
                IssueCopyOutRows<D_T>(packed, offset + r_, g56Rows_, V3UbGate(gateSlot + 2U, gateCount), rows,
                                      HC_GRAD_GATE_ALIGN);
                if (preparedG56A_) {
                    IssuePreparedG56A(base, rows, ddownHalf, V3UbGate(gateSlot + 2U, gateCount));
                }
            }
            if (layout_.denseDx != 0U) {
                const uint32_t columns = DenseDxColumns();
                const uint64_t denseOffset = static_cast<uint64_t>(base) * columns;
                GM_ADDR dense = DenseDxTaskPointer(0U);
                IssueCopyOutRows<D_T>(dense, denseOffset, columns, ddownHalf, rows, r_);
                if (hasGate_ != 0U) {
                    IssueCopyOutRows<D_T>(dense, denseOffset + r_, columns, V3UbGate(gateSlot + 2U, gateCount), rows,
                                          c_);
                }
            }
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            base = next;
            rows = nextRows;
            stage = nextStage;
        }
        for (uint32_t i = 0; i < HC_GRAD_PIPE_MAX_STAGES; ++i) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(i));
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(i));
            WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(i));
        }
    }

    template <bool FUSE_DACT_SUM = false, bool FUSE_ACT = false>
    __aicore__ inline void ProcessV3Pipe(uint32_t task, uint32_t begin, uint32_t end, uint32_t block,
                                         uint32_t gateCapacity, GM_ADDR dact, GM_ADDR packed)
    {
        const bool wideTile = block * r_ > bufCount_;
        const uint32_t capacity = wideTile ? block * r_ : bufCount_;
        const uint32_t sigmoidScratch = wideTile ? V3WideSigmoidOffset() : 0U;
        GM_ADDR act = FUSE_ACT ? TaskPointer(layout_.act, task, r_, sizeof(D_T)) : nullptr;
        for (uint32_t stage = 0; stage < HC_GRAD_PIPE_MAX_STAGES; ++stage) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            if constexpr (FUSE_ACT) {
                SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
            }
        }
        uint32_t stage = 0U;
        uint32_t rows = (end - begin < block) ? end - begin : block;
        IssueV3In<FUSE_ACT>(task, begin, rows, stage, gateCapacity, capacity, dact);
        for (uint32_t base = begin; base < end;) {
            const uint32_t next = base + rows;
            const uint32_t nextRows = (end - next < block) ? end - next : block;
            const uint32_t nextStage = stage ^ HC_GRAD_PIPE_STAGE_MASK;
            if (nextRows != 0U) {
                IssueV3In<FUSE_ACT>(task, next, nextRows, nextStage, gateCapacity, capacity, dact);
            }
            const uint32_t count = rows * r_;
            auto dactFloat = V3PipeFloat(stage, count, capacity);
            auto ddownHalf = V3PipeHalf(stage + HC_GRAD_PIPE_MAX_STAGES, count, capacity);
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            if constexpr (FUSE_DACT_SUM) {
                // The full-K C1a partial is the only reduction input. Preserve
                // its addition to +0, including signed-zero behavior.
                auto zero = V3PipeFloat(4U, count, capacity);
                Duplicate(zero, 0.0f, count);
                Add(dactFloat, zero, dactFloat, count);
            }
            auto down = V3PipeFloat(2U, count, capacity);
            auto sigmoid = V3PipeFloat(3U, count, capacity);
            auto derivative = V3PipeFloat(4U, count, capacity);
            auto downHalf = V3PipeHalf(stage, count, capacity);
            ComputeV3Down(dactFloat, down, sigmoid, derivative, downHalf, count, sigmoidScratch);
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            Cast(ddownHalf, dactFloat, RoundMode::CAST_RINT, count);
            if constexpr (FUSE_ACT) {
                // Both operands are exactly those of V1 Act. Reuse the dead derivative and down input slots.
                Mul(derivative, down, sigmoid, count);
                Cast(downHalf, derivative, RoundMode::CAST_RINT, count);
            }
            const uint32_t gateCount = rows * HC_GRAD_GATE_ALIGN;
            const uint32_t gateSlot = stage * V3_GATE_HALF_SLOTS;
            if (hasGate_ != 0U) {
                auto grad = V3PipeFloat(2U, gateCount, capacity);
                ComputeV3Gate(grad, V3PipeFloat(3U, gateCount, capacity), V3PipeFloat(4U, gateCount, capacity),
                              V3PipeGate(gateSlot, gateCount, gateCapacity, capacity),
                              V3PipeGate(gateSlot + 1U, gateCount, gateCapacity, capacity), gateCount);
                Cast(V3PipeGate(gateSlot + 2U, gateCount, gateCapacity, capacity), grad, RoundMode::CAST_RINT,
                     gateCount);
            }
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            if constexpr (FUSE_ACT) {
                IssueCopyOutRows<D_T>(act, static_cast<uint64_t>(base) * r_, count, downHalf, 1U, count);
                SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
            }
            const uint64_t offset = static_cast<uint64_t>(base) * g56Rows_;
            IssueCopyOutRows<D_T>(packed, offset, g56Rows_, ddownHalf, rows, r_);
            if (hasGate_ != 0U) {
                IssueCopyOutRows<D_T>(packed, offset + r_, g56Rows_,
                                      V3PipeGate(gateSlot + 2U, gateCount, gateCapacity, capacity), rows,
                                      HC_GRAD_GATE_ALIGN);
                if constexpr (FUSE_DACT_SUM) {
                    if (preparedG56A_) {
                        IssuePreparedG56A(base, rows, ddownHalf,
                                          V3PipeGate(gateSlot + 2U, gateCount, gateCapacity, capacity));
                    }
                }
            }
            if (layout_.denseDx != 0U) {
                const uint32_t columns = DenseDxColumns();
                const uint64_t denseOffset = static_cast<uint64_t>(base) * columns;
                GM_ADDR dense = DenseDxTaskPointer(task);
                IssueCopyOutRows<D_T>(dense, denseOffset, columns, ddownHalf, rows, r_);
                if (hasGate_ != 0U) {
                    IssueCopyOutRows<D_T>(dense, denseOffset + r_, columns,
                                          V3PipeGate(gateSlot + 2U, gateCount, gateCapacity, capacity), rows, c_);
                }
            }
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            base = next;
            rows = nextRows;
            stage = nextStage;
        }
        for (uint32_t i = 0; i < HC_GRAD_PIPE_MAX_STAGES; ++i) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(i));
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(i));
            if constexpr (FUSE_ACT) {
                WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(i));
            }
        }
    }
#else
    __aicore__ inline void IssueV3In(uint32_t task, uint32_t base, uint32_t rows, uint32_t stage, uint32_t gateCapacity,
                                     GM_ADDR dact)
    {
        const uint32_t count = rows * r_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        IssueCopyIn<float>(dact, static_cast<uint64_t>(base) * r_, FloatBuf(stage, count), count);
        IssueCopyIn<D_T>(gmDownPreAct_, token * r_, V3PipeHalf(stage, count), count);
        if (hasGate_ != 0U) {
            const uint32_t gateCount = rows * HC_GRAD_GATE_ALIGN;
            IssueV3GateIn(gmGradInject_, token, rows, V3PipeGate(stage * V3_GATE_HALF_SLOTS, gateCount, gateCapacity));
            IssueV3GateIn(gmInject_, token, rows, V3PipeGate(stage * V3_GATE_HALF_SLOTS + 1U, gateCount, gateCapacity));
        }
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
    }

    __aicore__ inline bool CanPipelineV3(uint32_t gateCapacity) const
    {
        const uint32_t gateBytes =
            hasGate_ != 0U ? HC_GRAD_PIPE_MAX_STAGES * V3_GATE_HALF_SLOTS * gateCapacity * sizeof(D_T) : 0U;
        return c_ == V3_PIPE_GATE_COLS && r_ != 0U && r_ % HC_GRAD_VECTOR_ALIGN == 0U &&
               (hasGate_ == 0U || (gateRows_ == HC_GRAD_GATE_ALIGN && gateCapacity <= bufCount_)) &&
               V3PipeGateBase() + gateBytes <= GammaOffset();
    }

    template <bool FUSE_DACT_SUM = false>
    __aicore__ inline void ProcessV3Pipe(uint32_t task, uint32_t begin, uint32_t end, uint32_t block,
                                         uint32_t gateCapacity, GM_ADDR dact, GM_ADDR packed)
    {
        for (uint32_t stage = 0; stage < HC_GRAD_PIPE_MAX_STAGES; ++stage) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        uint32_t stage = 0U;
        uint32_t rows = (end - begin < block) ? end - begin : block;
        IssueV3In(task, begin, rows, stage, gateCapacity, dact);
        for (uint32_t base = begin; base < end;) {
            const uint32_t next = base + rows;
            const uint32_t nextRows = (end - next < block) ? end - next : block;
            const uint32_t nextStage = stage ^ HC_GRAD_PIPE_STAGE_MASK;
            if (nextRows != 0U) {
                IssueV3In(task, next, nextRows, nextStage, gateCapacity, dact);
            }
            const uint32_t count = rows * r_;
            auto dactFloat = FloatBuf(stage, count);
            auto ddownHalf = V3PipeHalf(stage + HC_GRAD_PIPE_MAX_STAGES, count);
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            if constexpr (FUSE_DACT_SUM) {
                // The full-K C1a partial is the only reduction input. Preserve
                // its addition to +0, including signed-zero behavior.
                auto zero = FloatBuf(4U, count);
                Duplicate(zero, 0.0f, count);
                Add(dactFloat, zero, dactFloat, count);
            }
            ComputeV3Down(dactFloat, FloatBuf(2U, count), FloatBuf(3U, count), FloatBuf(4U, count),
                          V3PipeHalf(stage, count), count);
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            Cast(ddownHalf, dactFloat, RoundMode::CAST_RINT, count);
            const uint32_t gateCount = rows * HC_GRAD_GATE_ALIGN;
            const uint32_t gateSlot = stage * V3_GATE_HALF_SLOTS;
            if (hasGate_ != 0U) {
                auto grad = FloatBuf(2U, gateCount);
                ComputeV3Gate(grad, FloatBuf(3U, gateCount), FloatBuf(4U, gateCount),
                              V3PipeGate(gateSlot, gateCount, gateCapacity),
                              V3PipeGate(gateSlot + 1U, gateCount, gateCapacity), gateCount);
                Cast(V3PipeGate(gateSlot + 2U, gateCount, gateCapacity), grad, RoundMode::CAST_RINT, gateCount);
            }
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            const uint64_t offset = static_cast<uint64_t>(base) * g56Rows_;
            IssueCopyOutRows<D_T>(packed, offset, g56Rows_, ddownHalf, rows, r_);
            if (hasGate_ != 0U) {
                IssueCopyOutRows<D_T>(packed, offset + r_, g56Rows_, V3PipeGate(gateSlot + 2U, gateCount, gateCapacity),
                                      rows, HC_GRAD_GATE_ALIGN);
            }
            if (layout_.denseDx != 0U) {
                const uint32_t columns = DenseDxColumns();
                const uint64_t denseOffset = static_cast<uint64_t>(base) * columns;
                GM_ADDR dense = DenseDxTaskPointer(task);
                IssueCopyOutRows<D_T>(dense, denseOffset, columns, ddownHalf, rows, r_);
                if (hasGate_ != 0U) {
                    IssueCopyOutRows<D_T>(dense, denseOffset + r_, columns,
                                          V3PipeGate(gateSlot + 2U, gateCount, gateCapacity), rows, c_);
                }
            }
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            base = next;
            rows = nextRows;
            stage = nextStage;
        }
        for (uint32_t i = 0; i < HC_GRAD_PIPE_MAX_STAGES; ++i) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(i));
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(i));
        }
    }
#endif
#endif

    __aicore__ inline uint32_t V3BlockRows() const
    {
        const uint32_t blocks = bufCount_ / r_;
        return blocks != 0U ? blocks : 1U;
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline uint32_t SelectA5V3BlockRows(uint32_t block, uint32_t rows)
    {
        constexpr uint32_t WIDE_ROWS = 24U;
        constexpr uint32_t WIDE_GATE_CAPACITY = WIDE_ROWS * HC_GRAD_GATE_ALIGN;
        if (block < WIDE_ROWS && rows >= WIDE_ROWS) {
            const uint32_t bytes =
                V3PipeGateBase(WIDE_ROWS * r_) +
                (hasGate_ != 0U ? HC_GRAD_PIPE_MAX_STAGES * V3_GATE_HALF_SLOTS * WIDE_GATE_CAPACITY * sizeof(D_T) : 0U);
            if (bytes <= V3WideSigmoidOffset()) {
                // V3 may reuse the gamma cache while preserving the dgamma sum and compensation.
                gammaCachedCh_ = -1;
                return WIDE_ROWS;
            }
        }
        return block;
    }
#endif

    __aicore__ inline void ProcessV3(uint32_t task, uint32_t rows)
    {
        GM_ADDR dact = TaskPointer(layout_.dact, task, r_, sizeof(float));
        GM_ADDR packed = TaskPointer(layout_.packed, task, g56Rows_, sizeof(D_T));
        // As in ReduceDact, each core consumes the rows it produced.
        const uint32_t rowsPerCoreV3 = (rows + aivNum_ - 1U) / aivNum_;
        const uint32_t rowBeginV3 = aivId_ * rowsPerCoreV3;
        const uint32_t rowEndV3 = (rowBeginV3 + rowsPerCoreV3 < rows) ? (rowBeginV3 + rowsPerCoreV3) : rows;
        // dact and down_pre_act are contiguous across n rows; packed uses a g56Rows row stride.
        const uint32_t blk = V3BlockRows();
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (rowBeginV3 >= rowEndV3) {
            return;
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (CanPipelineV3(blk * HC_GRAD_GATE_ALIGN)) {
            const uint32_t block = SelectA5V3BlockRows(blk, rowEndV3 - rowBeginV3);
            ProcessV3Pipe(task, rowBeginV3, rowEndV3, block, block * HC_GRAD_GATE_ALIGN, dact, packed);
            return;
        }
#else
        const uint32_t gateCapacity = blk * HC_GRAD_GATE_ALIGN;
        if (CanPipelineV3(gateCapacity)) {
            ProcessV3Pipe(task, rowBeginV3, rowEndV3, blk, gateCapacity, dact, packed);
            return;
        }
#endif
#endif
        for (uint32_t base = rowBeginV3; base < rowEndV3; base += blk) {
            const uint32_t nb = (rowEndV3 - base < blk) ? (rowEndV3 - base) : blk;
            const uint32_t len = nb * r_;
            const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
            auto dactFloat = FloatBuf(HC_GRAD_V3_DACT, len);
            auto down = FloatBuf(HC_GRAD_V3_DOWN, len);
            auto sigmoid = FloatBuf(HC_GRAD_V3_SIGMOID, len);
            auto derivative = FloatBuf(HC_GRAD_V3_DERIVATIVE, len);
            auto downHalf = HalfBuf(HC_GRAD_HALF_INPUT, len);
            auto ddownHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
            // No Vector work occurs between these independent input copies.
            CopyIn<float>(dact, static_cast<uint64_t>(base) * r_, dactFloat, len, false);
            IssueCopyIn<D_T>(gmDownPreAct_, token * r_, downHalf, len);
            SyncCopy();
            ComputeV3Down(dactFloat, down, sigmoid, derivative, downHalf, len);
            Cast(ddownHalf, dactFloat, RoundMode::CAST_RINT, len);
            CopyOutRows<D_T>(packed, static_cast<uint64_t>(base) * g56Rows_, g56Rows_, ddownHalf, nb, r_);
            if (layout_.denseDx != 0U) {
                const uint32_t columns = DenseDxColumns();
                CopyOutRows<D_T>(DenseDxTaskPointer(task), static_cast<uint64_t>(base) * columns, columns, ddownHalf,
                                 nb, r_);
            }
            if (hasGate_ != 0) {
                ProcessGateLogits(task, base, nb, packed);
            }
        }
    }

    __aicore__ inline void TransposeDupRows(GlobalTensor<D_T>& source, LocalTensor<D_T> accum, LocalTensor<D_T> tile,
                                            LocalTensor<D_T> transposed, uint32_t rowBegin, uint32_t rowEnd,
                                            uint32_t rows, uint32_t columnBegin, uint32_t validCols)
    {
        DataCopyPadExtParams<D_T> pad{false, 0, 0, 0};
        for (uint32_t rowTile = rowBegin; rowTile < rowEnd; ++rowTile) {
            const uint32_t rowBase = rowTile * HC_GRAD_TRANSPOSE_TILE;
            const uint32_t validRows =
                (rows - rowBase < HC_GRAD_TRANSPOSE_TILE) ? (rows - rowBase) : HC_GRAD_TRANSPOSE_TILE;
            DataCopyExtParams inParams{1, static_cast<uint32_t>(validCols * sizeof(D_T)), 0, 0, 0};
            for (uint32_t tileRow = 0; tileRow < validRows; ++tileRow) {
                DataCopyPad(tile[tileRow * HC_GRAD_TRANSPOSE_TILE],
                            source[static_cast<uint64_t>(rowBase + tileRow) * d_ + columnBegin], inParams, pad);
            }
            SetFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
            Transpose16(transposed, tile);
            SetFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
            // Copy complete 16-element groups. Tail padding fits the 1024-row accumulator and is never stored to the
            // output.
            for (uint32_t i = 0; i < validCols; ++i) {
                DataCopy(accum[i * taskRows_ + rowBase], transposed[i * HC_GRAD_TRANSPOSE_TILE],
                         HC_GRAD_TRANSPOSE_TILE);
            }
        }
    }

    __aicore__ inline void StoreTransposeRows(GlobalTensor<D_T>& target, LocalTensor<D_T> accum, uint32_t rowBegin,
                                              uint32_t rowEnd, uint32_t rows, uint32_t columnBegin, uint32_t validCols,
                                              uint32_t targetStride)
    {
        SetFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_MTE3>(HC_GRAD_VECTOR_EVENT);
        uint32_t validCount = (rowEnd - rowBegin) * HC_GRAD_TRANSPOSE_TILE;
        if (validCount > rows - rowBegin * HC_GRAD_TRANSPOSE_TILE) {
            validCount = rows - rowBegin * HC_GRAD_TRANSPOSE_TILE;
        }
        DataCopyExtParams outParams{1, static_cast<uint32_t>(validCount * sizeof(D_T)), 0, 0, 0};
        for (uint32_t i = 0; i < validCols; ++i) {
            DataCopyPad(
                target[static_cast<uint64_t>(columnBegin + i) * targetStride + rowBegin * HC_GRAD_TRANSPOSE_TILE],
                accum[i * taskRows_ + rowBegin * HC_GRAD_TRANSPOSE_TILE], outParams);
        }
        SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
    }

    // Transpose fallback: convert dup [rows,D] to [D,rows] with a fixed TASK_ROWS row stride.
    // C1b reads the result as RowMajor. Both architectures use direct ColumnMajor reads by default.
    // Process 16x16 tiles: DataCopyPad gathers input rows, Transpose swaps axes,
    // and the output transfer gathers the corresponding columns.
    __aicore__ inline void TransposeDup(uint32_t task, uint32_t rows)
    {
        GM_ADDR src = TaskPointer(layout_.dup, task, d_, sizeof(D_T));
        GM_ADDR dst = TaskPointer(layout_.dupT, task, d_, sizeof(D_T));
        GlobalTensor<D_T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(src));
        GlobalTensor<D_T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(dst));
        const uint32_t columnTiles = (d_ + HC_GRAD_TRANSPOSE_TILE - 1U) / HC_GRAD_TRANSPOSE_TILE;
        const uint32_t rowTiles = (rows + HC_GRAD_TRANSPOSE_TILE - 1U) / HC_GRAD_TRANSPOSE_TILE;
        uint32_t groups = aivNum_ / columnTiles;
        if (groups == 0U) {
            groups = 1U;
        }
        if (groups > rowTiles) {
            groups = rowTiles;
        }
        const uint32_t tilesPerGroup = (rowTiles + groups - 1U) / groups;
        const uint32_t itemCount = columnTiles * groups;
        const uint32_t accumOffset = ubBytes_ - HC_GRAD_TRANSPOSE_SCRATCH;
        auto accum = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * taskRows_, accumOffset);
        auto tile = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                           accumOffset + HC_GRAD_TRANSPOSE_TILE * taskRows_ * sizeof(D_T));
        auto transposed = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                                 accumOffset + HC_GRAD_TRANSPOSE_TILE * taskRows_ * sizeof(D_T) +
                                                     HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE * sizeof(D_T));
        for (uint32_t item = aivId_; item < itemCount; item += aivNum_) {
            const uint32_t columnTile = item / groups;
            const uint32_t group = item - columnTile * groups;
            const uint32_t columnBegin = columnTile * HC_GRAD_TRANSPOSE_TILE;
            const uint32_t validCols =
                (d_ - columnBegin < HC_GRAD_TRANSPOSE_TILE) ? (d_ - columnBegin) : HC_GRAD_TRANSPOSE_TILE;
            const uint32_t rowBegin = group * tilesPerGroup;
            const uint32_t rowEnd = (rowBegin + tilesPerGroup) < rowTiles ? (rowBegin + tilesPerGroup) : rowTiles;
            if (rowBegin >= rowEnd) {
                continue;
            }
            TransposeDupRows(source, accum, tile, transposed, rowBegin, rowEnd, rows, columnBegin, validCols);
            StoreTransposeRows(target, accum, rowBegin, rowEnd, rows, columnBegin, validCols, taskRows_);
        }
    }

    __aicore__ inline void TransposePackedRows(GlobalTensor<D_T>& source, LocalTensor<D_T> accum, LocalTensor<D_T> tile,
                                               LocalTensor<D_T> transposed, uint32_t rowBegin, uint32_t rowEnd,
                                               uint32_t columnTile)
    {
        DataCopyPadExtParams<D_T> pad{false, 0, 0, 0};
        for (uint32_t rowTile = rowBegin; rowTile < rowEnd; ++rowTile) {
            DataCopyExtParams blockParams{
                HC_GRAD_TRANSPOSE_TILE, static_cast<uint32_t>(HC_GRAD_TRANSPOSE_TILE * sizeof(D_T)),
                static_cast<HcGradCopyStride>((g56Rows_ - HC_GRAD_TRANSPOSE_TILE) * sizeof(D_T)), 0, 0};
            DataCopyPad(tile,
                        source[static_cast<uint64_t>(rowTile) * HC_GRAD_TRANSPOSE_TILE * g56Rows_ +
                               columnTile * HC_GRAD_TRANSPOSE_TILE],
                        blockParams, pad);
            SetFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::MTE2_V>(HC_GRAD_VECTOR_EVENT);
            Transpose16(transposed, tile);
            SetFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
            for (uint32_t i = 0; i < HC_GRAD_TRANSPOSE_TILE; ++i) {
                const uint32_t rowOffset = rowTile * HC_GRAD_TRANSPOSE_TILE;
                // The five-argument A2 Copy operates on groups of 16 elements (32 bytes). Tail padding fits
                // the 1024-row accumulator and is never read when writing the output.
                DataCopy(accum[i * taskRows_ + rowOffset], transposed[i * HC_GRAD_TRANSPOSE_TILE],
                         HC_GRAD_TRANSPOSE_TILE);
            }
        }
    }

    __aicore__ inline void TransposePacked(uint32_t task, uint32_t rows)
    {
        GM_ADDR src = TaskPointer(layout_.packed, task, g56Rows_, sizeof(D_T));
        GM_ADDR dst = TaskPointer(layout_.packedT, task, g56Rows_, sizeof(D_T));
        GlobalTensor<D_T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(src));
        GlobalTensor<D_T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(dst));
        const uint32_t columnTiles = g56Rows_ / HC_GRAD_TRANSPOSE_TILE;
        const uint32_t rowTiles = (rows + HC_GRAD_TRANSPOSE_TILE - 1U) / HC_GRAD_TRANSPOSE_TILE;
        uint32_t groups = aivNum_ / columnTiles;
        if (groups == 0U) {
            groups = 1U;
        }
        if (groups > rowTiles) {
            groups = rowTiles;
        }
        const uint32_t tilesPerGroup = (rowTiles + groups - 1U) / groups;
        const uint32_t itemCount = columnTiles * groups;
        const uint32_t accumOffset = ubBytes_ - HC_GRAD_TRANSPOSE_SCRATCH;
        auto accum = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * taskRows_, accumOffset);
        auto tile = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                           accumOffset + HC_GRAD_TRANSPOSE_TILE * taskRows_ * sizeof(D_T));
        auto transposed = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                                 accumOffset + HC_GRAD_TRANSPOSE_TILE * taskRows_ * sizeof(D_T) +
                                                     HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE * sizeof(D_T));
        for (uint32_t item = aivId_; item < itemCount; item += aivNum_) {
            const uint32_t columnTile = item / groups;
            const uint32_t group = item - columnTile * groups;
            const uint32_t rowBegin = group * tilesPerGroup;
            const uint32_t rowEnd = (rowBegin + tilesPerGroup) < rowTiles ? (rowBegin + tilesPerGroup) : rowTiles;
            if (rowBegin >= rowEnd) {
                continue;
            }
            TransposePackedRows(source, accum, tile, transposed, rowBegin, rowEnd, columnTile);
            StoreTransposeRows(target, accum, rowBegin, rowEnd, rows, columnTile * HC_GRAD_TRANSPOSE_TILE,
                               HC_GRAD_TRANSPOSE_TILE, rows);
        }
    }

    __aicore__ inline void ComputeV6Gamma(uint32_t n, uint32_t channel, LocalTensor<float> total, LocalTensor<float> x,
                                          LocalTensor<float> gn, LocalTensor<float> product,
                                          LocalTensor<float> correction, LocalTensor<float> reduceWork)
    {
        if constexpr (HAS_GAMMA) {
            EnsureGammaCache(channel);
            if (useGammaCache_) {
                auto gammaCache = ub_.GetWithOffset<float>(h_, GammaOffset());
                for (uint32_t r = 0; r < n; ++r) {
                    const uint32_t off = r * h_;
                    Mul(gn[off], total[off], gammaCache, h_);
                }
            } else {
                // F4 holds gamma until gn is computed, then becomes scratch storage.
                auto gamma = FloatBuf(HC_GRAD_V6_SERIAL_CORRECTION, h_);
                CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_, gamma, h_);
                Adds(gamma, gamma, 1.0f, h_);
                for (uint32_t r = 0; r < n; ++r) {
                    const uint32_t off = r * h_;
                    Mul(gn[off], total[off], gamma, h_);
                }
            }
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t off = r * h_;
                Mul(product[off], x[off], total[off], h_);
                DgammaAccumAdd(product[off], h_, 0U, correction, reduceWork);
            }
        } else {
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t off = r * h_;
                Adds(gn[off], total[off], 0.0f, h_);
            }
        }
    }

    __aicore__ inline void ProjectV6Rows(uint32_t n, uint32_t channel, LocalTensor<float> x, LocalTensor<float> gn,
                                         LocalTensor<float> product, LocalTensor<float> reduceWork,
                                         LocalTensor<float> invBuf)
    {
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            Mul(product[off], gn[off], x[off], h_);
            ReduceRmsProjection(SumSlot(r), product[off], reduceWork[off], h_);
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
#else
        // A5 auto-sync already fences the following GetValue reads.
#endif
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            ReduceRmsResidual(product[off], reduceWork[off], r);
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
#else
        // A5 auto-sync already fences the following GetValue reads.
#endif
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            const float inv = invBuf.GetValue(r * c_ + channel);
            const float sum = SumSlot(r).GetValue(0);
            const float mean = sum / static_cast<float>(static_cast<int32_t>(h_));
            const float residual = ResidualSlot(r).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            Axpy(gn[off], x[off], -mean, h_);
            if (residual != 0.0f) {
                Axpy(gn[off], x[off], -residual, h_);
            }
            Muls(gn[off], gn[off], inv, h_);
        }
        SetFlag<HardEvent::S_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::S_V>(HC_GRAD_VECTOR_EVENT);
    }

    // For n rows of one channel, use nBurst for four inputs and one output. Reduce row sums first,
    // then centered residuals, with two V_S handshakes per block.
    __aicore__ inline void ProcessV6Block(uint32_t task, uint32_t rowBegin, uint32_t n, uint32_t channel)
    {
        const uint32_t len = n * h_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + rowBegin;
        const uint64_t inputOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto total = InFloat(0U, HC_GRAD_V6_DIRECT_INPUT, len);
        auto x = FloatBuf(HC_GRAD_V6_SERIAL_X, len);
        auto gn = FloatBuf(HC_GRAD_V6_SERIAL_GN, len);
        auto product = FloatBuf(HC_GRAD_V6_SERIAL_PRODUCT, len);
        auto correction = FloatBuf(HC_GRAD_V6_SERIAL_CORRECTION, len);
        auto reduceWork = FloatBuf(HC_GRAD_V6_SERIAL_REDUCE_WORK, len);
        auto dxHalf = HalfBuf(HC_GRAD_HALF_SECOND_INPUT, len);
        auto xHalf = InHalf(0U, len);
        // After adding dx2 into total, reuse its storage for the product and correction.
        auto dx2f = InFloat(0U, HC_GRAD_V6_DX2_INPUT, len);
        if (recomputeDirect_) {
            auto gmHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
            auto upHalf = dxHalf;
            CopyIn<D_T>(gmGradMixed_, token * h_, gmHalf, len, false);
            IssueCopyInRows<D_T>(gmUp_, inputOffset, d_, upHalf, n, h_);
        } else {
            CopyInRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)),
                              static_cast<uint64_t>(rowBegin) * d_ + static_cast<uint64_t>(channel) * h_, d_, total, n,
                              h_, false);
        }
        if (lowpDx_) {
            CopyInRows<D_T>(Dx2TaskPointer(task),
                            static_cast<uint64_t>(rowBegin) * d_ + static_cast<uint64_t>(channel) * h_, d_, dxHalf, n,
                            h_);
            Cast(dx2f, dxHalf, RoundMode::CAST_NONE, len);
        } else {
            IssueCopyInRows<float>(TaskPointer(layout_.dx2, task, d_, sizeof(float)),
                                   static_cast<uint64_t>(rowBegin) * d_ + static_cast<uint64_t>(channel) * h_, d_, dx2f,
                                   n, h_);
        }
        IssueCopyInRows<D_T>(gmX_, token * d_ + static_cast<uint64_t>(channel) * h_, d_, xHalf, n, h_);
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset(0U));
        IssueCopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_);
        SyncCopy();
        if (recomputeDirect_) {
            auto gmHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
            auto upHalf = dxHalf;
            Cast(total, gmHalf, RoundMode::CAST_NONE, len);
            Cast(x, upHalf, RoundMode::CAST_NONE, len);
            SigmoidFp32(x, x, len);
            Mul(total, total, x, len);
            ScaleByInvC(total, total, len);
        }
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
        Add(total, total, dx2f, len); // Inline the FP32 Cube-to-Vector dxnorm consumer.
        SetFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
        // Compute xhat = x * inv_rms in FP32 and apply the RMS projection in normalized coordinates.
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            const float inv = invBuf.GetValue(r * c_ + channel);
            Muls(x[off], x[off], inv, h_);
        }
        ComputeV6Gamma(n, channel, total, x, gn, product, correction, reduceWork);
        ProjectV6Rows(n, channel, x, gn, product, reduceWork, invBuf);
        Cast(dxHalf, gn, RoundMode::CAST_RINT, len);
        CopyOutRows<D_T>(gmDx_, inputOffset, d_, dxHalf, n, h_);
    }

    __aicore__ inline void ProcessV6(uint32_t task, uint32_t rows)
    {
        for (uint32_t channel = aivId_ % c_; channel < c_; channel += aivNum_) {
            const bool firstWave = channel < aivNum_;
            const uint32_t lane = firstWave ? aivId_ / c_ : 0U;
            const uint32_t laneCount = firstWave ? (aivNum_ - 1U - channel) / c_ + 1U : 1U;
            const uint32_t rowsPerCore = (rows + laneCount - 1U) / laneCount;
            const uint32_t rowBegin = lane * rowsPerCore;
            const uint32_t rowEnd = (rowBegin + rowsPerCore < rows) ? (rowBegin + rowsPerCore) : rows;
            // The V6 row pipeline has no scratch slot for 1+gamma. Without a cache, fall back to the serial path;
            // this only affects the single-task case, where pipelining offers limited benefit.
            if (pipeStages_ != 0U && ((HAS_GAMMA == 0) || useGammaCache_)) {
                ProcessV6Pipe(task, rowBegin, rowEnd, channel);
                continue;
            }
            for (uint32_t base = rowBegin; base < rowEnd; base += rowBlock_) {
                const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
                ProcessV6Block(task, base, n, channel);
            }
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr uint32_t V6_ROWS = 2U;

    __aicore__ inline uint32_t V6TwoRowStageBytes() const
    {
        // Ping-pong direct+product, dx2+xhat, x input, dx output and invRms.
        return V6_ROWS * h_ * (2U * sizeof(float) + 2U * sizeof(D_T)) + HC_GRAD_DATA_BLOCK_BYTES;
    }

    __aicore__ inline uint32_t V6TwoRowScratchOffset() const
    {
        return HC_GRAD_GAMMA_RESIDENT_BYTES * h_ + HC_GRAD_PIPE_MAX_STAGES * V6TwoRowStageBytes();
    }

    __aicore__ inline uint32_t V6TwoRowStorageBytes() const
    {
        return V6TwoRowScratchOffset() + (V6_ROWS + 2U) * h_ * sizeof(float) + 2U * V6_ROWS * HC_GRAD_DATA_BLOCK_BYTES;
    }

    __aicore__ inline uint32_t V6TwoRowStageOffset(uint32_t stage) const
    {
        return HC_GRAD_GAMMA_RESIDENT_BYTES * h_ + stage * V6TwoRowStageBytes();
    }

    __aicore__ inline LocalTensor<float> V6TwoRowScalar(uint32_t slot)
    {
        return ub_.GetWithOffset<float>(
            1U, V6TwoRowScratchOffset() + (V6_ROWS + 2U) * h_ * sizeof(float) + slot * HC_GRAD_DATA_BLOCK_BYTES);
    }

    __aicore__ inline void IssueV6TwoRowIn(uint32_t task, uint32_t row, uint32_t n, uint32_t channel, uint32_t stage)
    {
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + row;
        const uint64_t offset = static_cast<uint64_t>(row) * d_ + static_cast<uint64_t>(channel) * h_;
        const uint32_t len = n * h_;
        const uint32_t block = V6_ROWS * h_;
        const uint32_t base = V6TwoRowStageOffset(stage);
        auto direct = ub_.GetWithOffset<float>(len, base);
        auto dx2 = ub_.GetWithOffset<float>(len, base + block * sizeof(float));
        auto xHalf = ub_.GetWithOffset<D_T>(len, base + 2U * block * sizeof(float));
        auto inv = ub_.GetWithOffset<float>(HC_GRAD_DATA_BLOCK_BYTES / sizeof(float),
                                            base + 2U * block * (sizeof(float) + sizeof(D_T)));
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        if (recomputeDirect_) {
            auto gmHalf = ub_.GetWithOffset<D_T>(len, base);
            auto upHalf = ub_.GetWithOffset<D_T>(len, base + block * sizeof(D_T));
            IssueCopyIn<D_T>(gmGradMixed_, token * h_, gmHalf, len);
            IssueCopyInRows<D_T, (HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5)>(
                gmUp_, token * d_ + static_cast<uint64_t>(channel) * h_, d_, upHalf, n, h_);
        } else {
            IssueCopyInRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), offset, d_, direct, n, h_);
        }
        if (lowpDx_) {
            // dx2 reuses the DX output slot. Unlike the original x-only input,
            // its DMA must wait for the previous block's DX MTE3 read to finish.
            WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
            auto dx2Half = ub_.GetWithOffset<D_T>(len, base + block * (2U * sizeof(float) + sizeof(D_T)));
            IssueCopyInRows<D_T>(Dx2TaskPointer(task), offset, d_, dx2Half, n, h_);
        } else {
            IssueCopyInRows<float, (HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5)>(
                TaskPointer(layout_.dx2, task, d_, sizeof(float)), offset, d_, dx2, n, h_);
        }
        IssueCopyInRows<D_T, (HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5)>(
            gmX_, token * d_ + static_cast<uint64_t>(channel) * h_, d_, xHalf, n, h_);
        IssueCopyIn<float>(gmInvRms_, token * c_, inv, n * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        SetFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static __simd_vf__ inline void A5V6Direct(__ubuf__ float* total, __ubuf__ float* up, __ubuf__ float* dx2,
                                              uint16_t repeats)
    {
        namespace Reg = AscendC::Reg;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> ones;
        Reg::Duplicate(ones, 1.0f);
        for (uint16_t rep = 0; rep < repeats; ++rep) {
            const uint32_t off = static_cast<uint32_t>(rep) * 64U;
            Reg::RegTensor<float> z, grad, dx;
            Reg::LoadAlign(z, up + off);
            Reg::LoadAlign(grad, total + off);
            Reg::LoadAlign(dx, dx2 + off);
            Reg::Muls(z, z, -1.0f, mask);
            Reg::Exp(z, z, mask);
            Reg::Adds(z, z, 1.0f, mask);
            Reg::Div(z, ones, z, mask);
            Reg::Mul(grad, grad, z, mask);
            Reg::Muls(grad, grad, 0.25f, mask);
            Reg::Add(grad, grad, dx, mask);
            Reg::StoreAlign(total + off, grad, mask);
        }
    }

    static __simd_vf__ inline void A5V6Project(__ubuf__ float* x, __ubuf__ float* total, __ubuf__ float* gamma,
                                               __ubuf__ float* pending, __ubuf__ float* gn, __ubuf__ float* invPtr,
                                               uint16_t repeats)
    {
        namespace Reg = AscendC::Reg;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> inv;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(inv, invPtr);
        for (uint16_t rep = 0; rep < repeats; ++rep) {
            const uint32_t off = static_cast<uint32_t>(rep) * 64U;
            Reg::RegTensor<float> xhat, grad, g, p, product;
            Reg::LoadAlign(xhat, x + off);
            Reg::LoadAlign(grad, total + off);
            Reg::LoadAlign(g, gamma + off);
            Reg::LoadAlign(p, pending + off);
            Reg::Mul(xhat, xhat, inv, mask);
            Reg::Mul(product, xhat, grad, mask);
            Reg::Add(p, p, product, mask);
            Reg::Mul(grad, grad, g, mask);
            Reg::Mul(product, grad, xhat, mask);
            Reg::StoreAlign(x + off, xhat, mask);
            Reg::StoreAlign(pending + off, p, mask);
            Reg::StoreAlign(gn + off, grad, mask);
            Reg::StoreAlign(total + off, product, mask);
        }
    }
#endif

    __aicore__ inline void ComputeV6TwoRows(uint32_t task, uint32_t row, uint32_t n, uint32_t channel, uint32_t stage,
                                            bool flush)
    {
        const uint32_t len = n * h_;
        const uint32_t block = V6_ROWS * h_;
        const uint32_t base = V6TwoRowStageOffset(stage);
        const uint32_t scratch = V6TwoRowScratchOffset();
        auto pending = ub_.GetWithOffset<float>(h_, scratch + block * sizeof(float));
        auto reduceWork = ub_.GetWithOffset<float>(h_, scratch + (block + h_) * sizeof(float));
        const uint32_t totalOffset = recomputeDirect_ ? scratch : base;
        const uint32_t gnOffset = recomputeDirect_ ? base : scratch;
        auto gn = ub_.GetWithOffset<float>(len, gnOffset);
        auto gamma = ub_.GetWithOffset<float>(h_, HC_GRAD_KAHAN_RESIDENT_BYTES * h_);
        auto total = ub_.GetWithOffset<float>(len, totalOffset);
        auto x = ub_.GetWithOffset<float>(len, base + block * sizeof(float));
        auto xHalf = ub_.GetWithOffset<D_T>(len, base + 2U * block * sizeof(float));
        auto dxHalf = ub_.GetWithOffset<D_T>(len, base + block * (2U * sizeof(float) + sizeof(D_T)));
        auto inv = ub_.GetWithOffset<float>(HC_GRAD_DATA_BLOCK_BYTES / sizeof(float),
                                            base + 2U * block * (sizeof(float) + sizeof(D_T)));
        WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        if (lowpDx_) {
            Cast(x, dxHalf, RoundMode::CAST_NONE, len);
        }
        if (recomputeDirect_) {
            auto gmHalf = ub_.GetWithOffset<D_T>(len, base);
            auto upHalf = ub_.GetWithOffset<D_T>(len, base + block * sizeof(D_T));
            Cast(total, gmHalf, RoundMode::CAST_NONE, len);
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t off = r * h_;
                Cast(reduceWork, upHalf[off], RoundMode::CAST_NONE, h_);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                // h is 2560 on this two-row path. Preserve the FP32 sigmoid and scale order.
                asc_vf_call<A5V6Direct>(reinterpret_cast<__ubuf__ float*>(total[off].GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(reduceWork.GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(x[off].GetPhyAddr()),
                                        static_cast<uint16_t>(h_ / 64U));
#else
                auto work = ub_.GetWithOffset<float>(h_, base);
                Muls(work, reduceWork, -1.0f, h_);
                Exp(work, work, h_);
                Adds(work, work, 1.0f, h_);
                Duplicate(reduceWork, 1.0f, h_);
                Div(reduceWork, reduceWork, work, h_);
                Mul(total[off], total[off], reduceWork, h_);
                ScaleByInvC(total[off], total[off], h_);
                Add(total[off], total[off], x[off], h_);
#endif
            }
        } else {
            Add(total, total, x, len);
        }
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
#endif

        float invValues[V6_ROWS];
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            asc_vf_call<A5V6Project>(reinterpret_cast<__ubuf__ float*>(x[off].GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(total[off].GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(pending.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(gn[off].GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(inv.GetPhyAddr()) + r * c_ + channel,
                                     static_cast<uint16_t>(h_ / 64U));
#else
            invValues[r] = inv.GetValue(r * c_ + channel);
            Muls(x[off], x[off], invValues[r], h_);
            // Use the row output as temporary dgamma product before generating gn.
            Mul(gn[off], x[off], total[off], h_);
            Add(pending, pending, gn[off], h_);
            Mul(gn[off], total[off], gamma, h_);
            // Recycle total as the normalized projection product.
            Mul(total[off], gn[off], x[off], h_);
#endif
            ReduceRmsProjection(V6TwoRowScalar(r), total[off], reduceWork, h_);
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        // V6's projection consumed inv_rms on V. S first needs it for the final dx scale.
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
#endif
        float means[V6_ROWS];
        for (uint32_t r = 0; r < n; ++r) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            invValues[r] = inv.GetValue(r * c_ + channel);
#endif
            means[r] = V6TwoRowScalar(r).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            auto residual = V6TwoRowScalar(V6_ROWS + r);
            if (means[r] <= HC_GRAD_MAX_FINITE_FLOAT && means[r] >= -HC_GRAD_MAX_FINITE_FLOAT) {
                Adds(total[r * h_], total[r * h_], -means[r], h_);
                ReduceRmsProjection(residual, total[r * h_], reduceWork, h_);
            } else {
                Duplicate(residual, 0.0f, 1U);
            }
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            const float residual =
                V6TwoRowScalar(V6_ROWS + r).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            Axpy(gn[off], x[off], -means[r], h_);
            if (residual != 0.0f) {
                Axpy(gn[off], x[off], -residual, h_);
            }
            Muls(gn[off], gn[off], invValues[r], h_);
        }
        Cast(dxHalf, gn, RoundMode::CAST_RINT, len);
        SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + row;
        IssueCopyOutRows<D_T>(gmDx_, token * d_ + static_cast<uint64_t>(channel) * h_, d_, dxHalf, n, h_);
        if (lowpDx_) {
            SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        }
        SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        if (flush) {
            auto tmp = ub_.GetWithOffset<float>(h_, scratch);
            auto tmp2 = ub_.GetWithOffset<float>(h_, scratch + h_ * sizeof(float));
            DgammaAccumAdd(pending, h_, 0U, tmp, tmp2);
            Duplicate(pending, 0.0f, h_);
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 && defined(__CCE_ENABLE_ASAN__)
    // Keep dx writeback visible to CCEC's A2 GM store instrumentation.
    __aicore__ __attribute__((always_inline)) inline void ProcessV6TwoRows()
#else
    __aicore__ __attribute__((noinline)) void ProcessV6TwoRows()
#endif
    {
        const uint32_t channel = aivId_ % c_;
        const uint32_t lane = aivId_ / c_;
        const uint32_t laneCount = (aivNum_ - 1U - channel) / c_ + 1U;
        auto gamma = ub_.GetWithOffset<float>(h_, HC_GRAD_KAHAN_RESIDENT_BYTES * h_);
        CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_, gamma, h_);
        Adds(gamma, gamma, 1.0f, h_);
        const bool previousOutputMode = independentPipeOutput_;
        independentPipeOutput_ = true;
        PipePrime();
        if (lowpDx_) {
            for (uint32_t s = 0; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
                SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(s));
            }
        }
        uint32_t stage = 0U;
        for (uint32_t task = 0; task < taskCount_; ++task) {
            const uint32_t remaining = n_ - task * taskRows_;
            const uint32_t rows = remaining < taskRows_ ? remaining : taskRows_;
            const uint32_t perCore = (rows + laneCount - 1U) / laneCount;
            const uint32_t begin = lane * perCore;
            const uint32_t end = begin + perCore < rows ? begin + perCore : rows;
            if (end <= begin) {
                continue;
            }
            auto pending = ub_.GetWithOffset<float>(h_, V6TwoRowScratchOffset() + V6_ROWS * h_ * sizeof(float));
            Duplicate(pending, 0.0f, h_);
            IssueV6TwoRowIn(task, begin, end - begin < V6_ROWS ? end - begin : V6_ROWS, channel, stage);
            for (uint32_t row = begin; row < end; row += V6_ROWS) {
                const uint32_t count = end - row < V6_ROWS ? end - row : V6_ROWS;
                const uint32_t next = row + count;
                if (next < end) {
                    const uint32_t nextCount = end - next < V6_ROWS ? end - next : V6_ROWS;
                    IssueV6TwoRowIn(task, next, nextCount, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
                }
                // Keep the original one-row pipeline's Kahan update after each four rows and at task end.
                const bool flush = next == end || (next - begin) % HC_GRAD_DGAMMA_FLUSH_BLOCKS == 0U;
                ComputeV6TwoRows(task, row, count, channel, stage, flush);
                stage ^= HC_GRAD_PIPE_STAGE_MASK;
            }
        }
        PipeDrain();
        if (lowpDx_) {
            for (uint32_t s = 0; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
                WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(s));
            }
        }
        independentPipeOutput_ = previousOutputMode;
        gammaCachedCh_ = -1;
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr AscendC::Reg::CastTrait A5_V6_B16_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait A5_V6_FP32_TO_B16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};

    // Compute both RMS projection passes and the output in the vector pipeline.
    // Only dx2/xhat and gn cross the projection reduction; inv_rms stays in a register.
    static __simd_vf__ inline void A5V6PackedRow(__ubuf__ D_T* mixed, __ubuf__ D_T* up, __ubuf__ D_T* xIn,
                                                 __ubuf__ float* dx2, __ubuf__ float* gamma, __ubuf__ float* pending,
                                                 __ubuf__ float* gn, __ubuf__ float* invPtr, __ubuf__ D_T* dxOut)
    {
        namespace Reg = AscendC::Reg;
        constexpr uint16_t LANES = 64;
        constexpr uint16_t REPEATS = 2560 / LANES;
        constexpr uint16_t HALF_REPEATS = REPEATS / 2U;
        auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> zero, one, width, inv, sum, grad, xhat, product;
        Reg::Duplicate(zero, 0.0f);
        Reg::Duplicate(one, 1.0f);
        Reg::Duplicate(width, 2560.0f);
        Reg::Duplicate(sum, 0.0f);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(inv, invPtr);
        for (uint16_t base = 0U; base < REPEATS; base += HALF_REPEATS) {
            Reg::RegTensor<float> halfSum;
            Reg::Duplicate(halfSum, 0.0f);
            for (uint16_t i = 0U; i < HALF_REPEATS; ++i) {
                const uint16_t off = (base + i) * LANES;
                Reg::RegTensor<D_T> packed;
                Reg::RegTensor<float> gate, dx, scale, partial;
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, mixed + off);
                Reg::Cast<float, D_T, A5_V6_B16_TO_FP32>(grad, packed, mask);
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, up + off);
                Reg::Cast<float, D_T, A5_V6_B16_TO_FP32>(gate, packed, mask);
                Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(packed, xIn + off);
                Reg::Cast<float, D_T, A5_V6_B16_TO_FP32>(xhat, packed, mask);
                Reg::LoadAlign(dx, dx2 + off);
                Reg::LoadAlign(scale, gamma + off);
                Reg::LoadAlign(partial, pending + off);
                Reg::Muls(gate, gate, -1.0f, mask);
                Reg::Exp(gate, gate, mask);
                Reg::Adds(gate, gate, 1.0f, mask);
                Reg::Div(gate, one, gate, mask);
                Reg::Mul(grad, grad, gate, mask);
                Reg::Muls(grad, grad, 0.25f, mask);
                Reg::Add(grad, grad, dx, mask);
                Reg::Mul(xhat, xhat, inv, mask);
                Reg::Mul(product, xhat, grad, mask);
                Reg::Add(partial, partial, product, mask);
                Reg::Mul(grad, grad, scale, mask);
                Reg::Mul(product, grad, xhat, mask);
                Reg::Add(halfSum, halfSum, product, mask);
                Reg::StoreAlign(dx2 + off, xhat, mask);
                Reg::StoreAlign(gn + off, grad, mask);
                Reg::StoreAlign(pending + off, partial, mask);
            }
            Reg::Add(sum, sum, halfSum, mask);
        }
        Reg::RegTensor<float> reduced, mean, residual, absMean;
        Reg::ReduceSum(reduced, sum, mask);
        Reg::Duplicate(mean, reduced, mask);
        Reg::Div(mean, mean, width, mask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::Duplicate(sum, 0.0f);
        for (uint16_t base = 0U; base < REPEATS; base += HALF_REPEATS) {
            Reg::RegTensor<float> halfSum;
            Reg::Duplicate(halfSum, 0.0f);
            for (uint16_t i = 0U; i < HALF_REPEATS; ++i) {
                const uint16_t off = (base + i) * LANES;
                Reg::LoadAlign(xhat, dx2 + off);
                Reg::LoadAlign(grad, gn + off);
                Reg::Mul(product, grad, xhat, mask);
                Reg::Sub(product, product, mean, mask);
                Reg::Add(halfSum, halfSum, product, mask);
            }
            Reg::Add(sum, sum, halfSum, mask);
        }
        Reg::ReduceSum(reduced, sum, mask);
        Reg::Duplicate(residual, reduced, mask);
        Reg::Div(residual, residual, width, mask);
        Reg::MaskReg finiteMean, hasResidual;
        Reg::Abs(absMean, mean, mask);
        Reg::CompareScalar<float, CMPMODE::LE>(finiteMean, absMean, HC_GRAD_MAX_FINITE_FLOAT, mask);
        Reg::Select(residual, residual, zero, finiteMean);
        Reg::CompareScalar<float, CMPMODE::NE>(hasResidual, residual, 0.0f, mask);
        Reg::Muls(mean, mean, -1.0f, mask);
        Reg::Muls(residual, residual, -1.0f, mask);
        for (uint16_t i = 0; i < REPEATS; ++i) {
            const uint16_t off = i * LANES;
            Reg::RegTensor<D_T> packed;
            Reg::RegTensor<float> corrected;
            Reg::LoadAlign(xhat, dx2 + off);
            Reg::LoadAlign(grad, gn + off);
            Reg::MulAddDst(grad, xhat, mean, mask);
            Reg::Add(corrected, grad, zero, mask);
            Reg::MulAddDst(corrected, xhat, residual, mask);
            Reg::Select(grad, corrected, grad, hasResidual);
            Reg::Mul(grad, grad, inv, mask);
            Reg::Cast<D_T, float, A5_V6_FP32_TO_B16>(packed, grad, mask);
            Reg::StoreAlign<D_T, Reg::StoreDist::DIST_PACK_B32>(dxOut + off, packed, mask);
        }
    }

    __aicore__ __attribute__((noinline)) void ComputeV6PackedRows(uint32_t n, uint32_t channel, uint32_t stage)
    {
        const uint32_t block = V6_ROWS * h_;
        const uint32_t base = V6TwoRowStageOffset(stage);
        const uint32_t scratch = V6TwoRowScratchOffset();
        auto gn = ub_.GetWithOffset<float>(block, scratch);
        auto pending = ub_.GetWithOffset<float>(h_, scratch + block * sizeof(float));
        auto mixed = ub_.GetWithOffset<D_T>(block, base);
        auto up = ub_.GetWithOffset<D_T>(block, base + block * sizeof(D_T));
        auto dx2 = ub_.GetWithOffset<float>(block, base + block * sizeof(float));
        auto x = ub_.GetWithOffset<D_T>(block, base + 2U * block * sizeof(float));
        auto out = ub_.GetWithOffset<D_T>(block, base + block * (2U * sizeof(float) + sizeof(D_T)));
        auto inv = ub_.GetWithOffset<float>(V6_ROWS * c_, base + 2U * block * (sizeof(float) + sizeof(D_T)));
        auto gamma = ub_.GetWithOffset<float>(h_, HC_GRAD_KAHAN_RESIDENT_BYTES * h_);
        for (uint32_t i = 0U; i < n; ++i) {
            const uint32_t off = i * h_;
            asc_vf_call<A5V6PackedRow>(reinterpret_cast<__ubuf__ D_T*>(mixed.GetPhyAddr()) + off,
                                       reinterpret_cast<__ubuf__ D_T*>(up.GetPhyAddr()) + off,
                                       reinterpret_cast<__ubuf__ D_T*>(x.GetPhyAddr()) + off,
                                       reinterpret_cast<__ubuf__ float*>(dx2.GetPhyAddr()) + off,
                                       reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(pending.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(gn.GetPhyAddr()) + off,
                                       reinterpret_cast<__ubuf__ float*>(inv.GetPhyAddr()) + i * c_ + channel,
                                       reinterpret_cast<__ubuf__ D_T*>(out.GetPhyAddr()) + off);
        }
    }

    __aicore__ __attribute__((noinline)) void FlushV6PackedDgamma()
    {
        const uint32_t block = V6_ROWS * h_;
        const uint32_t scratch = V6TwoRowScratchOffset();
        auto gn = ub_.GetWithOffset<float>(block, scratch);
        auto pending = ub_.GetWithOffset<float>(h_, scratch + block * sizeof(float));
        DgammaAccumAdd(pending, h_, 0U, gn, gn[h_]);
        Duplicate(pending, 0.0f, h_);
    }

    __aicore__ __attribute__((always_inline)) inline void ProcessV6PackedRows()
    {
        const uint32_t channel = aivId_ % c_;
        const uint32_t lane = aivId_ / c_;
        const uint32_t laneCount = (aivNum_ - 1U - channel) / c_ + 1U;
        const uint32_t perCore = (n_ + laneCount - 1U) / laneCount;
        const uint32_t begin = lane * perCore;
        if (begin >= n_) {
            return;
        }
        const uint32_t end = n_ - begin < perCore ? n_ : begin + perCore;
        auto gamma = ub_.GetWithOffset<float>(h_, HC_GRAD_KAHAN_RESIDENT_BYTES * h_);
        CopyIn<float>(gmGamma_, static_cast<uint64_t>(channel) * h_, gamma, h_);
        Adds(gamma, gamma, 1.0f, h_);
        const bool previousOutputMode = independentPipeOutput_;
        independentPipeOutput_ = true;
        PipePrime();
        uint32_t stage = 0U;
        auto pending = ub_.GetWithOffset<float>(h_, V6TwoRowScratchOffset() + V6_ROWS * h_ * sizeof(float));
        Duplicate(pending, 0.0f, h_);
        IssueV6TwoRowIn(0U, begin, end - begin < V6_ROWS ? end - begin : V6_ROWS, channel, stage);
        for (uint32_t row = begin; row < end; row += V6_ROWS) {
            const uint32_t count = end - row < V6_ROWS ? end - row : V6_ROWS;
            const uint32_t next = row + count;
            if (next < end) {
                const uint32_t nextCount = end - next < V6_ROWS ? end - next : V6_ROWS;
                IssueV6TwoRowIn(0U, next, nextCount, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
            }
            // Keep the original one-row pipeline's Kahan update after each four rows and at task end.
            const bool flush = next == end || (next - begin) % HC_GRAD_DGAMMA_FLUSH_BLOCKS == 0U;
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            ComputeV6PackedRows(count, channel, stage);
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            const uint32_t block = V6_ROWS * h_;
            const uint32_t base = V6TwoRowStageOffset(stage);
            auto out = ub_.GetWithOffset<D_T>(block, base + block * (2U * sizeof(float) + sizeof(D_T)));
            SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
            IssueCopyOutRows<D_T, true>(gmDx_, static_cast<uint64_t>(row) * d_ + static_cast<uint64_t>(channel) * h_,
                                        d_, out, count, h_);
            SetFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
            if (flush) {
                FlushV6PackedDgamma();
            }
            stage ^= HC_GRAD_PIPE_STAGE_MASK;
        }
        PipeDrain();
        independentPipeOutput_ = previousOutputMode;
        gammaCachedCh_ = -1;
    }
#endif
#endif

    // V6 uses per-stage direct/dx2/x input and dx output slots to overlap MTE2/MTE3 with vector computation.
    __aicore__ inline void IssueV6In(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t channel,
                                     uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
        // direct and dx2 use the task's row base with stride D and the current channel's H-element segment.
        // x uses the global token row base.
        const uint64_t rowOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
        const uint64_t xOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto xHalf = V6InHalf(stage, len);
        auto directIn = V6InFloat(stage, HC_GRAD_V6_DIRECT_INPUT, len);
        auto dx2In = V6InFloat(stage, HC_GRAD_V6_DX2_INPUT, len);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), V6InvOffset(stage));
#else
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset(stage));
#endif
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        if (!independentPipeOutput_) {
            WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (recomputeDirect_) {
            IssueCopyIn<D_T>(gmGradMixed_, token * h_, V6DirectHalf(stage, 0U, len), len);
            IssueCopyInRows<D_T>(gmUp_, xOffset, d_, V6DirectHalf(stage, 1U, len), n, h_);
        } else
#endif
        {
            IssueCopyInRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), rowOffset, d_, directIn, n,
                                   h_);
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (lowpDx_) {
            auto dx2Half = directIn.template ReinterpretCast<D_T>();
            IssueCopyInRows<D_T>(Dx2TaskPointer(task), rowOffset, d_, dx2Half, n, h_);
        } else
#endif
        {
            IssueCopyInRows<float>(TaskPointer(layout_.dx2, task, d_, sizeof(float)), rowOffset, d_, dx2In, n, h_);
        }
        IssueCopyInRows<D_T>(gmX_, xOffset, d_, xHalf, n, h_);
        IssueCopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        SetFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
    }

    __aicore__ inline void ReduceV6Moments(uint32_t n, uint32_t channel, LocalTensor<float> total,
                                           LocalTensor<float> xBuf, LocalTensor<float> gnRow,
                                           LocalTensor<float> product, LocalTensor<float> correction,
                                           LocalTensor<float> reduceWork, LocalTensor<float> invBuf,
                                           bool reuseProjection)
    {
        // A single row can retain both the weighted gradient and its dot-product
        // vector in the existing row scratch buffers until projection finishes.
        if (reuseProjection) {
            const float inv = invBuf.GetValue(channel);
            Muls(xBuf, xBuf, inv, h_);
            if constexpr (HAS_GAMMA) {
                EnsureGammaCache(channel);
                auto gammaRow = ub_.GetWithOffset<float>(h_, GammaOffset());
                Mul(gnRow, total, gammaRow, h_);
                Mul(product, xBuf, total, h_);
                Add(correction, correction, product, h_);
            } else {
                Adds(gnRow, total, 0.0f, h_);
            }
            Mul(product, gnRow, xBuf, h_);
            ReduceRmsProjection(SumSlot(0), product, reduceWork, h_);
            return;
        }
        if constexpr (HAS_GAMMA) {
            // Only the gamma-cache branch uses the pipeline; otherwise gamma needs another row-sized slot.
            EnsureGammaCache(channel);
            auto gammaRow = ub_.GetWithOffset<float>(h_, GammaOffset());
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t off = r * h_;
                const float inv = invBuf.GetValue(r * c_ + channel);
                Mul(gnRow, total[off], gammaRow, h_);
                Muls(xBuf[off], xBuf[off], inv, h_);
                Mul(product, xBuf[off], total[off], h_);
                // Accumulate pending dgamma in one pass per row and apply Kahan once every F row blocks,
                // instead of running five vector passes for every row.
                Add(correction, correction, product, h_);
                Mul(product, gnRow, xBuf[off], h_);
                ReduceRmsProjection(SumSlot(r), product, reduceWork, h_);
            }
        } else {
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t off = r * h_;
                const float inv = invBuf.GetValue(r * c_ + channel);
                Muls(xBuf[off], xBuf[off], inv, h_);
                Adds(gnRow, total[off], 0.0f, h_);
                Mul(product, gnRow, xBuf[off], h_);
                ReduceRmsProjection(SumSlot(r), product, reduceWork, h_);
            }
        }
    }

    __aicore__ inline void ProjectV6PipeRows(uint32_t n, uint32_t channel, LocalTensor<float> total,
                                             LocalTensor<float> xBuf, LocalTensor<float> gnRow,
                                             LocalTensor<float> product, LocalTensor<float> reduceWork,
                                             LocalTensor<float> invBuf, LocalTensor<D_T> dxHalf, bool reuseProjection)
    {
        if (reuseProjection) {
            ReduceRmsResidual(product, reduceWork, 0U);
            SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
            const float inv = invBuf.GetValue(channel);
            const float mean = SumSlot(0).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            const float residual = ResidualSlot(0).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            Axpy(gnRow, xBuf, -mean, h_);
            if (residual != 0.0f) {
                Axpy(gnRow, xBuf, -residual, h_);
            }
            Muls(gnRow, gnRow, inv, h_);
            Cast(dxHalf, gnRow, RoundMode::CAST_RINT, h_);
            return;
        }
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            if constexpr (HAS_GAMMA) {
                auto gammaRow = ub_.GetWithOffset<float>(h_, GammaOffset());
                Mul(gnRow, total[off], gammaRow, h_);
            } else {
                Adds(gnRow, total[off], 0.0f, h_);
            }
            Mul(product, gnRow, xBuf[off], h_);
            ReduceRmsResidual(product, reduceWork, r);
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            const float inv = invBuf.GetValue(r * c_ + channel);
            const float sum = SumSlot(r).GetValue(0);
            if constexpr (HAS_GAMMA) {
                auto gammaRow = ub_.GetWithOffset<float>(h_, GammaOffset());
                Mul(gnRow, total[off], gammaRow, h_);
            } else {
                Adds(gnRow, total[off], 0.0f, h_);
            }
            const float mean = sum / static_cast<float>(static_cast<int32_t>(h_));
            const float residual = ResidualSlot(r).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            Axpy(gnRow, xBuf[off], -mean, h_);
            if (residual != 0.0f) {
                Axpy(gnRow, xBuf[off], -residual, h_);
            }
            Muls(gnRow, gnRow, inv, h_);
            Cast(dxHalf[off], gnRow, RoundMode::CAST_RINT, h_);
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    // Keep the original two-row traversal and dgamma flush points. Reuse the consumed dx2
    // input for xhat and the direct input for each row's projection product.
    template <bool THREE_SLOTS>
    __aicore__ inline void ComputeV6RetainedProjection(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk,
                                                       uint32_t blocks, uint32_t channel, uint32_t stage, uint32_t n)
    {
        const uint32_t len = n * h_;
        auto total = V6InFloat(stage, HC_GRAD_V6_DIRECT_INPUT, len);
        auto x = V6InFloat(stage, HC_GRAD_V6_DX2_INPUT, len);
        auto gn = V6X(len);
        auto product = V6Row(HC_GRAD_V6_ROW_PRODUCT, h_);
        auto pending = V6Row(HC_GRAD_V6_ROW_PENDING, h_);
        auto reduceWork = V6Row(HC_GRAD_V6_ROW_REDUCE_WORK, h_);
        auto inv = ub_.GetWithOffset<float>(invStride_ / sizeof(float), V6InvOffset(stage));

        Add(total, total, x, len);
        Cast(x, V6InHalf(stage, len), RoundMode::CAST_NONE, len);
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        if constexpr (THREE_SLOTS) {
            if (blk + 2U < blocks) {
                const uint32_t futureStage = stage == 0U ? V6_EXTRA_STAGE : stage - 1U;
                IssueV6In(task, rowBegin, rowEnd, blk + 2U, channel, futureStage);
            }
        } else if (!independentPipeOutput_ && blk + 1U < blocks) {
            IssueV6In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
        EnsureGammaCache(channel);
        auto gamma = ub_.GetWithOffset<float>(h_, GammaOffset());
        float invValues[2];
        for (uint32_t r = 0U; r < n; ++r) {
            const uint32_t off = r * h_;
            invValues[r] = inv.GetValue(r * c_ + channel);
            Mul(gn[off], total[off], gamma, h_);
            Muls(x[off], x[off], invValues[r], h_);
            Mul(product, x[off], total[off], h_);
            Add(pending, pending, product, h_);
            Mul(total[off], gn[off], x[off], h_);
            ReduceRmsProjection(SumSlot(r), total[off], reduceWork, h_);
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        for (uint32_t r = 0U; r < n; ++r) {
            ReduceRmsResidual(total[r * h_], reduceWork, r);
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        for (uint32_t r = 0U; r < n; ++r) {
            const uint32_t off = r * h_;
            const float mean = SumSlot(r).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            const float residual = ResidualSlot(r).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
            Axpy(gn[off], x[off], -mean, h_);
            if (residual != 0.0f) {
                Axpy(gn[off], x[off], -residual, h_);
            }
            Muls(gn[off], gn[off], invValues[r], h_);
        }
        if (independentPipeOutput_) {
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        Cast(V6OutHalf(stage, len), gn, RoundMode::CAST_RINT, len);
        SetFlag<HardEvent::S_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::S_V>(HC_GRAD_VECTOR_EVENT);
    }
#endif

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    template <bool THREE_SLOTS = false>
#endif
    __aicore__ inline void ComputeV6(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t blocks,
                                     uint32_t channel, uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            if (rowBlock_ == 2U && useGammaCache_ && !recomputeDirect_ && !lowpDx_) {
                ComputeV6RetainedProjection<THREE_SLOTS>(task, rowBegin, rowEnd, blk, blocks, channel, stage, n);
                return;
            }
        }
#endif
        const uint32_t len = n * h_;
        (void)task;
        auto xHalf = V6InHalf(stage, len);
        auto directIn = V6InFloat(stage, HC_GRAD_V6_DIRECT_INPUT, len);
        auto dx2In = V6InFloat(stage, HC_GRAD_V6_DX2_INPUT, len);
        auto dxHalf = V6OutHalf(stage, len);
        auto total = directIn; // Inline total = direct + dx2 in place.
        auto xBuf = V6X(len);
        auto gnRow = V6Row(HC_GRAD_V6_ROW_GN, h_);
        auto product = V6Row(HC_GRAD_V6_ROW_PRODUCT, h_);
        auto correction = V6Row(HC_GRAD_V6_ROW_PENDING, h_);
        auto reduceWork = V6Row(HC_GRAD_V6_ROW_REDUCE_WORK, h_);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), V6InvOffset(stage));
#else
        auto invBuf = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset(stage));
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (lowpDx_) {
            Cast(dx2In, directIn.template ReinterpretCast<D_T>(), RoundMode::CAST_NONE, len);
        }
        if (recomputeDirect_) {
            Cast(total, V6DirectHalf(stage, 0U, len), RoundMode::CAST_NONE, len);
            Cast(xBuf, V6DirectHalf(stage, 1U, len), RoundMode::CAST_NONE, len);
            SigmoidFp32(xBuf, xBuf, len);
            Mul(total, total, xBuf, len);
            ScaleByInvC(total, total, len);
            Cast(xBuf, xHalf, RoundMode::CAST_NONE, len);
            Add(total, total, dx2In, len);
        } else
#endif
        {
            Cast(xBuf, xHalf, RoundMode::CAST_NONE, len);
            Add(total, total, dx2In, len); // Inline total = direct + dxnorm in FP32.
        }
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        if (!independentPipeOutput_ && blk + 1U < blocks) {
            IssueV6In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        const bool reuseProjection = rowBlock_ == 1U;
#else
        constexpr bool reuseProjection = false;
#endif
        ReduceV6Moments(n, channel, total, xBuf, gnRow, product, correction, reduceWork, invBuf, reuseProjection);
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        if (independentPipeOutput_) {
            WaitFlag<HardEvent::MTE3_V>(PipeOutEvent(stage));
        }
        ProjectV6PipeRows(n, channel, total, xBuf, gnRow, product, reduceWork, invBuf, dxHalf, reuseProjection);
        SetFlag<HardEvent::S_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::S_V>(HC_GRAD_VECTOR_EVENT);
    }

    // Merge pending V6Row(2) into the Kahan accumulator every F blocks,
    // then clear the pending buffer.
    __aicore__ inline void FlushDgammaPending()
    {
        auto pending = V6Row(HC_GRAD_V6_ROW_PENDING, h_);
        auto tmp = V6Row(HC_GRAD_V6_ROW_PRODUCT, h_);
        auto tmp2 = V6Row(HC_GRAD_V6_ROW_REDUCE_WORK, h_);
        DgammaAccumAdd(pending, h_, 0U, tmp, tmp2);
        Duplicate(pending, 0.0f, h_);
    }

    __aicore__ inline void IssueV6Out(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t channel,
                                      uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        const uint64_t token = static_cast<uint64_t>(task) * taskRows_ + base;
        const uint64_t outOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto dxHalf = V6OutHalf(stage, len);
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        IssueCopyOutRows<D_T>(gmDx_, outOffset, d_, dxHalf, n, h_);
        SetPipeOutputReady(stage);
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    template <bool THREE_SLOTS = false>
    __aicore__ inline uint32_t ProcessV6Blocks(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t channel,
                                               uint32_t firstStage)
    {
        if (rowEnd <= rowBegin) {
            return firstStage;
        }
        const uint32_t blocks = (rowEnd - rowBegin + rowBlock_ - 1U) / rowBlock_;
        // Other operations overwrite the pending pool, so clear it on entry. Flush every four blocks
        // and at task end. Up to 4*rowBlock sequential additions precede each Kahan update;
        // Kahan cannot recover rounding error already introduced inside that batch.
        if constexpr (HAS_GAMMA) {
            auto pending = V6Row(HC_GRAD_V6_ROW_PENDING, h_);
            Duplicate(pending, 0.0f, h_);
        }
        IssueV6In(task, rowBegin, rowEnd, 0U, channel, firstStage);
        if constexpr (THREE_SLOTS) {
            if (blocks > 1U) {
                IssueV6In(task, rowBegin, rowEnd, 1U, channel, 1U);
            }
        }
        uint32_t stage = firstStage;
        for (uint32_t blk = 0U; blk < blocks; ++blk) {
            if (!THREE_SLOTS && independentPipeOutput_ && blk + 1U < blocks) {
                IssueV6In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
            }
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            ComputeV6<THREE_SLOTS>(task, rowBegin, rowEnd, blk, blocks, channel, stage);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
            // The Kahan update uses row scratch, so it can run while MTE3 reads the DX slot.
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV6Out(task, rowBegin, rowEnd, blk, channel, stage);
#endif
            if constexpr (HAS_GAMMA) {
                if (((blk + 1U) % HC_GRAD_DGAMMA_FLUSH_BLOCKS) == 0U || (blk + 1U) == blocks) {
                    FlushDgammaPending();
                }
            }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV6Out(task, rowBegin, rowEnd, blk, channel, stage);
#endif
            if constexpr (THREE_SLOTS) {
                stage = stage == HC_GRAD_PIPE_MAX_STAGES ? 0U : stage + 1U;
            } else {
                stage ^= HC_GRAD_PIPE_STAGE_MASK;
            }
        }
        return stage;
    }
#else
    __aicore__ inline uint32_t ProcessV6Blocks(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t channel,
                                               uint32_t firstStage)
    {
        if (rowEnd <= rowBegin) {
            return firstStage;
        }
        const uint32_t blocks = (rowEnd - rowBegin + rowBlock_ - 1U) / rowBlock_;
        // Other operations overwrite the pending pool, so clear it on entry. Flush every four blocks
        // and at task end. Up to 4*rowBlock sequential additions precede each Kahan update;
        // Kahan cannot recover rounding error already introduced inside that batch.
        if constexpr (HAS_GAMMA) {
            auto pending = V6Row(HC_GRAD_V6_ROW_PENDING, h_);
            Duplicate(pending, 0.0f, h_);
        }
        IssueV6In(task, rowBegin, rowEnd, 0U, channel, firstStage);
        for (uint32_t blk = 0U; blk < blocks; ++blk) {
            const uint32_t stage = (firstStage + blk) & HC_GRAD_PIPE_STAGE_MASK;
            if (independentPipeOutput_ && blk + 1U < blocks) {
                IssueV6In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
            }
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            ComputeV6(task, rowBegin, rowEnd, blk, blocks, channel, stage);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
            // The Kahan update uses row scratch, so it can run while MTE3 reads the DX slot.
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV6Out(task, rowBegin, rowEnd, blk, channel, stage);
#endif
            if constexpr (HAS_GAMMA) {
                if (((blk + 1U) % HC_GRAD_DGAMMA_FLUSH_BLOCKS) == 0U || (blk + 1U) == blocks) {
                    FlushDgammaPending();
                }
            }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV6Out(task, rowBegin, rowEnd, blk, channel, stage);
#endif
        }
        return (firstStage + blocks) & HC_GRAD_PIPE_STAGE_MASK;
    }
#endif

    __aicore__ inline void ProcessV6Pipe(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t channel)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (CanUseV6ThreeSlots()) {
            constexpr uint32_t stages = HC_GRAD_PIPE_MAX_STAGES + 1U;
            PipePrime(stages);
            ProcessV6Blocks<true>(task, rowBegin, rowEnd, channel, 0U);
            PipeDrain(stages);
            return;
        }
#endif
        PipePrime();
        ProcessV6Blocks(task, rowBegin, rowEnd, channel, 0U);
        PipeDrain();
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    // In the token pipeline the NN kernels also use UB. Give every public dgamma
    // element one owner and persist its compensation before the ring slot is released.
    __aicore__ __attribute__((noinline)) void AccumulateBoundedDgamma(uint32_t task, uint32_t rows)
    {
        PipeBarrier<PIPE_ALL>();
        const uint32_t alignment = HC_GRAD_VECTOR_ALIGN;
        const uint32_t lanes = (aivNum_ + c_ - 1U) / c_;
        const uint32_t tile = (h_ + lanes * alignment - 1U) / (lanes * alignment) * alignment;
        const uint32_t tiles = (h_ + tile - 1U) / tile;
        GM_ADDR direct = TaskPointer(layout_.direct, task, d_, sizeof(float));
        GM_ADDR dx2 = TaskPointer(layout_.dx2, task, d_, sizeof(float));
        GM_ADDR compensation = Pointer(layout_.dgammaPartial);
        const uint64_t firstToken = static_cast<uint64_t>(task) * taskRows_;
        for (uint32_t i = aivId_; i < c_ * tiles; i += aivNum_) {
            const uint32_t ch = i / tiles;
            const uint32_t col = (i % tiles) * tile;
            const uint32_t count = h_ - col < tile ? h_ - col : tile;
            const uint32_t feature = ch * h_ + col;
            const uint32_t elements = rowBlock_ * count;
            auto sum = FloatBuf(0U, count);
            auto comp = FloatBuf(1U, count);
            auto grad = FloatBuf(2U, elements);
            auto dx = FloatBuf(3U, elements);
            auto xhat = FloatBuf(4U, elements);
            auto work = FloatBuf(5U, elements);
            auto raw = work.template ReinterpretCast<D_T>();
            if (task == 0U) {
                Duplicate(sum, 0.0f, count);
                Duplicate(comp, 0.0f, count);
            } else {
                CopyIn<float>(gmDgamma_, feature, sum, count);
                CopyIn<float>(compensation, feature, comp, count);
            }
            auto inv = ub_.GetWithOffset<float>(invStride_ / sizeof(float), InvOffset(0U));
            for (uint32_t first = 0U; first < rows; first += rowBlock_) {
                const uint32_t n = rows - first < rowBlock_ ? rows - first : rowBlock_;
                const uint64_t local = static_cast<uint64_t>(first) * d_ + feature;
                SetFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
                WaitFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
                IssueCopyInRows<float>(direct, local, d_, grad, n, count);
                IssueCopyInRows<float>(dx2, local, d_, dx, n, count);
                IssueCopyInRows<D_T>(gmX_, firstToken * d_ + local, d_, raw, n, count);
                IssueCopyIn<float>(gmInvRms_, (firstToken + first) * c_, inv, n * c_);
                SyncCopy();
                Cast(xhat, raw, RoundMode::CAST_NONE, n * count);
                Add(grad, grad, dx, n * count);
                SetFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
                WaitFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t offset = j * count;
                    Muls(xhat[offset], xhat[offset], inv.GetValue(j * c_ + ch), count);
                    Mul(grad[offset], grad[offset], xhat[offset], count);
                    MergeDgammaGatherValue(sum, comp, dx, work, grad[offset], count);
                }
            }
            CopyOut<float>(gmDgamma_, feature, sum, count);
            if (task + 1U < taskCount_) {
                CopyOut<float>(compensation, feature, comp, count);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void MergeDgammaGatherValue(const LocalTensor<float>& accum, const LocalTensor<float>& comp,
                                                  const LocalTensor<float>& tmp, const LocalTensor<float>& tmp2,
                                                  const LocalTensor<float>& value, uint32_t count)
    {
        Sub(tmp, value, comp, count);
        Add(tmp2, accum, tmp, count);
        Sub(comp, tmp2, accum, count);
        Sub(comp, comp, tmp, count);
        Adds(accum, tmp2, 0.0f, count);
    }

    __aicore__ __attribute__((noinline)) bool TryReduceDgammaGather()
    {
        constexpr uint32_t COL_ALIGN = 16U;
        constexpr uint32_t MAX_DMA_BLOCKS = 4095U;
        constexpr uint32_t GATHER_SLOT = HC_GRAD_REDUCE_TMP_SUM + 1U;
        static_assert(COL_ALIGN * sizeof(float) % HC_GRAD_DATA_BLOCK_BYTES == 0U);
        static_assert(HC_GRAD_DGAMMA_REDUCE_CHUNK % COL_ALIGN == 0U);
        if (!layout_.fullBatch || !kahanDgamma_ || c_ != 4U || h_ != 2560U || aivNum_ < c_ || aivNum_ % c_ != 0U ||
            aivId_ >= aivNum_) {
            return false;
        }
        const uint32_t lanes = aivNum_ / c_;
        if (lanes > MAX_DMA_BLOCKS) {
            return false;
        }
        const uint32_t chunkAlign = lanes * COL_ALIGN;
        const uint32_t cols = (h_ + chunkAlign - 1U) / chunkAlign * COL_ALIGN;
        const uint32_t chunkLimit = cols < HC_GRAD_DGAMMA_REDUCE_CHUNK ? cols : HC_GRAD_DGAMMA_REDUCE_CHUNK;
        const uint32_t planeBytes = lanes * chunkLimit * sizeof(float);
        // Borrow the pool after the last live merge buffer, starting at FP32 slot 5.
        const uint32_t gatherOffset = GATHER_SLOT * bufCount_ * sizeof(float);
        const uint32_t gatherEnd = gatherOffset + 2U * planeBytes;
        if (chunkLimit > bufCount_ || gatherEnd > poolBytes_ || ubBufBase_ + gatherEnd > ubBytes_) {
            return false;
        }

        const uint32_t channel = aivId_ % c_;
        const uint32_t colBegin = (aivId_ / c_) * cols;
        const uint32_t colEnd = colBegin + cols < h_ ? colBegin + cols : h_;
        const uint32_t coreStride = HC_GRAD_DGAMMA_SLOTS * h_;
        const uint32_t channelStride = c_ * coreStride;
        GM_ADDR partial = Pointer(layout_.dgammaPartial);
        auto accum = FloatBuf(HC_GRAD_REDUCE_ACCUM, chunkLimit);
        auto comp = FloatBuf(HC_GRAD_REDUCE_COMP, chunkLimit);
        auto tmp = FloatBuf(HC_GRAD_REDUCE_TMP, chunkLimit);
        auto tmp2 = FloatBuf(HC_GRAD_REDUCE_TMP_SUM, chunkLimit);
        for (uint32_t off = colBegin; off < colEnd; off += chunkLimit) {
            const uint32_t n = colEnd - off < chunkLimit ? colEnd - off : chunkLimit;
            const uint32_t planeElems = lanes * n;
            auto sums = ub_.GetWithOffset<float>(planeElems, ubBufBase_ + gatherOffset);
            auto corrections = ub_.GetWithOffset<float>(planeElems, ubBufBase_ + gatherOffset + planeBytes);
            Duplicate(accum, 0.0f, n);
            Duplicate(comp, 0.0f, n);

            // Every burst is a multiple of 32 B, so dstStride=0 packs slices at i*n.
            // One readiness wait protects the borrowed UB and one publishes both planes.
            SetFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
            WaitFlag<HardEvent::V_MTE2>(HC_GRAD_VECTOR_EVENT);
            const uint64_t source = static_cast<uint64_t>(channel) * coreStride + off;
            IssueCopyInRows<float>(partial, source, channelStride, sums, lanes, n);
            IssueCopyInRows<float>(partial, source + h_, channelStride, corrections, lanes, n);
            SyncCopy();

            // Merge the sum and negated compensation for each producer before advancing to the next.
            for (uint32_t i = 0; i < lanes; ++i) {
                MergeDgammaGatherValue(accum, comp, tmp, tmp2, sums[i * n], n);
                auto correction = corrections[i * n];
                Muls(correction, correction, -1.0f, n);
                MergeDgammaGatherValue(accum, comp, tmp, tmp2, correction, n);
            }
            const uint64_t target = static_cast<uint64_t>(channel) * h_ + off;
            CopyOut<float>(gmDgamma_, target, accum, n);
        }
        return true;
    }
#endif

    __aicore__ inline void ReduceDgamma()
    {
        // Each core stores 2H partial values: accumulator and compensation. Sum cores assigned to the same channel.
        const uint64_t coreStride = static_cast<uint64_t>(HC_GRAD_DGAMMA_SLOTS) * h_;
        // Limit chunks to bufCount_ so accumulator and slice buffers cannot overlap.
        const uint32_t chunkLimit = (bufCount_ < HC_GRAD_DGAMMA_REDUCE_CHUNK) ? bufCount_ : HC_GRAD_DGAMMA_REDUCE_CHUNK;
        GM_ADDR partial = Pointer(layout_.dgammaPartial);
        auto accum = FloatBuf(HC_GRAD_REDUCE_ACCUM, chunkLimit);
        auto slice = FloatBuf(HC_GRAD_REDUCE_SLICE, chunkLimit);
        auto comp = FloatBuf(HC_GRAD_REDUCE_COMP, chunkLimit);
        auto tmp = FloatBuf(HC_GRAD_REDUCE_TMP, chunkLimit);
        auto tmp2 = FloatBuf(HC_GRAD_REDUCE_TMP_SUM, chunkLimit);
        uint32_t channelBegin = aivId_;
        uint32_t colBegin = 0U;
        uint32_t colEnd = h_;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (layout_.fullBatch && c_ == 4U && h_ == 2560U && aivNum_ >= c_ && aivNum_ % c_ == 0U) {
            // Keep each element's Kahan traversal unchanged. Split only the H axis.
            constexpr uint32_t COL_ALIGN = 16U;
            const uint32_t lanes = aivNum_ / c_;
            const uint32_t lane = aivId_ / c_;
            const uint32_t cols = (h_ + lanes * COL_ALIGN - 1U) / (lanes * COL_ALIGN) * COL_ALIGN;
            channelBegin = aivId_ % c_;
            colBegin = lane * cols;
            colEnd = (colBegin + cols < h_) ? colBegin + cols : h_;
        }
#endif
        for (uint32_t channel = channelBegin; channel < c_; channel += aivNum_) {
            // The first channel wave uses cores channel, channel+c, and so on. Later channel waves use
            // only channel % aivNum.
            const uint32_t first = (channel < aivNum_) ? channel : (channel % aivNum_);
            const uint32_t step = (channel < aivNum_) ? c_ : aivNum_;
            for (uint32_t off = colBegin; off < colEnd; off += chunkLimit) {
                const uint32_t n = (colEnd - off < chunkLimit) ? (colEnd - off) : chunkLimit;
                Duplicate(accum, 0.0f, n);
                Duplicate(comp, 0.0f, n);
                for (uint32_t core = first; core < aivNum_; core += step) {
                    CopyIn<float>(partial, static_cast<uint64_t>(core) * coreStride + off, slice, n);
                    if (kahanDgamma_) {
                        Sub(tmp, slice, comp, n);
                        Add(tmp2, accum, tmp, n);
                        Sub(comp, tmp2, accum, n);
                        Sub(comp, comp, tmp, n);
                        Adds(accum, tmp2, 0.0f, n);
                        CopyIn<float>(partial, static_cast<uint64_t>(core) * coreStride + h_ + off, slice, n);
                        // Kahan stores compensation = (t-sum)-y, so the corrected sum is accumulator-compensation.
                        // Negate each core's compensation when merging partials across cores.
                        Muls(slice, slice, -1.0f, n);
                        Sub(tmp, slice, comp, n);
                        Add(tmp2, accum, tmp, n);
                        Sub(comp, tmp2, accum, n);
                        Sub(comp, comp, tmp, n);
                        Adds(accum, tmp2, 0.0f, n);
                    } else {
                        Add(accum, accum, slice, n);
                    }
                }
                CopyOut<float>(gmDgamma_, static_cast<uint64_t>(channel) * h_ + off, accum, n);
            }
        }
    }

    __aicore__ inline void ReduceDact(uint32_t task, uint32_t rows)
    {
        GM_ADDR partial =
            Pointer(layout_.TaskBufferOffset(layout_.dactPartial, task, workspaceSlots_,
                                             static_cast<uint64_t>(splitkFactor_) * taskRows_ * r_ * sizeof(float)));
        GM_ADDR dact = TaskPointer(layout_.dact, task, r_, sizeof(float));
        // Batch contiguous rows of each split-K partial while preserving the addition order within every row.
        // This amortizes GM transfer overhead. Accumulators use F0/F1 and incoming slices use F4/F5;
        // each side can hold at most 2*bufCount_ elements within the six FP32 slots.
        // The batched layout must keep those accumulator and slice regions disjoint.
        const uint64_t rowStride = static_cast<uint64_t>(rows) * r_;
        uint32_t block = (HC_GRAD_DACT_ACCUM_SLOTS * bufCount_) / r_;
        if (block == 0U) {
            block = 1U;
        }
        if (block > HC_GRAD_DACT_REDUCE_ROWS) {
            block = HC_GRAD_DACT_REDUCE_ROWS;
        }
        const uint32_t rowsPerCore = (rows + aivNum_ - 1U) / aivNum_;
        const uint32_t rowBegin = aivId_ * rowsPerCore;
        const uint32_t rowEnd = (rowBegin + rowsPerCore < rows) ? (rowBegin + rowsPerCore) : rows;
        const uint32_t maxCount = block * r_;
        auto accum = FloatBuf(HC_GRAD_REDUCE_ACCUM, maxCount);
        auto slice = FloatBuf(HC_GRAD_DACT_REDUCE_SLICE, maxCount);
        for (uint32_t base = rowBegin; base < rowEnd; base += block) {
            const uint32_t nb = (rowEnd - base < block) ? (rowEnd - base) : block;
            const uint32_t count = nb * r_;
            const uint64_t baseOff = static_cast<uint64_t>(base) * r_;
            Duplicate(accum, 0.0f, count);
            for (uint32_t s = 0; s < splitkFactor_; ++s) {
                CopyIn<float>(partial, static_cast<uint64_t>(s) * rowStride + baseOff, slice, count);
                Add(accum, accum, slice, count);
            }
            CopyOut<float>(dact, baseOff, accum, count);
        }
    }

    __aicore__ inline void ReducePartialAtomic2(uint64_t partialOffset, GM_ADDR output, uint32_t task0, uint32_t task1,
                                                uint64_t slotElements, uint64_t sourceOffset, uint64_t sourceCount)
    {
        GM_ADDR partial0 =
            Pointer(layout_.TaskBufferOffset(partialOffset, task0, workspaceSlots_, slotElements * sizeof(float)));
        GM_ADDR partial1 =
            Pointer(layout_.TaskBufferOffset(partialOffset, task1, workspaceSlots_, slotElements * sizeof(float)));
        const uint32_t chunkLimit = h_ < bufCount_ ? h_ : bufCount_;
        const uint64_t coreStride = static_cast<uint64_t>(aivNum_) * chunkLimit;
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunkLimit; offset < sourceCount; offset += coreStride) {
            const uint32_t current =
                sourceCount - offset < chunkLimit ? static_cast<uint32_t>(sourceCount - offset) : chunkLimit;
            auto accum = FloatBuf(HC_GRAD_REDUCE_ACCUM, current);
            auto slice = FloatBuf(HC_GRAD_REDUCE_SLICE, current);
            CopyIn<float>(partial0, sourceOffset + offset, accum, current);
            CopyIn<float>(partial1, sourceOffset + offset, slice, current);
            Add(accum, accum, slice, current);
            if (task0 == 0U) {
                CopyOut<float>(output, offset, accum, current);
            } else {
                CopyOut<float, true>(output, offset, accum, current);
            }
        }
    }

    __aicore__ inline void ReducePartialFloat2(uint64_t partialOffset, GM_ADDR output, uint32_t task0, uint32_t task1,
                                               uint64_t slotElements, uint64_t count, uint64_t sourceOffset = 0U)
    {
        GM_ADDR partial0 =
            Pointer(layout_.TaskBufferOffset(partialOffset, task0, workspaceSlots_, slotElements * sizeof(float)));
        GM_ADDR partial1 =
            Pointer(layout_.TaskBufferOffset(partialOffset, task1, workspaceSlots_, slotElements * sizeof(float)));
        const uint32_t chunkLimit = h_ < bufCount_ ? h_ : bufCount_;
        const uint64_t coreStride = static_cast<uint64_t>(aivNum_) * chunkLimit;
        auto accum = FloatBuf(HC_GRAD_REDUCE_ACCUM, chunkLimit);
        auto slice = FloatBuf(HC_GRAD_REDUCE_SLICE, chunkLimit);
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunkLimit; offset < count; offset += coreStride) {
            const uint32_t current = count - offset < chunkLimit ? static_cast<uint32_t>(count - offset) : chunkLimit;
            CopyIn<float>(partial0, sourceOffset + offset, accum, current);
            CopyIn<float>(partial1, sourceOffset + offset, slice, current);
            Add(accum, accum, slice, current);
            if (task0 == 0U) {
                CopyOut<float>(output, offset, accum, current);
            } else {
                CopyOut<float, true>(output, offset, accum, current);
            }
        }
    }

    __aicore__ inline void ReducePartialAtomic(uint64_t partialOffset, GM_ADDR output, uint32_t task,
                                               uint64_t slotElements, uint64_t sourceOffset, uint64_t sourceCount)
    {
        GM_ADDR partial =
            Pointer(layout_.TaskBufferOffset(partialOffset, task, workspaceSlots_, slotElements * sizeof(float)));
        const uint32_t chunkLimit = h_ < bufCount_ ? h_ : bufCount_;
        const uint64_t coreStride = static_cast<uint64_t>(aivNum_) * chunkLimit;
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunkLimit; offset < sourceCount; offset += coreStride) {
            const uint32_t current =
                sourceCount - offset < chunkLimit ? static_cast<uint32_t>(sourceCount - offset) : chunkLimit;
            auto accum = FloatBuf(HC_GRAD_REDUCE_ACCUM, current);
            CopyIn<float>(partial, sourceOffset + offset, accum, current);
            if (task == 0U) {
                CopyOut<float>(output, offset, accum, current);
            } else {
                CopyOut<float, true>(output, offset, accum, current);
            }
        }
    }

    __aicore__ inline void ReducePartialFloat(uint64_t partialOffset, GM_ADDR output, uint32_t task,
                                              uint64_t slotElements, uint64_t count, uint64_t sourceOffset = 0U)
    {
        GM_ADDR partial =
            Pointer(layout_.TaskBufferOffset(partialOffset, task, workspaceSlots_, slotElements * sizeof(float)));
        const uint32_t chunkLimit = h_ < bufCount_ ? h_ : bufCount_;
        const uint64_t coreStride = static_cast<uint64_t>(aivNum_) * chunkLimit;
        auto value = FloatBuf(HC_GRAD_REDUCE_ACCUM, chunkLimit);
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunkLimit; offset < count; offset += coreStride) {
            const uint32_t current = count - offset < chunkLimit ? static_cast<uint32_t>(count - offset) : chunkLimit;
            CopyIn<float>(partial, sourceOffset + offset, value, current);
            if (task == 0U) {
                CopyOut<float>(output, offset, value, current);
            } else {
                CopyOut<float, true>(output, offset, value, current);
            }
        }
    }

    __aicore__ inline void CopyWeightRows(GM_ADDR output, uint64_t sourceOffset, uint64_t count, bool accumulate,
                                          uint64_t partialOffset = ~0ULL)
    {
        const uint32_t chunkLimit = h_ < bufCount_ ? h_ : bufCount_;
        const uint64_t coreStride = static_cast<uint64_t>(aivNum_) * chunkLimit;
        auto value = FloatBuf(HC_GRAD_REDUCE_ACCUM, chunkLimit);
        for (uint64_t offset = static_cast<uint64_t>(aivId_) * chunkLimit; offset < count; offset += coreStride) {
            const uint32_t current = count - offset < chunkLimit ? static_cast<uint32_t>(count - offset) : chunkLimit;
            CopyIn<float>(Pointer(partialOffset == ~0ULL ? layout_.dwPartial : partialOffset), sourceOffset + offset,
                          value, current);
            if (accumulate) {
                CopyOut<float, true>(output, offset, value, current);
            } else {
                CopyOut<float>(output, offset, value, current);
            }
        }
    }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    bool blockedG56Xn_{false};
    bool preparedG56A_{false};
#endif
    GM_ADDR gmGradMixed_{nullptr};
    GM_ADDR gmGradInject_{nullptr};
    GM_ADDR gmX_{nullptr};
    GM_ADDR gmWDown_{nullptr};
    GM_ADDR gmWGate_{nullptr};
    GM_ADDR gmGamma_{nullptr};
    GM_ADDR gmInvRms_{nullptr};
    GM_ADDR gmInject_{nullptr};
    GM_ADDR gmDownPreAct_{nullptr};
    GM_ADDR gmUp_{nullptr};
    GM_ADDR gmDx_{nullptr};
    GM_ADDR gmDwDown_{nullptr};
    GM_ADDR gmDwUp_{nullptr};
    GM_ADDR gmDwGate_{nullptr};
    GM_ADDR gmDwDownOutput_{nullptr};
    GM_ADDR gmDwUpOutput_{nullptr};
    GM_ADDR gmDwGateOutput_{nullptr};
    GM_ADDR gmDgamma_{nullptr};
    GM_ADDR workspace_{nullptr};
    HcGradWorkspaceLayout layout_;
    TBuf<TPosition::VECCALC> ub_;
    uint32_t n_{0};
    uint32_t c_{0};
    uint32_t h_{0};
    uint32_t r_{0};
    uint32_t d_{0};
    uint32_t taskRows_{HC_GRAD_TASK_ROWS};
    bool directWeights_{false};
    bool recomputeDirect_{false};
    bool lowpDx_{false};
    uint32_t taskCount_{0};
    uint32_t workspaceSlots_{0};
    uint32_t splitkFactor_{0};
    uint32_t hasGate_{0};
    uint32_t g56Rows_{0};
    uint32_t gateRows_{0};
    uint32_t bufCount_{0};
    uint32_t poolBytes_{0};
    uint32_t rowBlock_{1};
    uint32_t pipeStages_{0};  // pipeStages=0 selects serial execution.
    uint32_t invStride_{0};   // invStride_ is the per-stage inv_rms size aligned to 32 bytes.
    uint32_t scalarBytes_{0}; // scalarBytes_ follows the actual rowBlock_.
    bool useGammaCache_{false};
    bool independentPipeOutput_{false};
    uint32_t fullBatchRowBegin_{0};
    uint32_t parity_{0};
    uint32_t ubBytes_{HC_GRAD_VECTOR_UB_FALLBACK};
    uint32_t aivId_{0};
    uint32_t aivNum_{0};
    uint32_t dgammaBytes_{0};
    uint32_t ubBufBase_{0};
    int32_t gammaCachedCh_{-1};
    bool kahanDgamma_{false};
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_VECTOR_ARCH35_H
