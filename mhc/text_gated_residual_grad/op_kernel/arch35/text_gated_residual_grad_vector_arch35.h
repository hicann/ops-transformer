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
        ZeroRangeT<D_T>(Pointer(layout_.packed), static_cast<uint64_t>(workspaceSlots_) * HC_GRAD_TASK_ROWS * g56Rows_);
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

    __aicore__ inline void Process(HcGradVectorOp op, uint32_t task)
    {
        if (op == HcGradVectorOp::DGAMMA) {
            StoreDgammaPartial();
            return;
        }
        if (op == HcGradVectorOp::REDUCE_DGAMMA) {
            if constexpr (HAS_GAMMA) {
                ReduceDgamma();
            }
            return;
        }
        const uint32_t rowBegin = task * HC_GRAD_TASK_ROWS;
        const uint32_t rows = n_ - rowBegin < HC_GRAD_TASK_ROWS ? n_ - rowBegin : HC_GRAD_TASK_ROWS;
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
            const uint32_t v6 = HC_GRAD_V6_POOL_BYTES * bufCount_ + HC_GRAD_V6_ROW_BYTES * h_;
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
        pipe->InitBuffer(ub_, ubBytes_);
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

    __aicore__ inline void StoreDgammaPartial()
    {
        if constexpr (HAS_GAMMA) {
            GM_ADDR partial = Pointer(layout_.dgammaPartial) +
                              static_cast<uint64_t>(aivId_) * HC_GRAD_DGAMMA_SLOTS * h_ * sizeof(float);
            for (uint32_t off = 0; off < h_; off += HC_GRAD_DGAMMA_COPY_CHUNK) {
                const uint32_t n = h_ - off < HC_GRAD_DGAMMA_COPY_CHUNK ? h_ - off : HC_GRAD_DGAMMA_COPY_CHUNK;
                auto src = ub_.GetWithOffset<float>(n, off * sizeof(float));
                CopyOut<float>(partial, off, src, n);
            }
            if (kahanDgamma_) {
                for (uint32_t off = 0; off < h_; off += HC_GRAD_DGAMMA_COPY_CHUNK) {
                    const uint32_t n = h_ - off < HC_GRAD_DGAMMA_COPY_CHUNK ? h_ - off : HC_GRAD_DGAMMA_COPY_CHUNK;
                    auto src = ub_.GetWithOffset<float>(n, dgammaBytes_ + off * sizeof(float));
                    CopyOut<float>(partial, static_cast<uint64_t>(h_) + off, src, n);
                }
            }
        }
    }

    __aicore__ inline void ProcessWeightReduction(HcGradVectorOp op, uint32_t task)
    {
        switch (op) {
            case HcGradVectorOp::REDUCE_WU2:
                if (taskCount_ == 1U) {
                    break;
                }
                ReducePartialAtomic2(layout_.dwPartial, gmDwUp_, task, task + 1U, layout_.dwSlotElems, 0,
                                     static_cast<uint64_t>(r_) * d_);
                break;
            case HcGradVectorOp::REDUCE_WD2:
                if (taskCount_ == 1U) {
                    break;
                }
                ReducePartialAtomic2(layout_.dwPartial, gmDwDown_, task, task + 1U, layout_.dwSlotElems, 0,
                                     static_cast<uint64_t>(r_) * d_);
                if (hasGate_ != 0) {
                    ReducePartialFloat2(layout_.dwPartial, gmDwGate_, task, task + 1U, layout_.dwSlotElems,
                                        static_cast<uint64_t>(c_) * d_, layout_.dwGateRowElems);
                }
                break;
            case HcGradVectorOp::REDUCE_WU:
                if (taskCount_ == 1U) {
                    break;
                }
                ReducePartialAtomic(layout_.dwPartial, gmDwUp_, task, layout_.dwSlotElems, 0,
                                    static_cast<uint64_t>(r_) * d_);
                break;
            case HcGradVectorOp::REDUCE_WD:
                // For multiple tasks, G56 writes [R+gateRows,D]: the first R rows are dwDown, followed by C dwGate
                // rows. For one task, GEMM writes dwDown directly to its FP32 accumulator; only dwGate needs copying
                // here.
                if (taskCount_ != 1U) {
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
        const uint32_t slot = task % workspaceSlots_;
        return Pointer(offset) + static_cast<uint64_t>(slot) * HC_GRAD_TASK_ROWS * rowStride * elementSize;
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
        return ub_.GetWithOffset<float>(count, ubBufBase_ + index * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<D_T> HalfBuf(uint32_t index, uint32_t count)
    {
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + HC_GRAD_SERIAL_FLOAT_SLOTS * bufCount_ * sizeof(float) +
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

    __aicore__ inline LocalTensor<float> SumSlot(uint32_t row)
    {
        return ub_.GetWithOffset<float>(1, ScalarOffset() + row * HC_GRAD_DATA_BLOCK_BYTES);
    }

    __aicore__ inline LocalTensor<float> ResidualSlot(uint32_t row)
    {
        return ub_.GetWithOffset<float>(1, ScalarOffset() + (rowBlock_ + row) * HC_GRAD_DATA_BLOCK_BYTES);
    }

    // Keep the mean and centered residual separate to avoid rounding a small correction back into one FP32 mean.
    // For non-finite means, retain the original projection without centering Inf/NaN values.
    __aicore__ inline void ReduceRmsResidual(LocalTensor<float> product, LocalTensor<float> scratch, uint32_t row)
    {
        const float mean = SumSlot(row).GetValue(0) / static_cast<float>(static_cast<int32_t>(h_));
        if (mean <= HC_GRAD_MAX_FINITE_FLOAT && mean >= -HC_GRAD_MAX_FINITE_FLOAT) {
            Adds(product, product, -mean, h_);
            ReduceSum(ResidualSlot(row), product, scratch, h_);
        } else {
            Duplicate(ResidualSlot(row), 0.0f, 1U);
        }
    }

    // V1/V6 alternate two input stages; each 16-bit output reuses a consumed input slot.
    // Prefetch overlaps the latter part of vector computation. MTE2_S is stage-specific so
    // scalar waits do not include the next stage. MTE3_MTE2 protects output slots before reuse.
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

    // V1 pool offsets.
    __aicore__ inline LocalTensor<float> V1Scratch(uint32_t index, uint32_t count)
    {
        // Block-sized FP32 slots: 0 = x to xn, 1 = gate, 2 = temp.
        return ub_.GetWithOffset<float>(count, ubBufBase_ + index * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<float> V1DirectOut(uint32_t stage, uint32_t count)
    {
        return ub_.GetWithOffset<float>(count,
                                        ubBufBase_ + (HC_GRAD_V1_SCRATCH_SLOTS + stage) * bufCount_ * sizeof(float));
    }

    __aicore__ inline LocalTensor<D_T> V1InHalf(uint32_t stage, uint32_t index, uint32_t count)
    {
        // Three input slots per stage: x, up and grad_mixed.
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + HC_GRAD_V1_FLOAT_SLOTS * bufCount_ * sizeof(float) +
                                                 (stage * HC_GRAD_V1_INPUT_SLOTS + index) * bufCount_ * sizeof(D_T));
    }

    __aicore__ inline LocalTensor<D_T> V1OutHalf(uint32_t stage, uint32_t index, uint32_t count)
    {
        // After consumption, xn reuses the x slot and dup reuses the up slot. Each stage thus has its own
        // outputs without extra UB, allowing MTE3 to overlap computation of the next block.
        return V1InHalf(stage, index, count);
    }

    // V6 pool offsets.
    __aicore__ inline LocalTensor<float> V6InFloat(uint32_t stage, uint32_t index, uint32_t count)
    {
        // Two FP32 input slots per stage: direct and dx2.
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
        const uint32_t base =
            HC_GRAD_V6_FLOAT_SLOTS * bufCount_ * sizeof(float) + HC_GRAD_V6_ROW_SLOTS * h_ * sizeof(float);
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + base + stage * bufCount_ * sizeof(D_T));
    }

    __aicore__ inline LocalTensor<D_T> V6OutHalf(uint32_t stage, uint32_t count)
    {
        const uint32_t base =
            HC_GRAD_V6_FLOAT_SLOTS * bufCount_ * sizeof(float) + HC_GRAD_V6_ROW_SLOTS * h_ * sizeof(float);
        // dx reuses this stage's x input slot after Cast consumes xHalf, avoiding another UB output buffer.
        return ub_.GetWithOffset<D_T>(count, ubBufBase_ + base + stage * bufCount_ * sizeof(D_T));
    }

    // Prime V_MTE2, S_MTE2 and MTE3_MTE2 for each stage so the first wait cannot deadlock.
    __aicore__ inline void PipePrime()
    {
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(s));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(s));
        }
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(s));
        }
    }

    __aicore__ inline void PipeDrain()
    {
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            WaitFlag<HardEvent::V_MTE2>(PipeInEvent(s));
            WaitFlag<HardEvent::S_MTE2>(PipeInEvent(s));
        }
        for (uint32_t s = 0U; s < HC_GRAD_PIPE_MAX_STAGES; ++s) {
            WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(s));
        }
        // The next operation may reuse the pool for vector writes, such as clearing pending dgamma in V6.
        // MTE3_MTE2 protects subsequent input DMA only; it does not protect vector writes from outstanding MTE3 reads.
        SetFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE3_V>(HC_GRAD_VECTOR_EVENT);
    }

    // DMA helpers issue no flags; the pipeline controls synchronization.
    template <typename T>
    __aicore__ inline void IssueCopyInRows(GM_ADDR src, uint64_t srcRow0ElementOffset, uint32_t rowStride,
                                           const LocalTensor<T>& dst, uint32_t n, uint32_t count)
    {
        if (n == 0U || count == 0U) {
            return;
        }
        GlobalTensor<T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyExtParams params{static_cast<uint16_t>(n), static_cast<uint32_t>(count * sizeof(T)),
                                 static_cast<HcGradCopyStride>((rowStride - count) * sizeof(T)), 0, 0};
        DataCopyPad(dst, source[srcRow0ElementOffset], params, pad);
    }

    template <typename T>
    __aicore__ inline void IssueCopyIn(GM_ADDR src, uint64_t srcElementOffset, const LocalTensor<T>& dst,
                                       uint32_t count)
    {
        if (count == 0U) {
            return;
        }
        GlobalTensor<T> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, source[srcElementOffset], params, pad);
    }

    template <typename T>
    __aicore__ inline void IssueCopyOutRows(GM_ADDR dst, uint64_t dstRow0ElementOffset, uint32_t rowStride,
                                            const LocalTensor<T>& src, uint32_t n, uint32_t count)
    {
        if (n == 0U || count == 0U) {
            return;
        }
        GlobalTensor<T> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));
        DataCopyExtParams params{static_cast<uint16_t>(n), static_cast<uint32_t>(count * sizeof(T)), 0,
                                 static_cast<HcGradCopyStride>((rowStride - count) * sizeof(T)), 0};
        DataCopyPad(target[dstRow0ElementOffset], src, params);
    }

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
    __aicore__ inline void SigmoidFp32(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t count)
    {
        const uint32_t base = SigmoidOffset();
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
        const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + rowBegin;
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
        CopyInRows<D_T>(gmUp_, inputOffset, d_, upHalf, n, h_, false);
        // grad_mixed [B,S,H] is contiguous across the n rows.
        CopyIn<D_T>(gmGradMixed_, token * h_, gmHalf, len, false);
        // inv_rms contains n*C contiguous values and needs one transfer.
        auto invBuf = ub_.GetWithOffset<float>(n * c_, InvOffset());
        CopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_, false);
        SyncCopy();
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
        Cast(up, upHalf, RoundMode::CAST_NONE, len);
        Cast(gradMixed, gmHalf, RoundMode::CAST_NONE, len);
        SetFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::MTE2_S>(HC_GRAD_VECTOR_EVENT);
        NormalizeV1Rows(n, channel, x, xnFloat, invBuf);
        Cast(xnHalf, xnFloat, RoundMode::CAST_RINT, len);
        CopyOutRows<D_T>(xn, outputOffset, d_, xnHalf, n, h_);
        SigmoidFp32(gate, up, len);
        Mul(temp, gradMixed, gate, len);
        ScaleByInvC(temp, temp, len);
        CopyOutRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), outputOffset, d_, temp, n, h_);
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
        const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + base;
        const uint64_t inputOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto xHalf = V1InHalf(stage, HC_GRAD_V1_X, len);
        auto upHalf = V1InHalf(stage, HC_GRAD_V1_UP, len);
        auto gmHalf = V1InHalf(stage, HC_GRAD_V1_GRAD_MIXED, len);
        auto invBuf = ub_.GetWithOffset<float>(n * c_, InvOffset(stage));
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        // Scalar reads of inv_rms require S_MTE2 before overwrite; V_MTE2 protects vector reads only.
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        // Wait for the previous MTE3 transfer before reusing an input slot that also holds output.
        WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
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
        auto invBuf = ub_.GetWithOffset<float>(n * c_, InvOffset(stage));
        Cast(x, xHalf, RoundMode::CAST_NONE, len);
        Cast(gate, upHalf, RoundMode::CAST_NONE, len);
        Cast(temp, gmHalf, RoundMode::CAST_NONE, len);
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        // The first Cast gives the previous MTE3 transfer time to finish. Prefetch the next block afterward
        // so its MTE2 transfer overlaps the remaining computation, using stage-specific events.
        if (blk + 1U < blocks) {
            IssueV1In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
        NormalizeV1PipeRows(n, channel, x, directOut, invBuf);
        Cast(xnHalf, x, RoundMode::CAST_RINT, len);
        SigmoidFp32(gate, gate, len);
        // direct = grad_mixed * gate / C. temp already holds grad_mixed, so write direct to its output buffer
        // without another block copy or cast.
        Mul(directOut, temp, gate, len);
        ScaleByInvC(directOut, directOut, len);
        // dup = gm * xn / C * gate * (1 - gate) = direct * xn * (1 - gate):
        // Reuse direct to avoid multiplying grad_mixed and 1/C again.
        Mul(temp, directOut, x, len);
        Muls(gate, gate, -1.0f, len);
        Adds(gate, gate, 1.0f, len);
        Mul(temp, temp, gate, len);
        Cast(dupHalf, temp, RoundMode::CAST_RINT, len);
    }

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
        IssueCopyOutRows<D_T>(TaskPointer(layout_.xn, task, d_, sizeof(D_T)), outputOffset, d_, xnHalf, n, h_);
        IssueCopyOutRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), outputOffset, d_, directOut, n,
                                h_);
        IssueCopyOutRows<D_T>(TaskPointer(layout_.dup, task, d_, sizeof(D_T)), outputOffset, d_, dupHalf, n, h_);
        SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
    }

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
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            ComputeV1(task, rowBegin, rowEnd, blk, blocks, channel, stage);
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV1Out(task, rowBegin, rowEnd, blk, channel, stage);
        }
        PipeDrain();
    }

    __aicore__ inline void ProcessV1(uint32_t task, uint32_t rows)
    {
        GM_ADDR act = TaskPointer(layout_.act, task, r_, sizeof(D_T));
        // down_pre_act and act [token,R] are contiguous across n rows.
        {
            const uint32_t actRowsPerCore = (rows + aivNum_ - 1U) / aivNum_;
            const uint32_t actBegin = aivId_ * actRowsPerCore;
            const uint32_t actEnd = (actBegin + actRowsPerCore < rows) ? (actBegin + actRowsPerCore) : rows;
            uint32_t blk = bufCount_ / r_;
            if (blk < 1U) {
                blk = 1U;
            }
            for (uint32_t base = actBegin; base < actEnd; base += blk) {
                const uint32_t nb = (actEnd - base < blk) ? (actEnd - base) : blk;
                const uint32_t len = nb * r_;
                const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + base;
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

        for (uint32_t channel = aivId_ % c_; channel < c_; channel += aivNum_) {
            const bool firstWave = channel < aivNum_;
            const uint32_t lane = firstWave ? aivId_ / c_ : 0U;
            const uint32_t laneCount = firstWave ? (aivNum_ - 1U - channel) / c_ + 1U : 1U;
            // Partition rows contiguously across cores; row counts differ by at most one.
            const uint32_t rowsPerCore = (rows + laneCount - 1U) / laneCount;
            const uint32_t rowBegin = lane * rowsPerCore;
            const uint32_t rowEnd = (rowBegin + rowsPerCore < rows) ? (rowBegin + rowsPerCore) : rows;
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
            const uint64_t tok = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + row;
            for (uint32_t cOff = 0; cOff < c_; cOff += gateChunk) {
                const uint32_t count = c_ - cOff < gateChunk ? c_ - cOff : gateChunk;
                CopyIn<D_T>(gmGradInject_, tok * c_ + cOff, giHalf, count);
                CopyIn<D_T>(gmInject_, tok * c_ + cOff, injectHalf, count);
                Cast(gradInject, giHalf, RoundMode::CAST_NONE, count);
                Cast(inject, injectHalf, RoundMode::CAST_NONE, count);
                Muls(inject, inject, HC_GRAD_INJECT_TO_SIGMOID, count);
                Muls(temp, inject, -1.0f, count);
                Adds(temp, temp, 1.0f, count);
                Muls(inject, inject, HC_GRAD_SIGMOID_TO_INJECT, count);
                Mul(temp, inject, temp, count);
                Mul(gradInject, gradInject, temp, count);
                ScaleByInvC(gradInject, gradInject, count);
                Cast(dlogitsHalf, gradInject, RoundMode::CAST_RINT, count);
                CopyOut<D_T>(packed, static_cast<uint64_t>(row) * g56Rows_ + r_ + cOff, dlogitsHalf, count);
            }
        }
    }

    __aicore__ inline void ProcessV3(uint32_t task, uint32_t rows)
    {
        GM_ADDR dact = TaskPointer(layout_.dact, task, r_, sizeof(float));
        GM_ADDR packed = TaskPointer(layout_.packed, task, g56Rows_, sizeof(D_T));
        // As in ReduceDact, each core consumes the rows it produced.
        const uint32_t rowsPerCoreV3 = (rows + aivNum_ - 1U) / aivNum_;
        const uint32_t rowBeginV3 = aivId_ * rowsPerCoreV3;
        const uint32_t rowEndV3 = (rowBeginV3 + rowsPerCoreV3 < rows) ? (rowBeginV3 + rowsPerCoreV3) : rows;
        // dact and down_pre_act are contiguous across n rows; packed uses a g56Rows row stride.
        uint32_t blk = bufCount_ / r_;
        if (blk < 1U) {
            blk = 1U;
        }
        for (uint32_t base = rowBeginV3; base < rowEndV3; base += blk) {
            const uint32_t nb = (rowEndV3 - base < blk) ? (rowEndV3 - base) : blk;
            const uint32_t len = nb * r_;
            const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + base;
            auto dactFloat = FloatBuf(HC_GRAD_V3_DACT, len);
            auto down = FloatBuf(HC_GRAD_V3_DOWN, len);
            auto sigmoid = FloatBuf(HC_GRAD_V3_SIGMOID, len);
            auto derivative = FloatBuf(HC_GRAD_V3_DERIVATIVE, len);
            auto downHalf = HalfBuf(HC_GRAD_HALF_INPUT, len);
            auto ddownHalf = HalfBuf(HC_GRAD_HALF_IO_SCRATCH, len);
            CopyIn<float>(dact, static_cast<uint64_t>(base) * r_, dactFloat, len);
            CopyIn<D_T>(gmDownPreAct_, token * r_, downHalf, len);
            Cast(down, downHalf, RoundMode::CAST_NONE, len);
            ScaleByInvC(down, down, len);
            SigmoidFp32(sigmoid, down, len);
            Mul(dactFloat, dactFloat, sigmoid, len);
            Muls(derivative, sigmoid, -1.0f, len);
            Adds(derivative, derivative, 1.0f, len);
            Mul(derivative, derivative, down, len);
            Adds(derivative, derivative, 1.0f, len);
            Mul(dactFloat, dactFloat, derivative, len);
            ScaleByInvC(dactFloat, dactFloat, len);
            Cast(ddownHalf, dactFloat, RoundMode::CAST_RINT, len);
            CopyOutRows<D_T>(packed, static_cast<uint64_t>(base) * g56Rows_, g56Rows_, ddownHalf, nb, r_);
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
                DataCopy(accum[i * HC_GRAD_TASK_ROWS + rowBase], transposed[i * HC_GRAD_TRANSPOSE_TILE],
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
                accum[i * HC_GRAD_TASK_ROWS + rowBegin * HC_GRAD_TRANSPOSE_TILE], outParams);
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
        auto accum = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS, accumOffset);
        auto tile = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                           accumOffset + HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS * sizeof(D_T));
        auto transposed =
            ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                   accumOffset + HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS * sizeof(D_T) +
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
            StoreTransposeRows(target, accum, rowBegin, rowEnd, rows, columnBegin, validCols, HC_GRAD_TASK_ROWS);
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
                DataCopy(accum[i * HC_GRAD_TASK_ROWS + rowOffset], transposed[i * HC_GRAD_TRANSPOSE_TILE],
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
        auto accum = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS, accumOffset);
        auto tile = ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                           accumOffset + HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS * sizeof(D_T));
        auto transposed =
            ub_.GetWithOffset<D_T>(HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE,
                                   accumOffset + HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS * sizeof(D_T) +
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
            ReduceSum(SumSlot(r), product[off], reduceWork[off], h_);
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t off = r * h_;
            ReduceRmsResidual(product[off], reduceWork[off], r);
        }
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
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
        const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + rowBegin;
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
        CopyInRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)),
                          static_cast<uint64_t>(rowBegin) * d_ + static_cast<uint64_t>(channel) * h_, d_, total, n, h_,
                          false);
        CopyInRows<float>(TaskPointer(layout_.dx2, task, d_, sizeof(float)),
                          static_cast<uint64_t>(rowBegin) * d_ + static_cast<uint64_t>(channel) * h_, d_, dx2f, n, h_,
                          false);
        CopyInRows<D_T>(gmX_, token * d_ + static_cast<uint64_t>(channel) * h_, d_, xHalf, n, h_, false);
        auto invBuf = ub_.GetWithOffset<float>(n * c_, InvOffset(0U));
        CopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_, false);
        SyncCopy();
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

    // V6 uses per-stage direct/dx2/x input and dx output slots to overlap MTE2/MTE3 with vector computation.
    __aicore__ inline void IssueV6In(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t channel,
                                     uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
        const uint32_t len = n * h_;
        const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + base;
        // direct and dx2 use the task's row base with stride D and the current channel's H-element segment.
        // x uses the global token row base.
        const uint64_t rowOffset = static_cast<uint64_t>(base) * d_ + static_cast<uint64_t>(channel) * h_;
        const uint64_t xOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto xHalf = V6InHalf(stage, len);
        auto directIn = V6InFloat(stage, HC_GRAD_V6_DIRECT_INPUT, len);
        auto dx2In = V6InFloat(stage, HC_GRAD_V6_DX2_INPUT, len);
        auto invBuf = ub_.GetWithOffset<float>(n * c_, InvOffset(stage));
        WaitFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
        WaitFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
        IssueCopyInRows<float>(TaskPointer(layout_.direct, task, d_, sizeof(float)), rowOffset, d_, directIn, n, h_);
        IssueCopyInRows<float>(TaskPointer(layout_.dx2, task, d_, sizeof(float)), rowOffset, d_, dx2In, n, h_);
        IssueCopyInRows<D_T>(gmX_, xOffset, d_, xHalf, n, h_);
        IssueCopyIn<float>(gmInvRms_, token * c_, invBuf, n * c_);
        SetFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
        SetFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
    }

    __aicore__ inline void ReduceV6Moments(uint32_t n, uint32_t channel, LocalTensor<float> total,
                                           LocalTensor<float> xBuf, LocalTensor<float> gnRow,
                                           LocalTensor<float> product, LocalTensor<float> correction,
                                           LocalTensor<float> reduceWork, LocalTensor<float> invBuf)
    {
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
                ReduceSum(SumSlot(r), product, reduceWork, h_);
            }
        } else {
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t off = r * h_;
                const float inv = invBuf.GetValue(r * c_ + channel);
                Muls(xBuf[off], xBuf[off], inv, h_);
                Adds(gnRow, total[off], 0.0f, h_);
                Mul(product, gnRow, xBuf[off], h_);
                ReduceSum(SumSlot(r), product, reduceWork, h_);
            }
        }
    }

    __aicore__ inline void ProjectV6PipeRows(uint32_t n, uint32_t channel, LocalTensor<float> total,
                                             LocalTensor<float> xBuf, LocalTensor<float> gnRow,
                                             LocalTensor<float> product, LocalTensor<float> reduceWork,
                                             LocalTensor<float> invBuf, LocalTensor<D_T> dxHalf)
    {
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

    __aicore__ inline void ComputeV6(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t blk, uint32_t blocks,
                                     uint32_t channel, uint32_t stage)
    {
        const uint32_t base = rowBegin + blk * rowBlock_;
        const uint32_t n = (rowEnd - base < rowBlock_) ? (rowEnd - base) : rowBlock_;
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
        auto invBuf = ub_.GetWithOffset<float>(n * c_, InvOffset(stage));
        Cast(xBuf, xHalf, RoundMode::CAST_NONE, len);
        Add(total, total, dx2In, len); // Inline total = direct + dxnorm in FP32.
        WaitFlag<HardEvent::MTE2_S>(PipeScalarEvent(stage));
        // Cast/Add allow the previous MTE3 transfer to progress; prefetch overlaps the remaining row computation.
        if (blk + 1U < blocks) {
            IssueV6In(task, rowBegin, rowEnd, blk + 1U, channel, stage ^ HC_GRAD_PIPE_STAGE_MASK);
        }
        ReduceV6Moments(n, channel, total, xBuf, gnRow, product, correction, reduceWork, invBuf);
        SetFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        WaitFlag<HardEvent::V_S>(HC_GRAD_VECTOR_EVENT);
        ProjectV6PipeRows(n, channel, total, xBuf, gnRow, product, reduceWork, invBuf, dxHalf);
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
        const uint64_t token = static_cast<uint64_t>(task) * HC_GRAD_TASK_ROWS + base;
        const uint64_t outOffset = token * d_ + static_cast<uint64_t>(channel) * h_;
        auto dxHalf = V6OutHalf(stage, len);
        SetFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        WaitFlag<HardEvent::V_MTE3>(PipeOutEvent(stage));
        IssueCopyOutRows<D_T>(gmDx_, outOffset, d_, dxHalf, n, h_);
        SetFlag<HardEvent::MTE3_MTE2>(PipeOutEvent(stage));
    }

    __aicore__ inline void ProcessV6Pipe(uint32_t task, uint32_t rowBegin, uint32_t rowEnd, uint32_t channel)
    {
        if (rowEnd <= rowBegin) {
            return;
        }
        const uint32_t blocks = (rowEnd - rowBegin + rowBlock_ - 1U) / rowBlock_;
        // Other operations overwrite the pending pool, so clear it on entry. Flush every four blocks
        // and at task end. Up to 4*rowBlock sequential additions precede each Kahan update;
        // Kahan cannot recover rounding error already introduced inside that batch.
        if constexpr (HAS_GAMMA) {
            auto pending = V6Row(HC_GRAD_V6_ROW_PENDING, h_);
            Duplicate(pending, 0.0f, h_);
        }
        PipePrime();
        IssueV6In(task, rowBegin, rowEnd, 0U, channel, 0U);
        for (uint32_t blk = 0U; blk < blocks; ++blk) {
            const uint32_t stage = blk & HC_GRAD_PIPE_STAGE_MASK;
            WaitFlag<HardEvent::MTE2_V>(PipeInEvent(stage));
            ComputeV6(task, rowBegin, rowEnd, blk, blocks, channel, stage);
            if constexpr (HAS_GAMMA) {
                if (((blk + 1U) % HC_GRAD_DGAMMA_FLUSH_BLOCKS) == 0U || (blk + 1U) == blocks) {
                    FlushDgammaPending();
                }
            }
            SetFlag<HardEvent::V_MTE2>(PipeInEvent(stage));
            SetFlag<HardEvent::S_MTE2>(PipeInEvent(stage));
            IssueV6Out(task, rowBegin, rowEnd, blk, channel, stage);
        }
        PipeDrain();
    }

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
        for (uint32_t channel = aivId_; channel < c_; channel += aivNum_) {
            // The first channel wave uses cores channel, channel+c, and so on. Later channel waves use
            // only channel % aivNum.
            const uint32_t first = (channel < aivNum_) ? channel : (channel % aivNum_);
            const uint32_t step = (channel < aivNum_) ? c_ : aivNum_;
            for (uint32_t off = 0; off < h_; off += chunkLimit) {
                const uint32_t n = (h_ - off < chunkLimit) ? (h_ - off) : chunkLimit;
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
        const uint32_t slot = task % workspaceSlots_;
        GM_ADDR partial = Pointer(layout_.dactPartial) +
                          static_cast<uint64_t>(slot) * splitkFactor_ * HC_GRAD_TASK_ROWS * r_ * sizeof(float);
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
        const uint32_t slot0 = task0 % workspaceSlots_;
        const uint32_t slot1 = task1 % workspaceSlots_;
        GM_ADDR partial0 = Pointer(partialOffset) + static_cast<uint64_t>(slot0) * slotElements * sizeof(float);
        GM_ADDR partial1 = Pointer(partialOffset) + static_cast<uint64_t>(slot1) * slotElements * sizeof(float);
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
        const uint32_t slot0 = task0 % workspaceSlots_;
        const uint32_t slot1 = task1 % workspaceSlots_;
        GM_ADDR partial0 = Pointer(partialOffset) + static_cast<uint64_t>(slot0) * slotElements * sizeof(float);
        GM_ADDR partial1 = Pointer(partialOffset) + static_cast<uint64_t>(slot1) * slotElements * sizeof(float);
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
        const uint32_t slot = task % workspaceSlots_;
        GM_ADDR partial = Pointer(partialOffset) + static_cast<uint64_t>(slot) * slotElements * sizeof(float);
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
        const uint32_t slot = task % workspaceSlots_;
        GM_ADDR partial = Pointer(partialOffset) + static_cast<uint64_t>(slot) * slotElements * sizeof(float);
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
