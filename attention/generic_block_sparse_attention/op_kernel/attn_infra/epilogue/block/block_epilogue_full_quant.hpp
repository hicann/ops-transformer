/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GBSA_BLOCK_EPILOGUE_FULL_QUANT_HPP
#define GBSA_BLOCK_EPILOGUE_FULL_QUANT_HPP

#include "../../arch/gbsa_resource.hpp"
#include "../../../arch22/gbsa_full_quant_utils.hpp"
#include "../../../arch22/kernel_utils.hpp"

namespace GbsaKernelArch22::Int8 {

using namespace NpuArch;

// Byte offsets in each AIV's UB. Padding does not extend the valid GM read range.
struct UbLayout {
    static constexpr uint32_t S_INPUT = 0;
    static constexpr uint32_t SCORES = 32 * FullQuantLayout::KIB;
    static constexpr uint32_t P_INT32 = 48 * FullQuantLayout::KIB;
    static constexpr uint32_t P_FP16 = 64 * FullQuantLayout::KIB;
    static constexpr uint32_t P_OUTPUT = 72 * FullQuantLayout::KIB;
    static constexpr uint32_t O_INPUT = 80 * FullQuantLayout::KIB;
    static constexpr uint32_t O_TEMP = 96 * FullQuantLayout::KIB;
    static constexpr uint32_t O_ACC = 104 * FullQuantLayout::KIB;
    static constexpr uint32_t Q_SCALE = 168 * FullQuantLayout::KIB;
    static constexpr uint32_t K_SCALE = 169 * FullQuantLayout::KIB;
    static constexpr uint32_t V_SCALE = 173 * FullQuantLayout::KIB;
    static constexpr uint32_t P_SCALE = 174 * FullQuantLayout::KIB;
    static constexpr uint32_t RUNNING_MAX = 175 * FullQuantLayout::KIB;
    static constexpr uint32_t RUNNING_SUM = 176 * FullQuantLayout::KIB;
    // One version per pending PV; a single alpha would be overwritten by prelaunch.
    static constexpr uint32_t ALPHA = 177 * FullQuantLayout::KIB;
    static constexpr uint32_t ALPHA_SLOT_BYTES = FullQuantLayout::MAX_VECTOR_ROWS * 32U;
    static constexpr uint32_t REDUCE_MAX = 183 * FullQuantLayout::KIB;
    static constexpr uint32_t REDUCE_SUM = 184 * FullQuantLayout::KIB;
    // Four independent chunks allow all scale DMA to be issued before Gather.
    // They alias only transient buffers, never O_ACC, statistics or fixed offsets.
    // Kscale scratch is dead before S is loaded, so reuse the transient S/quant buffers.
    static constexpr uint32_t K_SCALE_GATHER = S_INPUT;
    static constexpr uint32_t K_SCALE_CHUNK_BYTES = 16 * FullQuantLayout::KIB;
    static constexpr uint32_t K_SCALE_OFFSETS = 185 * FullQuantLayout::KIB;
    static constexpr uint32_t END = K_SCALE_OFFSETS + FullQuantLayout::KV_TILE * sizeof(uint32_t);
};

static_assert(UbLayout::END <= 192 * FullQuantLayout::KIB);
// Two 16-row S buffers are converted to FP32 in place. Reduction scratch
// becomes FP16 quantization scratch; neither overlaps the S input buffers.
struct WideUbLayout {
    static constexpr uint32_t S_INPUT = 0;
    static constexpr uint32_t SCORES = 0;
    static constexpr uint32_t P_INT32 = 64 * FullQuantLayout::KIB;
    static constexpr uint32_t P_FP16 = P_INT32;
    static constexpr uint32_t P_OUTPUT = 80 * FullQuantLayout::KIB;
    static constexpr uint32_t O_INPUT = 0;
    static constexpr uint32_t O_TEMP = 32 * FullQuantLayout::KIB;
    static constexpr uint32_t O_ACC = 96 * FullQuantLayout::KIB;
    static constexpr uint32_t Q_SCALE = 160 * FullQuantLayout::KIB;
    static constexpr uint32_t K_SCALE = 161 * FullQuantLayout::KIB;
    static constexpr uint32_t V_SCALE = 165 * FullQuantLayout::KIB;
    static constexpr uint32_t P_SCALE = 166 * FullQuantLayout::KIB;
    static constexpr uint32_t RUNNING_MAX = 167 * FullQuantLayout::KIB;
    static constexpr uint32_t RUNNING_SUM = 168 * FullQuantLayout::KIB;
    static constexpr uint32_t ALPHA = 169 * FullQuantLayout::KIB;
    static constexpr uint32_t ALPHA_SLOT_BYTES = UbLayout::ALPHA_SLOT_BYTES;
    static constexpr uint32_t REDUCE_MAX = 175 * FullQuantLayout::KIB;
    static constexpr uint32_t REDUCE_SUM = 176 * FullQuantLayout::KIB;
    static constexpr uint32_t K_SCALE_GATHER = 0;
    static constexpr uint32_t K_SCALE_CHUNK_BYTES = UbLayout::K_SCALE_CHUNK_BYTES;
    static constexpr uint32_t K_SCALE_OFFSETS = 177 * FullQuantLayout::KIB;
    static constexpr uint32_t END = 179 * FullQuantLayout::KIB;
};
// The existing G<=16 tile gives each AIV at most eight rows.
// S and P copies are drained before reuse; O input has an explicit per-round reuse fence.
struct CompactUbLayout {
    static constexpr uint32_t S_INPUT = 0;
    static constexpr uint32_t SCORES = S_INPUT;
    static constexpr uint32_t P_INT32 = 16 * FullQuantLayout::KIB;
    static constexpr uint32_t P_FP16 = P_INT32;
    static constexpr uint32_t P_OUTPUT = 24 * FullQuantLayout::KIB;
    static constexpr uint32_t O_INPUT = 28 * FullQuantLayout::KIB;
    static constexpr uint32_t O_TEMP = 36 * FullQuantLayout::KIB;
    static constexpr uint32_t O_ACC = 44 * FullQuantLayout::KIB;
    static constexpr uint32_t Q_SCALE = 52 * FullQuantLayout::KIB;
    static constexpr uint32_t K_SCALE = Q_SCALE + 32U;
    static constexpr uint32_t V_SCALE = K_SCALE + 2U * FullQuantLayout::KV_TILE * sizeof(float);
    static constexpr uint32_t P_SCALE = V_SCALE + FullQuantLayout::HEAD_DIM * sizeof(float);
    static constexpr uint32_t RUNNING_MAX = P_SCALE + 32U;
    static constexpr uint32_t RUNNING_SUM = RUNNING_MAX + 32U;
    static constexpr uint32_t ALPHA = RUNNING_SUM + 32U;
    static constexpr uint32_t ALPHA_SLOT_BYTES = FullQuantLayout::VECTOR_ROWS * 32U;
    static constexpr uint32_t REDUCE_MAX = ALPHA + FullQuantLayout::STAGES * ALPHA_SLOT_BYTES;
    static constexpr uint32_t REDUCE_SUM = REDUCE_MAX + 32U;
    static constexpr uint32_t K_SCALE_GATHER = S_INPUT;
    static constexpr uint32_t K_SCALE_CHUNK_BYTES = 1024U;
    static constexpr uint32_t K_SCALE_OFFSETS = REDUCE_SUM + 32U;
    static constexpr uint32_t END = K_SCALE_OFFSETS + FullQuantLayout::PAGE_TOKENS * sizeof(uint32_t);
};
static_assert(CompactUbLayout::END + FullQuantLayout::MAX_VECTOR_ROWS * 32U <= 64U * FullQuantLayout::KIB);
template <bool singleRowTile, bool compactUb = false>
struct VectorUbLayout : WideUbLayout {};
template <>
struct VectorUbLayout<true, false> : UbLayout {};
template <>
struct VectorUbLayout<true, true> : CompactUbLayout {};
static_assert(WideUbLayout::END <= 192 * FullQuantLayout::KIB);

// The 8 KiB plan occupies FP16 quantization scratch only before Softmax.
struct UbCopyPlan {
    AscendC::LocalTensor<uint64_t> words;
    __aicore__ inline void SetSpan(uint32_t index, uint64_t token, uint32_t length)
    {
        words.SetValue(index * 2U, token);
        words.SetValue(index * 2U + 1U, static_cast<uint64_t>(length));
    }
    __aicore__ inline CopySpan GetSpan(uint32_t index) const
    {
        return {words.GetValue(index * 2U), static_cast<uint32_t>(words.GetValue(index * 2U + 1U))};
    }
};

template <class ElementO, bool singleRowTile = false, bool compactUb = false>
class GbsaFullQuantVectorArch22 {
    using BufferLayout = VectorUbLayout<singleRowTile, compactUb>;
    static_assert(!compactUb || singleRowTile);
    static constexpr uint32_t BATCH_ROWS = singleRowTile ? 8U : 16U;
    using Resource = NpuArch::Arch::Int8StaticResource;
    // Vector operands must start at a 32-byte boundary, including one-element Exp.
    static constexpr uint32_t STAT_STRIDE = 32 / sizeof(float);

public:
    __aicore__ inline explicit GbsaFullQuantVectorArch22(Resource& resource)
        : resource_(resource)
    {}

    __aicore__ inline void Init(uint32_t rows, AscendC::GlobalTensor<float> qScale, AscendC::GlobalTensor<float> vScale,
                                AscendC::GlobalTensor<float> pScale, uint32_t headDim = FullQuantLayout::HEAD_DIM,
                                uint32_t scaleHeadStride = 1U)
    {
        qScaleReady_ = false;
        headDim_ = headDim;
        alignedDim_ = (headDim + 31U) / 32U * 32U;
        const uint32_t split = rows / AscendC::GetSubBlockNum();
        rowStart_ = AscendC::GetSubBlockIdx() * split;
        totalRows_ = AscendC::GetSubBlockIdx() == 0U ? split : rows - split;
        rowOffset_ = 0U;
        rows_ = totalRows_ < BATCH_ROWS ? totalRows_ : BATCH_ROWS;
        if (totalRows_ == 0U) {
            return;
        }
        if (scaleHeadStride == 1U) {
            // Q head scales are contiguous. The existing MTE2_V fence below completes Qscale and Vscale.
            AscendC::DataCopyExtParams qCopy{1, totalRows_ * uint32_t{sizeof(float)}, 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<float> qPadding{true, 0U, static_cast<uint8_t>((8U - totalRows_ % 8U) % 8U),
                                                          0.0f};
            AscendC::DataCopyPad(Ub<float>(BufferLayout::Q_SCALE), qScale[rowStart_], qCopy, qPadding);
        } else {
            // Scalar access reads exactly the valid tail, not the 1 KiB allocation padding.
            for (uint32_t row = 0; row < totalRows_; ++row) {
                Ub<float>(BufferLayout::Q_SCALE)
                    .SetValue(row, qScale.GetValue(static_cast<uint64_t>(rowStart_ + row) * scaleHeadStride));
            }
            for (uint32_t row = totalRows_; row < (totalRows_ + 7U) / 8U * 8U; ++row) {
                Ub<float>(BufferLayout::Q_SCALE).SetValue(row, 0.0f);
            }
        }
        pScale_ = pScale.GetValue(0);
        Ub<float>(BufferLayout::P_SCALE).SetValue(0, pScale_);
        if (headDim_ != alignedDim_) {
            AscendC::Duplicate(Ub<float>(BufferLayout::V_SCALE), 0.0f, alignedDim_);
        }
        if (!singleRowTile || headDim_ != alignedDim_) {
            // Wide-row normalization stays on V. Finish's pipe waits do not
            // order the next MTE2 Vscale write after the preceding O update.
            Sync<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::DataCopyExtParams copy{1, headDim_ * uint32_t{sizeof(float)}, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(Ub<float>(BufferLayout::V_SCALE), vScale, copy, padding);
        Sync<AscendC::HardEvent::MTE2_V>();
        // Vscale and Pscale are invariant throughout this task.
        AscendC::Muls(Ub<float>(BufferLayout::V_SCALE), Ub<float>(BufferLayout::V_SCALE), 1.0f / pScale_, alignedDim_);
        AscendC::Duplicate(Ub<float>(BufferLayout::O_ACC), 0.0f, ((totalRows_ + 7U) / 8U * 8U) * alignedDim_);
        AscendC::Duplicate(Ub<float>(BufferLayout::RUNNING_SUM), 0.0f, (totalRows_ + 7U) / 8U * 8U);
        AscendC::Duplicate(Ub<float>(BufferLayout::RUNNING_MAX), -3.402823466e38f, (totalRows_ + 7U) / 8U * 8U);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (!singleRowTile) {
            for (uint32_t slot = 0U; slot < FullQuantLayout::DOUBLE_BUFFER; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
            }
        }
    }

    __aicore__ inline void Finish()
    {
        if constexpr (!singleRowTile) {
            if (totalRows_ != 0U) {
                for (uint32_t slot = 0U; slot < FullQuantLayout::DOUBLE_BUFFER; ++slot) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
                }
            }
        }
    }

    __aicore__ inline UbCopyPlan GetCopyPlan()
    {
        static_assert(BufferLayout::K_SCALE_GATHER +
                          FullQuantLayout::PAGES_PER_TILE * BufferLayout::K_SCALE_CHUNK_BYTES <=
                      BufferLayout::P_FP16);
        static_assert(BufferLayout::P_FP16 + FullQuantLayout::KV_TILE * 2U * sizeof(uint64_t) <=
                      BufferLayout::P_OUTPUT);
        return {Ub<uint64_t>(BufferLayout::P_FP16)};
    }
    __aicore__ inline void BeginCopyPlan()
    {
        Sync<AscendC::HardEvent::V_S>();
    }
    __aicore__ inline void EndCopyPlan()
    {
        Sync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline bool HasRows() const
    {
        return totalRows_ != 0U;
    }

    struct EpilogueOnlineSoftmax;
    struct EpilogueRescaleO;

private:
    template <AscendC::HardEvent EVENT>
    __aicore__ inline void Sync()
    {
        AscendC::SetFlag<EVENT>(7);
        AscendC::WaitFlag<EVENT>(7);
    }

    __aicore__ inline void FullMask()
    {
        AscendC::SetVectorMask<int8_t>(~uint64_t{0}, ~uint64_t{0});
    }

    // This helper is only used for masks of 1..64 FP32 elements.
    __aicore__ inline void RowMask(uint32_t elements)
    {
        AscendC::SetVectorMask<int8_t>(0, elements == 64U ? ~uint64_t{0} : (uint64_t{1} << elements) - 1U);
    }

    template <class T>
    __aicore__ inline AscendC::LocalTensor<T> Ub(uint32_t offset)
    {
        return resource_.ubBuf.template GetBufferByByte<T>(offset);
    }

    Resource& resource_;
    uint32_t headDim_ = FullQuantLayout::HEAD_DIM;
    uint32_t alignedDim_ = FullQuantLayout::HEAD_DIM;
    uint32_t rows_ = 0;
    uint32_t columns_ = FullQuantLayout::KV_TILE;
    uint32_t rowStart_ = 0;
    uint32_t rowOffset_ = 0;
    uint32_t totalRows_ = 0;
    uint32_t cachedKBatch_ = UINT32_MAX;
    uint32_t cachedKHead_ = UINT32_MAX;
    uint32_t cachedKLength_ = 0;
    bool cacheMapsReady_ = false;
    bool qScaleReady_ = false;
    bool shortScaleOffsetsReady_ = false;
    float pScale_ = 1.0f;
};

} // namespace GbsaKernelArch22::Int8

#endif
