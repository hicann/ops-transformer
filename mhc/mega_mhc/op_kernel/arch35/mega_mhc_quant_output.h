/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_QUANT_OUTPUT_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_QUANT_OUTPUT_H

#include "compute/mega_mhc_mx_quant.h"
#include "mega_mhc_quant_layout.h"

namespace MegaMhc::Compute {
using namespace AscendC;

// Match standalone MX quantizers: temporarily select saturation and preserve the caller's NaN mode.
constexpr int8_t kQuantOverflowControlBit = 60;
constexpr int64_t kQuantOverflowMode = 0;
constexpr uint32_t kQuantTileElements = QuantOutput::kTileElements;

/** Quantize RMS1's rounded UB result before its source slot is released.
 * Complete rows can use separate FP8 slots; fallback tiles compact into their consumed B16 slot.
 * Two max/scale slots follow the caller's row-output credits; inverse scales
 * are Vector-only. No x2/partial storage or input-prefetch credit is borrowed.
 * Scratch capacity follows the planned D tile; a separate padded input/output
 * tile is allocated only when the plan can produce a short physical tail.
 */
template <typename T>
class MxFp8Output {
public:
    __aicore__ inline void Init(TPipe* pipe, bool enabled, uint32_t maxElements, uint32_t lastElements)
    {
        if (!enabled) {
            return;
        }
        static_assert(MxQuant::kVectorBytes == QuantOutput::kVectorBytes, "Quant layout must match the vector width");
        static_assert(MxQuant::kInputElementsPerStep == kQuantTileElements, "Quant layout must match the input step");
        layout_ = QuantOutput::MakeScratchLayout(maxElements, lastElements);
        pipe->InitBuffer(scratchBuffer_, layout_.bytes);
    }

    /** count is a multiple of 16; each non-final D tile begins on a 128-element boundary. */
    __aicore__ inline void Write(const LocalTensor<T>& normalized, uint32_t count, const GlobalTensor<uint8_t>& quantGm,
                                 uint64_t quantOffset, const GlobalTensor<uint8_t>& scaleGm, uint64_t scaleOffset,
                                 bool asyncOutput = false, uint32_t outputSlot = 0U)
    {
        // Host sizes scratch for the largest call. Full blocks and the short tail
        // have disjoint scale/output regions, so no internal DMA drain is needed.
        const uint32_t fullElements = count / kQuantTileElements * kQuantTileElements;
        if (fullElements != 0U) {
            WriteBatch(normalized, fullElements, quantGm, quantOffset, scaleGm, scaleOffset,
                       asyncOutput || fullElements != count, outputSlot);
        }
        if (fullElements == count) {
            return;
        }
        auto scratch = scratchBuffer_.template Get<uint8_t>();
        auto tail =
            scratch[layout_.tailOffset + (outputSlot & QuantOutput::kOutputSlotMask) * kQuantTileElements * sizeof(T)]
                .template ReinterpretCast<T>();
        Duplicate<T>(tail, static_cast<T>(0), kQuantTileElements);
        CopyTail(normalized[fullElements], tail, count - fullElements);
        WriteBatch(tail, kQuantTileElements, quantGm, quantOffset + fullElements, scaleGm,
                   scaleOffset + fullElements / QuantOutput::kGroupElements, asyncOutput, outputSlot,
                   count - fullElements, true);
    }

    /** Complete physical tiles with disjoint B16 input and FP8 output; caller owns the output credit. */
    __aicore__ inline void WriteSeparate(const LocalTensor<T>& normalized, const LocalTensor<uint8_t>& output,
                                         uint32_t count, const GlobalTensor<uint8_t>& quantGm, uint64_t quantOffset,
                                         const GlobalTensor<uint8_t>& scaleGm, uint64_t scaleOffset,
                                         uint32_t outputSlot)
    {
        auto scratch = scratchBuffer_.template Get<uint8_t>();
        auto maxScale = scratch[(outputSlot & QuantOutput::kOutputSlotMask) * layout_.slotBytes];
        auto maxima = maxScale.template ReinterpretCast<uint16_t>();
        auto inverse = scratch[layout_.inverseOffset].template ReinterpretCast<uint16_t>();
        Duplicate<uint16_t>(maxima, 0U, layout_.slotBytes / sizeof(uint16_t));
        Duplicate<uint16_t>(inverse, 0U, layout_.slotBytes / sizeof(uint16_t));
        auto* values = reinterpret_cast<__ubuf__ T*>(normalized.GetPhyAddr());
        auto* maxAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxima.GetPhyAddr());
        auto* inverseAddr = reinterpret_cast<__ubuf__ uint16_t*>(inverse.GetPhyAddr());
        auto* outputAddr = reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr());
        const uint32_t scales = count / QuantOutput::kGroupElements;
        const uint16_t loops = static_cast<uint16_t>(count / kQuantTileElements);
        const uint16_t scaleLoops =
            static_cast<uint16_t>((scales + MxQuant::kB16VectorElements - 1U) / MxQuant::kB16VectorElements);
        const int64_t overflow = GetCtrlSpr<kQuantOverflowControlBit, kQuantOverflowControlBit>();
        SetCtrlSpr<kQuantOverflowControlBit, kQuantOverflowControlBit>(kQuantOverflowMode);
        MxQuant::ComputeMaxExpOCP<T>(values, maxAddr, count, loops);
        MxQuant::ComputeScaleOCP<T, fp8_e4m3fn_t>(maxAddr, maxAddr, inverseAddr, scales, scaleLoops);
        MxQuant::ComputeData<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, T, fp8_e4m3fn_t>(values, inverseAddr,
                                                                                           outputAddr, count, loops);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams dataCopy{1U, count, 0U, 0U, 0U};
        DataCopyExtParams scaleCopy{1U, scales, 0U, 0U, 0U};
        DataCopyPad(quantGm[quantOffset], output, dataCopy);
        DataCopyPad(scaleGm[scaleOffset], maxScale, scaleCopy);
        SetCtrlSpr<kQuantOverflowControlBit, kQuantOverflowControlBit>(overflow);
    }

private:
    /** The caller waits for the matching row credit before entering this method.
     * Each ComputeData iteration reads 512B before storing 256B at a lower
     * address: forward compaction cannot overwrite any future B16 input.
     * ComputeScale similarly compacts max exponents into the same scale slot.
     * The other row/scale slot may still be read by MTE3 throughout this work.
     */
    __aicore__ inline void WriteBatch(const LocalTensor<T>& normalized, uint32_t count,
                                      const GlobalTensor<uint8_t>& quantGm, uint64_t quantOffset,
                                      const GlobalTensor<uint8_t>& scaleGm, uint64_t scaleOffset, bool asyncOutput,
                                      uint32_t outputSlot, uint32_t validElements = 0U, bool tailBatch = false)
    {
        auto scratch = scratchBuffer_.template Get<uint8_t>();
        const uint32_t scaleBytes = tailBatch ? QuantOutput::kVectorBytes : layout_.slotBytes;
        auto maxScale = scratch[(outputSlot & QuantOutput::kOutputSlotMask) * layout_.slotBytes +
                                (tailBatch ? layout_.slotBytes - QuantOutput::kVectorBytes : 0U)];
        auto maxima = maxScale.template ReinterpretCast<uint16_t>();
        auto inverse = scratch[layout_.inverseOffset].template ReinterpretCast<uint16_t>();
        auto output = normalized.template ReinterpretCast<uint8_t>();
        Duplicate<uint16_t>(maxima, 0U, scaleBytes / sizeof(uint16_t));
        Duplicate<uint16_t>(inverse, 0U, layout_.slotBytes / sizeof(uint16_t));
        auto* values = reinterpret_cast<__ubuf__ T*>(normalized.GetPhyAddr());
        auto* maxAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxima.GetPhyAddr());
        auto* inverseAddr = reinterpret_cast<__ubuf__ uint16_t*>(inverse.GetPhyAddr());
        auto* outputAddr = reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr());
        const uint32_t scales = count / QuantOutput::kGroupElements;
        const uint16_t loops = static_cast<uint16_t>(count / kQuantTileElements);
        const uint16_t scaleLoops =
            static_cast<uint16_t>((scales + MxQuant::kB16VectorElements - 1U) / MxQuant::kB16VectorElements);
        const int64_t overflow = GetCtrlSpr<kQuantOverflowControlBit, kQuantOverflowControlBit>();
        SetCtrlSpr<kQuantOverflowControlBit, kQuantOverflowControlBit>(kQuantOverflowMode);
        MxQuant::ComputeMaxExpOCP<T>(values, maxAddr, count, loops);
        MxQuant::ComputeScaleOCP<T, fp8_e4m3fn_t>(maxAddr, maxAddr, inverseAddr, scales, scaleLoops);
        MxQuant::ComputeData<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, T, fp8_e4m3fn_t>(values, inverseAddr,
                                                                                           outputAddr, count, loops);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint32_t outputCount = validElements == 0U ? count : validElements;
        const uint32_t outputScales =
            (outputCount + QuantOutput::kPairElements - 1U) / QuantOutput::kPairElements * QuantOutput::kScalesPerPair;
        DataCopyExtParams dataCopy{1U, outputCount, 0U, 0U, 0U};
        DataCopyExtParams scaleCopy{1U, outputScales, 0U, 0U, 0U};
        DataCopyPad(quantGm[quantOffset], output, dataCopy);
        DataCopyPad(scaleGm[scaleOffset], maxScale, scaleCopy);
        if (!asyncOutput) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        SetCtrlSpr<kQuantOverflowControlBit, kQuantOverflowControlBit>(overflow);
    }

    /** Copy only valid B16 elements, preserving their bits and the zero-filled padding. */
    __aicore__ inline void CopyTail(const LocalTensor<T>& source, const LocalTensor<T>& destination, uint32_t count)
    {
        auto* src = reinterpret_cast<__ubuf__ T*>(source.GetPhyAddr());
        auto* dst = reinterpret_cast<__ubuf__ T*>(destination.GetPhyAddr());
        const uint16_t repeats =
            static_cast<uint16_t>((count + MxQuant::kFp32VectorElements - 1U) / MxQuant::kFp32VectorElements);
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> raw;
            uint32_t remaining = count;
            for (uint16_t repeat = 0U; repeat < repeats; ++repeat) {
                auto mask = Reg::UpdateMask<float>(remaining);
                const uint32_t offset = repeat * MxQuant::kFp32VectorElements;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(raw, src + offset);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst + offset, raw, mask);
            }
        }
    }

    QuantOutput::ScratchLayout layout_;
    TBuf<TPosition::VECCALC> scratchBuffer_;
};
} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_QUANT_OUTPUT_H
