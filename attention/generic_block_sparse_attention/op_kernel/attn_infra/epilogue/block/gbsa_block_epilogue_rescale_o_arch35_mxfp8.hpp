/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_RESCALE_O_ARCH35_MXFP8_HPP
#define GBSA_BLOCK_EPILOGUE_RESCALE_O_ARCH35_MXFP8_HPP

#include "gbsa_block_epilogue.hpp"
#include "gbsa_block_epilogue_arch35_utils.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block::GbsaMxfp8BaseVF {
using namespace AscendC;
using namespace AscendC::Reg;
constexpr float EMPTY_LSE = -3.402823466e+38f;

__simd_vf__ inline void CopyVector(__ubuf__ float* dst, __ubuf__ float* src, uint32_t validRows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> value;
    MaskReg mask = UpdateMask<float>(validRows);
    LoadAlign(value, src);
    StoreAlign(dst, value, mask);
}

__aicore__ inline void CopyVectorCallVF(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rows)
{
    CopyVector(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
               rows);
}

template <uint16_t D, bool ClearStats = true>
__simd_vf__ inline void StaticInitTask(__ubuf__ float* output, __ubuf__ float* sum, __ubuf__ float* maximum,
                                       __ubuf__ float* groupMax, uint32_t rows)
{
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    // UpdateMask consumes its count argument; preserve the loop bound.
    uint32_t maskRows = rows;
    MaskReg rowMask = UpdateMask<float>(maskRows);
    RegTensor<float> zero, negative;
    Duplicate(zero, 0.0f);
    Duplicate(negative, (-__builtin_inff()));
    StoreAlign(sum, zero, rowMask);
    StoreAlign(maximum, negative, rowMask);
    if constexpr (ClearStats) {
        StoreAlign(groupMax, negative, rowMask);
        StoreAlign(groupMax + 64, negative, rowMask);
        StoreAlign(groupMax + 128, negative, rowMask);
        StoreAlign(groupMax + 192, negative, rowMask);
    }
    for (uint16_t row = 0; row < rows; ++row) {
        StoreAlign(output + row * D, zero, all);
        if constexpr (D >= 128) {
            StoreAlign(output + row * D + 64, zero, all);
        }
        if constexpr (D >= 192) {
            StoreAlign(output + row * D + 128, zero, all);
        }
        if constexpr (D == 256) {
            StoreAlign(output + row * D + 192, zero, all);
        }
    }
}

template <uint16_t D, bool ClearStats = true>
__aicore__ inline void StaticInitTaskCallVF(const LocalTensor<float>& output, const LocalTensor<float>& sum,
                                            const LocalTensor<float>& maximum, const LocalTensor<float>& groupMax,
                                            uint32_t rows)
{
    StaticInitTask<D, ClearStats>(reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(groupMax.GetPhyAddr()), rows);
}

template <bool ClearStats = true>
__aicore__ inline void InitTaskCallVF(const LocalTensor<float>& output, const LocalTensor<float>& sum,
                                      const LocalTensor<float>& maximum, const LocalTensor<float>& groupMax,
                                      uint32_t rows, uint16_t D)
{
    if (D == 64) {
        StaticInitTaskCallVF<64, ClearStats>(output, sum, maximum, groupMax, rows);
    } else if (D == 128) {
        StaticInitTaskCallVF<128, ClearStats>(output, sum, maximum, groupMax, rows);
    } else if (D == 192) {
        StaticInitTaskCallVF<192, ClearStats>(output, sum, maximum, groupMax, rows);
    } else if (D == 256) {
        StaticInitTaskCallVF<256, ClearStats>(output, sum, maximum, groupMax, rows);
    }
}
template <typename OutputT>
__simd_vf__ inline void FillEmptyOutput(__ubuf__ OutputT* output, __ubuf__ float* lse, uint32_t elements, uint32_t rows,
                                        bool outputLse, bool packedLse)
{
    RegTensor<OutputT> zero;
    RegTensor<float> empty;
    RegTensor<uint32_t> indices;
    Duplicate(zero, static_cast<OutputT>(0));
    const uint32_t repeats = (elements + 127U) / 128U;
    for (uint32_t i = 0; i < repeats; ++i) {
        MaskReg mask = UpdateMask<OutputT>(elements);
        StoreAlign(output + i * 128U, zero, mask);
    }
    if (outputLse) {
        MaskReg mask = UpdateMask<float>(rows);
        Duplicate(empty, EMPTY_LSE);
        if (packedLse) {
            Arange((RegTensor<int32_t>&)indices, 0);
            Muls(indices, indices, static_cast<uint32_t>(8), mask);
            Scatter(lse, empty, indices, mask);
        } else {
            StoreAlign(lse, empty, mask);
        }
    }
}

template <typename OutputT>
__aicore__ inline void FillEmptyOutputCallVF(const LocalTensor<OutputT>& output, const LocalTensor<float>& lse,
                                             uint32_t elements, uint32_t rows, bool outputLse, bool packedLse)
{
    FillEmptyOutput<OutputT>(reinterpret_cast<__ubuf__ OutputT*>(output.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ float*>(lse.GetPhyAddr()), elements, rows, outputLse, packedLse);
}

template <uint16_t D, uint16_t R>
__simd_vf__ inline void StaticRescaleOutput(__ubuf__ float* accumulator, __ubuf__ float* pv,
                                            __ubuf__ float* localRowsum, __ubuf__ float* outL, __ubuf__ float* tileDm,
                                            uint32_t validRows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    uint32_t rows = validRows;
    MaskReg rowMask = UpdateMask<float>(rows);
    RegTensor<float> dm, previous, current;
    RegTensor<float> previous0, previous1, previous2, previous3;
    RegTensor<float> current0, current1, current2, current3;
    LoadAlign(dm, tileDm);
    LoadAlign(previous, outL);
    LoadAlign(current, localRowsum);
    Mul(previous, previous, dm, rowMask);
    Add(current, current, previous, rowMask);
    StoreAlign(outL, current, rowMask);
    for (uint16_t row = 0; row < validRows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dm, tileDm + row);
        LoadAlign(previous0, accumulator + row * D + 0);
        if constexpr (D >= 128) {
            LoadAlign(previous1, accumulator + row * D + 64);
        }
        if constexpr (D >= 192) {
            LoadAlign(previous2, accumulator + row * D + 128);
        }
        if constexpr (D >= 256) {
            LoadAlign(previous3, accumulator + row * D + 192);
        }
        LoadAlign(current0, pv + row * D + 0);
        if constexpr (D >= 128) {
            LoadAlign(current1, pv + row * D + 64);
        }
        if constexpr (D >= 192) {
            LoadAlign(current2, pv + row * D + 128);
        }
        if constexpr (D >= 256) {
            LoadAlign(current3, pv + row * D + 192);
        }
        Mul(previous0, previous0, dm, all);
        if constexpr (D >= 128) {
            Mul(previous1, previous1, dm, all);
        }
        if constexpr (D >= 192) {
            Mul(previous2, previous2, dm, all);
        }
        if constexpr (D >= 256) {
            Mul(previous3, previous3, dm, all);
        }
        Add(current0, current0, previous0, all);
        if constexpr (D >= 128) {
            Add(current1, current1, previous1, all);
        }
        if constexpr (D >= 192) {
            Add(current2, current2, previous2, all);
        }
        if constexpr (D >= 256) {
            Add(current3, current3, previous3, all);
        }
        StoreAlign(accumulator + row * D + 0, current0, all);
        if constexpr (D >= 128) {
            StoreAlign(accumulator + row * D + 64, current1, all);
        }
        if constexpr (D >= 192) {
            StoreAlign(accumulator + row * D + 128, current2, all);
        }
        if constexpr (D >= 256) {
            StoreAlign(accumulator + row * D + 192, current3, all);
        }
    }
}

__simd_vf__ inline void PackLse(__ubuf__ float* dst, __ubuf__ float* src, uint32_t rows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = UpdateMask<float>(rows);
    RegTensor<float> value;
    RegTensor<uint32_t> indices;
    Arange((RegTensor<int32_t>&)indices, 0);
    Muls(indices, indices, static_cast<uint32_t>(8), mask);
    LoadAlign(value, src);
    Scatter(dst, value, indices, mask);
}

__aicore__ inline void PackLseCallVF(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rows)
{
    PackLse(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
            rows);
}

template <uint16_t D, uint16_t R>
__aicore__ inline void StaticRescaleCallVF(const LocalTensor<float>& output, const LocalTensor<float>& pv,
                                           const LocalTensor<float>& rowSum, const LocalTensor<float>& sum,
                                           const LocalTensor<float>& dm, uint32_t rows)
{
    StaticRescaleOutput<D, R>(
        reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(pv.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(rowSum.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(sum.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(dm.GetPhyAddr()), rows);
}

__aicore__ inline void RescaleCallVF(const LocalTensor<float>& output, const LocalTensor<float>& pv,
                                     const LocalTensor<float>& rowSum, const LocalTensor<float>& sum,
                                     const LocalTensor<float>& dm, uint32_t rows, uint16_t D, uint16_t R)
{
    if (D == 64) {
        StaticRescaleCallVF<64, 64>(output, pv, rowSum, sum, dm, rows);
    } else if (D == 128) {
        StaticRescaleCallVF<128, 64>(output, pv, rowSum, sum, dm, rows);
    } else if (D == 192) {
        StaticRescaleCallVF<192, 64>(output, pv, rowSum, sum, dm, rows);
    } else if (D == 256) {
        StaticRescaleCallVF<256, 64>(output, pv, rowSum, sum, dm, rows);
    }
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp8BaseVF

namespace NpuArch::Epilogue::Block::GbsaMxfp8FullVF {
using namespace AscendC;
using namespace AscendC::Reg;
template <class OutputT>
__simd_vf__ inline void Normalize(__ubuf__ OutputT* output, __ubuf__ float* numerator, __ubuf__ float* sum,
                                  __ubuf__ uint16_t* channelScale, uint16_t firstRow, uint16_t rows, uint16_t dim,
                                  float inversePScale)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<uint16_t> packed;
    RegTensor<uint32_t> bits, minimum;
    RegTensor<float> denominator, value, zero, unit;
    RegTensor<OutputT> converted;
    Duplicate(zero, 0.0f);
    Duplicate(unit, 1.0f);
    Duplicate(minimum, uint32_t(0x00400000U), all);
    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(denominator, sum + firstRow + row);
        MaskReg valid;
        Compares<float, CMPMODE::GT>(valid, denominator, 0.0f, all);
        Select(denominator, denominator, unit, valid);
        for (uint16_t column = 0; column < dim; column += 64U) {
            LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(packed, channelScale + column);
            ShiftRights(bits, (RegTensor<uint32_t>&)packed, int16_t(8), all);
            ShiftLefts(bits, bits, int16_t(23), all);
            MaskReg subnormal;
            Compares<uint32_t, CMPMODE::EQ>(subnormal, bits, uint32_t(0), all);
            Select(bits, minimum, bits, subnormal);
            LoadAlign(value, numerator + (firstRow + row) * dim + column);
            Muls(value, value, inversePScale, all);
            Div(value, value, denominator, all);
            Mul(value, value, (RegTensor<float>&)bits, all);
            Select(value, value, zero, valid);
            if constexpr (AscendC::IsSameType<OutputT, float>::value) {
                StoreAlign(output + row * dim + column, value, all);
            } else {
                Cast<OutputT, float, FP8_RINT>(converted, value, all);
                StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + row * dim + column, converted, all);
            }
        }
    }
}

__simd_vf__ inline void Lse(__ubuf__ float* output, __ubuf__ float* maximum, __ubuf__ float* sum, uint32_t rows)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg mask = UpdateMask<float>(rows), valid;
    RegTensor<float> m, l, one, empty;
    LoadAlign(m, maximum);
    LoadAlign(l, sum);
    Duplicate(one, 1.0f);
    Duplicate(empty, -3.402823466e+38f);
    Compares<float, CMPMODE::GT>(valid, l, 0.0f, mask);
    Select(l, l, one, valid);
    Ln(l, l, mask);
    Add(l, l, m, mask);
    Select(l, l, empty, valid);
    StoreAlign(output, l, mask);
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp8FullVF

namespace NpuArch::Epilogue::Block {

template <typename OutputT, uint32_t QueryCapacity = 32U>
class RescaleOMxfp8Full {
public:
    using ArchTag = Arch::AtlasA5;
    uint32_t D, R;
    using ElementO = OutputT;
    using ElementLse = float;
    struct Buffers {
        uint32_t LOCAL_ROW_SUM_OFFSET;
        uint32_t LSE_OFFSET;
        uint32_t LSE_PACKED_OFFSET;
        uint32_t OUTPUT_SCRATCH_BYTES;
        uint32_t OUTPUT_SCRATCH_OFFSET;
        uint32_t OUT_G_OFFSET;
        uint32_t OUT_L_OFFSET;
        uint32_t O_OFFSET;
        uint32_t PV_OFFSET;
        uint32_t PV_SLOTS;
        uint32_t PV_SLOT_BYTES;
        uint32_t TILE_DM_OFFSET;
        uint32_t TILE_G_OFFSET;
        uint32_t VSCALE_OFFSET;
    };
    Buffers layout;
    using Resource = Arch::Resource<Arch::AtlasA5>;
    __aicore__ inline explicit RescaleOMxfp8Full(Resource& resource, uint32_t dim, uint32_t vecRows,
                                                 const Buffers& buffers, float pScale)
        : D(dim),
          R(vecRows),
          layout(buffers),
          inversePScale_(1.0f / pScale)
    {
        channelScaleUB = resource.ubBuf.template GetBufferByByte<uint16_t>(layout.VSCALE_OFFSET);
        outputUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.O_OFFSET);
        for (uint32_t slot = 0; slot < layout.PV_SLOTS; ++slot) {
            pvUBTensor[slot] =
                resource.ubBuf.template GetBufferByByte<float>(layout.PV_OFFSET + slot * layout.PV_SLOT_BYTES);
        }
        lsePackedUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LSE_PACKED_OFFSET);
        rowsumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LOCAL_ROW_SUM_OFFSET);
        sumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.OUT_L_OFFSET);
        maxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.OUT_G_OFFSET);
        lseUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LSE_OFFSET);
        resultUBTensor = resource.ubBuf.template GetBufferByByte<OutputT>(layout.OUTPUT_SCRATCH_OFFSET);
        for (uint32_t slot = 0; slot < 4; ++slot) {
            tileMaxUBTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_G_OFFSET + slot * 256);
            dmUBTensor[slot] = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_DM_OFFSET + slot * 256);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
    }
    __aicore__ inline ~RescaleOMxfp8Full()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
    }
    __aicore__ inline void SetValueScale(const AscendC::GlobalTensor<uint8_t>& scale)
    {
        const auto address = reinterpret_cast<uint64_t>(scale.GetPhyAddr());
        if (address == scaleAddress_) {
            return;
        }
        // Raw uses V_MTE2/MTE2_V 0..5; event 6 is private to channel VScale.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
        AscendC::DataCopy(channelScaleUB.ReinterpretCast<uint8_t>(), scale, D * 2U);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
        scaleAddress_ = address;
    }
    __aicore__ inline void BeginTask(uint32_t rows, bool flashDecode)
    {
        if (flashDecode) {
            // A previous partial DMA reads N/G/L until this credit arrives.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        }
        // The frontend may already own the statistics of the following task.
        GbsaMxfp8BaseVF::InitTaskCallVF<false>(outputUBTensor, sumUBTensor, maxUBTensor, tileMaxUBTensor[0], rows, D);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ConsumeTile(uint32_t statsSlot, uint32_t pvSlot, uint32_t rows, bool saveMax)
    {
        GbsaMxfp8BaseVF::RescaleCallVF(outputUBTensor, pvUBTensor[pvSlot], rowsumUBTensor[statsSlot * 64], sumUBTensor,
                                       dmUBTensor[statsSlot], rows, D, R);
        AscendC::PipeBarrier<PIPE_V>();
        // Only final LSE or FD partial output consumes this G snapshot.
        if (saveMax) {
            GbsaMxfp8BaseVF::CopyVectorCallVF(maxUBTensor, tileMaxUBTensor[statsSlot], rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void StorePartial(const AscendC::GlobalTensor<float>& numerator,
                                        const AscendC::GlobalTensor<float>& stats, uint32_t validRows,
                                        uint32_t subStride)
    {
        if (validRows == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        auto result = resultUBTensor.template ReinterpretCast<float>();
        AscendC::DataCopyExtParams copy = {1, validRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        if constexpr (QueryCapacity <= 32U) {
            Normalize<float>(result, outputUBTensor, sumUBTensor, 0U, validRows);
            ComputeLse(lseUBTensor, maxUBTensor, sumUBTensor, validRows);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
            copy.blockLen = validRows * D * sizeof(float);
            AscendC::DataCopyPad(numerator, result, copy);
        } else {
            // FP32 partials may exceed the P scratch region. Normalize rows
            // into the same bounded scratch only after its MTE3 reader retires.
            ComputeLse(lseUBTensor, maxUBTensor, sumUBTensor, validRows);
            const uint32_t chunkRows = layout.OUTPUT_SCRATCH_BYTES / (D * sizeof(float));
            for (uint32_t first = 0U; first < validRows; first += chunkRows) {
                const uint32_t rows = min(validRows - first, chunkRows);
                if (first != 0U) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
                }
                Normalize<float>(result, outputUBTensor, sumUBTensor, first, rows);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
                copy.blockLen = rows * D * sizeof(float);
                AscendC::DataCopyPad(numerator[first * D], result, copy);
                if (first + rows < validRows) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
                }
            }
        }
        copy.blockLen = validRows * sizeof(float);
        AscendC::DataCopyPad(stats, lseUBTensor, copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

    __aicore__ inline void Store(const AscendC::GlobalTensor<OutputT>& output, const AscendC::GlobalTensor<float>& lse,
                                 uint32_t validRows, bool outputLse, uint32_t headStride, uint32_t lseHeadStride)
    {
        StoreOutput(output, lse, validRows, outputLse, headStride, lseHeadStride, outputUBTensor, sumUBTensor,
                    maxUBTensor);
    }

    __aicore__ inline void StoreTile(const AscendC::GlobalTensor<OutputT>& output,
                                     const AscendC::GlobalTensor<float>& lse, uint32_t validRows, bool outputLse,
                                     uint32_t headStride, uint32_t lseHeadStride, uint32_t statsSlot, uint32_t pvSlot)
    {
        // A complete single-tile task has no earlier numerator to rescale.
        StoreOutput(output, lse, validRows, outputLse, headStride, lseHeadStride, pvUBTensor[pvSlot],
                    rowsumUBTensor[statsSlot * 64U], tileMaxUBTensor[statsSlot]);
    }

    __aicore__ inline void StoreEmpty(const AscendC::GlobalTensor<OutputT>& output,
                                      const AscendC::GlobalTensor<float>& lse, uint32_t validRows, bool outputLse,
                                      uint32_t headStride, uint32_t lseHeadStride)
    {
        if (validRows == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        const uint32_t chunkRows = layout.OUTPUT_SCRATCH_BYTES / (D * static_cast<uint32_t>(sizeof(OutputT)));
        auto lseSource = lseHeadStride == 1U ? lseUBTensor : lsePackedUBTensor;
        GbsaMxfp8BaseVF::FillEmptyOutputCallVF(resultUBTensor, lseSource, min(validRows, chunkRows) * D, validRows,
                                               outputLse, lseHeadStride != 1U);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        for (uint32_t first = 0U; first < validRows; first += chunkRows) {
            const uint32_t rows = min(validRows - first, chunkRows);
            AscendC::DataCopyExtParams copy = {static_cast<uint16_t>(rows), D * static_cast<uint32_t>(sizeof(OutputT)),
                                               0, static_cast<int64_t>((headStride - 1U) * D * sizeof(OutputT)), 0};
            AscendC::DataCopyPad(output[first * headStride * D], resultUBTensor, copy);
        }
        if (outputLse) {
            AscendC::DataCopyExtParams copy = {
                static_cast<uint16_t>(lseHeadStride == 1U ? 1U : validRows),
                static_cast<uint32_t>((lseHeadStride == 1U ? validRows : 1U) * sizeof(float)), 0,
                static_cast<int64_t>((lseHeadStride - 1U) * sizeof(float)), 0};
            AscendC::DataCopyPad(lse, lseSource, copy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

private:
    __aicore__ inline void StoreOutput(const AscendC::GlobalTensor<OutputT>& output,
                                       const AscendC::GlobalTensor<float>& lse, uint32_t validRows, bool outputLse,
                                       uint32_t headStride, uint32_t lseHeadStride,
                                       const AscendC::LocalTensor<float>& numerator,
                                       const AscendC::LocalTensor<float>& denominator,
                                       const AscendC::LocalTensor<float>& maximum)
    {
        if (validRows == 0) {
            return;
        }
        // Output conversion shares P storage with the frontend. Preserve its
        // credits while waiting for the current P stores to complete.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        const uint32_t CHUNK_ROWS = layout.OUTPUT_SCRATCH_BYTES / (D * static_cast<uint32_t>(sizeof(OutputT)));
        auto scratch = resultUBTensor;
        for (uint32_t first = 0; first < validRows; first += CHUNK_ROWS) {
            const uint32_t rows = validRows - first < CHUNK_ROWS ? validRows - first : CHUNK_ROWS;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
            Normalize<OutputT>(scratch, numerator, denominator, first, rows);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
            AscendC::DataCopyExtParams copy = {
                static_cast<uint16_t>(rows), D * static_cast<uint32_t>(sizeof(OutputT)), 0,
                static_cast<int64_t>((headStride - 1) * D * static_cast<uint32_t>(sizeof(OutputT))), 0};
            AscendC::DataCopyPad(output[first * headStride * D], scratch, copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
        if (outputLse) {
            ComputeLse(lseUBTensor, maximum, denominator, validRows);
            if (lseHeadStride == 1U) {
                // ComputeLse already stores contiguous FP32 values. Keep one
                // transfer for a contiguous GM view, including short row tails.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::DataCopyExtParams copy = {1, validRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(lse, lseUBTensor, copy);
            } else {
                GbsaMxfp8BaseVF::PackLseCallVF(lsePackedUBTensor, lseUBTensor, validRows);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
                AscendC::DataCopyExtParams copy = {static_cast<uint16_t>(validRows), sizeof(float), 0,
                                                   static_cast<int64_t>((lseHeadStride - 1U) * sizeof(float)), 0};
                AscendC::DataCopyPad(lse, lsePackedUBTensor, copy);
            }
        }
        // Only Vector reuses the output and LSE sources. Keep its credits;
        // independent Scalar descriptor work may proceed during the writeback.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    }

private:
    template <class T>
    __aicore__ void Normalize(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<float>& n,
                              const AscendC::LocalTensor<float>& l, uint32_t first, uint32_t rows)
    {
        GbsaMxfp8FullVF::Normalize<T>(
            reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(n.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(l.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint16_t*>(channelScaleUB.GetPhyAddr()), first, rows, D, inversePScale_);
    }
    __aicore__ void ComputeLse(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& m,
                               const AscendC::LocalTensor<float>& l, uint32_t rows)
    {
        GbsaMxfp8FullVF::Lse(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ float*>(m.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ float*>(l.GetPhyAddr()), rows);
    }
    float inversePScale_;
    uint64_t scaleAddress_ = 0;
    AscendC::LocalTensor<uint16_t> channelScaleUB;
    AscendC::LocalTensor<float> outputUBTensor, pvUBTensor[2], rowsumUBTensor;
    AscendC::LocalTensor<float> lsePackedUBTensor, sumUBTensor, maxUBTensor, lseUBTensor;
    AscendC::LocalTensor<float> tileMaxUBTensor[4], dmUBTensor[4];
    AscendC::LocalTensor<OutputT> resultUBTensor;
};

} // namespace NpuArch::Epilogue::Block
#endif
