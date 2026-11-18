/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_MXFP8_HPP
#define GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_MXFP8_HPP

#include "gbsa_block_epilogue.hpp"
#include "gbsa_block_epilogue_arch35_utils.hpp"
#include "../gbsa_epilogue_dispatch_policy.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"

namespace NpuArch::Epilogue::Block::GbsaMxfp8BaseVF {
using namespace AscendC;
using namespace AscendC::Reg;
constexpr float NEG_INF = -__builtin_inff();

__simd_vf__ inline void InitInitialMax(__ubuf__ float* maximum, uint32_t rows)
{
    RegTensor<float> negative;
    MaskReg mask = UpdateMask<float>(rows);
    Duplicate(negative, NEG_INF);
    StoreAlign(maximum, negative, mask);
}

__aicore__ inline void InitInitialMaxCallVF(const LocalTensor<float>& maximum, uint32_t rows)
{
    InitInitialMax(reinterpret_cast<__ubuf__ float*>(maximum.GetPhyAddr()), rows);
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp8BaseVF

namespace NpuArch::Epilogue::Block::GbsaMxfp8FullVF {
using namespace AscendC;
using namespace AscendC::Reg;

// Statistics are FP32 and precede quantization. Each tile has its own max,
// local sum and rescale snapshot so the PV consumer may lag by two tiles.
template <bool FullTile, uint32_t R = 16U, uint32_t N = 512U>
__simd_vf__ inline void Softmax(__ubuf__ fp8_e4m3fn_t* p, __ubuf__ float* s, __ubuf__ float* maximum,
                                __ubuf__ float* sum, __ubuf__ float* dm, __ubuf__ float* previous, uint16_t rows,
                                uint16_t tokens, float attentionScale, float pScale, bool initializePadding)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero, negative, rowMax, localMax, oldMax, delta, sumReg, localSum;
    RegTensor<float> score0, score1, exp0, exp1;
    RegTensor<fp8_e4m3fn_t> quant0, quant1, packed, tmp;
    RegTensor<uint8_t> padding;
    Duplicate(zero, 0.0f);
    Duplicate(negative, -__builtin_inff());
    const uint16_t planes = FullTile ? N / 32U : (tokens + 63U) / 64U * 2U;
    if (initializePadding) {
        Duplicate(padding, static_cast<uint8_t>(0));
        for (uint16_t plane = 0; plane < planes; ++plane) {
            uint32_t remaining = (R - rows) * 32U;
            for (uint16_t offset = 0; offset < (R - rows) * 32U; offset += 256U) {
                MaskReg mask = UpdateMask<uint8_t>(remaining);
                StoreAlign(reinterpret_cast<__ubuf__ uint8_t*>(p) + plane * R * 32U + rows * 32U + offset, padding,
                           mask);
            }
        }
    }
    UnalignReg maxStore, sumStore, dmStore;
    const uint16_t loops = FullTile ? N / 64U : (tokens + 63U) / 64U;
    const uint16_t pairs = FullTile ? N / 128U : (tokens + 127U) / 128U;
    for (uint16_t row = 0; row < rows; ++row) {
        Duplicate(localMax, -__builtin_inff());
        for (uint16_t block = 0; block < loops; ++block) {
            uint32_t valid = FullTile ? 64U : min(64U, static_cast<uint32_t>(tokens - block * 64U));
            MaskReg mask = UpdateMask<float>(valid);
            LoadAlign(score0, s + row * N + block * 64U);
            Muls(score0, score0, attentionScale, all);
            Select(score0, score0, negative, mask);
            Max(localMax, localMax, score0, all);
        }
        ReduceMax(rowMax, localMax, all);
        Duplicate(rowMax, rowMax, all);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(oldMax, previous + row);
        Max(rowMax, rowMax, oldMax, all);
        FusedExpSub(delta, oldMax, rowMax, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maximum, rowMax, maxStore, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dm, delta, dmStore, 1);
        Duplicate(sumReg, 0.0f);
        for (uint16_t block = 0; block < pairs; ++block) {
            const uint32_t begin = block * 128U;
            const uint32_t left = tokens - begin;
            uint32_t n0 = min(64U, left), n1 = left > 64U ? min(64U, left - 64U) : 0U;
            uint32_t stored = min(128U, planes * 32U - begin);
            MaskReg mask0 = UpdateMask<float>(n0), mask1 = UpdateMask<float>(n1);
            MaskReg storeMask = UpdateMask<uint8_t>(stored);
            LoadAlign(score0, s + row * N + begin);
            Muls(score0, score0, attentionScale, all);
            FusedExpSub(exp0, score0, rowMax, mask0);
            Select(exp0, exp0, zero, mask0);
            Duplicate(exp1, 0.0f);
            if (left > 64U) {
                LoadAlign(score1, s + row * N + begin + 64U);
                Muls(score1, score1, attentionScale, all);
                FusedExpSub(exp1, score1, rowMax, mask1);
                Select(exp1, exp1, zero, mask1);
            }
            Add(sumReg, sumReg, exp0, all);
            Add(sumReg, sumReg, exp1, all);
            Muls(exp0, exp0, pScale, all);
            Muls(exp1, exp1, pScale, all);
            Cast<fp8_e4m3fn_t, float, FP8_RINT>(quant0, exp0, all);
            Cast<fp8_e4m3fn_t, float, FP8_RINT>(quant1, exp1, all);
            DeInterleave((RegTensor<uint16_t>&)packed, (RegTensor<uint16_t>&)tmp, (RegTensor<uint16_t>&)quant0,
                         (RegTensor<uint16_t>&)quant1);
            DeInterleave(quant0, quant1, packed, packed);
            StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY>(p + row * 32U + begin * R, quant0, R, storeMask);
        }
        ReduceSum(localSum, sumReg, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum, localSum, sumStore, 1);
    }
    vstas(maxStore, maximum, 0, POST_UPDATE);
    vstas(dmStore, dm, 0, POST_UPDATE);
    vstas(sumStore, sum, 0, POST_UPDATE);
}

// A full tile fits in eight FP32 registers per row. Keep scores live
// through max/exp/quantization instead of reading S twice. The sum order,
// fixed P multiplier and FP8 rounding remain identical to the tail path.
__simd_vf__ inline void SoftmaxFull(__ubuf__ fp8_e4m3fn_t* p, __ubuf__ float* s, __ubuf__ float* maximum,
                                    __ubuf__ float* sum, __ubuf__ float* dm, __ubuf__ float* previous, uint16_t rows,
                                    float attentionScale, float pScale, bool initializePadding)
{
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    constexpr uint32_t R = 16, N = 512;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    uint32_t storeColumns = 128;
    MaskReg storeMask = UpdateMask<uint8_t>(storeColumns);
    RegTensor<float> s0, s1, s2, s3, s4, s5, s6, s7;
    RegTensor<float> localMax, rowMax, oldMax, delta, sumReg, localSum;
    RegTensor<fp8_e4m3fn_t> q0, q1, packed, tmp;
    RegTensor<uint8_t> padding;
    if (initializePadding) {
        Duplicate(padding, static_cast<uint8_t>(0));
        for (uint16_t plane = 0; plane < 16; ++plane) {
            uint32_t remaining = (R - rows) * 32U;
            for (uint16_t offset = 0; offset < (R - rows) * 32U; offset += 256U) {
                MaskReg mask = UpdateMask<uint8_t>(remaining);
                StoreAlign(reinterpret_cast<__ubuf__ uint8_t*>(p) + plane * R * 32U + rows * 32U + offset, padding,
                           mask);
            }
        }
    }
    UnalignReg maxStore, sumStore, dmStore;
    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign(s0, s + row * N + 0U);
        LoadAlign(s1, s + row * N + 64U);
        LoadAlign(s2, s + row * N + 128U);
        LoadAlign(s3, s + row * N + 192U);
        LoadAlign(s4, s + row * N + 256U);
        LoadAlign(s5, s + row * N + 320U);
        LoadAlign(s6, s + row * N + 384U);
        LoadAlign(s7, s + row * N + 448U);
        Muls(s0, s0, attentionScale, all);
        Muls(s1, s1, attentionScale, all);
        Muls(s2, s2, attentionScale, all);
        Muls(s3, s3, attentionScale, all);
        Muls(s4, s4, attentionScale, all);
        Muls(s5, s5, attentionScale, all);
        Muls(s6, s6, attentionScale, all);
        Muls(s7, s7, attentionScale, all);
        Max(localMax, s0, s1, all);
        Max(localMax, localMax, s2, all);
        Max(localMax, localMax, s3, all);
        Max(localMax, localMax, s4, all);
        Max(localMax, localMax, s5, all);
        Max(localMax, localMax, s6, all);
        Max(localMax, localMax, s7, all);
        ReduceMax(rowMax, localMax, all);
        Duplicate(rowMax, rowMax, all);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(oldMax, previous + row);
        Max(rowMax, rowMax, oldMax, all);
        FusedExpSub(delta, oldMax, rowMax, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maximum, rowMax, maxStore, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dm, delta, dmStore, 1);
        FusedExpSub(s0, s0, rowMax, all);
        FusedExpSub(s1, s1, rowMax, all);
        FusedExpSub(s2, s2, rowMax, all);
        FusedExpSub(s3, s3, rowMax, all);
        FusedExpSub(s4, s4, rowMax, all);
        FusedExpSub(s5, s5, rowMax, all);
        FusedExpSub(s6, s6, rowMax, all);
        FusedExpSub(s7, s7, rowMax, all);
        Duplicate(sumReg, 0.0f);
        Add(sumReg, sumReg, s0, all);
        Add(sumReg, sumReg, s1, all);
        Add(sumReg, sumReg, s2, all);
        Add(sumReg, sumReg, s3, all);
        Add(sumReg, sumReg, s4, all);
        Add(sumReg, sumReg, s5, all);
        Add(sumReg, sumReg, s6, all);
        Add(sumReg, sumReg, s7, all);
        ReduceSum(localSum, sumReg, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sum, localSum, sumStore, 1);
        Muls(s0, s0, pScale, all);
        Muls(s1, s1, pScale, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q0, s0, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q1, s1, all);
        DeInterleave((RegTensor<uint16_t>&)packed, (RegTensor<uint16_t>&)tmp, (RegTensor<uint16_t>&)q0,
                     (RegTensor<uint16_t>&)q1);
        DeInterleave(q0, q1, packed, packed);
        StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY>(p + row * 32U + 0U * R, q0, R, storeMask);
        Muls(s2, s2, pScale, all);
        Muls(s3, s3, pScale, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q0, s2, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q1, s3, all);
        DeInterleave((RegTensor<uint16_t>&)packed, (RegTensor<uint16_t>&)tmp, (RegTensor<uint16_t>&)q0,
                     (RegTensor<uint16_t>&)q1);
        DeInterleave(q0, q1, packed, packed);
        StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY>(p + row * 32U + 128U * R, q0, R, storeMask);
        Muls(s4, s4, pScale, all);
        Muls(s5, s5, pScale, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q0, s4, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q1, s5, all);
        DeInterleave((RegTensor<uint16_t>&)packed, (RegTensor<uint16_t>&)tmp, (RegTensor<uint16_t>&)q0,
                     (RegTensor<uint16_t>&)q1);
        DeInterleave(q0, q1, packed, packed);
        StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY>(p + row * 32U + 256U * R, q0, R, storeMask);
        Muls(s6, s6, pScale, all);
        Muls(s7, s7, pScale, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q0, s6, all);
        Cast<fp8_e4m3fn_t, float, FP8_RINT>(q1, s7, all);
        DeInterleave((RegTensor<uint16_t>&)packed, (RegTensor<uint16_t>&)tmp, (RegTensor<uint16_t>&)q0,
                     (RegTensor<uint16_t>&)q1);
        DeInterleave(q0, q1, packed, packed);
        StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY>(p + row * 32U + 384U * R, q0, R, storeMask);
    }
    vstas(maxStore, maximum, 0, POST_UPDATE);
    vstas(dmStore, dm, 0, POST_UPDATE);
    vstas(sumStore, sum, 0, POST_UPDATE);
}

} // namespace NpuArch::Epilogue::Block::GbsaMxfp8FullVF

namespace NpuArch::Epilogue::Block {

template <uint32_t RowCapacity = 16U, uint32_t TileTokens = 512U>
class SoftmaxMxfp8Full {
public:
    using ArchTag = Arch::AtlasA5;
    uint32_t D, R;
    struct Buffers {
        uint32_t KV;
        uint32_t LOCAL_ROW_SUM_OFFSET;
        uint32_t P_OFFSET;
        uint32_t P_SLOT_BYTES;
        uint32_t S_OFFSET;
        uint32_t S_SLOT_BYTES;
        uint32_t TILE_DM_OFFSET;
        uint32_t TILE_G_OFFSET;
        uint32_t TILE_MAX_OFFSET;
    };
    Buffers layout;
    using Resource = Arch::Resource<Arch::AtlasA5>;

    __aicore__ inline explicit SoftmaxMxfp8Full(Resource& resource, uint32_t dim, uint32_t vecRows,
                                                const Buffers& buffers, float pScale)
        : D(dim),
          R(vecRows),
          layout(buffers),
          pScale_(pScale)
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            sUBTensor[slot] =
                resource.ubBuf.template GetBufferByByte<float>(layout.S_OFFSET + slot * layout.S_SLOT_BYTES);
            pUBTensor[slot] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(layout.P_OFFSET + slot * layout.P_SLOT_BYTES);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
        localMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_DM_OFFSET);
        groupSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.LOCAL_ROW_SUM_OFFSET);
        tileMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_G_OFFSET);
        initialMaxUBTensor = resource.ubBuf.template GetBufferByByte<float>(layout.TILE_MAX_OFFSET);
    }
    __aicore__ inline ~SoftmaxMxfp8Full()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
    // Output conversion reuses P and invalidates both slots' padding.
    __aicore__ inline void InvalidatePadding()
    {
        paddingPlanes_[0] = 0;
        paddingPlanes_[1] = 0;
    }
    __aicore__ inline void InitTask(uint32_t validRows, float attentionScale)
    {
        InvalidatePadding();
        validRows_ = validRows;
        attentionScale_ = attentionScale;
        // The first tile reads this immutable seed directly.
        // RescaleO owns the numerator and denominator state for each task.
        GbsaMxfp8BaseVF::InitInitialMaxCallVF(initialMaxUBTensor, validRows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void operator()(uint32_t sSlot, uint32_t pSlot, uint32_t statsSlot, uint32_t validKv,
                                      const AscendC::LocalTensor<uint8_t>& l1P, uint32_t rowOffset, uint32_t queryRows,
                                      bool firstTile)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pSlot);
        if (validRows_ != 0) {
            const uint32_t bankRows = (validRows_ + 15U) / 16U * 16U;
            const uint32_t planes = (validKv + 63U) / 64U;
            const bool initializePadding = paddingPlanes_[pSlot] < planes && validRows_ != bankRows;
            paddingPlanes_[pSlot] = max(paddingPlanes_[pSlot], planes);
            auto previous = firstTile ? initialMaxUBTensor : tileMaxUBTensor[previousStats_ * 64U];
            if (validKv == layout.KV) {
                RunSoftmax<true>(sSlot, pSlot, statsSlot, validKv, previous, initializePadding);
            } else {
                RunSoftmax<false>(sSlot, pSlot, statsSlot, validKv, previous, initializePadding);
            }
            previousStats_ = statsSlot;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pSlot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pSlot);
            const uint32_t rows0 = (queryRows + 1) / 2;
            const uint32_t alignedRows = 2U * ((rows0 + 15U) / 16U * 16U);
            auto pLayout = tla::MakeLayout<uint8_t, layout::zN>(alignedRows, layout.KV);
            auto pL1Tensor = tla::MakeTensor(l1P, pLayout, Arch::PositionL1{});
            auto pTile = tla::GetTile(pL1Tensor, tla::MakeCoord(rowOffset, 0), tla::MakeShape(bankRows, layout.KV));
            AscendC::DataCopyParams copy = {static_cast<uint16_t>((validKv + 63) / 64 * 2),
                                            static_cast<uint16_t>(bankRows), static_cast<uint16_t>(R - bankRows),
                                            static_cast<uint16_t>(alignedRows - bankRows)};
            AscendC::DataCopy(pTile.data()[pLayout(pTile.coord())], pUBTensor[pSlot], copy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pSlot);
    }

private:
    template <bool Full>
    __aicore__ void RunSoftmax(uint32_t sSlot, uint32_t pSlot, uint32_t statsSlot, uint32_t tokens,
                               const AscendC::LocalTensor<float>& previous, bool padding)
    {
        if constexpr (Full && TileTokens == 512U) {
            GbsaMxfp8FullVF::SoftmaxFull(
                reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(pUBTensor[pSlot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(sUBTensor[sSlot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(tileMaxUBTensor[statsSlot * 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(groupSumUBTensor[statsSlot * 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(localMaxUBTensor[statsSlot * 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(previous.GetPhyAddr()), validRows_, attentionScale_, pScale_,
                padding);
        } else {
            GbsaMxfp8FullVF::Softmax<Full, RowCapacity, TileTokens>(
                reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(pUBTensor[pSlot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(sUBTensor[sSlot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(tileMaxUBTensor[statsSlot * 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(groupSumUBTensor[statsSlot * 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(localMaxUBTensor[statsSlot * 64U].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(previous.GetPhyAddr()), validRows_, tokens, attentionScale_, pScale_,
                padding);
        }
    }
    uint32_t previousStats_ = 0;
    AscendC::LocalTensor<float> sUBTensor[2], localMaxUBTensor, groupSumUBTensor;
    AscendC::LocalTensor<float> tileMaxUBTensor, initialMaxUBTensor;
    AscendC::LocalTensor<uint8_t> pUBTensor[2];
    uint32_t paddingPlanes_[2] = {0, 0};
    uint32_t validRows_ = 0;
    float attentionScale_ = 1.0f;
    float pScale_;
};

} // namespace NpuArch::Epilogue::Block
#endif
