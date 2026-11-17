/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GBSA_BLOCK_EPILOGUE_KV_DEQUANT_ARCH35_A8C4_HPP
#define GBSA_BLOCK_EPILOGUE_KV_DEQUANT_ARCH35_A8C4_HPP
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
namespace NpuArch::Epilogue::Block {
class A8C4KvDequant {
public:
    static constexpr uint32_t HEAD_DIM = 128U;
    static constexpr uint32_t SCALE_BYTES = 16U;
    static constexpr uint32_t HALF_TILE_TOKENS = 256U;
    static constexpr uint32_t PACKED_BYTES = HEAD_DIM / 2U;
    static constexpr uint32_t SCALE_SPAN_BYTES = 20U * 1024U;
    static constexpr uint32_t MAX_STORAGE_HEADS = SCALE_SPAN_BYTES / (HALF_TILE_TOKENS * SCALE_BYTES);
    static constexpr uint32_t SLOT_BYTES = HALF_TILE_TOKENS * (PACKED_BYTES + HEAD_DIM) + SCALE_SPAN_BYTES;
    static __aicore__ inline uint32_t ScratchBytes()
    {
        return 2U * SLOT_BYTES;
    }

    __aicore__ inline A8C4KvDequant(Arch::Resource<Arch::AtlasA5>& resource, uint32_t ubOffset, uint32_t storageHeads,
                                    uint32_t eventBase = 0U)
        : eventBase_(eventBase)
    {
        AscendC::SetSaturationStrategy<AscendC::OverrideStrategy::USE_API>();
        AscendC::SetSaturationFlag<AscendC::SaturationMode::FLOAT8>(true);
        packedTokenStride_ = storageHeads * PACKED_BYTES;
        scaleTokenStride_ = storageHeads * SCALE_BYTES;
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            const uint32_t offset = ubOffset + slot * SLOT_BYTES;
            packedUbTensor_[slot] = resource.ubBuf.template GetBufferByByte<uint8_t>(offset);
            scaleUbTensor_[slot] =
                resource.ubBuf.template GetBufferByByte<uint8_t>(offset + HALF_TILE_TOKENS * PACKED_BYTES);
            outputUbTensor_[slot] = resource.ubBuf.template GetBufferByByte<fp8_e4m3fn_t>(
                offset + HALF_TILE_TOKENS * PACKED_BYTES + SCALE_SPAN_BYTES);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventBase_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventBase_ + 1U);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventBase_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventBase_ + 1U);
    }
    __aicore__ inline ~A8C4KvDequant()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventBase_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventBase_ + 1U);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventBase_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventBase_ + 1U);
    }
    template <uint32_t RECORD_WORDS, uint32_t DATA_WORD, uint32_t SCALE_WORD>
    __aicore__ inline void ProduceTile(AscendC::LocalTensor<fp8_e4m3fn_t> dst, AscendC::GlobalTensor<uint8_t> data,
                                       AscendC::GlobalTensor<uint8_t> scales, AscendC::LocalTensor<uint64_t> descriptor,
                                       uint32_t tokens, uint32_t l1Tokens)
    {
        const uint32_t split = ((tokens + 1U) / 2U + 15U) / 16U * 16U;
        const uint32_t begin = AscendC::GetSubBlockIdx() * split;
        if (begin >= tokens) {
            return;
        }
        const uint32_t outputSlot = nextOutputSlot_;
        const uint32_t count = min(split, tokens - begin);
        const uint32_t padded = (count + 15U) / 16U * 16U;
        auto* output = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(outputUbTensor_[outputSlot].GetPhyAddr());
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventBase_ + outputSlot);
        if (padded != count) {
            ClearPadding(output, static_cast<uint16_t>(count), static_cast<uint16_t>(padded));
        }
        // The kernel validates that the scale span for 256 rows fits in one slot.
        const uint32_t inputSlot = nextInputSlot_;
        LoadChunk<RECORD_WORDS, DATA_WORD, SCALE_WORD>(data, scales, descriptor, begin, count, inputSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventBase_ + inputSlot);
        auto* packed = reinterpret_cast<__ubuf__ uint8_t*>(packedUbTensor_[inputSlot].GetPhyAddr());
        auto* scale = reinterpret_cast<__ubuf__ bfloat16_t*>(scaleUbTensor_[inputSlot].GetPhyAddr());
        if ((count & 1U) != 0U) {
            ComputeChunk<true>(packed, scale, output, count, padded, scaleTokenStride_);
        } else {
            ComputeChunk<false>(packed, scale, output, count, padded, scaleTokenStride_);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventBase_ + inputSlot);
        nextInputSlot_ ^= 1U;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventBase_ + outputSlot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventBase_ + outputSlot);
        const AscendC::DataCopyParams copy{4U, static_cast<uint16_t>(padded), 0U,
                                           static_cast<uint16_t>(l1Tokens - padded)};
        AscendC::DataCopy(dst[begin * 32U], outputUbTensor_[outputSlot], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventBase_ + outputSlot);
        nextOutputSlot_ ^= 1U;
    }

private:
    template <uint32_t RECORD_WORDS, uint32_t DATA_WORD, uint32_t SCALE_WORD>
    __aicore__ inline void LoadChunk(AscendC::GlobalTensor<uint8_t> data, AscendC::GlobalTensor<uint8_t> scales,
                                     AscendC::LocalTensor<uint64_t> descriptor, uint32_t begin, uint32_t rows,
                                     uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventBase_ + slot);
        const AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0U, 0U, 0U};
        for (uint32_t row = 0U; row < rows;) {
            const uint32_t token = begin + row;
            const uint32_t record = token / 128U * RECORD_WORDS;
            const uint32_t offset = token % 128U;
            const uint32_t count = min(128U - offset, rows - row);
            const uint64_t packedOffset = descriptor.GetValue(record + DATA_WORD) + offset * packedTokenStride_;
            const uint64_t scaleOffset = descriptor.GetValue(record + SCALE_WORD) + offset * scaleTokenStride_;
            if (packedTokenStride_ == PACKED_BYTES) {
                AscendC::DataCopyPad(packedUbTensor_[slot][row * PACKED_BYTES], data[packedOffset],
                                     AscendC::DataCopyExtParams{1U, count * PACKED_BYTES, 0U, 0U, 0U}, pad);
            } else {
                AscendC::DataCopyPad(packedUbTensor_[slot][row * PACKED_BYTES], data[packedOffset],
                                     AscendC::DataCopyExtParams{static_cast<uint16_t>(count), PACKED_BYTES,
                                                                packedTokenStride_ - PACKED_BYTES, 0U, 0U},
                                     pad);
            }
            const uint32_t spanBytes = (count - 1U) * scaleTokenStride_ + SCALE_BYTES;
            AscendC::DataCopyPad(scaleUbTensor_[slot][row * scaleTokenStride_], scales[scaleOffset],
                                 AscendC::DataCopyExtParams{1U, spanBytes, 0U, 0U, 0U}, pad);
            row += count;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventBase_ + slot);
    }
    static __simd_vf__ inline void ClearPadding(__ubuf__ fp8_e4m3fn_t* output, uint16_t count, uint16_t padded)
    {
        using namespace AscendC::Reg;
        RegTensor<uint8_t> zero;
        Duplicate(zero, static_cast<uint8_t>(0));
        uint32_t bytes = 32U;
        MaskReg block = UpdateMask<uint8_t>(bytes);
        for (uint16_t plane = 0U; plane < 4U; ++plane) {
            for (uint16_t row = count; row < padded; ++row) {
                StoreAlign<uint8_t>(reinterpret_cast<__ubuf__ uint8_t*>(output) + (plane * padded + row) * 32U, zero,
                                    block);
            }
        }
    }
    template <bool ODD_ROWS>
    static __simd_vf__ inline void ComputeChunk(__ubuf__ uint8_t* packedUb, __ubuf__ bfloat16_t* scaleUb,
                                                __ubuf__ fp8_e4m3fn_t* outputUb, uint16_t rows, uint16_t outputRows,
                                                uint32_t scaleStride)
    {
        using namespace AscendC::Reg;
        static constexpr CastTrait castFp4 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::UNKNOWN};
        static constexpr CastTrait castFp8 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castFp8Odd = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                 AscendC::RoundMode::CAST_RINT};
        MaskReg floatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg byteFull = CreateMask<uint8_t, MaskPattern::ALL>();
        static constexpr CastTrait widenEven = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                AscendC::RoundMode::UNKNOWN};
        static constexpr CastTrait widenOdd = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::UNKNOWN};
        MaskReg full = CreateMask<bfloat16_t, MaskPattern::ALL>();
        RegTensor<fp4x2_e2m1_t> codes0;
        RegTensor<bfloat16_t> values0, scales0, product0;
        RegTensor<float> even0, odd0;
        RegTensor<fp8_e4m3fn_t> quantEven0, quantOdd0;
        RegTensor<uint8_t> pair0;
        RegTensor<fp4x2_e2m1_t> codes1;
        RegTensor<bfloat16_t> values1, scales1;
        uint32_t c16 = 16, c32 = 32, c48 = 48;
        MaskReg m0 = UpdateMask<uint32_t>(c16);
        MaskReg first32 = UpdateMask<uint32_t>(c32);
        MaskReg first48 = UpdateMask<uint32_t>(c48);
        MaskReg m1, m2, m3;
        MaskXor(m1, m0, first32, floatFull);
        MaskXor(m2, first32, first48, floatFull);
        MaskXor(m3, first48, floatFull, floatFull);
        auto* readPacked = packedUb;
        auto* readScale = scaleUb;
        auto* write0 = reinterpret_cast<__ubuf__ uint32_t*>(outputUb);
        auto* write1 = write0 + 8U * (outputRows - 1U);
        auto* write2 = write0 + 16U * (outputRows - 1U);
        auto* write3 = write0 + 24U * (outputRows - 1U);
        MaskReg fp4Full = CreateMask<fp4x2_e2m1_t, MaskPattern::ALL>();

        LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
            scales0, readScale, static_cast<int32_t>(scaleStride / sizeof(bfloat16_t)));
        LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
            reinterpret_cast<RegTensor<uint8_t>&>(codes0), readPacked, HEAD_DIM / 2U);
        Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(values0, codes0, fp4Full);
        uint16_t row = 0;
        for (; row + 2U < rows; row += 2U) {
            {
                LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                    scales1, readScale, static_cast<int32_t>(scaleStride / sizeof(bfloat16_t)));
                LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                    reinterpret_cast<RegTensor<uint8_t>&>(codes1), readPacked, HEAD_DIM / 2U);
                Mul(product0, values0, scales0, full);
                Cast<float, bfloat16_t, widenEven>(even0, product0, full);
                Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(values1, codes1, fp4Full);
                Cast<float, bfloat16_t, widenOdd>(odd0, product0, full);
                Cast<fp8_e4m3fn_t, float, castFp8>(quantEven0, even0, floatFull);
                Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantOdd0, odd0, floatFull);
                Or(pair0, reinterpret_cast<RegTensor<uint8_t>&>(quantEven0),
                   reinterpret_cast<RegTensor<uint8_t>&>(quantOdd0), byteFull);
                auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (row) * 8U;
                auto& src = reinterpret_cast<RegTensor<uint32_t>&>(pair0);
                StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(dst, src, m0);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write1, src, 8, m1);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write2, src, 8, m2);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write3, src, 8, m3);
            }
            {
                LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                    scales0, readScale, static_cast<int32_t>(scaleStride / sizeof(bfloat16_t)));
                LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                    reinterpret_cast<RegTensor<uint8_t>&>(codes0), readPacked, HEAD_DIM / 2U);
                Mul(product0, values1, scales1, full);
                Cast<float, bfloat16_t, widenEven>(even0, product0, full);
                Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(values0, codes0, fp4Full);
                Cast<float, bfloat16_t, widenOdd>(odd0, product0, full);
                Cast<fp8_e4m3fn_t, float, castFp8>(quantEven0, even0, floatFull);
                Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantOdd0, odd0, floatFull);
                Or(pair0, reinterpret_cast<RegTensor<uint8_t>&>(quantEven0),
                   reinterpret_cast<RegTensor<uint8_t>&>(quantOdd0), byteFull);
                auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (row + 1U) * 8U;
                auto& src = reinterpret_cast<RegTensor<uint32_t>&>(pair0);
                StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(dst, src, m0);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write1, src, 8, m1);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write2, src, 8, m2);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write3, src, 8, m3);
            }
        }
        if constexpr (!ODD_ROWS) {
            {
                LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                    scales1, readScale, static_cast<int32_t>(scaleStride / sizeof(bfloat16_t)));
                LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                    reinterpret_cast<RegTensor<uint8_t>&>(codes1), readPacked, HEAD_DIM / 2U);
                Mul(product0, values0, scales0, full);
                Cast<float, bfloat16_t, widenEven>(even0, product0, full);
                Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(values1, codes1, fp4Full);
                Cast<float, bfloat16_t, widenOdd>(odd0, product0, full);
                Cast<fp8_e4m3fn_t, float, castFp8>(quantEven0, even0, floatFull);
                Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantOdd0, odd0, floatFull);
                Or(pair0, reinterpret_cast<RegTensor<uint8_t>&>(quantEven0),
                   reinterpret_cast<RegTensor<uint8_t>&>(quantOdd0), byteFull);
                auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (rows - 2U) * 8U;
                auto& src = reinterpret_cast<RegTensor<uint32_t>&>(pair0);
                StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(dst, src, m0);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write1, src, 8, m1);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write2, src, 8, m2);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write3, src, 8, m3);
            }
            {
                Mul(product0, values1, scales1, full);
                Cast<float, bfloat16_t, widenEven>(even0, product0, full);
                Cast<float, bfloat16_t, widenOdd>(odd0, product0, full);
                Cast<fp8_e4m3fn_t, float, castFp8>(quantEven0, even0, floatFull);
                Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantOdd0, odd0, floatFull);
                Or(pair0, reinterpret_cast<RegTensor<uint8_t>&>(quantEven0),
                   reinterpret_cast<RegTensor<uint8_t>&>(quantOdd0), byteFull);
                auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (rows - 1U) * 8U;
                auto& src = reinterpret_cast<RegTensor<uint32_t>&>(pair0);
                StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(dst, src, m0);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write1, src, 8, m1);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write2, src, 8, m2);
                StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write3, src, 8, m3);
            }
        } else {
            Mul(product0, values0, scales0, full);
            Cast<float, bfloat16_t, widenEven>(even0, product0, full);
            Cast<float, bfloat16_t, widenOdd>(odd0, product0, full);
            Cast<fp8_e4m3fn_t, float, castFp8>(quantEven0, even0, floatFull);
            Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantOdd0, odd0, floatFull);
            Or(pair0, reinterpret_cast<RegTensor<uint8_t>&>(quantEven0),
               reinterpret_cast<RegTensor<uint8_t>&>(quantOdd0), byteFull);
            auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (rows - 1U) * 8U;
            auto& src = reinterpret_cast<RegTensor<uint32_t>&>(pair0);
            StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(dst, src, m0);
            StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write1, src, 8, m1);
            StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write2, src, 8, m2);
            StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write3, src, 8, m3);
        }
    }

    uint32_t eventBase_, packedTokenStride_, scaleTokenStride_;
    uint32_t nextInputSlot_ = 0U;
    uint32_t nextOutputSlot_ = 0U;
    AscendC::LocalTensor<uint8_t> packedUbTensor_[2], scaleUbTensor_[2];
    AscendC::LocalTensor<fp8_e4m3fn_t> outputUbTensor_[2];
};
} // namespace NpuArch::Epilogue::Block
#endif
