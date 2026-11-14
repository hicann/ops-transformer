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
    static constexpr uint32_t SCALE_BYTES = (HEAD_DIM / 16U) * sizeof(bfloat16_t);
    static constexpr uint32_t HALF_TILE_TOKENS = 256U;
    static constexpr uint32_t STRIDED_SCALE_ROW_BYTES = 32U;

    static __aicore__ inline uint32_t HalfTileScratchBytes(uint32_t storageHeads)
    {
        const uint32_t scaleRowBytes = storageHeads == 1U ? SCALE_BYTES : STRIDED_SCALE_ROW_BYTES;
        return HALF_TILE_TOKENS * (HEAD_DIM / 2U + scaleRowBytes + HEAD_DIM);
    }

    __aicore__ inline A8C4KvDequant(Arch::Resource<Arch::AtlasA5>& resource, uint32_t ubOffset, uint32_t storageHeads,
                                    uint32_t eventBase = 0U)
        : eventBase_(eventBase)
    {
        AscendC::SetSaturationStrategy<AscendC::OverrideStrategy::USE_API>();
        AscendC::SetSaturationFlag<AscendC::SaturationMode::FLOAT8>(true);
        packedTokenStride_ = storageHeads * (HEAD_DIM / 2U);
        scaleTokenStride_ = storageHeads * SCALE_BYTES;
        scaleRowStride_ = storageHeads == 1U ? SCALE_BYTES : STRIDED_SCALE_ROW_BYTES;
        const uint32_t packedBytes = HALF_TILE_TOKENS * (HEAD_DIM / 2U);
        const uint32_t inputBytes = packedBytes + HALF_TILE_TOKENS * scaleRowStride_;
        const uint32_t slotBytes = HalfTileScratchBytes(storageHeads);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            const uint32_t inputOffset = ubOffset + slot * slotBytes;
            packedUbTensor_[slot] = resource.ubBuf.template GetBufferByByte<uint8_t>(inputOffset);
            scaleUbTensor_[slot] = resource.ubBuf.template GetBufferByByte<uint8_t>(inputOffset + packedBytes);
            outputUbTensor_[slot] = resource.ubBuf.template GetBufferByByte<fp8_e4m3fn_t>(inputOffset + inputBytes);
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
                                       uint32_t tokens, uint32_t l1Tokens, uint32_t slot)
    {
        LoadTile<RECORD_WORDS, DATA_WORD, SCALE_WORD>(data, scales, descriptor, tokens, slot);
        FinishTile(dst, tokens, l1Tokens, slot);
    }

    template <uint32_t RECORD_WORDS, uint32_t DATA_WORD, uint32_t SCALE_WORD>
    __aicore__ inline void LoadTile(AscendC::GlobalTensor<uint8_t> data, AscendC::GlobalTensor<uint8_t> scales,
                                    AscendC::LocalTensor<uint64_t> descriptor, uint32_t tokens, uint32_t slot)
    {
        constexpr uint32_t BLOCK_TOKENS = 128U;
        const uint32_t split = ((tokens + 1U) / 2U + 15U) / 16U * 16U;
        const uint32_t begin = AscendC::GetSubBlockIdx() * split;
        if (begin >= tokens) {
            return;
        }
        const uint32_t count = min(split, tokens - begin);
        const auto event = slot == 0U ? eventBase_ : eventBase_ + 1U;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        const bool fullTile = tokens == 2U * HALF_TILE_TOKENS;
        if (fullTile) {
            const uint32_t record = begin / BLOCK_TOKENS * RECORD_WORDS;
            LoadTileSlice<DATA_WORD, SCALE_WORD>(data, scales, descriptor, record, 0U, BLOCK_TOKENS, slot, 0U);
            LoadTileSlice<DATA_WORD, SCALE_WORD>(data, scales, descriptor, record + RECORD_WORDS, 0U, BLOCK_TOKENS,
                                                 slot, BLOCK_TOKENS);
        } else {
            for (uint32_t token = begin; token < begin + count;) {
                const uint32_t record = token / BLOCK_TOKENS * RECORD_WORDS;
                const uint32_t blockOffset = token % BLOCK_TOKENS;
                const uint32_t slice = min(BLOCK_TOKENS - blockOffset, begin + count - token);
                LoadTileSlice<DATA_WORD, SCALE_WORD>(data, scales, descriptor, record, blockOffset, slice, slot,
                                                     token - begin);
                token += slice;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    struct PreparedTile {
        uint32_t begin = 0U;
        uint32_t count = 0U;
        uint32_t padded = 0U;
        uint32_t event = 0U;
        bool full = false;
        __ubuf__ uint8_t* packed = nullptr;
        __ubuf__ bfloat16_t* scale = nullptr;
        __ubuf__ fp8_e4m3fn_t* output = nullptr;
    };

    __aicore__ inline PreparedTile PrepareTile(uint32_t tokens, uint32_t slot)
    {
        PreparedTile tile;
        const uint32_t split = ((tokens + 1U) / 2U + 15U) / 16U * 16U;
        tile.begin = AscendC::GetSubBlockIdx() * split;
        if (tile.begin < tokens) {
            tile.count = min(split, tokens - tile.begin);
            tile.padded = (tile.count + 15U) / 16U * 16U;
            tile.event = eventBase_ + slot;
            tile.full = tokens == 2U * HALF_TILE_TOKENS;
            tile.packed = reinterpret_cast<__ubuf__ uint8_t*>(packedUbTensor_[slot].GetPhyAddr());
            tile.scale = reinterpret_cast<__ubuf__ bfloat16_t*>(scaleUbTensor_[slot].GetPhyAddr());
            tile.output = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(outputUbTensor_[slot].GetPhyAddr());
        }
        return tile;
    }

    __aicore__ inline void RunTileVf(const PreparedTile& tile)
    {
        if (tile.count == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(tile.event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(tile.event);
        if (tile.full) {
            if (scaleRowStride_ == SCALE_BYTES) {
                ComputeKvDequantCompact<HALF_TILE_TOKENS>(tile.packed, tile.scale, tile.output);
            } else {
                ComputeKvDequantCompact<HALF_TILE_TOKENS, STRIDED_SCALE_ROW_BYTES>(tile.packed, tile.scale,
                                                                                   tile.output);
            }
        } else if (tile.count == tile.padded) {
            ComputeKvDequant<false>(tile.packed, tile.scale, tile.output, tile.count, HEAD_DIM / 2U, 0U,
                                    scaleRowStride_, 0U, tile.padded);
        } else {
            ComputeKvDequant<true>(tile.packed, tile.scale, tile.output, tile.count, HEAD_DIM / 2U, 0U, scaleRowStride_,
                                   0U, tile.padded);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(tile.event);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(tile.event);
    }

    __aicore__ inline void StoreTile(AscendC::LocalTensor<fp8_e4m3fn_t> dst, uint32_t l1Tokens, uint32_t slot,
                                     const PreparedTile& tile)
    {
        if (tile.count == 0U) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(tile.event);
        const AscendC::DataCopyParams copy{4U, static_cast<uint16_t>(tile.padded), 0U,
                                           static_cast<uint16_t>(l1Tokens - tile.padded)};
        AscendC::DataCopy(dst[tile.begin * 32U], outputUbTensor_[slot], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(tile.event);
    }

    __aicore__ inline void FinishTile(AscendC::LocalTensor<fp8_e4m3fn_t> dst, uint32_t tokens, uint32_t l1Tokens,
                                      uint32_t slot)
    {
        const auto tile = PrepareTile(tokens, slot);
        RunTileVf(tile);
        StoreTile(dst, l1Tokens, slot, tile);
    }

private:
    template <uint32_t DATA_WORD, uint32_t SCALE_WORD>
    __aicore__ inline void LoadTileSlice(AscendC::GlobalTensor<uint8_t> data, AscendC::GlobalTensor<uint8_t> scales,
                                         AscendC::LocalTensor<uint64_t> descriptor, uint32_t record,
                                         uint32_t blockOffset, uint32_t tokens, uint32_t slot, uint32_t dstOffset)
    {
        constexpr uint32_t PACKED_BYTES = HEAD_DIM / 2U;
        const AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        const uint64_t packedOffset = descriptor.GetValue(record + DATA_WORD) + blockOffset * packedTokenStride_;
        const uint64_t scaleOffset = descriptor.GetValue(record + SCALE_WORD) + blockOffset * scaleTokenStride_;
        if (packedTokenStride_ == PACKED_BYTES) {
            AscendC::DataCopyPad(packedUbTensor_[slot][dstOffset * PACKED_BYTES], data[packedOffset],
                                 AscendC::DataCopyExtParams{1, tokens * PACKED_BYTES, 0, 0, 0}, pad);
            AscendC::DataCopyPad(scaleUbTensor_[slot][dstOffset * SCALE_BYTES], scales[scaleOffset],
                                 AscendC::DataCopyExtParams{1, tokens * SCALE_BYTES, 0, 0, 0}, pad);
        } else {
            const auto blocks = static_cast<uint16_t>(tokens);
            AscendC::DataCopyPad(
                packedUbTensor_[slot][dstOffset * PACKED_BYTES], data[packedOffset],
                AscendC::DataCopyExtParams{blocks, PACKED_BYTES, packedTokenStride_ - PACKED_BYTES, 0, 0}, pad);
            AscendC::DataCopyPad(scaleUbTensor_[slot][dstOffset * STRIDED_SCALE_ROW_BYTES], scales[scaleOffset],
                                 AscendC::DataCopyExtParams{blocks, SCALE_BYTES, scaleTokenStride_ - SCALE_BYTES, 0, 0},
                                 pad);
        }
    }

    template <uint16_t ROWS, uint32_t SCALE_ROW_BYTES = SCALE_BYTES>
    static __simd_vf__ inline void ComputeKvDequantCompact(__ubuf__ uint8_t* packedUb, __ubuf__ bfloat16_t* scaleUb,
                                                           __ubuf__ fp8_e4m3fn_t* outputUb)
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
        auto* write1 = write0 + 8U * (ROWS - 1U);
        auto* write2 = write0 + 16U * (ROWS - 1U);
        auto* write3 = write0 + 24U * (ROWS - 1U);
        MaskReg fp4Full = CreateMask<fp4x2_e2m1_t, MaskPattern::ALL>();
        // Prime row 0 before overlapping each row with the next input load.
        LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
            scales0, readScale, SCALE_ROW_BYTES / sizeof(bfloat16_t));
        LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
            reinterpret_cast<RegTensor<uint8_t>&>(codes0), readPacked, 64);
        Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(values0, codes0, fp4Full);
        for (uint16_t row = 0; row < ROWS - 2U; row += 2U) {
            {
                LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                    scales1, readScale, SCALE_ROW_BYTES / sizeof(bfloat16_t));
                LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                    reinterpret_cast<RegTensor<uint8_t>&>(codes1), readPacked, 64);
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
                    scales0, readScale, SCALE_ROW_BYTES / sizeof(bfloat16_t));
                LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                    reinterpret_cast<RegTensor<uint8_t>&>(codes0), readPacked, 64);
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
        {
            LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                scales1, readScale, SCALE_ROW_BYTES / sizeof(bfloat16_t));
            LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                reinterpret_cast<RegTensor<uint8_t>&>(codes1), readPacked, 64);
            Mul(product0, values0, scales0, full);
            Cast<float, bfloat16_t, widenEven>(even0, product0, full);
            Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(values1, codes1, fp4Full);
            Cast<float, bfloat16_t, widenOdd>(odd0, product0, full);
            Cast<fp8_e4m3fn_t, float, castFp8>(quantEven0, even0, floatFull);
            Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantOdd0, odd0, floatFull);
            Or(pair0, reinterpret_cast<RegTensor<uint8_t>&>(quantEven0),
               reinterpret_cast<RegTensor<uint8_t>&>(quantOdd0), byteFull);
            auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (ROWS - 2U) * 8U;
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
            auto* dst = reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + (ROWS - 1U) * 8U;
            auto& src = reinterpret_cast<RegTensor<uint32_t>&>(pair0);
            StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(dst, src, m0);
            StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write1, src, 8, m1);
            StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write2, src, 8, m2);
            StoreAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B32>(write3, src, 8, m3);
        }
    }

    template <bool NEED_PADDING>
    static __simd_vf__ inline void ComputeKvDequant(__ubuf__ uint8_t* packedUb, __ubuf__ bfloat16_t* scaleUb,
                                                    __ubuf__ fp8_e4m3fn_t* outputUb, uint16_t rows,
                                                    uint16_t packedStride, uint16_t packedHead, uint16_t scaleStride,
                                                    uint16_t scaleHead, uint16_t outputRows)
    {
        using namespace AscendC::Reg;
        static constexpr CastTrait castFp4 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::UNKNOWN};
        static constexpr CastTrait castFp8 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castFp8Odd = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING,
                                                 AscendC::RoundMode::CAST_RINT};
        RegTensor<fp4x2_e2m1_t> codesVreg;
        RegTensor<bfloat16_t> valuesVreg, scalesVreg, productVreg;
        RegTensor<fp8_e4m3fn_t> quantizedVreg, quantizedOddVreg;
        RegTensor<float> evenVreg, oddVreg;
        MaskReg floatFull = CreateMask<float, MaskPattern::ALL>();
        RegTensor<uint8_t> pairVreg;
        RegTensor<uint16_t> packedResultVreg;
        MaskReg byteFull = CreateMask<uint8_t, MaskPattern::ALL>();
        uint32_t byteCount = 128U;
        MaskReg bytes128 = UpdateMask<uint8_t>(byteCount);
        static constexpr CastTrait widenEven = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                AscendC::RoundMode::UNKNOWN};
        static constexpr CastTrait widenOdd = {RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::UNKNOWN};
        MaskReg full = CreateMask<bfloat16_t, MaskPattern::ALL>();
        if constexpr (NEED_PADDING) {
            RegTensor<uint32_t> zeroVreg;
            Duplicate(zeroVreg, 0U);
            // outputRows is aligned to 16, so every clear iteration covers a full register.
            const uint32_t clearCount = outputRows * HEAD_DIM / sizeof(uint32_t);
            const uint16_t loops = static_cast<uint16_t>(clearCount / 64U);
            MaskReg outputMask = CreateMask<uint32_t, MaskPattern::ALL>();
            for (uint16_t i = 0; i < loops; ++i) {
                StoreAlign<uint32_t>(reinterpret_cast<__ubuf__ uint32_t*>(outputUb) + i * 64U, zeroVreg, outputMask);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>();
        }
        MaskReg fp4Full = CreateMask<fp4x2_e2m1_t, MaskPattern::ALL>();
        for (uint16_t row = 0; row < rows; ++row) {
            LoadAlign<bfloat16_t, LoadDist::DIST_E2B_B16>(
                scalesVreg, scaleUb + (row * scaleStride + scaleHead) / sizeof(bfloat16_t));
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK4_B8>(reinterpret_cast<RegTensor<uint8_t>&>(codesVreg),
                                                          packedUb + row * packedStride + packedHead);
            Cast<bfloat16_t, fp4x2_e2m1_t, castFp4>(valuesVreg, codesVreg, fp4Full);
            // Multiply decoded FP4 E2M1 by the per-block BF16 scale before casting to FP8 E4M3FN.
            Mul(productVreg, valuesVreg, scalesVreg, full);
            Cast<float, bfloat16_t, widenEven>(evenVreg, productVreg, full);
            Cast<float, bfloat16_t, widenOdd>(oddVreg, productVreg, full);
            Cast<fp8_e4m3fn_t, float, castFp8>(quantizedVreg, evenVreg, floatFull);
            Cast<fp8_e4m3fn_t, float, castFp8Odd>(quantizedOddVreg, oddVreg, floatFull);
            Or(pairVreg, reinterpret_cast<RegTensor<uint8_t>&>(quantizedVreg),
               reinterpret_cast<RegTensor<uint8_t>&>(quantizedOddVreg), byteFull);
            Pack(packedResultVreg, reinterpret_cast<RegTensor<uint32_t>&>(pairVreg));
            StoreAlign<uint8_t, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint8_t*>(outputUb) + row * 32U,
                reinterpret_cast<RegTensor<uint8_t>&>(packedResultVreg), static_cast<uint16_t>(outputRows), bytes128);
        }
    }

    uint32_t eventBase_;
    uint32_t scaleRowStride_;
    uint32_t packedTokenStride_;
    uint32_t scaleTokenStride_;
    AscendC::LocalTensor<uint8_t> packedUbTensor_[2];
    AscendC::LocalTensor<uint8_t> scaleUbTensor_[2];
    AscendC::LocalTensor<fp8_e4m3fn_t> outputUbTensor_[2];
};
} // namespace NpuArch::Epilogue::Block
#endif
