/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MX_PACKED_RECORDS_HPP
#define GBSA_MX_PACKED_RECORDS_HPP

// Extracted from the nonquant QK/PV descriptor consumers. It changes no MX scale,
// matrix, L0 or cross-core event handling.
namespace NpuArch::Gemm::Tile {

namespace Mxfp8DescriptorLayout {
constexpr uint32_t GROUP_TOKENS = 128U;
constexpr uint32_t RING_SLOTS = 4U;

__aicore__ inline uint64_t DescriptorGroupWords()
{
    // Reserve an eight-word header and two words per possible token span.
    // Keep this conservative descriptor capacity independent of blockY.
    return 8U + 2U * GROUP_TOKENS;
}

__aicore__ inline uint64_t DescriptorTileWords(uint64_t tile)
{
    // Keep a header for AIV1's empty group in the single-group tier.
    return 2U * static_cast<uint32_t>((tile / GROUP_TOKENS + 1U) / 2U) * DescriptorGroupWords();
}
} // namespace Mxfp8DescriptorLayout

// Caller must first wait for the producer's MTE3-ready cross-core event on PIPE_S.
// The same GM slot may hold a previous tile: a ready flag alone does not invalidate
// the AIC scalar data cache. QK refreshes once, and PV reuses these cache lines.
// Descriptor layout: 64-byte header, then {elementOffset, packedMeta} records.
__aicore__ inline uint32_t RefreshGbsaMxPackedRecords(AscendC::GlobalTensor<uint64_t> info)
{
    constexpr uint32_t CACHE_LINE_WORDS = 8U;
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        info);
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        info[CACHE_LINE_WORDS]);
    AscendC::DataCachePreload(info, static_cast<int64_t>(0));
    AscendC::DataCachePreload(info[CACHE_LINE_WORDS], static_cast<int64_t>(0));

    // READY and the cache refresh protect this descriptor generation. Read it
    // through the scalar tensor API; the producer cannot reuse the slot until
    // the delayed PV consumer releases it.
    const uint32_t recordCount = static_cast<uint32_t>(info.GetValue(0U));
    const uint32_t recordEnd = CACHE_LINE_WORDS + recordCount * 2U;
    // Keep invalidation and prefetch in separate loops: a later DCCI must not
    // drain an earlier descriptor prefetch. The first record line is done above.
    for (uint32_t word = 2U * CACHE_LINE_WORDS; word < recordEnd; word += CACHE_LINE_WORDS) {
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(info[word]);
    }
    for (uint32_t word = 2U * CACHE_LINE_WORDS; word < recordEnd; word += CACHE_LINE_WORDS) {
        AscendC::DataCachePreload(info[word], static_cast<int64_t>(0));
    }
    // Only the packed producer uses this tag; callers preserve it until copy.
    return recordCount | (info.GetValue(1U) == 1U ? 0x80000000U : 0U);
}

// Records stay within one QK L1-N group: 128 tokens in the grouped 512-token
// pipeline, or the non-grouped tile size. QK/PV use exactly the same
// logical row order; paired records must NOT reorder either source block.
// Source matrix strides are in ElementB elements. The initial producer uses a
// conservative 32-byte-aligned pairing guard; D=192/256, blockY=4/8 and dense
// BSND satisfy it. This is not a claim that all ND2NZ source strides need 32B.
// Any tail padding/zero-fill remains the caller's responsibility, as before.
template <bool IS_PV, class ElementB, class ArchTag, class TensorB, class TensorL1B>
__aicore__ inline void CopyGbsaMxPackedRecords(TensorB& gBTensor, TensorL1B& l1BTensor,
                                               AscendC::GlobalTensor<uint64_t> info, uint32_t recordCount,
                                               uint32_t embed, uint32_t embedInnerOffset = 0U)
{
    static_assert(sizeof(ElementB) == 1U, "This candidate targets packed FP8 K/V only");
    constexpr uint32_t ELEMENTS_PER_C0 = 32U / sizeof(ElementB);
    const auto& srcData = gBTensor.data();
    const auto& dstStorage = l1BTensor.data();
    // Preserve the proven nonquant consumer's explicit L1 address rebinding.
    const AscendC::LocalTensor<ElementB> l1Base(AscendC::TPosition::A1, 0, ArchTag::L1_SIZE / sizeof(ElementB));
    const uint32_t dstAddr = static_cast<uint32_t>((uint64_t)dstStorage.GetPhyAddr() - (uint64_t)l1Base.GetPhyAddr());
    const AscendC::LocalTensor<ElementB> dstData(AscendC::TPosition::A1, dstAddr, dstStorage.GetSize());
    const auto dstBase = l1BTensor.layout()(l1BTensor.coord());
    uint32_t dstTokenStride;
    AscendC::Nd2NzParams copyParams{};
    copyParams.dValue = embed;
    if constexpr (IS_PV) {
        dstTokenStride = tla::get<0, 0>(l1BTensor.stride());
        copyParams.srcDValue = tla::get<0>(gBTensor.stride());
        copyParams.dstNzC0Stride = tla::get<1, 1>(l1BTensor.stride()) / ELEMENTS_PER_C0;
    } else {
        dstTokenStride = tla::get<1, 0>(l1BTensor.stride());
        copyParams.srcDValue = tla::get<1>(gBTensor.stride());
        copyParams.dstNzC0Stride = tla::get<0, 1>(l1BTensor.stride()) / ELEMENTS_PER_C0;
    }
    copyParams.dstNzNStride = dstTokenStride / ELEMENTS_PER_C0;

    if ((recordCount & 0x80000000U) != 0U) {
        // Prepared FP8 NZ records: src{stride32, offset32},
        // dst{rows16, unused16, byteOffset16, ndNum16}. Both packed NZ
        // layouts have a 32-byte token stride. Preserve runtime C0 pitch:
        // QK rounds N to16 while PV pads the reduction to64.
        // These are the same intrinsics used by AscendC's dav_3510
        // DataCopyGM2L1ND2NZImplBase, with the per-record NZ bits prepacked.
        const uint64_t geometry = (static_cast<uint64_t>(copyParams.dstNzC0Stride) << 32U) |
                                  (static_cast<uint64_t>(copyParams.dstNzNStride) << 16U);
        auto dst = reinterpret_cast<__cbuf__ int8_t*>(dstData.GetPhyAddr()) + dstBase;
        // The intrinsic API takes a mutable GM pointer but only reads it.
        auto src =
            const_cast<__gm__ int8_t*>(reinterpret_cast<const __gm__ int8_t*>(srcData.GetPhyAddr())) + embedInnerOffset;
        // Use a counted loop. A pointer-inequality loop made this compiler
        // branch through an800-NOP prefix on every descriptor iteration.
        const uint32_t count = recordCount & 0x7fffffffU;
        for (uint32_t index = 0U; index < count; ++index) {
            const uint64_t source = info.GetValue(8U + index * 2U);
            const uint64_t target = info.GetValue(9U + index * 2U);
            set_mte2_nz_para((target & 0xffff00000000ffffULL) | geometry);
            copy_gm_to_cbuf_multi_nd2nz(dst + ((target >> 16U) & 0xffffU), src + static_cast<uint32_t>(source), 0U,
                                        copyParams.srcDValue, 0U, static_cast<uint16_t>(target >> 48U), embed,
                                        source >> 32U, false);
        }
        return;
    }
    uint32_t rowOffset = 0U;
    for (uint32_t record = 0U; record < recordCount; ++record) {
        const uint32_t word = 8U + record * 2U;
        const uint64_t sourceOffset = info.GetValue(word);
        const uint64_t meta = info.GetValue(word + 1U);
        const uint32_t rows = static_cast<uint32_t>(meta) >> 1U;
        const uint32_t matrices = 1U + static_cast<uint32_t>(meta & 1U);
        copyParams.ndNum = matrices;
        copyParams.nValue = rows;
        copyParams.srcNdMatrixStride = static_cast<uint32_t>(meta >> 32U);
        copyParams.dstNzMatrixStride = rows * dstTokenStride;
        AscendC::DataCopy(dstData[dstBase + rowOffset * dstTokenStride], srcData[sourceOffset + embedInnerOffset],
                          copyParams);
        rowOffset += matrices * rows;
    }
}

} // namespace NpuArch::Gemm::Tile

#endif // GBSA_MX_PACKED_RECORDS_HPP
