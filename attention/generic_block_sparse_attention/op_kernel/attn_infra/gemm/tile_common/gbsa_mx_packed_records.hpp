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

#include "gbsa_copy_gm_to_l1_a5.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"

namespace NpuArch::Gemm::Tile {

__aicore__ inline void PreloadMxfp8FullDescriptorHeader(const AscendC::GlobalTensor<uint64_t>& descriptor)
{
    // Called only after both AIVs publish this descriptor generation.
    for (uint32_t word = 0; word < 16; word += 8) {
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(descriptor[word]);
    }
    AscendC::DataCachePreload(descriptor, static_cast<int64_t>(0));
    AscendC::DataCachePreload(descriptor[8], static_cast<int64_t>(0));
}

template <bool Paged = false, bool HeaderReady = false>
__aicore__ inline void RefreshMxfp8FullDescriptor(const AscendC::GlobalTensor<uint64_t>& descriptor,
                                                  uint32_t groupWords)
{
    if constexpr (!HeaderReady) {
        PreloadMxfp8FullDescriptorHeader(descriptor);
    }
    const __gm__ uint64_t* records = descriptor.GetPhyAddr();
    // MXFP8 dense and PA both store [offset, meta]; reserved slot spans stay unchanged.
    const uint32_t end =
        min(groupWords, GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS +
                            static_cast<uint32_t>(records[0]) * GbsaKernelArch35::Mxfp8Descriptor::RECORD_WORDS);
    for (uint32_t word = 16; word < end; word += 8) {
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(descriptor[word]);
    }
    for (uint32_t word = 16; word < end; word += 8) {
        AscendC::DataCachePreload(descriptor[word], static_cast<int64_t>(0));
    }
}

// Both producers use this predicate on the same immutable descriptor. A direct
// group owns complete scale fractals; short or paired records remain on AIV.
template <class Records>
__aicore__ inline bool CanCopyKScaleDirectMxfp8Full(const AscendC::GlobalTensor<uint8_t>& keyScale, Records records,
                                                    uint64_t tokenStrideBytes, uint32_t D, uint32_t GroupRows)
{
    if (tokenStrideBytes != D || records[0] != 1) {
        return false;
    }
    const uint64_t offset = records[GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS];
    if (static_cast<uint32_t>(records[GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS + 1U]) != GroupRows * 2U ||
        offset % 32 != 0) {
        return false;
    }
    // Conservative first implementation: the complete source scale address
    // must be 32B-aligned, including its batch/head base.
    const uint64_t address = reinterpret_cast<uint64_t>(keyScale.GetPhyAddr()) + offset / 32;
    return address % 32 == 0;
}

// GM offsets and strides are bytes: one FP8 element occupies one byte.
struct CopyPackedGmToL1Mxfp8Full {
    __aicore__ inline void operator()(const AscendC::LocalTensor<uint8_t>& dst,
                                      const AscendC::GlobalTensor<uint8_t>& src, uint64_t sourceByteOffset,
                                      uint32_t dstRow, uint32_t rows, uint32_t headDim, uint64_t sourceTokenStrideBytes,
                                      uint32_t parentRows, uint32_t matrices = 1, uint32_t sourceMatrixStride = 0) const
    {
        if (rows == 0) {
            return;
        }
        auto gLayout =
            tla::MakeLayout(tla::MakeShape(rows, headDim), tla::MakeStride(sourceTokenStrideBytes, tla::Int<1>{}));
        auto gTensor = tla::MakeTensor(src[sourceByteOffset], gLayout, Arch::PositionGM{});
        auto l1Layout = tla::MakeLayout<uint8_t, layout::zN>(parentRows, headDim);
        auto l1Tensor = tla::MakeTensor(dst, l1Layout, Arch::PositionL1{});
        auto l1Tile = GetTile(l1Tensor, tla::MakeCoord(dstRow, 0), tla::MakeShape(rows, headDim));
        TileCopyTla<Arch::AtlasA5, decltype(gTensor), decltype(l1Tile)> copyGmToL1;
        copyGmToL1(l1Tile, gTensor, matrices, sourceMatrixStride, rows * BYTE_PER_C0);
    }
};

// Only used for contiguous Q banks starting on a 16-row boundary. Sparse K
// scale uses PrepareKScaleMxfp8Full, as Y=4/8 fragments cannot use this DN2NZ offset.
struct CopyQueryScaleGmToL1Mxfp8Full {
    __aicore__ inline void operator()(const AscendC::LocalTensor<uint8_t>& dst,
                                      const AscendC::GlobalTensor<uint8_t>& src, uint32_t sourceRow,
                                      uint32_t destinationRow, uint32_t rows, uint32_t headDim,
                                      uint32_t headStride = 1) const
    {
        if (rows == 0) {
            return;
        }
        const uint32_t pairs = headDim / 64;
        AscendC::GlobalTensor<half> source;
        source.SetGlobalBuffer((__gm__ half*)src.GetPhyAddr());
        AscendC::Dn2NzParams params;
        params.dnNum = 1;
        params.dValue = rows;
        params.nValue = pairs;
        params.srcDnMatrixStride = 0;
        params.srcDValue = pairs * headStride;
        params.dstNzC0Stride = pairs;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst.ReinterpretCast<half>()[destinationRow * pairs], source[sourceRow * pairs], params);
    }
};

template <bool Paged>
struct CopySparsePackedGmToL1Mxfp8Full {
    uint32_t D, KV_ROWS, GroupRows;
    __aicore__ CopySparsePackedGmToL1Mxfp8Full(uint32_t dim, uint32_t kvRows, uint32_t groupRows)
        : D(dim),
          KV_ROWS(kvRows),
          GroupRows(groupRows)
    {}

    template <bool Value>
    __aicore__ inline void operator()(const AscendC::LocalTensor<uint8_t>& dst,
                                      const AscendC::GlobalTensor<uint8_t>& src,
                                      const AscendC::GlobalTensor<uint64_t>& descriptors, uint32_t groupWords,
                                      uint32_t validTokens, uint64_t sourceTokenStrideBytes,
                                      uint32_t firstToken = 0U) const
    {
        Copy<Value, false>(dst, src, descriptors, groupWords, validTokens, sourceTokenStrideBytes, dst, src,
                           firstToken);
    }

    template <bool HeaderReady = false>
    __aicore__ inline void CopyKeyAndScale(const AscendC::LocalTensor<uint8_t>& dst,
                                           const AscendC::GlobalTensor<uint8_t>& src,
                                           const AscendC::LocalTensor<uint8_t>& dstScale,
                                           const AscendC::GlobalTensor<uint8_t>& srcScale,
                                           const AscendC::GlobalTensor<uint64_t>& descriptors, uint32_t groupWords,
                                           uint32_t validTokens, uint64_t sourceTokenStrideBytes,
                                           uint32_t firstToken = 0U) const
    {
        Copy<false, true, HeaderReady>(dst, src, descriptors, groupWords, validTokens, sourceTokenStrideBytes, dstScale,
                                       srcScale, firstToken);
    }

private:
    template <bool Value, bool WithScale, bool HeaderReady = false>
    __aicore__ inline void Copy(const AscendC::LocalTensor<uint8_t>& dst, const AscendC::GlobalTensor<uint8_t>& src,
                                const AscendC::GlobalTensor<uint64_t>& descriptors, uint32_t groupWords,
                                uint32_t validTokens, uint64_t sourceTokenStrideBytes,
                                const AscendC::LocalTensor<uint8_t>& dstScale,
                                const AscendC::GlobalTensor<uint8_t>& srcScale, uint32_t firstToken = 0U) const
    {
        constexpr uint32_t recordWords = GbsaKernelArch35::Mxfp8Descriptor::RECORD_WORDS;
        // Fix the logical position before the descriptor loop, as in the
        // non-quantized Block. DataCopy can then select L1 at compile time.
        const AscendC::LocalTensor<uint8_t> l1Base(AscendC::TPosition::A1, 0, Arch::AtlasA5::L1_SIZE);
        const uint32_t dstAddress = static_cast<uint32_t>((uint64_t)dst.GetPhyAddr() - (uint64_t)l1Base.GetPhyAddr());
        const AscendC::LocalTensor<uint8_t> dstData(AscendC::TPosition::A1, dstAddress, dst.GetSize());
        auto l1Layout = tla::MakeLayout<uint8_t, layout::zN>(KV_ROWS, D);
        auto l1Tensor = tla::MakeTensor(dstData, l1Layout, Arch::PositionL1{});
        AscendC::Nd2NzParams copyParams;
        copyParams.dValue = D;
        copyParams.srcDValue = sourceTokenStrideBytes;
        copyParams.dstNzC0Stride = tla::get<1, 1>(l1Tensor.stride()) / BYTE_PER_C0;
        copyParams.dstNzNStride = tla::get<0, 0>(l1Tensor.stride()) / BYTE_PER_C0;
        for (uint32_t group = 0; group * GroupRows < validTokens; ++group) {
            auto descriptor = descriptors[group * groupWords];
            if constexpr (!Value) {
                // Start this group's DMA before refreshing the next group.
                // PV reuses the immutable descriptor until it releases the slot.
                RefreshMxfp8FullDescriptor<Paged, HeaderReady>(descriptor, groupWords);
            }
            const __gm__ uint64_t* records = descriptor.GetPhyAddr();
            const uint32_t count = records[0];
            uint32_t dstRow = firstToken + group * GroupRows;
            if constexpr (WithScale && !Paged) {
                if (CanCopyKScaleDirectMxfp8Full(srcScale, records, sourceTokenStrideBytes, D, GroupRows)) {
                    CopyQueryScaleGmToL1Mxfp8Full copyScale;
                    copyScale(dstScale, srcScale[records[GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS] / 32], 0,
                              dstRow, GroupRows, D);
                }
            }
            const __gm__ uint64_t* record = records + GbsaKernelArch35::Mxfp8Descriptor::HEADER_WORDS;
            for (uint32_t index = 0; index < count; ++index, record += recordWords) {
                const uint64_t offset = record[0];
                const uint64_t meta = record[1];
                const uint32_t rows = static_cast<uint32_t>(meta) >> 1;
                const uint32_t matrices = 1 + static_cast<uint32_t>(meta & 1);
                const uint32_t stride = static_cast<uint32_t>(meta >> 32);
                // Paired records use one ND2NZ, preserving compact destination order.
                copyParams.ndNum = matrices;
                copyParams.nValue = rows;
                copyParams.srcNdMatrixStride = stride;
                copyParams.dstNzMatrixStride = rows * BYTE_PER_C0;
                // The producer emits only positive-length records; an empty
                // group has count == 0 and never enters this loop.
                AscendC::DataCopy(dstData[l1Layout(tla::MakeCoord(dstRow, 0))], src[offset], copyParams);
                dstRow += rows * matrices;
            }
        }
    }
};

} // namespace NpuArch::Gemm::Tile
#endif // GBSA_MX_PACKED_RECORDS_HPP
