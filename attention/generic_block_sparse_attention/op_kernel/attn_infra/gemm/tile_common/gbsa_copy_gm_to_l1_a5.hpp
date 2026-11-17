/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_GM_TO_L1_A5_HPP
#define GEMM_TILE_COPY_GM_TO_L1_A5_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_arch.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_tile_copy_tla.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"

namespace NpuArch::Gemm::Tile {

/// Partial specialization for CopyGmToL1, AtlasA5, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t ndNum = 1,
                                      uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN");

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint64_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
                   tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and zN, while TensorDst must be L1 and zN");

        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::AtlasA5, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint32_t ndNum = 1,
                                      uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        static_assert(tla::detail::isColumnMajor<typename TensorSrc::Layout>::value &&
                          tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
                      "while TensorDst must be L1 and nZ");

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint64_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzC0Stride = dstOuterStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, AtlasA5, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
                   tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value &&
                                    tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    __aicore__ inline TileCopyTla(){};

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor)
    {
        static_assert(tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
                          tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::GM &&
                          TensorDst::position == AscendC::TPosition::A1,
                      "The input parameters do not match. TensorSrc must be GM and nZ, "
                      "while TensorDst must be L1 and nZ");

        const uint32_t blockCount = tla::get<0, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

} // namespace NpuArch::Gemm::Tile

#if (__CCE_AICORE__ == 310) && defined(ORIG_DTYPE_QUERY) && defined(ORIG_DTYPE_KEY) && \
    (ORIG_DTYPE_QUERY == DT_FLOAT4_E2M1) && (ORIG_DTYPE_KEY == DT_FLOAT4_E2M1)
#include "../block/gbsa_block_mmad_arch35_mxfp4_utils.hpp"

namespace NpuArch::Gemm::Tile {

__aicore__ inline void PreloadMxfp4DescriptorHeader(const AscendC::GlobalTensor<uint64_t>& descriptor)
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
__aicore__ inline void RefreshMxfp4Descriptor(const AscendC::GlobalTensor<uint64_t>& descriptor, uint32_t groupWords)
{
    if constexpr (!HeaderReady) {
        PreloadMxfp4DescriptorHeader(descriptor);
    }
    const __gm__ uint64_t* records = descriptor.GetPhyAddr();
    // MXFP4 dense and PA both store [offset, meta]; reserved slot spans stay unchanged.
    const uint32_t end = min(groupWords, 8U + static_cast<uint32_t>(records[0]) * 2U);
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
__aicore__ inline bool CanCopyKScaleDirect(const AscendC::GlobalTensor<uint8_t>& keyScale, Records records,
                                           uint32_t tokenStrideBytes, uint32_t D, uint32_t GroupRows)
{
    if (tokenStrideBytes != D / 2 || records[0] != 1) {
        return false;
    }
    const uint64_t offset = records[8];
    if (static_cast<uint32_t>(records[9]) != GroupRows * 2U || offset % 16 != 0) {
        return false;
    }
    // Conservative first implementation: the complete source scale address
    // must be 32B-aligned, including its batch/head base.
    const uint64_t address = reinterpret_cast<uint64_t>(keyScale.GetPhyAddr()) + offset / 16;
    return address % 32 == 0;
}

// Only used for contiguous Q banks starting on a 16-row boundary. Sparse K
// scale uses PrepareKScaleMxfp4, as Y=4/8 fragments cannot use this DN2NZ offset.
struct CopyQueryScaleGmToL1Mxfp4 {
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
struct CopySparsePackedGmToL1Mxfp4 {
    uint32_t D, KV_ROWS, GroupRows;
    __aicore__ CopySparsePackedGmToL1Mxfp4(const GbsaKernelArch35::Mxfp4::L1TileHelper& l1Info,
                                           const GbsaKernelArch35::Mxfp4::DescriptorTileHelper<Paged>& descriptorInfo)
        : D(l1Info.D),
          KV_ROWS(l1Info.KV_ROWS),
          GroupRows(descriptorInfo.ROWS)
    {}

    template <bool Value>
    __aicore__ inline void operator()(const AscendC::LocalTensor<uint8_t>& dst,
                                      const AscendC::GlobalTensor<uint8_t>& src,
                                      const AscendC::GlobalTensor<uint64_t>& descriptors, uint32_t groupWords,
                                      uint32_t validTokens, uint32_t sourceTokenStrideBytes,
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
                                           uint32_t validTokens, uint32_t sourceTokenStrideBytes,
                                           uint32_t firstToken = 0U) const
    {
        Copy<false, true, HeaderReady>(dst, src, descriptors, groupWords, validTokens, sourceTokenStrideBytes, dstScale,
                                       srcScale, firstToken);
    }

private:
    template <bool Value, bool WithScale, bool HeaderReady = false>
    __aicore__ inline void Copy(const AscendC::LocalTensor<uint8_t>& dst, const AscendC::GlobalTensor<uint8_t>& src,
                                const AscendC::GlobalTensor<uint64_t>& descriptors, uint32_t groupWords,
                                uint32_t validTokens, uint32_t sourceTokenStrideBytes,
                                const AscendC::LocalTensor<uint8_t>& dstScale,
                                const AscendC::GlobalTensor<uint8_t>& srcScale, uint32_t firstToken = 0U) const
    {
        constexpr uint32_t recordWords = 2;
        // Fix the logical position before the descriptor loop, as in the
        // non-quantized Block. DataCopy can then select L1 at compile time.
        const AscendC::LocalTensor<uint8_t> l1Base(AscendC::TPosition::A1, 0, Arch::AtlasA5::L1_SIZE);
        const uint32_t dstAddress = static_cast<uint32_t>((uint64_t)dst.GetPhyAddr() - (uint64_t)l1Base.GetPhyAddr());
        const AscendC::LocalTensor<uint8_t> dstData(AscendC::TPosition::A1, dstAddress, dst.GetSize());
        auto l1Layout = tla::MakeLayout<uint8_t, layout::zN>(KV_ROWS, D / 2);
        auto l1Tensor = tla::MakeTensor(dstData, l1Layout, Arch::PositionL1{});
        AscendC::Nd2NzParams copyParams;
        copyParams.dValue = D / 2;
        copyParams.srcDValue = sourceTokenStrideBytes;
        copyParams.dstNzC0Stride = tla::get<1, 1>(l1Tensor.stride()) / BYTE_PER_C0;
        copyParams.dstNzNStride = tla::get<0, 0>(l1Tensor.stride()) / BYTE_PER_C0;
        for (uint32_t group = 0; group * GroupRows < validTokens; ++group) {
            auto descriptor = descriptors[group * groupWords];
            if constexpr (!Value) {
                // Start this group's DMA before refreshing the next group.
                // PV reuses the immutable descriptor until it releases the slot.
                RefreshMxfp4Descriptor<Paged, HeaderReady>(descriptor, groupWords);
            }
            const __gm__ uint64_t* records = descriptor.GetPhyAddr();
            const uint32_t count = records[0];
            uint32_t dstRow = firstToken + group * GroupRows;
            if constexpr (WithScale && !Paged) {
                if (CanCopyKScaleDirect(srcScale, records, sourceTokenStrideBytes, D, GroupRows)) {
                    CopyQueryScaleGmToL1Mxfp4 copyScale;
                    copyScale(dstScale, srcScale[records[8] / 16], 0, dstRow, GroupRows, D);
                }
            }
            const __gm__ uint64_t* record = records + 8;
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

// MX B scale: GM [KV/64,D,2] -> L1 [D/16,KV/64,16,2].
// Two E8M0 bytes are a B16 copy unit, with no numerical conversion.
template <class ElementMx, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA5,
                   tla::Tensor<AscendC::GlobalTensor<ElementMx>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
                   tla::Tensor<AscendC::LocalTensor<ElementMx>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
                   std::enable_if_t<tla::detail::isMxScaleBNoTrans<ElementMx, LayoutSrc>::value &&
                                    tla::detail::isMxScalenN<ElementMx, LayoutDst>::value>> {
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor) const
    {
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = tla::get<0, 1>(srcTensor.shape());
        params.dValue = tla::get<1>(srcTensor.shape());
        params.srcNdMatrixStride = 0;
        params.srcDValue = tla::get<0, 1>(srcTensor.stride()) / 2;
        params.dstNzC0Stride = tla::get<1, 1>(dstTensor.stride()) / BYTE_PER_C0;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        auto src = srcTensor.data()[srcTensor.layout()(srcTensor.coord())].template ReinterpretCast<half>();
        auto dst = dstTensor.data()[dstTensor.layout()(dstTensor.coord())].template ReinterpretCast<half>();
        AscendC::DataCopy(dst, src, params);
    }
};

} // namespace NpuArch::Gemm::Tile
#endif // MXFP4 on AtlasA5

#endif // GEMM_TILE_COPY_GM_TO_L1_A5_HPP
