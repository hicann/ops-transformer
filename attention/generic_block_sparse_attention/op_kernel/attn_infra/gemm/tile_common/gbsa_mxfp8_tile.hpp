/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_MXFP8_TILE_HPP
#define GBSA_MXFP8_TILE_HPP
#include "gbsa_gemm_tile_copy.hpp"
#include "gbsa_mx_packed_records.hpp"

namespace NpuArch::Gemm::Tile {
// Reserve the final 16 KiB of L1 for scales. Supported GBSA geometry uses
// at most 192 KiB of data buffers; scale slots are independent of data strides.
constexpr uint32_t GBSA_MX_SCALE_BASE = Arch::AtlasA5::L1_SIZE - 16384U;
constexpr uint32_t GBSA_MX_SCALE_SLOT = 2048U;

__aicore__ inline void CopyGbsaMxQKScale(AscendC::LocalTensor<uint8_t> dst, AscendC::GlobalTensor<uint8_t> src,
                                         uint32_t rows, uint32_t groups64, uint64_t srcRowStrideHalf)
{
    AscendC::GlobalTensor<half> src16;
    src16.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ uint8_t*>(src.GetPhyAddr())));
    AscendC::Dn2NzParams p;
    p.dnNum = 1;
    p.dValue = rows;
    p.nValue = groups64;
    p.srcDnMatrixStride = 0;
    p.srcDValue = srcRowStrideHalf;
    p.dstNzC0Stride = groups64;
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = groups64;
    AscendC::DataCopy(dst.ReinterpretCast<half>(), src16, p);
}

__aicore__ inline void CopyGbsaMxVScale(AscendC::LocalTensor<uint8_t> dst, AscendC::GlobalTensor<uint8_t> src,
                                        uint32_t channels, uint32_t groups64, uint32_t heads, uint32_t dstGroups64,
                                        uint32_t dstGroupOffset = 0U)
{
    AscendC::GlobalTensor<half> src16;
    src16.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ uint8_t*>(src.GetPhyAddr())));
    // GM [page, block/64, head, D, 2] -> scale NZ [D/16, block/64, 16, 2].
    AscendC::Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = groups64;
    p.dValue = channels;
    p.srcNdMatrixStride = 0;
    p.srcDValue = heads * channels;
    p.dstNzC0Stride = dstGroups64;
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = dstGroups64 * channels;
    AscendC::DataCopy(dst.ReinterpretCast<half>()[dstGroupOffset * 16U], src16, p);
}

// Keep the existing NZ row order while reading even and odd tokens as two
// interleaved ND matrices. This avoids read amplification for complete pages
// whose 128-byte rows have a 256-byte source stride, without adding a DMA.
template <class BaseCopy, bool IS_PV>
struct GbsaMxCopyGmToL1B : BaseCopy {
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src, uint32_t ndNum = 1,
                                      uint32_t srcNdMatrixStride = 0, uint32_t dstNzMatrixStride = 0)
    {
        if constexpr (sizeof(typename TensorSrc::Element) == 1 && sizeof(typename TensorDst::Element) == 1) {
            uint32_t nValue;
            uint32_t dValue;
            uint64_t srcDValue;
            uint32_t dstNzC0Stride;
            uint32_t dstNzNStride;
            if constexpr (IS_PV) {
                nValue = tla::get<0>(src.shape());
                dValue = tla::get<1>(src.shape());
                srcDValue = tla::get<0>(src.stride());
                dstNzC0Stride = tla::get<1, 1>(dst.stride()) / 32U;
                dstNzNStride = tla::get<0, 0>(dst.stride()) / 32U;
            } else {
                nValue = tla::get<1>(src.shape());
                dValue = tla::get<0>(src.shape());
                srcDValue = tla::get<1>(src.stride());
                dstNzC0Stride = tla::get<0, 1>(dst.stride()) / 32U;
                dstNzNStride = tla::get<1, 0>(dst.stride()) / 32U;
            }
            if (ndNum == 1U && nValue == 128U && dValue == 128U && srcDValue == 256U && dstNzNStride == 1U) {
                AscendC::Nd2NzParams p;
                p.ndNum = 2;
                p.nValue = 64;
                p.dValue = 128;
                // Source and matrix strides are in elements (one byte here).
                p.srcNdMatrixStride = 256;
                p.srcDValue = 512;
                // NZ N/C0 strides are in 32-byte blocks; retain the tile pitch.
                p.dstNzC0Stride = dstNzC0Stride;
                p.dstNzNStride = 2;
                p.dstNzMatrixStride = 32;
                const auto dstOffset = dst.layout()(dst.coord());
                const auto srcOffset = src.layout()(src.coord());
                AscendC::DataCopy(dst.data()[dstOffset], src.data()[srcOffset], p);
                return;
            }
        }
        BaseCopy::operator()(dst, src, ndNum, srcNdMatrixStride, dstNzMatrixStride);
    }
};

// A is Q/P (zN); QK B is K (nZ); PV B is V (zN, transposed by LoadData).
// MX scales use pairs of E8M0 bytes, one byte for each 32-element K group.
template <bool IS_A, bool IS_PV>
struct GbsaMxLoadL0 {
    AscendC::LocalTensor<uint8_t> scale;
    uint32_t scaleStride = 0;
    __aicore__ inline void SetScale(AscendC::LocalTensor<uint8_t> value, uint32_t stride)
    {
        scale = value;
        scaleStride = stride;
    }
    template <class Dst, class Src>
    __aicore__ inline void operator()(Dst const& dst, Src const& src)
    {
        const uint32_t rows = tla::get<0, 0>(dst.shape()) * tla::get<0, 1>(dst.shape());
        const uint32_t cols = tla::get<1, 0>(dst.shape()) * tla::get<1, 1>(dst.shape());
        const uint32_t rowStart = tla::get<0>(src.coord());
        const uint32_t colStart = tla::get<1>(src.coord());
        AscendC::LoadData2DParamsV2 data;
        AscendC::LoadData2DMxParams mx;
        if constexpr (IS_A) {
            data.mStartPosition = rowStart / 16;
            data.kStartPosition = colStart / 32;
            data.mStep = (rows + 15) / 16;
            data.kStep = (cols + 31) / 32;
            data.srcStride = tla::get<1, 1>(src.stride()) / 512;
            data.dstStride = tla::get<1, 1>(dst.stride()) / 512;
            data.ifTranspose = false;
            mx.xStartPosition = rowStart / 16;
            mx.yStartPosition = colStart / 64;
            mx.xStep = (rows + 15) / 16;
            mx.yStep = (cols + 63) / 64;
        } else {
            if constexpr (IS_PV) {
                data.mStartPosition = rowStart / 16;
                data.kStartPosition = colStart / 32;
                // FP8 transposition consumes pairs of 16-row fractals, including causal tails.
                data.mStep = (rows + 31) / 32 * 2;
                data.kStep = (cols + 31) / 32;
                data.srcStride = tla::get<1, 1>(src.stride()) / 512;
                data.dstStride = tla::get<0, 1>(dst.stride()) / 512;
                data.ifTranspose = true;
            } else {
                data.mStartPosition = colStart / 16;
                data.kStartPosition = rowStart / 32;
                data.mStep = (cols + 15) / 16;
                data.kStep = (rows + 31) / 32;
                data.srcStride = tla::get<0, 1>(src.stride()) / 512;
                data.dstStride = tla::get<0, 1>(dst.stride()) / 512;
                data.ifTranspose = false;
            }
            mx.xStartPosition = colStart / 16;
            mx.yStartPosition = rowStart / 64;
            mx.xStep = (cols + 15) / 16;
            mx.yStep = (rows + 63) / 64;
        }
        mx.srcStride = scaleStride;
        mx.dstStride = mx.yStep;
        const auto offset = dst.layout()(dst.coord());
        AscendC::LoadData(dst.data()[offset].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(), src.data(),
                          scale.template ReinterpretCast<AscendC::fp8_e8m0_t>(), data, mx);
    }
};

template <class Base, bool IS_PV>
struct GbsaMxTileCopy : Base {
    static constexpr bool IS_MXFP8 = true;
    static constexpr bool USE_MX_SCALE = true;
    template <class TensorB>
    using CopyGmToL1B = GbsaMxCopyGmToL1B<typename Base::template CopyGmToL1B<TensorB>, IS_PV>;
    using CopyL1ToL0A = GbsaMxLoadL0<true, IS_PV>;
    using CopyL1ToL0B = GbsaMxLoadL0<false, IS_PV>;
};

// V is quantized with one scale per sequence/head/channel. The scale can be
// factored out of PV and applied once to its FP32 output. Retain the MX page
// layout and complete 64-element K groups, but use ordinary FP8 L0 operands.
template <class Base>
struct GbsaMxHybridPVTileCopy : Base {
    static constexpr bool IS_MXFP8 = true;
    static constexpr bool USE_MX_SCALE = false;
    template <class TensorB>
    using CopyGmToL1B = GbsaMxCopyGmToL1B<typename Base::template CopyGmToL1B<TensorB>, true>;
};

// Shared MX copy helper consumes a contiguous span from the packed cursor. Keep
// token strides in 64 bits for dense head-major layouts with long sequences.
// The existing page-pair copy remains a separate specialization.
template <class BaseCopy>
struct GbsaMxGeneralCopyGmToL1K : BaseCopy {
    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const& dst, TensorSrc const& src)
    {
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = tla::get<1>(src.shape());
        p.dValue = tla::get<0>(src.shape());
        p.srcNdMatrixStride = 0;
        p.srcDValue = static_cast<uint64_t>(tla::get<1>(src.stride()));
        p.dstNzC0Stride = tla::get<0, 1>(dst.stride()) / 32U;
        p.dstNzNStride = tla::get<1, 0>(dst.stride()) / 32U;
        p.dstNzMatrixStride = 0;
        const auto dstOffset = dst.layout()(dst.coord());
        const auto srcOffset = src.layout()(src.coord());
        AscendC::DataCopy(dst.data()[dstOffset], src.data()[srcOffset], p);
    }
};

template <class Base>
struct GbsaMxGeneralQKTileCopy : Base {
    static constexpr bool IS_MXFP8 = true;
    static constexpr bool USE_MX_SCALE = true;
    static constexpr bool IS_GENERAL_MX = true;
    template <class TensorB>
    using CopyGmToL1B = GbsaMxGeneralCopyGmToL1K<typename Base::template CopyGmToL1B<TensorB>>;
    using CopyL1ToL0A = GbsaMxLoadL0<true, false>;
    using CopyL1ToL0B = GbsaMxLoadL0<false, false>;
};

template <class Base>
struct GbsaMxGeneralPVTileCopy : Base {
    static constexpr bool IS_MXFP8 = true;
    static constexpr bool USE_MX_SCALE = false;
    static constexpr bool IS_GENERAL_MX = true;
};

// Dense small-block MX tiles compact valid tokens in sparse-index order.
// This cursor is also used by the independent K/V loaders, so causal tails,
// invalid entries and arbitrary index order have identical packing semantics.
struct GbsaMxPackedCursor {
    AscendC::GlobalTensor<int32_t> indices;
    uint32_t count = 0;
    uint32_t blockY = 0;
    uint32_t visibleEnd = 0;
    uint32_t raw = 0;
    uint32_t inner = 0;
    // Nonzero only after a sorted-prefix endpoint check that id[i] == id[0] + i
    // and all selected entries form full blocks followed by an optional tail.
    uint32_t contiguousBegin = 0;
    uint32_t contiguousRows = 0;
    const __gm__ uint64_t* packedRecords = nullptr;
    uint32_t packedRecordCount = 0U;
    uint32_t packedRows = 0U;
    // Nonzero for a logical 512-token tile with four independent records.
    uint32_t packedGroupWords = 0U;

    __aicore__ inline GbsaMxPackedCursor Group(uint32_t group, uint32_t rows, bool refresh = true) const
    {
        auto result = *this;
        if (packedRecords != nullptr && packedGroupWords != 0U) {
            AscendC::GlobalTensor<uint64_t> info;
            info.SetGlobalBuffer(const_cast<__gm__ uint64_t*>(packedRecords + group * packedGroupWords));
            result.packedRecords = info.GetPhyAddr();
            if (refresh) {
                result.packedRecordCount = RefreshGbsaMxPackedRecords(info);
            } else {
                // Delayed PV owns the same descriptor generation as QK.
                // The producer cannot reuse this slot until PV releases it,
                // so the cache lines refreshed by QK remain valid here.
                result.packedRecordCount =
                    static_cast<uint32_t>(info.GetValue(0U)) | (info.GetValue(1U) == 1U ? 0x80000000U : 0U);
            }
            result.packedRows = rows;
            result.packedGroupWords = 0U;
        } else {
            result.Advance(group * 128U);
        }
        return result;
    }

    __aicore__ inline bool NextSpan(uint32_t limit, uint32_t& token, uint32_t& rows)
    {
        rows = 0;
        if (contiguousRows != 0U) {
            uint32_t position = raw * blockY + inner;
            if (position >= contiguousRows) {
                return false;
            }
            token = contiguousBegin + position;
            rows = min(limit, contiguousRows - position);
            position += rows;
            raw = position / blockY;
            inner = position % blockY;
            return rows != 0U;
        }
        while (raw < count && rows < limit) {
            const int32_t id = indices.GetValue(raw);
            if (id < 0 || static_cast<uint64_t>(id) * blockY >= visibleEnd) {
                ++raw;
                inner = 0;
                continue;
            }
            const uint32_t begin = static_cast<uint32_t>(id) * blockY;
            const uint32_t valid = min(blockY, visibleEnd - begin);
            if (inner >= valid) {
                ++raw;
                inner = 0;
                continue;
            }
            if (rows != 0U && begin + inner != token + rows) {
                break;
            }
            if (rows == 0U) {
                token = begin + inner;
            }
            const uint32_t take = min(limit - rows, valid - inner);
            rows += take;
            inner += take;
            if (inner == valid) {
                ++raw;
                inner = 0;
            }
        }
        return rows != 0U;
    }

    __aicore__ inline uint32_t Advance(uint32_t requested)
    {
        uint32_t token, rows, done = 0;
        while (done < requested && NextSpan(requested - done, token, rows)) {
            done += rows;
        }
        return done;
    }
};

template <class Base>
struct GbsaMxPackedQKTileCopy : GbsaMxGeneralQKTileCopy<Base> {
    static constexpr bool IS_PACKED_MX = true;
};

template <class Base>
struct GbsaMxPackedPVTileCopy : GbsaMxGeneralPVTileCopy<Base> {
    static constexpr bool IS_PACKED_MX = true;
};

struct GbsaMxMmad {
    template <class C, class A, class B>
    __aicore__ inline void operator()(C const& c, A const& a, B const& b, uint32_t m, uint32_t n, uint32_t k,
                                      bool initC = true)
    {
        AscendC::MmadParams p;
        p.m = m;
        p.n = n;
        p.k = k;
        p.cmatrixInitVal = initC;
        p.disableGemv = true;
        AscendC::Mmad(c.data(), a.data().template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                      b.data().template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(), p);
        if ((m / 16) * (n / 16) < 10) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};
} // namespace NpuArch::Gemm::Tile
#endif
