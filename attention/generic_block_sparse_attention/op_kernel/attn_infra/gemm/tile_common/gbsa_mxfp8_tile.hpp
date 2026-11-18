/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GBSA_MXFP8_TILE_HPP
#define GBSA_MXFP8_TILE_HPP
#include "gbsa_mx_packed_records.hpp"
#include "gbsa_gemm_tile_copy.hpp"

namespace NpuArch::Gemm::Tile {
template <bool IsPV>
struct CopyL0CToUbMxfp8Full {
    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
    template <class TensorUB>
    __aicore__ inline void operator()(TensorUB const& dst, const AscendC::LocalTensor<float>& src, uint32_t columns,
                                      uint32_t queryRows) const
    {
        const auto offset = dst.layout()(dst.coord());
        const uint32_t rows0 = (queryRows + 1) / 2;
        const uint32_t rowOffset = IsPV ? (rows0 + 15) / 16 * 16 : rows0;
        const uint32_t alignedRows = IsPV ? 2U * rowOffset : (queryRows + 15U) / 16U * 16U;
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.mSize = IsPV ? alignedRows : 2U * rows0;
        params.nSize = columns;
        params.srcStride = alignedRows;
        params.dstStride = tla::get<0>(dst.stride());
        params.quantPre = QuantMode_t::NoQuant;
        params.dualDstCtl = 1;
        params.params.ndNum = 1;
        params.params.srcNdStride = 0;
        params.params.dstNdStride = 0;
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(dst.data()[offset], src, params);
    }
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

struct CopyL1ToL0Mxfp8FullQK {
    template <class T>
    __aicore__ void operator()(const T& dst, const AscendC::LocalTensor<uint8_t>& src,
                               const AscendC::LocalTensor<uint8_t>& scale, uint32_t rows, uint32_t reduction,
                               uint32_t sourceRows, uint32_t rowStart = 0) const
    {
        AscendC::LoadData2DParamsV2 data;
        data.mStartPosition = rowStart / 16;
        data.kStartPosition = 0;
        data.mStep = rows / 16;
        data.kStep = reduction / 32;
        data.srcStride = sourceRows / 16;
        data.dstStride = rows / 16;
        data.ifTranspose = false;
        AscendC::LoadData2DMxParams mx;
        mx.xStartPosition = rowStart / 16;
        mx.yStartPosition = 0;
        mx.xStep = rows / 16;
        mx.yStep = reduction / 64;
        mx.srcStride = reduction / 64;
        mx.dstStride = reduction / 64;
        AscendC::LoadData(dst.template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(), src.ReinterpretCast<fp8_e4m3fn_t>(),
                          scale.ReinterpretCast<AscendC::fp8_e8m0_t>(), data, mx);
    }
};
struct CopyL1ToL0Mxfp8FullP {
    template <class T>
    __aicore__ void operator()(const T& dst, const AscendC::LocalTensor<uint8_t>& src, uint32_t rows,
                               uint32_t reduction, uint32_t sourceRows, uint32_t rowStart, uint32_t,
                               uint32_t reductionStart) const
    {
        AscendC::LoadData2DParamsV2 data;
        data.mStartPosition = rowStart / 16;
        data.kStartPosition = reductionStart / 32;
        data.mStep = rows / 16;
        data.kStep = reduction / 32;
        data.srcStride = sourceRows / 16;
        data.dstStride = rows / 16;
        data.ifTranspose = false;
        AscendC::LoadData(dst, src.ReinterpretCast<fp8_e4m3fn_t>(), data);
    }
};
struct CopyL1ToL0Mxfp8FullV {
    template <class T>
    __aicore__ void operator()(const T& dst, const AscendC::LocalTensor<uint8_t>& src, uint32_t channels, uint32_t rows,
                               uint32_t destinationRows, uint32_t sourceRows, uint32_t rowStart,
                               uint32_t channelStart) const
    {
        AscendC::LoadData2DParamsV2 data;
        data.mStartPosition = rowStart / 16;
        data.kStartPosition = channelStart / 32;
        data.mStep = rows / 16;
        data.kStep = channels / 32;
        data.srcStride = sourceRows / 16;
        data.dstStride = destinationRows / 16;
        data.ifTranspose = true;
        AscendC::LoadData(dst, src.ReinterpretCast<fp8_e4m3fn_t>(), data);
    }
};
template <bool Paged, bool IsPV>
struct PackedTileCopyMxfp8Full {
    using LayoutTagL1A = layout::zN;
    using LayoutTagL1B = layout::zN;
    using LayoutTagL0A = layout::zZ;
    using LayoutTagL0B = layout::nZ;
    using CopyGmToL1B = CopySparsePackedGmToL1Mxfp8Full<Paged>;
    using CopyGmToL1Scale = CopyQueryScaleGmToL1Mxfp8Full;
    using CopyL1ToL0A = std::conditional_t<IsPV, CopyL1ToL0Mxfp8FullP, CopyL1ToL0Mxfp8FullQK>;
    using CopyL1ToL0B = std::conditional_t<IsPV, CopyL1ToL0Mxfp8FullV, CopyL1ToL0Mxfp8FullQK>;
    using CopyL0CToUB = CopyL0CToUbMxfp8Full<IsPV>;
};
} // namespace NpuArch::Gemm::Tile
#endif
