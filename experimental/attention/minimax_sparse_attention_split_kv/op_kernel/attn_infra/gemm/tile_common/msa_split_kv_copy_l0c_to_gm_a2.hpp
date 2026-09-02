/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_MSA_SPLIT_KV_COPY_L0C_TO_GM_A2_HPP
#define GEMM_TILE_MSA_SPLIT_KV_COPY_L0C_TO_GM_A2_HPP

#include "../../../attn_infra/msa_split_kv_base_defs.hpp"
#include "../../../attn_infra/arch/msa_split_kv_arch.hpp"
#include "../../../attn_infra/gemm/tile_common/msa_split_kv_copy_l0c_to_dst.hpp"
#include "../../../attn_infra/gemm/msa_split_kv_gemm_type.hpp"
#include "../../../tla/msa_split_kv_tla_tensor.hpp"
namespace NpuArch::Gemm::Tile {

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<NpuArch::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT, ReluEnable_> {
    using ArchTag = NpuArch::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = NpuArch::layout::zN;
    using LayoutDst = NpuArch::layout::RowMajor;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
                                      AscendC::LocalTensor<ElementSrc> const &src, LayoutDst const &dstLayout,
                                      LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<NpuArch::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::zN>,
                   ScaleGranularity::NO_QUANT, ReluEnable_> {
    using ArchTag = NpuArch::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = NpuArch::layout::zN;
    using LayoutDst = NpuArch::layout::zN;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementDst> const &dst,
                                      AscendC::LocalTensor<ElementSrc> const &src, LayoutDst const &dstLayout,
                                      LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<NpuArch::Arch::AtlasA2, TensorSrc_,
                      tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
                      ScaleGranularity::NO_QUANT, ReluEnable_,
                      std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint16_t ndNum = 1,
                                      uint16_t srcNdStride = 1, uint32_t dstNdStride = 1, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor");

        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        // Fixpipe source stride is the distance between adjacent N fractals,
        // expressed in C0 units.  For an L0C zN tile this is derived from
        // the layout, not from the logical M extent.  Using ceil(M / 16)
        // repeats the first 16 score columns across every N fractal.
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / NpuArch::C0_NUM_PER_FRACTAL;
        // NZ2ND interprets dstStride in destination elements.  A tile keeps
        // the parent destination layout, so use that physical row stride
        // instead of the 128-column tile width (D can be 256).
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // A2 Fixpipe does not support ND multi-batch natively; loop over ND copies
        for (uint16_t nd = 0; nd < ndNum; ++nd) {
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
                dstTensor.data()[dstOffset + static_cast<uint64_t>(nd) * dstNdStride],
                srcTensor.data()[srcOffset + static_cast<uint64_t>(nd) * srcNdStride], intriParams);
        }
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<NpuArch::Arch::AtlasA2, TensorSrc_,
                      tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
                      ScaleGranularity::NO_QUANT, ReluEnable_,
                      std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint16_t ndNum = 1,
                                      uint16_t srcNdStride = 1, uint32_t dstNdStride = 1, uint8_t unitFlag = 0)
    {
        static_assert(tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                          TensorSrc::position == AscendC::TPosition::CO1 &&
                          TensorDst::position == AscendC::TPosition::GM,
                      "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and zN");

        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<1, 0>(srcTensor.shape());
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / (BYTE_PER_C0 / sizeof(ElementDst));

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.isChannelSplit = true;
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // A2 Fixpipe does not support ND multi-batch natively; loop over ND copies
        for (uint16_t nd = 0; nd < ndNum; ++nd) {
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(
                dstTensor.data()[dstOffset + static_cast<uint64_t>(nd) * dstNdStride],
                srcTensor.data()[srcOffset + static_cast<uint64_t>(nd) * srcNdStride], intriParams);
        }
    }
};

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_MSA_SPLIT_KV_COPY_L0C_TO_GM_A2_HPP
