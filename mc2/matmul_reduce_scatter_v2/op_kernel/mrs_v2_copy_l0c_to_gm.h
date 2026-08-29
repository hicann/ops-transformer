/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mrs_v2_copy_l0c_to_gm.h
 * \brief
 */

#ifndef MRS_V2_COPY_L0C_TO_GM_H
#define MRS_V2_COPY_L0C_TO_GM_H

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tla_gemm_copy_l0c_to_gm.hpp"

namespace Catlass::Gemm::Tile {
constexpr uint32_t PARAM_INDEX_0 = 0;
constexpr uint32_t PARAM_INDEX_1 = 1;
constexpr uint32_t PARAM_INDEX_2 = 2;
constexpr uint32_t PARAM_INDEX_3 = 3;
// Fixpipe with quant VDEQF16
template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<Catlass::Arch::AtlasA2, ElementAccumulator_, Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::PER_CHANNEL, ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre =
        CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
                    AscendC::LocalTensor<uint64_t> cbufWorkspace, LayoutDst const &dstLayout,
                    LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(PARAM_INDEX_1);
        intriParams.mSize = dstLayout.shape(PARAM_INDEX_0);
        intriParams.srcStride = srcLayout.stride(PARAM_INDEX_3) / srcLayout.stride(PARAM_INDEX_0);
        intriParams.dstStride = dstLayout.stride(PARAM_INDEX_0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, cbufWorkspace, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // COPY_L0C_TO_GM_H
