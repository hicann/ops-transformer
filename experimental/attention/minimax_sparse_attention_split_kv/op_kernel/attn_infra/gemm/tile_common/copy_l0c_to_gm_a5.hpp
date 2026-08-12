/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L0C_TO_GM_A5_HPP
#define GEMM_TILE_COPY_L0C_TO_GM_A5_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/arch.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l0c_to_dst.hpp"
#include "../../../tla/tensor.hpp"

#if (__CCE_AICORE__ == 310)
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_GM = {AscendC::CO2Layout::ROW_MAJOR, false};
#endif

namespace NpuArch::Gemm::Tile {

#if (__CCE_AICORE__ == 310)
template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    NpuArch::Arch::AtlasA5,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = NpuArch::Arch::AtlasA5;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(
        TensorDst const &dstTensor, TensorSrc const &srcTensor,
        uint16_t ndNum = 1, uint16_t srcNdStride = 1,
        uint32_t dstNdStride = 1, uint8_t unitFlag = 0)
    {
        // TODO: check why use fixpipe is not OK
        // L0C->GM via AscendC::DataCopy (NZ2ND) + SetFixpipeNz2ndFlag, matching matmul_act
        // block_mmad_pingpong CopyOut. The plain AscendC::Fixpipe<CFG_ROW_MAJOR_GM> path left
        // the L0C->GM DMA uncommitted for cross-core (AIC->AIV) visibility after SyncAll, so
        // the VEC-side post-SyncAll read saw 0 for the last-written slots. The DataCopy +
        // SetFixpipeNz2ndFlag idiom is the reliable CANN L0C->GM path. quantPre/reluEn/unitFlag
        // keep the current implementation's values (only the issuing primitive changes).
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.reluPre = reluEn ? 1 : 0;
        intriParams.unitFlag = unitFlag;
        intriParams.nz2ndEn = true;
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);

        // AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // intriParams.nSize = tla::get<1>(dstTensor.shape());
        // intriParams.mSize = tla::get<0>(dstTensor.shape());
        // intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // intriParams.quantPre = quantPre;
        // intriParams.reluEn = reluEn;
        // intriParams.unitFlag = unitFlag;

        // auto dstOffset = dstTensor.layout()(dstTensor.coord());
        // auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_GM>(
        //     dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};
#endif

}  // namespace NpuArch::Gemm::Tile

#endif  // GEMM_TILE_COPY_L0C_TO_GM_A5_HPP
