/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_MATMUL_AIV_PADDING_COMMON_H
#define MC2_MATMUL_AIV_PADDING_COMMON_H

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_padding_matmul.hpp"

namespace Mc2MatmulAivPadding {

template <typename Layout>
struct MatrixPaddingParams {
    GM_ADDR src;
    Layout srcLayout;
    GM_ADDR dst;
    Layout dstLayout;
    bool required;

    CATLASS_HOST_DEVICE
    MatrixPaddingParams() {}

    CATLASS_HOST_DEVICE
    MatrixPaddingParams(GM_ADDR src_, Layout srcLayout_, GM_ADDR dst_, Layout dstLayout_, bool required_)
        : src(src_),
          srcLayout(srcLayout_),
          dst(dst_),
          dstLayout(dstLayout_),
          required(required_)
    {}
};

template <typename LayoutA, typename LayoutB>
struct MatmulPaddingParams {
    MatrixPaddingParams<LayoutA> matrixA;
    MatrixPaddingParams<LayoutB> matrixB;

    CATLASS_HOST_DEVICE
    MatmulPaddingParams() {}

    CATLASS_HOST_DEVICE
    MatmulPaddingParams(GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrWA_,
                        LayoutA layoutWA_, GM_ADDR ptrWB_, LayoutB layoutWB_, bool alignA_, bool alignB_)
        : matrixA(ptrA_, layoutA_, ptrWA_, layoutWA_, alignA_),
          matrixB(ptrB_, layoutB_, ptrWB_, layoutWB_, alignB_)
    {}
};

template <class ArchTag, class Element, class Layout>
CATLASS_DEVICE void PadMatrix(Catlass::Arch::Resource<ArchTag> &resource, const MatrixPaddingParams<Layout> &params)
{
    if (!params.required) {
        return;
    }

    constexpr uint32_t computeLength = 96 * 1024 / sizeof(Element);
    using Padding = Catlass::Gemm::Kernel::PaddingMatrix<ArchTag, Element, Layout, computeLength>;
    AscendC::GlobalTensor<Element> src;
    AscendC::GlobalTensor<Element> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(params.src));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(params.dst));
    Padding padding(resource);
    padding(dst, src, params.dstLayout, params.srcLayout);
}

template <class ArchTag, class AType, class BType>
CATLASS_DEVICE void PadMatmulInputs(Catlass::Arch::Resource<ArchTag> &resource,
                                    const MatmulPaddingParams<typename AType::Layout, typename BType::Layout> &params)
{
    PadMatrix<ArchTag, typename AType::Element>(resource, params.matrixA);
    PadMatrix<ArchTag, typename BType::Element>(resource, params.matrixB);
}

CATLASS_DEVICE void NotifyMatmulPaddingFinished(Catlass::Arch::CrossCoreFlag &finishFlag)
{
    Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(finishFlag);
}

} // namespace Mc2MatmulAivPadding

#endif // MC2_MATMUL_AIV_PADDING_COMMON_H
