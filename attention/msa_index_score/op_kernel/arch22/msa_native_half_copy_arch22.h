/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MSA_NATIVE_HALF_COPY_ARCH22_H
#define MSA_NATIVE_HALF_COPY_ARCH22_H

#include "kernel_operator.h"
#include "../catlass/layout/msa_layout.hpp"

namespace MsaIndexScoreNs {

class MsaNativeHalfCopy {
public:
    struct Params {};
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;

    __aicore__ inline MsaNativeHalfCopy() = default;
    __aicore__ inline explicit MsaNativeHalfCopy(const Params&) {}

    __aicore__ inline void operator()(const AscendC::GlobalTensor<half>& dst, const AscendC::LocalTensor<int32_t>& src,
                                      const LayoutDst& dstLayout, const LayoutSrc& srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 params;
        params.mSize = dstLayout.shape(0);
        params.nSize = dstLayout.shape(1);
        params.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        params.dstStride = dstLayout.stride(0);
        params.reluEn = false;
        params.unitFlag = unitFlag;
        params.quantPre = QuantMode_t::DEQF16;
        params.deqScalar = 0x3a800000ULL; // FP32 encoding of 2^-10.
        AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>(dst, src, params);
    }
};

} // namespace MsaIndexScoreNs
#endif
