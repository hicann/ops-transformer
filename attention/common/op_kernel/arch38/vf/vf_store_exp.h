/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_store_exp.h
 * \brief StoreExp: stores the softmax exp result back to UB, shared by ND and DN layouts.
 */

#ifndef VF_STORE_EXP_H
#define VF_STORE_EXP_H

#include "kernel_tensor.h"

namespace FaVectorApi {
// Store one exp vector to UB, shared by ND and DN layouts (the layout difference is captured by the
// caller-provided blockStride/repeatStride). The source exp register is scaled by `scale` into a
// scratch register before store, so the caller's exp register stays unscaled (used later to compute
// expSum). For fp16→fp16 non-quant the scale is 2^-8 (range protection against the BMM2 numerator
// overflowing half), compensated by FP16_NOQUANT_RANGE_SCALE_BACK in vf_flashupdate_new_regbase_v2.h;
// for other non-int8 paths (e.g. fp16→int8) the scale is 1.0.
// The exp compute dtype (T) and the store dtype (T2) are both always half on this platform, so no cast
// is needed. The Muls (compute) uses a full per-lane predicate mask, while the DataCopy (block store)
// uses the caller-provided `preg` (an UpdateMask length/repeat mask in the DN layout) -- the two kinds
// must not be shared. src keeps a non-const reference because the underlying Reg::Muls/Reg::DataCopy
// interfaces take their source register as a non-const reference; It is read only.
template <typename T, typename T2>
__simd_callee__ inline void StoreExp(__ubuf__ T2*& dst, Reg::RegTensor<T>& src, uint16_t blockStride,
                                     uint16_t repeatStride, Reg::MaskReg& preg, float scale)
{
    static_assert(IsSameType<T2, T>::value, "StoreExp requires the store dtype to equal the compute dtype (both half)");
    Reg::RegTensor<T> scaledSrc;
    Reg::MaskReg pregFull = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(scaledSrc, src, static_cast<T>(scale), pregFull);
    Reg::DataCopy<T2, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        dst, scaledSrc, blockStride, repeatStride, preg);
}
} // namespace FaVectorApi

#endif // VF_STORE_EXP_H
