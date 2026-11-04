/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file vf_pair_permute_kl_loss_grad.h
 * \brief
 */
#ifndef VF_PAIR_PERMUTE_KL_LOSS_GRAD_H
#define VF_PAIR_PERMUTE_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
using namespace Reg;

template <bool EncodePairDelta = false, bool EncodePairBytes = false>
__simd_callee__ inline void InitSligPairPermutation32(RegTensor<uint32_t>& permutation, RegTensor<uint32_t>& identity,
                                                      RegTensor<uint32_t>& neighbor, __ubuf__ int32_t* indices,
                                                      MaskReg& allB32)
{
    RegTensor<int32_t> index, even, odd;
    MaskReg descending, descendingPairs, unused;
    LoadAlign(index, indices);
    DeInterleave(even, odd, index, index);
    Compare<int32_t, CMPMODE::GT>(descending, even, odd, allB32);
    Interleave<int32_t>(descendingPairs, unused, descending, descending);
    Select(permutation, neighbor, identity, descendingPairs);
    if constexpr (EncodePairDelta) {
        Sub<int32_t>(index, odd, even, allB32);
        Abs<int32_t>(index, index, allB32);
        Min<int32_t>(even, even, odd, allB32);
        Interleave<int32_t>(even, odd, even, index);
        if constexpr (EncodePairBytes) {
            auto& bytePairs = reinterpret_cast<RegTensor<uint32_t>&>(even);
            ShiftLefts<uint32_t, int16_t>(bytePairs, bytePairs, int16_t(9), allB32);
        }
        StoreAlign(indices, even, allB32);
    }
}

template <bool EncodePairDelta = false, bool EncodePairBytes = false>
__simd_vf__ inline void PairPermuteFloatVF(__ubuf__ float* dstUb, __ubuf__ float* srcUb, __ubuf__ int32_t* indices,
                                           uint32_t rows, uint32_t sortedRows)
{
    RegTensor<uint32_t> identity, neighbor, one, permutation;
    RegTensor<float> payload, permuted;
    MaskReg allB32 = CreateMask<float, MaskPattern::ALL>();
    Arange<int32_t>(reinterpret_cast<RegTensor<int32_t>&>(identity), int32_t(0));
    Duplicate(one, uint32_t(1));
    Xor(neighbor, identity, one, allB32);
    const uint32_t sortedChunks = sortedRows >> 6;
    for (uint16_t chunk = 0; chunk < sortedChunks; ++chunk) {
        InitSligPairPermutation32<EncodePairDelta, EncodePairBytes>(permutation, identity, neighbor,
                                                                    indices + chunk * 64, allB32);
        LoadAlign(payload, srcUb + chunk * 64);
        Gather<float, uint32_t>(permuted, payload, permutation);
        StoreAlign(dstUb + chunk * 64, permuted, allB32);
    }
    uint32_t remaining = rows - sortedRows;
    const uint32_t identityChunks = (remaining + 63) >> 6;
    for (uint16_t chunk = 0; chunk < identityChunks; ++chunk) {
        MaskReg valid = UpdateMask<float>(remaining);
        LoadAlign(payload, srcUb + sortedRows + chunk * 64);
        StoreAlign(dstUb + sortedRows + chunk * 64, payload, valid);
    }
}

template <typename T, bool EncodePairDelta = false, bool EncodePairBytes = false>
__aicore__ inline void PairPermuteFloatVf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                          const LocalTensor<int32_t>& indices, uint32_t rows, uint32_t sortedRows)
{
    PairPermuteFloatVF<EncodePairDelta, EncodePairBytes>(reinterpret_cast<__ubuf__ float*>(dstTensor.GetPhyAddr()),
                                                         reinterpret_cast<__ubuf__ float*>(srcTensor.GetPhyAddr()),
                                                         reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()),
                                                         rows, sortedRows);
}
} // namespace AscendC
#endif
