/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATTENTION_UPDATE_OUTPUT_REGBASE_H_
#define ATTENTION_UPDATE_OUTPUT_REGBASE_H_

#include "attention_update_base_regbase.h"

namespace AttentionUpdateOpt {
constexpr uint16_t CACHED_WEIGHT_MAX_SP = 16;
struct CachedOutputTile {
    uint32_t lseOffset;
    uint32_t lseStride;
    uint32_t headDim;
    uint32_t dAlign;
    uint32_t shardStride;
    uint16_t rows;
    uint16_t sp;
};
constexpr uint16_t CACHED_WEIGHT_MIN_ROWS = 2;

template <typename GoType, uint16_t Sp>
__aicore__ inline void ComputeOutputWithCachedWeights(__ubuf__ GoType* output, __ubuf__ float* weights,
                                                      __ubuf__ GoType* input, const CachedOutputTile& tile)
{
    constexpr uint32_t vl = VREG_SIZE / sizeof(float);
    const uint32_t headDim = tile.headDim;
    const uint16_t chunks = (headDim + vl - 1) / vl;
    const uint32_t lseOffset = tile.lseOffset;
    const uint32_t lseStride = tile.lseStride;
    const uint32_t dAlign = tile.dAlign;
    const uint32_t shardStride = tile.shardStride;
    const uint16_t rows = tile.rows;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> cachedWeights[Sp];
        AscendC::Reg::RegTensor<float> inputs[UNROLL_NUM];
        AscendC::Reg::RegTensor<float> sum;
        AscendC::Reg::MaskReg mask;
        for (uint16_t row = 0; row < rows; ++row) {
#pragma unroll
            for (uint16_t sp = 0; sp < Sp; ++sp) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                    cachedWeights[sp], weights + sp * lseStride + lseOffset + row);
            }
            uint32_t remaining = headDim;
            for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
                mask = AscendC::Reg::UpdateMask<float, Reg::RegTraitNumOne>(remaining);
                AscendC::Reg::Duplicate(sum, 0, mask);
#pragma unroll
                for (uint16_t sp = 0; sp < Sp; ++sp) {
                    ops::LoadOneTensorForDtypeT<GoType>(input, inputs[sp % UNROLL_NUM], mask,
                                                        sp * shardStride + row * dAlign + chunk * vl);
                    AscendC::Reg::MulAddDst(sum, cachedWeights[sp], inputs[sp % UNROLL_NUM], mask);
                }
                ops::StoreOneTensorForDtypeT<GoType>(output, sum, mask, row * dAlign + chunk * vl);
            }
        }
    }
}

__aicore__ inline bool ShouldCacheOutputWeights(uint16_t rows, uint32_t headDim, uint16_t sp)
{
    return rows >= CACHED_WEIGHT_MIN_ROWS && headDim > VREG_SIZE / sizeof(float) && sp >= 1 &&
           sp <= CACHED_WEIGHT_MAX_SP;
}

template <typename GoType, uint16_t Begin = 1, uint16_t End = CACHED_WEIGHT_MAX_SP>
__aicore__ inline void DispatchCachedOutput(__ubuf__ GoType* output, __ubuf__ float* weights, __ubuf__ GoType* input,
                                            const CachedOutputTile& tile)
{
    if constexpr (Begin == End) {
        ComputeOutputWithCachedWeights<GoType, Begin>(output, weights, input, tile);
    } else {
        constexpr uint16_t middle = (Begin + End) / 2;
        if (tile.sp <= middle) {
            DispatchCachedOutput<GoType, Begin, middle>(output, weights, input, tile);
        } else {
            DispatchCachedOutput<GoType, middle + 1, End>(output, weights, input, tile);
        }
    }
}
} // namespace AttentionUpdateOpt
#endif // ATTENTION_UPDATE_OUTPUT_REGBASE_H_
