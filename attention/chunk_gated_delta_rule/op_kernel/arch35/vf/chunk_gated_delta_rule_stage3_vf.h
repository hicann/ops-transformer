/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE3_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE3_VF_H

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace MicroAPI;

/*
 * gated 路径一次完成 masked_qkt：
 *   decay[i, j] = exp((g[i] - g[j]) * mask[i, j]);
 *   dst[i, j] = BF16(FP32(qkt[i, j]) * scale * decay[i, j] * mask[i, j])。
 * g 向量只加载一次，每行的 decay、Cast 和逐元素计算都保留在寄存器中，不写中间 UB。
 */
__simd_vf__ inline void ComputeMaskedQktGatedVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *qktAddr,
                                                __ubuf__ float *gAddr, __ubuf__ float *maskAddr, float scale,
                                                uint16_t validSize, uint16_t rowStride)
{
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};

    RegTensor<float> gColReg;
    RegTensor<float> gRowReg;
    RegTensor<float> decayReg;
    RegTensor<float> maskReg;
    RegTensor<float> qktFp32Reg;
    RegTensor<bfloat16_t> qktBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    uint32_t maskCount = validSize;
    MaskReg validMask = UpdateMask<float>(maskCount);

    LoadAlign(gColReg, gAddr);
    for (uint16_t row = 0; row < validSize; ++row) {
        uint32_t offset = static_cast<uint32_t>(row) * rowStride;
        DataCopy<float, LoadDist::DIST_BRC_B32>(gRowReg, gAddr + row);
        Sub(decayReg, gRowReg, gColReg, validMask);
        LoadAlign(maskReg, maskAddr + offset);
        Mul(decayReg, decayReg, maskReg, validMask);
        Exp<float, MaskMergeMode::ZEROING>(decayReg, decayReg, validMask);

        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qktBf16Reg, qktAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(qktFp32Reg, qktBf16Reg, validMask);
        Muls(qktFp32Reg, qktFp32Reg, scale, validMask);
        Mul(qktFp32Reg, qktFp32Reg, decayReg, validMask);
        Mul(qktFp32Reg, qktFp32Reg, maskReg, validMask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, qktFp32Reg, validMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, validMask);
    }
}

/*
 * 无门控路径一次完成：
 *   dst[i, j] = BF16(FP32(qkt[i, j]) * scale * 1.0 * mask[i, j])。
 * 保留原来的乘 1.0 计算顺序，但所有中间结果均位于寄存器。
 */
__simd_vf__ inline void ComputeMaskedQktNoGateVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *qktAddr,
                                                 __ubuf__ float *maskAddr, float scale, uint16_t validSize,
                                                 uint16_t rowStride)
{
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};

    RegTensor<float> maskReg;
    RegTensor<float> oneReg;
    RegTensor<float> qktFp32Reg;
    RegTensor<bfloat16_t> qktBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    uint32_t maskCount = validSize;
    MaskReg validMask = UpdateMask<float>(maskCount);

    Duplicate(oneReg, static_cast<float>(1.0f));
    for (uint16_t row = 0; row < validSize; ++row) {
        uint32_t offset = static_cast<uint32_t>(row) * rowStride;
        LoadAlign(maskReg, maskAddr + offset);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qktBf16Reg, qktAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(qktFp32Reg, qktBf16Reg, validMask);
        Muls(qktFp32Reg, qktFp32Reg, scale, validMask);
        Mul(qktFp32Reg, qktFp32Reg, oneReg, validMask);
        Mul(qktFp32Reg, qktFp32Reg, maskReg, validMask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, qktFp32Reg, validMask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, validMask);
    }
}

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_STAGE3_VF_H
