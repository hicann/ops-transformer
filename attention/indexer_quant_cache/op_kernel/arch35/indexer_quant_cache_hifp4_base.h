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
 * \file indexer_quant_cache_hifp4_base.h
 */

#ifndef INDEXER_QUANT_CACHE_HIFP4_BASE_H
#define INDEXER_QUANT_CACHE_HIFP4_BASE_H

#include "kernel_operator.h"
#include "indexer_quant_cache_base.h"

namespace IndexerQuantCache {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::MicroAPI::MaskDist;
using AscendC::MicroAPI::PostLiteral;
using AscendC::MicroAPI::StoreDist;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t CONST_TWO = 2;
constexpr int32_t ALIGN_PARAM1 = 64;
constexpr int32_t ALIGN_PARAM2 = 128;
constexpr int32_t ALIGN_PARAM3 = 512;
constexpr AscendC::MicroAPI::CastTrait castTraitReciprocal = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename TX>
__aicore__ inline void VFProcessSwigluHiF4InvScaleAndQuant(
    const LocalTensor<int32_t>& yQuantLocal, const LocalTensor<float>& xLocal, const LocalTensor<uint8_t>& eaLocal,
    const LocalTensor<uint32_t>& ebLocal, const LocalTensor<uint8_t>& ecLocal, const uint32_t curColNum)
{
    __local_mem__ int32_t* yQuantLocalAddr = (__local_mem__ int32_t*)yQuantLocal.GetPhyAddr();
    __local_mem__ float* xLocalAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
    __local_mem__ uint8_t* eaLocalAddr = (__local_mem__ uint8_t*)eaLocal.GetPhyAddr();
    __local_mem__ uint32_t* ebLocalAddr = (__local_mem__ uint32_t*)ebLocal.GetPhyAddr();
    __local_mem__ uint8_t* ecLocalAddr = (__local_mem__ uint8_t*)ecLocal.GetPhyAddr();
    const uint16_t VL_LEN = 256;
    uint16_t vlForB16 = VL_LEN / sizeof(bfloat16_t);
    uint16_t vlForB32 = VL_LEN / sizeof(float);
    uint16_t loopCount = CeilDiv(curColNum, vlForB16 * 4); // 一次搬入4个VL的输入，也就是512个b16数据
    __VEC_SCOPE__
    {
        RegTensor<half> xHalf1;
        RegTensor<half> xHalf2;
        RegTensor<half> xHalf3;
        RegTensor<half> xHalf4;
        RegTensor<bfloat16_t> x1;
        RegTensor<bfloat16_t> x2;
        RegTensor<bfloat16_t> x3;
        RegTensor<bfloat16_t> x4;

        RegTensor<bfloat16_t> x10;
        RegTensor<bfloat16_t> x20;
        RegTensor<bfloat16_t> x30;
        RegTensor<bfloat16_t> x40;

        RegTensor<bfloat16_t> max1;
        RegTensor<bfloat16_t> max2;
        RegTensor<bfloat16_t> maxPer4;
        RegTensor<bfloat16_t> maxPer4Tmp1;
        RegTensor<bfloat16_t> maxPer4Tmp2;
        RegTensor<bfloat16_t> vmaxPer4;
        RegTensor<bfloat16_t> maxPer4Coeff;
        RegTensor<bfloat16_t> maxPer8;
        RegTensor<bfloat16_t> maxPer8Tmp;
        RegTensor<bfloat16_t> maxPer8Tmp1;
        RegTensor<bfloat16_t> vmaxPer8;
        RegTensor<bfloat16_t> maxPer8Coeff;
        RegTensor<bfloat16_t> vb16EaRecBf16;
        RegTensor<bfloat16_t> tmp;
        RegTensor<bfloat16_t> numOne;
        RegTensor<bfloat16_t> numHalf;
        RegTensor<bfloat16_t> invScale0;
        RegTensor<bfloat16_t> invScale1;
        RegTensor<bfloat16_t> invScale2;
        RegTensor<bfloat16_t> invScale3;
        RegTensor<bfloat16_t> invScale4;
        RegTensor<fp4x2_e1m2_t> x1Out;
        RegTensor<fp4x2_e1m2_t> x2Out;
        RegTensor<fp4x2_e1m2_t> x3Out;
        RegTensor<fp4x2_e1m2_t> x4Out;
        RegTensor<bfloat16_t> maxPer64;
        RegTensor<uint16_t> expMaskBF16;
        RegTensor<uint16_t> numScale;
        RegTensor<uint16_t> e6m2;
        RegTensor<uint8_t> e6m2Res;
        RegTensor<uint8_t> tmp1;
        UnalignReg uregEa;
        UnalignReg uregEb;
        Duplicate(expMaskBF16, 0x7fff);
        Duplicate(numScale, 0x3e12);
        Duplicate(numOne, 1.0);
        float numberhalf = 0.5;
        Duplicate(numHalf, static_cast<bfloat16_t>(numberhalf));
        MaskReg ebMask;
        MaskReg ecMask;
        MaskReg maskCast = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg maskAll = CreateMask<uint8_t, MaskPattern::ALL>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitE2M1 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < loopCount; i++) {
            if constexpr (IsSameType<TX, half>::value) {
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (RegTensor<float>&)xHalf1, (RegTensor<float>&)xHalf2, xLocalAddr, vlForB32 * 2);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (RegTensor<float>&)xHalf3, (RegTensor<float>&)xHalf4, xLocalAddr, vlForB32 * 2);
                Cast<bfloat16_t, TX, castTraitHalf2Bf16>(x1, xHalf1, maskCast);
                Cast<bfloat16_t, TX, castTraitHalf2Bf16>(x2, xHalf2, maskCast);
                Cast<bfloat16_t, TX, castTraitHalf2Bf16>(x3, xHalf3, maskCast);
                Cast<bfloat16_t, TX, castTraitHalf2Bf16>(x4, xHalf4, maskCast);
            } else {
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (RegTensor<float>&)x1, (RegTensor<float>&)x2, xLocalAddr, vlForB32 * 2);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                    (RegTensor<float>&)x3, (RegTensor<float>&)x4, xLocalAddr, vlForB32 * 2);
            }
            And((RegTensor<uint16_t>&)x10, (RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            And((RegTensor<uint16_t>&)x20, (RegTensor<uint16_t>&)x2, expMaskBF16, maskAll);
            And((RegTensor<uint16_t>&)x30, (RegTensor<uint16_t>&)x3, expMaskBF16, maskAll);
            And((RegTensor<uint16_t>&)x40, (RegTensor<uint16_t>&)x4, expMaskBF16, maskAll);
            Max(max1, x10, x20, maskAll);
            Max(max2, x30, x40, maskAll);
            DeInterleave(max1, max2, max1, max2);
            Max(maxPer4, max1, max2, maskAll); // 512--> 128

            DeInterleave(maxPer4Tmp1, maxPer4Tmp2, maxPer4, maxPer4);
            Max(maxPer8, maxPer4Tmp1, maxPer4Tmp2, maskAll); // 128 -> 64

            ReduceMaxWithDataBlock<uint16_t>((RegTensor<uint16_t>&)maxPer64, (RegTensor<uint16_t>&)maxPer4,
                                             maskAll); // (128, 1) -> (8, 16) -> (8, 1)

            // Vmax * Const_rec
            Mul(maxPer64, maxPer64, (RegTensor<bfloat16_t>&)numScale, maskAll);

            Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((RegTensor<fp8_e6m2_t>&)e6m2, maxPer64, maskAll);
            Pack(e6m2Res, (RegTensor<uint16_t>&)e6m2);
            DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(eaLocalAddr, e6m2Res, uregEa, 8);

            ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocal>(vb16EaRecBf16, (RegTensor<fp8_e6m2_t>&)e6m2,
                                                                        maskAll);

            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 8 -> 16
            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 16 -> 32
            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 32 -> 64
            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 64 -> 128

            Interleave(maxPer8, tmp, maxPer8, maxPer8); // 64 -> 128

            Mul(maxPer8, maxPer8, vb16EaRecBf16, maskAll);
            CompareScalar<bfloat16_t, CMPMODE::GE>(ebMask, maxPer8, 4, maskAll);
            DataCopyUnAlign<uint32_t>(ebLocalAddr, ebMask, uregEb);
            Select(maxPer8Coeff, numHalf, numOne, ebMask);

            Mul(maxPer4, maxPer4, vb16EaRecBf16, maskAll);
            Mul(maxPer4, maxPer4, maxPer8Coeff, maskAll);
            CompareScalar<bfloat16_t, CMPMODE::GE>(ecMask, maxPer4, 2, maskAll);
            Select(maxPer4Coeff, numHalf, numOne, ecMask);
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_PACK>(ecLocalAddr, ecMask, 16);

            Mul(invScale0, vb16EaRecBf16, maxPer8Coeff, maskAll);
            Mul(invScale0, invScale0, maxPer4Coeff, maskAll);

            Interleave(invScale1, invScale3, invScale0, invScale0);
            Interleave(invScale1, invScale2, invScale1, invScale1);
            Interleave(invScale3, invScale4, invScale3, invScale3);

            Interleave((RegTensor<float>&)x1, (RegTensor<float>&)x2, (RegTensor<float>&)x1, (RegTensor<float>&)x2);
            Interleave((RegTensor<float>&)x3, (RegTensor<float>&)x4, (RegTensor<float>&)x3, (RegTensor<float>&)x4);

            Mul(x1, x1, invScale1, maskAll);
            Mul(x2, x2, invScale2, maskAll);
            Mul(x3, x3, invScale3, maskAll);
            Mul(x4, x4, invScale4, maskAll);

            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x1Out, x1, maskAll);
            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x2Out, x2, maskAll);
            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x3Out, x3, maskAll);
            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x4Out, x4, maskAll);

            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x1Out, 16, maskAll);
            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x2Out, 16, maskAll);
            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x3Out, 16, maskAll);
            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x4Out, 16, maskAll);
        }
        DataCopyUnAlignPost(eaLocalAddr, uregEa, 0);
        DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ebLocalAddr, uregEb, 0);
    }
}

__aicore__ inline void VFProcessSwigluGroupQuantScale(const LocalTensor<uint16_t>& scaleLocal,
                                                      const LocalTensor<uint8_t>& eaLocal,
                                                      const LocalTensor<uint8_t>& ebLocal,
                                                      const LocalTensor<uint16_t>& ecLocal, const uint16_t curColNum)
{
    __local_mem__ uint16_t* scaleLocalAddr = (__local_mem__ uint16_t*)scaleLocal.GetPhyAddr();
    __local_mem__ uint8_t* eaLocalAddr = (__local_mem__ uint8_t*)eaLocal.GetPhyAddr();
    __local_mem__ uint8_t* ebLocalAddr = (__local_mem__ uint8_t*)ebLocal.GetPhyAddr();
    __local_mem__ uint16_t* ecLocalAddr = (__local_mem__ uint16_t*)ecLocal.GetPhyAddr();
    const uint16_t VL_LEN = 256;
    uint16_t vlForB8 = VL_LEN / sizeof(uint8_t);
    uint16_t vlForB16 = VL_LEN / sizeof(uint16_t);
    uint16_t eaNum = CeilDiv(curColNum, 64);
    uint16_t loopCount = CeilDiv(eaNum, vlForB8);
    __VEC_SCOPE__
    {
        RegTensor<uint8_t> eA;
        RegTensor<uint8_t> eB;
        RegTensor<uint16_t> eC0;
        RegTensor<uint16_t> eC1;
        RegTensor<uint8_t> tmp;
        MaskReg maskAll = CreateMask<uint8_t, MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopCount; i++) {
            DataCopy<uint8_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eA, eaLocalAddr + i * vlForB8);
            DataCopy<uint8_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eB, ebLocalAddr + i * vlForB8);
            DataCopy<uint16_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eC0, ecLocalAddr + i * vlForB16 * 2);
            DataCopy<uint16_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eC1, ecLocalAddr + i * vlForB16 * 2 + vlForB16);

            Interleave(eA, eB, eA, eB);
            DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(scaleLocalAddr + i * vlForB16 * 4,
                                                                    (MicroAPI::RegTensor<uint16_t>&)eA, eC0, maskAll);
            DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(scaleLocalAddr + i * vlForB16 * 4 + vlForB16 * 2,
                                                                    (MicroAPI::RegTensor<uint16_t>&)eB, eC1, maskAll);
        }
    }
}

__aicore__ inline uint32_t ScaleRowAlignHiFp4(uint32_t scaleCol)
{
    return ((scaleCol + 511U) / 512U) * 512U;
}

template <typename TX, typename TS>
__aicore__ inline void VFProcessDynamicHiFp4Quant(const LocalTensor<int8_t>& yLocal, const LocalTensor<TS>& scaleLocal,
                                                  const LocalTensor<TX>& xLocal, const LocalTensor<uint8_t>& eaBuf,
                                                  const LocalTensor<uint32_t>& ebBuf,
                                                  const LocalTensor<uint8_t>& ecWriteBuf,
                                                  const LocalTensor<uint16_t>& ecReadBuf, const uint16_t curRowNum,
                                                  const uint32_t curColNum)
{
    const uint32_t xRowStride = CeilAlign(RoundUp<TX>(curColNum), 512);
    const uint32_t cacheCol = (curColNum + 1) / 2;
    uint32_t elemCnt = cacheCol;
    const uint32_t cacheRowStride = CeilAlign(RoundUp<int8_t>(elemCnt), 128);
    const uint32_t scaleColU16 = CeilDiv(curColNum, ALIGN_PARAM1) * 2;
    const uint32_t scaleRowAlignU16 = ScaleRowAlignHiFp4(scaleColU16);
    const uint32_t scaleRowStrideTs = scaleRowAlignU16 * sizeof(uint16_t) / sizeof(TS);

    for (uint16_t r = 0; r < curRowNum; r++) {
        LocalTensor<float> xFloat = xLocal[r * xRowStride].template ReinterpretCast<float>();
        LocalTensor<int32_t> quantOutI32 = yLocal.template ReinterpretCast<int32_t>();

        VFProcessSwigluHiF4InvScaleAndQuant<TX>(quantOutI32[r * cacheRowStride / sizeof(int32_t)], xFloat, eaBuf, ebBuf,
                                                ecWriteBuf, curColNum);
        VFProcessSwigluGroupQuantScale(scaleLocal[r * scaleRowStrideTs].template ReinterpretCast<uint16_t>(), eaBuf,
                                       ebBuf.template ReinterpretCast<uint8_t>(), ecReadBuf, curColNum);
    }
}

} // namespace IndexerQuantCache

#endif // INDEXER_QUANT_CACHE_HIFP4_BASE_H
