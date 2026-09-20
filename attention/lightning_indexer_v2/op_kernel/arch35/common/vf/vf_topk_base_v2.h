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
 * \file vf_topk_base_v2.h
 * \brief Common vector TopK helpers shared by LightningIndexerV2 and
 * QuantLightningIndexerV2.
 */

#ifndef VF_TOPK_BASE_V2_H
#define VF_TOPK_BASE_V2_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace liV2TopkCommon {
namespace Reg = AscendC::Reg;
using AscendC::CMPMODE;
using AscendC::LocalTensor;
using AscendC::RoundMode;

__simd_callee__ inline void StoreHistogramResult(__ubuf__ uint32_t *histogramsBuf, Reg::RegTensor<uint16_t> &cout0,
                                                 Reg::RegTensor<uint16_t> &cout1, Reg::MaskReg &pregB16,
                                                 Reg::MaskReg &pregB32)
{
    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_callee__ inline void FindTargetBinAndUpdateNextK(__ubuf__ uint32_t *idxBuf, __ubuf__ uint32_t *nkValueBuf,
                                                        __ubuf__ uint32_t *histogramsBuf,
                                                        Reg::RegTensor<uint32_t> &btmK, Reg::MaskReg &pregB32)
{
    Reg::UnalignRegForStore alignIdx;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxBuf, sqzIdx, alignIdx);
    }
    Reg::StoreUnAlignPost(idxBuf, alignIdx);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx, idxBuf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg pregZero = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(pregZero, idx, zeroAll, pregB32);
    Reg::Sub(idxPrev, idx, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32);
    Reg::ShiftRights(idxPrev, idxPrev, (int16_t)24, pregB32);

    Reg::Gather(prevBinValue, histogramsBuf, idxPrev, pregB32);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, pregZero);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK, prevBinValue, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

__simd_vf__ void IndicesAddOffsetVF(__ubuf__ uint32_t *indicesOutBuf, uint32_t outputIdxOffset, uint32_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> outIndices;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(outIndices, indicesOutBuf + i * 64);
        Reg::Adds(outIndices, outIndices, outputIdxOffset, pregB32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(indicesOutBuf + i * 64, outIndices, pregB32);
    }
}

__aicore__ inline void IndicesAddOffset(const LocalTensor<uint32_t> &indicesOutLocal, uint32_t outputIdxOffset,
                                        uint32_t topK)
{
    __ubuf__ uint32_t *indicesOutBuf = (__ubuf__ uint32_t *)indicesOutLocal.GetPhyAddr();
    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;
    IndicesAddOffsetVF(indicesOutBuf, outputIdxOffset, topkLoopNum32);
}

} // namespace liV2TopkCommon

#endif // VF_TOPK_BASE_V2_H
