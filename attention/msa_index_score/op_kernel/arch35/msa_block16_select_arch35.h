/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MSA_BLOCK16_SELECT_ARCH35_H
#define MSA_BLOCK16_SELECT_ARCH35_H

#include "kernel_operator.h"
#include "../msa_index_score_common.h"

// 基数选择原语保留在本算子内，支持独立构建。
namespace MsaSelectCommon {
template <typename T>
__aicore__ inline T Align(T value, T unit)
{
    return unit == 0 ? 0 : (value + unit - 1) / unit * unit;
}
} // namespace MsaSelectCommon

namespace MsaTopk32Gather {
using namespace AscendC;
// 保持直方图交错布局，转换计数后写出。
__simd_callee__ inline void StoreHistogram(__ubuf__ uint32_t* histogramsBuf, Reg::RegTensor<uint16_t>& cout0,
                                           Reg::RegTensor<uint16_t>& cout1, Reg::RegTensor<uint32_t>& cout0U32Even,
                                           Reg::RegTensor<uint32_t>& cout0U32Odd,
                                           Reg::RegTensor<uint32_t>& cout1U32Even,
                                           Reg::RegTensor<uint32_t>& cout1U32Odd, Reg::MaskReg& pregB16Reg,
                                           Reg::MaskReg& pregB32Reg)
{
    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg);
}

template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32Reg0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg0 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    // 计算直方图cout0 0-127 cout1 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;
    // 32bit 高16bit
    Reg::RegTensor<uint32_t> vreg0U16;
    // 32bit 低16bit
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;
    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);
        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t>&)vreg0U16, (Reg::RegTensor<uint8_t>&)vreg2U16);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg0,
                                                                                                          pregB8Reg0);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg0,
                                                                                                          pregB8Reg0);
    }
    StoreHistogram(histogramsBuf, cout0, cout1, cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd, pregB16Reg0,
                   pregB32Reg0);
}

template <bool LOAD_K>
__simd_vf__ void FindTargetBinVFImpl(__ubuf__ uint32_t* idxBuf, __ubuf__ uint32_t* nkValueBuf,
                                     __ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* kValue, uint32_t bottomK)
{
    Reg::MaskReg pregB32Reg = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> btmK;
    if constexpr (LOAD_K) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK, kValue);
    } else {
        Reg::Duplicate(btmK, bottomK);
    }
    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx;
        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32Reg);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx, (Reg::RegTensor<uint32_t>&)idxC, pregGE);
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
    Reg::Compare<uint32_t, CMPMODE::EQ>(pregZero, idx, zeroAll, pregB32Reg);
    Reg::Sub(idxPrev, idx, (Reg::RegTensor<uint32_t>&)idxAll1, pregB32Reg);
    Reg::ShiftRights(idxPrev, idxPrev, (int16_t)24, pregB32Reg);
    Reg::Gather(prevBinValue, histogramsBuf, idxPrev, pregB32Reg);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, pregZero);
    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK, prevBinValue, pregB32Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32Reg);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf,
                                        __ubuf__ uint32_t* idx0Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32Reg2 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg1 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;
    Reg::RegTensor<uint32_t> idx0;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;
    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);
        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t>&)vreg0U16, (Reg::RegTensor<uint8_t>&)vreg2U16);
        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (Reg::RegTensor<uint8_t>&)idx0, pregB8Reg1);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg1,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg1,
                                                                                                          pregEQ);
    }
    StoreHistogram(histogramsBuf, cout0, cout1, cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd, pregB16Reg1,
                   pregB32Reg2);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf,
                                       __ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* idx1Buf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32Reg4 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg2 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0, cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::RegTensor<uint32_t> cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd;
    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    Reg::RegTensor<uint32_t> vreg0U16, vreg1U16, vreg2U16, vreg3U16;
    Reg::RegTensor<uint8_t> vreg0, vreg1, vreg2, vreg3;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);
        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t>&)vreg0U16, (Reg::RegTensor<uint8_t>&)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t>&)vreg1U16, (Reg::RegTensor<uint8_t>&)vreg3U16);
        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t>&)idx0, pregB8Reg2);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t>&)idx1, pregB8Reg2);
        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ, pregEQ0, pregEQ1, pregB8Reg2);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg2,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg2,
                                                                                                          pregEQ);
    }
    StoreHistogram(histogramsBuf, cout0, cout1, cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd, pregB16Reg2,
                   pregB32Reg4);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf,
                                      __ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* idx1Buf,
                                      __ubuf__ uint32_t* idx2Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32Reg6 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg3 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg3 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0, cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::RegTensor<uint32_t> cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd;
    Reg::RegTensor<uint32_t> idx0, idx1, idx2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);
    Reg::RegTensor<uint32_t> vreg0U16, vreg1U16, vreg2U16, vreg3U16;
    Reg::RegTensor<uint8_t> vreg0, vreg1, vreg2, vreg3;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);
        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t>&)vreg0U16, (Reg::RegTensor<uint8_t>&)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t>&)vreg1U16, (Reg::RegTensor<uint8_t>&)vreg3U16);
        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t>&)idx0, pregB8Reg3);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t>&)idx1, pregB8Reg3);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (Reg::RegTensor<uint8_t>&)idx2, pregB8Reg3);
        Reg::MaskReg pregEQ0And1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8Reg3);
        Reg::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8Reg3);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg3,
                                                                                                          pregEQAll);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg3,
                                                                                                          pregEQAll);
    }
    StoreHistogram(histogramsBuf, cout0, cout1, cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd, pregB16Reg3,
                   pregB32Reg6);
}
__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t* kValue, __ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* idx0Buf,
                               __ubuf__ uint32_t* idx1Buf, __ubuf__ uint32_t* idx2Buf, __ubuf__ uint32_t* idx3Buf)
{
    Reg::MaskReg pregB32Reg7 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx3;
    Reg::RegTensor<uint32_t> btmK3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK3, kValue);
    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx3;
        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32Reg7);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx3, (Reg::RegTensor<uint32_t>&)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idx3Buf, sqzIdx3, alignIdx3);
    }
    Reg::StoreUnAlignPost(idx3Buf, alignIdx3);
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::RegTensor<uint32_t> idx2;
    Reg::RegTensor<uint32_t> idx3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx1, idx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx2, idx2Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx3, idx3Buf);
    Reg::ShiftLefts(idx0, idx0, (int16_t)24, pregB32Reg7);
    Reg::ShiftLefts(idx1, idx1, (int16_t)16, pregB32Reg7);
    Reg::ShiftLefts(idx2, idx2, (int16_t)8, pregB32Reg7);
    Reg::Add(idx0, idx0, idx1, pregB32Reg7);
    Reg::Add(idx0, idx0, idx2, pregB32Reg7);
    Reg::Add(idx0, idx0, idx3, pregB32Reg7);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(kValue, idx0, pregB32Reg7);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint32_t* inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg8 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);
    Reg::RegTensor<uint32_t> vregInput;
    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::Arange(idxC, beginIdx + i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);
        Reg::MaskReg poutGT = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint32_t> sqzIdxOut;
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32Reg8);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t>&)idxC, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint32_t* inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg9 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg poutEQ;
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);
    Reg::RegTensor<uint32_t> vregInput;
    Reg::RegTensor<int32_t> idxC;
    Reg::RegTensor<uint32_t> sqzIdxOut;
    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::Arange(idxC, beginIdx + i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32Reg9);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t>&)idxC, poutEQ);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

// 按选中编号收集最终分数。
__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint32_t* outputValueBuf, __ubuf__ uint32_t* inputValueBuf,
                                       __ubuf__ uint32_t* tmpIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg10 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputValue;
    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 64);
        Reg::Gather(outputValue, inputValueBuf, tmpIdx, pregB32Reg10);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputValueBuf + i * 64, outputValue, pregB32Reg10);
    }
}

__simd_vf__ void ComposeHalfThreshold(__ubuf__ uint32_t* out, __ubuf__ uint32_t* byte0, __ubuf__ uint32_t* byte1)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> hi, lo;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(hi, byte0);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(lo, byte1);
    Reg::ShiftLefts(hi, hi, static_cast<int16_t>(24), all);
    Reg::ShiftLefts(lo, lo, static_cast<int16_t>(16), all);
    Reg::Add(hi, hi, lo, all);
    Reg::StoreAlign<uint32_t>(out, hi, all);
}

// 输入按 256 对齐；FP16 键两轮、其余键四轮字节直方图，再筛选 TopK 编号。
template <bool ISOUTVALUE, bool HALF_KEYS = false>
__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t>& tmpIdxLocal, const LocalTensor<uint32_t>& outputValueLocal,
                                const LocalTensor<uint32_t>& inputValueLocal,
                                const LocalTensor<uint32_t>& histogramsLocal, const LocalTensor<uint32_t>& idx0Local,
                                const LocalTensor<uint32_t>& idx1Local, const LocalTensor<uint32_t>& idx2Local,
                                const LocalTensor<uint32_t>& idx3Local, const LocalTensor<uint32_t>& nkValueLocal,
                                uint32_t topK, uint32_t validLen)
{
    __ubuf__ uint32_t* tmpIdxBuf = (__ubuf__ uint32_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* outputValueBuf = (__ubuf__ uint32_t*)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t* inputValueBuf = (__ubuf__ uint32_t*)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t* histogramsBuf = (__ubuf__ uint32_t*)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t* idx0Buf = (__ubuf__ uint32_t*)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t* idx1Buf = (__ubuf__ uint32_t*)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t* idx2Buf = (__ubuf__ uint32_t*)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t* idx3Buf = (__ubuf__ uint32_t*)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t* nkValueBuf = (__ubuf__ uint32_t*)nkValueLocal.GetPhyAddr();
    uint32_t bottomK = validLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;
    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;
    uint16_t histogramsLoopNum = (validLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (validLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = (topK + 64 - 1) / 64;
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputValueBuf, histogramsLoopNum, flag);
    FindTargetBinVFImpl<false>(idx0Buf, nkValueBuf, histogramsBuf, nkValueBuf, bottomK);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, histogramsLoopNum, flag);
    FindTargetBinVFImpl<true>(idx1Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0);
    if constexpr (HALF_KEYS) {
        ComposeHalfThreshold(nkValueBuf, idx0Buf, idx1Buf);
    } else {
        HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, histogramsLoopNum, flag);
        FindTargetBinVFImpl<true>(idx2Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0);
        HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum,
                                       flag);
        FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);
    }
    int32_t count = MsaSelectCommon::Align(topK, (uint32_t)64) - topK / 64 * 64;
    AscendC::Duplicate(tmpIdxLocal[topK / 64 * 64], (uint32_t)(0), count);
    // 先输出大于阈值的编号。
    FindIdxGTOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // 再按输入顺序补齐等于阈值的编号。
    FindIdxEQOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    if constexpr (ISOUTVALUE) {
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, topkLoopNum);
    }
}

} // namespace MsaTopk32Gather

namespace MsaBlock16Select {
namespace FusedSelect {
using namespace AscendC;

// 工作区共 172 KiB：156 KiB 候选工作区和 16 KiB FP16 输入缓存。
constexpr uint32_t kScoreTile = 1024U;
constexpr uint32_t kRadixTile = 8192U;
constexpr uint32_t kRadixMaxTopK = 4096U;
constexpr uint32_t kMaxTopK = 4096U;
// 长历史中小于 8 的 K 使用小堆，其余使用两阶段基数合并。
constexpr uint32_t kSmallHeapMaxTopK = 7U;
constexpr uint32_t kSelectUbBytes = 176128U;
constexpr float kInvalidScore = -1.0e29F;
// 相同分数保留逻辑编号升序，稳定基数排序维持平局次序。
constexpr SortConfig kSortKeysDescending{SortType::RADIX_SORT, true};
constexpr Reg::CastTrait kIdIntToFloat = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                          RoundMode::CAST_ROUND};

__aicore__ inline uint32_t Min(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

__aicore__ inline uint32_t Align32(uint32_t value)
{
    return (value + 31U) / 32U * 32U;
}

// 保持浮点排序键转换及无效值语义。+0 的键高于 -0，与 FP16 位序一致。
__simd_vf__ void PrepareRadixKeys(__ubuf__ uint32_t* buffer, uint32_t validCount, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zeroBits;
    Reg::RegTensor<uint32_t> signBit;
    Reg::RegTensor<uint32_t> allBits;
    Reg::RegTensor<int32_t> validLimit;
    Reg::RegTensor<float> validThreshold;
    Reg::Duplicate(zeroBits, 0U);
    Reg::Duplicate(signBit, 0x80000000U);
    Reg::Duplicate(allBits, 0xffffffffU);
    Reg::Duplicate(validLimit, static_cast<int32_t>(validCount));
    Reg::Duplicate(validThreshold, -1.0e29f);
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> bits;
        Reg::RegTensor<int32_t> positions;
        Reg::RegTensor<uint32_t> sign;
        Reg::RegTensor<uint32_t> xorMask;
        Reg::RegTensor<uint32_t> key;
        Reg::MaskReg isNegative;
        Reg::MaskReg inRange;
        Reg::MaskReg scoreValid;
        Reg::MaskReg keep;
        Reg::LoadAlign<uint32_t>(bits, buffer + offset);
        Reg::Arange(positions, static_cast<int32_t>(offset));
        Reg::And(sign, bits, signBit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(isNegative, sign, zeroBits, all);
        Reg::Select(xorMask, allBits, signBit, isNegative);
        Reg::Xor(key, bits, xorMask, all);
        Reg::Compare<int32_t, CMPMODE::LT>(inRange, positions, validLimit, all);
        Reg::Compare<float, CMPMODE::GT>(scoreValid, reinterpret_cast<Reg::RegTensor<float>&>(bits), validThreshold,
                                         all);
        Reg::And(keep, inRange, scoreValid, all);
        Reg::Select(key, key, zeroBits, keep);
        Reg::StoreAlign<uint32_t>(buffer + offset, key, all);
    }
}

// FP16 bit order: -0 -> 0x7fff, +0 -> 0x8000; NaN ranks as -inf.
// Shift the 16-bit key into the high word so selection needs only two byte passes.
__simd_vf__ void PrepareHalfKeys(__ubuf__ uint32_t* dst, __ubuf__ uint16_t* src, uint32_t count, uint32_t aligned)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> lowMask, signBit, allBits, absMask, infBits, negInfKey, zero, limit;
    Reg::Duplicate(lowMask, 0xffffU);
    Reg::Duplicate(signBit, 0x8000U);
    Reg::Duplicate(allBits, 0xffffU);
    Reg::Duplicate(absMask, 0x7fffU);
    Reg::Duplicate(infBits, 0x7c00U);
    Reg::Duplicate(negInfKey, 0x03ffU);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(limit, count);
    for (uint32_t offset = 0U; offset < aligned; offset += 64U) {
        Reg::RegTensor<uint16_t> packed;
        Reg::RegTensor<uint32_t> bits, sign, xorMask, key, absBits, positions;
        Reg::MaskReg negative, nan, valid;
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_UNPACK_B16>(packed, src + offset);
        Reg::And(bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(packed), lowMask, all);
        Reg::And(sign, bits, signBit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(negative, sign, zero, all);
        Reg::Select(xorMask, allBits, signBit, negative);
        Reg::Xor(key, bits, xorMask, all);
        Reg::And(absBits, bits, absMask, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(nan, absBits, infBits, all);
        Reg::Select(key, negInfKey, key, nan);
        Reg::ShiftLefts(key, key, static_cast<int16_t>(16), all);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(positions), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, positions, limit, all);
        Reg::Select(key, key, zero, valid);
        Reg::StoreAlign<uint32_t>(dst + offset, key, all);
    }
}

// 将有效候选转为负编号并补负无穷，排序得到编号升序。
struct AscendingSortConstants {
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> limit;
    Reg::RegTensor<uint32_t> topKLimit;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> zeroFloat;
    Reg::RegTensor<float> negativeInfinity;
};

__simd_callee__ inline void InitAscendingSortConstants(AscendingSortConstants& constants, uint32_t count, uint32_t topK)
{
    Reg::Duplicate(constants.zero, 0U);
    Reg::Duplicate(constants.limit, count);
    Reg::Duplicate(constants.topKLimit, topK);
    Reg::Duplicate(constants.invalidId, 0xffffffffU);
    Reg::Duplicate(constants.one, 1.0F);
    Reg::Duplicate(constants.zeroFloat, 0.0F);
    Reg::Duplicate(constants.negativeInfinity, -__builtin_inff());
}

__simd_vf__ void PrepareAscendingIdSort(__ubuf__ float* values, __ubuf__ uint32_t* indices, __ubuf__ float* validFlags,
                                        __ubuf__ uint32_t* selected, __ubuf__ uint32_t* keys, uint32_t count,
                                        uint32_t topK, uint32_t alignedTopK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    AscendingSortConstants constants;
    InitAscendingSortConstants(constants, count, topK);
    for (uint32_t offset = 0U; offset < alignedTopK; offset += 64U) {
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> safeIds;
        Reg::RegTensor<uint32_t> key;
        Reg::RegTensor<float> idFloat;
        Reg::RegTensor<float> negativeId;
        Reg::RegTensor<float> valid;
        Reg::MaskReg inRange;
        Reg::MaskReg inTopK;
        Reg::MaskReg hasKey;
        Reg::MaskReg keep;
        Reg::LoadAlign<uint32_t>(ids, selected + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(positions), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(inRange, ids, constants.limit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(inTopK, positions, constants.topKLimit, all);
        Reg::And(inRange, inRange, inTopK, all);
        Reg::Select(safeIds, ids, constants.zero, inRange);
        Reg::Gather(key, keys, safeIds, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, key, constants.zero, all);
        Reg::And(keep, inRange, hasKey, all);
        Reg::Cast<float, int32_t, kIdIntToFloat>(idFloat, reinterpret_cast<Reg::RegTensor<int32_t>&>(safeIds), all);
        Reg::Muls(negativeId, idFloat, -1.0F, all);
        Reg::Select(negativeId, negativeId, constants.negativeInfinity, keep);
        Reg::Select(ids, ids, constants.invalidId, keep);
        Reg::Select(valid, constants.one, constants.zeroFloat, keep);
        Reg::StoreAlign<float>(values + offset, negativeId, all);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
        Reg::StoreAlign<float>(validFlags + offset, valid, all);
    }
}

__simd_vf__ void ExtractAscendingIds(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* sortedPairs, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    for (uint32_t offset = 0U; offset < topK; offset += 64U) {
        Reg::RegTensor<uint32_t> gatherIndex;
        Reg::RegTensor<uint32_t> ids;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(gatherIndex), static_cast<int32_t>(offset));
        Reg::Muls(gatherIndex, gatherIndex, 2U, all);
        Reg::Adds(gatherIndex, gatherIndex, 1U, all);
        Reg::Gather(ids, sortedPairs, gatherIndex, all);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
    }
}

__simd_vf__ void ExtractScoreOrderedIds(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* sortedPairs, uint32_t retained,
                                        uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::Duplicate(retainedLimit, retained);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    for (uint32_t offset = 0U; offset < topK; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> ids;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, lanes, retainedLimit, all);
        Reg::Muls(positions, lanes, 2U, all);
        Reg::Adds(positions, positions, 1U, all);
        Reg::Select(positions, positions, zero, valid);
        if (retained != 0U) {
            Reg::Gather(ids, sortedPairs, positions, all);
        } else {
            Reg::Duplicate(ids, 0U);
        }
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
    }
}

__simd_vf__ void PackScoreRadixInput(__ubuf__ uint32_t* dstIds, __ubuf__ uint32_t* dstKeys, __ubuf__ uint32_t* srcIds,
                                     __ubuf__ uint32_t* srcKeys, uint32_t retained, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::Duplicate(retainedLimit, retained);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        Reg::MaskReg inRange;
        Reg::MaskReg hasKey;
        Reg::MaskReg keep;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(positions), static_cast<int32_t>(offset));
        Reg::LoadAlign<uint32_t>(keys, srcKeys + offset);
        Reg::LoadAlign<uint32_t>(ids, srcIds + offset);
        Reg::Compare<uint32_t, CMPMODE::LT>(inRange, positions, retainedLimit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, keys, zero, all);
        Reg::And(keep, inRange, hasKey, all);
        Reg::Select(keys, keys, zero, keep);
        Reg::Select(ids, ids, invalidId, keep);
        Reg::StoreAlign<uint32_t>(dstIds + offset, ids, all);
        Reg::StoreAlign<uint32_t>(dstKeys + offset, keys, all);
    }
}

__simd_vf__ void UnpackScoreRadixIds(__ubuf__ uint32_t* dstIds, __ubuf__ uint32_t* sortedIds,
                                     __ubuf__ uint32_t* sortedKeys, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<uint32_t> keys;
        Reg::MaskReg hasKey;
        Reg::LoadAlign<uint32_t>(ids, sortedIds + offset);
        Reg::LoadAlign<uint32_t>(keys, sortedKeys + offset);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, keys, zero, all);
        Reg::Select(ids, ids, invalidId, hasKey);
        Reg::StoreAlign<uint32_t>(dstIds + offset, ids, all);
    }
}

// 从交错分数与编号中生成升序排序输入及无效尾部。
__simd_vf__ void PrepareSelectedIdsSort(__ubuf__ float* values, __ubuf__ uint32_t* indices,
                                        __ubuf__ uint32_t* retainedPairs, uint32_t retained, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<float> negativeInfinity;
    Reg::Duplicate(retainedLimit, retained);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::Duplicate(negativeInfinity, -__builtin_inff());
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> pairPositions;
        Reg::RegTensor<uint32_t> safePositions;
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<float> idFloat;
        Reg::RegTensor<float> negativeId;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, lanes, retainedLimit, all);
        Reg::Muls(pairPositions, lanes, 2U, all);
        Reg::Adds(pairPositions, pairPositions, 1U, all);
        Reg::Select(safePositions, pairPositions, zero, valid);
        if (retained != 0U) {
            Reg::Gather(ids, retainedPairs, safePositions, all);
        } else {
            Reg::Duplicate(ids, 0U);
        }
        Reg::Cast<float, int32_t, kIdIntToFloat>(idFloat, reinterpret_cast<Reg::RegTensor<int32_t>&>(ids), all);
        Reg::Muls(negativeId, idFloat, -1.0F, all);
        Reg::Select(negativeId, negativeId, negativeInfinity, valid);
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<float>(values + offset, negativeId, all);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
    }
}

// 用 Gather 拼接任意长度的候选，超出范围补零键。
__simd_vf__ void BuildCombinedKeys(__ubuf__ uint32_t* combinedKeys, __ubuf__ uint32_t* previousKeys,
                                   __ubuf__ uint32_t* incomingKeys, uint32_t previousCount, uint32_t combinedCount,
                                   uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> previousLimit;
    Reg::RegTensor<uint32_t> combinedLimit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(previousLimit, previousCount);
    Reg::Duplicate(combinedLimit, combinedCount);
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> previousPositions;
        Reg::RegTensor<uint32_t> incomingPositions;
        Reg::RegTensor<uint32_t> previousKey;
        Reg::RegTensor<uint32_t> incomingKey;
        Reg::RegTensor<uint32_t> key;
        Reg::MaskReg inCombined;
        Reg::MaskReg isPrevious;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(positions), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(inCombined, positions, combinedLimit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(isPrevious, positions, previousLimit, all);
        Reg::Select(previousPositions, positions, zero, isPrevious);
        Reg::Sub(incomingPositions, positions, previousLimit, all);
        Reg::Select(incomingPositions, zero, incomingPositions, isPrevious);
        Reg::Select(incomingPositions, incomingPositions, zero, inCombined);
        Reg::Gather(previousKey, previousKeys, previousPositions, all);
        Reg::Gather(incomingKey, incomingKeys, incomingPositions, all);
        Reg::Select(key, previousKey, incomingKey, isPrevious);
        Reg::Select(key, key, zero, inCombined);
        Reg::StoreAlign<uint32_t>(combinedKeys + offset, key, all);
    }
}

// 合并编号写入临时区，所有旧映射读取完成后再回写。
__simd_vf__ void GatherMergedCandidates(__ubuf__ uint32_t* outKeys, __ubuf__ uint32_t* mergedIds,
                                        __ubuf__ uint32_t* selectedPositions, __ubuf__ uint32_t* candidateKeys,
                                        __ubuf__ uint32_t* previousIds, __ubuf__ uint32_t* incomingIds,
                                        uint32_t previousCount, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> previousLimit;
    Reg::RegTensor<uint32_t> topKLimit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(previousLimit, previousCount);
    Reg::Duplicate(topKLimit, topK);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> safePositions;
        Reg::RegTensor<uint32_t> oldPositions;
        Reg::RegTensor<uint32_t> newPositions;
        Reg::RegTensor<uint32_t> safeOld;
        Reg::RegTensor<uint32_t> safeNew;
        Reg::RegTensor<uint32_t> key;
        Reg::RegTensor<uint32_t> oldId;
        Reg::RegTensor<uint32_t> newId;
        Reg::RegTensor<uint32_t> id;
        Reg::MaskReg inTopK;
        Reg::MaskReg isOld;
        Reg::LoadAlign<uint32_t>(positions, selectedPositions + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(inTopK, lanes, topKLimit, all);
        Reg::Select(safePositions, positions, zero, inTopK);
        Reg::Compare<uint32_t, CMPMODE::LT>(isOld, safePositions, previousLimit, all);
        Reg::Select(safeOld, safePositions, zero, isOld);
        Reg::Sub(newPositions, safePositions, previousLimit, all);
        Reg::Select(safeNew, zero, newPositions, isOld);
        Reg::Gather(key, candidateKeys, safePositions, all);
        Reg::Gather(oldId, previousIds, safeOld, all);
        Reg::Gather(newId, incomingIds, safeNew, all);
        Reg::Select(id, oldId, newId, isOld);
        Reg::StoreAlign<uint32_t>(outKeys + offset, key, all);
        Reg::StoreAlign<uint32_t>(mergedIds + offset, id, all);
    }
}

__simd_vf__ void CopyMergedIds(__ubuf__ uint32_t* previousIds, __ubuf__ uint32_t* mergedIds, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> id;
        Reg::LoadAlign<uint32_t>(id, mergedIds + offset);
        Reg::StoreAlign<uint32_t>(previousIds + offset, id, all);
    }
}

// 块内位置转为全局编号，零键候选留至行末统一处理。
__simd_vf__ void GatherChunkCandidates(__ubuf__ uint32_t* candidateKeys, __ubuf__ uint32_t* candidateIds,
                                       __ubuf__ uint32_t* selectedPositions, __ubuf__ uint32_t* chunkKeys,
                                       uint32_t count, uint32_t candidateCount, uint32_t chunkStart, bool useSelected)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> countLimit;
    Reg::RegTensor<uint32_t> candidateLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::Duplicate(countLimit, count);
    Reg::Duplicate(candidateLimit, candidateCount);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (candidateCount + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> safePositions;
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        Reg::MaskReg inCandidates;
        Reg::MaskReg inChunk;
        Reg::MaskReg validKey;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(offset));
        if (useSelected) {
            Reg::LoadAlign<uint32_t>(positions, selectedPositions + offset);
        } else {
            Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(positions), static_cast<int32_t>(offset));
        }
        Reg::Compare<uint32_t, CMPMODE::LT>(inCandidates, lanes, candidateLimit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(inChunk, positions, countLimit, all);
        Reg::And(inChunk, inChunk, inCandidates, all);
        Reg::Select(safePositions, positions, zero, inChunk);
        Reg::Gather(keys, chunkKeys, safePositions, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(validKey, keys, zero, all);
        Reg::And(valid, inChunk, validKey, all);
        Reg::Select(keys, keys, zero, valid);
        Reg::Adds(ids, safePositions, chunkStart, all);
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<uint32_t>(candidateKeys + offset, keys, all);
        Reg::StoreAlign<uint32_t>(candidateIds + offset, ids, all);
    }
}

__simd_vf__ void CopyChunkCandidates(__ubuf__ uint32_t* rankedKeys, __ubuf__ uint32_t* rankedIds,
                                     __ubuf__ uint32_t* candidateKeys, __ubuf__ uint32_t* candidateIds, uint32_t count,
                                     uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        if (offset < count) {
            Reg::LoadAlign<uint32_t>(keys, candidateKeys + offset);
            Reg::LoadAlign<uint32_t>(ids, candidateIds + offset);
        } else {
            Reg::Duplicate(keys, 0U);
            Reg::Duplicate(ids, 0xffffffffU);
        }
        Reg::StoreAlign<uint32_t>(rankedKeys + offset, keys, all);
        Reg::StoreAlign<uint32_t>(rankedIds + offset, ids, all);
    }
}

// 零键候选排序后转为编号 -1 的无效尾部。
__simd_vf__ void PrepareRankedIdsSort(__ubuf__ float* sortValues, __ubuf__ uint32_t* sortIds,
                                      __ubuf__ uint32_t* rankedKeys, uint32_t retained, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<float> negativeInfinity;
    Reg::Duplicate(retainedLimit, retained);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::Duplicate(negativeInfinity, -__builtin_inff());
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<float> idFloat;
        Reg::RegTensor<float> negativeId;
        Reg::MaskReg inRange;
        Reg::MaskReg hasKey;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(offset));
        Reg::LoadAlign<uint32_t>(keys, rankedKeys + offset);
        Reg::LoadAlign<uint32_t>(ids, sortIds + offset);
        Reg::Compare<uint32_t, CMPMODE::LT>(inRange, lanes, retainedLimit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, keys, zero, all);
        Reg::And(valid, inRange, hasKey, all);
        Reg::Cast<float, int32_t, kIdIntToFloat>(idFloat, reinterpret_cast<Reg::RegTensor<int32_t>&>(ids), all);
        Reg::Muls(negativeId, idFloat, -1.0F, all);
        Reg::Select(negativeId, negativeId, negativeInfinity, valid);
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<float>(sortValues + offset, negativeId, all);
        Reg::StoreAlign<uint32_t>(sortIds + offset, ids, all);
    }
}

// 小分数或同分的大编号更差，平局优先保留小编号。
__aicore__ inline bool ScoreWorse(float a, uint32_t ai, float b, uint32_t bi)
{
    return a < b || (a == b && ai > bi);
}

struct VisibleMeta {
    GM_ADDR actualSeqQ;
    GM_ADDR actualSeqK;
    uint32_t batch;
    uint32_t sparseMode;
    uint32_t initBlocks;
    uint32_t localBlocks;
    uint32_t compressFactor;
    uint32_t keyLayout;
    uint32_t qMaxSeq;
    uint32_t kMaxSeq;
    uint32_t maxBlocksPerBatch;
    uint32_t tableWidth;
    uint32_t pageSize;
};

// Validate the common full-K selection and locate the forced final visible block.
// Each statistic occupies its own aligned UB block for scalar consumption.
__simd_vf__ void FullSelectionStats128(__ubuf__ uint32_t* stats, __ubuf__ uint32_t* ids, uint32_t visible,
                                       uint32_t initEnd, uint32_t localBegin)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg first = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
    Reg::RegTensor<uint32_t> zero, one, visibleLimit, initialLimit, localLimit, target;
    Reg::RegTensor<uint32_t> validCounts, forcedCounts, positions, reduced;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(one, 1U);
    Reg::Duplicate(visibleLimit, visible);
    Reg::Duplicate(initialLimit, initEnd);
    Reg::Duplicate(localLimit, localBegin);
    Reg::Duplicate(target, visible - 1U);
    Reg::Duplicate(validCounts, 0U);
    Reg::Duplicate(forcedCounts, 0U);
    Reg::Duplicate(positions, 0U);
    for (uint32_t offset = 0U; offset < 128U; offset += 64U) {
        Reg::RegTensor<uint32_t> id, lanes, count, initial, local, targetPos;
        Reg::MaskReg valid, isInitial, isLocal, isTarget;
        Reg::LoadAlign(id, ids + offset);
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, id, visibleLimit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(isInitial, id, initialLimit, all);
        Reg::Compare<uint32_t, CMPMODE::GE>(isLocal, id, localLimit, all);
        Reg::And(isLocal, isLocal, valid, all);
        Reg::Select(count, one, zero, valid);
        Reg::Select(initial, one, zero, isInitial);
        Reg::Select(local, one, zero, isLocal);
        Reg::Add(validCounts, validCounts, count, all);
        Reg::Add(forcedCounts, forcedCounts, initial, all);
        Reg::Add(forcedCounts, forcedCounts, local, all);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(offset + 1U));
        Reg::Compare<uint32_t, CMPMODE::EQ>(isTarget, id, target, all);
        Reg::Select(targetPos, lanes, zero, isTarget);
        Reg::Max(positions, positions, targetPos, all);
    }
    Reg::ReduceSum(reduced, validCounts, all);
    Reg::StoreAlign<uint32_t>(stats, reduced, first);
    Reg::ReduceSum(reduced, forcedCounts, all);
    Reg::StoreAlign<uint32_t>(stats + 8U, reduced, first);
    Reg::ReduceMax(reduced, positions, all);
    Reg::StoreAlign<uint32_t>(stats + 16U, reduced, first);
}

class Selector final {
public:
    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR selectedIds, GM_ADDR selectedValues, uint32_t tokens,
                                uint32_t scoreStride, uint32_t usedCubeCores, bool returnValues,
                                const VisibleMeta& meta, TPipe* pipe)
    {
        groups_ = 1U;
        tokens_ = tokens;
        scoreStrideGroup_ = static_cast<uint64_t>(tokens) * scoreStride;
        scoreStrideToken_ = scoreStride;
        rout_ = scoreStride;
        maxRegionLen_ = scoreStride;
        topK_ = 128U;
        selectedStride_ = topK_;
        sortIds_ = false;
        usedCubeCores_ = usedCubeCores;
        returnValues_ = returnValues;
        pipe_ = pipe;
        batch_ = meta.batch;
        sparseMode_ = meta.sparseMode;
        initBlocks_ = meta.initBlocks;
        localBlocks_ = meta.localBlocks;
        compressFactor_ = meta.compressFactor < 1U ? 1U : meta.compressFactor;
        keyLayout_ = meta.keyLayout;
        qMaxSeq_ = meta.qMaxSeq;
        kMaxSeq_ = meta.kMaxSeq;
        maxBlocksPerBatch_ = meta.maxBlocksPerBatch;
        tableWidth_ = meta.tableWidth;
        pageSize_ = meta.pageSize;
        hasMeta_ = meta.actualSeqQ != nullptr && meta.actualSeqK != nullptr && meta.batch > 0U;
        if (hasMeta_) {
            seqQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(meta.actualSeqQ));
            seqK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(meta.actualSeqK));
        }
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scores));
        selectedIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selectedIds));
        selectedValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(selectedValues));
        // 评分释放并重置流水后，为每个 AIV 独立申请工作区。
        pipe_->InitBuffer(ub_, kSelectUbBytes);
        LocalTensor<uint8_t> base = ub_.Get<uint8_t>();
        halfInput_ = base[159744].ReinterpretCast<half>();
        score_ = base[0].ReinterpretCast<float>();
        index_ = base[4096].ReinterpretCast<uint32_t>();
        sortTmp_ = base[8192].ReinterpretCast<float>();
        chunkSorted_ = base[40960].ReinterpretCast<float>();
        retained_ = base[49152].ReinterpretCast<float>();
        merge_ = base[81952].ReinterpretCast<float>();
        orderValues_ = base[122912].ReinterpretCast<float>();
        orderIndices_ = base[139296].ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void Process()
    {
        if (topK_ == 0U || topK_ > kMaxTopK || groups_ == 0U || tokens_ == 0U || usedCubeCores_ == 0U) {
            return;
        }
        // 每个 AIV 独占行条带，使用原始工作核编号。
        const uint32_t workerCount = usedCubeCores_ * GetTaskRation();
        const uint32_t worker = GetBlockIdx();
        if (worker >= workerCount) {
            return;
        }
        const uint64_t rows = static_cast<uint64_t>(groups_) * tokens_;
        // 相邻短行共享搬运单元时，由同一 AIV 写入以避免竞争。
        uint32_t rowsPerUnit = 1U;
        while ((rowsPerUnit * selectedStride_) % 8U != 0U) {
            ++rowsPerUnit;
        }
        const uint64_t rowUnits = (rows + rowsPerUnit - 1U) / rowsPerUnit;
        for (uint64_t unit = worker; unit < rowUnits; unit += workerCount) {
            const uint64_t begin = unit * rowsPerUnit;
            const uint64_t end = Min(static_cast<uint32_t>(rows - begin), rowsPerUnit) + begin;
            for (uint64_t row = begin; row < end; ++row) {
                ProcessRow(static_cast<uint32_t>(row));
            }
        }
    }

private:
    __aicore__ inline void LoadHalfScoresRaw(uint64_t offset, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(half)), 0U, 0U, 0U};
        DataCopyPadExtParams<half> pad{false, 0U, 0U, static_cast<half>(0)};
        DataCopyPad(halfInput_, scores_[offset], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void LoadHalfScores(const LocalTensor<float>& dst, uint64_t offset, uint32_t count)
    {
        LoadHalfScoresRaw(offset, count);
        Cast(dst, halfInput_, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t HistoryLength(uint32_t) const
    {
        // Producer writes all chunk-owned columns, including causal/tail invalid fill.
        return maxRegionLen_;
    }

    __aicore__ inline void HeapSwap(uint32_t left, uint32_t right)
    {
        const float score = orderValues_.GetValue(left);
        const uint32_t id = orderIndices_.GetValue(left);
        orderValues_.SetValue(left, orderValues_.GetValue(right));
        orderIndices_.SetValue(left, orderIndices_.GetValue(right));
        orderValues_.SetValue(right, score);
        orderIndices_.SetValue(right, id);
    }

    // 最小堆根保存当前最差候选。
    __aicore__ inline void SiftScoreHeap(uint32_t root, uint32_t count)
    {
        while (2U * root + 1U < count) {
            uint32_t child = 2U * root + 1U;
            if (child + 1U < count && ScoreWorse(orderValues_.GetValue(child + 1U), orderIndices_.GetValue(child + 1U),
                                                 orderValues_.GetValue(child), orderIndices_.GetValue(child))) {
                ++child;
            }
            if (!ScoreWorse(orderValues_.GetValue(child), orderIndices_.GetValue(child), orderValues_.GetValue(root),
                            orderIndices_.GetValue(root))) {
                return;
            }
            HeapSwap(child, root);
            root = child;
        }
    }

    __aicore__ inline void Consider(float value, uint32_t id, uint32_t& retained)
    {
        // 有序比较同时排除 NaN 和无效分数哨兵。
        if (!(value > kInvalidScore)) {
            return;
        }
        if (retained < topK_) {
            uint32_t pos = retained++;
            while (pos > 0U) {
                const uint32_t parent = (pos - 1U) / 2U;
                if (!ScoreWorse(value, id, orderValues_.GetValue(parent), orderIndices_.GetValue(parent))) {
                    break;
                }
                orderValues_.SetValue(pos, orderValues_.GetValue(parent));
                orderIndices_.SetValue(pos, orderIndices_.GetValue(parent));
                pos = parent;
            }
            orderValues_.SetValue(pos, value);
            orderIndices_.SetValue(pos, id);
        } else if (ScoreWorse(orderValues_.GetValue(0U), orderIndices_.GetValue(0U), value, id)) {
            orderValues_.SetValue(0U, value);
            orderIndices_.SetValue(0U, id);
            SiftScoreHeap(0U, retained);
        }
    }

    __aicore__ inline void SelectHistoryTopK(uint32_t group, uint32_t token, uint32_t history, uint32_t& retained,
                                             uint32_t& directMode)
    {
        // K 不小于 8 时统一基数合并，小 K 使用有界堆。
        if (topK_ >= 8U) {
            SelectHistoryTopKRadixMerge(group, token, history, retained);
        } else if (history > kRadixTile) {
            SelectHistoryTopKRadixHeapSmall(group, token, history, retained);
        } else if (history >= kScoreTile) {
            SelectHistoryTopKRadix(group, token, history, retained, directMode);
        } else {
            SelectHistoryTopKSortHeap(group, token, history, retained);
        }
    }

    __aicore__ inline void SelectHistoryTopKSortHeap(uint32_t group, uint32_t token, uint32_t history,
                                                     uint32_t& retained)
    {
        retained = 0U;
        const uint64_t rowOffset =
            static_cast<uint64_t>(group) * scoreStrideGroup_ + static_cast<uint64_t>(token) * scoreStrideToken_;
        for (uint32_t start = 0U; start < history; start += kScoreTile) {
            const uint32_t count = Min(kScoreTile, history - start);
            LoadHalfScores(score_, rowOffset + start, count);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            // 完整块在向量侧排序，边界同分时扫描全块保证小编号优先。
            bool uniform = false;
            bool boundaryTie = true;
            if (count == kScoreTile) {
                SortFullScoreTile(start, count, uniform, boundaryTie);
            } else {
                for (uint32_t i = 0U; i < count; ++i) {
                    const float value = score_.GetValue(i);
                    score_.SetValue(i, value > kInvalidScore ? value : -3.4028234663852886e+38F);
                }
            }
            CollectSortedTile(start, count, uniform, boundaryTie, retained);
        }
        // 逆序提取最小堆，得到分数降序且同分编号升序。
        DrainScoreHeap(retained);
    }

    // 逆序提取最小堆，保持同分小编号优先。
    __aicore__ inline void DrainScoreHeap(uint32_t retained)
    {
        LocalTensor<uint32_t> retainedIds = retained_.ReinterpretCast<uint32_t>();
        for (uint32_t remaining = retained; remaining > 0U; --remaining) {
            retained_.SetValue(2U * (remaining - 1U), orderValues_.GetValue(0U));
            retainedIds.SetValue(2U * (remaining - 1U) + 1U, orderIndices_.GetValue(0U));
            if (remaining > 1U) {
                orderValues_.SetValue(0U, orderValues_.GetValue(remaining - 1U));
                orderIndices_.SetValue(0U, orderIndices_.GetValue(remaining - 1U));
                SiftScoreHeap(0U, remaining - 1U);
            }
        }
    }

    __aicore__ inline void SortFullScoreTile(uint32_t start, uint32_t count, bool& uniform, bool& boundaryTie)
    {
        LocalTensor<uint8_t> validMask = sortTmp_.ReinterpretCast<uint8_t>();
        CompareScalar(validMask, score_, kInvalidScore, CMPMODE::GT, kScoreTile);
        PipeBarrier<PIPE_V>();
        Select(score_, validMask, score_, -__builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, kScoreTile);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        ReduceMax(sortTmp_, score_, sortTmp_[128], static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceMin(sortTmp_[64], score_, sortTmp_[128], static_cast<int32_t>(count), false);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uniform = sortTmp_.GetValue(0U) == sortTmp_.GetValue(64U);
        if (!uniform) {
            ArithProgression<int32_t>(index_.ReinterpretCast<int32_t>(), static_cast<int32_t>(start), 1, kScoreTile);
            // 编号写入后同步至向量侧，排序结果另行同步至标量侧。
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Sort<float, true>(chunkSorted_, score_, index_, sortTmp_, kScoreTile / 32U);
            const uint32_t localK = Min(topK_, count);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            boundaryTie =
                localK < count && chunkSorted_.GetValue(2U * (localK - 1U)) == chunkSorted_.GetValue(2U * localK);
        }
    }

    __aicore__ inline void CollectSortedTile(uint32_t start, uint32_t count, bool uniform, bool boundaryTie,
                                             uint32_t& retained)
    {
        const uint32_t localK = Min(topK_, count);
        const uint32_t candidates = count == kScoreTile && !uniform && !boundaryTie ? localK : count;
        LocalTensor<uint32_t> sortedIds = chunkSorted_.ReinterpretCast<uint32_t>();
        for (uint32_t i = 0U; i < candidates; ++i) {
            const float value =
                count == kScoreTile && !uniform && !boundaryTie ? chunkSorted_.GetValue(2U * i) : score_.GetValue(i);
            const uint32_t id =
                count == kScoreTile && !uniform && !boundaryTie ? sortedIds.GetValue(2U * i + 1U) : start + i;
            Consider(value, id, retained);
        }
    }

    template <bool HALF_KEYS = false>
    __aicore__ inline void FilterRadixCandidates(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
                                                 uint32_t aligned)
    {
        LocalTensor<uint32_t> hist = chunkSorted_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> bucket0 = hist[256U];
        LocalTensor<uint32_t> bucket1 = hist[512U];
        LocalTensor<uint32_t> bucket2 = hist[768U];
        LocalTensor<uint32_t> bucket3 = hist[1024U];
        LocalTensor<uint32_t> kth = hist[1280U];
        MsaTopk32Gather::LiTopKVF<false, HALF_KEYS>(ids, index_, keys, hist, bucket0, bucket1, bucket2, bucket3, kth,
                                                    topK_, aligned);
    }

    __aicore__ inline void CollectRadixHeapCandidates(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
                                                      uint32_t count, uint32_t start, uint32_t& retained)
    {
        const uint32_t candidateCount = count > topK_ ? topK_ : count;
        for (uint32_t i = 0U; i < candidateCount; ++i) {
            const uint32_t localId = count > topK_ ? ids.GetValue(i) : i;
            if (localId < count) {
                const uint32_t key = keys.GetValue(localId);
                if (key != 0U) {
                    Consider(DecodeRadixKey(key), start + localId, retained);
                }
            }
        }
    }

    __aicore__ inline void SortFinalRadixScores(uint32_t retained, LocalTensor<uint32_t> rankedKeys)
    {
        const uint32_t sortAligned = Align32(topK_);
        LocalTensor<uint32_t> inputIds = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> inputKeys = inputIds[kMaxTopK];
        LocalTensor<uint32_t> outputIds = retained_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> outputKeys = outputIds[kMaxTopK];
        LocalTensor<uint8_t> radixScratch = merge_.ReinterpretCast<uint8_t>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PackScoreRadixInput(reinterpret_cast<__ubuf__ uint32_t*>(inputIds.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(inputKeys.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(rankedKeys.GetPhyAddr()), retained, sortAligned);
        PipeBarrier<PIPE_V>();
        Sort<uint32_t, uint32_t, false, kSortKeysDescending>(outputKeys, outputIds, inputKeys, inputIds, radixScratch,
                                                             sortAligned);
        PipeBarrier<PIPE_V>();
        UnpackScoreRadixIds(reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(outputIds.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(outputKeys.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        scoreIdsReady_ = true;
    }

    __aicore__ inline void MergeRadixCandidates(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
                                                LocalTensor<uint32_t> rankedKeys, LocalTensor<uint32_t> incomingKeys,
                                                LocalTensor<uint32_t> incomingIds, LocalTensor<uint32_t> mergedIds,
                                                uint32_t retained, uint32_t combined)
    {
        const uint32_t mergeAligned = (combined + 255U) / 256U * 256U;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        BuildCombinedKeys(reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t*>(rankedKeys.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t*>(incomingKeys.GetPhyAddr()), retained, combined,
                          mergeAligned);
        PipeBarrier<PIPE_V>();
        FilterRadixCandidates<true>(ids, keys, mergeAligned);
        PipeBarrier<PIPE_V>();
        GatherMergedCandidates(reinterpret_cast<__ubuf__ uint32_t*>(rankedKeys.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(mergedIds.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(incomingIds.GetPhyAddr()), retained, topK_);
        PipeBarrier<PIPE_V>();
        CopyMergedIds(reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t*>(mergedIds.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        // 所有旧映射读取完成后回写，同分候选保持块间编号升序。
    }

    __aicore__ inline void AppendRadixCandidates(LocalTensor<uint32_t> rankedKeys, LocalTensor<uint32_t> incomingKeys,
                                                 LocalTensor<uint32_t> incomingIds, uint32_t retained,
                                                 uint32_t incoming)
    {
        if (retained == 0U) {
            CopyChunkCandidates(reinterpret_cast<__ubuf__ uint32_t*>(rankedKeys.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint32_t*>(incomingKeys.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint32_t*>(incomingIds.GetPhyAddr()), incoming, topK_);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            for (uint32_t i = 0U; i < incoming; ++i) {
                rankedKeys.SetValue(retained + i, incomingKeys.GetValue(i));
                orderIndices_.SetValue(retained + i, incomingIds.GetValue(i));
            }
        }
    }

    __aicore__ inline void PrepareIncomingRadixCandidates(uint64_t rowOffset, uint32_t start, uint32_t history,
                                                          LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
                                                          LocalTensor<uint32_t> incomingKeys,
                                                          LocalTensor<uint32_t> incomingIds, uint32_t& candidateCount)
    {
        const uint32_t count = Min(kRadixTile, history - start);
        const uint32_t aligned = (count + 255U) / 256U * 256U;
        LoadHalfScoresRaw(rowOffset + start, count);
        PrepareHalfKeys(reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint16_t*>(halfInput_.GetPhyAddr()), count, aligned);
        PipeBarrier<PIPE_V>();
        if (boostOn_ && ForcedTileHits(start, count)) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            BoostForcedKeys(keys, start, count);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        const bool reduceChunk = count > topK_;
        if (reduceChunk) {
            // 上一轮标量写入的索引工作区复用前同步至向量侧。
            if (start != 0U) {
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            FilterRadixCandidates<true>(ids, keys, aligned);
        }
        candidateCount = reduceChunk ? topK_ : count;
        GatherChunkCandidates(reinterpret_cast<__ubuf__ uint32_t*>(incomingKeys.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ uint32_t*>(incomingIds.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()), count, candidateCount, start,
                              reduceChunk);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void CollectShortRadixTile(uint64_t rowOffset, uint32_t start, uint32_t count, uint32_t& retained)
    {
        for (uint32_t offset = 0U; offset < count; offset += kScoreTile) {
            const uint32_t n = Min(kScoreTile, count - offset);
            LoadHalfScores(score_, rowOffset + start + offset, n);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            for (uint32_t i = 0U; i < n; ++i) {
                Consider(score_.GetValue(i), start + offset + i, retained);
            }
        }
    }

    __aicore__ inline void CheckUniformRadixTile(uint32_t count, uint32_t& retained, uint32_t& directMode)
    {
        // 最大最小值及求和联合检查同分有效行，拒绝 NaN。
        ReduceMax(score_, sortTmp_, merge_, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceMin(score_[2], sortTmp_, merge_, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceSum(score_[4], sortTmp_, merge_, static_cast<int32_t>(count));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float minimum = score_.GetValue(2U);
        const float maximum = score_.GetValue(0U);
        const float total = score_.GetValue(4U);
        if (minimum == maximum && maximum > kInvalidScore && total == maximum * static_cast<float>(count)) {
            retained = topK_;
            directMode = 1U;
            // 单块路径由循环后的公共出口收尾。
            return;
        }
    }

    __aicore__ inline void TryAscendingRadixIds(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys, uint32_t count,
                                                uint32_t& retained, uint32_t& directMode)
    {
        // 全有效候选在向量侧排序，无效候选沿用标量回退。
        const uint32_t alignedTopK = (topK_ + 63U) / 64U * 64U;
        PipeBarrier<PIPE_V>();
        PrepareAscendingIdSort(reinterpret_cast<__ubuf__ float*>(orderValues_.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(score_.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()), count, topK_, alignedTopK);
        PipeBarrier<PIPE_V>();
        // 有效性结果置于排序输入之外，避免覆盖完整评分块。
        ReduceMin(orderValues_[alignedTopK], score_, chunkSorted_, static_cast<int32_t>(topK_), false);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (orderValues_.GetValue(alignedTopK) == 1.0F) {
            Sort<float, true>(merge_, orderValues_, orderIndices_, sortTmp_, alignedTopK / 32U);
            PipeBarrier<PIPE_V>();
            ExtractAscendingIds(reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint32_t*>(merge_.GetPhyAddr()), topK_);
            retained = topK_;
            directMode = 2U;
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            return;
        }
    }

    __aicore__ inline void CollectRadixSelection(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
                                                 LocalTensor<uint32_t> selected, uint32_t count, uint32_t start,
                                                 bool singleTrunk, uint32_t& retained)
    {
        for (uint32_t i = 0U; i < topK_; ++i) {
            const uint32_t localId = ids.GetValue(i);
            if (localId < count && keys.GetValue(localId) != 0U) {
                if (singleTrunk) {
                    retained_.SetValue(2U * retained, 0.0F);
                    selected.SetValue(2U * retained + 1U, localId);
                    ++retained;
                } else {
                    Consider(DecodeRadixKey(keys.GetValue(localId)), start + localId, retained);
                }
            }
        }
    }

    __aicore__ inline void SortSingleRadixSelection(LocalTensor<uint32_t> keys, LocalTensor<uint32_t> selected,
                                                    uint32_t retained)
    {
        const uint32_t alignedK = Align32(topK_);
        for (uint32_t i = 0U; i < alignedK; ++i) {
            if (i < retained) {
                const uint32_t id = selected.GetValue(2U * i + 1U);
                score_.SetValue(i, DecodeRadixKey(keys.GetValue(id)));
                index_.SetValue(i, id);
            } else {
                score_.SetValue(i, -__builtin_inff());
                index_.SetValue(i, 0xffffffffU);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sort<float, true>(chunkSorted_, score_, index_, sortTmp_, alignedK / 32U);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        LocalTensor<uint32_t> sorted = chunkSorted_.ReinterpretCast<uint32_t>();
        RestoreScoreTieOrder(sorted, selected, retained);
    }

    __aicore__ inline void RestoreScoreTieOrder(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> selected,
                                                uint32_t retained)
    {
        // 排序无稳定平局保证，单独恢复同分候选的小编号优先。
        uint32_t begin = 0U;
        while (begin < retained) {
            const float value = chunkSorted_.GetValue(2U * begin);
            uint32_t end = begin + 1U;
            while (end < retained && chunkSorted_.GetValue(2U * end) == value) {
                ++end;
            }
            if (end - begin >= 32U) {
                // 长同分段使用负编号向量排序，避免标量插入的平方开销。
                SortLongScoreTie(sorted, selected, begin, end);
            } else {
                SortShortScoreTie(sorted, selected, begin, end);
            }
            begin = end;
        }
    }

    __aicore__ inline void SortLongScoreTie(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> selected,
                                            uint32_t begin, uint32_t end)
    {
        const uint32_t tieCount = end - begin;
        const uint32_t tieAligned = Align32(tieCount);
        for (uint32_t i = 0U; i < tieAligned; ++i) {
            const uint32_t id = i < tieCount ? sorted.GetValue(2U * (begin + i) + 1U) : 0xffffffffU;
            score_.SetValue(i, i < tieCount ? -static_cast<float>(id) : -3.4028234663852886e+38F);
            index_.SetValue(i, id);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sort<float, true>(merge_, score_, index_, sortTmp_, tieAligned / 32U);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        LocalTensor<uint32_t> sortedTie = merge_.ReinterpretCast<uint32_t>();
        for (uint32_t i = 0U; i < tieCount; ++i) {
            selected.SetValue(2U * (begin + i) + 1U, sortedTie.GetValue(2U * i + 1U));
        }
    }

    __aicore__ inline void SortShortScoreTie(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> selected,
                                             uint32_t begin, uint32_t end)
    {
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t id = sorted.GetValue(2U * i + 1U);
            uint32_t pos = i;
            while (pos > begin && selected.GetValue(2U * (pos - 1U) + 1U) > id) {
                selected.SetValue(2U * pos + 1U, selected.GetValue(2U * (pos - 1U) + 1U));
                --pos;
            }
            selected.SetValue(2U * pos + 1U, id);
        }
    }

    __aicore__ inline void LoadRadixScoreTile(uint64_t offset, uint32_t count)
    {
        LoadHalfScores(sortTmp_, offset, count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline float DecodeRadixKey(uint32_t key) const
    {
        const uint32_t bits = (key & 0x80000000U) != 0U ? key ^ 0x80000000U : ~key;
        return __builtin_bit_cast(float, bits);
    }

    // 小 K 每块基数筛选后合入小堆，避免重复基数合并。
    __aicore__ inline void SelectHistoryTopKRadixHeapSmall(uint32_t group, uint32_t token, uint32_t history,
                                                           uint32_t& retained)
    {
        const uint64_t rowOffset =
            static_cast<uint64_t>(group) * scoreStrideGroup_ + static_cast<uint64_t>(token) * scoreStrideToken_;
        LocalTensor<uint32_t> ids = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> keys = merge_.ReinterpretCast<uint32_t>();
        retained = 0U;
        for (uint32_t start = 0U; start < history; start += kRadixTile) {
            const uint32_t count = Min(kRadixTile, history - start);
            const uint32_t aligned = (count + 255U) / 256U * 256U;
            LoadHalfScores(merge_, rowOffset + start, count);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PrepareRadixKeys(reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()), count, aligned);
            PipeBarrier<PIPE_V>();
            if (count > topK_) {
                FilterRadixCandidates(ids, keys, aligned);
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            CollectRadixHeapCandidates(ids, keys, count, start, retained);
        }
        DrainScoreHeap(retained);
    }

    __aicore__ inline void SelectHistoryTopKRadixMerge(uint32_t group, uint32_t token, uint32_t history,
                                                       uint32_t& retained)
    {
        const uint64_t rowOffset =
            static_cast<uint64_t>(group) * scoreStrideGroup_ + static_cast<uint64_t>(token) * scoreStrideToken_;
        uint32_t visible = 0U;
        cachedReady_ = VisibleBlocksOf(token, visible);
        cachedToken_ = token;
        cachedVisible_ = visible;
        if (cachedReady_) {
            history = Min(history, visible);
        }
        boostOn_ = false;
        if (cachedReady_ && visible > topK_) {
            boostLocalBegin_ = visible > localBlocks_ ? (visible - localBlocks_) : 0U;
            boostInitEnd_ = initBlocks_ < boostLocalBegin_ ? initBlocks_ : boostLocalBegin_;
            boostVisible_ = visible;
            boostOn_ = (boostInitEnd_ > 0U) || (boostVisible_ > boostLocalBegin_);
        }
        // 每块缩减至 K 个候选，复用工作区进行第二阶段合并。
        LocalTensor<uint32_t> ids = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> keys = merge_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> rankedKeys = orderValues_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> incomingKeys = retained_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> incomingIds = incomingKeys[kMaxTopK];
        LocalTensor<uint32_t> mergedIds = ids[kMaxTopK];
        retained = 0U;
        for (uint32_t start = 0U; start < history; start += kRadixTile) {
            uint32_t candidateCount;
            PrepareIncomingRadixCandidates(rowOffset, start, history, ids, keys, incomingKeys, incomingIds,
                                           candidateCount);
            const uint32_t incoming = candidateCount;
            const uint32_t combined = retained + incoming;
            if (combined <= topK_) {
                AppendRadixCandidates(rankedKeys, incomingKeys, incomingIds, retained, incoming);
                retained = combined;
                continue;
            }
            MergeRadixCandidates(ids, keys, rankedKeys, incomingKeys, incomingIds, mergedIds, retained, combined);
            retained = topK_;
        }
        // 编号排序只需候选集合。分数基数排序已取消。
        if (sortIds_) {
            rankedIdsReady_ = true;
            return;
        }
        // 候选块号已在 orderIndices_。行末只把最大块号换到最后一个有效槽。
        scoreIdsReady_ = true;
    }

    __aicore__ inline void SelectHistoryTopKRadix(uint32_t group, uint32_t token, uint32_t history, uint32_t& retained,
                                                  uint32_t& directMode)
    {
        const uint64_t rowOffset =
            static_cast<uint64_t>(group) * scoreStrideGroup_ + static_cast<uint64_t>(token) * scoreStrideToken_;
        LocalTensor<uint32_t> keys = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> ids = merge_.ReinterpretCast<uint32_t>();
        retained = 0U;
        const bool singleTrunk = history <= kRadixTile;
        LocalTensor<uint32_t> selected = retained_.ReinterpretCast<uint32_t>();
        for (uint32_t start = 0U; start < history; start += kRadixTile) {
            const uint32_t count = Min(kRadixTile, history - start);
            // 尾块不足 K 时，全部元素参与全局候选竞争。
            if (count < topK_) {
                CollectShortRadixTile(rowOffset, start, count, retained);
                continue;
            }
            const uint32_t aligned = (count + 255U) / 256U * 256U;
            LoadRadixScoreTile(rowOffset + start, count);
            if (singleTrunk) {
                CheckUniformRadixTile(count, retained, directMode);
                if (directMode == 1U) {
                    continue;
                }
            }
            PrepareRadixKeys(reinterpret_cast<__ubuf__ uint32_t*>(keys.GetPhyAddr()), count, aligned);
            PipeBarrier<PIPE_V>();
            FilterRadixCandidates(ids, keys, aligned);
            if (singleTrunk && sortIds_) {
                TryAscendingRadixIds(ids, keys, count, retained, directMode);
                if (directMode == 2U) {
                    continue;
                }
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            CollectRadixSelection(ids, keys, selected, count, start, singleTrunk, retained);
            if (singleTrunk && !sortIds_ && retained != 0U) {
                SortSingleRadixSelection(keys, selected, retained);
            }
        }
        if (singleTrunk) {
            return;
        }
        DrainScoreHeap(retained);
    }

    __aicore__ inline void SortSelectedIds(uint32_t retained)
    {
        LocalTensor<uint32_t> retainedIds = retained_.ReinterpretCast<uint32_t>();
        const uint32_t aligned = Align32(topK_);
        // 负编号降序对应逻辑编号升序。
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PrepareSelectedIdsSort(reinterpret_cast<__ubuf__ float*>(orderValues_.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(retainedIds.GetPhyAddr()), retained, aligned);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(merge_, orderValues_, orderIndices_, sortTmp_, aligned / 32U);
        PipeBarrier<PIPE_V>();
        ExtractAscendingIds(reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(merge_.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void SortRankedIds(uint32_t retained)
    {
        const uint32_t aligned = Align32(topK_);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PrepareRankedIdsSort(reinterpret_cast<__ubuf__ float*>(orderValues_.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ uint32_t*>(orderValues_.GetPhyAddr()), retained, aligned);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(merge_, orderValues_, orderIndices_, sortTmp_, aligned / 32U);
        PipeBarrier<PIPE_V>();
        ExtractAscendingIds(reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ uint32_t*>(merge_.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline int32_t ClampI32(int32_t v, int32_t lo, int32_t hi) const
    {
        if (v < lo) {
            return lo;
        }
        if (v > hi) {
            return hi;
        }
        return v;
    }

    __aicore__ inline uint32_t RowsPerBlock() const
    {
        uint32_t rows = MsaIndexScoreNs::MSA_BLOCK16_SIZE / compressFactor_;
        return rows == 0U ? 1U : rows;
    }

    __aicore__ inline uint32_t BlockCap() const
    {
        uint32_t cap = maxBlocksPerBatch_;
        if (MsaIndexScoreNs::MsaIsPaKeyLayout(keyLayout_) && tableWidth_ > 0U) {
            const uint32_t perPage = pageSize_ / RowsPerBlock();
            const uint64_t tableBlocks = static_cast<uint64_t>(tableWidth_) * perPage;
            if (tableBlocks < cap) {
                cap = static_cast<uint32_t>(tableBlocks);
            }
        }
        if (cap > maxRegionLen_) {
            cap = maxRegionLen_;
        }
        return cap;
    }

    __aicore__ inline int32_t VisibleKeyEndOf(int32_t kvLen, int32_t kvTokenLen, int32_t qLen, int32_t tOff) const
    {
        if (sparseMode_ == MsaIndexScoreNs::MSA_SPARSE_MODE_RIGHT_DOWN) {
            const int32_t cap = kvLen < 0 ? 0 : kvLen;
            if (compressFactor_ == 1U) {
                return ClampI32(kvLen - qLen + tOff + 1, 0, cap);
            }
            const int32_t cf = static_cast<int32_t>(compressFactor_);
            const int32_t position = kvTokenLen - qLen + tOff;
            return position < 0 ? 0 : ClampI32(position / cf + 1, 0, cap);
        }
        return kvLen < 0 ? 0 : kvLen;
    }

    // 与 LoadBatchMeta / ApplyLocalMaskBlock16 同一可见块数。找不到 token 时返回 false。
    __aicore__ inline bool VisibleBlocksOf(uint32_t token, uint32_t& numVisible) const
    {
        numVisible = 0U;
        if (!hasMeta_) {
            return false;
        }
        for (uint32_t b = 0U; b < batch_; ++b) {
            const uint32_t begin = static_cast<uint32_t>(seqQ_.GetValue(b));
            const uint32_t end = static_cast<uint32_t>(seqQ_.GetValue(b + 1U));
            if (token < begin || token >= end) {
                continue;
            }
            int32_t qLen = static_cast<int32_t>(end - begin);
            if (qLen < 0) {
                qLen = 0;
            }
            if (qMaxSeq_ > 0U && qLen > static_cast<int32_t>(qMaxSeq_)) {
                qLen = static_cast<int32_t>(qMaxSeq_);
            }
            const int32_t tOff = static_cast<int32_t>(token - begin);
            int32_t kvLen = 0;
            int32_t kvTokenLen = 0;
            if (compressFactor_ == 1U) {
                if (MsaIndexScoreNs::MsaIsSeqKeyLayout(keyLayout_)) {
                    kvLen = static_cast<int32_t>(seqK_.GetValue(b + 1U)) - static_cast<int32_t>(seqK_.GetValue(b));
                    if (kvLen < 0) {
                        kvLen = 0;
                    }
                    if (kMaxSeq_ > 0U && kvLen > static_cast<int32_t>(kMaxSeq_)) {
                        kvLen = static_cast<int32_t>(kMaxSeq_);
                    }
                } else {
                    kvLen = seqK_.GetValue(b);
                    if (kvLen < 0) {
                        kvLen = 0;
                    }
                }
            } else {
                int32_t tokenLen = 0;
                if (MsaIndexScoreNs::MsaIsSeqKeyLayout(keyLayout_)) {
                    tokenLen = static_cast<int32_t>(seqK_.GetValue(b + 1U)) - static_cast<int32_t>(seqK_.GetValue(b));
                } else {
                    tokenLen = seqK_.GetValue(b);
                }
                if (tokenLen < 0) {
                    tokenLen = 0;
                }
                kvTokenLen = tokenLen;
                const int32_t cf = static_cast<int32_t>(compressFactor_);
                kvLen = (tokenLen + cf - 1) / cf;
                if (kMaxSeq_ > 0U && kvLen > static_cast<int32_t>(kMaxSeq_)) {
                    kvLen = static_cast<int32_t>(kMaxSeq_);
                }
            }
            const int32_t visibleKeyEnd = VisibleKeyEndOf(kvLen, kvTokenLen, qLen, tOff);
            int32_t blocks = 0;
            if (visibleKeyEnd > 0) {
                const int32_t rows = static_cast<int32_t>(RowsPerBlock());
                blocks = (visibleKeyEnd + rows - 1) / rows;
            }
            const uint32_t cap = BlockCap();
            if (blocks < 0) {
                blocks = 0;
            }
            if (blocks > static_cast<int32_t>(cap)) {
                blocks = static_cast<int32_t>(cap);
            }
            numVisible = static_cast<uint32_t>(blocks);
            return true;
        }
        return true;
    }

    __aicore__ inline int32_t ForcedSlot(uint32_t id, uint32_t initEnd, uint32_t localBegin, uint32_t visible) const
    {
        if (id < initEnd) {
            return static_cast<int32_t>(id);
        }
        if (id >= localBegin && id < visible) {
            return static_cast<int32_t>(initEnd + (id - localBegin));
        }
        return -1;
    }

    // 可见块不超过 topk：整段保留，块号从小到大，尾部 -1。
    // 超过 topk：缺的前缀和局部块替换最差的非强制块，再把最大块号换到最后一个有效槽。
    __aicore__ inline bool RangeHits(uint32_t start, uint32_t count, uint32_t lo, uint32_t hi) const
    {
        return lo < hi && start < hi && (start + count) > lo;
    }

    __aicore__ inline bool ForcedTileHits(uint32_t start, uint32_t count) const
    {
        return RangeHits(start, count, 0U, boostInitEnd_) || RangeHits(start, count, boostLocalBegin_, boostVisible_);
    }

    // 真实分数不再抬到 1e30/1e29。把强制块的排序键设成最大，使它们留在 128 个候选里。
    __aicore__ inline void BoostForcedKeys(LocalTensor<uint32_t> keys, uint32_t start, uint32_t count)
    {
        const uint32_t end = start + count;
        uint32_t from = boostInitEnd_ < end ? 0U : end;
        if (from < start) {
            from = start;
        }
        uint32_t to = boostInitEnd_ < end ? boostInitEnd_ : end;
        for (uint32_t id = from; id < to; ++id) {
            keys.SetValue(id - start, 0xffffffffU);
        }
        from = boostLocalBegin_ > start ? boostLocalBegin_ : start;
        to = boostVisible_ < end ? boostVisible_ : end;
        for (uint32_t id = from; id < to; ++id) {
            keys.SetValue(id - start, 0xffffffffU);
        }
    }

    __aicore__ inline void AlignVisibleSelection(uint32_t row)
    {
        uint32_t visible = 0U;
        const uint32_t token = tokens_ == 0U ? row : (row - (row / tokens_) * tokens_);
        if (cachedReady_ && cachedToken_ == token) {
            visible = cachedVisible_;
        } else if (!VisibleBlocksOf(token, visible)) {
            MoveMaxBlockToLast();
            return;
        }
        if (visible <= topK_) {
            for (uint32_t i = 0U; i < topK_; ++i) {
                orderIndices_.SetValue(i, i < visible ? i : 0xffffffffU);
            }
            return;
        }
        const uint32_t localBegin = visible > localBlocks_ ? (visible - localBlocks_) : 0U;
        const uint32_t initEnd = initBlocks_ < localBegin ? initBlocks_ : localBegin;
        const uint32_t forced = initEnd + (visible - localBegin);
        const uint32_t forcedUse = forced > topK_ ? topK_ : forced;
        if (topK_ == 128U && boostOn_ && localBlocks_ > 0U && forced == forcedUse) {
            auto stats = score_.ReinterpretCast<uint32_t>();
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            FullSelectionStats128(reinterpret_cast<__ubuf__ uint32_t*>(stats.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()), visible, initEnd,
                                  localBegin);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t pos = stats.GetValue(16U);
            if (stats.GetValue(0U) == 128U && stats.GetValue(8U) == forcedUse && pos > 0U && pos <= 128U) {
                if (pos != 128U) {
                    orderIndices_.SetValue(pos - 1U, orderIndices_.GetValue(127U));
                    orderIndices_.SetValue(127U, visible - 1U);
                }
                return;
            }
        }
        uint32_t seen[4] = {0U, 0U, 0U, 0U};
        uint32_t maxPos = 0U;
        uint32_t maxId = 0U;
        bool found = false;
        uint32_t last = 0U;
        for (uint32_t i = 0U; i < topK_; ++i) {
            const uint32_t id = orderIndices_.GetValue(i);
            if (id == 0xffffffffU) {
                continue;
            }
            last = i + 1U;
            if (!found || id > maxId) {
                found = true;
                maxId = id;
                maxPos = i;
            }
            const int32_t slot = ForcedSlot(id, initEnd, localBegin, visible);
            if (slot >= 0 && static_cast<uint32_t>(slot) < forcedUse) {
                seen[static_cast<uint32_t>(slot) >> 5] |= 1U << (static_cast<uint32_t>(slot) & 31U);
            }
        }
        uint32_t missing = 0U;
        for (uint32_t slot = 0U; slot < forcedUse; ++slot) {
            if ((seen[slot >> 5] & (1U << (slot & 31U))) == 0U) {
                ++missing;
            }
        }
        if (missing == 0U) {
            if (found && last > 1U && maxPos + 1U != last) {
                const uint32_t tail = orderIndices_.GetValue(last - 1U);
                orderIndices_.SetValue(maxPos, tail);
                orderIndices_.SetValue(last - 1U, maxId);
            }
            return;
        }
        PipeBarrier<PIPE_ALL>();
        const uint64_t scoreOffset = static_cast<uint64_t>(row) * scoreStrideToken_;
        for (uint32_t i = 0U; i < topK_; ++i) {
            const uint32_t id = orderIndices_.GetValue(i);
            float value = -3.4028234663852886e+38F;
            if (id != 0xffffffffU && id < scoreStrideToken_) {
                value = scores_.GetValue(scoreOffset + id);
            }
            score_.SetValue(i, value);
        }
        for (uint32_t slot = 0U; slot < forcedUse; ++slot) {
            if ((seen[slot >> 5] & (1U << (slot & 31U))) != 0U) {
                continue;
            }
            const uint32_t fid = slot < initEnd ? slot : (localBegin + (slot - initEnd));
            int32_t hole = -1;
            int32_t evict = -1;
            float worstScore = 0.0F;
            uint32_t worstId = 0U;
            for (uint32_t i = 0U; i < topK_; ++i) {
                const uint32_t id = orderIndices_.GetValue(i);
                if (id == 0xffffffffU) {
                    if (hole < 0) {
                        hole = static_cast<int32_t>(i);
                    }
                    continue;
                }
                const int32_t owned = ForcedSlot(id, initEnd, localBegin, visible);
                if (owned >= 0 && static_cast<uint32_t>(owned) < forcedUse) {
                    continue;
                }
                const float value = score_.GetValue(i);
                if (evict < 0 || value < worstScore || (value == worstScore && id > worstId)) {
                    evict = static_cast<int32_t>(i);
                    worstScore = value;
                    worstId = id;
                }
            }
            const int32_t dst = hole >= 0 ? hole : evict;
            if (dst < 0) {
                break;
            }
            orderIndices_.SetValue(static_cast<uint32_t>(dst), fid);
            seen[slot >> 5] |= 1U << (slot & 31U);
        }
        MoveMaxBlockToLast();
    }

    // 分数顺序保持不变，只把最大有效块号换到最后一个有效槽。-1 尾部不动。
    __aicore__ inline void MoveMaxBlockToLast()
    {
        uint32_t last = topK_;
        while (last > 0U && orderIndices_.GetValue(last - 1U) == 0xffffffffU) {
            --last;
        }
        if (last <= 1U) {
            return;
        }
        uint32_t maxPos = 0U;
        uint32_t maxId = 0U;
        bool found = false;
        for (uint32_t i = 0U; i < last; ++i) {
            const uint32_t id = orderIndices_.GetValue(i);
            if (id == 0xffffffffU) {
                continue;
            }
            if (!found || id > maxId) {
                found = true;
                maxId = id;
                maxPos = i;
            }
        }
        if (!found || maxPos + 1U == last) {
            return;
        }
        const uint32_t tail = orderIndices_.GetValue(last - 1U);
        orderIndices_.SetValue(maxPos, tail);
        orderIndices_.SetValue(last - 1U, maxId);
    }

    __aicore__ inline void Store(uint32_t row, uint32_t retained, uint32_t directMode)
    {
        if (directMode != 0U) {
            if (directMode == 1U) {
                ArithProgression<int32_t>(orderIndices_.ReinterpretCast<int32_t>(), 0, 1, topK_);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams directCopy{1U, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(selectedIds_[static_cast<uint64_t>(row) * selectedStride_],
                        orderIndices_.ReinterpretCast<int32_t>(), directCopy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        } else {
            LocalTensor<uint32_t> retainedIds = retained_.ReinterpretCast<uint32_t>();
            if (sortIds_) {
                if (rankedIdsReady_) {
                    SortRankedIds(retained);
                } else {
                    SortSelectedIds(retained);
                }
            } else if (!scoreIdsReady_) {
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                ExtractScoreOrderedIds(reinterpret_cast<__ubuf__ uint32_t*>(orderIndices_.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ uint32_t*>(retainedIds.GetPhyAddr()), retained, topK_);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
            if (!sortIds_) {
                AlignVisibleSelection(row);
            }
            DataCopyExtParams copy{1U, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0U, 0U, 0U};
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(selectedIds_[static_cast<uint64_t>(row) * selectedStride_],
                        orderIndices_.ReinterpretCast<int32_t>(), copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        if (returnValues_) {
            const uint64_t scoreOffset = static_cast<uint64_t>(row) * scoreStrideToken_;
            for (uint32_t i = 0U; i < topK_; ++i) {
                const int32_t id = static_cast<int32_t>(orderIndices_.GetValue(i));
                score_.SetValue(i, id >= 0 ?
                                       static_cast<float>(scores_.GetValue(scoreOffset + static_cast<uint32_t>(id))) :
                                       -__builtin_inff());
            }
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(selectedValues_[static_cast<uint64_t>(row) * topK_], score_, topK_);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        rankedIdsReady_ = false;
        scoreIdsReady_ = false;
        const uint32_t group = row / tokens_;
        const uint32_t token = row - group * tokens_;
        const uint32_t history = HistoryLength(token);
        uint32_t retained = 0U;
        uint32_t directMode = 0U;
        if (history != 0U) {
            SelectHistoryTopK(group, token, history, retained, directMode);
        }
        Store(row, retained, directMode);
    }

    uint32_t groups_ = 0U;
    uint32_t tokens_ = 0U;
    uint64_t scoreStrideGroup_ = 0U;
    uint64_t scoreStrideToken_ = 0U;
    uint32_t rout_ = 0U;
    uint32_t maxRegionLen_ = 0U;
    uint32_t topK_ = 0U;
    uint32_t selectedStride_ = 0U;
    bool sortIds_ = false;
    bool rankedIdsReady_ = false;
    bool scoreIdsReady_ = false;
    uint32_t usedCubeCores_ = 0U;
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<half> scores_;
    LocalTensor<half> halfInput_;
    GlobalTensor<int32_t> selectedIds_;
    GlobalTensor<float> selectedValues_;
    bool returnValues_ = false;
    bool hasMeta_ = false;
    bool cachedReady_ = false;
    bool boostOn_ = false;
    uint32_t cachedToken_ = 0U;
    uint32_t cachedVisible_ = 0U;
    uint32_t boostInitEnd_ = 0U;
    uint32_t boostLocalBegin_ = 0U;
    uint32_t boostVisible_ = 0U;
    uint32_t batch_ = 0U;
    uint32_t sparseMode_ = MsaIndexScoreNs::MSA_SPARSE_MODE_RIGHT_DOWN;
    uint32_t initBlocks_ = 0U;
    uint32_t localBlocks_ = 0U;
    uint32_t compressFactor_ = 1U;
    uint32_t keyLayout_ = 0U;
    uint32_t qMaxSeq_ = 0U;
    uint32_t kMaxSeq_ = 0U;
    uint32_t maxBlocksPerBatch_ = 0U;
    uint32_t tableWidth_ = 0U;
    uint32_t pageSize_ = MsaIndexScoreNs::MSA_BLOCK16_SIZE;
    GlobalTensor<int32_t> seqQ_;
    GlobalTensor<int32_t> seqK_;
    LocalTensor<float> score_, sortTmp_, chunkSorted_, retained_, merge_, orderValues_;
    LocalTensor<uint32_t> index_, orderIndices_;
};

__aicore__ inline void Run(GM_ADDR scores, GM_ADDR indices, GM_ADDR values, uint32_t tokens, uint32_t stride,
                           uint32_t cubeCores, bool returnValues, const VisibleMeta& meta)
{
    TPipe pipe;
    Selector selector;
    selector.Init(scores, indices, values, tokens, stride, cubeCores, returnValues, meta, &pipe);
    selector.Process();
}
} // namespace FusedSelect
} // namespace MsaBlock16Select
#endif // MSA_BLOCK16_SELECT_ARCH35_H
