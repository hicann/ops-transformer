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
 * \file vf_topk_gather.h
 * \brief
 */

#ifndef SI_VF_TOP_K_GATHER_H
#define SI_VF_TOP_K_GATHER_H

namespace SITopkb32gather {
constexpr uint32_t HISTOGRAM_BIN_COUNT = 256U;
constexpr uint32_t HISTOGRAM_HALF_BIN_COUNT = 128U;
constexpr uint32_t VF_INPUT_CHUNK_ELEMS = 256U;
constexpr uint32_t VF_B32_ELEMS = 64U;
constexpr uint32_t TOPK_ALIGN_ELEMS = 256U;

template <typename T>
__simd_callee__ inline void HistogramsFirstProcessRow(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                                      uint16_t vfLoop, uint32_t offset, uint32_t rowSlot,
                                                      uint32_t realRowIdx)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;

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

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                 roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg0,
                                                                                                          pregB8);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg0,
                                                                                                          pregB8);
    }
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT, cout1U32Even,
                                                              cout1U32Odd, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf, uint16_t vfLoop,
                                       uint32_t offset, uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2,
                                       uint32_t rowIdx3)
{
    HistogramsFirstProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 0, rowIdx0);
    HistogramsFirstProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 1, rowIdx1);
    HistogramsFirstProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 2, rowIdx2);
    HistogramsFirstProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 3, rowIdx3);
}

__simd_callee__ inline void FindFirstTargetBinProcessRow(__ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *nkValueBuf,
                                                         __ubuf__ uint32_t *histogramsBuf, uint32_t topK,
                                                         uint32_t validLen, uint32_t rowSlot)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx0;

    Reg::RegTensor<uint32_t> topKReg;
    Reg::Duplicate(topKReg, topK);
    Reg::RegTensor<uint32_t> validLenPlus1;
    Reg::Duplicate(validLenPlus1, validLen + 1);
    Reg::RegTensor<uint32_t> btmK;
    Reg::Sub(btmK, validLenPlus1, topKReg, pregB32);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx0;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * VF_B32_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx0, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(roundIdx0Buf, sqzIdx0, alignIdx0);
    }
    Reg::StoreUnAlignPost(roundIdx0Buf, alignIdx0);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx0;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev0;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg0, idx0, zeroAll, pregB32);
    Reg::Sub(idxPrev0, idx0, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32);
    Reg::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    Reg::Gather(prevBinValue, roundHistBuf, idxPrev0, pregB32);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK, prevBinValue, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, uint32_t validLen, uint32_t topkNum0,
                                          uint32_t topkNum1, uint32_t topkNum2, uint32_t topkNum3)
{
    FindFirstTargetBinProcessRow(idx0Buf, nkValueBuf, histogramsBuf, topkNum0, validLen, 0);
    FindFirstTargetBinProcessRow(idx0Buf, nkValueBuf, histogramsBuf, topkNum1, validLen, 1);
    FindFirstTargetBinProcessRow(idx0Buf, nkValueBuf, histogramsBuf, topkNum2, validLen, 2);
    FindFirstTargetBinProcessRow(idx0Buf, nkValueBuf, histogramsBuf, topkNum3, validLen, 3);
}

template <typename T>
__simd_callee__ inline void HistogramsSecondProcessRow(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                                       __ubuf__ uint32_t *idx0Buf, uint16_t vfLoop, uint32_t offset,
                                                       uint32_t rowSlot, uint32_t realRowIdx)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    Reg::RegTensor<uint32_t> idx0;

    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                 roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg1,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg1,
                                                                                                          pregEQ);
    }
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT, cout1U32Even,
                                                              cout1U32Odd, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                        __ubuf__ uint32_t *idx0Buf, uint16_t vfLoop, uint32_t offset, uint32_t rowIdx0,
                                        uint32_t rowIdx1, uint32_t rowIdx2, uint32_t rowIdx3)
{
    HistogramsSecondProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, vfLoop, offset, 0, rowIdx0);
    HistogramsSecondProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, vfLoop, offset, 1, rowIdx1);
    HistogramsSecondProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, vfLoop, offset, 2, rowIdx2);
    HistogramsSecondProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, vfLoop, offset, 3, rowIdx3);
}

// kValue新的bottomK
__simd_callee__ inline void FindSecondTargetBinProcessRow(__ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *nkValueBuf,
                                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf,
                                                          uint32_t rowSlot)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx1;

    Reg::RegTensor<uint32_t> btmK1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK1, roundKValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx1;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * VF_B32_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK1, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx1, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(roundIdx1Buf, sqzIdx1, alignIdx1);
    }
    Reg::StoreUnAlignPost(roundIdx1Buf, alignIdx1);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, roundIdx1Buf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev1;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg1 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg1, idx1, zeroAll, pregB32);
    Reg::Sub(idxPrev1, idx1, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32);
    Reg::ShiftRights(idxPrev1, idxPrev1, (int16_t)24, pregB32);

    Reg::Gather(prevBinValue, roundHistBuf, idxPrev1, pregB32);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg1);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK1, prevBinValue, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
}

__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *nkValueBuf,
                                           __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    FindSecondTargetBinProcessRow(idx1Buf, nkValueBuf, kValue, histogramsBuf, 0);
    FindSecondTargetBinProcessRow(idx1Buf, nkValueBuf, kValue, histogramsBuf, 1);
    FindSecondTargetBinProcessRow(idx1Buf, nkValueBuf, kValue, histogramsBuf, 2);
    FindSecondTargetBinProcessRow(idx1Buf, nkValueBuf, kValue, histogramsBuf, 3);
}

template <typename T>
__simd_callee__ inline void HistogramsThirdProcessRow(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                                      __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                                      uint16_t vfLoop, uint32_t offset, uint32_t rowSlot,
                                                      uint32_t realRowIdx)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;

    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, roundIdx1Buf);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                 roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t> &)vreg1U16, (Reg::RegTensor<uint8_t> &)vreg3U16);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t> &)idx1, pregB8);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ, pregEQ0, pregEQ1, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg2,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg2,
                                                                                                          pregEQ);
    }
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT, cout1U32Even,
                                                              cout1U32Odd, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                       __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, uint16_t vfLoop,
                                       uint32_t offset, uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2,
                                       uint32_t rowIdx3)
{
    HistogramsThirdProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, vfLoop, offset, 0, rowIdx0);
    HistogramsThirdProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, vfLoop, offset, 1, rowIdx1);
    HistogramsThirdProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, vfLoop, offset, 2, rowIdx2);
    HistogramsThirdProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, vfLoop, offset, 3, rowIdx3);
}

__simd_callee__ inline void FindThirdTargetBinProcessRow(__ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *nkValueBuf,
                                                         __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf,
                                                         uint32_t rowSlot)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdx2Buf = idx2Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx2;

    Reg::RegTensor<uint32_t> btmK2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK2, roundKValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx2;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * VF_B32_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK2, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx2, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(roundIdx2Buf, sqzIdx2, alignIdx2);
    }
    Reg::StoreUnAlignPost(roundIdx2Buf, alignIdx2);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, roundIdx2Buf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev2;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg2 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg2, idx2, zeroAll, pregB32);
    Reg::Sub(idxPrev2, idx2, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32);
    Reg::ShiftRights(idxPrev2, idxPrev2, (int16_t)24, pregB32);

    Reg::Gather(prevBinValue, roundHistBuf, idxPrev2, pregB32);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg2);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK2, prevBinValue, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    FindThirdTargetBinProcessRow(idx2Buf, nkValueBuf, kValue, histogramsBuf, 0);
    FindThirdTargetBinProcessRow(idx2Buf, nkValueBuf, kValue, histogramsBuf, 1);
    FindThirdTargetBinProcessRow(idx2Buf, nkValueBuf, kValue, histogramsBuf, 2);
    FindThirdTargetBinProcessRow(idx2Buf, nkValueBuf, kValue, histogramsBuf, 3);
}

template <typename T>
__simd_callee__ inline void HistogramsLastProcessRow(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                                     __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                                     __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, uint32_t offset,
                                                     uint32_t rowSlot, uint32_t realRowIdx)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::RegTensor<uint32_t> idx2;

    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx2Buf = idx2Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, roundIdx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, roundIdx2Buf);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                 roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t> &)vreg1U16, (Reg::RegTensor<uint8_t> &)vreg3U16);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t> &)idx1, pregB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (Reg::RegTensor<uint8_t> &)idx2, pregB8);

        Reg::MaskReg pregEQ0And1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8);
        Reg::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg3,
                                                                                                          pregEQAll);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg3,
                                                                                                          pregEQAll);
    }
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT, cout1U32Even,
                                                              cout1U32Odd, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                      __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                      __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, uint32_t offset, uint32_t rowIdx0,
                                      uint32_t rowIdx1, uint32_t rowIdx2, uint32_t rowIdx3)
{
    HistogramsLastProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, vfLoop, offset, 0, rowIdx0);
    HistogramsLastProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, vfLoop, offset, 1, rowIdx1);
    HistogramsLastProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, vfLoop, offset, 2, rowIdx2);
    HistogramsLastProcessRow<T>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, vfLoop, offset, 3, rowIdx3);
}

__simd_callee__ inline void FindKthProcessRow(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf,
                                              __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                              __ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *idx3Buf, uint32_t rowSlot)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx2Buf = idx2Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx3Buf = idx3Buf + rowSlot * HISTOGRAM_BIN_COUNT;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx3;

    Reg::RegTensor<uint32_t> btmK3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK3, roundKValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx3;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * VF_B32_ELEMS);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx3, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(roundIdx3Buf, sqzIdx3, alignIdx3);
    }
    Reg::StoreUnAlignPost(roundIdx3Buf, alignIdx3);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::RegTensor<uint32_t> idx2;
    Reg::RegTensor<uint32_t> idx3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx0, roundIdx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx1, roundIdx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx2, roundIdx2Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx3, roundIdx3Buf);

    Reg::ShiftLefts(idx0, idx0, (int16_t)24, pregB32);
    Reg::ShiftLefts(idx1, idx1, (int16_t)16, pregB32);
    Reg::ShiftLefts(idx2, idx2, (int16_t)8, pregB32);

    Reg::Add(idx0, idx0, idx1, pregB32);
    Reg::Add(idx0, idx0, idx2, pregB32);
    Reg::Add(idx0, idx0, idx3, pregB32);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(roundKValue, idx0, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *idx0Buf,
                               __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *idx3Buf)
{
    FindKthProcessRow(kValue, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf, 0);
    FindKthProcessRow(kValue, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf, 1);
    FindKthProcessRow(kValue, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf, 2);
    FindKthProcessRow(kValue, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf, 3);
}

__simd_callee__ inline void FindIdxOutputProcessRow(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf,
                                                    __ubuf__ uint32_t *kValue, uint16_t vfLoop, uint32_t offset,
                                                    uint32_t tmpIdxOffset, uint32_t rowSlot, uint32_t realRowIdx)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundOutputIdxBuf = outputIdxBuf + rowSlot * tmpIdxOffset;
    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, roundKValue);

    Reg::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::Arange(idxC, i * VF_B32_ELEMS);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, roundInputBuf + i * VF_B32_ELEMS);

        Reg::MaskReg poutGT = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint32_t> sqzIdxOut;
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(roundOutputIdxBuf, sqzIdxOut, alignIdx);
    }

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::Arange(idxC, i * VF_B32_ELEMS);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, roundInputBuf + i * VF_B32_ELEMS);

        Reg::MaskReg poutEQ;

        Reg::RegTensor<uint32_t> sqzIdxOut;
        Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutEQ);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(roundOutputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(roundOutputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf,
                                     __ubuf__ uint32_t *kValue, uint16_t vfLoop, uint32_t offset, uint32_t tmpIdxOffset,
                                     uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2, uint32_t rowIdx3)
{
    FindIdxOutputProcessRow(outputIdxBuf, inputBuf, kValue, vfLoop, offset, tmpIdxOffset, 0, rowIdx0);
    FindIdxOutputProcessRow(outputIdxBuf, inputBuf, kValue, vfLoop, offset, tmpIdxOffset, 1, rowIdx1);
    FindIdxOutputProcessRow(outputIdxBuf, inputBuf, kValue, vfLoop, offset, tmpIdxOffset, 2, rowIdx2);
    FindIdxOutputProcessRow(outputIdxBuf, inputBuf, kValue, vfLoop, offset, tmpIdxOffset, 3, rowIdx3);
}

/**
    输出最终的Value
 */
__simd_callee__ inline void FindValueOutputProcessRow(__ubuf__ uint32_t *outputValueBuf,
                                                      __ubuf__ uint32_t *inputValueBuf, __ubuf__ uint32_t *tmpIdxBuf,
                                                      uint16_t vfLoop, uint32_t inputOffset, uint32_t tmpIdxOffset,
                                                      uint32_t rowSlot, uint32_t realRowIdx)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    uint32_t outputOffset = tmpIdxOffset;

    __ubuf__ uint32_t *roundOutputValueBuf = outputValueBuf + rowSlot * outputOffset;
    __ubuf__ uint32_t *roundInputValueBuf = inputValueBuf + realRowIdx * inputOffset;
    __ubuf__ uint32_t *roundTmpIdxBuf = tmpIdxBuf + rowSlot * tmpIdxOffset;

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, roundTmpIdxBuf + i * VF_B32_ELEMS);

        Reg::Gather(outputValue, roundInputValueBuf, tmpIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(roundOutputValueBuf + i * VF_B32_ELEMS, outputValue,
                                                             pregB32);
    }
}

__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputValueBuf,
                                       __ubuf__ uint32_t *tmpIdxBuf, uint16_t vfLoop, uint32_t inputOffset,
                                       uint32_t tmpIdxOffset, uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2,
                                       uint32_t rowIdx3)
{
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset, 0, rowIdx0);
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset, 1, rowIdx1);
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset, 2, rowIdx2);
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset, 3, rowIdx3);
}

/**
    输出最终的Idx
 */
__simd_callee__ inline void FindRealIndexProcessRow(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *tmpIdxBuf,
                                                    __ubuf__ uint32_t *hisIdxBuf, uint32_t topK, uint32_t loopIndex,
                                                    uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregNow;
    Reg::MaskReg pregHis;

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputGatherIdx;
    Reg::RegTensor<uint32_t> outputAddsIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * VF_B32_ELEMS);

        Reg::Compares<uint32_t, CMPMODE::GT>(pregNow, tmpIdx, topK - 1, pregB32);
        Reg::Xor(pregHis, pregNow, pregB32, pregB32);

        Reg::Gather(outputGatherIdx, hisIdxBuf, tmpIdx, pregHis);
        Reg::Adds(outputAddsIdx, tmpIdx, loopIndex, pregNow);

        Reg::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputIdxBuf + i * VF_B32_ELEMS, outputGatherIdx, pregB32);
    }
}

__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *tmpIdxBuf,
                                     __ubuf__ uint32_t *hisIdxBuf, uint32_t outputIdxStride, uint32_t tmpIdxStride,
                                     uint32_t hisIdxStride, uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2,
                                     uint32_t rowIdx3, uint32_t topK0, uint32_t topK1, uint32_t topK2, uint32_t topK3,
                                     uint32_t loopIndex, uint16_t vfLoop)
{
    FindRealIndexProcessRow(outputIdxBuf, tmpIdxBuf, hisIdxBuf + rowIdx0 * hisIdxStride, topK0, loopIndex, vfLoop);
    FindRealIndexProcessRow(outputIdxBuf + outputIdxStride, tmpIdxBuf + tmpIdxStride,
                            hisIdxBuf + rowIdx1 * hisIdxStride, topK1, loopIndex, vfLoop);
    FindRealIndexProcessRow(outputIdxBuf + 2U * outputIdxStride, tmpIdxBuf + 2U * tmpIdxStride,
                            hisIdxBuf + rowIdx2 * hisIdxStride, topK2, loopIndex, vfLoop);
    FindRealIndexProcessRow(outputIdxBuf + 3U * outputIdxStride, tmpIdxBuf + 3U * tmpIdxStride,
                            hisIdxBuf + rowIdx3 * hisIdxStride, topK3, loopIndex, vfLoop);
}

__simd_vf__ void IndicesAddOffsetVF(__ubuf__ uint32_t *indicesOutBuf, uint32_t outputIdxOffset, uint32_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> outIndices;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(outIndices, indicesOutBuf + i * VF_B32_ELEMS);
        Reg::Adds(outIndices, outIndices, outputIdxOffset, pregB32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(indicesOutBuf + i * VF_B32_ELEMS, outIndices, pregB32);
    }
}

/**
 * @brief SITopKVF 对loopM行输入数据各自进行topk算法，输出idx_tmp
 * @param tmpIdxLocal Temp阶段输出的TopKIndex;如果s2SeqLen < 8K作为最终输出 loopM * (Align(maxTopK,64) * 4B)
 * @param outputValueLocal 如果s2SeqLen > 8K并且是首轮输出Value loopM * (topK * 4B)
 * @param inputValueLocal 输入Value基地址，第m个compact行通过rowIdx映射到 inputValueLocal + rowIdx * offset
 * @param histogramsLocal 直方图 loopM * (256 * 4B)，第m行在 histogramsLocal + m * 256
 * @param idx0Local 目标桶第一个八位 loopM * (256 * 4B)
 * @param idx1Local 目标桶第二个八位 loopM * (256 * 4B)
 * @param idx2Local 目标桶第三个八位 loopM * (256 * 4B)
 * @param idx3Local 目标桶第四个八位 loopM * (256 * 4B)
 * @param nkValueLocal 存储next_k的值 loopM * (64 * 4B)
 * @param topkNum0-topkNum3 compact行对应的topK元素个数
 * @param validLen 每行有效元素个数:SICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256)
 * @param loopM 循环轮数(行数)，多少行输入数据拼成1轮做一次topk算法
 * @param offset 每行输入数据之间的偏移(元素个数)
 * @param rowIdx0-rowIdx3 compact行对应的真实mInnerIdx
 */
template <bool ISOUTVALUE>
__aicore__ inline void SiTopKVF(const LocalTensor<uint32_t> &tmpIdxLocal, const LocalTensor<uint32_t> &outputValueLocal,
                                const LocalTensor<uint32_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t validLen, uint32_t loopM, uint32_t offset, uint32_t rowIdx0, uint32_t rowIdx1,
                                uint32_t rowIdx2, uint32_t rowIdx3, uint32_t topkNum0, uint32_t topkNum1,
                                uint32_t topkNum2, uint32_t topkNum3)
{
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputValueBuf = (__ubuf__ uint32_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint16_t histogramsLoopNum = (validLen + VF_INPUT_CHUNK_ELEMS - 1U) / VF_INPUT_CHUNK_ELEMS;
    uint16_t inputLoopNum = (validLen + VF_B32_ELEMS - 1U) / VF_B32_ELEMS;

    uint32_t maxTopK = topkNum0;
    uint32_t curTopK = topkNum1;
    if (curTopK > maxTopK) {
        maxTopK = curTopK;
    }
    curTopK = topkNum2;
    if (curTopK > maxTopK) {
        maxTopK = curTopK;
    }
    curTopK = topkNum3;
    if (curTopK > maxTopK) {
        maxTopK = curTopK;
    }
    uint16_t topkLoopNum = (maxTopK + VF_B32_ELEMS - 1U) / VF_B32_ELEMS;
    // tmpIdxOffset 对齐 256，与外部 indicesOutLocal/hisValueLocal 行步长一致
    uint32_t tmpIdxOffset = (maxTopK + TOPK_ALIGN_ELEMS - 1U) & ~(TOPK_ALIGN_ELEMS - 1U);

    // find kth-value
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputValueBuf, histogramsLoopNum, offset, rowIdx0, rowIdx1, rowIdx2,
                                    rowIdx3);
    FindFirstTargetBinVFImpl(idx0Buf, nkValueBuf, histogramsBuf, validLen, topkNum0, topkNum1, topkNum2, topkNum3);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, histogramsLoopNum, offset, rowIdx0, rowIdx1,
                                     rowIdx2, rowIdx3);
    FindSecondTargetBinVFImpl(idx1Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, histogramsLoopNum, offset, rowIdx0,
                                    rowIdx1, rowIdx2, rowIdx3);
    FindThirdTargetBinVFImpl(idx2Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum, offset,
                                   rowIdx0, rowIdx1, rowIdx2, rowIdx3);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);

    // filter
    AscendC::Duplicate(tmpIdxLocal, (uint32_t)(0), loopM * tmpIdxOffset);
    FindIdxOutputVFImpl(tmpIdxBuf, inputValueBuf, nkValueBuf, inputLoopNum, offset, tmpIdxOffset, rowIdx0, rowIdx1,
                        rowIdx2, rowIdx3);

    if constexpr (ISOUTVALUE) {
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, topkLoopNum, offset, tmpIdxOffset, rowIdx0,
                              rowIdx1, rowIdx2, rowIdx3);
    }
}

/**
 * @brief 一次VF处理4行，通过tmpIdx gather出实际的TopK索引
 * @param outputIdxLocal 4个compact行的输出索引
 * @param tmpIdxLocal 4个compact行的本轮临时索引
 * @param hisIdxLocal 上一轮各实际行的全局索引
 * @param outputIdxStride/tmpIdxStride/hisIdxStride 对应buffer的行步长
 * @param rowIdx0-rowIdx3 compact行对应的实际行号
 * @param topK0-topK3 每个compact行的topK元素个数
 * @param loopBasicIdx 当前循环需要加上的基准Index
 */
__aicore__ inline void SiTopKGatherVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                      const LocalTensor<uint32_t> &tmpIdxLocal,
                                      const LocalTensor<uint32_t> &hisIdxLocal, uint32_t outputIdxStride,
                                      uint32_t tmpIdxStride, uint32_t hisIdxStride, uint32_t rowIdx0, uint32_t rowIdx1,
                                      uint32_t rowIdx2, uint32_t rowIdx3, uint32_t topK0, uint32_t topK1,
                                      uint32_t topK2, uint32_t topK3, uint32_t loopBasicIdx)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    uint32_t maxTopK = topK0;
    maxTopK = topK1 > maxTopK ? topK1 : maxTopK;
    maxTopK = topK2 > maxTopK ? topK2 : maxTopK;
    maxTopK = topK3 > maxTopK ? topK3 : maxTopK;
    uint16_t topkLoopNum32 = (maxTopK + 63U) >> 6U;

    FindRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, outputIdxStride, tmpIdxStride, hisIdxStride, rowIdx0,
                        rowIdx1, rowIdx2, rowIdx3, topK0, topK1, topK2, topK3, loopBasicIdx, topkLoopNum32);
}

__aicore__ inline void IndicesAddOffset(const LocalTensor<uint32_t> &indicesOutLocal, uint32_t outputIdxOffset,
                                        uint32_t topK)
{
    __ubuf__ uint32_t *indicesOutBuf = (__ubuf__ uint32_t *)indicesOutLocal.GetPhyAddr();
    uint16_t topkLoopNum32 = (topK + VF_B32_ELEMS - 1U) / VF_B32_ELEMS;
    IndicesAddOffsetVF(indicesOutBuf, outputIdxOffset, topkLoopNum32);
}
} // namespace SITopkb32gather
#endif
