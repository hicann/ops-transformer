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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    // 32bit 高16bit
    MicroAPI::RegTensor<uint32_t> vreg0U16;
    // 32bit 低16bit
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                           roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t> &)vreg0U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg2U16);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg0, pregB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg0, pregB8);
    }
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT,
                                                                        cout1U32Even, cout1U32Odd, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx0;

    MicroAPI::RegTensor<uint32_t> topKReg;
    MicroAPI::Duplicate(topKReg, topK);
    MicroAPI::RegTensor<uint32_t> validLenPlus1;
    MicroAPI::Duplicate(validLenPlus1, validLen + 1);
    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Sub(btmK, validLenPlus1, topKReg, pregB32);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx0;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx0, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                         pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundIdx0Buf, sqzIdx0, alignIdx0);
    }
    MicroAPI::StoreUnAlignPost(roundIdx0Buf, alignIdx0);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev0;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg0, idx0, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev0, idx0, (MicroAPI::RegTensor<uint32_t> &)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, roundHistBuf, idxPrev0, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                           roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t> &)vreg0U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg2U16);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (MicroAPI::RegTensor<uint8_t> &)idx0, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg1, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg1, pregEQ);
    }
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT,
                                                                        cout1U32Even, cout1U32Odd, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx1;

    MicroAPI::RegTensor<uint32_t> btmK1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK1, roundKValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx1;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK1, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx1, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                         pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundIdx1Buf, sqzIdx1, alignIdx1);
    }
    MicroAPI::StoreUnAlignPost(roundIdx1Buf, alignIdx1);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, roundIdx1Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev1;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg1, idx1, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev1, idx1, (MicroAPI::RegTensor<uint32_t> &)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev1, idxPrev1, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, roundHistBuf, idxPrev1, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg1);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK1, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, roundIdx1Buf);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                           roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t> &)vreg0U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg2U16);
        MicroAPI::DeInterleave(vreg3, vreg2, (MicroAPI::RegTensor<uint8_t> &)vreg1U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg3U16);

        MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t> &)idx0, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t> &)idx1, pregB8);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::And(pregEQ, pregEQ0, pregEQ1, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg2, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg2, pregEQ);
    }
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT,
                                                                        cout1U32Even, cout1U32Odd, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdx2Buf = idx2Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx2;

    MicroAPI::RegTensor<uint32_t> btmK2;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK2, roundKValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx2;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK2, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx2, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                         pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundIdx2Buf, sqzIdx2, alignIdx2);
    }
    MicroAPI::StoreUnAlignPost(roundIdx2Buf, alignIdx2);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx2;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx2, roundIdx2Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev2;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg2, idx2, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev2, idx2, (MicroAPI::RegTensor<uint32_t> &)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev2, idxPrev2, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, roundHistBuf, idxPrev2, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg2);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK2, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::RegTensor<uint32_t> idx2;

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx2Buf = idx2Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, roundIdx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, roundIdx1Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx2, roundIdx2Buf);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16,
                                                                           roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            vreg3U16, vreg2U16, roundInputBuf + (i * VF_INPUT_CHUNK_ELEMS) + HISTOGRAM_HALF_BIN_COUNT);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t> &)vreg0U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg2U16);
        MicroAPI::DeInterleave(vreg3, vreg2, (MicroAPI::RegTensor<uint8_t> &)vreg1U16,
                               (MicroAPI::RegTensor<uint8_t> &)vreg3U16);

        MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ2 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t> &)idx0, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t> &)idx1, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (MicroAPI::RegTensor<uint8_t> &)idx2, pregB8);

        MicroAPI::MaskReg pregEQ0And1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8);
        MicroAPI::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg3, pregEQAll);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg3, pregEQAll);
    }
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf + HISTOGRAM_HALF_BIN_COUNT,
                                                                        cout1U32Even, cout1U32Odd, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx0Buf = idx0Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx1Buf = idx1Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx2Buf = idx2Buf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdx3Buf = idx3Buf + rowSlot * HISTOGRAM_BIN_COUNT;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx3;

    MicroAPI::RegTensor<uint32_t> btmK3;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK3, roundKValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx3;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx3, (MicroAPI::RegTensor<uint32_t> &)idxC,
                                                                         pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundIdx3Buf, sqzIdx3, alignIdx3);
    }
    MicroAPI::StoreUnAlignPost(roundIdx3Buf, alignIdx3);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::RegTensor<uint32_t> idx2;
    MicroAPI::RegTensor<uint32_t> idx3;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx0, roundIdx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx1, roundIdx1Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx2, roundIdx2Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx3, roundIdx3Buf);

    MicroAPI::ShiftLefts(idx0, idx0, (int16_t)24, pregB32);
    MicroAPI::ShiftLefts(idx1, idx1, (int16_t)16, pregB32);
    MicroAPI::ShiftLefts(idx2, idx2, (int16_t)8, pregB32);

    MicroAPI::Add(idx0, idx0, idx1, pregB32);
    MicroAPI::Add(idx0, idx0, idx2, pregB32);
    MicroAPI::Add(idx0, idx0, idx3, pregB32);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(roundKValue, idx0, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundOutputIdxBuf = outputIdxBuf + rowSlot * tmpIdxOffset;
    __ubuf__ uint32_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, roundKValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, roundInputBuf + i * VF_B32_ELEMS);

        MicroAPI::MaskReg poutGT = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<uint32_t> sqzIdxOut;
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint32_t> &)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundOutputIdxBuf, sqzIdxOut,
                                                                                  alignIdx);
    }

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, roundInputBuf + i * VF_B32_ELEMS);

        MicroAPI::MaskReg poutEQ;

        MicroAPI::RegTensor<uint32_t> sqzIdxOut;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint32_t> &)idxC, poutEQ);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundOutputIdxBuf, sqzIdxOut,
                                                                                  alignIdx);
    }
    MicroAPI::StoreUnAlignPost(roundOutputIdxBuf, alignIdx);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    uint32_t outputOffset = tmpIdxOffset;

    __ubuf__ uint32_t *roundOutputValueBuf = outputValueBuf + rowSlot * outputOffset;
    __ubuf__ uint32_t *roundInputValueBuf = inputValueBuf + realRowIdx * inputOffset;
    __ubuf__ uint32_t *roundTmpIdxBuf = tmpIdxBuf + rowSlot * tmpIdxOffset;

    MicroAPI::RegTensor<uint32_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(tmpIdx, roundTmpIdxBuf + i * VF_B32_ELEMS);

        MicroAPI::Gather(outputValue, roundInputValueBuf, tmpIdx, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(roundOutputValueBuf + i * VF_B32_ELEMS,
                                                                       outputValue, pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg pregNow;
    MicroAPI::MaskReg pregHis;

    MicroAPI::RegTensor<uint32_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputGatherIdx;
    MicroAPI::RegTensor<uint32_t> outputAddsIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * VF_B32_ELEMS);

        MicroAPI::Compares<uint32_t, CMPMODE::GT>(pregNow, tmpIdx, topK - 1, pregB32);
        MicroAPI::Xor(pregHis, pregNow, pregB32, pregB32);

        MicroAPI::Gather(outputGatherIdx, hisIdxBuf, tmpIdx, pregHis);
        MicroAPI::Adds(outputAddsIdx, tmpIdx, loopIndex, pregNow);

        MicroAPI::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outputIdxBuf + i * VF_B32_ELEMS, outputGatherIdx,
                                                                       pregB32);
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
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> outIndices;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(outIndices, indicesOutBuf + i * VF_B32_ELEMS);
        MicroAPI::Adds(outIndices, outIndices, outputIdxOffset, pregB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(indicesOutBuf + i * VF_B32_ELEMS, outIndices,
                                                                       pregB32);
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
