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
 * \file vf_top_k_16_gather.h
 * \brief
 */

#ifndef STEM_INDEXER_VF_TOP_K_16_GATHER_H
#define STEM_INDEXER_VF_TOP_K_16_GATHER_H

namespace SITopkb16gather {
constexpr uint16_t HISTOGRAM_BIN_COUNT = 256U;
constexpr uint16_t VF_INPUT_CHUNK_ELEMS = 256U;
constexpr uint16_t VF_B16_ELEMS = 128U;
constexpr uint16_t VF_B32_ELEMS = 64U;

template <typename T>
__simd_callee__ inline void HistogramsHighProcessRow(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint16_t *inputBuf,
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

    MicroAPI::RegTensor<uint16_t> vregHigh;
    MicroAPI::RegTensor<uint16_t> vregLow;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint16_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh,
                                                                          roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, (MicroAPI::RegTensor<uint8_t> &)vregHigh,
                                                                   pregB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, (MicroAPI::RegTensor<uint8_t> &)vregHigh,
                                                                   pregB8);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf + VF_B16_ELEMS, cout1U32Even,
                                                                        cout1U32Odd, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsHighVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint16_t *inputBuf, uint16_t vfLoop,
                                      uint32_t offset, uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2,
                                      uint32_t rowIdx3)
{
    HistogramsHighProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 0, rowIdx0);
    HistogramsHighProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 1, rowIdx1);
    HistogramsHighProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 2, rowIdx2);
    HistogramsHighProcessRow<T>(histogramsBuf, inputBuf, vfLoop, offset, 3, rowIdx3);
}

__simd_callee__ inline void FindHighTargetBinStoreRow(__ubuf__ uint32_t *idxHighBuf, __ubuf__ uint32_t *histogramsBuf,
                                                      uint32_t topK, uint32_t validLen, uint32_t rowSlot)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdxHighBuf = idxHighBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    MicroAPI::MaskReg pregGE;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdxHigh;

    MicroAPI::RegTensor<uint32_t> topKReg;
    MicroAPI::Duplicate(topKReg, topK);
    MicroAPI::RegTensor<uint32_t> validLenPlus1;
    MicroAPI::Duplicate(validLenPlus1, validLen + 1);
    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Sub(btmK, validLenPlus1, topKReg, pregB32);

    MicroAPI::RegTensor<int32_t> idxC;
    MicroAPI::RegTensor<uint32_t> cout;
    MicroAPI::RegTensor<uint32_t> sqzIdxHigh;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::Arange(idxC, i * VF_B32_ELEMS);

        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);

        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxHigh,
                                                                         (MicroAPI::RegTensor<uint32_t> &)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundIdxHighBuf, sqzIdxHigh,
                                                                                  alignIdxHigh);
    }
    MicroAPI::StoreUnAlignPost(roundIdxHighBuf, alignIdxHigh);
}

__simd_callee__ inline void FindHighTargetBinFinishRow(__ubuf__ uint32_t *idxHighBuf, __ubuf__ uint32_t *nkValueBuf,
                                                       __ubuf__ uint32_t *histogramsBuf, uint32_t topK,
                                                       uint32_t validLen, uint32_t rowSlot)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint32_t *roundIdxHighBuf = idxHighBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundNkValueBuf = nkValueBuf + rowSlot * VF_B32_ELEMS;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;

    MicroAPI::RegTensor<uint32_t> topKReg;
    MicroAPI::Duplicate(topKReg, topK);
    MicroAPI::RegTensor<uint32_t> validLenPlus1;
    MicroAPI::Duplicate(validLenPlus1, validLen + 1);
    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Sub(btmK, validLenPlus1, topKReg, pregB32);

    MicroAPI::RegTensor<uint32_t> idxHigh;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idxHigh, roundIdxHighBuf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev0;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg0, idxHigh, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev0, idxHigh, (MicroAPI::RegTensor<uint32_t> &)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, roundHistBuf, idxPrev0, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(roundNkValueBuf, nextK, pregB32);
}

__simd_vf__ void FindHighTargetBinVFImpl(__ubuf__ uint32_t *idxHighBuf, __ubuf__ uint32_t *nkValueBuf,
                                         __ubuf__ uint32_t *histogramsBuf, uint32_t validLen, uint32_t topK0,
                                         uint32_t topK1, uint32_t topK2, uint32_t topK3)
{
    FindHighTargetBinStoreRow(idxHighBuf, histogramsBuf, topK0, validLen, 0);
    FindHighTargetBinStoreRow(idxHighBuf, histogramsBuf, topK1, validLen, 1);
    FindHighTargetBinStoreRow(idxHighBuf, histogramsBuf, topK2, validLen, 2);
    FindHighTargetBinStoreRow(idxHighBuf, histogramsBuf, topK3, validLen, 3);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    FindHighTargetBinFinishRow(idxHighBuf, nkValueBuf, histogramsBuf, topK0, validLen, 0);
    FindHighTargetBinFinishRow(idxHighBuf, nkValueBuf, histogramsBuf, topK1, validLen, 1);
    FindHighTargetBinFinishRow(idxHighBuf, nkValueBuf, histogramsBuf, topK2, validLen, 2);
    FindHighTargetBinFinishRow(idxHighBuf, nkValueBuf, histogramsBuf, topK3, validLen, 3);
}

template <typename T>
__simd_callee__ inline void HistogramsLowProcessRow(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint16_t *inputBuf,
                                                    __ubuf__ uint32_t *idxHighBuf, uint16_t vfLoop, uint32_t offset,
                                                    uint32_t rowSlot, uint32_t realRowIdx)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg pregEQ;

    // 计算直方图0-127 128-255
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idxHigh;

    MicroAPI::RegTensor<uint16_t> vregHigh;
    MicroAPI::RegTensor<uint16_t> vregLow;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    __ubuf__ uint16_t *roundInputBuf = inputBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundHistBuf = histogramsBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    __ubuf__ uint32_t *roundIdxHighBuf = idxHighBuf + rowSlot * HISTOGRAM_BIN_COUNT;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idxHigh, roundIdxHighBuf);

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh,
                                                                          roundInputBuf + i * VF_INPUT_CHUNK_ELEMS);

        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ, (MicroAPI::RegTensor<uint8_t> &)vregHigh,
                                                (MicroAPI::RegTensor<uint8_t> &)idxHigh, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, (MicroAPI::RegTensor<uint8_t> &)vregLow,
                                                                   pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, (MicroAPI::RegTensor<uint8_t> &)vregLow,
                                                                   pregEQ);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf, cout0U32Even, cout0U32Odd,
                                                                        pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(roundHistBuf + VF_B16_ELEMS, cout1U32Even,
                                                                        cout1U32Odd, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsLowVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint16_t *inputBuf,
                                     __ubuf__ uint32_t *idxHighBuf, uint16_t vfLoop, uint32_t offset, uint32_t rowIdx0,
                                     uint32_t rowIdx1, uint32_t rowIdx2, uint32_t rowIdx3)
{
    HistogramsLowProcessRow<T>(histogramsBuf, inputBuf, idxHighBuf, vfLoop, offset, 0, rowIdx0);
    HistogramsLowProcessRow<T>(histogramsBuf, inputBuf, idxHighBuf, vfLoop, offset, 1, rowIdx1);
    HistogramsLowProcessRow<T>(histogramsBuf, inputBuf, idxHighBuf, vfLoop, offset, 2, rowIdx2);
    HistogramsLowProcessRow<T>(histogramsBuf, inputBuf, idxHighBuf, vfLoop, offset, 3, rowIdx3);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf,
                               __ubuf__ uint32_t *idxHighBuf, __ubuf__ uint32_t *idxLowBuf, uint16_t loopM)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    for (uint16_t m = 0; m < loopM; ++m) {
        __ubuf__ uint32_t *roundKValue = kValue + m * VF_B32_ELEMS;
        __ubuf__ uint32_t *roundHistBuf = histogramsBuf + m * HISTOGRAM_BIN_COUNT;
        __ubuf__ uint32_t *roundIdxLowBuf = idxLowBuf + m * HISTOGRAM_BIN_COUNT;

        MicroAPI::MaskReg pregGE;

        MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        MicroAPI::UnalignRegForStore alignIdxLow;

        MicroAPI::RegTensor<uint32_t> btmK;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK, roundKValue);

        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdxLow;

        for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
            MicroAPI::Arange(idxC, i * VF_B32_ELEMS);

            MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, roundHistBuf + i * VF_B32_ELEMS);

            MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);

            MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
                sqzIdxLow, (MicroAPI::RegTensor<uint32_t> &)idxC, pregGE);
            MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundIdxLowBuf, sqzIdxLow,
                                                                                      alignIdxLow);
        }
        MicroAPI::StoreUnAlignPost(roundIdxLowBuf, alignIdxLow);
    }

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint16_t> idxTmp;
    MicroAPI::Duplicate(idxTmp, 0xff00);

    for (uint16_t m = 0; m < loopM; ++m) {
        __ubuf__ uint32_t *roundKValue = kValue + m * VF_B32_ELEMS;
        __ubuf__ uint32_t *roundIdxHighBuf = idxHighBuf + m * HISTOGRAM_BIN_COUNT;
        __ubuf__ uint32_t *roundIdxLowBuf = idxLowBuf + m * HISTOGRAM_BIN_COUNT;

        MicroAPI::RegTensor<uint32_t> idxHigh;
        MicroAPI::RegTensor<uint32_t> idxLow;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idxHigh, roundIdxHighBuf);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(idxLow, roundIdxLowBuf);

        MicroAPI::And(idxHigh, idxHigh, (MicroAPI::RegTensor<uint32_t> &)idxTmp, pregB32);

        MicroAPI::RegTensor<uint32_t> idxK;
        MicroAPI::Add(idxK, idxHigh, idxLow, pregB16);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM_B16>(roundKValue, idxK, pregB32);
    }
}

__simd_callee__ inline void FindIdxOutputProcessRow(__ubuf__ uint16_t *outputIdxBuf, __ubuf__ uint16_t *inputValueBuf,
                                                    __ubuf__ uint32_t *kValue, uint16_t vfLoop, uint32_t offset,
                                                    uint32_t tmpIdxOffset, uint32_t rowSlot, uint32_t realRowIdx)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint16_t *roundOutputIdxBuf = outputIdxBuf + rowSlot * tmpIdxOffset;
    __ubuf__ uint16_t *roundInputBuf = inputValueBuf + realRowIdx * offset;
    __ubuf__ uint32_t *roundKValue = kValue + rowSlot * VF_B32_ELEMS;

    MicroAPI::MaskReg poutGT;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(kthValue, roundKValue);

    MicroAPI::RegTensor<uint16_t> vregInput;
    MicroAPI::RegTensor<int16_t> idxC;
    MicroAPI::RegTensor<uint16_t> sqzIdxOut;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::Arange(idxC, i * VF_B16_ELEMS);

        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, roundInputBuf + i * VF_B16_ELEMS);

        MicroAPI::Compare<uint16_t, CMPMODE::GT>(poutGT, vregInput, (MicroAPI::RegTensor<uint16_t> &)kthValue, pregB16);

        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint16_t> &)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundOutputIdxBuf, sqzIdxOut,
                                                                                  alignIdx);
    }

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::Arange(idxC, i * VF_B16_ELEMS);

        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, roundInputBuf + i * VF_B16_ELEMS);

        MicroAPI::MaskReg poutEQ;
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(poutEQ, vregInput, (MicroAPI::RegTensor<uint16_t> &)kthValue, pregB16);

        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint16_t> &)idxC, poutEQ);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(roundOutputIdxBuf, sqzIdxOut,
                                                                                  alignIdx);
    }
    MicroAPI::StoreUnAlignPost(roundOutputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxOutputVFImpl(__ubuf__ uint16_t *outputIdxBuf, __ubuf__ uint16_t *inputValueBuf,
                                     __ubuf__ uint32_t *kValue, uint16_t vfLoop, uint32_t offset, uint32_t tmpIdxOffset,
                                     uint32_t rowIdx0, uint32_t rowIdx1, uint32_t rowIdx2, uint32_t rowIdx3)
{
    FindIdxOutputProcessRow(outputIdxBuf, inputValueBuf, kValue, vfLoop, offset, tmpIdxOffset, 0, rowIdx0);
    FindIdxOutputProcessRow(outputIdxBuf, inputValueBuf, kValue, vfLoop, offset, tmpIdxOffset, 1, rowIdx1);
    FindIdxOutputProcessRow(outputIdxBuf, inputValueBuf, kValue, vfLoop, offset, tmpIdxOffset, 2, rowIdx2);
    FindIdxOutputProcessRow(outputIdxBuf, inputValueBuf, kValue, vfLoop, offset, tmpIdxOffset, 3, rowIdx3);
}

/**
    输出最终的Value
 */
__simd_callee__ inline void FindValueOutputProcessRow(__ubuf__ uint16_t *outputValueBuf,
                                                      __ubuf__ uint16_t *inputValueBuf, __ubuf__ uint16_t *tmpIdxBuf,
                                                      uint16_t vfLoop, uint32_t inputOffset, uint32_t tmpIdxOffset,
                                                      uint32_t outputValueOffset, uint32_t rowSlot, uint32_t realRowIdx)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    __ubuf__ uint16_t *roundOutputValueBuf = outputValueBuf + rowSlot * outputValueOffset;
    __ubuf__ uint16_t *roundInputValueBuf = inputValueBuf + realRowIdx * inputOffset;
    __ubuf__ uint16_t *roundTmpIdxBuf = tmpIdxBuf + rowSlot * tmpIdxOffset;

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint16_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(tmpIdx, roundTmpIdxBuf + i * VF_B16_ELEMS);

        MicroAPI::Gather(outputValue, roundInputValueBuf, tmpIdx, pregB16);

        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(roundOutputValueBuf + i * VF_B16_ELEMS,
                                                                       outputValue, pregB16);
    }
}

__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint16_t *outputValueBuf, __ubuf__ uint16_t *inputValueBuf,
                                       __ubuf__ uint16_t *tmpIdxBuf, uint16_t vfLoop, uint32_t inputOffset,
                                       uint32_t tmpIdxOffset, uint32_t outputValueOffset, uint32_t rowIdx0,
                                       uint32_t rowIdx1, uint32_t rowIdx2, uint32_t rowIdx3)
{
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset,
                              outputValueOffset, 0, rowIdx0);
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset,
                              outputValueOffset, 1, rowIdx1);
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset,
                              outputValueOffset, 2, rowIdx2);
    FindValueOutputProcessRow(outputValueBuf, inputValueBuf, tmpIdxBuf, vfLoop, inputOffset, tmpIdxOffset,
                              outputValueOffset, 3, rowIdx3);
}

/**
    输出最终的Idx
 */
__simd_callee__ inline void FindRealIndexProcessRow(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint16_t *tmpIdxBuf,
                                                    __ubuf__ uint32_t *hisIdxBuf, uint32_t topK, uint32_t loopIndex,
                                                    uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg pregNow;
    MicroAPI::MaskReg pregHis;

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputGatherIdx;
    MicroAPI::RegTensor<uint32_t> outputAddsIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * VF_B32_ELEMS);

        MicroAPI::Compares<uint32_t, CMPMODE::GT>(pregNow, (MicroAPI::RegTensor<uint32_t> &)tmpIdx, topK - 1, pregB32);
        MicroAPI::Xor(pregHis, pregNow, pregB32, pregB32);

        MicroAPI::Gather(outputGatherIdx, hisIdxBuf, (MicroAPI::RegTensor<uint32_t> &)tmpIdx, pregHis);
        MicroAPI::Adds(outputAddsIdx, (MicroAPI::RegTensor<uint32_t> &)tmpIdx, loopIndex, pregNow);

        MicroAPI::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outputIdxBuf + i * VF_B32_ELEMS, outputGatherIdx,
                                                                       pregB32);
    }
}

__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint16_t *tmpIdxBuf,
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

/**
 * @brief SiTopKVF 对loopM行输入数据各自进行topk算法，输出idx_tmp
 * @param tmpIdxLocal Temp阶段输出的TopKIndex;如果s2SeqLen < 16K作为最终输出 loopM * (Align(maxTopK,128) * 2B)
 * @param outputValueLocal 如果s2SeqLen > 16K并且是首轮输出Value loopM * (topK * 2B)
 * @param inputValueLocal 输入Value loopM * (validLen * 2B)，第m行在 inputValueLocal + m * offset
 * @param histogramsLocal 直方图 loopM * (256 * 4B)，第m行在 histogramsLocal + m * 256
 * @param idxHighLocal 目标桶高八位 loopM * (256 * 4B)
 * @param idxLowLocal 目标桶低八位 loopM * (256 * 4B)
 * @param nkValueLocal 存储next_k的值 loopM * (64 * 4B)
 * @param validLen 每行有效元素个数:SICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256)
 * @param loopM 循环轮数(行数)，多少行输入数据拼成1轮做一次topk算法
 * @param offset 每行输入数据之间的偏移(元素个数)
 * @param rowIdx0-rowIdx3 compact行对应的真实mInnerIdx
 * @param topkNum0-topkNum3 compact行对应的topK元素个数
 * @param tmpIdxStride tmpIdxLocal 每行之间的偏移(元素个数)
 * @param outputValueStride outputValueLocal 每行之间的偏移(元素个数)
 */
template <bool ISOUTVALUE> // 是否输出VALUE
__aicore__ inline void SiTopKVF(const LocalTensor<uint16_t> &tmpIdxLocal, const LocalTensor<uint16_t> &outputValueLocal,
                                const LocalTensor<uint16_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idxHighLocal,
                                const LocalTensor<uint32_t> &idxLowLocal, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t validLen, uint32_t loopM, uint32_t offset, uint32_t rowIdx0, uint32_t rowIdx1,
                                uint32_t rowIdx2, uint32_t rowIdx3, uint32_t topkNum0, uint32_t topkNum1,
                                uint32_t topkNum2, uint32_t topkNum3, uint32_t tmpIdxStride, uint32_t outputValueStride)
{
    __ubuf__ uint16_t *tmpIdxBuf = (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t *outputValueBuf = (__ubuf__ uint16_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t *inputValueBuf = (__ubuf__ uint16_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idxHighBuf = (__ubuf__ uint32_t *)idxHighLocal.GetPhyAddr();
    __ubuf__ uint32_t *idxLowBuf = (__ubuf__ uint32_t *)idxLowLocal.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint16_t histogramsLoopNum = (validLen + VF_INPUT_CHUNK_ELEMS - 1U) / VF_INPUT_CHUNK_ELEMS;
    uint16_t inputLoopNum = (validLen + VF_B16_ELEMS - 1U) / VF_B16_ELEMS;

    uint32_t maxTopK = topkNum0;
    if (topkNum1 > maxTopK) {
        maxTopK = topkNum1;
    }
    if (topkNum2 > maxTopK) {
        maxTopK = topkNum2;
    }
    if (topkNum3 > maxTopK) {
        maxTopK = topkNum3;
    }
    uint16_t topkLoopNum16 = (maxTopK + VF_B16_ELEMS - 1U) / VF_B16_ELEMS;
    // tmpIdx/outputValue 的行距由外部传入，确保 VF 内部写入和外部后处理读取使用同一个 stride。
    uint32_t tmpIdxOffset = tmpIdxStride;

    // find kth-value
    HistogramsHighVFImpl<uint16_t>(histogramsBuf, inputValueBuf, histogramsLoopNum, offset, rowIdx0, rowIdx1, rowIdx2,
                                   rowIdx3);
    FindHighTargetBinVFImpl(idxHighBuf, nkValueBuf, histogramsBuf, validLen, topkNum0, topkNum1, topkNum2, topkNum3);

    HistogramsLowVFImpl<uint16_t>(histogramsBuf, inputValueBuf, idxHighBuf, histogramsLoopNum, offset, rowIdx0, rowIdx1,
                                  rowIdx2, rowIdx3);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idxHighBuf, idxLowBuf, (uint16_t)loopM);

    // filter
    AscendC::Duplicate(tmpIdxLocal, (uint16_t)(0), loopM * tmpIdxOffset);
    // 先连续写入所有大于 kth-value 的 idx，再追加等于 kth-value 的 idx。
    // 这里必须复用同一个 UnalignRegForStore 状态，避免 EQ 阶段从行首重新写导致 GT 结果被覆盖。
    FindIdxOutputVFImpl(tmpIdxBuf, inputValueBuf, nkValueBuf, inputLoopNum, offset, tmpIdxOffset, rowIdx0, rowIdx1,
                        rowIdx2, rowIdx3);

    // 是否输出Value
    if constexpr (ISOUTVALUE) {
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, topkLoopNum16, offset, tmpIdxOffset,
                              outputValueStride, rowIdx0, rowIdx1, rowIdx2, rowIdx3);
    }
}

/**
    LD:输出最终的Idx
*/
__simd_vf__ void FindLDRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint16_t *tmpIdxBuf,
                                       __ubuf__ uint32_t *hisIdxBuf, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * VF_B32_ELEMS);

        MicroAPI::Gather(outputIdx, hisIdxBuf, (MicroAPI::RegTensor<uint32_t> &)tmpIdx, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outputIdxBuf + i * VF_B32_ELEMS, outputIdx,
                                                                       pregB32);
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
                                      const LocalTensor<uint16_t> &tmpIdxLocal,
                                      const LocalTensor<uint32_t> &hisIdxLocal, uint32_t outputIdxStride,
                                      uint32_t tmpIdxStride, uint32_t hisIdxStride, uint32_t rowIdx0, uint32_t rowIdx1,
                                      uint32_t rowIdx2, uint32_t rowIdx3, uint32_t topK0, uint32_t topK1,
                                      uint32_t topK2, uint32_t topK3, uint32_t loopBasicIdx)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t *tmpIdxBuf = (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    uint32_t maxTopK = topK0;
    maxTopK = topK1 > maxTopK ? topK1 : maxTopK;
    maxTopK = topK2 > maxTopK ? topK2 : maxTopK;
    maxTopK = topK3 > maxTopK ? topK3 : maxTopK;
    uint16_t topkLoopNum32 = (maxTopK + 63U) >> 6U;

    FindRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, outputIdxStride, tmpIdxStride, hisIdxStride, rowIdx0,
                        rowIdx1, rowIdx2, rowIdx3, topK0, topK1, topK2, topK3, loopBasicIdx, topkLoopNum32);
}

/**
    LD:gather最终的Idx
*/
__aicore__ inline void SiTopKLDGatherVF(const LocalTensor<uint32_t> &outputIdxLocal, // 输出Idx topK * 2B
                                        const LocalTensor<uint16_t> &tmpIdxLocal,    // 本轮tmpIdx输入 validLen * 2B
                                        const LocalTensor<uint32_t> &hisIdxLocal,    // 上一轮Idx输入 topK * 4B
                                        uint32_t topK)                               // topK元素个数
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t *tmpIdxBuf = (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    uint16_t topkLoopNum16 = (topK + VF_B16_ELEMS - 1U) / VF_B16_ELEMS;
    uint16_t topkLoopNum32 = (topK + VF_B32_ELEMS - 1U) / VF_B32_ELEMS;

    FindLDRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topkLoopNum32);
}

} // namespace SITopkb16gather
#endif
