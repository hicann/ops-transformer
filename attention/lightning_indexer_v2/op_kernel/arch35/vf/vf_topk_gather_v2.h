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
 * \file vf_topk_gather_v2.h
 * \brief
 */

#ifndef VF_TOPK_GATHER_V2_H
#define VF_TOPK_GATHER_V2_H

#include "../common/vf/vf_topk_base_v2.h"

namespace liV2Topkb32gather {
using liV2TopkCommon::IndicesAddOffset;

template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

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

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg0,
                                                                                                          pregB8);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg0,
                                                                                                          pregB8);
    }
    liV2TopkCommon::StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, uint32_t bottomK)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::RegTensor<uint32_t> btmK;
    Reg::Duplicate(btmK, bottomK);

    liV2TopkCommon::FindTargetBinAndUpdateNextK(idx0Buf, nkValueBuf, histogramsBuf, btmK, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                        __ubuf__ uint32_t *idx0Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> idx0;
    // 0x000000fc -> 0xfcfcfcfc
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

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg1,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg1,
                                                                                                          pregEQ);
    }

    liV2TopkCommon::StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

// kValue新的bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *nkValueBuf,
                                           __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::RegTensor<uint32_t> btmK1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK1, kValue);

    liV2TopkCommon::FindTargetBinAndUpdateNextK(idx1Buf, nkValueBuf, histogramsBuf, btmK1, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                       __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

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

    liV2TopkCommon::StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::RegTensor<uint32_t> btmK2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK2, kValue);

    liV2TopkCommon::FindTargetBinAndUpdateNextK(idx2Buf, nkValueBuf, histogramsBuf, btmK2, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                      __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                      __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::RegTensor<uint32_t> idx2;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);

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

    liV2TopkCommon::StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *idx0Buf,
                               __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *idx3Buf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

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
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx3, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
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

    Reg::ShiftLefts(idx0, idx0, (int16_t)24, pregB32);
    Reg::ShiftLefts(idx1, idx1, (int16_t)16, pregB32);
    Reg::ShiftLefts(idx2, idx2, (int16_t)8, pregB32);

    // ADD
    Reg::Add(idx0, idx0, idx1, pregB32);
    Reg::Add(idx0, idx0, idx2, pregB32);
    Reg::Add(idx0, idx0, idx3, pregB32);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(kValue, idx0, pregB32);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

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
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

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

        Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutEQ);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出最终的Value
 */
__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputValueBuf,
                                       __ubuf__ uint32_t *tmpIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Gather(outputValue, inputValueBuf, tmpIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputValueBuf + i * 64, outputValue, pregB32);
    }
}

/**
    输出最终的Idx
 */
__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *tmpIdxBuf,
                                     __ubuf__ uint32_t *hisIdxBuf, uint32_t topK, uint32_t loopIndex, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregNow;
    Reg::MaskReg pregHis;

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputGatherIdx;
    Reg::RegTensor<uint32_t> outputAddsIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Compares<uint32_t, CMPMODE::GT>(pregNow, tmpIdx, topK - 1, pregB32);
        Reg::Xor(pregHis, pregNow, pregB32, pregB32);

        Reg::Gather(outputGatherIdx, hisIdxBuf, tmpIdx, pregHis);
        Reg::Adds(outputAddsIdx, tmpIdx, loopIndex, pregNow);

        Reg::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputGatherIdx, pregB32);
    }
}
/**
    LD:输出最终的Idx
*/
__simd_vf__ void FindLDRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *tmpIdxBuf,
                                       __ubuf__ uint32_t *hisIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Gather(outputIdx, hisIdxBuf, tmpIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputIdx, pregB32);
    }
}

/**
 * @brief LiTopKVF 对一个validLen的输入进行topk算法，输出idx_tmp
 * @param tmpIdxLocal Temp阶段输出的TopKIndex;如果s2SeqLen < 8K作为最终输出 validLen * 4B
 * @param outputValueLocal 如果s2SeqLen > 8K并且是首轮输出Value topK * 4B
 * @param inputValueLocal 输入Value validLen * 4B
 * @param histogramsLocal 直方图 256 * 4B
 * @param idx0Local 目标桶第一个八位 256 * 4B
 * @param idx1Local 目标桶第二个八位 256 * 4B
 * @param idx2Local 目标桶第三个八位 256 * 4B
 * @param idx3Local 目标桶第四个八位 256 * 4B
 * @param nkValueLocal 存储next_k的值 64 * 4B
 * @param topK topK元素
 * @param validLen 有效元素个数:QLICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256)
 */
template <bool ISOUTVALUE>
__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &tmpIdxLocal, const LocalTensor<uint32_t> &outputValueLocal,
                                const LocalTensor<uint32_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t validLen)
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

    uint32_t bottomK = validLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (validLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (validLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = (topK + 64 - 1) / 64;

    // find kth-value
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputValueBuf, histogramsLoopNum, flag);
    FindFirstTargetBinVFImpl(idx0Buf, nkValueBuf, histogramsBuf, bottomK);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, histogramsLoopNum, flag);
    FindSecondTargetBinVFImpl(idx1Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, histogramsLoopNum, flag);
    FindThirdTargetBinVFImpl(idx2Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);

    // filter
    int32_t count = LIV2Common::Align(topK, (uint32_t)64) - topK / 64 * 64;
    AscendC::Duplicate(tmpIdxLocal[topK / 64 * 64], (uint32_t)(0), count);
    // 输出大于k-value的值idx
    FindIdxGTOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // 输出等于k-value的值idx
    FindIdxEQOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);

    if constexpr (ISOUTVALUE) {
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, topkLoopNum);
    }
}

/**
 * @brief 通过idx_tmp gather出实际的TopKIndex，s2SeqLen > 8K才会执行
 * @param outputIdxLocal 输出Idx 有效:topK * 4B
 * @param outputValueLocal 输出Value topK * 4B(以后需要输出实际value使用)
 * @param inputValueLocal 输入Value validLen * 4B
 * @param tmpIdxLocal 本轮tmpIdx输入 validLen * 4B (0 ~ validLen - 1)
 * @param hisIdxLocal 上一轮实际Idx输入 有效:topK * 4B
 * @param topK topK元素个数
 * @param loopBasicIdx 当前循环需要加上得基准Index
 * @param validLen 有效元素个数
 */
__aicore__ inline void LiTopKGatherVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                      const LocalTensor<uint32_t> &outputValueLocal,
                                      const LocalTensor<uint32_t> &inputValueLocal,
                                      const LocalTensor<uint32_t> &tmpIdxLocal,
                                      const LocalTensor<uint32_t> &hisIdxLocal, uint32_t topK, uint32_t loopBasicIdx,
                                      uint32_t validLen)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputValueBuf = (__ubuf__ uint32_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;

    FindRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topK, loopBasicIdx, topkLoopNum32);
}

/**
    LD:gather最终的Idx
*/
__aicore__ inline void LiTopKLDGatherVF(const LocalTensor<uint32_t> &outputIdxLocal, // 输出Idx topK * 2B
                                        const LocalTensor<uint32_t> &tmpIdxLocal,    // 本轮tmpIdx输入 validLen * 2B
                                        const LocalTensor<uint32_t> &hisIdxLocal,    // 上一轮Idx输入 topK * 4B
                                        uint32_t topK)                               // topK元素个数
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;

    FindLDRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topkLoopNum32);
}

} // namespace liV2Topkb32gather
#endif
