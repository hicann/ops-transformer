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
 * \file vf_topk.h
 * \brief
 */

#ifndef VF_TOP_K_H
#define VF_TOP_K_H

#include "../../../../lightning_indexer_v2/op_kernel/arch35/common/vf/vf_topk_base_v2.h"

namespace topkb32 {
using liV2TopkCommon::StoreHistogramResult;

template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t *qliV2HistogramOutput, __ubuf__ uint32_t *qliV2HistogramInput,
                                       uint16_t qliV2LoopCount, bool qliV2Init)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    Reg::RegTensor<uint16_t> qliV2Histogram0;
    Reg::RegTensor<uint16_t> qliV2Histogram1;
    Reg::Duplicate(qliV2Histogram0, 0);
    Reg::Duplicate(qliV2Histogram1, 0);

    Reg::RegTensor<uint8_t> qliV2ByteReg0;
    Reg::RegTensor<uint8_t> qliV2ByteReg1;
    Reg::RegTensor<uint8_t> qliV2ByteReg2;
    Reg::RegTensor<uint8_t> qliV2ByteReg3;

    // 32bit 高16bit
    Reg::RegTensor<uint32_t> qliV2PackedReg0;
    // 32bit 低16bit
    Reg::RegTensor<uint32_t> qliV2PackedReg1;
    Reg::RegTensor<uint32_t> qliV2PackedReg2;
    Reg::RegTensor<uint32_t> qliV2PackedReg3;

    for (uint16_t qliV2LoopIndex = 0; qliV2LoopIndex < qliV2LoopCount; ++qliV2LoopIndex) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg1, qliV2PackedReg0,
                                                                 qliV2HistogramInput + qliV2LoopIndex * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg3, qliV2PackedReg2,
                                                                 qliV2HistogramInput + (qliV2LoopIndex * 256) + 128);

        Reg::DeInterleave(qliV2ByteReg1, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2PackedReg0,
                          (Reg::RegTensor<uint8_t> &)qliV2PackedReg2);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram0, qliV2ByteReg0, qliV2MaskB8);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram1, qliV2ByteReg0, qliV2MaskB8);
    }

    StoreHistogramResult(qliV2HistogramOutput, qliV2Histogram0, qliV2Histogram1, qliV2MaskB16, qliV2MaskB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t *qliV2Index0Buffer, __ubuf__ uint32_t *qliV2NkValueBuf,
                                          __ubuf__ uint32_t *qliV2HistogramOutput, uint32_t qliV2BottomK)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> qliV2BottomKReg;
    Reg::Duplicate(qliV2BottomKReg, qliV2BottomK);

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    liV2TopkCommon::FindTargetBinAndUpdateNextK(qliV2Index0Buffer, qliV2NkValueBuf, qliV2HistogramOutput,
                                                qliV2BottomKReg, qliV2MaskB32);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t *qliV2HistogramOutput, __ubuf__ uint32_t *qliV2HistogramInput,
                                        __ubuf__ uint32_t *qliV2Index0Buffer, uint16_t qliV2LoopCount, bool qliV2Init)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> qliV2Histogram0;
    Reg::RegTensor<uint16_t> qliV2Histogram1;
    Reg::Duplicate(qliV2Histogram0, 0);
    Reg::Duplicate(qliV2Histogram1, 0);

    Reg::RegTensor<uint32_t> qliV2Index0;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(qliV2Index0, qliV2Index0Buffer);

    Reg::RegTensor<uint32_t> qliV2PackedReg0;
    Reg::RegTensor<uint32_t> qliV2PackedReg1;
    Reg::RegTensor<uint32_t> qliV2PackedReg2;
    Reg::RegTensor<uint32_t> qliV2PackedReg3;

    Reg::RegTensor<uint8_t> qliV2ByteReg0;
    Reg::RegTensor<uint8_t> qliV2ByteReg1;
    Reg::RegTensor<uint8_t> qliV2ByteReg2;
    Reg::RegTensor<uint8_t> qliV2ByteReg3;

    for (uint16_t qliV2LoopIndex = 0; qliV2LoopIndex < qliV2LoopCount; ++qliV2LoopIndex) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg1, qliV2PackedReg0,
                                                                 qliV2HistogramInput + qliV2LoopIndex * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg3, qliV2PackedReg2,
                                                                 qliV2HistogramInput + (qliV2LoopIndex * 256) + 128);

        Reg::DeInterleave(qliV2ByteReg1, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2PackedReg0,
                          (Reg::RegTensor<uint8_t> &)qliV2PackedReg2);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2Index0, qliV2MaskB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram0, qliV2ByteReg1, pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram1, qliV2ByteReg1, pregEQ);
    }

    StoreHistogramResult(qliV2HistogramOutput, qliV2Histogram0, qliV2Histogram1, qliV2MaskB16, qliV2MaskB32);
}

// kValue新的bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t *qliV2Index1Buffer, __ubuf__ uint32_t *qliV2NkValueBuf,
                                           __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *qliV2HistogramOutput)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> btmK1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK1, kValue);

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    liV2TopkCommon::FindTargetBinAndUpdateNextK(qliV2Index1Buffer, qliV2NkValueBuf, qliV2HistogramOutput, btmK1,
                                                qliV2MaskB32);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t *qliV2HistogramOutput, __ubuf__ uint32_t *qliV2HistogramInput,
                                       __ubuf__ uint32_t *qliV2Index0Buffer, __ubuf__ uint32_t *qliV2Index1Buffer,
                                       uint16_t qliV2LoopCount, bool qliV2Init)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> qliV2Histogram0;
    Reg::RegTensor<uint16_t> qliV2Histogram1;
    Reg::Duplicate(qliV2Histogram0, 0);
    Reg::Duplicate(qliV2Histogram1, 0);

    Reg::RegTensor<uint32_t> qliV2Index0;
    Reg::RegTensor<uint32_t> qliV2Index1;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(qliV2Index0, qliV2Index0Buffer);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(qliV2Index1, qliV2Index1Buffer);

    Reg::RegTensor<uint8_t> qliV2ByteReg0;
    Reg::RegTensor<uint8_t> qliV2ByteReg1;
    Reg::RegTensor<uint8_t> qliV2ByteReg2;
    Reg::RegTensor<uint8_t> qliV2ByteReg3;

    Reg::RegTensor<uint32_t> qliV2PackedReg0;
    Reg::RegTensor<uint32_t> qliV2PackedReg1;
    Reg::RegTensor<uint32_t> qliV2PackedReg2;
    Reg::RegTensor<uint32_t> qliV2PackedReg3;

    for (uint16_t qliV2LoopIndex = 0; qliV2LoopIndex < qliV2LoopCount; ++qliV2LoopIndex) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg1, qliV2PackedReg0,
                                                                 qliV2HistogramInput + qliV2LoopIndex * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg3, qliV2PackedReg2,
                                                                 qliV2HistogramInput + (qliV2LoopIndex * 256) + 128);

        Reg::DeInterleave(qliV2ByteReg1, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2PackedReg0,
                          (Reg::RegTensor<uint8_t> &)qliV2PackedReg2);
        Reg::DeInterleave(qliV2ByteReg3, qliV2ByteReg2, (Reg::RegTensor<uint8_t> &)qliV2PackedReg1,
                          (Reg::RegTensor<uint8_t> &)qliV2PackedReg3);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2Index0, qliV2MaskB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, qliV2ByteReg1, (Reg::RegTensor<uint8_t> &)qliV2Index1, qliV2MaskB8);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ, pregEQ0, pregEQ1, qliV2MaskB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram0, qliV2ByteReg2, pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram1, qliV2ByteReg2, pregEQ);
    }

    StoreHistogramResult(qliV2HistogramOutput, qliV2Histogram0, qliV2Histogram1, qliV2MaskB16, qliV2MaskB32);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t *qliV2Idx2Buf, __ubuf__ uint32_t *qliV2NkValueBuf,
                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *qliV2HistogramOutput)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> btmK2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK2, kValue);

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    liV2TopkCommon::FindTargetBinAndUpdateNextK(qliV2Idx2Buf, qliV2NkValueBuf, qliV2HistogramOutput, btmK2,
                                                qliV2MaskB32);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t *qliV2HistogramOutput, __ubuf__ uint32_t *qliV2HistogramInput,
                                      __ubuf__ uint32_t *qliV2Index0Buffer, __ubuf__ uint32_t *qliV2Index1Buffer,
                                      __ubuf__ uint32_t *qliV2Idx2Buf, uint16_t qliV2LoopCount, bool qliV2Init)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg qliV2MaskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> qliV2Index0;
    Reg::RegTensor<uint32_t> qliV2Index1;
    Reg::RegTensor<uint32_t> idx2;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(qliV2Index0, qliV2Index0Buffer);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(qliV2Index1, qliV2Index1Buffer);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, qliV2Idx2Buf);

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> qliV2Histogram0;
    Reg::RegTensor<uint16_t> qliV2Histogram1;
    Reg::Duplicate(qliV2Histogram0, 0);
    Reg::Duplicate(qliV2Histogram1, 0);

    Reg::RegTensor<uint32_t> qliV2PackedReg0;
    Reg::RegTensor<uint32_t> qliV2PackedReg1;
    Reg::RegTensor<uint32_t> qliV2PackedReg2;
    Reg::RegTensor<uint32_t> qliV2PackedReg3;

    Reg::RegTensor<uint8_t> qliV2ByteReg0;
    Reg::RegTensor<uint8_t> qliV2ByteReg1;
    Reg::RegTensor<uint8_t> qliV2ByteReg2;
    Reg::RegTensor<uint8_t> qliV2ByteReg3;

    for (uint16_t qliV2LoopIndex = 0; qliV2LoopIndex < qliV2LoopCount; ++qliV2LoopIndex) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg1, qliV2PackedReg0,
                                                                 qliV2HistogramInput + qliV2LoopIndex * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(qliV2PackedReg3, qliV2PackedReg2,
                                                                 qliV2HistogramInput + (qliV2LoopIndex * 256) + 128);

        Reg::DeInterleave(qliV2ByteReg1, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2PackedReg0,
                          (Reg::RegTensor<uint8_t> &)qliV2PackedReg2);
        Reg::DeInterleave(qliV2ByteReg3, qliV2ByteReg2, (Reg::RegTensor<uint8_t> &)qliV2PackedReg1,
                          (Reg::RegTensor<uint8_t> &)qliV2PackedReg3);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, qliV2ByteReg0, (Reg::RegTensor<uint8_t> &)qliV2Index0, qliV2MaskB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, qliV2ByteReg1, (Reg::RegTensor<uint8_t> &)qliV2Index1, qliV2MaskB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, qliV2ByteReg2, (Reg::RegTensor<uint8_t> &)idx2, qliV2MaskB8);

        Reg::MaskReg pregEQ0And1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ0And1, pregEQ0, pregEQ1, qliV2MaskB8);
        Reg::And(pregEQAll, pregEQ0And1, pregEQ2, qliV2MaskB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram0, qliV2ByteReg3, pregEQAll);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            qliV2Histogram1, qliV2ByteReg3, pregEQAll);
    }

    StoreHistogramResult(qliV2HistogramOutput, qliV2Histogram0, qliV2Histogram1, qliV2MaskB16, qliV2MaskB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *qliV2KValue, __ubuf__ uint32_t *qliV2HistogramOutput,
                               __ubuf__ uint32_t *qliV2Index0Buffer, __ubuf__ uint32_t *qliV2Index1Buffer,
                               __ubuf__ uint32_t *qliV2Idx2Buf, __ubuf__ uint32_t *qliV2Idx3Buf)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore qliV2AlignIdx3;

    Reg::RegTensor<uint32_t> qliV2BtmK3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(qliV2BtmK3, qliV2KValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> qliV2IdxC;
        Reg::RegTensor<uint32_t> qliV2Cout;
        Reg::RegTensor<uint32_t> qliV2SqzIdx3;

        Reg::MaskReg qliV2PregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(qliV2IdxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(qliV2Cout, qliV2HistogramOutput + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(qliV2PregGE, qliV2Cout, qliV2BtmK3, qliV2MaskB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(qliV2SqzIdx3, (Reg::RegTensor<uint32_t> &)qliV2IdxC,
                                                               qliV2PregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(qliV2Idx3Buf, qliV2SqzIdx3, qliV2AlignIdx3);
    }
    Reg::StoreUnAlignPost(qliV2Idx3Buf, qliV2AlignIdx3);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> qliV2Index0;
    Reg::RegTensor<uint32_t> qliV2Index1;
    Reg::RegTensor<uint32_t> qliV2Idx2;
    Reg::RegTensor<uint32_t> qliV2Idx3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(qliV2Index0, qliV2Index0Buffer);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(qliV2Index1, qliV2Index1Buffer);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(qliV2Idx2, qliV2Idx2Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(qliV2Idx3, qliV2Idx3Buf);

    Reg::ShiftLefts(qliV2Index0, qliV2Index0, (int16_t)24, qliV2MaskB32);
    Reg::ShiftLefts(qliV2Index1, qliV2Index1, (int16_t)16, qliV2MaskB32);
    Reg::ShiftLefts(qliV2Idx2, qliV2Idx2, (int16_t)8, qliV2MaskB32);

    // ADD
    Reg::Add(qliV2Index0, qliV2Index0, qliV2Index1, qliV2MaskB32);
    Reg::Add(qliV2Index0, qliV2Index0, qliV2Idx2, qliV2MaskB32);
    Reg::Add(qliV2Index0, qliV2Index0, qliV2Idx3, qliV2MaskB32);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(qliV2KValue, qliV2Index0, qliV2MaskB32);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t *qliV2OutputIdxBuf, __ubuf__ uint32_t *qliV2HistogramInput,
                                       uint32_t qliV2BeginIdx, __ubuf__ uint32_t *qliV2KValue, uint16_t qliV2LoopCount)
{
    Reg::MaskReg qliV2MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore qliV2AlignIdx;

    Reg::RegTensor<uint32_t> qliV2KthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(qliV2KthValue, qliV2KValue);

    Reg::RegTensor<uint32_t> qliV2VregInput;

    for (uint16_t i = 0; i < (uint16_t)(qliV2LoopCount); ++i) {
        Reg::RegTensor<int32_t> qliV2IdxC;
        Reg::Arange(qliV2IdxC, qliV2BeginIdx + i * 64);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(qliV2VregInput, qliV2HistogramInput + i * 64);

        Reg::MaskReg qliV2PoutGT = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint32_t> sqzIdxOut;
        Reg::Compare<uint32_t, CMPMODE::GT>(qliV2PoutGT, qliV2VregInput, qliV2KthValue, qliV2MaskB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)qliV2IdxC,
                                                               qliV2PoutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(qliV2OutputIdxBuf, sqzIdxOut, qliV2AlignIdx);
    }
    Reg::StoreUnAlignPost(qliV2OutputIdxBuf, qliV2AlignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    Reg::RegTensor<int32_t> idxC;
    Reg::Arange(idxC, beginIdx);

    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf);

    Reg::MaskReg poutEQ = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> sqzIdxOut;
    Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutEQ);
    Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindValueGTOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputBuf,
                                         __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignValue;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        Reg::MaskReg poutGT = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint32_t> sqzValueOut;
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    }
    Reg::StoreUnAlignPost(outputValueBuf, alignValue);
}

__simd_vf__ void FindValueEQOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputBuf,
                                         __ubuf__ uint32_t *kValue)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::UnalignRegForStore alignValue;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf);

    Reg::MaskReg poutEQ = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> sqzValueOut;
    Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutEQ);
    Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    Reg::StoreUnAlignPost(outputValueBuf, alignValue);
}

__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                const LocalTensor<uint32_t> &outputValueLocal, const LocalTensor<uint32_t> &inputLocal,
                                const LocalTensor<uint32_t> &tmpIdxLocal, const LocalTensor<uint32_t> &tmpValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t s2SeqLen)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputBuf = (__ubuf__ uint32_t *)inputLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpValueBuf = (__ubuf__ uint32_t *)tmpValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint32_t bottomK = s2SeqLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (s2SeqLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (s2SeqLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = (topK + 64 - 1) / 64;

    // find kth-value
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputBuf, histogramsLoopNum, flag);
    FindFirstTargetBinVFImpl(idx0Buf, nkValueBuf, histogramsBuf, bottomK);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, histogramsLoopNum, flag);
    FindSecondTargetBinVFImpl(idx1Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, histogramsLoopNum, flag);
    FindThirdTargetBinVFImpl(idx2Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);

    // filter
    // 输出大于k-value的值value
    FindValueGTOutputVFImpl(outputValueBuf, inputBuf, nkValueBuf, inputLoopNum);
    // value-当前偏移大于k-value的值在AR特殊寄存器中的有效字节数
    int64_t arValueNum = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    // value-剩余需要输出等于k-value的数量
    int64_t remainValueNum = topK - (arValueNum / sizeof(uint32_t));
    for (uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arValueNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arValueNumPerLoop - arValueNum) / sizeof(uint32_t)) < remainValueNum) {
            // 调用一次查找等于k-value情况的过程；64: 单次循环处理的元素块大小
            FindValueEQOutputVFImpl(outputValueBuf, inputBuf + i * 64, nkValueBuf);
        } else {
            break;
        }
    }

    // 输出大于k-value的值idx
    FindIdxGTOutputVFImpl(outputIdxBuf, inputBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // idx-当前偏移大于k-value的值在AR特殊寄存器中的有效字节数
    int64_t arIdxNum = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    int64_t remainIdxNum = topK - (arIdxNum / sizeof(uint32_t));
    for (uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arIdxNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arIdxNumPerLoop - arIdxNum) / sizeof(uint32_t)) < remainIdxNum) {
            // 调用一次查找等于k-value情况的过程
            beginIdx = i * 64;                                                            // 64: 块起始偏移量
            FindIdxEQOutputVFImpl(outputIdxBuf, inputBuf + i * 64, beginIdx, nkValueBuf); // 64: 块起始偏移量
        } else {
            break;
        }
    }
}
} // namespace topkb32
#endif
