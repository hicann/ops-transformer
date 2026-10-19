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
 * \file attenmask_david.h
 * \brief
 */

#ifndef ATTENMASK_DAVID_H
#define ATTENMASK_DAVID_H

#include "util_david.h"

namespace davidutil {
enum class AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE = 1,
    RIGHT_DOWN_CAUSAL_MODE = 2,
    BAND_MODE = 3,
    PREFIX_MODE = 4,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE = 6
};

enum class AttenMaskComputeMode {
    NORMAL_MODE = 0,
    CAUSAL_OR_NEXT_ONLY_MODE,
    PRE_ONLY_MODE,
    PRE_AND_NEXT_MODE,
    NO_NEED_COMPUTE_MODE,
    PREFIX_COMPUTE_MODE,
    PREFIX_N_COMPUTE_MODE
};

struct AttenMaskInfo {
    int64_t preTokens;
    int64_t nextTokens;
    uint8_t compressMode = 0;
    int64_t attenMaskShapeType;
    int64_t attenMaskS2Size;
    int64_t bandIndex;
    GM_ADDR prefixNAddr;
    int64_t attenMaskOffsetPre;
    AttenMaskComputeMode computeMode = AttenMaskComputeMode::NORMAL_MODE;
};

__aicore__ inline void BoolCopyInRegbase(const LocalTensor<uint8_t>& dstTensor, const GlobalTensor<uint8_t>& srcTensor,
                                         int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size,
                                         int64_t s2BaseSize)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = CeilDiv(s2Size, blockBytes);
    dataCopyParams.dstStride = CeilDiv(s2BaseSize, blockBytes) - dataCopyParams.blockLen;
    if (totalS2Size % blockBytes == 0) {
        dataCopyParams.srcStride = (totalS2Size - s2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyParams, dataCopyPadParams);
    }
}

template <bool hasAtten, bool isInfer = false>
__aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre, int64_t s1Offset,
                                               RunInfo<isInfer>& runInfo, ConstInfo<isInfer>& constInfo,
                                               AttenMaskInfo& attenMaskInfo)
{
    if constexpr (hasAtten == true) {
        int64_t causalOrNextFactor = deltaCausalOrNext - constInfo.s2BaseSize;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
            attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            if (causalOrNextFactor >= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            }
            return;
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            int64_t preFactor = deltaPre + 1 + constInfo.s1BaseSize;
            if (causalOrNextFactor >= 0 && preFactor <= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else if (causalOrNextFactor < 0 && preFactor <= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            } else if (causalOrNextFactor >= 0 && preFactor > 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
            } else {
                attenMaskInfo.computeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
            }
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            int64_t preFactor = deltaPre - constInfo.s2BaseSize;
            // Triangular part and rectangular part have one is not counted, then the whole is not counted,
            // otherwise it needs to be calculated
            if (causalOrNextFactor >= 0 || preFactor >= 0 || deltaPre > runInfo.s2SizeThisBatch) {
                // attenmask value is all 0, no need to compute
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                int64_t intersectionX = runInfo.s1SizeThisBatch - runInfo.s2SizeThisBatch +
                                        ((__gm__ int64_t*)attenMaskInfo.prefixNAddr)[runInfo.boIdx];
                if (s1Offset >= intersectionX) {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
                } else if (s1Offset + constInfo.s1BaseSize <= intersectionX) {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE;
                } else {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::PREFIX_COMPUTE_MODE;
                }
            }
        }
        return;
    }
}

template <bool hasAtten, bool isInfer = false>
__aicore__ inline int64_t ComputeOffsetForNoCompress(RunInfo<isInfer>& runInfo, ConstInfo<isInfer>& constInfo,
                                                     AttenMaskInfo& attenMaskInfo)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize * attenMaskInfo.attenMaskS2Size +
                           runInfo.vecCoreOffset * attenMaskInfo.attenMaskS2Size +
                           runInfo.vecS1Offset * attenMaskInfo.attenMaskS2Size;
        int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
        if (attenMaskInfo.attenMaskShapeType == attenMaskBN2GS1S2) {
            bOffset = runInfo.b1SSOffset * constInfo.n2G;
            n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.s1SizeThisBatch * runInfo.s2SizeThisBatch;
            gOffset = runInfo.goIdx * runInfo.s1SizeThisBatch * runInfo.s2SizeThisBatch;
        } else if (attenMaskInfo.attenMaskShapeType == attenMaskBS1S2) {
            bOffset = runInfo.b1SSOffset;
        }
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    }
}

template <bool isInfer = false>
__aicore__ inline int64_t ComputeOffsetForCausal(const int64_t& delta, const uint32_t& s1BaseSize,
                                                 const uint32_t& s2BaseSize, const uint32_t& attenMaskS2Size,
                                                 RunInfo<isInfer>& runInfo, const bool useDn = false)
{
    if (useDn) {
        if (delta >= 0) {
            return Min(delta, s2BaseSize) + runInfo.vecS1Offset + runInfo.vecS2Offset * attenMaskS2Size + 1;
        } else {
            return Min(-1 * delta, s1BaseSize) * attenMaskS2Size + runInfo.vecS1Offset +
                   runInfo.vecS2Offset * attenMaskS2Size + 1;
        }
    } else {
        if (delta <= 0) {
            return Min(-1 * delta, s1BaseSize) + runInfo.vecS1Offset * attenMaskS2Size;
        } else {
            return (Min(delta, s2BaseSize) + runInfo.vecS1Offset) * attenMaskS2Size;
        }
    }
}

__aicore__ inline int64_t ComputeOffsetForPrefixRectangle(const int64_t& delta, const uint32_t& s2BaseSize,
                                                          const uint32_t& attenMaskS2Size)
{
    // attenMask S1 is same to S2
    if (delta <= 0) {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2; // 2048 * 2048 + 1024
    } else if (delta >= s2BaseSize) {
        return attenMaskS2Size * attenMaskS2Size; // 2048 * 2048 + 0
    } else {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2 - delta; // 2048 * 2048 + (1024 - delta)
    }
}

#ifndef __CCE_KT_TEST__
template <bool hasAtten>
__aicore__ inline void MergeBandModeMask(const LocalTensor<uint8_t>& maskPre, const LocalTensor<uint8_t>& maskNext,
                                         int32_t& halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = halfS1RealSizeLoop / rowNumEachLoop;

    __VEC_SCOPE__
    {
        vector_u32 vreg_pre;
        vector_u32 vreg_next;
        vector_u32 vreg_xor;
        vector_u32 vreg_not;
        vector_u32 vreg_or;
        vector_bool preg_all = pge_b32(PAT_ALL);
        vbr(vreg_xor, 0x1010101);

        for (uint16_t i = 0; i < loopCount; ++i) {
            vlds(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb, i * 64, NORM);
            vlds(vreg_next, (__ubuf__ uint32_t*&)maskNextUb, i * 64, NORM);
            vxor(vreg_not, vreg_pre, vreg_xor, preg_all);
            vor(vreg_or, vreg_not, vreg_next, preg_all);
            vsts(vreg_or, (__ubuf__ uint32_t*&)maskNextUb, i * 64, NORM_B32, preg_all);
        }
    }
}

template <bool hasAtten>
__aicore__ inline void MergePrefixModeMask(const LocalTensor<uint8_t>& maskPre, const LocalTensor<uint8_t>& maskNext,
                                           int32_t& halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = halfS1RealSizeLoop / rowNumEachLoop;

    __VEC_SCOPE__
    {
        vector_u32 vreg_pre;
        vector_u32 vreg_next;
        vector_u32 vreg_and;
        vector_bool preg_all = pge_b32(PAT_ALL);

        for (uint16_t i = 0; i < loopCount; ++i) {
            vlds(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb, i * 64, NORM);
            vlds(vreg_next, (__ubuf__ uint32_t*&)maskNextUb, i * 64, NORM);
            vand(vreg_and, vreg_pre, vreg_next, preg_all);
            vsts(vreg_and, (__ubuf__ uint32_t*&)maskNextUb, i * 64, NORM_B32, preg_all);
        }
    }
}
#else
template <bool hasAtten>
__aicore__ inline void MergeBandModeMask(const LocalTensor<uint8_t>& maskPre, const LocalTensor<uint8_t>& maskNext,
                                         int32_t& halfS1RealSize, int64_t s2BaseSize)
{}

template <bool hasAtten>
__aicore__ inline void MergePrefixModeMask(const LocalTensor<uint8_t>& maskPre, const LocalTensor<uint8_t>& maskNext,
                                           int32_t& halfS1RealSize, int64_t s2BaseSize)
{}
#endif

template <bool hasAtten, bool isInfer = false>
__aicore__ inline int64_t ComputeAttenMaskOffset(RunInfo<isInfer>& runInfo, ConstInfo<isInfer>& constInfo,
                                                 AttenMaskInfo& attenMaskInfo, const bool useDn = false)
{
    if constexpr (hasAtten == true) {
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return ComputeOffsetForNoCompress<hasAtten>(runInfo, constInfo, attenMaskInfo);
        }
        if (constInfo.layoutType == (uint32_t)LayOutTypeEnum::LAYOUT_TND) {
            // compress mode
            int64_t delta = 0;
            int64_t deltaPre = 0;
            int64_t deltaN = runInfo.s1SizeThisBatch - runInfo.s2SizeThisBatch;
            int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize;
            int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
            if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset;
            } else if (attenMaskInfo.compressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
            } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
                int64_t tmpPre = attenMaskInfo.preTokens;
                int64_t tmpNext = attenMaskInfo.nextTokens;
                int64_t transPreTokens = runInfo.s1SizeThisBatch - Max(runInfo.s2SizeThisBatch - tmpPre, 0);
                int64_t transNextTokens = runInfo.s2SizeThisBatch - Max(runInfo.s1SizeThisBatch - tmpNext, 0);
                deltaPre = s1Offset - s2Offset - transPreTokens - 1;
                int64_t maskOffsetPre = ComputeOffsetForCausal<isInfer>(
                    deltaPre, constInfo.s1BaseSize, constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size, runInfo);
                attenMaskInfo.attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask operation.
                delta = s1Offset - s2Offset + transNextTokens;
            } else if (attenMaskInfo.compressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE)) {
                if (runInfo.boIdx == attenMaskInfo.bandIndex) {
                    delta = s1Offset - s2Offset - deltaN + attenMaskInfo.nextTokens;
                } else {
                    delta = s1Offset - s2Offset - deltaN;
                }
            } else if (attenMaskInfo.compressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE)) {
                if (runInfo.boIdx == attenMaskInfo.bandIndex) {
                    delta = s1Offset - s2Offset + runInfo.s2SizeThisBatch -
                            Max(runInfo.s1SizeThisBatch - attenMaskInfo.nextTokens, 0);
                } else {
                    delta = s1Offset - s2Offset;
                }
            } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
                if ((runInfo.s1SizeThisBatch + ((__gm__ int64_t*)attenMaskInfo.prefixNAddr)[runInfo.boIdx]) >
                    runInfo.s2SizeThisBatch) {
                    // prefix reuse attenMaskOffsetPre
                    deltaPre = ((__gm__ int64_t*)attenMaskInfo.prefixNAddr)[runInfo.boIdx] - runInfo.s2StartIdx -
                               runInfo.s2LoopCount * constInfo.s2BaseSize;
                    attenMaskInfo.attenMaskOffsetPre =
                        ComputeOffsetForPrefixRectangle(deltaPre, constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size);
                }
            } else {
                return 0;
            }
            GetAttenMaskComputeMode<hasAtten>(delta, deltaPre, s1Offset, runInfo, constInfo, attenMaskInfo);
            return ComputeOffsetForCausal<isInfer>(delta, constInfo.s1BaseSize, constInfo.s2BaseSize,
                                                   attenMaskInfo.attenMaskS2Size, runInfo);
        }
        // compress mode
        int64_t deltaCausalOrNext = 0;
        int64_t deltaPre = 0;
        int64_t deltaN = runInfo.s1SizeThisBatch - runInfo.s2SizeThisBatch;
        int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize;
        int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset;
        } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
        } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            deltaPre = s1Offset - s2Offset - attenMaskInfo.preTokens - 1;
            attenMaskInfo.attenMaskOffsetPre = ComputeOffsetForCausal<isInfer>(
                deltaPre, constInfo.s1BaseSize, constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size, runInfo, useDn);
            deltaCausalOrNext = s1Offset - s2Offset + attenMaskInfo.nextTokens;
        } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
            if ((runInfo.s1SizeThisBatch + ((__gm__ int64_t*)attenMaskInfo.prefixNAddr)[runInfo.boIdx]) >
                runInfo.s2SizeThisBatch) {
                // prefix reuse attenMaskOffsetPre
                deltaPre = ((__gm__ int64_t*)attenMaskInfo.prefixNAddr)[runInfo.boIdx] - runInfo.s2StartIdx -
                           runInfo.s2LoopCount * constInfo.s2BaseSize;
                attenMaskInfo.attenMaskOffsetPre =
                    ComputeOffsetForPrefixRectangle(deltaPre, constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size);
            }
        } else {
            return 0;
        }
        GetAttenMaskComputeMode<hasAtten>(deltaCausalOrNext, deltaPre, s1Offset, runInfo, constInfo, attenMaskInfo);
        return ComputeOffsetForCausal<isInfer>(deltaCausalOrNext, constInfo.s1BaseSize, constInfo.s2BaseSize,
                                               attenMaskInfo.attenMaskS2Size, runInfo, useDn);
    }
}

template <bool hasAtten, bool isInfer = false>
__aicore__ inline void AttenMaskCopyIn(TQue<QuePosition::VECIN, 1>& attenMaskInQue, GlobalTensor<uint8_t>& srcTensor,
                                       RunInfo<isInfer>& runInfo, ConstInfo<isInfer>& constInfo,
                                       AttenMaskInfo& attenMaskInfo, bool isPre = false)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten>(runInfo, constInfo, attenMaskInfo);
        BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
        attenMaskInQue.template EnQue(attenMaskUb);
        return;
    }
}

template <bool isInfer = false>
__aicore__ inline void AttenMaskCopyIn(TBuf<>& attenMaskBuf, GlobalTensor<uint8_t>& srcTensor,
                                       RunInfo<isInfer>& runInfo, ConstInfo<isInfer>& constInfo,
                                       AttenMaskInfo& attenMaskInfo, const bool useDn = false)
{
    uint32_t s1BaseSize = constInfo.s1BaseSize;
    uint32_t s2BaseSize = constInfo.s2BaseSize;
    LocalTensor<uint8_t> attenMaskUb = attenMaskBuf.Get<uint8_t>();
    int64_t maskOffset = ComputeAttenMaskOffset<true>(runInfo, constInfo, attenMaskInfo, useDn);
    if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
        if (attenMaskInfo.computeMode == AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
            maskOffset = 4194304; // 2048*2048
        }
        if (attenMaskInfo.computeMode == AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE) {
            maskOffset = attenMaskInfo.attenMaskOffsetPre;
        }
        BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
        if (attenMaskInfo.computeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
            BoolCopyInRegbase(attenMaskUb[constInfo.s1VecBaseSize * constInfo.s2BaseSize], srcTensor,
                              attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize, runInfo.s2RealSize,
                              attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
            AscendC::TEventID vWaitMte2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
            MergePrefixModeMask<true>(attenMaskUb[constInfo.s1VecBaseSize * constInfo.s2BaseSize], attenMaskUb,
                                      runInfo.halfS1RealSize, constInfo.s2BaseSize);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        }
        return;
    }
    if (useDn) {
        BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.s2VecSize, runInfo.halfS1RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s1VecBaseSize);
    } else {
        BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
    }
    if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE) &&
        (attenMaskInfo.computeMode == AttenMaskComputeMode::PRE_ONLY_MODE ||
         attenMaskInfo.computeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE)) {
        BoolCopyInRegbase(attenMaskUb[constInfo.s1VecBaseSize * constInfo.s2BaseSize], srcTensor,
                          attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
        AscendC::TEventID vWaitMte2 = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        MergeBandModeMask<true>(attenMaskUb[constInfo.s1VecBaseSize * constInfo.s2BaseSize], attenMaskUb,
                                runInfo.halfS1RealSize, constInfo.s2BaseSize);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(vWaitMte2);
    }
    return;
}

template <bool hasAtten, bool isInfer = false>
__aicore__ inline void AttenMaskCopyIn(TQue<QuePosition::VECIN, 1>& attenMaskInQue,
                                       TQue<QuePosition::VECIN, 1>& attenMaskInQuePre, GlobalTensor<uint8_t>& srcTensor,
                                       RunInfo<isInfer>& runInfo, ConstInfo<isInfer>& constInfo,
                                       AttenMaskInfo& attenMaskInfo, const bool useDn = false)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten>(runInfo, constInfo, attenMaskInfo, useDn);
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            if (attenMaskInfo.computeMode == AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
                maskOffset = 4194304; // 2048*2048
            }
            if (attenMaskInfo.computeMode == AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE) {
                maskOffset = attenMaskInfo.attenMaskOffsetPre;
            }
            BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                              attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
            attenMaskInQue.template EnQue(attenMaskUb);
            if (attenMaskInfo.computeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                LocalTensor<uint8_t> attenMaskUbPre = attenMaskInQuePre.template AllocTensor<uint8_t>();
                BoolCopyInRegbase(attenMaskUbPre, srcTensor, attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize,
                                  runInfo.s2RealSize, attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
                attenMaskInQuePre.template EnQue(attenMaskUbPre);
                attenMaskInQuePre.template DeQue<uint8_t>();
                attenMaskInQue.template DeQue<uint8_t>();
                MergePrefixModeMask<hasAtten>(attenMaskUbPre, attenMaskUb, runInfo.halfS1RealSize,
                                              constInfo.s2BaseSize);
                attenMaskInQuePre.template FreeTensor(attenMaskUbPre);
                attenMaskInQue.template EnQue(attenMaskUb);
            }
            return;
        }
        BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
        attenMaskInQue.template EnQue(attenMaskUb);
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE) &&
            (attenMaskInfo.computeMode == AttenMaskComputeMode::PRE_ONLY_MODE ||
             attenMaskInfo.computeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE)) {
            LocalTensor<uint8_t> attenMaskUbPre = attenMaskInQuePre.template AllocTensor<uint8_t>();
            BoolCopyInRegbase(attenMaskUbPre, srcTensor, attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize,
                              runInfo.s2RealSize, attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize);
            attenMaskInQuePre.template EnQue(attenMaskUbPre);
            attenMaskInQuePre.template DeQue<uint8_t>();
            attenMaskInQue.template DeQue<uint8_t>();
            MergeBandModeMask<hasAtten>(attenMaskUbPre, attenMaskUb, runInfo.halfS1RealSize, constInfo.s2BaseSize);
            attenMaskInQuePre.template FreeTensor(attenMaskUbPre);
            attenMaskInQue.template EnQue(attenMaskUb);
        }
        return;
    }
}
} // namespace davidutil

#endif // ATTENMASK_DAVID_H
