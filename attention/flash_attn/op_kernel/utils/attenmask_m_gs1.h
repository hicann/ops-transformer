/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attenmask_m_gs1.h
 * \brief attention mask 合轴（GS/SG，G 与 S1 合并为 gs1 轴）处理
 */

#ifndef ATTENMASK_M_GS1_H
#define ATTENMASK_M_GS1_H

#include "kernel_operator.h"
#include "flash_attn_common_def.h"
#include "../../../common/op_kernel/const_def.h"

enum MaskFormat {
    GS,
    SG,
};

struct MaskInfoMerged {
    uint32_t s1Size;
    uint32_t gSize;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;

    int64_t preToken = 0;
    int64_t nextToken = 0;

    uint32_t attenMaskS1Stride;
    uint32_t attenMaskDstStride = 0;

    FAMaskMode maskMode;

    uint32_t gs1StartIdx;
    uint32_t gs1dealNum;
};

__simd_vf__ inline void MergeBand(const uint64_t maskNextUb, const uint64_t maskPreUb, const uint16_t loopCount)
{
    AscendC::Reg::RegTensor<uint32_t> vreg_pre;
    AscendC::Reg::RegTensor<uint32_t> vreg_next;
    AscendC::Reg::RegTensor<uint32_t> vreg_xor;
    AscendC::Reg::RegTensor<uint32_t> vreg_not;
    AscendC::Reg::RegTensor<uint32_t> vreg_or;
    AscendC::Reg::MaskReg preg_all = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(vreg_xor, 0x1010101);

    for (uint16_t i = 0; i < loopCount; ++i) {
        AscendC::Reg::LoadAlign(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb + i * 64);
        AscendC::Reg::LoadAlign(vreg_next, (__ubuf__ uint32_t*&)maskNextUb + i * 64);
        AscendC::Reg::Xor(vreg_not, vreg_pre, vreg_xor, preg_all);
        AscendC::Reg::Or(vreg_or, vreg_not, vreg_next, preg_all);
        AscendC::Reg::StoreAlign((__ubuf__ uint32_t*&)maskNextUb + i * 64, vreg_or, preg_all);
    }
}

template <typename T, uint32_t s2BaseSize>
__simd_vf__ void MaskUbCopyS1GVF(__ubuf__ T* maskUb, uint16_t headGLoop, uint16_t midGLoop, uint16_t tailGLoop,
                                 uint16_t midS1Count)
{
    constexpr uint32_t repeatStride = s2BaseSize >> 5;

    AscendC::Reg::RegTensor<T> vreg_g_head;
    AscendC::Reg::RegTensor<T> vreg_g_mid;
    AscendC::Reg::RegTensor<T> vreg_g_tail;

    AscendC::Reg::MaskReg preg_all;
    if constexpr (s2BaseSize == 128) {
        preg_all = AscendC::Reg::CreateMask<bool, AscendC::Reg::MaskPattern::VL128>();
    } else { // s2BaseSize = 256
        preg_all = AscendC::Reg::CreateMask<bool, AscendC::Reg::MaskPattern::ALL>();
    }

    for (uint16_t x = headGLoop; x > 0; x = 0) { // if (headGLoop > 0) {}
        AscendC::Reg::LoadAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                                MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg_g_head, maskUb, 1, repeatStride,
                                                                         preg_all);
        for (uint16_t i = 1; i < headGLoop; ++i) {
            AscendC::Reg::StoreAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(maskUb, vreg_g_head, 1, repeatStride,
                                                                              preg_all);
        }
    }

    for (uint16_t x = midGLoop; x > 0; x = 0) { // if (midGLoop > 0) {}
        for (uint16_t i = 0; i < midS1Count; ++i) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg_g_mid, maskUb, 1, repeatStride,
                                                                             preg_all);
            for (uint16_t j = 1; j < midGLoop; ++j) {
                AscendC::Reg::StoreAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                                         MicroAPI::PostLiteral::POST_MODE_UPDATE>(maskUb, vreg_g_mid, 1, repeatStride,
                                                                                  preg_all);
            }
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }
    }

    for (uint16_t x = tailGLoop; x > 0; x = 0) { // if (tailGLoop > 0) {}
        AscendC::Reg::LoadAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                                MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg_g_tail, maskUb, 1, repeatStride,
                                                                         preg_all);
        for (uint16_t i = 1; i < tailGLoop; ++i) {
            AscendC::Reg::StoreAlign<T, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY,
                                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(maskUb, vreg_g_tail, 1, repeatStride,
                                                                              preg_all);
        }
    }
}

template <typename T>
__aicore__ inline void MergeMask(LocalTensor<T>& maskNext, LocalTensor<T>& maskPre, uint32_t& halfS1RealSize,
                                 int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop;
    uint64_t rowNumTimesEachLoop;
    if (s2BaseSize > regBytes) {
        rowNumEachLoop = 1;
        rowNumTimesEachLoop = static_cast<uint16_t>(s2BaseSize) / regBytes;
    } else {
        rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
        rowNumTimesEachLoop = 1;
    }
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize);
    uint16_t loopCount = (halfS1RealSizeLoop + rowNumEachLoop - 1) / rowNumEachLoop * rowNumTimesEachLoop;

    MergeBand(maskNextUb, maskPreUb, loopCount);
}

template <typename T, uint32_t s2BaseSize>
__aicore__ inline void MaskUbCopyS1G(const LocalTensor<T>& maskTensor, int32_t headGSize, int32_t gSize,
                                     int32_t tailGSize, int32_t midS1Count)
{
    if (gSize <= 1) {
        return;
    }
    __ubuf__ T* maskUb = (__ubuf__ T*)maskTensor.GetPhyAddr();
    MaskUbCopyS1GVF<T, s2BaseSize>(maskUb, headGSize, gSize, tailGSize, midS1Count);
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskInfoMerged& info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0;
    if (info.maskMode == FAMaskMode::CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.maskMode == FAMaskMode::BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressPre(MaskInfoMerged& info, uint32_t s1StartIdx)
{
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) -
                       static_cast<int64_t>(info.s2Size); // 统一以左上角为原点计算token
    int64_t delta = -preToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(info.s2StartIdx) - 1;
    uint64_t offset = 0;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskInfoMerged& info, uint32_t s1StartIdx = 0, bool isPre = false)
{
    if (isPre) {
        return ComputeAttenMaskOffsetCompressPre(info, s1StartIdx);
    } else {
        return ComputeAttenMaskOffsetCompress(info, s1StartIdx);
    }
}

template <MaskFormat MASK_FORMAT>
__aicore__ inline bool IsSkipAttentionmaskForPre(MaskInfoMerged& info)
{
    if (info.maskMode != FAMaskMode::BAND) {
        return true;
    }

    int32_t s1StartIdx;
    if constexpr (MASK_FORMAT == MaskFormat::GS) {
        s1StartIdx = info.gs1StartIdx % info.s1Size;
    } else { // SG
        s1StartIdx = info.gs1StartIdx / info.gSize;
    }

    if constexpr (MASK_FORMAT == MaskFormat::GS) {
        if (s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
            return false;
        }
    }

    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) -
                       static_cast<int64_t>(info.s2Size); // 统一以左上角为原点计算Token
    int32_t s1EndIdx;
    if constexpr (MASK_FORMAT == MaskFormat::GS) {
        s1EndIdx = s1StartIdx + info.gs1dealNum;
    } else { // SG
        s1EndIdx = (info.gs1StartIdx + info.gs1dealNum) / info.gSize;
    }

    if (static_cast<int64_t>(info.s2StartIdx) + preToken >= static_cast<int64_t>(s1EndIdx)) {
        return true;
    }
    return false;
}

template <MaskFormat MASK_FORMAT>
__aicore__ inline bool IsSkipAttentionmask(MaskInfoMerged& info)
{
    int32_t s1StartIdx;
    if constexpr (MASK_FORMAT == MaskFormat::GS) {
        s1StartIdx = info.gs1StartIdx % info.s1Size;
    } else { // SG
        s1StartIdx = info.gs1StartIdx / info.gSize;
    }

    if constexpr (MASK_FORMAT == MaskFormat::GS) {
        if (s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
            return false;
        }
    }

    int64_t nextToken = 0;
    if (info.maskMode == FAMaskMode::CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算Token
    } else if (info.maskMode == FAMaskMode::BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                             MaskInfoMerged& info, uint32_t s1StartIdx, uint32_t s1EndIdx,
                                             bool isPre = false)
{
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
    dataCopyParams.dstStride = info.attenMaskDstStride;
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
    DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForGsLayout(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                                      MaskInfoMerged& info, bool isPre, uint32_t mte2ToVEventId)
{
    int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
    uint32_t attenMaskS2Stride = AttentionCommon::Align(info.s2dealNum, 32U) + 32 * info.attenMaskDstStride;
    uint32_t headS1Count = (s1StartIdx == 0) ? 0 : (info.s1Size - s1StartIdx);
    // midGCount: gs1dealNum 跨 G 边界时，中间完整 s1Size 行的重复次数
    uint32_t midGCount = (info.gs1dealNum > headS1Count) ? (info.gs1dealNum - headS1Count) / info.s1Size : 0;
    // 无完整G（midGCount==0）时，模板写入会超出 buffer 边界，改为走 if 分支直接从 GM 拷贝
    if (info.gs1dealNum <= info.s1Size || midGCount == 0) {
        Mutex::Lock<PIPE_MTE2>(mte2ToVEventId);
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, info.s1Size, isPre);
            LocalTensor<T> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskS2Stride];
            AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, s1EndIdx, isPre);
        } else {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx, isPre);
        }
        Mutex::Unlock<PIPE_MTE2>(mte2ToVEventId);
    } else {
        // 有完整G：先从GM拷贝一个完整G到UB作为模板，再用UB→UB DataCopy填充head/mid/tail
        uint32_t tailS1Size = (info.gs1dealNum - headS1Count) % info.s1Size;

        Mutex::Lock<PIPE_MTE2>(mte2ToVEventId);
        LocalTensor<T> attenMaskSecUb = attenMaskUb[headS1Count * attenMaskS2Stride];
        AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, info.s1Size, isPre);
        Mutex::Unlock<PIPE_MTE2>(mte2ToVEventId);

        Mutex::Lock<PIPE_V>(mte2ToVEventId);
        // head
        if (headS1Count > 0) {
            DataCopy(attenMaskUb, attenMaskUb[info.s1Size * attenMaskS2Stride], headS1Count * attenMaskS2Stride);
        }
        // mid
        for (uint32_t i = 1; i < midGCount; i++) {
            DataCopy(attenMaskUb[(headS1Count + i * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
                     info.s1Size * attenMaskS2Stride);
        }
        // tail
        if (tailS1Size > 0) {
            DataCopy(attenMaskUb[(headS1Count + midGCount * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
                     tailS1Size * attenMaskS2Stride);
        }
        Mutex::Unlock<PIPE_V>(mte2ToVEventId);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                                      MaskInfoMerged& info, bool isPre, uint32_t mte2ToVEventId)
{
    uint32_t s1StartIdx = info.gs1StartIdx / info.gSize;
    uint32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) / info.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t headGSize = info.gs1dealNum;
    if (s1Count > 1) {
        headGSize = (info.gs1StartIdx % info.gSize == 0) ? 0 : (info.gSize - info.gs1StartIdx % info.gSize);
    }
    uint32_t remainRowCount = info.gs1dealNum - headGSize;
    uint32_t midS1Count = remainRowCount / info.gSize;
    uint32_t tailGSize = remainRowCount % info.gSize;
    uint32_t attenMaskSizeAlign = AttentionCommon::Align(info.s2dealNum, 32U);
    uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;

    Mutex::Lock<PIPE_MTE2>(mte2ToVEventId);
    // ub-head
    if (headGSize > 0) {
        AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1, isPre);
        s1StartIdx++;
    }

    // ub-remain
    if (remainRowCount > 0) {
        uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
        dataCopyParams.blockLen = info.s2dealNum;
        dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
        dataCopyParams.dstStride = (info.gSize - 1) * attenMaskS2Stride / 32 + info.attenMaskDstStride;
        // blockLen + padParams.len + dstStride = gSize * s2Size
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum),
                                                   1U};
        DataCopyPad(attenMaskUb[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    }
    Mutex::Unlock<PIPE_MTE2>(mte2ToVEventId);

    Mutex::Lock<PIPE_V>(mte2ToVEventId);
    MaskUbCopyS1G<T, s2BaseSize>(attenMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
    Mutex::Unlock<PIPE_V>(mte2ToVEventId);
    PipeBarrier<PIPE_V>();
}

template <typename T, MaskFormat MASK_FORMAT, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                           MaskInfoMerged& info, bool isPre, uint32_t mte2ToVEventId)
{
    if constexpr (MASK_FORMAT == MaskFormat::GS) {
        AttentionmaskCopyInForGsLayout<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, isPre,
                                                                         mte2ToVEventId);
    } else { // SG
        AttentionmaskCopyInForSgLayout<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, isPre,
                                                                         mte2ToVEventId);
    }
}

#endif
