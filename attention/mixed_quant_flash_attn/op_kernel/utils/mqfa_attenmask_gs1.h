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
 * \file mqfa_attenmask_gs1.h
 * \brief
 */

// TODO，与A3的合并策略待定

#ifndef MQFA_ATTENMASK_GS1_H
#define MQFA_ATTENMASK_GS1_H

using namespace AscendC;
using namespace AscendC::MicroAPI;
namespace fa_base_vector_gs1 {
enum LAYOUT_Q {
    GS,
    SG,
};

enum MaskDataType : uint8_t {
    MASK_BOOL,
    MASK_INT8,
    MASK_UINT8,
    MASK_FP16,
};

enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

struct MaskInfo {
    uint32_t gs1StartIdx;
    uint32_t gs1dealNum;
    uint32_t s1Size;
    uint32_t gSize;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;
    uint32_t nBaseSize;

    int64_t preToken = 0;
    int64_t nextToken = 0;

    // for bss & bs
    uint32_t batchIdx;
    uint32_t attenMaskBatchStride;
    uint32_t attenMaskS1Stride; // 这个变量名与A3 代码不一致，统一整改
    uint32_t attenMaskDstStride = 0;

    LAYOUT_Q layout;
    MaskDataType attenMaskType;
    SparseMode sparseMode;
    uint32_t maskValue;

    uint64_t s1LeftPaddingSize = 0;
    uint64_t s2LeftPaddingSize = 0;
};

template <typename T1>
static __aicore__ inline constexpr T1 Align(T1 s, T1 alignment)
{
    if constexpr (IsSameType<T1, uint64_t>::value || IsSameType<T1, uint32_t>::value ||
                  IsSameType<T1, uint16_t>::value || IsSameType<T1, uint8_t>::value) {
        return (s + alignment - 1) & (~(alignment - 1));
    } else {
        return uint64_t(s + alignment - 1) & (~uint64_t(alignment - 1));
    }
}
// TODO，后续放到VF层
__simd_vf__ inline void MergeBand(const uint64_t maskNextUb, const uint64_t maskPreUb, const uint16_t loopCount)
{
    RegTensor<uint32_t> vreg_pre;
    RegTensor<uint32_t> vreg_next;
    RegTensor<uint32_t> vreg_xor;
    RegTensor<uint32_t> vreg_not;
    RegTensor<uint32_t> vreg_or;
    MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();
    Duplicate(vreg_xor, 0x1010101);

    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb + i * 64);
        LoadAlign(vreg_next, (__ubuf__ uint32_t*&)maskNextUb + i * 64);
        Xor(vreg_not, vreg_pre, vreg_xor, preg_all);
        Or(vreg_or, vreg_not, vreg_next, preg_all);
        StoreAlign((__ubuf__ uint32_t*&)maskNextUb + i * 64, vreg_or, preg_all);
    }
}

template <typename T, uint32_t s2BaseSize>
__simd_vf__ void MaskUbCopyS1GVF(__ubuf__ T* maskUb, uint16_t headGLoop, uint16_t midGLoop, uint16_t tailGLoop,
                                 uint16_t midS1Count)
{
    MaskReg preg_all;
    if constexpr (s2BaseSize == 128) {
        preg_all = CreateMask<bool, MaskPattern::VL128>();
    } else { // s2BaseSize = 256 or 512
        preg_all = CreateMask<bool, MaskPattern::ALL>();
    }

    if constexpr (s2BaseSize == 512) {
        // 512B 需要拆分成 2 次 256B 的操作，步长固定为 256B 对应的 block 数
        constexpr uint32_t repeatStride = 256 >> 5;
        // head
        RegTensor<T> vreg_g_head1;
        RegTensor<T> vreg_g_head2;
        for (uint16_t x = headGLoop; x > 0; x = 0) { // if (headGLoop > 0) {}
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_head1, maskUb, 1, repeatStride, preg_all);
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_head2, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_head1, 1, repeatStride, preg_all);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_head2, 1, repeatStride, preg_all);
            }
        }

        // mid
        RegTensor<T> vreg_g_mid1;
        RegTensor<T> vreg_g_mid2;
        for (uint16_t x = midGLoop; x > 0; x = 0) { // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg_g_mid1, maskUb, 1, repeatStride, preg_all);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg_g_mid2, maskUb, 1, repeatStride, preg_all);
                for (uint16_t j = 1; j < midGLoop; ++j) {
                    StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maskUb, vreg_g_mid1, 1, repeatStride, preg_all);
                    StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maskUb, vreg_g_mid2, 1, repeatStride, preg_all);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
        }

        // tail
        RegTensor<T> vreg_g_tail1;
        RegTensor<T> vreg_g_tail2;
        for (uint16_t x = tailGLoop; x > 1; x = 0) { // if (tailGLoop > 1) {}
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_tail1, maskUb, 1, repeatStride, preg_all);
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_tail2, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_tail1, 1, repeatStride, preg_all);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_tail2, 1, repeatStride, preg_all);
            }
        }
    } else {
        // 原有逻辑处理 128/256
        constexpr uint32_t repeatStride = s2BaseSize >> 5;
        RegTensor<T> vreg_g_head;
        RegTensor<T> vreg_g_mid;
        RegTensor<T> vreg_g_tail;

        for (uint16_t x = headGLoop; x > 0; x = 0) { // if (headGLoop > 0) {}
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_head, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_head, 1, repeatStride, preg_all);
            }
        }

        for (uint16_t x = midGLoop; x > 0; x = 0) { // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    vreg_g_mid, maskUb, 1, repeatStride, preg_all);
                for (uint16_t j = 1; j < midGLoop; ++j) {
                    StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        maskUb, vreg_g_mid, 1, repeatStride, preg_all);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            }
        }

        for (uint16_t x = tailGLoop; x > 1; x = 0) { // if (tailGLoop > 1) {}
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vreg_g_tail, maskUb, 1, repeatStride, preg_all);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maskUb, vreg_g_tail, 1, repeatStride, preg_all);
            }
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
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = (halfS1RealSizeLoop / rowNumEachLoop) * rowNumTimesEachLoop;

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

__aicore__ inline uint64_t ComputeAttenMaskOffsetNoCompress(MaskInfo& info, uint32_t s1StartIdx)
{
    uint64_t bOffset = static_cast<uint64_t>(info.batchIdx) * static_cast<uint64_t>(info.attenMaskBatchStride);
    uint64_t s1Offset = (info.s1LeftPaddingSize + s1StartIdx % info.s1Size) * info.attenMaskS1Stride;
    uint64_t s2Offset = info.s2LeftPaddingSize + info.s2StartIdx;
    return bOffset + s1Offset + s2Offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskInfo& info, uint32_t s1StartIdx)
{
    int64_t nextToken = 0; // sparse2 本身原点就是左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.sparseMode == BAND) {                                          // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    uint64_t offset = 0;
    int64_t delta = nextToken + s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressPre(MaskInfo& info, uint32_t s1StartIdx)
{
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) -
                       static_cast<int64_t>(info.s2Size); // 统一以左上角为原点计算token
    int64_t delta = -preToken + static_cast<int64_t>(s1StartIdx) - static_cast<int64_t>(info.s2StartIdx) - 1;
    uint64_t offset = 0;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskInfo& info, uint32_t s1StartIdx = 0, bool isPre = false)
{
    if (isPre) {
        return ComputeAttenMaskOffsetCompressPre(info, s1StartIdx);
    } else {
        if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
            return ComputeAttenMaskOffsetNoCompress(info, s1StartIdx);
        } else {
            return ComputeAttenMaskOffsetCompress(info, s1StartIdx);
        }
    }
}

__aicore__ inline bool IsSkipAttentionmaskForPre(MaskInfo& info)
{
    if (info.sparseMode != BAND) {
        return true;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }

    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) -
                       static_cast<int64_t>(info.s2Size); // 统一以左上角为原点计算Token
    int32_t s1EndIdx =
        info.layout == GS ? s1StartIdx + info.gs1dealNum : (info.gs1StartIdx + info.gs1dealNum) / info.gSize;

    if (static_cast<int64_t>(info.s2StartIdx) + preToken >= static_cast<int64_t>(s1EndIdx)) {
        return true;
    }
    return false;
}

__aicore__ inline bool IsSkipAttentionmask(MaskInfo& info)
{
    if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
        return false;
    }
    int64_t nextToken = 0;
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken = static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    } else if (info.sparseMode == BAND) {
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }
    int64_t s2End = static_cast<int64_t>(info.s2StartIdx) + static_cast<int64_t>(info.s2dealNum);
    // SG类
    if (info.layout == SG) {
        int32_t s1StartIdx = info.gs1StartIdx / info.gSize;
        if (s2End <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            return true;
        }
        return false;
    }
    // GS类
    // 分头中尾判断
    int32_t headS1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t gs1EndInCurrentS1 = headS1StartIdx + info.gs1dealNum;
    // 不跨G，只判断头
    bool headCanSkip = s2End <= static_cast<int64_t>(headS1StartIdx) + nextToken;
    if (gs1EndInCurrentS1 <= info.s1Size) {
        if (headCanSkip) {
            return true;
        }
        return false;
    }
    // 跨G 中间/尾部 s1StartIdx = 0
    bool middleOrTailCanSkip = (s2End <= nextToken);
    if (middleOrTailCanSkip) {
        return true;
    }
    return false;
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr, MaskInfo& info,
                                             uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
{
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttenMaskOffset(info, s1StartIdx, isPre);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskS1Stride - info.s2dealNum;
    dataCopyParams.dstStride = info.attenMaskDstStride;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
    DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T, uint32_t s2BaseSize>
__simd_vf__ void MaskUbCopyGS1VF(__ubuf__ T* maskUb, __ubuf__ T* maskSecUb, uint16_t headLoop, uint16_t midRowCount,
                                 uint16_t midGLoop, uint16_t tailLoop)
{
    MaskReg preg_all;
    if constexpr (s2BaseSize == 128) {
        preg_all = CreateMask<bool, MaskPattern::VL128>();
    } else { // s2BaseSize = 256 or 512
        preg_all = CreateMask<bool, MaskPattern::ALL>();
    }

    if constexpr (s2BaseSize == 512) {
        constexpr uint32_t repeatStride = 256 >> 5;
        constexpr uint32_t halfStride = 256;
        // head: copy from maskSecUb[s1StartIdx .. s1StartIdx+headLoop)
        RegTensor<T> vreg_head1;
        RegTensor<T> vreg_head2;
        __ubuf__ T* srcHead = maskSecUb + (midRowCount - headLoop) * 512;
        __ubuf__ T* dstHead = maskUb;
        for (uint16_t i = 0; i < headLoop; ++i) {
            LoadAlign(vreg_head1, srcHead + i * 512);
            LoadAlign(vreg_head2, srcHead + i * 512 + halfStride);
            StoreAlign(dstHead + i * 512, vreg_head1, preg_all);
            StoreAlign(dstHead + i * 512 + halfStride, vreg_head2, preg_all);
        }

        // mid
        RegTensor<T> vreg_mid1;
        RegTensor<T> vreg_mid2;
        for (uint16_t row = 0; row < midRowCount; ++row) {
            __ubuf__ T* srcRow = maskSecUb + row * 512;
            LoadAlign(vreg_mid1, srcRow);
            LoadAlign(vreg_mid2, srcRow + halfStride);
            for (uint16_t sg = 1; sg < midGLoop; ++sg) {
                __ubuf__ T* dstRow = maskUb + (headLoop + sg * midRowCount + row) * 512;
                StoreAlign(dstRow, vreg_mid1, preg_all);
                StoreAlign(dstRow + halfStride, vreg_mid2, preg_all);
            }
        }

        // tail
        RegTensor<T> vreg_tail1;
        RegTensor<T> vreg_tail2;
        __ubuf__ T* srcTail = maskSecUb;
        __ubuf__ T* dstTailBase = maskUb + (headLoop + midGLoop * midRowCount) * 512;
        for (uint16_t i = 0; i < tailLoop; ++i) {
            LoadAlign(vreg_tail1, srcTail + i * 512);
            LoadAlign(vreg_tail2, srcTail + i * 512 + halfStride);
            StoreAlign(dstTailBase + i * 512, vreg_tail1, preg_all);
            StoreAlign(dstTailBase + i * 512 + halfStride, vreg_tail2, preg_all);
        }
    } else {
        // constexpr uint32_t repeatStride = s2BaseSize >> 5;
        // // head
        // RegTensor<T> vreg_head;
        // __ubuf__ T * srcHead = maskSecUb + (midRowCount - headLoop) * s2BaseSize;
        // __ubuf__ T * dstHead = maskUb;
        // for (uint16_t i = 0; i < headLoop; ++i) {
        //     LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
        //         vreg_head, srcHead + i * s2BaseSize, 1, repeatStride, preg_all);
        //     StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
        //         dstHead + i * s2BaseSize, vreg_head, 1, repeatStride, preg_all);
        // }

        // // mid
        // RegTensor<T> vreg_mid;
        // for (uint16_t row = 0; row < midRowCount; ++row) {
        //     __ubuf__ T * srcRow = maskSecUb + row * s2BaseSize;
        //     LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
        //         vreg_mid, srcRow, 1, repeatStride, preg_all);
        //     for (uint16_t sg = 0; sg < midGLoop; ++sg) {
        //         __ubuf__ T * dstRow = maskUb + (headLoop + sg * midRowCount + row) * s2BaseSize;
        //         StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
        //             dstRow, vreg_mid, 1, repeatStride, preg_all);
        //     }
        // }

        // // tail
        // RegTensor<T> vreg_tail;
        // __ubuf__ T * srcTail = maskSecUb;
        // __ubuf__ T * dstTailBase = maskUb + (headLoop + midGLoop * midRowCount) * s2BaseSize;
        // for (uint16_t i = 0; i < tailLoop; ++i) {
        //     LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
        //         vreg_tail, srcTail + i * s2BaseSize, 1, repeatStride, preg_all);
        //     StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
        //         dstTailBase + i * s2BaseSize, vreg_tail, 1, repeatStride, preg_all);
        // }
    }
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForGsLayout(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                                      MaskInfo& info, TEventID eventId, bool isPre = false)
{
    int32_t s1StartIdx = info.gs1StartIdx % info.s1Size;
    int32_t s1EndIdx = (info.gs1StartIdx + info.gs1dealNum - 1) % info.s1Size + 1;
    uint32_t attenMaskS2Stride = Align(info.s2dealNum, 32U) + 32 * info.attenMaskDstStride;
    if (info.gs1dealNum <= info.s1Size) {
        if (s1StartIdx + info.gs1dealNum > info.s1Size) {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, info.s1Size, isPre);
            LocalTensor<T> attenMaskSecUb = attenMaskUb[(info.s1Size - s1StartIdx) * attenMaskS2Stride];
            AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, s1EndIdx, isPre);
        } else {
            AttentionmaskDataCopy(attenMaskUb, srcGmAddr, info, s1StartIdx, s1EndIdx, isPre);
        }
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    } else {
        uint32_t headS1Count = info.s1Size - s1StartIdx;
        uint32_t remainRowCount = info.gs1dealNum - headS1Count;
        uint32_t midGCount = remainRowCount / info.s1Size;
        uint32_t tailS1Size = remainRowCount % info.s1Size;

        // 第一块完整的mask
        LocalTensor<T> attenMaskSecUb = attenMaskUb[headS1Count * attenMaskS2Stride];
        AttentionmaskDataCopy(attenMaskSecUb, srcGmAddr, info, 0, info.s1Size, isPre);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        __ubuf__ T* maskUb = (__ubuf__ T*)attenMaskUb.GetPhyAddr();
        __ubuf__ T* maskSecUb = (__ubuf__ T*)attenMaskSecUb.GetPhyAddr();
        MaskUbCopyGS1VF<T, s2BaseSize>(maskUb, maskSecUb, headS1Count, info.s1Size, midGCount, tailS1Size);
        // // TODO，待优化，后续改成VF
        // // head
        // DataCopy(attenMaskUb, attenMaskUb[info.s1Size * attenMaskS2Stride], headS1Count * attenMaskS2Stride);
        // // mid
        // for (uint32_t i = 1; i < midGCount; i++) {
        //     DataCopy(attenMaskUb[(headS1Count + i * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
        //              info.s1Size * attenMaskS2Stride);
        // }
        // // tail
        // if (tailS1Size > 0) {
        //     DataCopy(attenMaskUb[(headS1Count + midGCount * info.s1Size) * attenMaskS2Stride], attenMaskSecUb,
        //              tailS1Size * attenMaskS2Stride);
        // }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr,
                                                      MaskInfo& info, TEventID eventId, bool isPre = false)
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
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * info.attenMaskDstStride;

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
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};
        DataCopyPad(attenMaskUb[headGSize * attenMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    MaskUbCopyS1G<T, s2BaseSize>(attenMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
}

// TODO，老模板收编完成后删除该函数
template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyGS1(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr, MaskInfo& info,
                                            TEventID eventId, bool isPre = false)
{
    if (info.layout == LAYOUT_Q::GS) {
        AttentionmaskCopyInForGsLayout<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, eventId, isPre);
    } else if (info.layout == LAYOUT_Q::SG) {
        AttentionmaskCopyInForSgLayout<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, eventId, isPre);
    }
}

template <typename T, LAYOUT_Q layoutQ, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T>& attenMaskUb, GlobalTensor<T>& srcGmAddr, MaskInfo& info,
                                           TEventID eventId, bool isPre = false)
{
    if constexpr (layoutQ == LAYOUT_Q::GS) {
        AttentionmaskCopyInForGsLayout<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, eventId, isPre);
    } else if constexpr (layoutQ == LAYOUT_Q::SG) { // sg
        AttentionmaskCopyInForSgLayout<T, isReconstructTemp, s2BaseSize>(attenMaskUb, srcGmAddr, info, eventId, isPre);
    }
}
} // namespace fa_base_vector_gs1
#endif
