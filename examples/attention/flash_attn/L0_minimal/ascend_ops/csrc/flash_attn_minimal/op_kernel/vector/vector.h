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
 * \file vector.h
 * \brief
 */
#ifndef VECTOR_H
#define VECTOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../fa_kernel_public.h"

using namespace fa_kernel::util;
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::LocalTensor;

namespace fa_base_vector {

enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    TREE = 9,
};

enum LAYOUT_Q {
    GS,
    SG,
};

enum MaskDataType : uint8_t {
    MASK_BOOL,
    MASK_INT8,
    MASK_UINT8,
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
    uint32_t attnMaskBatchStride;
    uint32_t attnMaskS1Stride; // 这个变量名与A3 代码不一致，统一整改
    uint32_t attnMaskDstStride = 0;

    LAYOUT_Q layout;
    MaskDataType attnMaskType;
    SparseMode sparseMode;
    uint32_t maskValue;

    uint64_t s1LeftPaddingSize = 0;
    uint64_t s2LeftPaddingSize = 0;
};

template <typename T, uint32_t s2BaseSize>
__simd_vf__ void MaskUbCopyS1GVF(__ubuf__ T * maskUb, uint16_t headGLoop, uint16_t midGLoop,
    uint16_t tailGLoop, uint16_t midS1Count)
{
    constexpr uint32_t repeatStride = s2BaseSize >> 5;

    RegTensor<T> vreg_g_head;
    RegTensor<T> vreg_g_mid;
    RegTensor<T> vreg_g_tail;

    MaskReg preg_all;
    if constexpr (s2BaseSize == 128) {
        preg_all = CreateMask<bool, MaskPattern::VL128>();
    } else {    // s2BaseSize = 256
        preg_all = CreateMask<bool, MaskPattern::ALL>();
    }

    for (uint16_t x = headGLoop; x > 0; x = 0) {     // if (headGLoop > 0) {}
        LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            vreg_g_head, maskUb, 1, repeatStride, preg_all);
        for (uint16_t i = 1; i < headGLoop; ++i) {
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maskUb, vreg_g_head, 1, repeatStride, preg_all);
        }
    }

    for (uint16_t x = midGLoop; x > 0; x = 0) {     // if (midGLoop > 0) {}
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

    for (uint16_t x = tailGLoop; x > 1; x = 0) {     // if (tailGLoop > 1) {}
        LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            vreg_g_tail, maskUb, 1, repeatStride, preg_all);
        for (uint16_t i = 1; i < tailGLoop; ++i) {
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maskUb, vreg_g_tail, 1, repeatStride, preg_all);
        }
    }
}

template <typename T, uint32_t s2BaseSize>
__aicore__ inline void MaskUbCopyS1G(const LocalTensor<T>& maskTensor, int32_t headGSize, int32_t gSize,
    int32_t tailGSize, int32_t midS1Count)
{
    if (gSize <= 1) {
        return;
    }

    __ubuf__ T * maskUb = (__ubuf__ T*)maskTensor.GetPhyAddr();
    MaskUbCopyS1GVF<T, s2BaseSize>(maskUb, headGSize, gSize, tailGSize, midS1Count);
}

__aicore__ inline uint64_t ComputeAttnMaskOffsetCompress(MaskInfo &info, uint32_t s1StartIdx)
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
    uint32_t attnMaskSizeAlign = AlignUp(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attnMaskSizeAlign) ? delta : attnMaskSizeAlign) *
                 info.attnMaskS1Stride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttnMaskOffset(MaskInfo &info, uint32_t s1StartIdx = 0)
{
    return ComputeAttnMaskOffsetCompress(info, s1StartIdx);
}

__aicore__ inline bool IsSkipAttentionmask(MaskInfo &info)
{
    if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
        return false;
    }

    int32_t s1StartIdx = info.layout == GS ? info.gs1StartIdx % info.s1Size : info.gs1StartIdx / info.gSize;
    if (info.layout == GS && s1StartIdx + info.gs1dealNum > info.s1Size) { // 当跨多个s1时，不再支持跳过计算
        return false;
    }

    int64_t nextToken = 0; // sparse2 本身原点就在左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算Token
    } else if (info.sparseMode == BAND) {                                          // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }

    if (static_cast<int64_t>(info.s2StartIdx + info.s2dealNum) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
        return true;
    }
    return false;
}

template <typename T>
__aicore__ inline void AttentionmaskDataCopy(LocalTensor<T> &attnMaskUb, GlobalTensor<T> &srcGmAddr, MaskInfo &info,
                                             uint32_t s1StartIdx, uint32_t s1EndIdx)
{
    uint32_t attnMaskSizeAlign = AlignUp(info.s2dealNum, 32U);
    uint64_t maskOffset = ComputeAttnMaskOffset(info, s1StartIdx);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attnMaskS1Stride - info.s2dealNum;
    dataCopyParams.dstStride = info.attnMaskDstStride;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attnMaskSizeAlign - info.s2dealNum), 1U};
    DataCopyPad(attnMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

template <typename T, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyInForSgLayout(LocalTensor<T> &attnMaskUb, GlobalTensor<T> &srcGmAddr,
                                                      MaskInfo &info)
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
    uint32_t attnMaskSizeAlign = AlignUp(info.s2dealNum, 32U);
    uint32_t attnMaskS2Stride = attnMaskSizeAlign + 32 * info.attnMaskDstStride;

    // ub-head
    if (headGSize > 0) {
        AttentionmaskDataCopy(attnMaskUb, srcGmAddr, info, s1StartIdx, s1StartIdx + 1);
        s1StartIdx++;
    }

    // ub-remain
    if (remainRowCount > 0) {
        uint64_t maskOffset = ComputeAttnMaskOffset(info, s1StartIdx);
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = midS1Count + (tailGSize > 0);
        dataCopyParams.blockLen = info.s2dealNum;
        dataCopyParams.srcStride = info.attnMaskS1Stride - info.s2dealNum;
        dataCopyParams.dstStride = (info.gSize - 1) * attnMaskS2Stride / 32 + info.attnMaskDstStride;
        // blockLen + padParams.len + dstStride = gSize * s2Size
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attnMaskSizeAlign - info.s2dealNum), 1U};
        DataCopyPad(attnMaskUb[headGSize * attnMaskS2Stride], srcGmAddr[maskOffset], dataCopyParams, padParams);
    }

    event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(enQueEvtID);
    WaitFlag<HardEvent::MTE2_V>(enQueEvtID);

    MaskUbCopyS1G<T, s2BaseSize>(attnMaskUb, headGSize, info.gSize, tailGSize, midS1Count);
}

template <typename T, LAYOUT_Q layoutQ, bool isReconstructTemp, uint32_t s2BaseSize>
__aicore__ inline void AttentionmaskCopyIn(LocalTensor<T> &attnMaskUb, GlobalTensor<T> &srcGmAddr, MaskInfo &info)
{
    AttentionmaskCopyInForSgLayout<T, isReconstructTemp, s2BaseSize>(attnMaskUb, srcGmAddr, info);
}

} // namespace fa_base_vector
#endif
