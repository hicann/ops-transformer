/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEMORY_COPY_ARCH35_QUANT_FLASH_ATTN_H
#define MEMORY_COPY_ARCH35_QUANT_FLASH_ATTN_H

#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memcopy/gm_layout.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/offset_calculator_v2.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_l1_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../../common/op_kernel/memcopy/gm_coord.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_ub.h"
#include "../../../common/op_kernel/memcopy/copy_ub_to_gm.h"

// 格式转换, 暂时放在这里
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, bool useDn = false, bool isPerTokenHead = false>
__aicore__ inline constexpr GmFormat GetQueryScaleGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        if constexpr (isPerTokenHead) {
            return GmFormat::NGT;
        }
        if constexpr (!useDn) {
            return GmFormat::NTGD;
        } else {
            return GmFormat::TNGD;
        }
    } else {
        return GmFormat::TNGD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKeyScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ_K_SCALE;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetValueScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND2;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return UbFormat::GS1;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) { // v100 没有bsnd
        return keyTensorDesc.GetShape(1);                 // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

// ----------------------------------------------Copy LSE UB To Gm arch35--------------------------------
template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseBSNDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                    uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                    const CONST_INFO_T &constInfo, uint64_t s1LeftPaddingSize = 0)
{
    // USE_DN时，realGSize=1(G合入N2)
    uint32_t startS1Idx = mOffset / constInfo.realGSize;
    uint32_t startGIdx = mOffset % constInfo.realGSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.realGSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.realGSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.s1Size + s1Idx + s1LeftPaddingSize;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.realGSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.s1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseBNSDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                    uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                    const CONST_INFO_T &constInfo, uint64_t qActSeqLens,
                                                    uint64_t s1LeftPaddingSize = 0)
{
    uint64_t gOffset = mOffset / qActSeqLens * constInfo.s1Size;
    uint64_t seqOffset = mOffset % qActSeqLens;
    uint64_t outOffset = bN2Offset + gOffset + seqOffset + s1LeftPaddingSize;
    uint64_t ubOffset = 0;

    // dealCount ≤ 当前actQs剩余部分，则直接搬运全部dealCount
    if ((qActSeqLens - seqOffset) >= dealCount) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        return;
    }
    // dealCount > 当前actQs剩余部分，分块搬运dealCount
    // dealCount首块
    uint64_t headActSeq = qActSeqLens - seqOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = headActSeq;
    dataCopyParams.blockLen = sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    outOffset += constInfo.s1Size - qActSeqLens + headActSeq;
    // ubOffset += headActSeq * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    ubOffset += headActSeq * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    // dealCount中间块
    uint64_t pendingCount = dealCount - headActSeq;
    while (pendingCount > qActSeqLens) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = qActSeqLens;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += constInfo.s1Size;
        ubOffset += qActSeqLens * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        pendingCount -= qActSeqLens;
    }
    // dealCount尾块
    if (pendingCount > 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = pendingCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                   uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                   const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + s1Idx * constInfo.n2Size * constInfo.gSize + startGIdx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDArch35NoGS1Merge(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                             uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                             const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.realGSize;
    uint32_t startGIdx = mOffset % constInfo.realGSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.realGSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.realGSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + s1Idx * constInfo.realN2Size * constInfo.realGSize + startGIdx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.realGSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

// TND layout LSE 输出转成 N-major (N 在外, T 在内) 排布: [N2*G, T]
// bN2Offset = n2Idx * realGSize * t1Size + prefixBS1
// 单次写入: outOffset = bN2Offset + gIdx * t1Size + s1Idx
//         = n2Idx * realGSize * t1Size + prefixBS1 + gIdx * t1Size + s1Idx
// 即最终 GM 排布为 [N2, G, T], 与 host infershape 声明的 (N, T) 对齐 (N=N2*G)
template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge(GlobalTensor<float> softmaxLseGm,
                                                                 LocalTensor<T> lseSrc, uint64_t bN2Offset,
                                                                 uint32_t mOffset, uint32_t dealCount,
                                                                 const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.realGSize;
    uint32_t startGIdx = mOffset % constInfo.realGSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.realGSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.realGSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.t1Size + s1Idx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.realGSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        // 每写一个 float, 跳到下一个 gIdx 对应的 T 行起点: stride = (t1Size - 1) * sizeof(float)
        dataCopyParams.dstStride = (constInfo.t1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

// TND layout LSE 输出转成 N-major 排布 (非 NoGS1Merge 版本, 用 gSize/n2Size)
// 与 DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge 同语义, 仅把 realGSize 换成 gSize
template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseTNDtoNTArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                       uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                       const CONST_INFO_T &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.t1Size + s1Idx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.t1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void DataCopySoftmaxLseNTDArch35(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                                   uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                                   const CONST_INFO_T &constInfo, uint32_t s1Size)
{
    uint32_t startS1Idx = mOffset % s1Size;
    uint32_t startGIdx = mOffset / s1Size;
    uint32_t endS1Idx = (mOffset + dealCount - 1) % s1Size;
    uint32_t endGIdx = (mOffset + dealCount - 1) / s1Size;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t gIdx = startGIdx; gIdx <= endGIdx; gIdx++) {
        outOffset = bN2Offset + startS1Idx * constInfo.n2Size * constInfo.gSize + gIdx;
        if (gIdx != endGIdx) {
            curDealRowCount = s1Size - startS1Idx;
        } else {
            curDealRowCount = endS1Idx + 1 - startS1Idx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.gSize * constInfo.n2Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startS1Idx = 0;
        ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    }
}

#endif
