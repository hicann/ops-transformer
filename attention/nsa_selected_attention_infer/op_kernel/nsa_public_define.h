/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_public_define.h
 * \brief
 */

#ifndef NSA_PUBLIC_DEFINE_H
#define NSA_PUBLIC_DEFINE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::SoftmaxConfig;

constexpr float FLOAT_ZERO = 0;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);

enum class LAYOUT {
    BSH = 0,
    SBH,
    BNSD,
    BSND,
    TND
};
constexpr SoftmaxConfig IFA_SOFTMAX_FLASHV2_CFG = {false}; // 将isCheckTiling设置为false

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
        const bool FLASH_DECODE = false, LAYOUT LAYOUT_T = LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
        const bool SHARED_PREFIX = false, typename... Args>
struct NSAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
};

__aicore__ inline void RowMuls(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                            uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;

    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            // offset = i * dtypeMask
            Mul(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParams);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask,
                columnRepeatCount, columnRepeatParams);
        }
    }

    if (dRemain > 0) {
        Mul(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParams);
    }
}

__aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                            uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

#endif // NSA_PUBLIC_DEFINE_H