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
 * \file memory_copy.h
   GM->L1
   PA
   PARope
 * \brief
 */
#ifndef MEMORY_COPY_H
#define MEMORY_COPY_H
#include "offset_calculator.h"

static constexpr uint64_t BYTE_BLOCK = 32UL;

template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor, uint32_t nValue,
                                              uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    constexpr uint32_t SRCDVALUE_LIMIT = 65536;
    Nd2NzParams nd2nzPara;
    if (unlikely(srcDValue < SRCDVALUE_LIMIT)) {
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = nValue; // nd矩阵的行数
        nd2nzPara.dValue = dValue;       // nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    } else {
        constexpr uint32_t BLOCK_ELEMENT_CNT = BYTE_BLOCK / sizeof(T);
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < nValue; i++) {
            nd2nzPara.ndNum = 1;
            nd2nzPara.nValue = 1;
            nd2nzPara.dValue = dValue;
            nd2nzPara.srcDValue = dValue;
            nd2nzPara.dstNzC0Stride = dstNzC0Stride;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = 0;
            nd2nzPara.dstNzMatrixStride = 0;
            DataCopy(l1Tensor[l1Offset], gmTensor[gmOffset], nd2nzPara);

            gmOffset += srcDValue;
            l1Offset += BLOCK_ELEMENT_CNT;
        }
    }
}

template <typename INPUT_T, GmFormat GM_FORMAT>
__aicore__ inline void CopyMatrixGmToL1(FaL1Tensor<INPUT_T> &dstTensor,
    FaGmTensor<INPUT_T, GM_FORMAT> &srcGm,
    uint32_t bIdx, uint32_t nIdx, uint32_t sIdx, uint32_t sDealSize)
{
    auto &info = srcGm.offsetInfo;
    uint64_t offset = GetOffset(info, bIdx, nIdx, sIdx, 0);
    CopySingleMatrixNDToNZ(dstTensor.tensor, srcGm.gmTensor[offset],
        sDealSize, info.dimD, info.strideS, dstTensor.rowCount);
}

template <typename OUT_T>
__aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                      uint32_t blockCount, uint32_t blockLen, uint32_t srcStride, uint64_t dstStride)
{
    DataCopyExtParams dataCopyParams;
    // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
    if (dstStride > UINT32_MAX) {
        uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
        uint64_t ubSingleStride = (srcStride * BYTE_BLOCK + blockLen) / sizeof(OUT_T);
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0; // 单位为Byte
        for (uint32_t i = 0; i < blockCount; i++) {
            DataCopyPad(gmTensor[i * gmSingleStride], ubTensor[i * ubSingleStride], dataCopyParams);
        }
    } else {
        // dataCopyParams.dstStride(uint32_t)没有溢出时，进行跳写
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = srcStride;
        dataCopyParams.dstStride = dstStride; // 单位为Byte
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }
}

// ----------------------------------------------CopyAttnOutBN1UbToGm--------------------------------
template <typename OUT_T>
__aicore__ inline void CopyAttnOutBN1UbToGm(FaGmTensor<OUT_T, GmFormat::BSND> &dstTensor,
                                            FaUbTensor<OUT_T> &srcTensor, uint32_t bIdx, uint32_t n1Head,
                                            uint32_t s1Start)
{
    auto &info = dstTensor.offsetInfo;
    uint64_t gmOffset = GetOffset(info, bIdx, n1Head, s1Start, 0);
    uint32_t blockCount = srcTensor.rowCount;
    uint32_t blockLen = srcTensor.colCount * sizeof(OUT_T);
    uint64_t dstStride = (info.strideS - srcTensor.colCount) * sizeof(OUT_T);
    SafeStrideCopy(dstTensor.gmTensor[gmOffset], srcTensor.tensor, blockCount, blockLen, 0, dstStride);
}
#endif