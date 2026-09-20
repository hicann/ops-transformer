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
 * \file lightning_indexer_v2_const_info_common.h
 * \brief Common ConstInfo fields shared by LightningIndexerV2 and QuantLightningIndexerV2.
 */
#ifndef LIGHTNING_INDEXER_V2_CONST_INFO_COMMON_H
#define LIGHTNING_INDEXER_V2_CONST_INFO_COMMON_H

namespace LIV2Common {

template <typename LayoutT>
struct ConstInfoCommon {
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // 8字节字段
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t kSeqSize = 0ULL; // kv最大S长度
    uint64_t qSeqSize = 1ULL; // q最大S长度

    // 4字节字段
    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;
    // 基本块大小
    uint32_t mBaseSize = 1U;
    uint32_t s1BaseSize = 1U;
    uint32_t s2BaseSize = 1U;
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t keyStride0 = 0;
    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    LayoutT outputLayout;         // 输出的格式

    // 1字节字段
    bool attenMaskFlag = false;
    bool isAccumSeqS1 = false; // 是否累加模式
    bool isAccumSeqS2 = false; // 是否累加模式
    bool isLDOpen = false;
    bool returnValue = false;
};

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_CONST_INFO_COMMON_H
