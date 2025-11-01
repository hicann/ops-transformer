/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _MOE_GATING_TOPK_SOFTMAX_V2_TILING_H_
#define _MOE_GATING_TOPK_SOFTMAX_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct MoeGatingTopKSoftmaxV2EKFullLoadTilingData {
    uint32_t row;
    uint32_t col;
    uint32_t colAlign;
    uint32_t k;
    uint32_t kAlignB16;
    uint32_t kAlignB32;
    uint32_t kAlignT;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoopOfFormerBlock;
    uint32_t ubLoopOfTailBlock;
    uint32_t ubFormer;
    uint32_t ubTailOfFormerBlock;
    uint32_t ubTailOfTailBlock;
    uint32_t softmaxFlag;
    uint32_t copyUbToUbBlockCount;
    SoftMaxTiling formerSoftmaxTilingData;
    SoftMaxTiling formerBlockTailSoftmaxTilingData;
    SoftMaxTiling tailBlockTailSoftmaxTilingData;
    TopkTiling formerTopkTilingData;
    TopkTiling formerBlockTailTopkTilingData;
    TopkTiling tailBlockTailTopkTilingData;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeGatingTopKSoftmaxV2EKFullLoadTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(MoeGatingTopKSoftmaxV2EKFullLoadTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeGatingTopKSoftmaxV2EKFullLoadTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeGatingTopKSoftmaxV2EKFullLoadTilingData));
}
#endif

#pragma pack(1)

struct MoeGatingTopKSoftmaxV2KFullLoadTilingData {
    uint32_t row;
    uint32_t col;
    uint32_t k;
    uint32_t kAlign;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoop;
    uint32_t ubFormer;
    uint32_t ubTail;
    uint32_t ubFormerAlign;
    uint32_t ubTailAlign;
    uint32_t softmaxFlag;
    SoftMaxTiling ubFormerSoftmaxTilingData;
    SoftMaxTiling ubTailSoftmaxTilingData;
    TopkTiling topkFormerTilingData;
    TopkTiling topkTailTilingData;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeGatingTopKSoftmaxV2KFullLoadTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(MoeGatingTopKSoftmaxV2KFullLoadTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeGatingTopKSoftmaxV2KFullLoadTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeGatingTopKSoftmaxV2KFullLoadTilingData));
}
#endif

#pragma pack(1)

struct MoeGatingTopKSoftmaxV2PerfTilingData {
    uint32_t row;
    uint32_t col;
    uint32_t colBytesAlign;
    uint32_t colAlign;
    uint32_t k;
    uint32_t kAlign;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoopOfFormerBlock;
    uint32_t ubLoopOfTailBlock;
    uint32_t ubFormer;
    uint32_t ubTailOfFormerBlock;
    uint32_t ubTailOfTailBlock;
    uint32_t topKValuesMask;
    uint32_t topKIndicesMask;
    uint32_t bufferElemSize;
    uint32_t softmaxFlag;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeGatingTopKSoftmaxV2PerfTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(MoeGatingTopKSoftmaxV2PerfTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeGatingTopKSoftmaxV2PerfTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MoeGatingTopKSoftmaxV2PerfTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    MoeInitRoutingTilingData tiling_data;        \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif
