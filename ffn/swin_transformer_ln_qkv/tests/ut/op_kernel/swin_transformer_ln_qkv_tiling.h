/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SWIN_ATTENTION_FFN_TILING_H_
#define _SWIN_ATTENTION_FFN_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct SwinTransformerLnQKVBaseInfo {
    uint32_t inputsize;
    uint32_t hSize;
    uint32_t baseLoopNum;
    uint32_t remainderBlockNum;
};

struct SwinTransformerLnQKVLayernormTilingData {
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t bsLength;
    uint32_t shLength;
    uint32_t loopSize;
    uint32_t elementPerBlock;
    uint32_t remainderElementPerBlock;
    uint32_t innerLoopLength;
    uint32_t innerLoopNum;
    uint32_t normalBlockElementOffset;
    uint32_t rollOffset;
};

struct SwinTransformerLnQKVTilingData {
    uint32_t maxCoreNum;
    uint32_t useVectorNum;
    uint32_t workspaceSize;
    uint32_t inputSizeSum;
    SwinTransformerLnQKVBaseInfo opBaseInfo;
    SwinTransformerLnQKVLayernormTilingData layernormTilingParams;
    TCubeTiling mmTilingParams;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, SwinTransformerLnQKVTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(SwinTransformerLnQKVTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, SwinTransformerLnQKVTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SwinTransformerLnQKVTilingData));
}
#endif


#define GET_TILING_DATA(tiling_data, tiling_arg) \
    SwinTransformerLnQKVTilingData tiling_data;        \
    InitTilingData(tiling_arg, &tiling_data)


#endif