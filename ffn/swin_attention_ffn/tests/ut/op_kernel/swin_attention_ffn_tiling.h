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

struct SwinAttentionFFNTilingData {
    uint32_t batchSize;
    uint32_t bmmFormerNum;
    uint32_t bmmTailNum;
    uint32_t bmmFormerBatchNum;
    uint32_t bmmTailBatchNum;
    uint32_t aivNum;
    uint32_t shift1;
    uint32_t shift2;
    uint32_t tpBlockSize;
    uint32_t tpSpaceCnt;
    uint32_t tpSpaceH;
    uint32_t tpSpaceW;
    uint32_t blockInSpace;
    uint32_t tpSpaceSize;
    uint32_t tpSpaceWTransposed;
    uint32_t tpSpaceHTransposed;
    TCubeTiling bmmTilingData;
    uint32_t dataNumPerBatchA;
    uint32_t dataNumPerBatchD;
    uint32_t dataNumPerLoop;
    uint32_t reserved;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, SwinAttentionFFNTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(SwinAttentionFFNTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, SwinAttentionFFNTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(SwinAttentionFFNTilingData));
}
#endif


#define GET_TILING_DATA(tiling_data, tiling_arg) \
    SwinAttentionFFNTilingData tiling_data;        \
    InitTilingData(tiling_arg, &tiling_data)


#endif