/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef APPLY_RAOTRY_POS_EMB_REGBASE_H
#define APPLY_RAOTRY_POS_EMB_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#define __CCE_UT_TEST__

#define TILING_KEY_ABA 20010
#define TILING_KEY_BA 20011
#define TILING_KEY_BAB 20020
#define TILING_KEY_AB 20030

struct ApplyRotaryPosEmbRegbaseTilingData {
    int64_t B = 0;
    int64_t CosB = 0;
    int64_t S = 0;
    int64_t D = 0;
    int64_t QN = 0;
    int64_t KN = 0;
    int64_t blockNumB = 0;
    int64_t blockFactorB = 0;
    int64_t blockNumS = 0;
    int64_t blockFactorS = 0;
    int64_t ubLoopNumS = 0;
    int64_t ubFactorS = 0;
    int64_t ubTailFactorS = 0;
    int64_t ubLoopNumB = 0;
    int64_t ubFactorB = 0;
    int64_t ubTailFactorB = 0;
    int64_t ubLoopNumQN = 0;
    int64_t ubFactorQN = 0;
    int64_t ubTailFactorQN = 0;
    int64_t ubLoopNumKN = 0;
    int64_t ubFactorKN = 0;
    int64_t ubTailFactorKN = 0;
    int64_t rotaryMode = 0;
};

struct ApplyRotaryPosEmbRegbaseABTilingData {
    int64_t B = 0;
    int64_t CosB = 0;
    int64_t S = 0;
    int64_t D = 0;
    int64_t QN = 0;
    int64_t KN = 0;
    int64_t dAlign = 0;
    int64_t dSplitCoef = 0;
    int64_t blockNumBS = 0;
    int64_t blockFactorBS = 0;
    int64_t blockTailBS = 0;
    int64_t ubLoopN = 0;
    int64_t ubFactorN = 0;
    int64_t ubTailN = 0;
    int64_t rotaryMode = 0;
    int64_t isCos1D = 0;
};

#define DTYPE_QUERY float

inline void InitTilingData(uint8_t *tiling, ApplyRotaryPosEmbRegbaseTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(ApplyRotaryPosEmbRegbaseTilingData));
}

inline void InitTilingData(uint8_t *tiling, ApplyRotaryPosEmbRegbaseABTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(ApplyRotaryPosEmbRegbaseABTilingData));
}

#undef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#endif
