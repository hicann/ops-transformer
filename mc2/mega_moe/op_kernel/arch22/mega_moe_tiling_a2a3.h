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
 * \file mega_moe_tiling_a2a3.h
 * \brief
 */

#ifndef ASCENDC_MEGA_MOE_TILING_H
#define ASCENDC_MEGA_MOE_TILING_H

#include "moe_permute_prologue/moe_permute_prologue_tiling.h"

using namespace Mc2Tiling;

#define MEGA_MOE_QUANT_MODE_NO_QUANT 0
#define MEGA_MOE_QUANT_MODE_PER_TOKEN 1

#define MEGA_MOE_QUANT_OUT_TYPE_UNDEFINED 0
#define MEGA_MOE_QUANT_OUT_TYPE_INT8 1
#define MEGA_MOE_QUANT_OUT_TYPE_INT4 2

#define SOC_ASCEND910B 0
#define SOC_ASCEND910_93 1

struct MegaMoeA2A3TilingData {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t listLen;

    uint32_t moeExpertNum;
    uint32_t epWorldSize;
    uint32_t dispatchQuantMode;
    int64_t cclBufferSize;
    uint64_t maxRecvTokenNum;

    int32_t dispatchQuantOutDtype;
    uint32_t combineQuantMode;
    uint32_t commAlgCode;
    uint32_t numMaxTokensPerRank;
    uint32_t activationCode;
    float activationClamp;
    float activationParams1;
    float activationParams2;
    uint32_t isTransposeW1;
    uint32_t isTransposeW2;

    uint32_t hasBias1;
    uint32_t hasBias2;
    uint32_t hasXActiveMask;
    uint32_t hasScales;

    uint32_t isQuantRouting;
    uint32_t isW4A8;

    int32_t activationOutDtype;
    uint32_t weight1Interleave;

    // A3 接收侧 chunk（轮次切分）：
    // recvRoundBudget = 单轮接收预算 B（route 行，= min(maxRecvTokenNum, PERMUTE_CHUNK*EP*min(topK,epr))）
    // recvRoundsMax   = 轮表份数上限 = ceil(PERMUTE_CHUNK*EP*min(topK,epr) / B)，≥1
    // A2（发送侧 chunk）：recvRoundsMax = 1，B = maxRecvTokenNum（无接收侧轮次切分；
    //   单 chunk 接收行数 <= 总接收行数 <= maxRecvTokenNum，kernel 的 maxOutputSize 取 B）。
    uint64_t recvRoundBudget;
    uint64_t recvRoundsMax;
};

struct MegaMoeTilingDataQuant {
    MegaMoeA2A3TilingData common;
};

struct MegaMoeTilingDataNonQuant {
    MegaMoeA2A3TilingData common;
};
#endif
