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
 * \file moe_re_routing_v2_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#if __has_include("../moe_re_routing/arch35/moe_re_routing_re_regbase.h")
#include "../moe_re_routing/arch35/moe_re_routing_re_regbase.h"
#include "../moe_re_routing/arch35/moe_re_routing_r_regbase.h"
#else
#include "../../moe_re_routing/op_kernel/arch35/moe_re_routing_re_regbase.h"
#include "../../moe_re_routing/op_kernel/arch35/moe_re_routing_r_regbase.h"
#endif

using namespace MoeReRouting;

#define MOE_RE_ROUTING_RE_WITHOUT_SCALE 200000UL
#define MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT 200100UL
// Note: MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0 key covers multiple quantization type combinations:
// FP8/HIF8/FP4 with float8_e8m0 or float32 scale. All reuse int8_t template since they are 1-byte packed data.
#define MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0 200200UL
#define MOE_RE_ROUTING_R_WITHOUT_SCALE 210000UL
#define MOE_RE_ROUTING_R_WITH_SCALE_FLOAT 210100UL
#define MOE_RE_ROUTING_R_WITH_SCALE_FLOAT8_E8M0 210200UL

template <typename OpType>
__aicore__ void InitAndProcess(OpType &op, GM_ADDR tokens, GM_ADDR expertTokenNumPerRank, GM_ADDR perTokenScales,
                               GM_ADDR expertTopkWeight, GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales,
                               GM_ADDR permuteTokenIdx, GM_ADDR expertTokenNum, GM_ADDR permuteTopkWeight)
{
    op.Init(tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight, permuteTokens, permutePerTokenScales,
            permuteTokenIdx, expertTokenNum, permuteTopkWeight);
    op.Process();
}

extern "C" __global__ __aicore__ void moe_re_routing_v2(GM_ADDR tokens, GM_ADDR expertTokenNumPerRank,
                                                        GM_ADDR perTokenScales, GM_ADDR expertTopkWeight,
                                                        GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales,
                                                        GM_ADDR permuteTokenIdx, GM_ADDR expertTokenNum,
                                                        GM_ADDR permuteTopkWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    if (TILING_KEY_IS(MOE_RE_ROUTING_RE_WITHOUT_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingReTilingData, tiling_data_in, tiling);
        const MoeReRoutingReTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value &&
                      !IsSameType<DTYPE_TOKENS, hifloat8_t>::value && !IsSameType<DTYPE_TOKENS, fp4x2_e2m1_t>::value &&
                      !IsSameType<DTYPE_TOKENS, fp4x2_e1m2_t>::value) {
            MoeReRoutingReRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, false> op(&pipe, tilingData);
            InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                           permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
        } else {
            MoeReRoutingReRegbase<int8_t, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, false> op(&pipe, tilingData);
            InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                           permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingReTilingData, tiling_data_in, tiling);
        const MoeReRoutingReTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value &&
                      !IsSameType<DTYPE_TOKENS, hifloat8_t>::value && !IsSameType<DTYPE_TOKENS, fp4x2_e2m1_t>::value &&
                      !IsSameType<DTYPE_TOKENS, fp4x2_e1m2_t>::value) {
            MoeReRoutingReRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, true> op(&pipe, tilingData);
            InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                           permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_RE_WITH_SCALE_FLOAT8_E8M0)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingReTilingData, tiling_data_in, tiling);
        const MoeReRoutingReTilingData *__restrict tilingData = &tiling_data_in;
        MoeReRoutingReRegbase<int8_t, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, int8_t, true> op(&pipe, tilingData);
        InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                       permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_R_WITHOUT_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingRTilingData, tiling_data_in, tiling);
        const MoeReRoutingRTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value &&
                      !IsSameType<DTYPE_TOKENS, hifloat8_t>::value && !IsSameType<DTYPE_TOKENS, fp4x2_e2m1_t>::value &&
                      !IsSameType<DTYPE_TOKENS, fp4x2_e1m2_t>::value) {
            MoeReRoutingRRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, false> op(&pipe, tilingData);
            InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                           permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
        } else {
            MoeReRoutingRRegbase<int8_t, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, false> op(&pipe, tilingData);
            InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                           permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_R_WITH_SCALE_FLOAT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingRTilingData, tiling_data_in, tiling);
        const MoeReRoutingRTilingData *__restrict tilingData = &tiling_data_in;
        if constexpr (!IsSameType<DTYPE_TOKENS, fp8_e5m2_t>::value && !IsSameType<DTYPE_TOKENS, fp8_e4m3fn_t>::value &&
                      !IsSameType<DTYPE_TOKENS, hifloat8_t>::value && !IsSameType<DTYPE_TOKENS, fp4x2_e2m1_t>::value &&
                      !IsSameType<DTYPE_TOKENS, fp4x2_e1m2_t>::value) {
            MoeReRoutingRRegbase<DTYPE_TOKENS, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, float, true> op(&pipe, tilingData);
            InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                           permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
        }
    } else if (TILING_KEY_IS(MOE_RE_ROUTING_R_WITH_SCALE_FLOAT8_E8M0)) {
        GET_TILING_DATA_WITH_STRUCT(MoeReRoutingRTilingData, tiling_data_in, tiling);
        const MoeReRoutingRTilingData *__restrict tilingData = &tiling_data_in;
        MoeReRoutingRRegbase<int8_t, DTYPE_EXPERT_TOKEN_NUM_PER_RANK, int8_t, true> op(&pipe, tilingData);
        InitAndProcess(op, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
                       permuteTokens, permutePerTokenScales, permuteTokenIdx, expertTokenNum, permuteTopkWeight);
    }
}
