/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_arch38_entry_regbase.h
 * \brief
 */

#ifndef FIA_ARCH38_ENTRY_REGBASE_H_
#define FIA_ARCH38_ENTRY_REGBASE_H_

#include "flash_attention_score_kernel_infer_regbase_v2.h"
#include "fia_arch38_dummy.h"
namespace optiling {};

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(templateClass, ...) \
    do { \
        if (query == nullptr) { \
            return; \
        } \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling); \
        const FlashAttentionScoreSimplifiedTilingData* __restrict tilingData = &tilingDataIn; \
        TPipe tPipe; \
        using CubeBlockType = BaseApi::FABlockCube<__VA_ARGS__>; \
        using VecBlockType = BaseApi::FABlockVecInfer<__VA_ARGS__>; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths, \
                       actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, nullptr, nullptr, deq_scale2, \
                       deq_scale1, quant_scale1, postQuantScale, postQuantOffset, queryRope, keyRope, nullptr, \
                       nullptr, nullptr, softmaxLse, attentionOut, user, tilingData, &tPipe); \
        op.Process(); \
    } while (0)

// kv is empty tensor, return zero output
#define INVOKE_FIA_DUMMY(templateClass, ...) \
    do { \
        TPipe tPipe; \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling); \
        const FlashAttentionScoreSimplifiedTilingData* __restrict tilingData = &tilingDataIn; \
        FiaArch38Dummy<half> op; \
        op.Init(attentionOut, tilingData); \
        op.Process(); \
        return \
    } while (0)

inline __aicore__ void fia_arch38_regbase(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
    __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
    __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1, __gm__ uint8_t* deq_scale2,
    __gm__ uint8_t* postQuantScale, __gm__ uint8_t* postQuantOffset, __gm__ uint8_t* antiquant_scale,
    __gm__ uint8_t* antiquant_offset, __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
    __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
    __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset, __gm__ uint8_t* keySharedPrefix,
    __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen, __gm__ uint8_t* queryRope,
    __gm__ uint8_t* keyRope, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
#if (__NPU_ARCH__ == 5102)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TILING_KEY_IS(1002312000040001212);
    TILING_KEY_IS(1002312000040021212);
    TILING_KEY_IS(1001311000000001212);
    TILING_KEY_IS(1001311000000021212);
    TILING_KEY_IS(1002122000000021012);
    TILING_KEY_IS(1001121000000021012);
    // attenMask使能场景，与上面各key一一对应：mask位(1e8)置1，且DN位(4e7)被host强制清零
    TILING_KEY_IS(1002312000100001212);
    TILING_KEY_IS(1002312000100021212);
    TILING_KEY_IS(1001311000100001212);
    TILING_KEY_IS(1001311000100021212);
    TILING_KEY_IS(1002122000100021012);

    TILING_KEY_IS(1002312000040001012);
    TILING_KEY_IS(1002312000100001012);
    TILING_KEY_IS(1003113000100001012);
    TILING_KEY_IS(1001311000100001012);

    TILING_KEY_IS(1000000000000000090);

#if TILING_KEY_VAR == 1002312000040001212
    // BNSD layout HighPerformance, No mask, No pse, 常量化, D128, useDn
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002312000040021212
    // BNSD layout HighPerformance, No mask, No pse, 常量化, D128, useDn
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1001311000000001212
    // BNSD layout HighPerformance, No mask, No pse, 常量化, D128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned64,
        DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1001311000000021212
    // BNSD layout HighPerformance, No mask, No pse, 常量化, D64
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned64,
        DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002122000000021012
    // BNSD layout HighPerformance, No mask, No pse, 常量化, D128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned64, S2TemplateType::Aligned256, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1001121000000021012
    // BNSD layout HighPerformance, No mask, No pse, 常量化, D64, S1Base=64, S2Base=256, int8输出
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned64, S2TemplateType::Aligned256, DTemplateType::Aligned64,
        DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002312000100001212
    // BNSD layout HighPerformance, With mask, ND, No pse, 常量化, D128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002312000100021212
    // BNSD layout HighPerformance, With mask, ND, No pse, 常量化, D128, out int8
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1001311000100001212
    // BNSD layout HighPerformance, With mask, ND, No pse, 常量化, D64
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned64,
        DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1001311000100021212
    // BNSD layout HighPerformance, With mask, ND, No pse, 常量化, D64, out int8
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, int8_t, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned64,
        DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002122000100021012
    // BNSD layout HighPerformance, With mask, ND, No pse, 常量化, D128, S1=64/S2=256
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, int8_t, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned64, S2TemplateType::Aligned256, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002312000040001012
    // BNSD layout HighPerformance, no mask, no pse, D128, FP16 noquant, FP16 out, S1=128/S2=128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, false, false, false, true, false, false);
#elif TILING_KEY_VAR == 1002312000100001012
    // BNSD layout HighPerformance, with mask, no pse, D128, FP16 noquant, FP16 out, S1=128/S2=128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned128,
        DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1003113000100001012
    // BNSD layout HighPerformance, with mask, no pse, D256, FP16 noquant, FP16 out, S1=64/S2=128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned64, S2TemplateType::Aligned128, DTemplateType::Aligned256,
        DTemplateType::Aligned256, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1001311000100001012
    // BNSD layout HighPerformance, with mask, no pse, D64, FP16 noquant, FP16 out, S1=128/S2=128
    INVOKE_FIA_GENERAL_OP_IMPL_REGBASE_V2_FA_BASEAPI(
        BaseApi::FlashAttentionScoreKernelInferRegbaseV2, half, half, half, ImplModeEnum::AA_HIGH_PERFORMANCE,
        LayOutTypeEnum::LAYOUT_BNSD, S1TemplateType::Aligned128, S2TemplateType::Aligned128, DTemplateType::Aligned64,
        DTemplateType::Aligned64, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, false, false);
#elif TILING_KEY_VAR == 1000000000000000090
    INVOKE_FIA_DUMMY();
#endif

#endif
}
#endif // end of FIA_ARCH38_ENTRY_REGBASE_H_
