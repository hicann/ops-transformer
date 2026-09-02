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
 * \file flash_attention_score_entry.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_ENTRY_310_H_
#define FLASH_ATTENTION_SCORE_ENTRY_310_H_
#include "flash_attention_score_drop_mask_adapter_regbase.h"
#include "flash_attention_score_kernel_train.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __DAV_C310_CUBE__ // CUBE 实现

#define INVOKE_FA_OP_IMPL_BASEAPI(templateClass, ...) \
    do { \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        TPipe tPipe; \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FANoQuantBlockCube<__VA_ARGS__>, \
                                      BaseApi::FANoQuantBlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FANoQuantBlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::FANoQuantBlockVecTrain<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, pse, dropMask, paddingMask, attenMask, prefix, actualSeqLengths, \
                       actualSeqLengthsKv, nullptr, nullptr, nullptr, deqScaleQ, deqScaleK, deqScaleV, pScale, \
                       nullptr, nullptr, nullptr, nullptr, nullptr, queryRope, keyRope, sink, softmaxMax, softmaxSum, \
                       softmaxOut, nullptr, attentionOut, user, nullptr, &tPipe); \
        op.Process(); \
    } while (0)
#else // VECTOR 实现
#define REGBASE_COPY_TILING_DATA(tiling) \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling); \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn;

#ifdef __CCE_KT_TEST__
#define INVOKE_FA_OP_IMPL_BASEAPI(templateClass, ...) \
    do { \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        REGBASE_COPY_TILING_DATA(tiling); \
        TPipe tPipe; \
        if (tilingData->inputParamsRegbase.needDropMaskOp) { \
            FlashAttentionScoreDropMaskAdapterRegbase dropMaskAdapter; \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe); \
            dropMaskAdapter.Process(); \
            tPipe.Reset(); \
        } \
        using CubeBlockType = typename BaseApi::FANoQuantBlockCube<__VA_ARGS__>; \
        using VecBlockType = typename BaseApi::FANoQuantBlockVecTrain<__VA_ARGS__>; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, pse, dropMask, paddingMask, attenMask, prefix, actualSeqLengths, \
                       actualSeqLengthsKv, nullptr, nullptr, nullptr, deqScaleQ, deqScaleK, deqScaleV, pScale, \
                       nullptr, nullptr, queryRope, keyRope, softmaxMax, softmaxSum, softmaxOut, nullptr, \
                       attentionOut, user, tilingData, &tPipe); \
        op.Process(); \
    } while (0)
#else
#define INVOKE_FA_OP_IMPL_BASEAPI(templateClass, ...) \
    do { \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        REGBASE_COPY_TILING_DATA(tiling); \
        TPipe tPipe; \
        if (tilingData->inputParamsRegbase.needDropMaskOp) { \
            FlashAttentionScoreDropMaskAdapterRegbase dropMaskAdapter; \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe); \
            dropMaskAdapter.Process(); \
            tPipe.Reset(); \
        } \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FANoQuantBlockCube<__VA_ARGS__>, \
                                      BaseApi::FANoQuantBlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FANoQuantBlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::FANoQuantBlockVecTrain<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, pse, dropMask, paddingMask, attenMask, prefix, actualSeqLengths, \
                       actualSeqLengthsKv, nullptr, nullptr, nullptr, deqScaleQ, deqScaleK, deqScaleV, pScale, \
                       nullptr, nullptr, nullptr, nullptr, nullptr, queryRope, keyRope, sink, softmaxMax, softmaxSum, \
                       softmaxOut, nullptr, attentionOut, user, tilingData, &tPipe); \
        op.Process(); \
    } while (0)
#endif
#endif

template <uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, uint16_t s2TemplateType, uint16_t dTemplateType,
          uint16_t dvTemplateType, uint8_t pseMode, bool hasAtten, bool hasDrop, bool hasRope, uint8_t outDtype,
          uint8_t regbase, bool optionalDn>
inline __aicore__ void flash_attention_score_regbase(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *sink, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
#if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_QUERY == DT_FLOAT)
    INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, float, float, float, ImplModeEnum(implMode),
                              LayOutTypeEnum(layout), S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                              DTemplateType(dTemplateType),
                              DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType), PseTypeEnum(pseMode),
                              hasAtten, hasDrop, hasRope, false, false, false, false, false, optionalDn);
    return;
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
    INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, bfloat16_t, float, bfloat16_t,
                              ImplModeEnum(implMode), LayOutTypeEnum(layout), S1TemplateType(s1TemplateType),
                              S2TemplateType(s2TemplateType), DTemplateType(dTemplateType),
                              DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType), PseTypeEnum(pseMode),
                              hasAtten, hasDrop, hasRope, false, false, false, false, false, optionalDn);
    return;
#endif
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, half, float, half, ImplModeEnum(implMode),
                              LayOutTypeEnum(layout), S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                              DTemplateType(dTemplateType),
                              DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType), PseTypeEnum(pseMode),
                              hasAtten, hasDrop, hasRope, false, false, false, false, false, optionalDn);
    return;
#endif

// fp8
#if (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
    INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, hifloat8_t, float, bfloat16_t,
                              ImplModeEnum(implMode), LayOutTypeEnum(layout), S1TemplateType(s1TemplateType),
                              S2TemplateType(s2TemplateType), DTemplateType(dTemplateType),
                              DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType), PseTypeEnum(pseMode),
                              hasAtten, hasDrop, hasRope, false, false, false, false, false, optionalDn);
    return;
#endif
#endif
}
#endif // end of FLASH_ATTENTION_SCORE_ENTRY_310_H_
