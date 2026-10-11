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
 * \file incre_flash_attention_arch22.h
 * \brief
 */
#include "incre_flash_attention_preload_dd.h"

template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitMetaData(const __gm__ uint8_t* p_metadata, T* metadata)
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(ASCENDC_CPU_DEBUG) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || \
    defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64((uint8_t*)metadata, (__gm__ uint8_t*)p_metadata, all_bytes);
#else
    __ubuf__ uint8_t* metadata_in_ub = (__ubuf__ uint8_t*)get_imm(0);
    constexpr uint32_t len_burst = (all_bytes + 31) / 32;
    copy_gm_to_ubuf(((__ubuf__ uint8_t*)metadata_in_ub), p_metadata, 0, 1, len_burst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64((uint8_t*)metadata, (__ubuf__ uint8_t*)metadata_in_ub, all_bytes);
#endif
}

#define INVOKE_IFA_NO_KFC_DD_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<IFAType<__VA_ARGS__>> op; \
        const optiling::IncreFlashAttentionTilingData* __restrict tiling_data = &tiling; \
        const IncreFlashAttentionMetaData* __restrict meta_data = nullptr; \
        IncreFlashAttentionMetaData metaDataTmp; \
        if (metaData != nullptr) { \
            InitMetaData<IncreFlashAttentionMetaData>(metaData, &metaDataTmp); \
            meta_data = &metaDataTmp; \
        } \
        op.blockNum = GetBlockNum(); \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, \
                kvPaddingSize, meta_data, attentionOut, softmaxLse, user, tiling_data, &tPipe); \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user); \
        op.Process(); \
    } while (0)

#define INVOKE_IFA_NO_KFC_DD_OP_IMPL_SK(templateClass, ...) \
    do { \
        templateClass<IFAType<__VA_ARGS__>> op; \
        const optiling::IncreFlashAttentionTilingData* __restrict tiling_data = &tiling; \
        const IncreFlashAttentionMetaData* __restrict meta_data = nullptr; \
        IncreFlashAttentionMetaData metaDataTmp; \
        if (metaData != nullptr) { \
            InitMetaData<IncreFlashAttentionMetaData>(metaData, &metaDataTmp); \
            meta_data = &metaDataTmp; \
        } \
        if ASCEND_IS_AIC { \
            op.blockNum = sysArgs->skNumBlocks; \
        } else { \
            op.blockNum = sysArgs->skNumBlocks / 2; \
        } \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, \
                kvPaddingSize, meta_data, attentionOut, softmaxLse, user, tiling_data, &tPipe); \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user); \
        op.Process(); \
    } while (0)

template <uint8_t FLASH_DECODE, uint8_t LAYOUT_T, uint8_t ANTIQUANT_MODE>
__global__ __mix__(1, 2) void incre_flash_attention(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
    __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2, __gm__ uint8_t* quantScale2,
    __gm__ uint8_t* quantOffset2, __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset, __gm__ uint8_t* valueAntiquantScale,
    __gm__ uint8_t* valueAntiquantOffset, __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix,
    __gm__ uint8_t* actualSharedPrefixLen, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
    __gm__ uint8_t* keyRopeAntiquantScale, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* metaData,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    optiling::IncreFlashAttentionTilingData tiling)
{
    TPipe tPipe;
    __gm__ uint8_t* user = workspace;

    INVOKE_IFA_NO_KFC_DD_OP_IMPL(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t, true,
                                 FLASH_DECODE, static_cast<LAYOUT>(LAYOUT_T), ANTIQUANT_MODE, false, LAYOUT::NZ,
                                 AMLAMODE::NORMAL);
}

// SK Adapt
struct SKArgsStruct {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR pseShift;
    GM_ADDR attenMask;
    GM_ADDR actualSeqLengthsQ;
    GM_ADDR actualSeqLengths;
    GM_ADDR deqScale1;
    GM_ADDR quantScale1;
    GM_ADDR deqScale2;
    GM_ADDR quantScale2;
    GM_ADDR quantOffset2;
    GM_ADDR antiquantScale;
    GM_ADDR antiquantOffset;
    GM_ADDR blocktable;
    GM_ADDR queryPaddingSize;
    GM_ADDR kvPaddingSize;
    GM_ADDR keyAntiquantScale;
    GM_ADDR keyAntiquantOffset;
    GM_ADDR valueAntiquantScale;
    GM_ADDR valueAntiquantOffset;
    GM_ADDR keySharedPrefix;
    GM_ADDR valueSharedPrefix;
    GM_ADDR actualSharedPrefixLen;
    GM_ADDR queryRope;
    GM_ADDR keyRope;
    GM_ADDR keyRopeAntiquantScale;
    GM_ADDR dequantScaleQuery;
    GM_ADDR metaData;
    GM_ADDR attentionOut;
    GM_ADDR softmaxLse;
    GM_ADDR workspace;
    optiling::IncreFlashAttentionTilingData tiling;
};

template <uint8_t FLASH_DECODE, uint8_t LAYOUT_T, uint8_t ANTIQUANT_MODE, uint32_t splitNum>
__sk__ void incre_flash_attention_sk(const SKArgsStruct* args, const sk::SkSystemArgs* sysArgs)
{
    GM_ADDR query = args->query;
    GM_ADDR key = args->key;
    GM_ADDR value = args->value;
    GM_ADDR pseShift = args->pseShift;
    GM_ADDR attenMask = args->attenMask;
    GM_ADDR actualSeqLengthsQ = args->actualSeqLengthsQ;
    GM_ADDR actualSeqLengths = args->actualSeqLengths;
    GM_ADDR deqScale1 = args->deqScale1;
    GM_ADDR quantScale1 = args->quantScale1;
    GM_ADDR deqScale2 = args->deqScale2;
    GM_ADDR quantScale2 = args->quantScale2;
    GM_ADDR quantOffset2 = args->quantOffset2;
    GM_ADDR antiquantScale = args->antiquantScale;
    GM_ADDR antiquantOffset = args->antiquantOffset;
    GM_ADDR blocktable = args->blocktable;
    GM_ADDR queryPaddingSize = args->queryPaddingSize;
    GM_ADDR kvPaddingSize = args->kvPaddingSize;
    GM_ADDR keyAntiquantScale = args->keyAntiquantScale;
    GM_ADDR keyAntiquantOffset = args->keyAntiquantOffset;
    GM_ADDR valueAntiquantScale = args->valueAntiquantScale;
    GM_ADDR valueAntiquantOffset = args->valueAntiquantOffset;
    GM_ADDR keySharedPrefix = args->keySharedPrefix;
    GM_ADDR valueSharedPrefix = args->valueSharedPrefix;
    GM_ADDR actualSharedPrefixLen = args->actualSharedPrefixLen;
    GM_ADDR queryRope = args->queryRope;
    GM_ADDR keyRope = args->keyRope;
    GM_ADDR keyRopeAntiquantScale = args->keyRopeAntiquantScale;
    GM_ADDR dequantScaleQuery = args->dequantScaleQuery;
    GM_ADDR metaData = args->metaData;
    GM_ADDR attentionOut = args->attentionOut;
    GM_ADDR softmaxLse = args->softmaxLse;
    GM_ADDR workspace = args->workspace;
    const optiling::IncreFlashAttentionTilingData& tiling = args->tiling;

    TPipe tPipe;
    __gm__ uint8_t* user = workspace;

    INVOKE_IFA_NO_KFC_DD_OP_IMPL_SK(IncreFlashAttentionAttenPreloadDD, bfloat16_t, int8_t, bfloat16_t, bfloat16_t, true,
                                    FLASH_DECODE, static_cast<LAYOUT>(LAYOUT_T), ANTIQUANT_MODE, false, LAYOUT::NZ,
                                    AMLAMODE::NORMAL);
    tPipe.DestroyWithoutPipeAll();
}

#define FA_SK_BIND(FLASH_DECODE, LAYOUT_T, ANTIQUANT_MODE) \
    SK_BIND(incre_flash_attention<FLASH_DECODE, LAYOUT_T, ANTIQUANT_MODE>, 4, \
            incre_flash_attention_sk<FLASH_DECODE, LAYOUT_T, ANTIQUANT_MODE, 0>, \
            incre_flash_attention_sk<FLASH_DECODE, LAYOUT_T, ANTIQUANT_MODE, 1>, \
            incre_flash_attention_sk<FLASH_DECODE, LAYOUT_T, ANTIQUANT_MODE, 2>, \
            incre_flash_attention_sk<FLASH_DECODE, LAYOUT_T, ANTIQUANT_MODE, 3>)

FA_SK_BIND(0, 0, 0);
FA_SK_BIND(0, 0, 1);
FA_SK_BIND(0, 1, 0);
FA_SK_BIND(0, 1, 1);
FA_SK_BIND(1, 0, 0);
FA_SK_BIND(1, 0, 1);
FA_SK_BIND(1, 1, 0);
FA_SK_BIND(1, 1, 1);
