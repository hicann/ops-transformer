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
 * \file fia_arch35_fullquant.h
 * \brief arch35 FIA 全量化路由
 */

#ifndef FIA_ARCH35_FULLQUANT_H_
#define FIA_ARCH35_FULLQUANT_H_

#include "fia_arch35_common.h"
#include "fia_arch35_template_tiling_key_enum.h"
#include "flash_attention_score_kernel_infer_mla_fullquant.h"
#include "fia_template_dispatcher.h"

using namespace regbaseutil;

#define INVOKE_MLA_FULLQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(templateClass, vec1ResultSize, qkvSize, ...) \
    do { \
        if (query == nullptr) { \
            return; \
        } \
        FIA_REGBASE_COPY_TILING_DATA(tiling); \
        TPipe tPipe; \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockCubeMlaFullquant<__VA_ARGS__>, \
                                      BaseApi::FABlockCubeMlaFullquantDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FABlockVecFullquantDummy<__VA_ARGS__>, \
                                      BaseApi::FABlockVecInferMlaFullquant<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths, \
                       actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, dequantScaleQuery, \
                       key_antiquant_scale, value_antiquant_scale, nullptr, postQuantScale, postQuantOffset, \
                       keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, learnableSink, \
                       nullptr, nullptr, nullptr, softmaxLse, attentionOut, user, tilingData, &tPipe); \
        op.Process(); \
    } while (0)

template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope,
          uint8_t KvLayoutType, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit,
          bool isReconstructTemp>
inline __aicore__ void fia_fullquant_regbase(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *deq_scale1, __gm__ uint8_t *quant_scale1, __gm__ uint8_t *deq_scale2,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *antiquant_scale,
    __gm__ uint8_t *antiquant_offset, __gm__ uint8_t *blocktable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *key_antiquant_scale, __gm__ uint8_t *key_antiquant_offset,
    __gm__ uint8_t *value_antiquant_scale, __gm__ uint8_t *value_antiquant_offset, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr bool isPa = KvLayoutType != 0;
#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && \
     ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    if constexpr (isReconstructTemp == true) {
        if constexpr (quantMode == FULLQUANT_MODE_QKV_MXFP8_PREFILL || quantMode == FULLQUANT_MODE_QKV_MXFP8_DECODE) {
            run_fia_fullquant_mx_kernel<fp8_e4m3fn_t, half, inOutLayoutType, config, pseMode, quantMode, hasAttenMask,
                                        hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(
                query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                dequantScaleQuery, key_antiquant_scale, value_antiquant_scale, quant_scale1, queryRope, keyRope,
                attentionOut, softmaxLse, user, tiling);
        } else if constexpr (quantMode == FULLQUANT_MODE_QK_PER_TOKEN_HEAD_V_PER_HEAD) {
            run_fia_fullquant_gqa_kernel<fp8_e4m3fn_t, half, inOutLayoutType, config, pseMode, quantMode, hasAttenMask,
                                         hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(
                query, key, value, pseShift, actualSeqLengths, actualSeqLengthsKV, blocktable, dequantScaleQuery,
                key_antiquant_scale, value_antiquant_scale, quant_scale1, queryRope, keyRope, attentionOut, softmaxLse,
                user, tiling);
        }
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_HIFLOAT8 && ORIG_DTYPE_KEY == DT_HIFLOAT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (quantMode == FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR) { // mla fullquant
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dVTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dVTemplateType) >> 1)) *
            2;
        INVOKE_MLA_FULLQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionScoreKernelInferMlaFullquant, vec1ResultSize, qkvSizeRsv2, hifloat8_t, float,
            bfloat16_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    if constexpr (isReconstructTemp == true) {
        if constexpr (quantMode == FULLQUANT_MODE_QKV_MXFP8_PREFILL || quantMode == FULLQUANT_MODE_QKV_MXFP8_DECODE) {
            run_fia_fullquant_mx_kernel<fp8_e4m3fn_t, bfloat16_t, inOutLayoutType, config, pseMode, quantMode,
                                        hasAttenMask, hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix,
                                        enableS1OutSplit>(query, key, value, pseShift, attenMask, actualSeqLengths,
                                                          actualSeqLengthsKV, blocktable, dequantScaleQuery,
                                                          key_antiquant_scale, value_antiquant_scale, quant_scale1,
                                                          queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        } else if constexpr (quantMode == FULLQUANT_MODE_QK_PER_TOKEN_HEAD_V_PER_HEAD) {
            run_fia_fullquant_gqa_kernel<fp8_e4m3fn_t, bfloat16_t, inOutLayoutType, config, pseMode, quantMode,
                                         hasAttenMask, hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix,
                                         enableS1OutSplit>(query, key, value, pseShift, actualSeqLengths,
                                                           actualSeqLengthsKV, blocktable, dequantScaleQuery,
                                                           key_antiquant_scale, value_antiquant_scale, quant_scale1,
                                                           queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        } else if constexpr (quantMode == FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR) {
            run_fia_fullquant_mla_kernel<fp8_e4m3fn_t, bfloat16_t, inOutLayoutType, config, pseMode, quantMode,
                                         hasAttenMask, hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix,
                                         enableS1OutSplit>(query, key, value, pseShift, attenMask, actualSeqLengths,
                                                           actualSeqLengthsKV, blocktable, dequantScaleQuery,
                                                           key_antiquant_scale, value_antiquant_scale, quant_scale1,
                                                           queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        }
    } else {
        PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd,
                               emptyTensor, enableKVPrefix);
        constexpr uint64_t vec1ResultSize =
            static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr (quantMode == FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR) { // mla fullquant
            constexpr uint64_t qkvSizeRsv2 =
                MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                        (static_cast<uint64_t>(dVTemplateType) >> 1),
                    static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dVTemplateType) >> 1)) *
                2;
            INVOKE_MLA_FULLQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
                BaseApi::FlashAttentionScoreKernelInferMlaFullquant, vec1ResultSize, qkvSizeRsv2, fp8_e4m3fn_t, float,
                bfloat16_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType,
                dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true,
                isPa, isFd, enableKVPrefix);
        }
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_INT8 && ORIG_DTYPE_KEY == DT_INT8 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                           enableKVPrefix);
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (quantMode == FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR) { // mla fullquant
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dVTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dVTemplateType) >> 1)) *
            2;
        INVOKE_MLA_FULLQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionScoreKernelInferMlaFullquant, vec1ResultSize, qkvSizeRsv2, int8_t, float, bfloat16_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix);
    }
#endif
}

#endif // FIA_ARCH35_FULLQUANT_H_
