/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "incre_flash_attention_allvec_new.h"
#if (__CCE_AICORE__ > 200)
#include "incre_flash_attention_split_Bbn2s2_Us2.h"
#include "incre_flash_attention_preload.h"
#include "incre_flash_attention_preload_dd.h"

// #ifdef FIA_ENABLE_MLA
// // mla模板使用私有tiling结构，框架编译时根据一组DType预编译获取keylist，根据keylist找到对应的tiling结构
// // 在这组DType中，若没有mla模板的key，包含mla模板编译会报错：unknown type name 'IncreFlashAttentionTilingDataMla'
// #if ((ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_INT8))
#include "incre_flash_attention_preload_mla.h"
// #endif
// #endif // FIA_ENABLE_MLA

#endif

#include "incre_flash_attention_template_tiling_key.h"
#include "incre_flash_attention_tiling.h"
using namespace AscendC;

#define NEED_CUBE_TILING (true)
#define NOT_NEED_CUBE_TILING (false)

#define INVOKE_IFA_GENERAL_OP_IMPL_PREFIX(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_PREFIX(tiling, NEED_CUBE_TILING);                                                             \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling, op.mm1Sp,              \
                          bmm1tilingPrefix, op.mm2Sp, bmm2tilingPrefix);                                               \
        op.InitPrefix(query, keySharedPrefix, valueSharedPrefix, pseShift, attenMask, actualSharedPrefixLen,           \
                      blocktable, kvPaddingSize, attentionOut, softmaxLse, user, &tiling_data->tilingPrefix, tiling,   \
                      &tPipe);                                                                                         \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
        SyncAll(); /* workspace改为每个核单独使用即可去掉此处同步 */                                                      \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, &tiling_data->tilingBase, tiling, nullptr);                                          \
        op.Process();                                                                                                  \
        op.ProcessSysPrefixCombine();                                                                                  \
    } while (0)

#define INVOKE_IFA_GENERAL_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA(tiling, NEED_CUBE_TILING);                                                                    \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                       \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, tiling_data, tiling, &tPipe);                                                        \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_ALL_VEC_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_NO_CUBE(tiling);                                                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, tiling_data, &tPipe);                                                                \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     user);                                                                                            \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_NO_KFC_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_ALL(tiling);                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, blocktable, kvPaddingSize, attentionOut,     \
                softmaxLse, user, tiling_data, tiling, &tPipe);                                                        \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_IFA_ANTIQUANT_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingAtbDataV2, tiling_data_in, tiling);                       \
        const IncreFlashAttentionTilingAtbDataV2 *__restrict tiling_data = &tiling_data_in;                            \
        op.Init(query, key, value, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, attentionOut, user,     \
                tiling_data);                                                                                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, user);          \
        op.Process();                                                                                                  \
    } while (0)

#ifdef FIA_ENABLE_MLA
#define INVOKE_IFA_NO_KFC_MLA_OP_IMPL(templateClass, ...)                                                              \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        COPY_TILING_DATA_MLA(tiling);                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, kvPaddingSize,\
            queryRope, keyRope, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,                 \
                     keyRopeAntiquantScale, dequantScaleQuery, user);                                                  \
        op.Process();                                                                                                  \
    } while (0)

#else
#define INVOKE_IFA_NO_KFC_MLA_OP_IMPL(templateClass, ...)  do {} while (0)
#endif

#define COPY_TILING_DATA_ALL(tiling)                                                                                   \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;                                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr

#define INVOKE_IFA_NEW_GQA_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingAtbDataV2, tiling_data_in, tiling);                       \
        const IncreFlashAttentionTilingAtbDataV2 *__restrict tiling_data = &tiling_data_in;                            \
        op.Init(query, key, value, pseShift, actualSeqLengths, blocktable, attentionOut, user, tiling_data);           \
        op.Process();                                                                                                  \
    } while (0)


#define COPY_TILING_DATA_MLA(tiling)                                                                                   \
    GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingDataMla, tiling_data_in, tiling);                             \
    const IncreFlashAttentionTilingDataMla *__restrict tiling_data = &tiling_data_in


#ifdef __DAV_C220_CUBE__
#define COPY_TILING_DATA(tiling, need_cube)                                                                            \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);     \
    const IncreFlashAttentionTilingData *__restrict tiling_data = nullptr;                                             \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar

#define COPY_TILING_DATA_PREFIX(tiling, need_cube)                                                                     \
    if constexpr (!need_cube) {                                                                                        \
        return;                                                                                                        \
    }                                                                                                                  \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm1TilingData, bmm1TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase.bmm2TilingData, bmm2TilingDataVar, tiling);     \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingPrefix.base.bmm1TilingData, bmm1TilingDataVarPrefix, \
                           tiling);                                                                                    \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingPrefix.base.bmm2TilingData, bmm2TilingDataVarPrefix, \
                           tiling);                                                                                    \
    const IncreFlashAttentionTilingDataV2 *__restrict tiling_data = nullptr;                                           \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm1tilingPrefix = &bmm1TilingDataVarPrefix;                                         \
    const TCubeTiling *__restrict bmm2tilingPrefix = &bmm2TilingDataVarPrefix

#define COPY_TILING_DATA_NO_CUBE(tiling) COPY_TILING_DATA(tiling, NOT_NEED_CUBE_TILING)

#else
#if (__CCE_AICORE__ != 310) && (!defined (__DAV_310R6__))
#define COPY_TILING_DATA(tiling, need_cube)                                                                            \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in;                                     \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr

#define COPY_TILING_DATA_PREFIX(tiling, need_cube)                                                                     \
    GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingDataV2, tiling_data_in, tiling);                              \
    const IncreFlashAttentionTilingDataV2 *__restrict tiling_data = &tiling_data_in;                                   \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm1tilingPrefix = nullptr;                                                          \
    const TCubeTiling *__restrict bmm2tilingPrefix = nullptr

#define COPY_TILING_DATA_NO_CUBE(tiling)                                                                               \
    GET_TILING_DATA_MEMBER(IncreFlashAttentionTilingDataV2, tilingBase, tiling_data_in, tiling);                       \
    const IncreFlashAttentionTilingData *__restrict tiling_data = &tiling_data_in

#endif
#endif

template<int LAYOUT_T>
__aicore__ constexpr LAYOUT get_layout_type() {
    if constexpr (LAYOUT_T == 0) {
        return LAYOUT::BNSD;
    } else if constexpr (LAYOUT_T == 1) {
        return LAYOUT::BSH;
    } else if constexpr (LAYOUT_T == 2) {
        return LAYOUT::TND;
    }
    return LAYOUT::BNSD;
}

template<int KV_LAYOUT_T>
__aicore__ constexpr LAYOUT get_kv_layout_type() {
    if constexpr (KV_LAYOUT_T == 0) {
        return LAYOUT::BNSD;
    } else if (KV_LAYOUT_T == 1) {
        return LAYOUT::BSH;
    } else if (KV_LAYOUT_T == 2) {
        return LAYOUT::NZ;
    }
    return LAYOUT::BNSD;
}

template<int M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA>
__aicore__ constexpr AMLAMODE get_amla_mode() {
    if constexpr (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) {
        return AMLAMODE::NORMAL;
    } else if (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) {
        return AMLAMODE::AMLA;
    }   
    return AMLAMODE::NORMAL;
}

template<uint8_t Q_T, uint8_t KV_T, uint16_t OUT_T, uint8_t PAGE_ATTENTIOND, uint16_t LAYOUT_T,  uint16_t KV_LAYOUT_T, uint16_t FLASH_DECODE,
            uint8_t ENABLE_PREFIX, uint8_t M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, uint8_t M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, uint8_t M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA,  uint16_t M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE, 
            uint16_t M_FIAFLAG_P_MMTYPETMP_I_MODEVAL,   uint8_t P_CVDIFF_BASE_FLAG, uint8_t P_CVDIFF_MLA_FLAG,  uint8_t P_TEMPLATE_VERSION, uint8_t TEMPLATE_MODE>
__global__ __aicore__ void incre_flash_attention_FIAS(
    __gm__ uint8_t *query, 
    __gm__ uint8_t *key,
    __gm__ uint8_t *value, 
    __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, 
    __gm__ uint8_t *actualSeqLengthsQ, 
    __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, 
    __gm__ uint8_t *quantScale1, 
    __gm__ uint8_t *deqScale2, 
    __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, 
    __gm__ uint8_t *antiquantScale, 
    __gm__ uint8_t *antiquantOffset, 
    __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, 
    __gm__ uint8_t *kvPaddingSize, 
    __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, 
    __gm__ uint8_t *valueAntiquantScale, 
    __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *keySharedPrefix, 
    __gm__ uint8_t *valueSharedPrefix, 
    __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *queryRope, 
    __gm__ uint8_t *keyRope, 
    __gm__ uint8_t *keyRopeAntiquantScale, 
    __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, 
    __gm__ uint8_t *workspace, 
    __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    constexpr LAYOUT LAYOUT_TYPE = get_layout_type<LAYOUT_T>();
    constexpr LAYOUT KV_LAYOUT_TYPE = get_kv_layout_type<KV_LAYOUT_T>();
    constexpr AMLAMODE AMLA_TYPE = get_amla_mode<M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA>();
    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
#if (__CCE_AICORE__ > 200)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    #ifndef FIA_IFA
    REGISTER_TILING_DEFAULT(IncreFlashAttentionTilingDataV2);
    #endif
#if (__CCE_AICORE__ > 200)
    if constexpr (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 1) {
        using Q_TYPE = half;
        using KV_TYPE = std::conditional_t<KV_T == 0, half,
                        std::conditional_t<KV_T == 3, int8_t,
                        half>>;
        using OUT_TYPE = half;
        using ORIGIN_TYPE = half;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataV2);
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE);
    } else if (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 0) {
        using Q_TYPE = std::conditional_t<Q_T == 0, half,
                       std::conditional_t<Q_T == 2, bfloat16_t,
                       half>>;
        using KV_TYPE = std::conditional_t<KV_T == 0, half,
                        std::conditional_t<KV_T == 2, bfloat16_t,
                        std::conditional_t<KV_T == 3, int8_t,
                        std::conditional_t<KV_T == 4, int4b_t,
                        half>>>>;
        using OUT_TYPE = std::conditional_t<OUT_T == 0, half,
                          std::conditional_t<OUT_T == 2, bfloat16_t,
                          std::conditional_t<OUT_T == 3, int8_t,
                          half>>>;
        using ORIGIN_TYPE = std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0, half,
                            std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2, bfloat16_t,
                            half>>;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataV2);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, false);
    } else if (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 2) {
        using Q_TYPE = std::conditional_t<Q_T == 0, half,
                       std::conditional_t<Q_T == 2, bfloat16_t,
                       half>>;
        using KV_TYPE = std::conditional_t<KV_T == 0, half,
                        std::conditional_t<KV_T == 2, bfloat16_t,
                        std::conditional_t<KV_T == 3, int8_t,
                        std::conditional_t<KV_T == 4, int4b_t,
                        half>>>>;
        using OUT_TYPE = std::conditional_t<OUT_T == 0, half,
                         std::conditional_t<OUT_T == 2, bfloat16_t,
                         std::conditional_t<OUT_T == 3, int8_t,
                         half>>>;
        using ORIGIN_TYPE = std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0, half,
                            std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2, bfloat16_t,
                            half>>;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataV2);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, false);
    } else if (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2) {
        using Q_TYPE = std::conditional_t<Q_T == 0, half,
                       std::conditional_t<Q_T == 2, bfloat16_t,
                       half>>;
        using KV_TYPE = std::conditional_t<KV_T == 0, half,
                        std::conditional_t<KV_T == 2, bfloat16_t,
                        std::conditional_t<KV_T == 3, int8_t,
                        std::conditional_t<KV_T == 4, int4b_t,
                        half>>>>;
        using OUT_TYPE = std::conditional_t<OUT_T == 0, half,
                         std::conditional_t<OUT_T == 2, bfloat16_t,
                         std::conditional_t<OUT_T == 3, int8_t,
                         half>>>;
        using ORIGIN_TYPE = std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0, half,
                            std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2, bfloat16_t,
                            half>>;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataV2);
        INVOKE_IFA_GENERAL_OP_IMPL(IncreFlashAttentionAttenSplitBbn2s2Us2, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, true);
    } else if (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 3) {
        using Q_TYPE = std::conditional_t<Q_T == 0, half,
                       std::conditional_t<Q_T == 2, bfloat16_t,
                       half>>;
        using KV_TYPE = std::conditional_t<KV_T == 0, half,
                        std::conditional_t<KV_T == 2, bfloat16_t,
                        std::conditional_t<KV_T == 3, int8_t,
                        std::conditional_t<KV_T == 4, int4b_t,
                        half>>>>;
        using OUT_TYPE = std::conditional_t<OUT_T == 0, half,
                         std::conditional_t<OUT_T == 2, bfloat16_t,
                         std::conditional_t<OUT_T == 3, int8_t,
                         half>>>;
        using ORIGIN_TYPE = std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0, half,
                            std::conditional_t<M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2, bfloat16_t,
                            half>>;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataV2);
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreload, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, false);
    } else if (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 6) {
        using Q_TYPE = half;
        using KV_TYPE = int8_t;
        using OUT_TYPE = bfloat16_t;
        using ORIGIN_TYPE = bfloat16_t;
        INVOKE_IFA_NO_KFC_OP_IMPL(IncreFlashAttentionAttenPreloadDD, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, false, KV_LAYOUT_TYPE, AMLA_TYPE, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE);
    } else if (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 5) {
        using Q_TYPE = int8_t;
        using KV_TYPE = int8_t;
        using OUT_TYPE = bfloat16_t;
        using ORIGIN_TYPE = bfloat16_t;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataMla);
        INVOKE_IFA_NO_KFC_MLA_OP_IMPL(IncreFlashAttentionAttenPreloadMla, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, false, KV_LAYOUT_TYPE, AMLA_TYPE, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE);
    } 
#else
    REGISTER_TILING_DEFAULT(IncreFlashAttentionTilingData);
    if constexpr (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && P_CVDIFF_BASE_FLAG == 1) {
        using Q_TYPE = half;
        using KV_TYPE = std::conditional_t<KV_T == 0, half,
                        std::conditional_t<KV_T == 3, int8_t,
                        half>>;
        using OUT_TYPE = half;
        using ORIGIN_TYPE = half;
        REGISTER_TILING_FOR_TILINGKEY("TRUE", IncreFlashAttentionTilingDataV2);  
        INVOKE_IFA_ALL_VEC_OP_IMPL(IncreFlashAttentionAttenAllVecNew, Q_TYPE, KV_TYPE, OUT_TYPE, ORIGIN_TYPE, PAGE_ATTENTIOND, FLASH_DECODE,
                                    LAYOUT_TYPE, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, false);
    } 
#endif
}

template<uint8_t Q_T, uint8_t KV_T, uint8_t OUT_T, uint8_t PAGE_ATTENTIOND, uint8_t LAYOUT_T,  uint8_t KV_LAYOUT_T, uint8_t FLASH_DECODE,
            uint8_t ENABLE_PREFIX, uint8_t M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, uint8_t M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, uint8_t M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA,  uint8_t M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE, 
            uint8_t M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, uint8_t P_CVDIFF_BASE_FLAG, uint8_t P_CVDIFF_MLA_FLAG,  uint8_t P_TEMPLATE_VERSION, uint8_t TEMPLATE_MODE>
__global__ __aicore__ void
incre_flash_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
                      __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1,
                      __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                      __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                      __gm__ uint8_t *blocktable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    incre_flash_attention_FIAS<Q_T, KV_T, OUT_T, PAGE_ATTENTIOND, LAYOUT_T, KV_LAYOUT_T, FLASH_DECODE, ENABLE_PREFIX, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE,
                                    M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
                                    M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, P_CVDIFF_BASE_FLAG, P_CVDIFF_MLA_FLAG, P_TEMPLATE_VERSION, TEMPLATE_MODE>
          (query, key, value, pseShift, attenMask, nullptr, actualSeqLengths, deqScale1, quantScale1,
                               deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, nullptr,
                               kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                               attentionOut, nullptr, workspace, tiling);
}