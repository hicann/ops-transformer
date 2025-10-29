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
 * \file prompt_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "prompt_flash_attention_tilingkey.h"
#if (__CCE_AICORE__ > 200)
#include "prompt_flash_attention_base.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tailWBNSD.h"
#include "prompt_flash_attention_bnstilling_n_s_tailWBNSD.h"
#include "prompt_flash_attention_s1s2_bns1_x910.h"
#include "prompt_flash_attention_base_api.h"
#include "prompt_flash_attention_base_api_high_precision_no_mask.h"
#include "prompt_flash_attention_base_mla.h"
#include "prompt_flash_attention_base_mla_high_precision.h"
#include "prompt_flash_attention_s1s2_bns1_mla.h"
#include "prompt_flash_attention_var_len_score_sab.h"
#include "prompt_flash_attention_s1s2_bns1_mla_baseapi.h"
#include "prompt_flash_attention_var_len_score_sab_baseapi.h"
#include "prompt_flash_attention_empty_tensor.h"
#include "prompt_flash_attention_tiling_data.h"
#include "prompt_flash_attention_template_tiling_key.h"
#else
#include "prompt_attention_prefill.h"
#include "prompt_flash_attention_s1s2_bns1_x310_base.h"
#include "prompt_flash_attention_s1s2_bns1_x310.h"
#include "prompt_flash_attention_tiling_data.h"
#include "prompt_flash_attention_template_tiling_key.h"
#endif

#define INVOKE_PFA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitMsd(key_antiquant_scale, key_antiquant_offset,value_antiquant_scale, value_antiquant_offset);                     \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_INT8_OP_IMPL(templateClass, ...)                                                                     \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.InitMsd(key_antiquant_scale, key_antiquant_offset,value_antiquant_scale, value_antiquant_offset);            \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_KVANTIQUANT_OP_IMPL(templateClass, ...)                                                              \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitKvAntiquant(antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                              \
        matmul::l1Global = l1Array;                                                                                     \
        matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                      \
        op.Init(query, key, value, attenMask, attentionOut, softmaxLse, user, tilingData, &tPipe);                                  \
        op.Process();                                                                                                   \
    } while (0)
// PFA TND
#define INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2, bmm2tiling); \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, queryRope, keyRope, attentionOut, softmaxLse, user, tilingData, &tPipe);   \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_NO_KFC_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, queryRope, keyRope, blocktable, learnableSink, \
                     attentionOut, softmaxLse, user, tilingData, &tPipe);                                               \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                        \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, nullptr, \
                deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, nullptr);                                    \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_MLA(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                        \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, nullptr, nullptr); \
    } while (0)
#define INVOKE_PFA_NO_KFC_TILING_DATA_MLA(tiling)                                                                      \
    GET_TILING_DATA_WITH_STRUCT(MLAGeneralTilingData, tiling_data_in, tiling);                                         \
    const MLAGeneralTilingData *__restrict tilingData = &tiling_data_in;                                               \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr
#ifdef __DAV_C220_CUBE__
#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm1TilingDataRect, bmm1TilingData, tiling);                \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm2TilingDataRect, bmm2TilingData, tiling);                \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;                                                        \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;                                                        \
    const PromptFlashAttentionTilingData* __restrict tiling_data = nullptr

#define INVOKE_PFA_TILING_DATA_BASE_API(tiling)                                                                        \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionBaseApiTilingData, tiling_data_in, tiling);                        \
    const PromptFlashAttentionBaseApiTilingData* __restrict tiling_data = &tiling_data_in

#define INVOKE_PFA_TILING_DATA_MLA(tiling)                                                                             \
    GET_TILING_DATA_MEMBER(MLAGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);                           \
    GET_TILING_DATA_MEMBER(MLAGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);                           \
    const MLAGeneralTilingData * __restrict tilingData = nullptr;                                                      \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingDataVar 
#else
#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionTilingData, tiling_data_in, tiling);                               \
    const PromptFlashAttentionTilingData* __restrict tiling_data = &tiling_data_in;                                    \
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);                                     \
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

#define INVOKE_PFA_TILING_DATA_MLA(tiling)                                                                              \
    GET_TILING_DATA_WITH_STRUCT(MLAGeneralTilingData, tiling_data_in, tiling);                                          \
    const MLAGeneralTilingData * __restrict tilingData = &tiling_data_in;                                               \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->bmm1TilingData);                                           \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->bmm2TilingData)

#define INVOKE_PFA_TILING_DATA_BASE_API(tiling)                                                                        \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionBaseApiTilingData, tiling_data_in, tiling);                        \
    const PromptFlashAttentionBaseApiTilingData* __restrict tiling_data = &tiling_data_in
#endif
#define INVOKE_PFA_NEW_GQA_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        if (query == nullptr) {return;}                                                                              \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                                \
        templateClass<__VA_ARGS__> op;                                                                          \
        op.Init(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, attentionOut, user, tiling_data);                                 \
        op.Process();                                                                                                  \
    } while (0)


constexpr uint32_t FLOATBYTENUM = 8;
constexpr uint32_t FLOAT16BYTENUM = 16;
constexpr uint32_t INT8BYTENUM = 32;

template<uint8_t Q_T, uint8_t KV_T, uint16_t OUT_T, uint8_t PAGE_ATTENTIOND, uint16_t LAYOUT_T,  uint16_t KV_LAYOUT_T, uint16_t FLASH_DECODE,
            uint8_t ENABLE_PREFIX, uint8_t M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, uint8_t M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, uint8_t M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA,  uint16_t M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE, 
                uint16_t M_FIAFLAG_P_MMTYPETMP_I_MODEVAL,   uint8_t P_CVDIFF_BASE_FLAG,
                uint8_t P_CVDIFF_MLA_FLAG,  uint8_t P_TEMPLATE_VERSION, uint8_t TEMPLATE_MODE>
__global__ __aicore__ void prompt_flash_attention_FIAS(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                        __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                        __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                        __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                        __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset,
                                                        __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                        __gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
                                                        __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
                                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                        __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* learnableSink,
                                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                        __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    {
    #if (__CCE_AICORE__ > 200)
        // bit 2
        using Q_DTYPE = std::conditional_t<Q_T == 0, half,
                        std::conditional_t<Q_T == 1, bfloat16_t,
                        std::conditional_t<Q_T == 2, int8_t,
                        std::conditional_t<Q_T == 6, half, half>>>>; // 默认回退

        constexpr OptimizationMode PRECISION_TYPE = (Q_T == 0) ? OptimizationMode::HighPerformance :
                                                    (Q_T == 6) ? OptimizationMode::HighPrecision : OptimizationMode::HighPerformance;
        // bit 4
        using OUT_DTYPE = std::conditional_t<OUT_T == 0, half,
                          std::conditional_t<OUT_T == 1, bfloat16_t,
                          std::conditional_t<OUT_T == 2, int8_t, half>>>; // 默认回退
        // bit 5
        constexpr PFALayout LAYOUT_DTYPE = (LAYOUT_T == 0) ? PFALayout::BNSD :
                                           (LAYOUT_T == 1) ? PFALayout::BSH : PFALayout::BNSD;
        // bit 6
        constexpr MatMulType MATMUL_TYPE = (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) ? MatMulType::MM_MDL :
                                           (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1) ? MatMulType::MM_NORM :
                                           (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2) ? MatMulType::MM_IBSHARE_NORM : MatMulType::MM_MDL;

        // bit 7
        constexpr MmPolicyType PA_TYPE = (PAGE_ATTENTIOND == 0) ? MmPolicyType::NORMAL :
                                         (PAGE_ATTENTIOND == 1) ? MmPolicyType::PA_ND :
                                         (PAGE_ATTENTIOND == 2) ? MmPolicyType::PA_NZ : MmPolicyType::NORMAL;

        // bit 8-2
        constexpr MsdMode MSD_TYPE = (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) ? MsdMode::MSD_OFF :
                                     (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1) ? MsdMode::MSD_ON : MsdMode::MSD_OFF;
        // bit 8-1
        constexpr bool PREFIX_MODE = (ENABLE_PREFIX == 0) ? false :
                                     (ENABLE_PREFIX == 1) ? true : false;
        #ifndef FIA_PFA
        REGISTER_TILING_DEFAULT(PromptFlashAttentionTilingData);
        #endif
        GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
        auto maskByteNum = baseParams.maskTypeByteNum;
    
        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        if constexpr ((Q_T == 0 || Q_T == 6) && (KV_T != 1)){
            if constexpr ((Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0)
                && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
                if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1)){
                    // Non-BNSD layout, split NS no tail
                    if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, Q_DTYPE, half, CubeFormat::ND, OUT_DTYPE);
                    }
                    else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
                    }
                }
                else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1)){
                    // Non-BNSD layout, split NS with tail
                    if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, Q_DTYPE, half, CubeFormat::ND, OUT_DTYPE);
                    }
                    else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
                    }
                }
                else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1)){
                    // BNSD layout, split NS no tail
                    if (maskByteNum == FLOAT16BYTENUM) {
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, Q_DTYPE, half, CubeFormat::ND, OUT_DTYPE);
                    }
                    else {
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
                    }
                }
                else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1)){
                    // BNSD layout, split NS with tail
                    if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, Q_DTYPE, half, CubeFormat::ND, OUT_DTYPE);
                    }
                    else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
                    }
                }
            }   
            if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0) && (LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0)    
                && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
                // no anti-quant path for CVDIFF-BSH, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, half>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool>);
                }
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0)   
                && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
                // no anti-quant path for CVDIFF-BNSD, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, half>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, uint8_t>);
                }
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 2) && (P_CVDIFF_MLA_FLAG == 0)   
                && (KV_T == 0) && (P_TEMPLATE_VERSION == 2)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionNoMask, PFAHighPrecisionBaseType<PromptFlashAttentionBaseApiTilingData, float, half, half, half, half, float>);
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0)   
                && (KV_T == 0) && (P_TEMPLATE_VERSION == 2)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPerformance, PFATypeNew<PromptFlashAttentionBaseApiTilingData, Q_DTYPE, half, OUT_DTYPE, half, half, float, half>);
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 1)   
                && (KV_T == 0) && (P_TEMPLATE_VERSION == 2)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLA, PFAMLAType<PromptFlashAttentionBaseApiTilingData>);
            }
            if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0 || Q_T == 6) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0 || M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0)   
                && (LAYOUT_T == 0 || LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 || M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 || M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 || M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)   
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0 || M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0 || P_CVDIFF_MLA_FLAG == 1) && (KV_T == 0 || KV_T == 4 || KV_T == 8) && (P_TEMPLATE_VERSION == 1 || P_TEMPLATE_VERSION == 2)){
                if constexpr ((P_TEMPLATE_VERSION == 1) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2)){   
                    if constexpr ((KV_T == 0) && (PAGE_ATTENTIOND == 0) && (Q_T == 6) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1) 
                    && ((LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 1) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && ENABLE_PREFIX == 0) 
                    || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && ENABLE_PREFIX == 1) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 && ENABLE_PREFIX == 0) || (LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) 
                    || (LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 1) || (LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 && ENABLE_PREFIX == 0))){
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, half, OptimizationMode::HighPrecision, MATMUL_TYPE, PREFIX_MODE, MsdMode::MSD_OFF>);
                    }
                    else if ((KV_T == 0) && (PAGE_ATTENTIOND == 1) && (Q_T == 6) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1) 
                    && ((LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0))){
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
                    }
                    else if ((KV_T == 8) && (PAGE_ATTENTIOND == 0) && (Q_T == 6) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1) 
                    && ((LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 1) 
                    || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 1))){
                        INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPrecision, MATMUL_TYPE, PREFIX_MODE, MsdMode::MSD_OFF>);
                    }
                    else if ((KV_T == 4)  && (PAGE_ATTENTIOND == 0) && (Q_T == 6) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1) 
                    && ((LAYOUT_T == 1 && ENABLE_PREFIX == 1) || (LAYOUT_T == 0 && ENABLE_PREFIX == 1) || (LAYOUT_T == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 1 && ENABLE_PREFIX == 0))){
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPrecision, MATMUL_TYPE, PREFIX_MODE, MSD_TYPE>);
                    }
                    else if( (KV_T == 0) && (PAGE_ATTENTIOND == 0) && (Q_T == 0) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 1) 
                    && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1)){
                        if (maskByteNum == FLOAT16BYTENUM) {
                            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, half, OUT_DTYPE, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                        } else {
                            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, uint8_t, OUT_DTYPE, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                        }
                    }
                    else if ((KV_T == 0) && (PAGE_ATTENTIOND == 0) && (Q_T == 0) && (OUT_T == 0) && (LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 1) 
                    && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1)){
                        if (maskByteNum == FLOAT16BYTENUM) {
                            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, half, OUT_DTYPE, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                        } else {
                            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                        }
                    }
                    else if ((KV_T == 0) && (PAGE_ATTENTIOND == 0) && (Q_T == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1) 
                    && ((LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && ENABLE_PREFIX == 0) 
                    || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 1 && ENABLE_PREFIX == 1) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 && ENABLE_PREFIX == 0))){
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, uint8_t, OUT_DTYPE, half, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE, MsdMode::MSD_OFF>);
                    }
                    else if ((KV_T == 0) && (PAGE_ATTENTIOND == 1) && (Q_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0) 
                    && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1)){
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, uint8_t, OUT_DTYPE, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                    }
                    else if ((KV_T == 0) && (PAGE_ATTENTIOND == 1) && (Q_T == 0) && (LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0) 
                    && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1)){
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                    }
                    else if ((KV_T == 8) && (PAGE_ATTENTIOND == 0) && (Q_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1) 
                    && ((LAYOUT_T == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && ENABLE_PREFIX == 1) || (LAYOUT_T == 1 && ENABLE_PREFIX == 0) ||(LAYOUT_T == 1 && ENABLE_PREFIX == 1))){
                        INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE, MsdMode::MSD_OFF>);
                    }
                    else if ((KV_T == 8) && (PAGE_ATTENTIOND == 1) && (Q_T == 0) && (LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0) 
                    && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1)){
                        INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA_ANTIQUANT>);
                    }
                    else if ((KV_T == 8) && (PAGE_ATTENTIOND == 1) && (Q_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0) 
                    && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0) && (P_TEMPLATE_VERSION == 1)){
                        INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t>);
                    }
                }   
                else if ((P_TEMPLATE_VERSION == 2) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE) && (OUT_T == 0)   
                && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_MLA_FLAG == 0 || P_CVDIFF_MLA_FLAG == 1) && (KV_T == 0)){
                    if (P_CVDIFF_MLA_FLAG == 0){
                        REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                        INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionV, PFATypeNew<PromptFlashAttentionBaseApiTilingData, Q_DTYPE, half, OUT_DTYPE, float, half, float, half, OptimizationMode::HighPrecision>);
                    }
                    else if (P_CVDIFF_MLA_FLAG == 1){
                        REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                        INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLAHighPrecision, PFAHighPrecisionMLAType<PromptFlashAttentionBaseApiTilingData, half, false>);
                    }
                }
            }

            if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0 || Q_T == 6) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 2)   
                && (LAYOUT_T == 0 || LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)   
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0 || M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0) && (KV_T == 0 || KV_T == 4 || KV_T == 8) && (P_TEMPLATE_VERSION == 1)){
                if constexpr ((Q_T == 0 || Q_T == 6) && (KV_T == 0) && (PAGE_ATTENTIOND == 0) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0)){
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, half, PRECISION_TYPE, MATMUL_TYPE, PREFIX_MODE, MsdMode::MSD_OFF>);
                }
                else if ((Q_T == 0 || Q_T == 6) && (KV_T == 0) && (PAGE_ATTENTIOND == 1) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0)){
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, half, PRECISION_TYPE, MatMulType::MM_PA, PREFIX_MODE, MsdMode::MSD_OFF>);
                }
                else if ((Q_T == 0 || Q_T == 6) && (KV_T == 8) && (PAGE_ATTENTIOND == 0) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0)){
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, PRECISION_TYPE, MATMUL_TYPE, PREFIX_MODE, MsdMode::MSD_OFF>);
                }
                else if ((Q_T == 6) && (KV_T == 4) && (PAGE_ATTENTIOND == 0) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1)){
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPrecision, MATMUL_TYPE, PREFIX_MODE, MSD_TYPE>);
                }
            }
        }
            if constexpr ((Q_T == 0 || Q_T == 1) && (KV_T != 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3)     
                && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0 || M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 0 || Q_T == 1) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0 || M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0 || OUT_T == 1 || OUT_T == 2)                   
                && (LAYOUT_T == 0 || LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 || M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1 || PAGE_ATTENTIOND == 2) && (ENABLE_PREFIX == 0)       
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0 || P_CVDIFF_BASE_FLAG == 2) && (P_CVDIFF_MLA_FLAG == 0 || P_CVDIFF_MLA_FLAG == 1) && (KV_T == 0) && (P_TEMPLATE_VERSION == 1 || P_TEMPLATE_VERSION == 2 || P_TEMPLATE_VERSION == 4)){
            if constexpr ((P_TEMPLATE_VERSION == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0) && (Q_T == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                // Non-BNSD layout, split NS no tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            }
            else if ((P_TEMPLATE_VERSION == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1) && (Q_T == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                // Non-BNSD layout, split NS with tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            }
            else if ((P_TEMPLATE_VERSION == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5) && (Q_T == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                // BNSD layout, split NS no tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            }
            else if ((P_TEMPLATE_VERSION == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6) && (Q_T == 1) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                    // BNSD layout, split NS with tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            }
            else if ((P_TEMPLATE_VERSION == 2) && (P_CVDIFF_MLA_FLAG == 0) && (P_CVDIFF_BASE_FLAG == 2) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 1) && (OUT_T == 1) && (LAYOUT_T == 0) 
            && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionNoMask, PFAHighPrecisionBaseType<PromptFlashAttentionBaseApiTilingData, float, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, float>);
            }
            else if ((P_TEMPLATE_VERSION == 2) && (P_CVDIFF_MLA_FLAG == 0) && (P_CVDIFF_BASE_FLAG == 0) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 1) && (OUT_T == 1) && (LAYOUT_T == 0) 
            && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionV,
                                                    PFATypeNew<PromptFlashAttentionBaseApiTilingData, bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t, float, bfloat16_t, OptimizationMode::HighPrecision>);
            }
            else if ((P_TEMPLATE_VERSION == 2) && (P_CVDIFF_MLA_FLAG == 1) && (P_CVDIFF_BASE_FLAG == 0) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 1) && (OUT_T == 1) && (LAYOUT_T == 0) 
            && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 4) && (PAGE_ATTENTIOND == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", PromptFlashAttentionBaseApiTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLAHighPrecision, PFAHighPrecisionMLAType<PromptFlashAttentionBaseApiTilingData, bfloat16_t, true>);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (PAGE_ATTENTIOND == 0) && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameAB, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (PAGE_ATTENTIOND == 0) && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameAB, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (LAYOUT_T == 0) && (PAGE_ATTENTIOND == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (LAYOUT_T == 1) && (PAGE_ATTENTIOND == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (LAYOUT_T == 0) && (PAGE_ATTENTIOND == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (LAYOUT_T == 1) && (PAGE_ATTENTIOND == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (LAYOUT_T == 0) && (PAGE_ATTENTIOND == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (LAYOUT_T == 1) && (PAGE_ATTENTIOND == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3) && (LAYOUT_T == 0) && (PAGE_ATTENTIOND == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3) && (LAYOUT_T == 1) && (PAGE_ATTENTIOND == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3) && (LAYOUT_T == 0) && (PAGE_ATTENTIOND == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3) && (LAYOUT_T == 1) && (PAGE_ATTENTIOND == 1) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3) && (LAYOUT_T == 0) && (PAGE_ATTENTIOND == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            }
            else if ((P_TEMPLATE_VERSION == 4) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 3) && (LAYOUT_T == 1) && (PAGE_ATTENTIOND == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0) && (Q_T == 0) && (OUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) 
            && (P_CVDIFF_BASE_FLAG == 0)  && (P_CVDIFF_MLA_FLAG == 0)){
                REGISTER_TILING_FOR_TILINGKEY("TRUE", MLAGeneralTilingData);
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            }

            if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 1) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 1)   
                && (LAYOUT_T == 0 || LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 || M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)   
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0 || M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0) && (KV_T == 0 || KV_T == 4) && (P_TEMPLATE_VERSION == 1)){
                if constexpr ((KV_T == 0) && (PAGE_ATTENTIOND == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) 
                && ((LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 && ENABLE_PREFIX == 0) || (LAYOUT_T == 1 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 1) 
                || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 2 && ENABLE_PREFIX == 0) || (LAYOUT_T == 0 && M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0 && ENABLE_PREFIX == 1))){
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, bfloat16_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE, MSD_TYPE>);
                }
                else if ((KV_T == 0) && (PAGE_ATTENTIOND == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) 
                && ((LAYOUT_T == 1  && ENABLE_PREFIX == 0) || (LAYOUT_T == 0  && ENABLE_PREFIX == 0))){
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA, PREFIX_MODE, MSD_TYPE>);
                }
                else if ((KV_T == 4) && (PAGE_ATTENTIOND == 0) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1)){
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE, MSD_TYPE>);
                }   
            }

            if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 1) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 2)   
                && (LAYOUT_T == 0 || LAYOUT_T == 1) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)   
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0 || M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0) && (KV_T == 0 || KV_T == 4) && (P_TEMPLATE_VERSION == 1)){
                if constexpr ((KV_T == 0) && (PAGE_ATTENTIOND == 0) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0)){
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, bfloat16_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE, MSD_TYPE>);
                }
                else if ((KV_T == 0) && (PAGE_ATTENTIOND == 1) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0)){
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA, PREFIX_MODE, MSD_TYPE>);
                }
                else if ((KV_T == 4) && (PAGE_ATTENTIOND == 0) && (LAYOUT_T == 0 || LAYOUT_T == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 1)){
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<LAYOUT_DTYPE, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE, MSD_TYPE>);
                }
            }
        }
        if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 7)    
                && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0 || M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 2) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0 || M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 2)   
                && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)   
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0) && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
            if constexpr ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, Q_DTYPE, bool, CubeFormat::ND, OUT_DTYPE);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) &&  (PAGE_ATTENTIOND == 0) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE>);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (PAGE_ATTENTIOND == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, Q_DTYPE, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1)  && (PAGE_ATTENTIOND == 0) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==7) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE>);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1)  && (PAGE_ATTENTIOND == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==7) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            }
        }
        if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 2 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6 || M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 7) 
                && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 0 || M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (Q_T == 2) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0 || M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) && (OUT_T == 0)   
                && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0 || PAGE_ATTENTIOND == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)   
                && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0) && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
            if constexpr ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, half);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 1) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, half);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 5) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, half);
            }  
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 6) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, half);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) &&  (PAGE_ATTENTIOND == 0) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MATMUL_TYPE, PREFIX_MODE>);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) &&  (PAGE_ATTENTIOND == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==2) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) &&  (PAGE_ATTENTIOND == 0) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==7) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0 || ENABLE_PREFIX == 1)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, PREFIX_MODE>);
            }
            else if ((M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 1) &&  (PAGE_ATTENTIOND == 1) && (M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T ==7) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 1) && (ENABLE_PREFIX == 0)){
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, Q_DTYPE, bool, OUT_DTYPE, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            }
        }
        if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 0) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 2) && (Q_T == 0) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 0) && (OUT_T == 0) && (LAYOUT_T == 0) && (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0)   
            && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
            // kv is empty tensor, return zero output
            TPipe tPipe;
            INVOKE_PFA_TILING_DATA(tiling);
            PromptFlashAttentionEmptyTensor<half> op;
            op.Init(attentionOut, tiling_data, &tPipe);
            op.Process();
            return;
        }
    #else
        #ifndef FIA_PFA
        REGISTER_TILING_DEFAULT(PromptFlashAttentionTilingData);
        #endif
        GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
        auto maskByteNum = baseParams.maskTypeByteNum;

        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        if constexpr( (M_FIAFLAG_P_MMTYPETMP_I_MODEVAL == 0) && (PAGE_ATTENTIOND == 0) && (ENABLE_PREFIX == 0) && (M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE == 0) && (P_CVDIFF_BASE_FLAG == 0) && (P_CVDIFF_MLA_FLAG == 0)   
            && (KV_T == 0) && (P_TEMPLATE_VERSION == 1)){
            if constexpr ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 8) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 8) && (Q_T == 2) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 2) && (OUT_T == 1) && (LAYOUT_T == 1)){
                INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 8) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 8) && (Q_T == 2) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 2) && (OUT_T == 2) && (LAYOUT_T == 1)){
                INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 8) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 8) && (Q_T == 2) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 2) && (OUT_T == 1) && (LAYOUT_T == 0)){
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>);
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 8) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 8) && (Q_T == 2) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 2) && (OUT_T == 2) && (LAYOUT_T == 0)){
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half>);
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 8) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 8) && (Q_T == 8) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 2) && (OUT_T == 1) && (LAYOUT_T == 0)){
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
            }
            else if ((M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T == 8) && (M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA == 8) && (Q_T == 8) && (M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE == 2) && (OUT_T == 2) && (LAYOUT_T == 0)){
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
            }
        }
    #endif
    }    
}

template<uint8_t Q_T, uint8_t KV_T, uint8_t OUT_T, uint8_t PAGE_ATTENTIOND, uint8_t LAYOUT_T, uint16_t KV_LAYOUT_T, uint16_t FLASH_DECODE, uint8_t ENABLE_PREFIX,
            uint8_t M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE, uint8_t M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, uint8_t M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, uint8_t M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
            uint8_t M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, uint8_t P_CVDIFF_BASE_FLAG, uint8_t P_CVDIFF_MLA_FLAG, uint8_t P_TEMPLATE_VERSION, uint8_t TEMPLATE_MODE>
__global__ __aicore__ void prompt_flash_attention(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                            __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                            __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                            __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                            __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                            __gm__ uint8_t* quant_offset2, __gm__ uint8_t* attentionOut,
                                            __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
        prompt_flash_attention_FIAS<Q_T, KV_T, OUT_T, PAGE_ATTENTIOND, LAYOUT_T, KV_LAYOUT_T, FLASH_DECODE, ENABLE_PREFIX, M_Q_QUANTMODE_P_MSD_MODE_I_ANTIQUANTMODE,
                                    M_OUTLAYOUT_P_TAIL_MODE_I_ORIGIN_T, M_K_QUANTMODE_P_NEWTILINGFLAG_I_AMLA, M_V_QUANTMODE_P_PRECISION_MODE_I_BALANCE,
                                    M_FIAFLAG_P_MMTYPETMP_I_MODEVAL, P_CVDIFF_BASE_FLAG, P_CVDIFF_MLA_FLAG, P_TEMPLATE_VERSION, TEMPLATE_MODE>
           (query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                quant_offset2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, attentionOut, nullptr, workspace, tiling);
}