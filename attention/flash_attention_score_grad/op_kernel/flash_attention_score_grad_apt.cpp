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
 * \file flash_attention_score_grad.cpp
 * \brief
 */
#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#else
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

#if (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#include "arch92/flash_attention_score_grad_entry_david.h"
#include "arch92/flash_attention_score_grad_template_tiling_key_david.h"
#include "arch92/flash_attention_score_grad_tiling_data_david.h"
#include "arch92/flash_attention_score_grad_empty_tensor_david.h"
#else
#include "arch35/flash_attention_score_grad_entry_regbase.h"
#include "arch35/flash_attention_score_grad_template_tiling_key.h"
#include "arch35/flash_attention_score_grad_tiling_data_regbase.h"
#include "arch35/flash_attention_score_grad_empty_tensor_regbase.h"
#endif

#if (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
template <uint8_t IsEmptyTensor, uint8_t SplitAxis, uint8_t InputDType, bool IsTnd, bool IsDrop, bool IsPse,
          bool IsAttenMask, uint16_t S1TemplateNum, uint16_t S2TemplateNum, uint16_t DTemplateNum, uint8_t DeterType,
          bool IsNEqual, bool IS_BN2_MULTIBLK, bool IsDNoEqual, uint8_t OutDType, bool UseDn, bool HasRope,
          bool IsDavid>
__global__ __aicore__ void flash_attention_score_grad(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_in, __gm__ uint8_t* attention_in, __gm__ uint8_t* prefix,
    __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* q_start_idx,
    __gm__ uint8_t* kv_start_idx, __gm__ uint8_t* deqScaleQ, __gm__ uint8_t* deqScaleK, __gm__ uint8_t* deqScaleV,
    __gm__ uint8_t* deqScaleDy, __gm__ uint8_t* deqScaleO, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
    __gm__ uint8_t* sink, __gm__ uint8_t* dsScale, __gm__ uint8_t* pScale, __gm__ uint8_t* dq, __gm__ uint8_t* dk,
    __gm__ uint8_t* dv, __gm__ uint8_t* dpse, __gm__ uint8_t* dqRope, __gm__ uint8_t* dkRope, __gm__ uint8_t* dsink,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling_data)
#else
// implementation of kernel function
template <uint8_t IsEmptyTensor, uint8_t SplitAxis, uint8_t InputDType, bool IsTnd, bool IsDrop, bool IsPse,
          bool IsAttenMask, uint16_t S1TemplateNum, uint16_t S2TemplateNum, uint16_t DTemplateNum, uint8_t DeterType,
          bool IsNEqual, bool IsBn2MultiBlk, bool IsDNoEqual, bool IsRope, uint8_t OutDType, bool IsNzOut,
          bool IsTndSwizzle, bool IsRegbase>
__global__ __aicore__ void flash_attention_score_grad(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_in, __gm__ uint8_t* attention_in, __gm__ uint8_t* prefix,
    __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* q_start_idx,
    __gm__ uint8_t* kv_start_idx, __gm__ uint8_t* deqScaleQ, __gm__ uint8_t* deqScaleK, __gm__ uint8_t* deqScaleV,
    __gm__ uint8_t* deqScaleDy, __gm__ uint8_t* deqScaleO, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
    __gm__ uint8_t* sink, __gm__ uint8_t* dsScale, __gm__ uint8_t* pScale, __gm__ uint8_t* dq, __gm__ uint8_t* dk,
    __gm__ uint8_t* dv, __gm__ uint8_t* dpse, __gm__ uint8_t* dqRope, __gm__ uint8_t* dkRope, __gm__ uint8_t* dsink,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling_data)
#endif
{
#if (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
    using fagTiling = optiling::fag::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2David<IS_DETER_NEW(DeterType)>;
    REGISTER_TILING_DEFAULT(fagTiling);
    if constexpr (IsEmptyTensor) {
        TPipe tPipe;
        REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR & 0x1)",
                                      optiling::fag::FlashAttentionScoreGradEmptyTensorTilingDataDavid);
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradEmptyTensorTilingDataDavid, tiling_data_in, tiling_data);
        const FlashAttentionScoreGradEmptyTensorTilingDataDavid* __restrict empty_tensor_tiling_data = &tiling_data_in;
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        FlashAttentionScoreGradEmptyTensorDavid<half> op;
        op.Init(dq, dk, dv, dpse, dqRope, dkRope, empty_tensor_tiling_data);
        op.Process();
#endif
#if (ORIG_DTYPE_QUERY == DT_FLOAT)
        FlashAttentionScoreGradEmptyTensorDavid<float> op;
        op.Init(dq, dk, dv, dpse, dqRope, dkRope, empty_tensor_tiling_data);
        op.Process();
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
        FlashAttentionScoreGradEmptyTensorDavid<bfloat16_t> op;
        op.Init(dq, dk, dv, dpse, dqRope, dkRope, empty_tensor_tiling_data);
        op.Process();
#endif
        return;
    } else {
        DavidFAG<SplitAxis, InputDType, IsTnd, IsDrop, IsPse, IsAttenMask, S1TemplateNum, S2TemplateNum, DTemplateNum,
                 DeterType, IsNEqual, IS_BN2_MULTIBLK, IsDNoEqual, OutDType, UseDn, HasRope, IsDavid>(
            query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,
            attention_in, prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy,
            deqScaleO, queryRope, keyRope, sink, dq, dk, dv, dpse, dqRope, dkRope, dsink, workspace, tiling_data);
    }
#else
    constexpr bool needDeterPrefix = NEED_DETER_PREFIX(DeterType, IsTnd);
    using fagTiling =
        optiling::fag::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER(DeterType), needDeterPrefix,
                                                                              IsTnd, IsTndSwizzle>;
    REGISTER_TILING_DEFAULT(fagTiling);
    if constexpr (IsEmptyTensor) {
        TPipe tPipe;
        REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR & 0x1)",
                                      optiling::fag::FlashAttentionScoreGradEmptyTensorTilingDataRegbase);
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradEmptyTensorTilingDataRegbase, tiling_data_in, tiling_data);
        const FlashAttentionScoreGradEmptyTensorTilingDataRegbase* __restrict empty_tensor_tiling_data =
            &tiling_data_in;
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        FlashAttentionScoreGradEmptyTensorRegbase<half> op;
        op.Init(dq, dk, dv, dpse, dqRope, dkRope, empty_tensor_tiling_data);
        op.Process();
#endif
#if (ORIG_DTYPE_QUERY == DT_FLOAT)
        FlashAttentionScoreGradEmptyTensorRegbase<float> op;
        op.Init(dq, dk, dv, dpse, dqRope, dkRope, empty_tensor_tiling_data);
        op.Process();
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
        FlashAttentionScoreGradEmptyTensorRegbase<bfloat16_t> op;
        op.Init(dq, dk, dv, dpse, dqRope, dkRope, empty_tensor_tiling_data);
        op.Process();
#endif
        return;
    } else {
        RegbaseFAG<SplitAxis, InputDType, IsTnd, IsDrop, IsPse, IsAttenMask, S1TemplateNum, S2TemplateNum, DTemplateNum,
                   DeterType, IsNEqual, IsBn2MultiBlk, IsDNoEqual, IsRope, OutDType, IsNzOut, IsTndSwizzle, IsRegbase>(
            query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,
            attention_in, prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy,
            dsScale, pScale, queryRope, keyRope, sink, dq, dk, dv, dpse, dqRope, dkRope, dsink, workspace, tiling_data);
    }
#endif
}
