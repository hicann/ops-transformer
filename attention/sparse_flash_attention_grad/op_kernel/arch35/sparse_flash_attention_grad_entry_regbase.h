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
 * \file sparse_flash_attention_grad_entry_regbase.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_ENTRY_REGBASE_H_
#define SPARSE_FLASH_ATTENTION_GRAD_ENTRY_REGBASE_H_

#include "sparse_flash_attention_grad_arch35_common.h"

#include "sparse_flash_attention_grad_post_regbase.h"
#include "sparse_flash_attention_grad_pre_regbase.h"
#include "kernel_operator.h"

#include "sparse_flash_attention_grad_block_vec.h"
#include "sparse_flash_attention_grad_block_cube.h"
#include "sparse_flash_attention_grad_kernel.h"

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_DROP, IS_TND, IS_ROPE, \
                                                      IS_DETER, KV_MERGE, IS_SINKS, s2TemplateType, dTemplateType) \
    do { \
        FlashAttentionGradPreRegbase<INPUT_TYPE, float, IS_TND, IS_SINKS, IS_DETER, KV_MERGE> opPre; \
        opPre.Init(dq, dk, dv, d_sinks, actual_seq_kvlen, user, tilingData, &pipeIn); \
        opPre.Process(); \
        opPre.SyncALLCores(); \
        pipeIn.Destroy(); \
        TPipe pipeBase; \
        using CubeBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            SfagBaseApi::FAGBlockCube<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_DROP, IS_TND, IS_ROPE, IS_DETER, KV_MERGE, \
                                      IS_SINKS, s2TemplateType, dTemplateType>, \
            SfagBaseApi::FAGBlockCubeDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_DROP, IS_TND, IS_ROPE, IS_DETER, \
                                           KV_MERGE, IS_SINKS, s2TemplateType, dTemplateType>>::type; \
        using VecBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            SfagBaseApi::FAGBlockVecDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_DROP, IS_TND, IS_ROPE, IS_DETER, \
                                          KV_MERGE, IS_SINKS, s2TemplateType, dTemplateType>, \
            SfagBaseApi::FAGBlockVec<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_DROP, IS_TND, IS_ROPE, IS_DETER, KV_MERGE, \
                                     IS_SINKS, s2TemplateType, dTemplateType>>::type; \
\
        SfagBaseApi::FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType> op; \
        __gm__ uint8_t *valueIn = KV_MERGE ? key : value; \
        op.Init(query, key, valueIn, sparse_indices, d_out, out, softmax_max, softmax_sum, actual_seq_qlen, \
                actual_seq_kvlen, query_rope, key_rope, dq, dk, dv, dq_rope, dk_rope, user, tilingData, &pipeBase, \
                sinks, d_sinks); \
        op.Process(); \
        if (ORIG_DTYPE_QUERY != DT_FLOAT) { \
            op.SyncALLCores(); \
            pipeBase.Destroy(); \
            TPipe pipePost; \
            SparseFlashAttentionGradPostRegbase<INPUT_TYPE, float, OUTDTYPE, IS_ROPE, IS_TND, KV_MERGE> opPost; \
            opPost.Init(dq, dk, dv, dq_rope, dk_rope, user, tilingData, &pipePost); \
            opPost.Process(); \
        } else { \
            pipeBase.Destroy(); \
        } \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(...) \
    if (ORIG_DTYPE_QUERY == DT_FLOAT16) \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(__VA_ARGS__)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_BF16(...) \
    if (ORIG_DTYPE_QUERY == DT_BF16) \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(__VA_ARGS__)

// implementation of kernel function
template <uint8_t inputDType, bool isTnd, uint16_t gTemplateType, uint16_t s2TemplateType, uint16_t dTemplateType,
          bool isRope, bool deterministic, bool kvMerge, bool isSinks>
inline __aicore__ void RegbaseSFAG(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                   __gm__ uint8_t *sparse_indices, __gm__ uint8_t *d_out, __gm__ uint8_t *out,
                                   __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                   __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                                   __gm__ uint8_t *query_rope, __gm__ uint8_t *key_rope, __gm__ uint8_t *dq,
                                   __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dq_rope,
                                   __gm__ uint8_t *dk_rope, __gm__ uint8_t *sinks, __gm__ uint8_t *d_sinks,
                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeIn;
    SetMaskNorm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    using fagTiling = optiling::sfag::SparseFlashAttentionGradTilingDataRegbase;
    GET_TILING_DATA_WITH_STRUCT(fagTiling, tiling_data_in, tiling_data);
    const optiling::sfag::SparseFlashAttentionGradTilingDataRegbase *__restrict tilingData = &tiling_data_in;
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(half, float, half, false, isTnd, isRope, deterministic, kvMerge,
                                                       isSinks, S2TemplateType(s2TemplateType),
                                                       DTemplateType(dTemplateType));
    return;
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_BF16(bfloat16_t, float, bfloat16_t, false, isTnd, isRope,
                                                       deterministic, kvMerge, isSinks, S2TemplateType(s2TemplateType),
                                                       DTemplateType(dTemplateType));
    return;
#endif
}
#endif // _SPARSE_FLASH_ATTENTION_GRAD_ENTRY_REGBASE_H_
