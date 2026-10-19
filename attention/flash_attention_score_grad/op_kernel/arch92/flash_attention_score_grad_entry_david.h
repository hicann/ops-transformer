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
 * \file flash_attention_score_grad_entry_david.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_ENTRY_DAVID_H_
#define _FLASH_ATTENTION_SCORE_GRAD_ENTRY_DAVID_H_

#include "kernel_tiling/kernel_tiling.h"
#include "flash_attention_score_grad_common_david.h"
#include "flash_attention_score_grad_block_vec_david.h"
#include "flash_attention_score_grad_block_cube_david.h"
#include "flash_attention_score_grad_kernel_david.h"
#include "flash_attention_score_grad_kernel_deter_david.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_post_david.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_pre_david.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2_dropmask_david.h"
#include "kernel_operator.h"

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL( \
    INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, DETER_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, \
    SPLIT_AXIS, s1TemplateType, s2TemplateType, dTemplateType, OUTDTYPE, IS_SWIZZLE, IS_ROPE) \
    do { \
        if ASCEND_IS_AIV { \
            FlashAttentionScoreGradS1S2BNGS1S2PreDavid<INPUT_TYPE, float, dTemplateType, IS_D_NO_EQUAL, IS_TND, \
                                                       SPLIT_AXIS, DETER_TYPE, IS_ATTEN_MASK> \
                opPre; \
            opPre.Init(dq, dk, dv, dy, attention_in, actual_seq_qlen, actual_seq_kvlen, drop_mask, deqScaleDy, \
                       deqScaleO, user, tilingData, &pipeIn); \
            opPre.Process(); \
            opPre.SyncALLCores(); \
        } \
        pipeIn.Destroy(); \
\
        if constexpr (IS_DROP) { \
            TPipe pipeDrop; \
            FlashAttentionScoreGradS1S2BNGS1S2DropmaskDavid<INPUT_TYPE, float, IS_TND, SPLIT_AXIS, DETER_TYPE> opDrop; \
            opDrop.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, drop_mask, user, tilingData, &pipeDrop); \
            opDrop.Process(); \
            opDrop.SyncALLCores(); \
            pipeDrop.Destroy(); \
        } \
        TPipe pipeBase; \
        using CubeBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            FagBaseApi::FAGBlockCube<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, \
                                     DETER_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, SPLIT_AXIS, s1TemplateType, \
                                     s2TemplateType, dTemplateType, IS_SWIZZLE, IS_ROPE>, \
            FagBaseApi::FAGBlockCubeDummy<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, \
                                          DETER_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, SPLIT_AXIS, s1TemplateType, \
                                          s2TemplateType, dTemplateType, IS_SWIZZLE, IS_ROPE>>::type; \
        using VecBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            FagBaseApi::FAGBlockVecDummy<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, \
                                         DETER_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, SPLIT_AXIS, s1TemplateType, \
                                         s2TemplateType, dTemplateType, IS_SWIZZLE, IS_ROPE>, \
            FagBaseApi::FAGBlockVec<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, IS_BN2_MULTIBLK, \
                                    DETER_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, SPLIT_AXIS, s1TemplateType, s2TemplateType, \
                                    dTemplateType, IS_SWIZZLE, IS_ROPE>>::type; \
\
        typename std::conditional< \
            (DETER_TYPE) == NO_DETER, FagBaseApi::FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>, \
            FagBaseApi::FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>>::type op; \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, deqScaleO, \
                queryRope, keyRope, sink, dq, dk, dv, dpse, dsink, user, tilingData, &pipeBase); \
        op.Process(); \
        if ASCEND_IS_AIC { \
            pipeBase.Destroy(); \
            return; \
        } \
        if (ORIG_DTYPE_QUERY != DT_FLOAT && \
            (SPLIT_AXIS == BN2GS1S2 || tilingData->s1s2BNGS1S2BaseParams.sinkOptional)) { \
            op.SyncALLCores(); \
            pipeBase.Destroy(); \
            TPipe pipePost; \
            FlashAttentionScoreGradS1S2BNGS1S2PostDavid<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE, DETER_TYPE> \
                opPost; \
            opPost.Init(dq, dk, dv, dqRope, dkRope, dsink, user, tilingData, &pipePost); \
            opPost.Process(); \
            pipePost.Destroy(); \
        } else { \
            pipeBase.Destroy(); \
        } \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_FP16(...) \
    if (ORIG_DTYPE_QUERY == DT_FLOAT16) \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL(__VA_ARGS__)
#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_BF16(...) \
    if (ORIG_DTYPE_QUERY == DT_BF16) \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL(__VA_ARGS__)
#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_FP32(...) \
    if (ORIG_DTYPE_QUERY == DT_FLOAT) \
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL(__VA_ARGS__)
// implementation of kernel function
template <uint8_t splitAxis, uint8_t inputDType, bool isTnd, bool isDrop, bool isPse, bool isAttenMask,
          uint16_t s1TemplateNum, uint16_t s2TemplateNum, uint16_t dTemplateNum, uint8_t deterType, bool isNEqual,
          bool isBn2MultiBlk, bool isDNoEqual, uint8_t outDType, bool isSwizzle, bool hasRope, bool isDavid>
inline __aicore__ void DavidFAG(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy,
                                __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask,
                                __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
                                __gm__ uint8_t* softmax_in, __gm__ uint8_t* attention_in, __gm__ uint8_t* prefix,
                                __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen,
                                __gm__ uint8_t* deqScaleQ, __gm__ uint8_t* deqScaleK, __gm__ uint8_t* deqScaleV,
                                __gm__ uint8_t* deqScaleDy, __gm__ uint8_t* deqScaleO, __gm__ uint8_t* queryRope,
                                __gm__ uint8_t* keyRope, __gm__ uint8_t* sink, __gm__ uint8_t* dq, __gm__ uint8_t* dk,
                                __gm__ uint8_t* dv, __gm__ uint8_t* dpse, __gm__ uint8_t* dqRope,
                                __gm__ uint8_t* dkRope, __gm__ uint8_t* dsink, __gm__ uint8_t* workspace,
                                __gm__ uint8_t* tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipeIn;
    set_mask_norm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    using fagTiling = optiling::fag::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2David<IS_DETER_NEW(deterType)>;
    const __gm__ fagTiling* __restrict tilingData = (__gm__ fagTiling*)tiling_data;

#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if constexpr (splitAxis == BN2GS1S2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_FP16(half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                                         deterType, isNEqual, isDNoEqual, BN2GS1S2,
                                                         S1TemplateType(s1TemplateNum), S2TemplateType(s2TemplateNum),
                                                         DTemplateType(dTemplateNum), half, isSwizzle, hasRope);
        return;
    } else if (splitAxis == BN2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_FP16(half, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                                         deterType, isNEqual, isDNoEqual, BN2,
                                                         S1TemplateType(s1TemplateNum), S2TemplateType(s2TemplateNum),
                                                         DTemplateType(dTemplateNum), half, isSwizzle, hasRope);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    if constexpr (splitAxis == BN2GS1S2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_BF16(bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                                         deterType, isNEqual, isDNoEqual, BN2GS1S2,
                                                         S1TemplateType(s1TemplateNum), S2TemplateType(s2TemplateNum),
                                                         DTemplateType(dTemplateNum), bfloat16_t, isSwizzle, hasRope);
        return;
    } else if (splitAxis == BN2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_BF16(bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                                         deterType, isNEqual, isDNoEqual, BN2,
                                                         S1TemplateType(s1TemplateNum), S2TemplateType(s2TemplateNum),
                                                         DTemplateType(dTemplateNum), bfloat16_t, isSwizzle, hasRope);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT)
    if constexpr (splitAxis == BN2GS1S2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_FP32(float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                                         deterType, isNEqual, isDNoEqual, BN2GS1S2,
                                                         S1TemplateType(s1TemplateNum), S2TemplateType(s2TemplateNum),
                                                         DTemplateType(dTemplateNum), float, isSwizzle, hasRope);
        return;
    } else if (splitAxis == BN2) {
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_BASIC_IMPL_FP32(float, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                                         deterType, isNEqual, isDNoEqual, BN2,
                                                         S1TemplateType(s1TemplateNum), S2TemplateType(s2TemplateNum),
                                                         DTemplateType(dTemplateNum), float, isSwizzle, hasRope);
        return;
    }
#endif
}
#endif // _FLASH_ATTENTION_SCORE_GRAD_ENTRY_DAVID_H_
