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
 * \file flash_attention_score_grad_entry_regbase_arch92.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_ENTRY_REGBASE_ARCH92_QUANT_H_
#define FLASH_ATTENTION_SCORE_GRAD_ENTRY_REGBASE_ARCH92_QUANT_H_
#include "flash_attention_score_grad_common_quant_arch92.h"
#include "flash_attention_score_grad_tiling_data_regbase_arch92.h"
using namespace commondef::quant;
using namespace optiling::fag::quant;
using namespace AscendC;
using namespace AscendC::Reg;
#include "flash_attention_score_grad_presfmg_regbase_arch92.h"

#include "flash_attention_score_grad_block_cube_quant_arch92.h"
#include "flash_attention_score_grad_block_vec_quant_arch92.h"
#include "flash_attention_score_grad_kernel_quant_arch92.h"
#include "flash_attention_score_grad_post_quant_arch92.h"
#include "flash_attention_score_grad_reduce_quant_arch92.h"

#define INVOKE_FAG_GENERAL_HIFP8_ARCH92_IMPL(INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, IS_TND, \
                                             IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, IS_ROPE, \
                                             IS_NZ_OUT, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, s2TemplateType, \
                                             dTemplateType) \
    do { \
        FlashAttentionScoreGradPresfmgRegbase<INPUT_TYPE, CALC_TYPE, OUTDTYPE, dTemplateType, IS_D_NO_EQUAL, \
                                              DETER_SPARSE_TYPE, IS_TND, SPLIT_AXIS, IS_TND_SWIZZLE> \
            opPre; \
        opPre.Init(dq, dk, dv, dy, attention_in, deqScaleDy, actual_seq_kvlen, user, tilingData, &pipeIn); \
        opPre.Process(); \
        opPre.SyncALLCores(); \
        pipeIn.Destroy(); \
        TPipe pipeBase; \
        using CubeBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            FagBaseApi::quant::FAGBlockCubeQuant<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, \
                                                 IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, \
                                                 IS_D_NO_EQUAL, IS_ROPE, IS_NZ_OUT, IS_TND_SWIZZLE, SPLIT_AXIS, \
                                                 s1TemplateType, s2TemplateType, dTemplateType>, \
            FagBaseApi::quant::FAGBlockCubeQuantDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, \
                                                      IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, \
                                                      IS_D_NO_EQUAL, IS_ROPE, IS_NZ_OUT, IS_TND_SWIZZLE, SPLIT_AXIS, \
                                                      s1TemplateType, s2TemplateType, dTemplateType>>::type; \
        using VecBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            FagBaseApi::quant::FAGBlockVecQuantDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, \
                                                     IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, \
                                                     IS_D_NO_EQUAL, IS_ROPE, IS_NZ_OUT, IS_TND_SWIZZLE, SPLIT_AXIS, \
                                                     s1TemplateType, s2TemplateType, dTemplateType>, \
            FagBaseApi::quant::FAGBlockVecQuant<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, \
                                                IS_TND, IS_BN2_MULTIBLK, DETER_SPARSE_TYPE, IS_N_EQUAL, IS_D_NO_EQUAL, \
                                                IS_ROPE, IS_NZ_OUT, IS_TND_SWIZZLE, SPLIT_AXIS, s1TemplateType, \
                                                s2TemplateType, dTemplateType>>::type; \
        using QuantKernel = FagBaseApi::quant::FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>; \
        QuantKernel op; \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, \
                prefix, actual_seq_qlen, actual_seq_kvlen, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, dsScale, \
                pScale, queryRope, keyRope, dq, dk, dv, dpse, dqRope, dkRope, user, tilingData, &pipeBase); \
        op.Process(); \
        op.SyncALLCores(); \
        pipeBase.Destroy(); \
        if (FagBaseApi::quant::BalancedShape(tilingData)) { \
            TPipe pipeReduce; \
            FagBaseApi::quant::ReduceBalancedPartials(user, tilingData, &pipeReduce); \
            pipeReduce.Destroy(); \
        } \
        TPipe pipePost; \
        FlashAttentionScoreGradPostQuant<INPUT_TYPE, float, OUTDTYPE, SPLIT_AXIS, IS_ROPE, DETER_SPARSE_TYPE, IS_TND, \
                                         IS_TND_SWIZZLE> \
            opPost; \
        opPost.Init(dq, dk, dv, dqRope, dkRope, dsink, user, tilingData, &pipePost); \
        opPost.Process(); \
    } while (0)

#define INVOKE_FAG_ARCH92_HIFP8_IMPL(...) \
    if (ORIG_DTYPE_QUERY == DT_HIFLOAT8) \
    INVOKE_FAG_GENERAL_HIFP8_ARCH92_IMPL(__VA_ARGS__)

template <uint8_t splitAxis, uint8_t inputDType, bool isTnd, bool isDrop, bool isPse, bool isAttenMask,
          uint16_t s1TemplateType, uint16_t s2TemplateType, uint16_t dTemplateType, uint8_t deterType, bool isNEqual,
          bool isBn2MultiBlk, bool isDNoEqual, bool isRope, uint8_t outDType, bool isNzOut, bool isTndSwizzle,
          bool isRegbase>
inline __aicore__ void QuantFAG(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy,
                                __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask,
                                __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
                                __gm__ uint8_t* softmax_in, __gm__ uint8_t* attention_in, __gm__ uint8_t* prefix,
                                __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen,
                                __gm__ uint8_t* deqScaleQ, __gm__ uint8_t* deqScaleK, __gm__ uint8_t* deqScaleV,
                                __gm__ uint8_t* deqScaleDy, __gm__ uint8_t* dsScale, __gm__ uint8_t* pScale,
                                __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope, __gm__ uint8_t* sink,
                                __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
                                __gm__ uint8_t* dqRope, __gm__ uint8_t* dkRope, __gm__ uint8_t* dsink,
                                __gm__ uint8_t* workspace, __gm__ uint8_t* tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipeIn;
    SetMaskNorm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    constexpr static bool needDeterPrefix = NEED_DETER_PREFIX(deterType, isTnd);
    using fagTiling = FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER(deterType), needDeterPrefix,
                                                                             isTnd, isTndSwizzle>;
    size_t offset = (size_t)(&((fagTiling*)0)->s1s2BNGS1S2BaseParams);

    const __gm__ fagTiling* __restrict tilingData = (const __gm__ fagTiling* __restrict)(tiling_data + offset);
#if (ORIG_DTYPE_QUERY == DT_HIFLOAT8)
    static_assert(inputDType == 6 && splitAxis == BN2GS1S2 && outDType == BFLOAT16 && s1TemplateType == 512 &&
                      s2TemplateType == 512 && dTemplateType == 128 && !isTnd && !isDrop && !isPse && !isAttenMask &&
                      !isRope && !isDNoEqual && !isBn2MultiBlk && !isNzOut && !isTndSwizzle && isRegbase &&
                      (deterType == NO_DETER || deterType == DETER_DENSE),
                  "Unsupported arch92 quant specialization");
    if constexpr (outDType == BFLOAT16) {
        INVOKE_FAG_ARCH92_HIFP8_IMPL(hifloat8_t, float, bfloat16_t, isAttenMask, isPse, isDrop, isTnd, isBn2MultiBlk,
                                     deterType, isNEqual, isDNoEqual, isRope, isNzOut, isTndSwizzle, BN2GS1S2,
                                     S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                                     DTemplateType(dTemplateType));
        return;
    }
#endif
}
#endif // FLASH_ATTENTION_SCORE_GRAD_ENTRY_REGBASE_ARCH92_QUANT_H_
