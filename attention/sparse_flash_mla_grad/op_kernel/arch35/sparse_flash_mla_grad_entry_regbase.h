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
 * \file sparse_flash_mla_grad_entry_regbase.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_ENTRY_REGBASE_H_
#define SPARSE_FLASH_MLA_GRAD_ENTRY_REGBASE_H_

#include "sparse_flash_mla_grad_arch35_common.h"

#include "sparse_flash_mla_grad_post_regbase.h"
#include "sparse_flash_mla_grad_pre_regbase.h"
#include "kernel_operator.h"

#include "sparse_flash_mla_grad_block_vec.h"
#include "sparse_flash_mla_grad_block_cube.h"
#include "sparse_flash_mla_grad_kernel.h"

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL(INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, s2TemplateType, \
                                                      dTemplateType, isOriKVExist, isCmpKVExist, isOriKVSparse, \
                                                      isCmpKVSparse, isDeter) \
    do { \
        SparseFlashMlaGradPreRegbase<INPUT_TYPE, float, IS_TND, isOriKVExist, isCmpKVExist, isOriKVSparse, \
                                     isCmpKVSparse> \
            opPre; \
        opPre.Init(user, dsinks, ori_softmax_l1, cmp_softmax_l1, tilingData, &pipeIn); \
        opPre.Process(); \
        opPre.SyncALLCores(); \
        pipeIn.Destroy(); \
        TPipe pipeBase; \
        using CubeBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            SfagBaseApi::FAGBlockCube<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, s2TemplateType, dTemplateType, \
                                      isOriKVExist, isCmpKVExist, isOriKVSparse, isCmpKVSparse, isDeter>, \
            SfagBaseApi::FAGBlockCubeDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, s2TemplateType, dTemplateType, \
                                           isOriKVExist, isCmpKVExist, isOriKVSparse, isCmpKVSparse, isDeter>>::type; \
        using VecBlockType = typename std::conditional< \
            g_coreType == AscendC::AIC, \
            SfagBaseApi::FAGBlockVecDummy<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, s2TemplateType, dTemplateType, \
                                          isOriKVExist, isCmpKVExist, isOriKVSparse, isCmpKVSparse, isDeter>, \
            SfagBaseApi::FAGBlockVec<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, s2TemplateType, dTemplateType, \
                                     isOriKVExist, isCmpKVExist, isOriKVSparse, isCmpKVSparse, isDeter>>::type; \
\
        SfagBaseApi::SparseFlashMlaGradKernel<CubeBlockType, VecBlockType> op; \
        op.Init(query, ori_kv, cmp_kv, d_out, out, ori_sparse_indices, cmp_sparse_indices, cu_seqlens_q, \
                cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, \
                ori_topk_length, cmp_topk_length, softmax_lse, sinks, metadata, dq, dori_kv, dcmp_kv, dsinks, \
                ori_softmax_l1, cmp_softmax_l1, user, tilingData, &pipeBase); \
        op.Process(); \
        if (ORIG_DTYPE_QUERY != DT_FLOAT) { \
            op.SyncALLCores(); \
            pipeBase.Destroy(); \
            TPipe pipePost; \
            SparseFlashMlaGradPostRegbase<INPUT_TYPE, float, OUTDTYPE, IS_TND, isOriKVExist, isCmpKVExist> opPost; \
            opPost.Init(dq, dori_kv, dcmp_kv, dsinks, user, tilingData, &pipePost); \
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
          bool isOriKVExist, bool isCmpKVExist, bool isOriKVSparse, bool isCmpKVSparse, bool deterministic>
inline __aicore__ void RegbaseSFAG(
    __gm__ uint8_t *query, __gm__ uint8_t *ori_kv, __gm__ uint8_t *cmp_kv, __gm__ uint8_t *d_out, __gm__ uint8_t *out,
    __gm__ uint8_t *ori_sparse_indices, __gm__ uint8_t *cmp_sparse_indices, __gm__ uint8_t *cu_seqlens_q,
    __gm__ uint8_t *cu_seqlens_ori_kv, __gm__ uint8_t *cu_seqlens_cmp_kv, __gm__ uint8_t *seqused_q,
    __gm__ uint8_t *seqused_ori_kv, __gm__ uint8_t *seqused_cmp_kv, __gm__ uint8_t *cmp_residual_kv,
    __gm__ uint8_t *ori_topk_length, __gm__ uint8_t *cmp_topk_length, __gm__ uint8_t *softmax_lse,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *dq, __gm__ uint8_t *dori_kv,
    __gm__ uint8_t *dcmp_kv, __gm__ uint8_t *dsinks, __gm__ uint8_t *ori_softmax_l1, __gm__ uint8_t *cmp_softmax_l1,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeIn;
    SetMaskNorm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    using fagTiling = optiling::smlag::SparseFlashMlaGradTilingDataRegbase;
    GET_TILING_DATA_WITH_STRUCT(fagTiling, tiling_data_in, tiling_data);
    const optiling::smlag::SparseFlashMlaGradTilingDataRegbase *__restrict tilingData = &tiling_data_in;
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_FP16(half, float, half, isTnd, S2TemplateType(s2TemplateType),
                                                       DTemplateType(dTemplateType), isOriKVExist, isCmpKVExist,
                                                       isOriKVSparse, isCmpKVSparse, deterministic);
    return;
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_REGBASE_IMPL_BF16(
        bfloat16_t, float, bfloat16_t, isTnd, S2TemplateType(s2TemplateType), DTemplateType(dTemplateType),
        isOriKVExist, isCmpKVExist, isOriKVSparse, isCmpKVSparse, deterministic);
    return;
#endif
}
#endif // _SPARSE_FLASH_MLA_GRAD_ENTRY_REGBASE_H_
