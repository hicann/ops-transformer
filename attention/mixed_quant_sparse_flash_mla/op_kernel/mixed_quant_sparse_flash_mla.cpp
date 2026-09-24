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
 * \file mixed_quant_sparse_flash_mla.cpp
 * \brief
 */

#if (__CCE_AICORE__ == 310)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "arch35/mixed_quant_sparse_flash_mla_template_tiling_key_arch35.h"
#include "arch35/mixed_quant_sparse_flash_mla_csa_kernel.h"
#include "arch35/mixed_quant_sparse_flash_mla_common.h"
#else
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#include "lib/matmul_intf.h"
#include "arch22/mixed_quant_sparse_flash_mla_template_tiling_key_arch22.h"
#include "arch22/mixed_quant_sparse_flash_mla_tq_csa_kernel.h"
#include "mixed_quant_sparse_flash_mla_metadata.h"
#endif

using namespace AscendC;

#if (__CCE_AICORE__ == 310)
#define QSMLA_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::CSABlockCube<__VA_ARGS__>, \
                                      BaseApi::CSABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::CSABlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::CSABlockVec<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tilingDataIn, tiling); \
        const tilingdataClass *__restrict tilingData = &tilingDataIn; \
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, \
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv, oriTopkLength, \
                cmpTopkLength, sinks, metadata, attentionOut, softmax_lse, user, tilingData); \
        op.Process(); \
    } while (0)
#else
#define QSMLA_TQ_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        templateClass<SASKernel::SASType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tilingDataIn, tiling); \
        const tilingdataClass *__restrict tilingData = &tilingDataIn; \
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, \
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv, oriTopkLength, \
                cmpTopkLength, sinks, metadata, attentionOut, softmax_lse, user, tilingData, tiling, &tPipe); \
        op.Process(); \
    } while (0)
#endif

template <int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G, int QUANT_MODE, int KV_DTYPE,
          int BATCH_CONSISTENCY, int IS_VEC_S2PHYADDR, int HIGH_PERF>
__global__ __aicore__ void mixed_quant_sparse_flash_mla(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmax_lse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310)
    REGISTER_TILING_DEFAULT(optiling::MixedQuantSparseFlashMlaTilingData);
#else
    REGISTER_TILING_DEFAULT(optiling::MixedQuantSparseFlashMlaTqTilingData);
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

#if (__CCE_AICORE__ == 310)
    if constexpr (KV_DTYPE == DTYPE_FP8_E4M3FN) {
        QSMLA_OP_IMPL(BaseApi::MixedQuantSparseFlashMlaCsa, MixedQuantSparseFlashMlaTilingData, bfloat16_t,
                      fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, KV_LAYOUT_T == QSMLA_LAYOUT_PA_BBND,
                      static_cast<QSMLA_LAYOUT>(LAYOUT_T), static_cast<QSMLA_LAYOUT>(KV_LAYOUT_T),
                      static_cast<QSMLATemplateMode>(TEMPLATE_MODE), SPLIT_G,
                      static_cast<SCALE_CONTIGUOUS_MODE>(QUANT_MODE), BATCH_CONSISTENCY, IS_VEC_S2PHYADDR, HIGH_PERF);
    } else {
        QSMLA_OP_IMPL(BaseApi::MixedQuantSparseFlashMlaCsa, MixedQuantSparseFlashMlaTilingData, bfloat16_t, hifloat8_t,
                      float, bfloat16_t, FLASH_DECODE, KV_LAYOUT_T == QSMLA_LAYOUT_PA_BBND,
                      static_cast<QSMLA_LAYOUT>(LAYOUT_T), static_cast<QSMLA_LAYOUT>(KV_LAYOUT_T),
                      static_cast<QSMLATemplateMode>(TEMPLATE_MODE), SPLIT_G,
                      static_cast<SCALE_CONTIGUOUS_MODE>(QUANT_MODE), BATCH_CONSISTENCY, IS_VEC_S2PHYADDR, HIGH_PERF);
    }
#else
    TPipe tPipe;
    if constexpr (QUANT_MODE == TURBO_QUANT && TEMPLATE_MODE == CSA_TEMPLATE) {
        if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && ORIG_DTYPE_ORI_KV == DT_FLOAT16 && ORIG_DTYPE_CMP_KV == DT_UINT8 &&
                      ORIG_DTYPE_ATTN_OUT == DT_FLOAT16) {
            QSMLA_TQ_OP_IMPL(SASKernel::MixedQuantSparseFlashMlaTqCsaKernel,
                             optiling::MixedQuantSparseFlashMlaTqTilingData, half, half, half, FLASH_DECODE,
                             static_cast<SASKernel::SAS_LAYOUT>(LAYOUT_T),
                             static_cast<SASKernel::SAS_LAYOUT>(KV_LAYOUT_T), CSA_TEMPLATE);
        }
        if constexpr (ORIG_DTYPE_Q == DT_BF16 && ORIG_DTYPE_ORI_KV == DT_BF16 && ORIG_DTYPE_CMP_KV == DT_UINT8 &&
                      ORIG_DTYPE_ATTN_OUT == DT_BF16) {
            QSMLA_TQ_OP_IMPL(SASKernel::MixedQuantSparseFlashMlaTqCsaKernel,
                             optiling::MixedQuantSparseFlashMlaTqTilingData, bfloat16_t, bfloat16_t, bfloat16_t,
                             FLASH_DECODE, static_cast<SASKernel::SAS_LAYOUT>(LAYOUT_T),
                             static_cast<SASKernel::SAS_LAYOUT>(KV_LAYOUT_T), CSA_TEMPLATE);
        }
    }
#endif
}
