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
 * \file dense_lightning_indexer_kl_loss_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
using namespace AscendC;

#include "arch35/dense_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "arch35/dense_lightning_indexer_kl_loss_grad_tiling_regbase.h"
#include "arch35/dense_lightning_indexer_kl_loss_grad_template_tiling_key_regbase.h"
#include "arch35/dense_lightning_indexer_kl_loss_grad_vector_block.h"
#include "arch35/dense_lightning_indexer_kl_loss_grad_cube_block.h"
#include "arch35/dense_lightning_indexer_kl_loss_grad_kernel_base.h"
#include "arch35/dense_lightning_indexer_kl_loss_grad_post.h"
using namespace Dlikg;
#define DLI_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, Dlikg::DlikgBlockCube<__VA_ARGS__>, \
                                      Dlikg::DlikgBlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIV, Dlikg::DlikgBlockVec<__VA_ARGS__>, \
                                                       Dlikg::DlikgBlockVecDummy<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \
        op.Init(q, k, w, attnSoftmaxL1, softmaxLse, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, \
                metadata, dq, dk, dw, softmaxOut, user, tiling_data, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
        TPipe pipePost; \
        DenseLightningIndexerKLLossGradPost<__VA_ARGS__> opPost; \
        opPost.Init(dk, user, tiling_data, &pipePost); \
        opPost.Process(); \
    } while (0)

template <int TopKRange, int LayoutT_Q, int LayoutT_KT, int SparseMode, bool HasCuSeqlensQ, bool HasCuSeqlensK,
          bool HasSequsedQ, bool HasSequsedK, bool HasCmpResidualK, bool Deterministic>
__global__ __aicore__ void dense_lightning_indexer_kl_loss_grad(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w, __gm__ uint8_t *attnSoftmaxL1, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
    __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *metadata, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dw,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::DenseLightningIndexerGradRegBaseTilingData);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && ORIG_DTYPE_K == DT_FLOAT16) {
        DLI_OP_IMPL(DenseLightningIndexerKLLossGradKernelBase, optiling::DenseLightningIndexerGradRegBaseTilingData,
                    half, half, float, static_cast<DLILayout>(LayoutT_Q), static_cast<DLILayout>(LayoutT_KT),
                    static_cast<DLISparseMode>(SparseMode), HasCuSeqlensQ, HasCuSeqlensK, HasSequsedQ, HasSequsedK,
                    HasCmpResidualK, Deterministic);
    }
    if constexpr (ORIG_DTYPE_Q == DT_BF16 && ORIG_DTYPE_K == DT_BF16) {
        DLI_OP_IMPL(DenseLightningIndexerKLLossGradKernelBase, optiling::DenseLightningIndexerGradRegBaseTilingData,
                    bfloat16_t, bfloat16_t, float, static_cast<DLILayout>(LayoutT_Q),
                    static_cast<DLILayout>(LayoutT_KT), static_cast<DLISparseMode>(SparseMode), HasCuSeqlensQ,
                    HasCuSeqlensK, HasSequsedQ, HasSequsedK, HasCmpResidualK, Deterministic);
    }
}
