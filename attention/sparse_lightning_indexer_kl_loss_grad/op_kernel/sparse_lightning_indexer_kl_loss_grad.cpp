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
 * \file sparse_lightning_indexer_kl_loss_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
using namespace AscendC;

#if __CCE_AICORE__ != 310
#include "sparse_lightning_indexer_kl_loss_grad_template_tiling_key.h"
#include "arch22/sparse_lightning_indexer_kl_loss_grad_base.h"
#define SLI_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        templateClass<SLIType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \
        op.Init(q, k, w, sparseIndices, attnSoftmaxL1Norm, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, \
                metadata, dq, dk, dw, softmaxOut, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)
#else
#include "arch35/sparse_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_tiling_regbase.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_template_tiling_key_regbase.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_vector_block.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_cube_block.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_kernel_base.h"
using namespace Slig;
#define SLI_OP_IMPL(templateClass, tilingdataClass, ...) \
    do { \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC, Slig::SligBlockCube<__VA_ARGS__>, \
                                                        Slig::SligBlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIV, Slig::SligBlockVec<__VA_ARGS__>, \
                                                       Slig::SligBlockVecDummy<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling); \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in; \
        op.Init(q, k, w, sparseIndices, attnSoftmaxL1, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, \
                metadata, dq, dk, dw, softmaxOut, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)
#endif

#if __CCE_AICORE__ != 310
template <bool HasRope, int TopKRange, int LayoutT_QT, int LayoutT_KT, int SparseMode, bool HasSequsedQ,
          bool HasSequsedK, bool Deterministic, bool PrivateScatter>
__global__ __aicore__ void sparse_lightning_indexer_kl_loss_grad(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *attnSoftmaxL1Norm, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *metadata, __gm__ uint8_t *dq,
    __gm__ uint8_t *dk, __gm__ uint8_t *dw, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::SparseLightningIndexerKLLossGradTilingData);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && ORIG_DTYPE_K == DT_FLOAT16) {
        SLI_OP_IMPL(SparseLightningIndexerKLLossGradBase, optiling::SparseLightningIndexerKLLossGradTilingData, half,
                    half, half, static_cast<SLITopKRange>(TopKRange), static_cast<SLILayout>(LayoutT_QT),
                    static_cast<SLILayout>(LayoutT_KT), static_cast<SLISparseMode>(SparseMode), HasRope, HasSequsedQ,
                    HasSequsedK, Deterministic, PrivateScatter);
    }
    if constexpr (ORIG_DTYPE_Q == DT_BF16 && ORIG_DTYPE_K == DT_BF16) {
        SLI_OP_IMPL(SparseLightningIndexerKLLossGradBase, optiling::SparseLightningIndexerKLLossGradTilingData,
                    bfloat16_t, bfloat16_t, bfloat16_t, static_cast<SLITopKRange>(TopKRange),
                    static_cast<SLILayout>(LayoutT_QT), static_cast<SLILayout>(LayoutT_KT),
                    static_cast<SLISparseMode>(SparseMode), HasRope, HasSequsedQ, HasSequsedK, Deterministic,
                    PrivateScatter);
    }
}

#else
template <int TopKRange, int LayoutT_Q, int LayoutT_KT, int SparseMode, bool HasCuSeqlensQ, bool HasCuSeqlensK,
          bool HasSequsedQ, bool HasSequsedK, bool HasCmpResidualK, bool Deterministic>
__global__ __aicore__ void sparse_lightning_indexer_kl_loss_grad(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *attnSoftmaxL1, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *metadata, __gm__ uint8_t *dq,
    __gm__ uint8_t *dk, __gm__ uint8_t *dw, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::SparseLightningIndexerGradRegBaseTilingData);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ORIG_DTYPE_Q == DT_FLOAT16 && ORIG_DTYPE_K == DT_FLOAT16) {
        SLI_OP_IMPL(SparseLightningIndexerKLLossGradKernelBase, optiling::SparseLightningIndexerGradRegBaseTilingData,
                    half, half, float, static_cast<SLILayout>(LayoutT_Q), static_cast<SLILayout>(LayoutT_KT),
                    static_cast<SLISparseMode>(SparseMode), HasCuSeqlensQ, HasCuSeqlensK, HasSequsedQ, HasSequsedK,
                    HasCmpResidualK, Deterministic);
    }
    if constexpr (ORIG_DTYPE_Q == DT_BF16 && ORIG_DTYPE_K == DT_BF16) {
        SLI_OP_IMPL(SparseLightningIndexerKLLossGradKernelBase, optiling::SparseLightningIndexerGradRegBaseTilingData,
                    bfloat16_t, bfloat16_t, float, static_cast<SLILayout>(LayoutT_Q),
                    static_cast<SLILayout>(LayoutT_KT), static_cast<SLISparseMode>(SparseMode), HasCuSeqlensQ,
                    HasCuSeqlensK, HasSequsedQ, HasSequsedK, HasCmpResidualK, Deterministic);
    }
}
#endif
