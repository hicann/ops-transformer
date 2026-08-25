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
 * \file moe_token_permute_with_routing_map.cpp
 * \brief
 */
#include "moe_mrgsort_out_with_routing_map.h"
#include "moe_mrgsort_with_routing_map.h"
#include "moe_sort_multi_core_with_routing_map.h"
#include "moe_sort_multi_core_last_dim.h"
#include "moe_sort_one_core_with_routing_map.h"
#include "moe_index_copy_with_routing_map.h"
#include "moe_index_copy_split_d.h"
#include "moe_token_permute_masked_select_v3.h"
#include "moe_permute_prob.h"

#include "moe_routing_rank_multi_core.h"
#include "moe_sortindices_scatter_multi_core.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/gather_v2_simd_two_dim.h"
#include "arch35/gather_v2_simt_two_dim.h"
#endif

#if !defined(DTYPE_TOKENS)
#define DTYPE_TOKENS bfloat16_t
#endif

using namespace AscendC;
using namespace MoeTokenPermute;

#define MASKSELECT_ONLY_IMPL()                                                                                         \
    do {                                                                                                               \
        TPipe sortPipe;                                                                                                \
        KernelMaskedSelectV3<DTYPE_PERMUTE_PROBS> opMS;                                                                \
        auto maskedSelectTilingData = &(t->maskedSelectParamsOp);                                                      \
        opMS.Init(probs, routingMap, permuteProbs, sortedIndices, userWS, maskedSelectTilingData, t->topK, hasProb,    \
                  &sortPipe);                                                                                          \
        opMS.Process(permuteProbs, sortedIndices, actualOutTokens);                                                    \
        sortPipe.Destroy();                                                                                            \
        AscendC::SyncAll();                                                                                            \
    } while (0)

#define MASKSELECT_GATHER_RANK_IMPL(sortPositionGm, valueStreamOffset)                                                 \
    do {                                                                                                               \
        TPipe sortPipe;                                                                                                \
        KernelMaskedSelectV3<DTYPE_PERMUTE_PROBS> opMS;                                                                \
        auto maskedSelectTilingData = &(t->maskedSelectParamsOp);                                                      \
        opMS.Init(probs, routingMap, permuteProbs, sortedIndices, userWS, maskedSelectTilingData, t->topK, hasProb,    \
                  &sortPipe, AscendC::MaskedSelectMode::kGatherExternalIndex, sortPositionGm, valueStreamOffset);      \
        opMS.Process(permuteProbs, sortedIndices, actualOutTokens);                                                    \
        sortPipe.Destroy();                                                                                            \
        AscendC::SyncAll();                                                                                            \
    } while (0)

#define SORT_COPY_IMPL(sortClass2, indexCopyClass, tokenDtype, ifNumOutTokens)          \
    do {                                                                               \
        TPipe sortPipe2;                                                               \
        sortClass2<int32_t> op2;                                                       \
        op2.Init(sortedIndices, sortedIndices, userWS, t, &sortPipe2);                 \
        op2.Process();                                                                 \
        sortPipe2.Destroy();                                                           \
        TPipe MoeindexCopyPipe;                                                        \
        indexCopyClass<tokenDtype, false> indexCopyOp;                        \
        indexCopyOp.Init(tokens, sortedIndices, permuteTokens, t, &MoeindexCopyPipe);  \
        indexCopyOp.Process();                                                         \
    } while (0)

#define NONALIGNED_SORT_COPY_IMPL(indexCopyClass, tokenDtype)                                                          \
    do {                                                                                                               \
        TPipe sortPipe2;                                                                                               \
        int64_t msNeedCoreNum = t->maskedSelectParamsOp.needCoreNum;                                                   \
        int64_t msNumTokens = t->n;                                                                                    \
        int64_t msNumExperts = t->expertNum;                                                                           \
        int64_t partialOffset = MoeTokenPermute::Align(msNeedCoreNum, sizeof(int32_t));                                \
        int64_t sortPositionOffset = partialOffset + msNeedCoreNum * msNumTokens;                                      \
        int64_t valueStreamOffset = sortPositionOffset + msNumExperts * msNumTokens;                                   \
        GM_ADDR sortPositionGm = userWS + sortPositionOffset * sizeof(int32_t);                                        \
        GM_ADDR valueStreamGm = userWS + valueStreamOffset * sizeof(int32_t);                                          \
        MoeRoutingRankMultiCore rankOp;                                                                                \
        rankOp.Init(routingMap, userWS, &(t->maskedSelectParamsOp), msNumExperts, t->topK, &sortPipe2);                \
        rankOp.Process();                                                                                              \
        sortPipe2.Destroy();                                                                                           \
        MASKSELECT_GATHER_RANK_IMPL(sortPositionGm, valueStreamOffset);                                                \
        TPipe scatterPipe;                                                                                             \
        MoeSortIndicesScatterMultiCore scatterOp;                                                                      \
        scatterOp.Init(sortedIndices, valueStreamGm, actualOutTokens, t, userWS, &scatterPipe);                        \
        scatterOp.Process();                                                                                           \
        scatterPipe.Destroy();                                                                                         \
        TPipe MoeindexCopyPipe;                                                                                        \
        indexCopyClass<tokenDtype, true> indexCopyOp;                                                                  \
        indexCopyOp.Init(tokens, sortedIndices, permuteTokens, t, &MoeindexCopyPipe);                                  \
        indexCopyOp.Process();                                                                                         \
    } while (0)

#define GENERAL_OP_IMPL(sortClass1, sortClass2, indexCopyClass, tokenDtype, ifNumOutTokens) \
    do {                                                                               \
        int32_t actualOutTokens = 0;                                                   \
        MASKSELECT_ONLY_IMPL();                                                        \
        if (actualOutTokens != t->n * t->topK) {                                       \
            NONALIGNED_SORT_COPY_IMPL(indexCopyClass, tokenDtype);                     \
        } else {                                                                       \
            SORT_COPY_IMPL(sortClass2, indexCopyClass, tokenDtype, ifNumOutTokens);    \
        }                                                                              \
    } while (0)

#define GENERAL_PAD_OP_IMPL(sortClass1, sortClass2, indexCopyClass, ...)         \
    do {                                                                         \
        TPipe sortPipe2;                                                         \
        MoeSortMultiLastDimCore<uint8_t> op;                                     \
        op.Init(routingMap, sortedIndices, userWS, t, &sortPipe2);               \
        op.Process();                                                            \
        sortPipe2.Destroy();                                                     \
        if (hasProb) {                                                           \
            TPipe sortPipe;                                                      \
            AscendC::SyncAll();                                                  \
            MoePermuteProb<DTYPE_PERMUTE_PROBS> op2;                             \
            op2.Init(permuteProbs, probs, sortedIndices, t, &sortPipe);          \
            op2.Process();                                                       \
            sortPipe.Destroy();                                                  \
        }                                                                        \
    } while (0)                                                                  \

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define GENERAL_PAD_OP_IMPL_3510(sortClass1, sortClass2, indexCopyClass, ...)                                          \
    do {                                                                                                               \
        TPipe sortPipe2;                                                                                               \
        MoeSortMultiLastDimCore<uint8_t> op;                                                                           \
        op.Init(routingMap, sortedIndices, userWS, t, &sortPipe2);                                                     \
        op.Process();                                                                                                  \
        sortPipe2.Destroy();                                                                                           \
        if (hasProb) {                                                                                                 \
            AscendC::SyncAll();                                                                                        \
            gatherv2::Gatherv2SimtTwoDim<DTYPE_PERMUTE_PROBS, int32_t, uint32_t> gatherv2Op;                           \
            gatherv2Op.Init(probs, sortedIndices, permuteProbs, t, true);                                              \
            gatherv2Op.ProbProcess();                                                                                  \
        }                                                                                                              \
    } while (0)

#define GATHER_IMPL()                                                                                                  \
    do {                                                                                                               \
        AscendC::SyncAll();                                                                                            \
        TPipe MoeindexCopyPipe;                                                                                        \
        gatherv2::Gatherv2SimdTwoDim<int32_t> gatherv2Op(&MoeindexCopyPipe);                                           \
        gatherv2Op.Init(tokens, sortedIndices, permuteTokens, t);                                                      \
        gatherv2Op.Process();                                                                                          \
    } while (0)
#endif

extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map(GM_ADDR tokens, GM_ADDR routingMap,
                                                                         GM_ADDR probs, GM_ADDR permuteTokens,
                                                                         GM_ADDR permuteProbs, GM_ADDR sortedIndices,
                                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;
    int64_t hasProb = t->hasProb;
    if (TILING_KEY_IS(1)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopyOp, DTYPE_TOKENS, false);
    } else if (TILING_KEY_IS(3)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopySplitDOp, DTYPE_TOKENS, false);
    } else if (TILING_KEY_IS(2)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopyOp, DTYPE_TOKENS, false);
    } else if (TILING_KEY_IS(4)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySplitDOp, DTYPE_TOKENS, false);
    } else if (TILING_KEY_IS(5)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopyOp, DTYPE_TOKENS, true);
    } else if (TILING_KEY_IS(7)) {
        GENERAL_OP_IMPL(MoeSortOneCore, MoeSortOneCore, MoeindexCopySplitDOp, DTYPE_TOKENS, true);
    } else if (TILING_KEY_IS(6)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopyOp, DTYPE_TOKENS, true);
    } else if (TILING_KEY_IS(8)) {
        GENERAL_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySplitDOp, DTYPE_TOKENS, true);
    } else if (TILING_KEY_IS(9)) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        GENERAL_PAD_OP_IMPL_3510(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySplitDOp, DTYPE_TOKENS, true);
        AscendC::SyncAll();
        TPipe MoeindexCopyPipe;
        gatherv2::Gatherv2SimdTwoDim<int32_t> gatherTokenOp(&MoeindexCopyPipe);
        gatherTokenOp.Init(tokens, sortedIndices, permuteTokens, t);
        gatherTokenOp.Process();
    } else if (TILING_KEY_IS(19)) {
        GENERAL_PAD_OP_IMPL_3510(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySplitDOp, DTYPE_TOKENS, true);
        AscendC::SyncAll();
        gatherv2::Gatherv2SimtTwoDim<DTYPE_TOKENS, int32_t, uint32_t> gatherTokenOp;
        gatherTokenOp.Init(tokens, sortedIndices, permuteTokens, t, false);
        gatherTokenOp.Process();
#else
        GENERAL_PAD_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySplitDOp, DTYPE_TOKENS, true);
#endif
    }
}
