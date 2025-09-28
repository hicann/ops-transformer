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
 * \file moe_token_permute_with_routing_map.cpp
 * \brief
 */
#include "moe_mrgsort_out.h"
#include "moe_mrgsort.h"
#include "moe_sort_multi_core.h"
#include "moe_sort_multi_core_last_dim.h"
#include "moe_sort_one_core.h"
#include "moe_index_copy.h"
#include "moe_index_copy_split_d.h"
#include "masked_select_v3.h"
#include "moe_permute_prob.h"
#if !defined(DTYPE_TOKENS)
#define DTYPE_TOKENS bfloat16_t
#endif

using namespace AscendC;
using namespace MoeTokenPermute;
#define GENERAL_OP_IMPL(sortClass1, sortClass2, indexCopyClass, ...)                                                   \
    do {                                                                                                               \
        TPipe sortPipe;                                                                                                \
        KernelMaskedSelectV3<DTYPE_PERMUTE_PROBS> opMS;                                                                \
        auto maskedSelectTilingData = &(t->maskedSelectParamsOp);                                                      \
        opMS.Init(probs, routingMap, permuteProbs, sortedIndices, userWS, maskedSelectTilingData, hasProb, &sortPipe); \
        opMS.Process(permuteProbs, sortedIndices);                                                                     \
        sortPipe.Destroy();                                                                                            \
        AscendC::SyncAll();                                                                                            \
        TPipe sortPipe2;                                                                                               \
        sortClass2<int32_t> op2;                                                                                       \
        op2.Init(sortedIndices, sortedIndices, userWS, t, &sortPipe2);                                                 \
        op2.Process();                                                                                                 \
        sortPipe2.Destroy();                                                                                           \
        TPipe MoeindexCopyPipe;                                                                                        \
        indexCopyClass<__VA_ARGS__> indexCopyOp;                                                                       \
        indexCopyOp.Init(tokens, sortedIndices, permuteTokens, t, &MoeindexCopyPipe);                                  \
        indexCopyOp.Process();                                                                                         \
    } while (0)
#define GENERAL_PAD_OP_IMPL(sortClass1, sortClass2, indexCopyClass, ...) \
    do {                                                                 \
        TPipe sortPipe2;                                                 \
        MoeSortMultiLastDimCore<uint8_t> op;                             \
        op.Init(routingMap, sortedIndices, userWS, t, &sortPipe2);       \
        op.Process();                                                    \
        sortPipe2.Destroy();                                             \
        if (hasProb) {                                                   \
            TPipe sortPipe;                                              \
            AscendC::SyncAll();                                          \
            MoePermuteProb<DTYPE_PERMUTE_PROBS> op2;                     \
            op2.Init(permuteProbs, probs, sortedIndices, t, &sortPipe);  \
            op2.Process();                                               \
        }                                                                \
    } while (0)
extern "C" __global__ __aicore__ void moe_token_permute_with_routing_map(
    GM_ADDR tokens, GM_ADDR routingMap, GM_ADDR probs, GM_ADDR permuteTokens, GM_ADDR permuteProbs,
    GM_ADDR sortedIndices, GM_ADDR workspace, GM_ADDR tiling)
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
        GENERAL_PAD_OP_IMPL(MoeSortMultiCore, MoeSortMultiCore, MoeindexCopySplitDOp, DTYPE_TOKENS, true);
    }
}
