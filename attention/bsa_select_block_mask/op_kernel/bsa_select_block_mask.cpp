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
 * \file bsa_select_block_mask.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "bsa_select_block_mask_tiling_key.h"

#if (__CCE_AICORE__ == 310)
#pragma message("[BSASelectBlockMask] compiling arch35 kernel path")
#include "arch35/bsa_select_block_mask_base_arch35.h"
#include "arch35/bsa_select_block_mask_common_arch35.h"
#else
#include "bsa_select_block_mask_base.h"
#include "bsa_select_block_mask_common.h"
#endif

using namespace AscendC;

#define BSA_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        templateClass<BSAType<__VA_ARGS__>> op;                                                   \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        op.Init(query, key, blockShape, postBlockShape,                                           \
                actualSeqLensQ, actualSeqLensKV,                                                  \
                actualBlockLenQ, actualBlockLenKV,                                                \
                maskOut,                                                                    \
                user, tiling_data, &tPipe);                                              \
        op.Process();                                                                             \
    } while (0)

template<int LayoutQ, int LayoutKV>
__global__ __aicore__ void bsa_select_block_mask(
    __gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *blockShape, __gm__ uint8_t *postBlockShape,
    __gm__ uint8_t *actualSeqLensQ, __gm__ uint8_t *actualSeqLensKV,
    __gm__ uint8_t *actualBlockLenQ, __gm__ uint8_t *actualBlockLenKV,
    __gm__ uint8_t *maskOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::BSASelectBlockMaskTilingData);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    BSA_OP_IMPL(BSASelectBlockMaskBase, optiling::BSASelectBlockMaskTilingData,
                DTYPE_QUERY, int8_t,
                static_cast<BSALayout>(LayoutQ), static_cast<BSALayout>(LayoutKV));
}
