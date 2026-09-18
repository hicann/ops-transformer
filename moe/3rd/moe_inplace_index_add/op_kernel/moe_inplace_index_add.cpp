/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_inplace_index_add.cpp
 * \brief moe_inplace_index_add
 */
#include "arch35/moe_inplace_index_add_simt.h"
#include "arch35/moe_inplace_index_add_determinstic.h"
#include "arch35/moe_inplace_index_add_determinstic_notquant.h"
#include "arch35/moe_inplace_index_add_simd.h"
#include "arch35/moe_inplace_index_add_simd_sort.h"
#include "arch35/moe_inplace_index_add_simt_sort.h"

using namespace MoeInplaceIndexAdd;

#define IIA_UINT32_FP32 100000
#define IIA_UINT32_FP16 100001
#define IIA_UINT32_INT8 100002
#define IIA_UINT32_INT32 100003
#define IIA_UINT32_UINT8 100004
#define IIA_UINT32_INT16 100006
#define IIA_UINT32_BF16 100027
#define IIA_UINT32_INT64 100009
#define IIA_UINT32_BOOL 100012

#define IIA_UINT64_FP32 100100
#define IIA_UINT64_FP16 100101
#define IIA_UINT64_INT8 100102
#define IIA_UINT64_INT32 100103
#define IIA_UINT64_UINT8 100104
#define IIA_UINT64_INT16 100106
#define IIA_UINT64_BF16 100127
#define IIA_UINT64_INT64 100109
#define IIA_UINT64_BOOL 100112

#define IIA_WITH_ALPHA_UINT32_FP32 101000
#define IIA_WITH_ALPHA_UINT32_FP16 101001
#define IIA_WITH_ALPHA_UINT32_INT8 101002
#define IIA_WITH_ALPHA_UINT32_INT32 101003
#define IIA_WITH_ALPHA_UINT32_UINT8 101004
#define IIA_WITH_ALPHA_UINT32_INT16 101006
#define IIA_WITH_ALPHA_UINT32_BF16 101027
#define IIA_WITH_ALPHA_UINT32_INT64 101009
#define IIA_WITH_ALPHA_UINT32_BOOL 101012

#define IIA_WITH_ALPHA_UINT64_FP32 101100
#define IIA_WITH_ALPHA_UINT64_FP16 101101
#define IIA_WITH_ALPHA_UINT64_INT8 101102
#define IIA_WITH_ALPHA_UINT64_INT32 101103
#define IIA_WITH_ALPHA_UINT64_UINT8 101104
#define IIA_WITH_ALPHA_UINT64_INT16 101106
#define IIA_WITH_ALPHA_UINT64_BF16 101127
#define IIA_WITH_ALPHA_UINT64_INT64 101109
#define IIA_WITH_ALPHA_UINT64_BOOL 101112

#define DETERMINSTIC_FLOAT32 300000
#define DETERMINSTIC_FLOAT16 300001
#define DETERMINSTIC_BFLOAT16 300002
#define TEMPLATE_SIMD_SORT 200000
#define TEMPLATE_SIMD 400000
#define TEMPLATE_SIMT_SORT_UINT32_NO_ALPHA 500000
#define TEMPLATE_SIMT_SORT_UINT32_ALPHA 500001
#define TEMPLATE_SIMT_SORT_UINT64_NO_ALPHA 500010
#define TEMPLATE_SIMT_SORT_UINT64_ALPHA 500011

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimdSortCast(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha,
                                                      GM_ADDR var_out, GM_ADDR userWS, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimdSortTilingData, tilingData, tiling);
    uint32_t cast_mode = tilingData.indicesCastMode;
    if (cast_mode == CAST_1) {
        MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_1> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_2) {
        MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_2> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_3) {
        MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_3> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_4) {
        MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_4> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_5) {
        MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_5> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else {
        MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_0> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimtSortCast(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha,
                                                      GM_ADDR var_out, GM_ADDR userWS, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtSortTilingData, tilingData, tiling);
    uint32_t cast_mode = tilingData.indicesCastMode;
    if (cast_mode == CAST_1) {
        MoeInplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS, CAST_1> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_2) {
        MoeInplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS, CAST_2> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_3) {
        MoeInplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS, CAST_3> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_4) {
        MoeInplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS, CAST_4> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else if (cast_mode == CAST_5) {
        MoeInplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS, CAST_5> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    } else {
        MoeInplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS, CAST_0> op(tilingData, pipe);
        op.Init(var, indices, updates, alpha, userWS);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void moe_inplace_index_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                                            GM_ADDR alpha, GM_ADDR var_out, GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(IIA_UINT32_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }

    } else if (TILING_KEY_IS(IIA_UINT32_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(DETERMINSTIC_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddDeterminsticTilingData, tilingData, tiling);
        if (tilingData.isSplitPreAxis == 1 || tilingData.isSplitAfterAxis == 1) {
            MoeInplaceIndexAddDeterminsticNotQuant<bfloat16_t, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddDeterminstic<bfloat16_t, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(DETERMINSTIC_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddDeterminsticTilingData, tilingData, tiling);
        if (tilingData.isSplitPreAxis == 1 || tilingData.isSplitAfterAxis == 1) {
            MoeInplaceIndexAddDeterminsticNotQuant<half, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddDeterminstic<half, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(DETERMINSTIC_FLOAT32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddDeterminsticTilingData, tilingData, tiling);
        if (tilingData.isSplitPreAxis == 1 || tilingData.isSplitAfterAxis == 1) {
            MoeInplaceIndexAddDeterminsticNotQuant<float, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            MoeInplaceIndexAddDeterminstic<float, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_SORT)) {
        if constexpr (std::is_same<int64_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimdSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                MoeInplaceIndexAddSimdSortCast<DTYPE_VAR, DTYPE_INDICES, true>(var, indices, updates, alpha, var_out,
                                                                               userWS, tiling, pipe);
            } else {
                MoeInplaceIndexAddSimdSortCast<DTYPE_VAR, DTYPE_INDICES, false>(var, indices, updates, alpha, var_out,
                                                                                userWS, tiling, pipe);
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMD)) {
        if constexpr (std::is_same<int64_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimdTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                MoeInplaceIndexAddSimd<DTYPE_VAR, DTYPE_INDICES, DTYPE_VAR, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                MoeInplaceIndexAddSimd<DTYPE_VAR, DTYPE_INDICES, DTYPE_VAR, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT32_NO_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
                      std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint32_t, false, true>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            } else {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint32_t, false, false>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT32_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
                      std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint32_t, true, true>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            } else {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint32_t, true, false>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT64_NO_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
                      std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint64_t, false, true>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            } else {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint64_t, false, false>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT64_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
                      std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(MoeInplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint64_t, true, true>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            } else {
                MoeInplaceIndexAddSimtSortCast<DTYPE_VAR, DTYPE_INDICES, uint64_t, true, false>(
                    var, indices, updates, alpha, var_out, userWS, tiling, pipe);
            }
        }
    }
}
