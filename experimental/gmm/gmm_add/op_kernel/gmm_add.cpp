/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "gmm_add.h"
#include "gmm_add_C_16.h"

using namespace AscendC;
using namespace matmul;

#define EXEC_GMM_ADD \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, trans_a>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using matmulType = matmul::Matmul<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, mmTiling); \
    ops_gmm_add::GMMCompute<decltype(mm), tensor_type, float, int32_t, trans_a, true> computeOp(mm, tPipe, \
                                                                                                tiling_data); \
    computeOp.Init(x, weight, group_list, weightGrad, y, user); \
    ops_gmm_add::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

#define EXEC_GMM_ADD_C_16 \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, trans_a>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using matmulType = matmul::Matmul<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, mmTiling); \
    ops_gmm_add_c_16::GMMCompute<decltype(mm), tensor_type, tensor_type, int32_t, trans_a, true> computeOp( \
        mm, tPipe, tiling_data); \
    computeOp.Init(x, weight, group_list, weightGrad, y, user); \
    ops_gmm_add_c_16::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

extern "C" __global__ __aicore__ void gmm_add(GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR weightGrad,
                                              GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    TPipe tPipe;
    // TODO: SetOverflow?
    AscendCUtils::SetOverflow(1);

    const TCubeTiling *__restrict mmTiling = &(tiling_data.mmTilingData);

    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(0)) {
        constexpr bool trans_a = false;
        using tensor_type = half;
        EXEC_GMM_ADD
    } else if (TILING_KEY_IS(1)) {
        constexpr bool trans_a = false;
        using tensor_type = half;
        EXEC_GMM_ADD_C_16
    } else if (TILING_KEY_IS(2)) {
        constexpr bool trans_a = true;
        using tensor_type = half;
        EXEC_GMM_ADD
    } else if (TILING_KEY_IS(3)) {
        constexpr bool trans_a = true;
        using tensor_type = half;
        EXEC_GMM_ADD_C_16
    } else if (TILING_KEY_IS(4)) {
        constexpr bool trans_a = false;
        using tensor_type = bfloat16_t;
        EXEC_GMM_ADD
    } else if (TILING_KEY_IS(5)) {
        constexpr bool trans_a = false;
        using tensor_type = bfloat16_t;
        EXEC_GMM_ADD_C_16
    } else if (TILING_KEY_IS(6)) {
        constexpr bool trans_a = true;
        using tensor_type = bfloat16_t;
        EXEC_GMM_ADD
    } else if (TILING_KEY_IS(7)) {
        constexpr bool trans_a = true;
        using tensor_type = bfloat16_t;
        EXEC_GMM_ADD_C_16
    }
}
