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
#include "gmm_local_exp_with_zero.h"

using namespace AscendC;
using namespace matmul;

#define EXEC_GMM_RL_STEP0 \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, trans_b>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using matmulType = matmul::MatmulImpl<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    mm.SetSubBlockIdx(0);

#define EXEC_GMM_RL_STEP0_NZ \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, tensor_type, trans_b>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type>; \
    using matmulType = matmul::MatmulImpl<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    mm.SetSubBlockIdx(0);

#define EXEC_GMM_RL_STEP1 \
    mm.Init(mmTiling, &tPipe); \
    ops_gmm_local_exp_with_zero::GMMCompute<decltype(mm), tensor_type, tensor_type, int32_t, trans_b, false> \
        computeOp(mm, tiling_data); \
    computeOp.Init(x, weight, group_list, y); \
    ops_gmm_local_exp_with_zero::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

#define EXEC_GMM_RL_STEP1_NZ \
    mm.Init(mmTiling, &tPipe); \
    ops_gmm_local_exp_with_zero::GMMCompute<decltype(mm), tensor_type, tensor_type, int32_t, trans_b, false, true> \
        computeOp(mm, tiling_data); \
    computeOp.Init(x, weight, group_list, y); \
    ops_gmm_local_exp_with_zero::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

#define EXEC_GMM_RL_STEP0_MIXED_OUTPUT(output_type) \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, trans_b>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, output_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, output_type>; \
    using matmulType = matmul::MatmulImpl<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    mm.SetSubBlockIdx(0);

#define EXEC_GMM_RL_STEP0_NZ_MIXED_OUTPUT(output_type) \
    using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, tensor_type, false>; \
    using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, tensor_type, trans_b>; \
    using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, output_type>; \
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, output_type>; \
    using matmulType = matmul::MatmulImpl<xType, weightType, yType, biasType, CFG_MDL>; \
    matmulType mm; \
    mm.SetSubBlockIdx(0);

#define EXEC_GMM_RL_STEP1_MIXED_OUTPUT(output_type) \
    mm.Init(mmTiling, &tPipe); \
    ops_gmm_local_exp_with_zero::GMMCompute<decltype(mm), tensor_type, output_type, int32_t, trans_b, false> \
        computeOp(mm, tiling_data); \
    computeOp.Init(x, weight, group_list, y); \
    ops_gmm_local_exp_with_zero::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

#define EXEC_GMM_RL_STEP1_NZ_MIXED_OUTPUT(output_type) \
    mm.Init(mmTiling, &tPipe); \
    ops_gmm_local_exp_with_zero::GMMCompute<decltype(mm), tensor_type, output_type, int32_t, trans_b, false, true> \
        computeOp(mm, tiling_data); \
    computeOp.Init(x, weight, group_list, y); \
    ops_gmm_local_exp_with_zero::GMMProcess<decltype(computeOp)> op(computeOp, tiling_data); \
    op.Init(); \
    op.Process();

extern "C" __global__ __aicore__ void gmm_local_exp_with_zero(GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    // PRINTF("lg test in kernel");
    GET_TILING_DATA(tiling_data, tiling);

    TPipe tPipe;
    // TODO: SetOverflow?
    AscendCUtils::SetOverflow(1);

    const TCubeTiling *__restrict mmTiling = &(tiling_data.mmTilingData);

    if (TILING_KEY_IS(0)) {
        constexpr bool trans_b = false;
        using tensor_type = half;
        EXEC_GMM_RL_STEP0
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1
    } else if (TILING_KEY_IS(1)) {
        constexpr bool trans_b = true;
        using tensor_type = half;
        EXEC_GMM_RL_STEP0
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1
    } else if (TILING_KEY_IS(2)) {
        constexpr bool trans_b = false;
        using tensor_type = bfloat16_t;
        EXEC_GMM_RL_STEP0
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1
    } else if (TILING_KEY_IS(3)) {
        constexpr bool trans_b = true;
        using tensor_type = bfloat16_t;
        EXEC_GMM_RL_STEP0
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1
    } else if (TILING_KEY_IS(4)) {
        constexpr bool trans_b = true;
        using tensor_type = half;
        EXEC_GMM_RL_STEP0_NZ
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_NZ
    } else if (TILING_KEY_IS(5)) {
        constexpr bool trans_b = true;
        using tensor_type = bfloat16_t;
        EXEC_GMM_RL_STEP0_NZ
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_NZ
    } else if (TILING_KEY_IS(6)) {
        constexpr bool trans_b = false;
        using tensor_type = half;
        using output_type = float;
        EXEC_GMM_RL_STEP0_MIXED_OUTPUT(output_type)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(7)) {
        constexpr bool trans_b = true;
        using tensor_type = half;
        using output_type = float;
        EXEC_GMM_RL_STEP0_MIXED_OUTPUT(output_type)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(8)) {
        constexpr bool trans_b = false;
        using tensor_type = bfloat16_t;
        using output_type = float;
        EXEC_GMM_RL_STEP0_MIXED_OUTPUT(output_type)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(9)) {
        constexpr bool trans_b = true;
        using tensor_type = bfloat16_t;
        using output_type = float;
        EXEC_GMM_RL_STEP0_MIXED_OUTPUT(output_type)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(10)) {
        constexpr bool trans_b = true;
        using tensor_type = half;
        using output_type = float;
        EXEC_GMM_RL_STEP0_NZ_MIXED_OUTPUT(output_type)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_NZ_MIXED_OUTPUT(output_type)
    } else if (TILING_KEY_IS(11)) {
        constexpr bool trans_b = true;
        using tensor_type = bfloat16_t;
        using output_type = float;
        EXEC_GMM_RL_STEP0_NZ_MIXED_OUTPUT(output_type)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        EXEC_GMM_RL_STEP1_NZ_MIXED_OUTPUT(output_type)
    }
}
