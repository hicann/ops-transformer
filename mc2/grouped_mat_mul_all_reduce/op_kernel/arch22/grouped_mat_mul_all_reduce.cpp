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
 * \file grouped_mat_mul_all_reduce.cpp
 * \brief
 */
#define K_MAX_SHAPE_DIM 0

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../grouped_mat_mul_all_reduce_utils.h"
#include "../grouped_mat_mul_all_reduce.h"

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MAT_MUL_ALL_REDUCE;

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[AC_MAX_RANK_NUM];
    uint64_t windowsOut[AC_MAX_RANK_NUM];
    char hcomId[HCCL_COMM_DOMAIN_KEY_MAX_LEN];
    HcclStreamInfo streamInfo[AC_MAX_RANK_NUM];
    HcclCombinOpSignalParam signalInfo;
    HcclConfig config; // 配置参数
};

// for oom check
__aicore__ inline void OOMInit(__gm__ HcclCombinOpParam* context)
{
#ifndef __CCE_KT_TEST__
    AscendC::OOMCheckAddrRange((__gm__ uint8_t*)(context->WorkSpace), context->WorkSpaceSize);
    AscendC::OOMCheckAddrRange((__gm__ uint8_t*)(context->windowsIn[context->rankId]), context->winSize);
#endif
}

using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, false>;
using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, false>;
using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;
using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

extern "C" __global__ __aicore__ void grouped_mat_mul_all_reduce(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                 GM_ADDR group_list, GM_ADDR y, GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
#ifdef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(GMMAllReduceTilingData);
#endif
    GET_TILING_DATA_MEMBER(GMMAllReduceTilingData, aicoreTiling, aicore_tiling_data, tiling);
    if (aicore_tiling_data.debugMode == static_cast<uint32_t>(DebugMode::MC2_DEBUG_ONLY_AICPU)) {
        return;
    }

    const TCubeTiling* __restrict mmTiling = &(aicore_tiling_data.mmTilingData);
    GM_ADDR user1 = GetUserWorkspace(workspace);

    TPipe tPipe;
    HcclServer hcclServer;
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    OOMInit(context);
    __gm__ uint8_t* workspaceMsg = (__gm__ uint8_t*)(context->WorkSpace + aicore_tiling_data.notifyOff);
    hcclServer.Init(workspaceMsg, aicore_tiling_data.debugMode);

    if (TILING_KEY_IS(0)) { // float16 and bf16
        if ASCEND_IS_AIV {
            return;
        }
        using mmType = MMImplType<xType, weightType, yType, biasType, CFG_MDL>;
        mmType::MT mm;
        mm.SetSubBlockIdx(0);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        PRELOAD(4); // If comment this line, the program will be blocked.
#endif
        mm.Init(mmTiling, &tPipe);
        GMMCompute<mmType> computeOp(mm);
        computeOp.Init(x, weight, bias, y, user1);
        GMMProcess<decltype(computeOp)> op(computeOp);
        op.Init(&aicore_tiling_data, &tPipe);
        op.Process(hcclServer);
    }
}
