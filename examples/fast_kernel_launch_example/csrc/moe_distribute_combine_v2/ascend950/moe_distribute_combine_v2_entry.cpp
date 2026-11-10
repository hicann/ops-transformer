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
 * \file moe_distribute_combine_v2_entry.cpp
 * \brief A5 (ascend950/arch35) <<<>>> direct-launch entry with SuperKernel support.
 *        Kernel implementations are referenced in place from mc2 op_kernel/arch35
 *        (no local copies), resolved via the repo-root include path.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"

#include "mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h"
#include "mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"

#include "moe_distribute_combine_v2_entry.h"

using namespace MoeDistributeCombineV2A5MteImpl;
using namespace Mc2Kernel;
using namespace AscendC;

/*
 * A5 example 本地补充（不引入 mc2 tiling_key_decl.h）：
 * tiling_key_decl.h 内含 host-only TPL 头 template_argument.h（FastEncodeTilingKeyDirect 用 printf），
 * 在含 __aicore__ printf 的 device TU 中展开会报 "call to __aicore__ function from __host__ function"，
 * 故 device 侧不复用该头，仅镜像其 TILINGKEY_* 声明值（与 ops 构建框架注入的
 * GET_TILING_DATA_WITH_STRUCT 语义等价：从 tilingGM 读出本地 TilingData 结构体）。
 */
#define GET_TILING_DATA_WITH_STRUCT(TilingDataStru, tilingData, tilingArg) \
    TilingDataStru tilingData; \
    { \
        const __gm__ uint32_t* __gm_src = reinterpret_cast<const __gm__ uint32_t*>(tilingArg); \
        uint32_t* __local_dst = reinterpret_cast<uint32_t*>(&tilingData); \
        constexpr uint32_t __words = sizeof(TilingDataStru) / sizeof(uint32_t); \
        for (uint32_t __i = 0; __i < __words; ++__i) { \
            __local_dst[__i] = __gm_src[__i]; \
        } \
    }

constexpr uint32_t TILINGKEY_NO_QUANT = 0;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 2;

/*
 * A5 tilingkey说明（example自有编码，非mc2 TPL注册键，仅在本直调样例内使用）
 * 3位的十进制数，基值200
 * 第1位（个位）：commQuantMode:
 *     0: 不量化, 2: int8通信量化
 * 第2位（十位）：expand_x输入类型:
 *     0: float16, 1: bfloat16
 * 仅支持 MTE 通信
 */

template <typename ExpandXType, typename XType, int32_t QuantMode>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_combine_v2_a5_impl(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
    GM_ADDR tpSendCounts, GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale, GM_ADDR groupList,
    GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
    GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    TPipe pipe;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // A3 式直接传值：mc2Context 为 HcclCombinOpParam 布局上下文设备地址
    GM_ADDR contextGM0 = mc2Context;
    MoeDistributeCombineV2A5Mte<ExpandXType, XType, int32_t, QuantMode, false, false> op;
    // residualX/gamma/yOut/rstdOut在apt.cpp中固定为nullptr
    op.Init(contextGM0, expandX, expertIds, assistInfoForCombine, epSendCount, nullptr, nullptr, scales, xActiveMask,
            sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo,
            nullptr, nullptr, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    pipe.DestroyWithoutPipeAll();
    return;
}

__aicore__ inline void moe_combine_a5_switch(int32_t tilingKey, GM_ADDR expandX, GM_ADDR expertIds,
                                             GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
                                             GM_ADDR tpSendCounts, GM_ADDR xActiveMask, GM_ADDR activationScale,
                                             GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales,
                                             GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                             GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
                                             GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
                                             GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    switch (tilingKey) {
        case 200: // float16, 不量化
            moe_distribute_combine_v2_a5_impl<float16_t, float16_t, TILINGKEY_NO_QUANT>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts, xActiveMask,
                activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM,
                mc2Context);
            break;

        case 210: // bfloat16, 不量化
            moe_distribute_combine_v2_a5_impl<bfloat16_t, bfloat16_t, TILINGKEY_NO_QUANT>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts, xActiveMask,
                activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM,
                mc2Context);
            break;

        case 202: // float16, int8通信量化
            moe_distribute_combine_v2_a5_impl<float16_t, float16_t, TILINGKEY_INT8_COMM_QUANT>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts, xActiveMask,
                activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM,
                mc2Context);
            break;

        case 212: // bfloat16, int8通信量化
            moe_distribute_combine_v2_a5_impl<bfloat16_t, bfloat16_t, TILINGKEY_INT8_COMM_QUANT>(
                expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts, xActiveMask,
                activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX,
                constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM,
                mc2Context);
            break;

        default:
            AscendC::PRINTF("moe_distribute_combine_v2_a5 Error: invalid tilingKey = %d\n", tilingKey);
            break;
    }
}

extern "C" __global__ __aicore__ void moe_distribute_combine_v2_generic(
    int32_t tilingKey, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount,
    GM_ADDR scales, GM_ADDR tpSendCounts, GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
    GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    // 对齐 A5-MTE：SetHcclContext<0>(mc2Context) 供经 GetHcclContext 读取同一 HcclCombinOpParam 上下文
    AscendC::SetHcclContext<0>(mc2Context);
    moe_combine_a5_switch(tilingKey, expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts,
                          xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX,
                          elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut,
                          workspaceGM, tilingGM, mc2Context);
}

// SK孪生模板函数：从SkArgs解包参数后复用moe_combine_a5_switch，与generic走相同分发逻辑
template <int32_t split_num>
__sk__ __vector__ void moe_distribute_combine_v2_generic_sk(MoeDistributeCombineV2SkArgs* args)
{
    int32_t tilingKey = args->tilingKey;
    GM_ADDR expandX = args->expandX;
    GM_ADDR expertIds = args->expertIds;
    GM_ADDR assistInfoForCombine = args->assistInfoForCombine;
    GM_ADDR epSendCount = args->epSendCount;
    GM_ADDR scales = args->scales;
    GM_ADDR tpSendCounts = args->tpSendCounts;
    GM_ADDR xActiveMask = args->xActiveMask;
    GM_ADDR activationScale = args->activationScale;
    GM_ADDR weightScale = args->weightScale;
    GM_ADDR groupList = args->groupList;
    GM_ADDR expandScales = args->expandScales;
    GM_ADDR sharedExpertX = args->sharedExpertX;
    GM_ADDR elasticInfo = args->elasticInfo;
    GM_ADDR oriX = args->oriX;
    GM_ADDR constExpertAlpha1 = args->constExpertAlpha1;
    GM_ADDR constExpertAlpha2 = args->constExpertAlpha2;
    GM_ADDR constExpertV = args->constExpertV;
    GM_ADDR performanceInfo = args->performanceInfo;
    GM_ADDR XOut = args->XOut;
    GM_ADDR workspaceGM = args->workspaceGM;
    GM_ADDR tilingGM = args->tilingGM;
    GM_ADDR mc2Context = args->mc2Context;

    AscendC::SetHcclContext<0>(mc2Context);
    moe_combine_a5_switch(tilingKey, expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts,
                          xActiveMask, activationScale, weightScale, groupList, expandScales, sharedExpertX,
                          elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut,
                          workspaceGM, tilingGM, mc2Context);
}

SK_BIND(moe_distribute_combine_v2_generic, 4, moe_distribute_combine_v2_generic_sk<0>,
        moe_distribute_combine_v2_generic_sk<1>, moe_distribute_combine_v2_generic_sk<2>,
        moe_distribute_combine_v2_generic_sk<3>);

// <<<>>>调用函数
void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR expandX,
                                     GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount,
                                     GM_ADDR scales, GM_ADDR tpSendCounts, GM_ADDR xActiveMask, GM_ADDR activationScale,
                                     GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales,
                                     GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                     GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
                                     GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                     GM_ADDR mc2Context)
{
    moe_distribute_combine_v2_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCounts, xActiveMask,
        activationScale, weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1,
        constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM, mc2Context);
}
