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
 * \file moe_distribute_combine_v2_apt.cpp
 * \brief A5 (arch35) kernel entry for MoeDistributeCombineV2.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_apt_tiling_key.h"
#include "moe_distribute_combine_v2_a5_ccu.h"
#include "moe_distribute_combine_v2_a5_mte.h"

using namespace MoeDistributeCombineV2A5MteImpl;
using namespace Mc2Kernel;
using namespace Mc2Tiling;
using namespace AscendC;

template <uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine_v2(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                     GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCounts,
                                                     GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                     GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX,
                                                     GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                                     GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
                                                     GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
                                                     GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if constexpr ((ArchTag == TILINGKEY_TPL_A5) && (LayeredMode == TILINGKEY_TPL_CCU)) {
        MoeDistributeCombineA5Impl::MoeDistributeCombineA5<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, xActiveMask, scales, sharedExpertX, XOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A5) && (LayeredMode == TILINGKEY_TPL_MTE)) {
        GM_ADDR contextGM0 = (GM_ADDR)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        MoeDistributeCombineV2A5Mte<DTYPE_EXPAND_X, DTYPE_X, int32_t, QuantMode, false> op;
        op.Init(contextGM0, expandX, expertIds, assistInfoForCombine, epSendCount, nullptr, nullptr, scales,
                xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, nullptr, nullptr, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
