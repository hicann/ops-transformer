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
 * \file moe_distribute_combine_v3.cpp
 * \brief A3 (arch22) kernel entry for MoeDistributeCombineV3.
 *        Reuses combine_v2's A3 combine implementation.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#if __has_include("../../moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2.h")
#include "../../moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2.h"
#include "../../moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_tiling_key.h"
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
#elif __has_include("../../moe_distribute_combine_v2/arch22/moe_distribute_combine_v2.h")
#include "../../moe_distribute_combine_v2/arch22/moe_distribute_combine_v2.h"
#include "../../moe_distribute_combine_v2/arch22/moe_distribute_combine_v2_tiling_key.h"
#include "../../moe_distribute_combine_v2/moe_distribute_combine_tiling.h"
#else
#include "../../../moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2.h"
#include "../../../moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_tiling_key.h"
#include "../../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
#endif
using namespace AscendC;
using namespace Mc2Kernel;
using namespace Mc2Tiling;

template <uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void
moe_distribute_combine_v3(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                          GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount, GM_ADDR xActiveMask,
                          GM_ADDR activationScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales,
                          GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                          GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
                          GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        MoeDistributeCombineV2<Mc2Kernel::Mc2MoeContextHolder, DTYPE_EXPAND_X, DTYPE_X, int32_t, QuantMode, false> op;
        op.Init(mc2Context, expandX, expertIds, assistInfoForCombine, epSendCount, nullptr, nullptr, scales,
                xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, nullptr, nullptr, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
