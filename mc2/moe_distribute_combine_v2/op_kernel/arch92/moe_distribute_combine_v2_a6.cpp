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
 * \file moe_distribute_combine_v2_a6.cpp
 * \brief A6 (arch92) kernel entry for MoeDistributeCombineV2.
 *        Dedicated entry so that Ascend960 (NPU_ARCH 9201/9202) builds a source compiled with its own
 *        __NPU_ARCH__, keeping it separated from the Ascend950 (NPU_ARCH 3510, arch35) entry.
 *        Supports the RSW data-as-flag MTE combine implementation only (no ROCE on arch92).
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_a6_tiling_key.h"
#include "moe_distribute_combine_v2_a6_mte.h"

using namespace MoeDistributeCombineV2A6MteImpl;
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
    GET_TILING_DATA_MEMBER(MoeDistributeCombineV2TilingData, moeDistributeCombineV2Info, combineV2Info, tilingGM);
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if constexpr ((ArchTag == TILINGKEY_TPL_A6) && (LayeredMode == TILINGKEY_TPL_MTE)) {
        GM_ADDR contextGM0 = (GM_ADDR)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        MoeDistributeCombineV2A6Mte<DTYPE_EXPAND_X, DTYPE_X, int32_t, QuantMode, false> op;
        op.Init(contextGM0, expandX, expertIds, assistInfoForCombine, epSendCount, nullptr, nullptr, scales,
                xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
                performanceInfo, nullptr, nullptr, XOut, workspaceGM, &pipe, &combineV2Info);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
