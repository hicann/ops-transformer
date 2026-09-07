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
 * \file moe_distribute_combine_a3.cpp
 * \brief A3 (arch22) kernel entry for MoeDistributeCombine.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_tiling_key.h"
#include "../moe_distribute_combine.h"
using namespace MoeDistributeCombineImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template <uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                                  GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                  GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                  GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR XOut,
                                                  GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingData, tilingData, tilingGM);
    TPipe pipe;
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        MoeDistributeCombine<DTYPE_EXPAND_X, int32_t, QuantMode == TILINGKEY_INT8_QUANT> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
