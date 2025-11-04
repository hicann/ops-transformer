/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_buffer_reset.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_distribute_buffer_reset_tiling.h"
#include "moe_distribute_buffer_reset.h"

using namespace AscendC;
using namespace MoeDistributeBufferResetImpl;

extern "C" __global__ __aicore__ void moe_distribute_buffer_reset(GM_ADDR elasticInfo, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeBufferResetTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeBufferResetTilingData, tilingData, tilingGM);
        MoeDistributeBufferReset<DTYPE_ELASTIC_INFO> op;
        op.Init(elasticInfo, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}