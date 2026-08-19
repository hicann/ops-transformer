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
 * \file lightning_indexer_kl_loss.cpp
 * \brief AscendC kernel entry for LightningIndexerKL_Loss
 */

#include "lightning_indexer_kl_loss_kernel.h"
#include "lightning_indexer_kl_loss_tiling_key.h"

using namespace NsLightningIndexerKLLoss;

/*
 * Kernel entry: 8 TilingKey modes via one bool + one uint8:
 *   DeterType=0/1  |  DataType=0(FLOAT16),1(FLOAT32),2(BFLOAT16),3(FLOAT16_PRECISION)
 *   isHalf = (dataType != 1)  i.e. only FLOAT32 is full-precision
 */
template <bool isDeterministic, uint8_t dataType, bool weightType>
__global__ __aicore__ void lightning_indexer_kl_loss(GM_ADDR targetScore, GM_ADDR indexProbs, GM_ADDR loss,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LightningIndexerKLLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(LightningIndexerKLLossTilingData, tilingData, tiling);
    TPipe pipe;
    GM_ADDR user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if constexpr (dataType == 0) { // FLOAT16
        LightningIndexerKLLoss<half, isDeterministic, weightType> op;
        op.Init(targetScore, indexProbs, loss, user, &tilingData, &pipe);
        op.InitHalfBufs();
        op.Process();
        op.WriteBackDet();
    } else if constexpr (dataType == 2) { // BFLOAT16
        LightningIndexerKLLoss<bfloat16_t, isDeterministic, weightType> op;
        op.Init(targetScore, indexProbs, loss, user, &tilingData, &pipe);
        op.InitHalfBufs();
        op.Process();
        op.WriteBackDet();
    } else { // FLOAT32
        LightningIndexerKLLoss<float, isDeterministic, weightType> op;
        GM_ADDR user = GetUserWorkspace(workspace);
        op.Init(targetScore, indexProbs, loss, user, &tilingData, &pipe);
        op.Process();
        if constexpr (isDeterministic) {
            op.WriteBackDet();
        }
    }
}
