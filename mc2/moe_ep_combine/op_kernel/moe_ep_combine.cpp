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
 * \file moe_ep_combine.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_combine_tiling_key.h"
#include "moe_ep_combine_tiling.h"
#include "moe_ep_combine.h"

using namespace MoeEpCombineImpl;
using namespace Mc2Tiling;
using namespace AscendC;


template <uint32_t HasTopkWeight, uint32_t ArchTag>
__global__ __aicore__ void moe_ep_combine(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                          GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR combinedX,
                                          GM_ADDR combinedTopkWeights, GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpCombineTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeEpCombineTilingData);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeEpCombineTilingData, tilingData, tilingGM);
    MoeEpCombine<DTYPE_X, HasTopkWeight> op;
    op.Init(context, x, topkIdx, recvSrcMetadata, numRecvPerExpert, topkWeights, combinedX, combinedTopkWeights,
            workspace, tilingGM, &pipe, &tilingData.moeEpCombineInfo);
    op.Process();
}
