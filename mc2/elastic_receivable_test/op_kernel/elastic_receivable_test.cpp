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
 * \file elastic_receivable_test.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "elastic_receivable_test_tiling.h"
#include "elastic_receivable_test.h"

using namespace AscendC;
using namespace ElasticReceivableTestImpl;

extern "C" __global__ __aicore__ void elastic_receivable_test(GM_ADDR dstRank, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ElasticReceivableTestTilingData);
    TPipe pipe;

    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(ElasticReceivableTestTilingData, tilingData, tilingGM);
        ElasticReceivableTest<DTYPE_DST_RANK> op;
        op.Init(dstRank, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}