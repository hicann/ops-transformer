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
 * \file mhc_sinkhorn_backward_apt.cpp
 * \brief mhc_sinkhorn_backward
 */

#include "arch35/mhc_sinkhorn_backward_simd.h"

using namespace AscendC;
using namespace MhcSinkhornBackward;

template <int64_t TILING_KEY>
__global__ __aicore__ void mhc_sinkhorn_backward(GM_ADDR gradY, GM_ADDR norm, GM_ADDR sum, GM_ADDR gradInput,
                                                 GM_ADDR workSpace, GM_ADDR tiling)
{
    GM_ADDR user = GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcSinkhornBackwardTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    MhcSinkhornBackwardSimd op(pipe, tilingData);
    op.Init(gradY, norm, sum, gradInput, tiling);
    op.Process();
}
