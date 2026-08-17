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
 * \file mhc_sinkhorn_apt.cpp
 * \brief mhc_sinkhorn
 */

#include "arch35/mhc_sinkhorn_arch35.h"

using namespace AscendC;
using namespace MhcSinkhorn;

template <int64_t TEMPLATE_MODE>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out,
                                        GM_ADDR sum_out, GM_ADDR workSpace, GM_ADDR tiling)
{
    GM_ADDR user = GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    AscendC::TPipe pipe;
    if constexpr (TEMPLATE_MODE == 0) {
        GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tilingData, tiling);
        MhcSinkhornSimd<float, false> op(pipe, tilingData);
        op.Init(h_res, y, norm_out, sum_out, tiling);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == 1) {
        GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tilingData, tiling);
        MhcSinkhornSimd<float, true> op(pipe, tilingData);
        op.Init(h_res, y, norm_out, sum_out, tiling);
        op.Process();
    }
}
