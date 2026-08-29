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
 * \file all_gather_matmul_v3_apt.cpp
 * \brief Kernel entry for AllGatherMatmulV3 (MX-quant FP8, apace UDMA path)
 */

#include <kernel_operator.h>
#include "all_gather_matmul_v3_tiling_key.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_urma_tiling_data.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_urma_impl.h"

using namespace AscendC;
using namespace Apace;

template <uint32_t QUANTMODE>
__global__ __aicore__ void all_gather_matmul_v3(GM_ADDR context, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                GM_ADDR x2_scale, GM_ADDR y, GM_ADDR gather_out, GM_ADDR amax_out,
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)gather_out;
    (void)amax_out;
    (void)workspaceGM;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(AllGatherMxMatmulUrmaTilingData);
    GET_TILING_DATA_WITH_STRUCT(AllGatherMxMatmulUrmaTilingData, tilingData, tilingGM);

    __gm__ CommContext *hcommCtx = reinterpret_cast<__gm__ CommContext *>(context);

    using Impl = AllGatherMxMatmulUrmaImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y>;

    Impl impl;
    impl.Init(hcommCtx, x1, x1_scale, x2, x2_scale, y, bias, &tilingData);
    impl.Process();
}
