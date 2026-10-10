/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file grouped_matmul_quant.cpp
 */

#include "grouped_matmul_quant_w4a16.cpp"

extern "C" __global__ __aicore__ void grouped_matmul_quant(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale,
                                                           GM_ADDR weight_offset, GM_ADDR group_list, GM_ADDR y,
                                                           GM_ADDR usrworkspace, GM_ADDR gmmTiling)
{
    AscendC::SetAtomicNone();
    GET_TILING_DATA(tiling_data, gmmTiling);
    const GroupedMatmulQuantTilingData* __restrict tilingData = &tiling_data;
    if (TILING_KEY_IS(10000001)) {
        if (tilingData->dataType == 1) {
            GroupedMatmulQuantW4A16<half> gmm_handle;
            gmm_handle.Init(x, quantized_weight, weight_scale, weight_offset, group_list, y, usrworkspace, tilingData);
            gmm_handle.Process();
        } else if (tilingData->dataType == 27) {
            GroupedMatmulQuantW4A16<bfloat16_t> gmm_handle;
            gmm_handle.Init(x, quantized_weight, weight_scale, weight_offset, group_list, y, usrworkspace, tilingData);
            gmm_handle.Process();
        }
    }
}
