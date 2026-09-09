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
 * \file quant_all_reduce.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_all_reduce_tiling_data.h"
#include "quant_all_reduce_tiling_key.h"
#include "quant_all_reduce_mte_one_shot.h"

using namespace AscendC;
using namespace QuantAllReduceImpl;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#endif

template <uint32_t quantAllReduceCommMode>
__global__ __aicore__ void quant_all_reduce(GM_ADDR context, GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(QuantAllReduceTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantAllReduceTilingData, tilingData, tilingGM);
    TPipe pipe;
    if constexpr (quantAllReduceCommMode == MTE_ONE_SHOT) {
        QuantAllReduceMteOneShot<DTYPE_X, DTYPE_SCALES, DTYPE_OUT_PUT> op;
        op.Init(context, x, scales, output, &pipe, &tilingData);
        op.Process();
    }
}
