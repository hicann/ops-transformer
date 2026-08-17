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
 * \file mhc_post_backward.cpp
 * \brief
 */

#include "./arch22/mhc_post_backward_arch22.h"
#include "./arch22/mhc_post_backward_tiling_data_arch22.h"
#include "./arch22/mhc_post_backward_tiling_key_arch22.h"
#include "kernel_operator.h"

using namespace AscendC;

template <uint32_t SCHMODE = 0>
__global__ __aicore__ void mhc_post_backward(
    GM_ADDR grad_y, GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out, GM_ADDR h_post,
    GM_ADDR grad_x, GM_ADDR grad_h_res, GM_ADDR grad_h_out, GM_ADDR grad_h_post,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(MhcPostBackwardTilingDataArch22);
    GET_TILING_DATA(tiling_data, tiling);
    KernelMhcPostBackward<DTYPE_GRAD_Y> op;
    op.Init(grad_y, x, h_res, h_out, h_post, grad_x, grad_h_res, grad_h_out,
            grad_h_post, tiling_data, &pipe);
    op.Process();
}
