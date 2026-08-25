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
 * \file moe_init_routing_grad.cpp
 * \brief ACLNN Wrapper for aclnnMoeInitRoutingV2Grad
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DROP_PAD_MODE_DROPLESS = 0;
constexpr int64_t DROP_PAD_MODE_DROP_PAD = 1;

at::Tensor moe_init_routing_grad(const at::Tensor &gradExpandedX, const at::Tensor &expandedRowIdx, int64_t topK,
                                 int64_t dropPadMode, int64_t activeNum)
{
    TORCH_CHECK(gradExpandedX.dim() == 2 || gradExpandedX.dim() == 3, "grad_expanded_x should be 2D or 3D, but got ",
                gradExpandedX.dim(), "-D.");
    TORCH_CHECK(gradExpandedX.scalar_type() == at::kHalf || gradExpandedX.scalar_type() == at::kFloat ||
                    gradExpandedX.scalar_type() == at::kBFloat16,
                "grad_expanded_x dtype should be float16, float32 or bfloat16, but got ", gradExpandedX.scalar_type(),
                ".");
    TORCH_CHECK(expandedRowIdx.dim() == 1, "expanded_row_idx should be 1D, but got ", expandedRowIdx.dim(), "-D.");
    TORCH_CHECK(expandedRowIdx.scalar_type() == at::kInt, "expanded_row_idx dtype should be int32, but got ",
                expandedRowIdx.scalar_type(), ".");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_DROP_PAD,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");
    TORCH_CHECK(topK > 0, "top_k must be greater than 0, but got ", topK, ".");
    TORCH_CHECK(activeNum >= 0, "active_num must be non-negative, but got ", activeNum, ".");
    if (dropPadMode == DROP_PAD_MODE_DROP_PAD) {
        TORCH_CHECK(gradExpandedX.dim() == 3, "grad_expanded_x should be 3D when drop_pad_mode=1, but got ",
                    gradExpandedX.dim(), "-D.");
    }

    int64_t gradXDim0 = expandedRowIdx.numel() / topK;
    int64_t gradXDim1 = (dropPadMode == DROP_PAD_MODE_DROP_PAD) ? gradExpandedX.size(2) : gradExpandedX.size(1);

    at::Tensor out;
    {
        auto localDevice = c10::Device(gradExpandedX.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        out = at::empty({gradXDim0, gradXDim1}, gradExpandedX.options());
    }

    ACLNN_CMD(aclnnMoeInitRoutingV2Grad, gradExpandedX, expandedRowIdx, topK, dropPadMode, activeNum, out);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_init_routing_grad", &moe_init_routing_grad, "moe_init_routing_grad");
}

} // namespace op_api
