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
 * \file moe_finalize_routing_grad.cpp
 * \brief ACLNN Wrapper for aclnnMoeFinalizeRoutingV2Grad
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DROP_PAD_MODE_DROPLESS = 0;
constexpr int64_t DROP_PAD_MODE_DROP_PAD = 1;

static bool HasValue(const c10::optional<at::Tensor> &opt)
{
    return opt.has_value() && opt.value().defined() && opt.value().numel() > 0;
}

std::tuple<at::Tensor, at::Tensor> moe_finalize_routing_grad(const at::Tensor &gradY, const at::Tensor &expandedRowIdx,
                                                             const c10::optional<at::Tensor> &expandedXOptional,
                                                             const c10::optional<at::Tensor> &scalesOptional,
                                                             const c10::optional<at::Tensor> &expertIdxOptional,
                                                             const c10::optional<at::Tensor> &biasOptional,
                                                             int64_t dropPadMode, int64_t activeNum, int64_t expertNum,
                                                             int64_t expertCapacity)
{
    TORCH_CHECK(gradY.dim() == DIM_TWO, "grad_y should be 2D, but got ", gradY.dim(), "-D.");
    TORCH_CHECK(
        gradY.scalar_type() == at::kHalf || gradY.scalar_type() == at::kFloat || gradY.scalar_type() == at::kBFloat16,
        "grad_y dtype should be float16, float32 or bfloat16, but got ", gradY.scalar_type(), ".");
    TORCH_CHECK(expandedRowIdx.dim() == DIM_ONE, "expanded_row_idx should be 1D, but got ", expandedRowIdx.dim(),
                "-D.");
    TORCH_CHECK(expandedRowIdx.scalar_type() == at::kInt, "expanded_row_idx dtype should be int32, but got ",
                expandedRowIdx.scalar_type(), ".");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_DROP_PAD,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");
    TORCH_CHECK(!biasOptional.has_value() || expertIdxOptional.has_value(),
                "expert_idx must be provided when bias is provided.");
    TORCH_CHECK(!scalesOptional.has_value() || expandedXOptional.has_value(),
                "expanded_x must be provided when scales is provided.");
    if (dropPadMode == DROP_PAD_MODE_DROP_PAD) {
        TORCH_CHECK(expertNum > 0, "expert_num must be positive when drop_pad_mode=1, but got ", expertNum, ".");
        TORCH_CHECK(expertCapacity > 0, "expert_capacity must be positive when drop_pad_mode=1, but got ",
                    expertCapacity, ".");
    }

    int64_t hidden = gradY.size(1);

    at::Tensor gradExpandedXOut;
    at::Tensor gradScalesOut;
    {
        auto localDevice = c10::Device(gradY.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        if (dropPadMode == DROP_PAD_MODE_DROP_PAD) {
            gradExpandedXOut = at::empty({expertNum, expertCapacity, hidden}, gradY.options());
        } else {
            int64_t dim0 = expandedRowIdx.numel();
            if (dropPadMode == DROP_PAD_MODE_DROPLESS && activeNum > 0 && activeNum < dim0) {
                dim0 = activeNum;
            }
            gradExpandedXOut = at::empty({dim0, hidden}, gradY.options());
        }

        int64_t scalesDim1 = 1;
        at::TensorOptions scalesOpts = gradY.options();
        if (HasValue(scalesOptional)) {
            scalesDim1 = scalesOptional.value().size(1);
            scalesOpts = scalesOptional.value().options();
        }
        gradScalesOut = at::empty({gradY.size(0), scalesDim1}, scalesOpts);
    }

    ACLNN_CMD(aclnnMoeFinalizeRoutingV2Grad, gradY, expandedRowIdx, expandedXOptional, scalesOptional,
              expertIdxOptional, biasOptional, dropPadMode, activeNum, expertNum, expertCapacity, gradExpandedXOut,
              gradScalesOut);

    return std::make_tuple(std::move(gradExpandedXOut), std::move(gradScalesOut));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_finalize_routing_grad", &moe_finalize_routing_grad, "moe_finalize_routing_grad");
}

} // namespace op_api
