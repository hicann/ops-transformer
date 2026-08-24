/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update.cpp
 * \brief Torch bridge — decode / update, delegates to aclnnCausalConv1dUpdate.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor CausalConv1dUpdate(const at::Tensor &x, at::Tensor &convStates, const at::Tensor &weight,
                              const c10::optional<at::Tensor> &bias, const std::string &activation,
                              const c10::optional<at::Tensor> &cacheIndices,
                              const c10::optional<at::Tensor> &numAcceptedTokens,
                              const c10::optional<at::Tensor> &queryStartLoc, int64_t maxQueryLen, int64_t nullBlockId,
                              const c10::optional<at::Tensor> &blockIdxLastScheduledToken,
                              const c10::optional<at::Tensor> &initialStateIdx)
{
    TORCH_CHECK(activation == "silu" || activation == "none", "activation must be 'silu' or 'none', got: ", activation);

    at::Tensor y{nullptr};
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        y = at::empty_like(x);
    }

    const char *activationMode = activation.c_str();
    ACLNN_CMD(aclnnCausalConv1dUpdate, x, weight, convStates, bias, queryStartLoc, cacheIndices, numAcceptedTokens,
              blockIdxLastScheduledToken, initialStateIdx, activationMode, nullBlockId, maxQueryLen, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("causal_conv1d_update", &CausalConv1dUpdate, "causal_conv1d decode/update");
}

} // namespace op_api
