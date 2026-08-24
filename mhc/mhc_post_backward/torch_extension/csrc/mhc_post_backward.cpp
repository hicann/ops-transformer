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
 * \brief ACLNN Wrapper for aclnnMhcPostBackward
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPostBackward(const at::Tensor &gradOutput,
                                                                           const at::Tensor &x, const at::Tensor &hRes,
                                                                           const at::Tensor &hOut,
                                                                           const at::Tensor &hPost)
{
    at::Tensor gradX = at::empty_like(x);
    at::Tensor gradHres = at::empty_like(hRes);
    at::Tensor gradHout = at::empty_like(hOut);
    at::Tensor gradHpost = at::empty_like(hPost);

    ACLNN_CMD(aclnnMhcPostBackward, gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(gradX, gradHres, gradHout, gradHpost);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mhc_post_backward", &MhcPostBackward, "mhc_post_backward"); }

} // namespace op_api
