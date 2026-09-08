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
 * \file flash_attn_grad.cpp
 * \brief C++ wrapper for flash_attn_grad operator.
 */

#include <torch/extension.h>
#include <string>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor> FlashAttnGrad(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &doTensor,
    const at::Tensor &attnOut, const at::Tensor &softmaxLse, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &attnMask, const c10::optional<at::Tensor> &metadata, double softmaxScale,
    int64_t maskMode, int64_t winLeft, int64_t winRight, int64_t maxSeqlenQ, int64_t maxSeqlenKv, std::string layoutQ,
    std::string layoutKv, std::string layoutOut)
{
    TORCH_CHECK(q.defined(), "q must be defined");
    TORCH_CHECK(k.defined(), "k must be defined");
    TORCH_CHECK(v.defined(), "v must be defined");
    TORCH_CHECK(doTensor.defined(), "dout must be defined");
    TORCH_CHECK(attnOut.defined(), "attn_out must be defined");
    TORCH_CHECK(softmaxLse.defined(), "softmax_lse must be defined");

    auto device = q.device();
    c10::OptionalDeviceGuard device_guard(device);

    at::Tensor dq;
    at::Tensor dk;
    at::Tensor dv;
    {
        dq = at::empty(q.sizes(), q.options());
        dk = at::empty(k.sizes(), k.options());
        dv = at::empty(v.sizes(), v.options());
    }

    char *layoutQPtr = const_cast<char *>(layoutQ.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKv.c_str());
    char *layoutOutPtr = const_cast<char *>(layoutOut.c_str());

    ACLNN_CMD(aclnnInnerFlashAttnGrad, q, k, v, doTensor, attnOut, softmaxLse, cuSeqlensQ, cuSeqlensKv, sequsedQ,
              sequsedKv, sinks, attnMask, metadata, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKv,
              layoutQPtr, layoutKvPtr, layoutOutPtr, dq, dk, dv);

    return std::make_tuple(dq, dk, dv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn_grad", &FlashAttnGrad, "flash_attn_grad");
}

} // namespace op_api
