/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <torch/extension.h>
#include "aclnn_common.h"
#include <torch/torch.h>

namespace op_api {

at::Tensor gmm(const at::Tensor &a, const at::Tensor &b, const at::Tensor &batch_sizes, const bool trans_a,
               const bool trans_b, const c10::optional<at::Tensor> &c, const int64_t aicore_num, bool type_promotion)
{
    TORCH_CHECK(a.device().type() == at::kPrivateUse1, "a must be on NPU");
    TORCH_CHECK(b.device() == a.device(), "a and b must be on the same device");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16, "a must be float16 or bfloat16");
    TORCH_CHECK(b.scalar_type() == a.scalar_type(), "a and b must have the same dtype");
    TORCH_CHECK(a.is_contiguous(), "a should be contiguous.");
    TORCH_CHECK(b.is_contiguous(), "b should be contiguous.");
    TORCH_CHECK(batch_sizes.is_contiguous(), "batch_sizes should be contiguous.");
    if (a.dim() != 2) {
        throw std::runtime_error("[gmm] Input tensor 'a' must be a 2-D tensor.");
    }
    if (b.dim() != 2) {
        throw std::runtime_error("[gmm] Input tensor 'b' must be a 2-D tensor.");
    }
    if (batch_sizes.scalar_type() != at::kInt && batch_sizes.scalar_type() != at::kLong) {
        throw std::runtime_error("[gmm] batch_sizes must be an integer tensor");
    }
    if (batch_sizes.dim() != 1) {
        throw std::runtime_error("[gmm] batch_sizes must be a 1-D tensor");
    }
    if (trans_b) {
        throw std::runtime_error("[gmm] Do not support trans_b=True.");
    }

    TORCH_CHECK(batch_sizes.device() == a.device(), "batch_sizes must be on the same NPU");
    const int64_t m = a.size(trans_a ? 1 : 0);
    TORCH_CHECK(a.size(trans_a ? 0 : 1) == b.size(0), "a and b K dimensions must match");
    auto batch_sizes_int32 = batch_sizes.to(at::kInt);

    if (c.has_value()) {
        // Backward-bgrad: c[g] += X_g.T @ W_g  (in-place)
        at::Tensor c_tensor = c.value();
        TORCH_CHECK(c_tensor.device() == a.device(), "c must be on the same NPU");
        TORCH_CHECK(c_tensor.dim() == 3 && c_tensor.size(0) == batch_sizes.size(0) && c_tensor.size(1) == m &&
                        c_tensor.size(2) == b.size(1),
                    "c shape must be [groups, M, N]");
        TORCH_CHECK(c_tensor.scalar_type() == at::kFloat || c_tensor.scalar_type() == a.scalar_type(),
                    "c dtype must be float32 or the input dtype");
        TORCH_CHECK(c_tensor.is_contiguous(), "c_tensor should be contiguous.");
        ACLNN_CMD(aclnnGmmAdd, a, b, batch_sizes_int32, c_tensor, trans_a, aicore_num, c_tensor);
        return c_tensor;
    } else {
        // Backward-bgrad: y[g] = X_g.T @ W_g
        auto num_group = batch_sizes_int32.sizes()[0];
        c10::TensorOptions options = a.options().dtype(a.scalar_type());
        auto out_size = std::vector<int64_t>({num_group, m, b.size(1)});
        if (type_promotion) {
            options = options.dtype(at::kFloat);
        }
        auto out = at::empty(out_size, options);
        out.zero_();
        ACLNN_CMD(aclnnGmmKDim, a, b, batch_sizes_int32, trans_a, out);
        return out;
    }
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gmm", &op_api::gmm);
}
