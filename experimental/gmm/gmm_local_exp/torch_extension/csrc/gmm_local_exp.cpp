/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <acl/acl.h>
#include <torch/extension.h>
#include "aclnn_common.h"
#include <torch/torch.h>

namespace op_api {

at::Tensor local_exp_gmm(const at::Tensor &a, const at::Tensor &b, const at::Tensor &problemList,
                         const int64_t expStartIdx, const int64_t expEndIdx, const bool trans_a, const bool trans_b,
                         const bool is_b_nz, bool type_promotion)
{
    TORCH_CHECK(a.device().type() == at::kPrivateUse1, "a must be on NPU");
    TORCH_CHECK(b.device() == a.device(), "a and b must be on the same device");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16, "a must be float16 or bfloat16");
    TORCH_CHECK(b.scalar_type() == a.scalar_type(), "a and b must have the same dtype");
    TORCH_CHECK(a.is_contiguous(), "a should be contiguous.");
    TORCH_CHECK(b.is_contiguous(), "b should be contiguous.");
    TORCH_CHECK(problemList.is_contiguous(), "problemList should be contiguous.");

    if (a.dim() != 2) {
        throw std::runtime_error("[local_exp_gmm] Input tensor 'a' must be a 2-D tensor.");
    }
    if (b.dim() != 3) {
        throw std::runtime_error("[local_exp_gmm] Input tensor 'b' must be a 3-D tensor.");
    }
    if (problemList.scalar_type() != at::kInt && problemList.scalar_type() != at::kLong) {
        throw std::runtime_error("[local_exp_gmm] problemList must be an integer tensor");
    }
    if (problemList.dim() != 1) {
        throw std::runtime_error("[local_exp_gmm] problemList must be a 1-D tensor");
    }

    TORCH_CHECK(!trans_a, "local expert GMM supports only trans_a=False");
    TORCH_CHECK(problemList.device() == a.device(), "problemList must be on the same NPU");
    TORCH_CHECK(expStartIdx >= 0 && expEndIdx > expStartIdx && expEndIdx <= problemList.size(0),
                "invalid expert range");
    TORCH_CHECK(b.size(0) == expEndIdx - expStartIdx, "weight expert count must match the selected range");
    TORCH_CHECK(a.size(1) == b.size(trans_b ? 2 : 1), "a and b K dimensions must match");
    TORCH_CHECK(!is_b_nz || trans_b, "NZ weights require trans_b=True");
    auto problemList_int32 = problemList.to(at::kInt);

    {
        auto n = b.sizes()[2];
        if (trans_b) {
            n = b.sizes()[1];
        }
        c10::TensorOptions options = a.options().dtype(a.scalar_type());
        auto out_size = std::vector<int64_t>({a.sizes()[0], n});
        // auto    out = at::empty(out_size, options).zero_();
        if (type_promotion) {
            options = options.dtype(at::kFloat);
        }
        auto out = at::empty(out_size, options);
        ACLNN_CMD(aclnnGmmLocalExp, a, b, problemList_int32, trans_b, is_b_nz, expStartIdx, expEndIdx, out);
        return out;
    }
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("local_exp_gmm", &op_api::local_exp_gmm);
}
