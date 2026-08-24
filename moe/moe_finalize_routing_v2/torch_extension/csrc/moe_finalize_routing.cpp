/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr size_t INDEX_ONE = 1;
constexpr size_t INDEX_TWO = 2;
constexpr size_t DIM_THREE = 3;

at::Tensor MoeFinalizeRouting(
    const at::Tensor &expandedX, const at::Tensor &expandedRowIdx, const c10::optional<at::Tensor> &x1Optional,
    const c10::optional<at::Tensor> &x2Optional, const c10::optional<at::Tensor> &biasOptional,
    const c10::optional<at::Tensor> &scalesOptional, const c10::optional<at::Tensor> &expertIdxOptional,
    const c10::optional<at::Tensor> &xOptional, const c10::optional<at::Tensor> &alpha1Optional,
    const c10::optional<at::Tensor> &alpha2Optional, const c10::optional<at::Tensor> &vOptional,
    c10::optional<int64_t> dropPadMode, const c10::optional<std::vector<int64_t>> &zeroExpertRange,
    const c10::optional<std::vector<int64_t>> &copyExpertRange,
    const c10::optional<std::vector<int64_t>> &constantExpertRange, c10::optional<int64_t> k)
{
    int64_t kAttr = c10::value_or_else(k, [] { return 1; });

    int64_t dim0 = expandedRowIdx.size(0);
    if (scalesOptional.has_value()) {
        dim0 = scalesOptional.value().size(0);
    } else if (kAttr > 0) {
        dim0 = dim0 / kAttr;
    }
    size_t dim1Index = INDEX_ONE;
    if (expandedX.dim() == DIM_THREE) {
        dim1Index = INDEX_TWO;
    }
    at::Tensor result = at::empty({dim0, expandedX.size(dim1Index)}, expandedX.options());
    int64_t mode = c10::value_or_else(dropPadMode, [] { return 0; });

    std::vector<int64_t> zeroVec = zeroExpertRange.value_or(std::vector<int64_t>{});
    std::vector<int64_t> copyVec = copyExpertRange.value_or(std::vector<int64_t>{});
    std::vector<int64_t> constVec = constantExpertRange.value_or(std::vector<int64_t>{});
    at::IntArrayRef zeroRange(zeroVec);
    at::IntArrayRef copyRange(copyVec);
    at::IntArrayRef constRange(constVec);
    ACLNN_CMD(aclnnMoeFinalizeRoutingV4, expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
              scalesOptional, expertIdxOptional, xOptional, alpha1Optional, alpha2Optional, vOptional, mode, zeroRange,
              copyRange, constRange, kAttr, result);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_finalize_routing", &MoeFinalizeRouting, "moe_finalize_routing"); }

} // namespace op_api
