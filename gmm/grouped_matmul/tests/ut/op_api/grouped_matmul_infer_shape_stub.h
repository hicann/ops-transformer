/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_INFER_SHAPE_STUB_H
#define GROUPED_MATMUL_INFER_SHAPE_STUB_H

#include "opdev/make_op_executor.h"

bool& GroupedMatmulInferShapeStubEnabled()
{
    static thread_local bool enabled = false;
    return enabled;
}

// API UTs exercise parameter checks; real inference is covered by op_host UTs.
// The common stub returns success without setting shapes. Model single-tensor
// split-M here so the production output-shape check can still run unchanged.
inline aclnnStatus InferGroupedMatmulShapeForUt(const aclTensorList* x, const aclTensorList* weight,
                                                const aclTensorList* out, int64_t groupType, bool transposeWeight)
{
    if (!GroupedMatmulInferShapeStubEnabled() || groupType != 0 || x == nullptr || weight == nullptr ||
        out == nullptr || x->Size() != 1 || weight->Size() != 1 || out->Size() != 1) {
        return ACLNN_SUCCESS;
    }
    const auto& xShape = (*x)[0]->GetViewShape();
    const auto& wShape = (*weight)[0]->GetViewShape();
    if (xShape.GetDimNum() != 2 || wShape.GetDimNum() != 3) {
        return ACLNN_SUCCESS;
    }
    const op::Shape shape{xShape.GetDim(0), wShape.GetDim(transposeWeight ? 1 : 2)};
    auto* output = const_cast<aclTensor*>((*out)[0]);
    output->SetViewShape(shape);
    output->SetStorageShape(shape);
    output->SetOriginalShape(shape);
    return ACLNN_SUCCESS;
}

#undef INFER_SHAPE
#define INFER_SHAPE(KERNEL_NAME, ...) InferGroupedMatmulShapeForUt(x, weight, out, groupType, transposeWeight)

#endif
