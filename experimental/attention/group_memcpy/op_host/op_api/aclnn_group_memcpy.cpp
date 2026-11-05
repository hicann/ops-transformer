/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_group_memcpy.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include <limits>

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(GroupMemcpy);
static aclnnStatus GroupMemcpy(const aclTensorList* src, const aclTensorList* dst, aclOpExecutor* executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(GroupMemcpy, OP_INPUT(src), OP_OUTPUT(dst), OP_ATTR());
}
} // namespace l0op

namespace {
uint64_t ItemBytes(DataType type)
{
    switch (type) {
        case DataType::DT_INT8:
        case DataType::DT_UINT8:
        case DataType::DT_BOOL:
            return 1;
        case DataType::DT_INT16:
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
            return 2;
        case DataType::DT_INT32:
        case DataType::DT_FLOAT:
            return 4;
        case DataType::DT_INT64:
            return 8;
        default:
            return 0;
    }
}
bool Bytes(const aclTensor* tensor, uint64_t itemBytes, uint64_t& bytes)
{
    const auto& shape = tensor->GetViewShape();
    bool empty = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape[i] < 0) {
            return false;
        }
        empty = empty || shape[i] == 0;
    }
    bytes = empty ? 0 : itemBytes;
    for (size_t i = 0; !empty && i < shape.GetDimNum(); ++i) {
        const auto dim = static_cast<uint64_t>(shape[i]);
        if (dim != 0 && bytes > (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 64) / dim) {
            return false;
        }
        bytes *= dim;
    }
    return true;
}
} // namespace

extern "C" aclnnStatus aclnnGroupMemcpyGetWorkspaceSize(const aclTensorList* src, const aclTensorList* dst,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    if (src == nullptr || dst == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (src->Size() == 0 || src->Size() != dst->Size() ||
        src->Size() > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*src)[0] == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const auto dtype = (*src)[0]->GetDataType();
    const uint64_t itemBytes = ItemBytes(dtype);
    if (itemBytes == 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t commonBytes = 0;
    for (size_t i = 0; i < src->Size(); ++i) {
        const auto* input = (*src)[i];
        const auto* output = (*dst)[i];
        if (input == nullptr || output == nullptr) {
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        uint64_t bytes = 0;
        if (input->GetDataType() != dtype || output->GetDataType() != dtype ||
            input->GetViewShape() != output->GetViewShape() || input->GetStorageFormat() != Format::FORMAT_ND ||
            output->GetStorageFormat() != Format::FORMAT_ND || !IsContiguous(input) || !IsContiguous(output) ||
            !Bytes(input, itemBytes, bytes) || (i != 0 && bytes != commonBytes)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        commonBytes = bytes;
    }
    // Address validity, same-device placement and non-overlap are the ACL caller's responsibility.
    L2_DFX_PHASE_1(aclnnGroupMemcpy, DFX_IN(src), DFX_OUT(dst));
    auto owner = CREATE_EXECUTOR();
    if (owner.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    if (commonBytes != 0) {
        const auto status = l0op::GroupMemcpy(src, dst, owner.get());
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnGroupMemcpy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    if (executor == nullptr || (workspaceSize != 0 && workspace == nullptr)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_2(aclnnGroupMemcpy);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
