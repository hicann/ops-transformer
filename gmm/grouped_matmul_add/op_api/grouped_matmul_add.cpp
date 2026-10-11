/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "grouped_matmul_add.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulAdd);

aclTensor* GroupedMatmulAdd(const aclTensor* x1, const aclTensor* x2, const aclTensor* groupList, aclTensor* yRef,
                            bool transposeX, bool transposeWeight, int64_t groupType, int64_t groupListType,
                            aclOpExecutor* executor)
{
    L0_DFX(GroupedMatmulAdd, x1, x2, groupList, yRef);
    auto ret = INFER_SHAPE(GroupedMatmulAdd, OP_INPUT(x1, x2, groupList, yRef), OP_OUTPUT(yRef),
                           OP_ATTR(transposeX, transposeWeight, groupType, groupListType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulAdd, OP_INPUT(x1, x2, groupList, yRef), OP_OUTPUT(yRef),
                                      OP_ATTR(transposeX, transposeWeight, groupType, groupListType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return yRef;
}

} // namespace l0op
