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
#include "quant_grouped_matmul_inplace_add.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(QuantGroupedMatmulInplaceAdd);

aclTensor* QuantGroupedMatmulInplaceAdd(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale1Optional,
                                        const aclTensor* scale2, const aclTensor* groupList, aclTensor* yRef,
                                        int64_t groupListType, int64_t groupSize, aclOpExecutor* executor)
{
    L0_DFX(QuantGroupedMatmulInplaceAdd, x1, x2, scale1Optional, scale2, groupList, yRef);
    auto ret = INFER_SHAPE(QuantGroupedMatmulInplaceAdd, OP_INPUT(x1, x2, scale2, groupList, yRef, scale1Optional),
                           OP_OUTPUT(yRef), OP_ATTR(groupListType, groupSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantGroupedMatmulInplaceAdd,
                                      OP_INPUT(x1, x2, scale2, groupList, yRef, scale1Optional), OP_OUTPUT(yRef),
                                      OP_ATTR(groupListType, groupSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return yRef;
}

} // namespace l0op
