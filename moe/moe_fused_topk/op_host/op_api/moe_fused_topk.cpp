/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_topk.cpp
 * \brief
 */

#include "moe_fused_topk.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace {
static const int64_t OUT_TOPK_INDEX = 1;
}

namespace l0op {
OP_TYPE_REGISTER(MoeFusedTopk);

const std::array<const aclTensor*, 2> MoeFusedTopk(const aclTensor* x, const aclTensor* addNum,
                                                   const aclTensor* mappingNum, const aclTensor* mappingTable,
                                                   uint32_t groupNum, uint32_t groupTopk, uint32_t topN, uint32_t topK,
                                                   uint32_t activateType, bool isNorm, float scale,
                                                   bool enableExpertMapping, aclOpExecutor* executor)
{
    L0_DFX(MoeFusedTopk, x, addNum, mappingNum, mappingTable, groupNum, groupTopk, topN, topK, activateType, isNorm,
           scale, enableExpertMapping);

    op::Shape outShape = x->GetViewShape();
    outShape.SetDim(OUT_TOPK_INDEX, topK);
    auto y = executor->AllocTensor(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto indices = executor->AllocTensor(outShape, op::DataType::DT_INT32, op::Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MoeFusedTopk, OP_INPUT(x, addNum, mappingNum, mappingTable), OP_OUTPUT(y, indices),
        OP_ATTR(groupNum, groupTopk, topN, topK, activateType, isNorm, scale, enableExpertMapping));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MoeFusedTopkAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {y, indices};
}
} // namespace l0op
