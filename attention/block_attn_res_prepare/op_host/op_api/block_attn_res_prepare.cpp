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
 * \file block_attn_res_prepare.cpp
 * \brief L0 operator implementation for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {
namespace {

bool CheckNotNull(const void *pointer, const char *parameterName)
{
    if (pointer == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s must not be nullptr.", parameterName);
        return false;
    }
    return true;
}

} // namespace

OP_TYPE_REGISTER(BlockAttnResPrepare);

const std::array<const aclTensor *, BLOCK_ATTN_RES_PREPARE_OUTPUT_NUM> BlockAttnResPrepare(const aclTensor *blockRes,
                                                                                           const aclTensor *validBlocks,
                                                                                           const aclTensor *pseudoQuery,
                                                                                           float eps,
                                                                                           aclOpExecutor *executor)
{
    L0_DFX(BlockAttnResPrepare, blockRes, validBlocks, pseudoQuery, eps);
    if (!CheckNotNull(blockRes, "blockRes") || !CheckNotNull(validBlocks, "validBlocks") ||
        !CheckNotNull(pseudoQuery, "pseudoQuery") || !CheckNotNull(executor, "executor")) {
        return {nullptr, nullptr, nullptr};
    }

    // valid_blocks is a device-side control tensor, so the L0 API must not
    // dereference it. The kernel clamps values above N to N at run time.
    auto numerator = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    if (numerator == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate the numerator tensor.");
        return {nullptr, nullptr, nullptr};
    }
    auto logitMax = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    if (logitMax == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate the logitMax tensor.");
        return {nullptr, nullptr, nullptr};
    }
    auto expSum = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    if (expSum == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate the expSum tensor.");
        return {nullptr, nullptr, nullptr};
    }
    const aclnnStatus ret = INFER_SHAPE(BlockAttnResPrepare, OP_INPUT(blockRes, validBlocks, pseudoQuery),
                                        OP_OUTPUT(numerator, logitMax, expSum), OP_ATTR(eps));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "BlockAttnResPrepare infer shape failed: ret=%d, blockRes.shape=%s, validBlocks.shape=%s, "
                "pseudoQuery.shape=%s, eps=%f.",
                ret, op::ToString(blockRes->GetViewShape()).GetString(),
                op::ToString(validBlocks->GetViewShape()).GetString(),
                op::ToString(pseudoQuery->GetViewShape()).GetString(), eps);
        return {nullptr, nullptr, nullptr};
    }

    // T == 0 or S == 0 produces correctly shaped empty outputs.  Do not add
    // an AICore launch for that case: all output elements are already empty.
    if (blockRes->IsEmpty() || pseudoQuery->IsEmpty()) {
        return {numerator, logitMax, expSum};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(BlockAttnResPrepare, OP_INPUT(blockRes, validBlocks, pseudoQuery),
                                OP_OUTPUT(numerator, logitMax, expSum), OP_ATTR(eps));
    return {numerator, logitMax, expSum};
}

} // namespace l0op
