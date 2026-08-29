/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>

#include "block_attn_res_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockAttnResUpdate);

namespace {
constexpr size_t MATRIX_DIM_NUM = 2UL;
constexpr size_t TOKEN_DIM_INDEX = 0UL;
constexpr size_t HIDDEN_DIM_INDEX = 1UL;

static bool CheckNotNull(const aclTensor *partialBlockRef, const aclTensor *delta, const aclTensor *pseudoQuery,
                         const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum,
                         const aclOpExecutor *executor)
{
    OP_CHECK(partialBlockRef != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate partialBlockRef is nullptr."), return false);
    OP_CHECK(delta != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate delta is nullptr."), return false);
    OP_CHECK(pseudoQuery != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate pseudoQuery is nullptr."),
             return false);
    OP_CHECK(numerator != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate numerator is nullptr."),
             return false);
    OP_CHECK(logitMax != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate logitMax is nullptr."),
             return false);
    OP_CHECK(expSum != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate expSum is nullptr."),
             return false);
    OP_CHECK(executor != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate executor is nullptr."),
             return false);
    return true;
}

static bool IsEmptyWorkload(const aclTensor *partialBlockRef)
{
    const auto &shape = partialBlockRef->GetViewShape();
    return shape.GetDimNum() == MATRIX_DIM_NUM &&
           (shape.GetDim(TOKEN_DIM_INDEX) == 0 || shape.GetDim(HIDDEN_DIM_INDEX) == 0);
}
} // namespace

const aclTensor *BlockAttnResUpdate(aclTensor *partialBlockRef, const aclTensor *delta, const aclTensor *pseudoQuery,
                                    const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum,
                                    float eps, aclOpExecutor *executor)
{
    L0_DFX(BlockAttnResUpdate, partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps);

    if (!CheckNotNull(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, executor)) {
        return nullptr;
    }

    if (IsEmptyWorkload(partialBlockRef)) {
        auto emptyH = executor->AllocTensor(delta->GetViewShape(), DataType::DT_BF16, Format::FORMAT_ND);
        OP_CHECK(emptyH != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockAttnResUpdate AllocTensor for empty output failed."),
                 return nullptr);
        return emptyH;
    }

    auto h = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK(h != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockAttnResUpdate AllocTensor failed."), return nullptr);

    auto ret =
        INFER_SHAPE(BlockAttnResUpdate, OP_INPUT(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum),
                    OP_OUTPUT(partialBlockRef, h), OP_ATTR(eps));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "BlockAttnResUpdate InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(BlockAttnResUpdate,
                                      OP_INPUT(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum),
                                      OP_OUTPUT(partialBlockRef, h), OP_ATTR(eps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "BlockAttnResUpdate ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return h;
}

} // namespace l0op
