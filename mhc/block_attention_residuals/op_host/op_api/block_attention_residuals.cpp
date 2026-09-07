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
 * \file block_attention_residuals.cpp
 * \brief BlockAttentionResiduals L0 op_api
 */
#include "block_attention_residuals.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(BlockAttentionResiduals);

const std::array<const aclTensor *, 3> BlockAttentionResiduals(const aclTensor *partialBlock, const aclTensor *blockRes,
                                                               const aclTensor *projWeight, const aclTensor *normWeight,
                                                               int64_t validBlockNum, double normEps, bool needBackward,
                                                               aclOpExecutor *executor)
{
    L0_DFX(BlockAttentionResiduals, partialBlock, blockRes, projWeight, normWeight, validBlockNum, normEps,
           needBackward);

    DataType outType = partialBlock->GetDataType();
    Format format = Format::FORMAT_ND;
    auto hiddenStates = executor->AllocTensor(outType, format, format);
    auto invNorm = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    auto probs = executor->AllocTensor(DataType::DT_FLOAT, format, format);

    auto ret = INFER_SHAPE(BlockAttentionResiduals, OP_INPUT(partialBlock, blockRes, projWeight, normWeight),
                           OP_OUTPUT(hiddenStates, invNorm, probs),
                           OP_ATTR(validBlockNum, static_cast<float>(normEps), needBackward));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttentionResiduals InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(BlockAttentionResiduals, OP_INPUT(partialBlock, blockRes, projWeight, normWeight),
                                      OP_OUTPUT(hiddenStates, invNorm, probs),
                                      OP_ATTR(validBlockNum, static_cast<float>(normEps), needBackward));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttentionResiduals ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }

    return {hiddenStates, invNorm, probs};
}
} // namespace l0op
