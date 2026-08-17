/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lightning_indexer_kl_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LightningIndexerKLLoss);

const aclTensor *LightningIndexerKLLoss(const aclTensor *targetScore, const aclTensor *indexProbs, double eps,
                                        const char *weightType, aclOpExecutor *executor)
{
    L0_DFX(LightningIndexerKLLoss, targetScore, indexProbs, eps, weightType);
    DataType outputDtype = targetScore->GetDataType();
    auto loss = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);

    auto ret = INFER_SHAPE(LightningIndexerKLLoss, OP_INPUT(targetScore, indexProbs), OP_OUTPUT(loss),
                           OP_ATTR(eps, weightType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LightningIndexerKLLoss InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(LightningIndexerKLLoss, OP_INPUT(targetScore, indexProbs), OP_OUTPUT(loss),
                                      OP_ATTR(eps, weightType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LightningIndexerKLLoss launch kernel failed.");
        return nullptr;
    }
    return loss;
}

} // namespace l0op
