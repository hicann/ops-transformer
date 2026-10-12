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
 * \file attention_update.cpp
 * \brief
 */

#include "attention_update.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AttentionUpdate);

const std::tuple<const aclTensor*, const aclTensor*> AttentionUpdate(const aclTensorList* lse, const aclTensorList* go,
                                                                     int64_t updateType, int64_t sp,
                                                                     aclOpExecutor* executor)
{
    L0_DFX(AttentionUpdate, lse, go, updateType, sp);

    op::Shape outShape = (*go)[0]->GetViewShape();
    auto out = executor->AllocTensor(outShape, (*go)[0]->GetDataType(), op::Format::FORMAT_ND);

    op::Shape lseOutShape = (*lse)[0]->GetViewShape();
    auto lseOut = executor->AllocTensor(lseOutShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AttentionUpdate, OP_INPUT(lse, go), OP_OUTPUT(out, lseOut),
                                           OP_ATTR(updateType, sp));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AttentionUpdateAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    return std::tuple<aclTensor*, aclTensor*>(out, lseOut);
}
} // namespace l0op
