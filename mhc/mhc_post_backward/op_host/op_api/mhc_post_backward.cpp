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
 * \file mhc_post_backward.cpp
 * \brief MhcPostBackward L0 API implementation
 */

#include "mhc_post_backward.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

// 注册算子类型
OP_TYPE_REGISTER(MhcPostBackward);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPostBackwardAICore(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                      const aclTensor *hPost, aclTensor *gradX, aclTensor *gradHres, aclTensor *gradHout,
                      aclTensor *gradHpost, aclOpExecutor *executor)
{
    L0_DFX(MhcPostBackwardAICore, gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(MhcPostBackward, OP_INPUT(gradOutput, x, hRes, hOut, hPost),
                                                 OP_OUTPUT(gradX, gradHres, gradHout, gradHpost));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MhcPostBackward launch kernel failed.");
        return std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *>(nullptr, nullptr, nullptr, nullptr);
    }
    return std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *>(gradX, gradHres, gradHout, gradHpost);
}

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPostBackward(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                const aclTensor *hPost, aclOpExecutor *executor)
{
    L0_DFX(MhcPostBackward, gradOutput, x, hRes, hOut, hPost);

    // 根据输入x的shape分配输出tensor
    auto gradX = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    aclTensor *gradHres = nullptr;
    const aclTensor *hResInput = hRes;
    if (hRes != nullptr) {
        gradHres = executor->AllocTensor(hRes->GetViewShape(), hRes->GetDataType(), hRes->GetViewFormat());
    } else {
        // hRes 为空时，gradHres shape 从 x 推导: [T,n,D]->[T,n,n] 或 [B,S,n,D]->[B,S,n,n]
        auto gradHresShape = x->GetViewShape();
        gradHresShape.SetDim(gradHresShape.GetDimNum() - 1, gradHresShape.GetDim(gradHresShape.GetDimNum() - 2));
        gradHres = executor->AllocTensor(gradHresShape, op::DataType::DT_FLOAT, x->GetViewFormat());
        // A5 no-hres: create empty tensor for operator layer
        op::Shape emptyShape({0});
        hResInput = executor->AllocTensor(emptyShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    }
    auto gradHout = executor->AllocTensor(hOut->GetViewShape(), hOut->GetDataType(), hOut->GetViewFormat());
    auto gradHpost = executor->AllocTensor(hPost->GetViewShape(), hPost->GetDataType(), hPost->GetViewFormat());

    return MhcPostBackwardAICore(gradOutput, x, hResInput, hOut, hPost, gradX, gradHres, gradHout, gradHpost, executor);
}

} // namespace l0op
