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
 * \file mhc_post.cpp
 * \brief MhcPost L0 API implementation
 */

#include "mhc_post.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include <string>
#include <vector>
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

// 注册算子类型
OP_TYPE_REGISTER(MhcPost);

// AICORE算子kernel
const aclTensor* MhcPostAICore(const aclTensor* x, const aclTensor* hRes, const aclTensor* hOut, const aclTensor* hPost,
                               const aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(MhcPostAICore, x, hRes, hOut, hPost, output);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(MhcPost, OP_INPUT(x, hRes, hOut, hPost), OP_OUTPUT(output));
    if (retAicore != ACLNN_SUCCESS) {
        const std::string errorReason = "MhcPost launch kernel failed; launcher status: " + std::to_string(retAicore);
        OP_LOGE_LIBOPAPI_REPORT("MhcPost", "%s", errorReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1009", std::vector<const char*>({"opName", "reason"}),
                                  std::vector<const char*>({"MhcPost", errorReason.c_str()}));
        return nullptr;
    }
    return output;
}

const aclTensor* MhcPost(const aclTensor* x, const aclTensor* hRes, const aclTensor* hOut, const aclTensor* hPost,
                         aclOpExecutor* executor)
{
    auto output = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    if (output == nullptr) {
        const std::string errorReason = "MhcPost AllocTensor returned nullptr.";
        OP_LOGE_LIBOPAPI_REPORT("MhcPost", "%s", errorReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1009", std::vector<const char*>({"opName", "reason"}),
                                  std::vector<const char*>({"MhcPost", errorReason.c_str()}));
        return nullptr;
    }

    return MhcPostAICore(x, hRes, hOut, hPost, output, executor);
}

} // namespace l0op
