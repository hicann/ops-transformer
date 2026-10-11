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
 * \file mhc_sinkhorn.cpp
 * \brief mhc_sinkhorn
 */

#include "mhc_sinkhorn.h"
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

OP_TYPE_REGISTER(MhcSinkhorn);

// AICORE算子kernel
static const aclTensor* MhcSinkhornAiCore(const aclTensor* x, int64_t outFlag, float eps, int64_t numIters,
                                          aclTensor* output, aclTensor* normOut, aclTensor* sumOut,
                                          aclOpExecutor* executor)
{
    L0_DFX(MhcSinkhornAiCore, x, eps, numIters, outFlag, output, normOut, sumOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将MhcSinkhorn算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MhcSinkhorn, OP_INPUT(x), OP_OUTPUT(output, normOut, sumOut),
                                           OP_ATTR(eps, numIters, outFlag));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MhcSinkhorn ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return output;
}

const aclTensor* MhcSinkhorn(const aclTensor* x, int64_t outFlag, float eps, int64_t numIters, aclTensor* output,
                             aclTensor* normOut, aclTensor* sumOut, aclOpExecutor* executor)
{
    return MhcSinkhornAiCore(x, outFlag, eps, numIters, output, normOut, sumOut, executor);
}
} // namespace l0op
