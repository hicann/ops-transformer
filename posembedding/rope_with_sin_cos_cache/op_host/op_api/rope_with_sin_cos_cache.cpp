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
 * \file rope_with_sin_cos_cache.cpp
 * \brief
 */

#include "rope_with_sin_cos_cache.h"
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
OP_TYPE_REGISTER(RopeWithSinCosCache);

// AICORE算子kernel
const std::tuple<aclTensor*, aclTensor*> RoepWithSinCosCacheAICore(const aclTensor* positions, const aclTensor* queryIn,
                                                                   const aclTensor* keyIn, const aclTensor* cosSinCache,
                                                                   const aclIntArray* mropeSection, int64_t headSize,
                                                                   bool isNeoxStyle, int64_t qStride, int64_t kStride,
                                                                   int64_t numQHeads, int64_t numKHeads,
                                                                   int64_t cacheMode, aclTensor* queryOut,
                                                                   aclTensor* keyOut, aclOpExecutor* executor)
{
    L0_DFX(RoepWithSinCosCacheAICore, positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle,
           qStride, kStride, numQHeads, numKHeads, cacheMode, queryOut, keyOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        RopeWithSinCosCache, OP_INPUT(positions, queryIn, keyIn, cosSinCache), OP_OUTPUT(queryOut, keyOut),
        OP_ATTR(numQHeads, numKHeads, headSize, mropeSection, qStride, kStride, isNeoxStyle, cacheMode));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RoepWithSinCosCacheAICore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tie(queryOut, keyOut);
}

const std::tuple<aclTensor*, aclTensor*> RopeWithSinCosCache(const aclTensor* positions, const aclTensor* queryIn,
                                                             const aclTensor* keyIn, const aclTensor* cosSinCache,
                                                             const aclIntArray* mropeSection, int64_t headSize,
                                                             bool isNeoxStyle, int64_t qStride, int64_t kStride,
                                                             int64_t numQHeads, int64_t numKHeads, int64_t cacheMode,
                                                             aclOpExecutor* executor)
{
    // 根据推导出的输出shape申请输出tensor
    auto queryOut = executor->AllocTensor(queryIn->GetViewShape(), queryIn->GetDataType(), queryIn->GetViewFormat());
    auto keyOut = executor->AllocTensor(keyIn->GetViewShape(), keyIn->GetDataType(), keyIn->GetViewFormat());

    return RoepWithSinCosCacheAICore(positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle,
                                     qStride, kStride, numQHeads, numKHeads, cacheMode, queryOut, keyOut, executor);
}
} // namespace l0op
