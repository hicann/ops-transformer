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
 * \file moe_distribute_combine_gen_task.cpp
 * \brief
 */
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"

namespace ops {
ge::Status MoeDistributeCombineCalcParamFunc(gert::ExeResGenerationContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Do general calc param");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MoeDistributeCombineGenTaskFunc(const gert::ExeResGenerationContext* context,
                                           std::vector<std::vector<uint8_t>>& tasks)
{
    const char* nodeName = context->GetNodeName();
    if (IsTargetPlatformSocVersion(nodeName, PLATFORM_A2)) {
        OPS_LOG_D(context->GetNodeName(), "Do A2 gen task");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3/A5 gen task");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeCombine)
    .CalcOpParam(MoeDistributeCombineCalcParamFunc)
    .GenerateTask(MoeDistributeCombineGenTaskFunc);
} // namespace ops
