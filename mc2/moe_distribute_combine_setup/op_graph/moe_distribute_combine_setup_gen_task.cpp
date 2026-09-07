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
 * \file moe_distribute_combine_setup_gen_task.cpp
 * \brief
 */
#include <vector>

#include "common/utils/op_mc2.h"
#include "platform/platform_info.h"

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
namespace ops {

ge::Status MoeDistributeCombineSetupCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    OPS_LOG_D(context->GetNodeName(), "Do A3 aicpu CalcParam.");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeCombineSetupGenTaskFunc(const gert::ExeResGenerationContext *context,
                                                std::vector<std::vector<uint8_t>> &tasks)
{
    OPS_LOG_I(context->GetNodeName(), "Do A3 aicpu GenTask.");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeCombineSetup)
    .CalcOpParam(MoeDistributeCombineSetupCalcParamFunc)
    .GenerateTask(MoeDistributeCombineSetupGenTaskFunc);
} // namespace ops
